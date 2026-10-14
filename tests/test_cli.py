# Project uniurdf
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Tests for the command line interface.

The CLI holds no alignment math; these tests cover argument handling, exit
codes, output formatting, and the ``align`` verb contract.

Exit code conventions under test:
    EXIT_OK            = 0  -- clean completion
    EXIT_ERROR         = 1  -- bad input, bad config, or fatal diagnostics
    EXIT_VERIFY_FAILED = 3  -- geometry regression in ``align``

``argparse`` owns 2 for a malformed command line, so it is absent here.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from uniurdf.cli import EXIT_ERROR, EXIT_OK, EXIT_VERIFY_FAILED, main

# ---------------------------------------------------------------------------
# Synthetic URDF content
# ---------------------------------------------------------------------------

# A two-joint arm where neither axis is +Z; alignment will normalize both.
ARM_URDF = b"""<?xml version="1.0"?>
<robot name="arm">
  <link name="base"/><link name="l1"/><link name="l2"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/><axis xyz="0 0 -1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
  </joint>
</robot>"""

# A URDF with a cycle (fatal).
CYCLE_URDF = b"""<?xml version="1.0"?>
<robot name="cycle">
  <link name="a"/>
  <link name="b"/>
  <joint name="ab" type="revolute">
    <parent link="a"/><child link="b"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="ba" type="revolute">
    <parent link="b"/><child link="a"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_spec(
    path: Path,
    source_urdf: Path | None = None,
    output: str | None = None,
    extra: dict | None = None,
) -> Path:
    data: dict = {}
    if source_urdf is not None:
        data["source_urdf"] = str(source_urdf)
    if output is not None:
        data["output"] = output
    if extra:
        data.update(extra)
    path.write_text(yaml.dump(data))
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def urdf(tmp_path: Path) -> Path:
    path = tmp_path / "arm.urdf"
    path.write_bytes(ARM_URDF)
    return path


@pytest.fixture()
def spec(tmp_path: Path, urdf: Path) -> Path:
    return _write_spec(tmp_path / "arm.yaml", source_urdf=urdf)


@pytest.fixture()
def spec_with_tcp(tmp_path: Path, urdf: Path) -> Path:
    data: dict = {
        "source_urdf": str(urdf),
        "chains": [["j1", "j2"]],
        "tcp": [
            {
                "base_link": "l2",
                "xyz": [0.0, 0.0, 0.15],
                "rpy": [0.0, 0.0, 270.0],
            }
        ],
    }
    path = tmp_path / "arm_tcp.yaml"
    path.write_text(yaml.dump(data))
    return path


# ---------------------------------------------------------------------------
# Exit code constants
# ---------------------------------------------------------------------------


def test_exit_codes_are_distinct() -> None:
    """Scripts must be able to tell a bad input from a failed check."""
    assert len({EXIT_OK, EXIT_ERROR, EXIT_VERIFY_FAILED}) == 3


def test_unknown_command_exits_nonzero() -> None:
    """An unrecognised subcommand exits with a non-zero code."""
    with pytest.raises(SystemExit) as exc:
        main(["nonsense"])
    assert exc.value.code != 0


# ---------------------------------------------------------------------------
# ``run`` verb
# ---------------------------------------------------------------------------


class TestRun:
    """The ``run`` subcommand."""

    def test_produces_source_and_aligned_urdf(
        self, spec: Path, tmp_path: Path, capsys
    ) -> None:
        """Run writes <stem>.source.urdf and <stem>.urdf side by side."""
        out = tmp_path / "out"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = main(["align", str(spec), "--output", str(out)])
        assert code == EXIT_OK
        assert (out / "arm.source.urdf").exists()
        assert (out / "arm.urdf").exists()

    def test_produces_a_pruned_urdf_and_its_own_inventory(
        self, spec_with_tcp: Path, tmp_path: Path, capsys
    ) -> None:
        """Four files, because two of them index different joint vectors."""
        out = tmp_path / "out"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert main(["align", str(spec_with_tcp), "--output", str(out)]) == EXIT_OK
        assert (out / "arm.pruned.urdf").exists()
        assert (out / "arm.pruned.inventory.json").exists()
        assert (out / "arm.inventory.json").exists()

    def test_pruned_urdf_keeps_the_tcp_link(
        self, spec_with_tcp: Path, tmp_path: Path, capsys
    ) -> None:
        """Alignment's own output hangs below its base link.

        Ancestor closure alone would delete it, so this is the end-to-end
        check that the seeds reach the CLI path too.
        """
        from uniurdf.kinematics.chain import build_chain

        out = tmp_path / "out"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main(["align", str(spec_with_tcp), "--output", str(out)])
        pruned = build_chain((out / "arm.pruned.urdf").read_bytes())
        assert "l2_tcp" in pruned.link_names

    def test_run_warns_when_pruning_renumbers_the_joint_vector(
        self, tmp_path: Path, capsys
    ) -> None:
        """The one silent failure mode gets said out loud.

        A stale index still moves the robot -- just not the joint the caller
        meant -- so the terminal has to name which inventory to trust.
        """
        urdf = tmp_path / "arm.urdf"
        urdf.write_text(
            """<?xml version="1.0"?>
<robot name="arm">
  <link name="base"/><link name="l1"/><link name="spare"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="unused" type="revolute">
    <parent link="base"/><child link="spare"/>
    <origin xyz="1 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>"""
        )
        config = tmp_path / "spec.yaml"
        config.write_text(
            yaml.safe_dump(
                {"source_urdf": str(urdf), "chains": [["j1"]], "tcp": [None]}
            )
        )
        out = tmp_path / "out"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main(["align", str(config), "--output", str(out)])
        text = capsys.readouterr().out
        assert "renumbers the joint vector" in text
        assert ".pruned.inventory.json" in text

    def test_exit_ok_on_clean_run(self, spec: Path, tmp_path: Path, capsys) -> None:
        """Run returns EXIT_OK when everything succeeds."""
        out = tmp_path / "out"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = main(["align", str(spec), "--output", str(out)])
        assert code == EXIT_OK

    def test_malformed_xml_is_reported_not_raised(self, tmp_path: Path, capsys) -> None:
        """A truncated URDF is a bad input, and reads like one.

        Staging copies the file before anything parses it, so the copy has
        to survive XML the parser rejects for the diagnostic to be the thing
        the user sees.
        """
        urdf = tmp_path / "broken.urdf"
        urdf.write_text('<?xml version="1.0"?>\n<robot name="broken">\n  <link')
        config = tmp_path / "spec.yaml"
        config.write_text(yaml.safe_dump({"source_urdf": str(urdf)}))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = main(["align", str(config), "--output", str(tmp_path / "out")])

        captured = capsys.readouterr()
        assert code == EXIT_ERROR
        assert "fatal" in (captured.out + captured.err).lower()

    def test_run_reports_normalized_joints(
        self, spec: Path, tmp_path: Path, capsys
    ) -> None:
        """Run prints how many joints were normalized."""
        out = tmp_path / "out"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main(["align", str(spec), "--output", str(out)])
        out_text = capsys.readouterr().out
        assert "normalized" in out_text
        assert "joint" in out_text

    def test_run_reports_tcp_links(
        self, spec_with_tcp: Path, tmp_path: Path, capsys
    ) -> None:
        """Run prints TCP link names when the spec declares them."""
        out = tmp_path / "out"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main(["align", str(spec_with_tcp), "--output", str(out)])
        out_text = capsys.readouterr().out
        assert "l2_tcp" in out_text

    def test_run_reports_verify_verdict(
        self, spec: Path, tmp_path: Path, capsys
    ) -> None:
        """Run prints the three verification error magnitudes."""
        out = tmp_path / "out"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main(["align", str(spec), "--output", str(out)])
        out_text = capsys.readouterr().out
        assert "verify" in out_text.lower() or "verified" in out_text.lower()

    def test_run_prints_absolute_output_path(
        self, spec: Path, tmp_path: Path, capsys
    ) -> None:
        """Run prints the absolute output path before writing."""
        out = tmp_path / "out"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main(["align", str(spec), "--output", str(out)])
        out_text = capsys.readouterr().out
        assert str(out.resolve()) in out_text

    def test_run_fatal_diagnostics_exits_1(self, tmp_path: Path, capsys) -> None:
        """Run exits EXIT_ERROR when the URDF has fatal diagnostics."""
        urdf_path = tmp_path / "cycle.urdf"
        urdf_path.write_bytes(CYCLE_URDF)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)
        out = tmp_path / "out"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = main(["align", str(spec_path), "--output", str(out)])
        assert code == EXIT_ERROR
        err = capsys.readouterr().err
        assert "fatal" in err.lower() or "error" in err.lower()

    def test_run_missing_config_exits_1(self, tmp_path: Path, capsys) -> None:
        """Run exits EXIT_ERROR when the config file does not exist."""
        code = main(["align", str(tmp_path / "ghost.yaml")])
        assert code == EXIT_ERROR

    def test_run_config_without_source_urdf_exits_1(
        self, tmp_path: Path, capsys
    ) -> None:
        """Run exits EXIT_ERROR when the spec has no source_urdf."""
        spec_path = _write_spec(tmp_path / "spec.yaml")
        code = main(["align", str(spec_path)])
        assert code == EXIT_ERROR

    def test_run_output_override_beats_spec(self, tmp_path: Path, capsys) -> None:
        """--output overrides the spec's output field."""
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(ARM_URDF)
        spec_path = _write_spec(
            tmp_path / "spec.yaml",
            source_urdf=urdf_path,
            output="./spec_output",
        )
        explicit_out = tmp_path / "explicit"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = main(["align", str(spec_path), "--output", str(explicit_out)])
        assert code == EXIT_OK
        assert (explicit_out / "robot.urdf").exists()
        # spec_output must NOT have been created.
        assert not (Path.cwd() / "spec_output").exists()

    def test_run_rerun_overwrites_with_warning_in_stderr(
        self, spec: Path, tmp_path: Path, capsys
    ) -> None:
        """Re-running prints an overwrite warning to stderr."""
        out = tmp_path / "out"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main(["align", str(spec), "--output", str(out)])

        # Second run.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = main(["align", str(spec), "--output", str(out)])

        assert code == EXIT_OK
        err = capsys.readouterr().err
        assert "overwriting" in err.lower() or "warning" in err.lower()

    def test_run_verify_failed_exits_2(self, tmp_path: Path, capsys) -> None:
        """Exit 2 when verification fails (distinguishable from exit 1)."""
        # Fabricate a VerifyReport that says ok=False, then mock it in.
        urdf_path = tmp_path / "arm.urdf"
        urdf_path.write_bytes(ARM_URDF)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)
        out = tmp_path / "out"

        from uniurdf.robot import VerifyReport

        bad_report = VerifyReport(
            ok=False,
            max_link_position_error=1.0,
            max_mesh_position_error=0.0,
            max_mesh_rotation_error=0.0,
            moved_links=("base",),
        )

        with patch("uniurdf.robot.AlignedRobot.verify", return_value=bad_report):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                code = main(["align", str(spec_path), "--output", str(out)])

        assert code == EXIT_VERIFY_FAILED


# ---------------------------------------------------------------------------
# Exit code distinctness (belt and suspenders)
# ---------------------------------------------------------------------------


def test_exit_ok_is_zero() -> None:
    assert EXIT_OK == 0


def test_exit_error_is_one() -> None:
    assert EXIT_ERROR == 1


def test_exit_verify_failed_is_three() -> None:
    """`argparse` owns 2, so a failed check must not claim it too.

    A CI script testing for 2 would otherwise read a mistyped flag as
    geometry the tool could not vouch for.
    """
    assert EXIT_VERIFY_FAILED == 3
    assert EXIT_VERIFY_FAILED != 2
