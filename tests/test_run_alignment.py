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

"""The alignment workflow and the two refusals built into it.

`run_alignment` is where the sequence and its gates live, so the CLI and the
viewer enforce the same rules. These tests drive it directly: the gates are
properties of the workflow, not of any one front end.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from uniurdf.project import (
    AlignmentStatus,
    load_project,
    run_alignment,
)

# Two joints, neither axis on +Z, so alignment has work to do.
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

# Two joints forming a cycle, so the URDF cannot be read as a tree at all.
# A defect that merely *warns* is not enough here: diagnostics can flag a
# URDF that still loads, and those go on to align.
# `ghost` is never declared, so j2 cannot be placed. Parsing skips it and the
# robot loads -- with one fewer joint than the file declares. The URDF is
# fatal and loadable at once, which is the case the gate turns on.
FATAL_BUT_LOADABLE_URDF = b"""<?xml version="1.0"?>
<robot name="ghost_parent">
  <link name="base"/><link name="l1"/><link name="l2"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="ghost"/><child link="l2"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>"""

UNLOADABLE_URDF = b"""<?xml version="1.0"?>
<robot name="cycle">
  <link name="a"/><link name="b"/>
  <joint name="ab" type="revolute">
    <parent link="a"/><child link="b"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="ba" type="revolute">
    <parent link="b"/><child link="a"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>"""


def _project(tmp_path: Path, urdf: bytes, **spec_fields):
    """Write a URDF and a spec naming it, and load the project."""
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_bytes(urdf)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.dump({"source_urdf": str(urdf_path), **spec_fields}))
    return load_project(spec_path, output=tmp_path / "out")


def _run(project, **kwargs):
    """Run the workflow with warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_alignment(project, **kwargs)


class TestCleanRun:
    """The path where everything holds."""

    def test_reports_ok_and_writes_the_bundle(self, tmp_path: Path) -> None:
        outcome = _run(_project(tmp_path, ARM_URDF))
        assert outcome.status is AlignmentStatus.OK
        assert outcome.ok
        assert outcome.bundle is not None
        assert outcome.bundle.write.urdf.exists()
        assert outcome.bundle.inventory.exists()

    def test_carries_every_report_it_produced(self, tmp_path: Path) -> None:
        """A caller renders these; the workflow does not print them."""
        outcome = _run(_project(tmp_path, ARM_URDF))
        assert outcome.staged is not None
        assert outcome.aligned is not None
        assert outcome.verify is not None
        assert outcome.verify.ok
        assert not outcome.error

    def test_writes_nothing_to_stdout_or_stderr(self, tmp_path: Path, capsys) -> None:
        """The workflow is usable from a server, where printing is wrong."""
        _run(_project(tmp_path, ARM_URDF))
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


class TestFatalDiagnosticsGate:
    """A URDF with a fatal finding is not aligned, loadable or not."""

    def test_refuses_to_align(self, tmp_path: Path) -> None:
        """A URDF with a cycle has no root, so there is nothing to align."""
        outcome = _run(_project(tmp_path, UNLOADABLE_URDF))
        assert outcome.status is AlignmentStatus.FATAL_DIAGNOSTICS
        assert not outcome.ok
        assert outcome.aligned is None
        assert outcome.bundle is None

    def test_names_the_file_to_repair(self, tmp_path: Path) -> None:
        """The staged copy is what the user edits, so the error names it."""
        outcome = _run(_project(tmp_path, UNLOADABLE_URDF))
        assert outcome.staged is not None
        assert str(outcome.staged.staged_path) in outcome.error

    def test_keeps_the_diagnostics(self, tmp_path: Path) -> None:
        """The findings are the whole point of the refusal."""
        outcome = _run(_project(tmp_path, UNLOADABLE_URDF))
        assert outcome.staged is not None
        assert outcome.staged.diagnostics.is_fatal
        assert outcome.staged.diagnostics.findings

    def test_writes_no_files(self, tmp_path: Path) -> None:
        outcome = _run(_project(tmp_path, UNLOADABLE_URDF))
        assert not list(outcome.project.output.glob("*.inventory.json"))

    def test_refuses_a_fatal_urdf_that_still_loads(self, tmp_path: Path) -> None:
        """The dangerous case: flagged fatal, yet it parses.

        Parsing skips the joint it cannot place, so alignment would succeed
        against a robot short a joint and write an inventory whose vector
        matches nothing anyone reviewed. Loadability is not the question.
        """
        project = _project(tmp_path, FATAL_BUT_LOADABLE_URDF)
        outcome = _run(project)

        assert outcome.staged is not None
        assert outcome.staged.robot is not None, "fixture must still load"
        assert outcome.staged.diagnostics.is_fatal

        assert outcome.status is AlignmentStatus.FATAL_DIAGNOSTICS
        assert outcome.aligned is None
        assert outcome.bundle is None

    def test_writes_nothing_for_a_fatal_urdf_that_loads(self, tmp_path: Path) -> None:
        """No aligned URDF, and no inventory to index against it."""
        project = _project(tmp_path, FATAL_BUT_LOADABLE_URDF)
        _run(project)

        aligned = [
            path
            for path in project.output.glob("*.urdf")
            if not path.name.endswith(".source.urdf")
        ]
        assert aligned == []
        assert list(project.output.glob("*.inventory.json")) == []

    def test_the_dropped_joint_is_why(self, tmp_path: Path) -> None:
        """Pins the reason the gate exists, not just that it fires."""
        outcome = _run(_project(tmp_path, FATAL_BUT_LOADABLE_URDF))
        assert outcome.staged is not None
        assert outcome.staged.robot is not None
        # The file declares j1 and j2; only j1 survives parsing.
        assert outcome.staged.robot.joint_names == ("j1",)


class TestSpecErrorGate:
    """A spec that disagrees with its URDF stops the run."""

    def test_reports_spec_error(self, tmp_path: Path) -> None:
        project = _project(tmp_path, ARM_URDF, tcp=[{"base_link": "no_such_link"}])
        outcome = _run(project)
        assert outcome.status is AlignmentStatus.SPEC_ERROR
        assert outcome.error
        assert outcome.bundle is None


class TestVerifyGate:
    """Geometry that failed its check is not written.

    A correct aligner cannot be made to fail verification through a spec --
    that is the guarantee holding -- so the failing report is substituted
    here. The gate, not the check, is what these cover.
    """

    @staticmethod
    def _failing_report():
        from uniurdf.robot import VerifyReport

        return VerifyReport(
            ok=False,
            max_link_position_error=1.0,
            max_mesh_position_error=0.0,
            max_mesh_rotation_error=0.0,
            moved_links=("base",),
            compared_links=3,
        )

    def test_refuses_to_write(self, tmp_path: Path) -> None:
        project = _project(tmp_path, ARM_URDF)
        with patch(
            "uniurdf.robot.AlignedRobot.verify",
            return_value=self._failing_report(),
        ):
            outcome = _run(project)

        assert outcome.status is AlignmentStatus.VERIFY_FAILED
        assert outcome.bundle is None
        assert not (project.output / "robot.urdf").exists()

    def test_keeps_the_failing_report(self, tmp_path: Path) -> None:
        """A caller needs to show which links moved."""
        project = _project(tmp_path, ARM_URDF)
        with patch(
            "uniurdf.robot.AlignedRobot.verify",
            return_value=self._failing_report(),
        ):
            outcome = _run(project)

        assert outcome.verify is not None
        assert outcome.verify.moved_links == ("base",)

    def test_force_writes_and_still_reports_the_failure(self, tmp_path: Path) -> None:
        """Forcing produces files; it does not produce a passing status."""
        project = _project(tmp_path, ARM_URDF)
        with patch(
            "uniurdf.robot.AlignedRobot.verify",
            return_value=self._failing_report(),
        ):
            outcome = _run(project, force=True)

        assert outcome.status is AlignmentStatus.VERIFY_FAILED
        assert outcome.forced
        assert not outcome.ok
        assert outcome.bundle is not None
        assert outcome.bundle.write.urdf.exists()


class TestForce:
    """Writing despite a failed check is possible, never silent."""

    def test_force_is_not_needed_for_a_clean_run(self, tmp_path: Path) -> None:
        """A passing run reports OK whether or not force was given."""
        outcome = _run(_project(tmp_path, ARM_URDF), force=True)
        assert outcome.status is AlignmentStatus.OK
        assert not outcome.forced


class TestStagingReuse:
    """A caller that has already staged does not pay for it twice."""

    def test_accepts_an_already_staged_robot(self, tmp_path: Path) -> None:
        """The viewer stages before serving, then runs alignments per request."""
        project = _project(tmp_path, ARM_URDF)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            staged = project.stage()
            outcome = run_alignment(project, staged=staged)
        assert outcome.status is AlignmentStatus.OK
        assert outcome.staged is staged


def test_status_values_are_distinct() -> None:
    """A caller switches on these, so they must not collide."""
    assert len({status.value for status in AlignmentStatus}) == len(
        list(AlignmentStatus)
    )


@pytest.mark.parametrize(
    "status",
    [
        AlignmentStatus.FATAL_DIAGNOSTICS,
        AlignmentStatus.SPEC_ERROR,
        AlignmentStatus.VERIFY_FAILED,
        AlignmentStatus.WRITE_FAILED,
    ],
)
def test_only_ok_counts_as_ok(status: AlignmentStatus) -> None:
    """`ok` must not drift from the status it summarises."""
    assert status is not AlignmentStatus.OK
