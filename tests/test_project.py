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

"""Tests for uniurdf.project.

All tests are hermetic: they use ``tmp_path`` and synthetic URDFs.  No
dependency on external robot assets.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from conftest import PACKAGE_DIR, iter_specs
from uniurdf.project import load_project
from uniurdf.spec import SpecError

# ---------------------------------------------------------------------------
# Synthetic URDF fixtures
# ---------------------------------------------------------------------------

SIMPLE_URDF = b"""<?xml version="1.0"?>
<robot name="simple">
  <link name="base"/>
  <link name="upper"/>
  <link name="tool"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="upper"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="1"/>
  </joint>
  <joint name="wrist" type="prismatic">
    <parent link="upper"/>
    <child link="tool"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.3" effort="10" velocity="1"/>
  </joint>
</robot>
"""

# A URDF with a mesh so we can test the asset copy logic.
MESH_URDF_TEMPLATE = """\
<?xml version="1.0"?>
<robot name="meshed">
  <link name="base">
    <visual>
      <geometry>
        <mesh filename="{mesh_ref}"/>
      </geometry>
    </visual>
  </link>
  <link name="arm"/>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="1"/>
  </joint>
</robot>
"""

# A URDF with a cycle (fatal).
CYCLE_URDF = b"""<?xml version="1.0"?>
<robot name="cycle">
  <link name="a"/>
  <link name="b"/>
  <joint name="ab" type="revolute">
    <parent link="a"/>
    <child link="b"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="ba" type="revolute">
    <parent link="b"/>
    <child link="a"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
"""


# ---------------------------------------------------------------------------
# Helper: write a minimal spec YAML
# ---------------------------------------------------------------------------


def _write_spec(
    path: Path,
    source_urdf: Path | str | None = None,
    output: str | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a minimal YAML spec and return the path.

    Args:
        path (Path): File to write.
        source_urdf (Path | str | None): ``source_urdf`` field value.
        output (str | None): ``output`` field value.
        extra (dict | None): Additional fields to merge in.

    Returns:
        Path: The written spec file.
    """
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
# load_project: path resolution
# ---------------------------------------------------------------------------


class TestLoadProjectPathResolution:
    """Output path resolution precedence and absolute-path printing."""

    def test_explicit_output_wins_over_spec(self, tmp_path: Path, capsys) -> None:
        """--output overrides the spec's output field (decision 2)."""
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(
            tmp_path / "spec.yaml",
            source_urdf=str(urdf_path),
            output="./spec_output",
        )
        explicit = tmp_path / "explicit_output"

        project = load_project(spec_path, output=explicit)
        assert project.output == explicit.resolve()

    def test_spec_output_used_when_no_explicit(self, tmp_path: Path, capsys) -> None:
        """Spec's ``output`` is used when no override is given."""
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(
            tmp_path / "spec.yaml",
            source_urdf=str(urdf_path),
            output="./from_spec",
        )

        # Spec-relative: resolve against the spec file's directory.
        expected = (tmp_path / "from_spec").resolve()
        project = load_project(spec_path)
        assert project.output == expected

    def test_default_output_when_neither_is_set(self, tmp_path: Path, capsys) -> None:
        """Default is workspace/aligned_urdf/<stem> beside the spec."""
        urdf_path = tmp_path / "my_robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(
            tmp_path / "spec.yaml",
            source_urdf=str(urdf_path),
        )

        expected = (tmp_path / "workspace" / "aligned_urdf" / "my_robot").resolve()
        project = load_project(spec_path)
        assert project.output == expected

    def test_output_is_always_absolute(self, tmp_path: Path, capsys) -> None:
        """Regardless of input, ``project.output`` is always absolute."""
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(
            tmp_path / "spec.yaml",
            source_urdf=str(urdf_path),
            output="./relative/path",
        )

        project = load_project(spec_path)
        assert project.output.is_absolute()

    def test_resolves_output_to_an_absolute_path(self, tmp_path: Path, capsys) -> None:
        """The output path is resolved, and reported by return, not stdout.

        A library that writes to stdout cannot be silenced by its caller,
        so the path is carried on the Project instead.
        """
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(
            tmp_path / "spec.yaml",
            source_urdf=str(urdf_path),
        )

        project = load_project(spec_path)
        assert project.output.is_absolute()
        assert capsys.readouterr().out == ""

    def test_spec_relative_not_cwd_relative(self, tmp_path: Path, capsys) -> None:
        """Relative ``output`` is anchored to the spec file, not the CWD."""
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        # spec file is in a subdirectory of tmp_path
        sub = tmp_path / "configs"
        sub.mkdir()
        spec_path = _write_spec(
            sub / "spec.yaml",
            source_urdf=str(urdf_path),
            output="./out",
        )

        # The CWD is not `sub`, so a CWD anchor would land elsewhere.
        project = load_project(spec_path)
        assert project.output == (sub / "out").resolve()
        assert project.output != (Path.cwd() / "out").resolve()

    def test_explicit_output_is_cwd_relative(self, tmp_path: Path, capsys) -> None:
        """``--output`` is a run-time choice, so it anchors to the CWD."""
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        sub = tmp_path / "configs"
        sub.mkdir()
        spec_path = _write_spec(
            sub / "spec.yaml",
            source_urdf=str(urdf_path),
            output="./from_spec",
        )

        project = load_project(spec_path, output=Path("./explicit"))
        assert project.output == (Path.cwd() / "explicit").resolve()

    def test_stem_matches_source_urdf(self, tmp_path: Path, capsys) -> None:
        """project.stem is the source URDF's filename stem."""
        urdf_path = tmp_path / "my_robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(
            tmp_path / "spec.yaml",
            source_urdf=str(urdf_path),
        )

        project = load_project(spec_path)
        assert project.stem == "my_robot"

    def test_missing_source_urdf_raises(self, tmp_path: Path, capsys) -> None:
        """Raise SpecError when source_urdf does not exist."""
        spec_path = _write_spec(
            tmp_path / "spec.yaml",
            source_urdf=str(tmp_path / "ghost.urdf"),
        )
        with pytest.raises(SpecError, match="does not exist"):
            load_project(spec_path)

    def test_spec_without_source_urdf_raises(self, tmp_path: Path, capsys) -> None:
        """Raise SpecError when spec declares no source_urdf."""
        spec_path = _write_spec(tmp_path / "spec.yaml")
        with pytest.raises(SpecError, match="source_urdf"):
            load_project(spec_path)


# ---------------------------------------------------------------------------
# stage: file layout
# ---------------------------------------------------------------------------


class TestBadPathsAreReported:
    """Each error names the path the user typed.

    All three fail deep enough that the OS's own text -- "File exists",
    "Permission denied" -- names the operation and not the path behind it.
    """

    def test_output_pointing_at_a_file(self, tmp_path: Path) -> None:
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")

        project = load_project(spec_path, output=blocker)
        with pytest.raises(SpecError, match="output path is not a directory"):
            project.stage()

    def test_unwritable_output_directory(self, tmp_path: Path) -> None:
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)
        readonly = tmp_path / "readonly"
        readonly.mkdir()
        os.chmod(readonly, 0o500)
        try:
            project = load_project(spec_path, output=readonly)
            with pytest.raises(SpecError, match="could not stage"):
                project.stage()
        finally:
            os.chmod(readonly, 0o700)

    def test_mesh_root_that_does_not_exist(self, tmp_path: Path) -> None:
        """Every mesh resolves through it, so a wrong one is named once."""
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(
            tmp_path / "spec.yaml",
            source_urdf=urdf_path,
            extra={"mesh_root": str(tmp_path / "absent")},
        )
        with pytest.raises(SpecError, match="mesh_root is not a directory"):
            load_project(spec_path)

    def test_mesh_root_pointing_at_a_file(self, tmp_path: Path) -> None:
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        not_a_dir = tmp_path / "meshes.txt"
        not_a_dir.write_text("")
        spec_path = _write_spec(
            tmp_path / "spec.yaml",
            source_urdf=urdf_path,
            extra={"mesh_root": str(not_a_dir)},
        )
        with pytest.raises(SpecError, match="mesh_root is not a directory"):
            load_project(spec_path)


class TestStageFileLayout:
    """The staged output directory has the right structure."""

    def test_staged_file_is_named_stem_dot_source_dot_urdf(
        self, tmp_path: Path, capsys
    ) -> None:
        """Staged URDF is named <stem>.source.urdf."""
        urdf_path = tmp_path / "my_robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)

        out_dir = tmp_path / "output"
        project = load_project(spec_path, output=out_dir)
        staged = project.stage()

        assert staged.staged_path == out_dir / "my_robot.source.urdf"
        assert staged.staged_path.exists()

    def test_output_directory_is_created(self, tmp_path: Path, capsys) -> None:
        """Output directory is created when it does not exist."""
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)

        out_dir = tmp_path / "new_dir" / "nested"
        assert not out_dir.exists()

        project = load_project(spec_path, output=out_dir)
        project.stage()

        assert out_dir.is_dir()

    def test_staged_urdf_content_matches_source(self, tmp_path: Path, capsys) -> None:
        """Staged URDF bytes are identical to the original."""
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)

        out_dir = tmp_path / "out"
        project = load_project(spec_path, output=out_dir)
        staged = project.stage()

        assert staged.staged_path.read_bytes() == SIMPLE_URDF

    def test_staged_robot_is_loaded(self, tmp_path: Path, capsys) -> None:
        """StagedRobot.robot is a populated Robot."""
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)

        out_dir = tmp_path / "out"
        project = load_project(spec_path, output=out_dir)
        staged = project.stage()

        assert staged.robot is not None
        assert "shoulder" in staged.robot.joint_names

    def test_diagnostics_are_attached(self, tmp_path: Path, capsys) -> None:
        """Diagnostics are populated from the staged URDF."""
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)

        out_dir = tmp_path / "out"
        project = load_project(spec_path, output=out_dir)
        staged = project.stage()

        # SIMPLE_URDF has no issues; diagnostics should be empty and non-fatal.
        assert not staged.diagnostics.is_fatal
        assert staged.diagnostics.findings == ()

    def test_mesh_copied_beside_urdf(self, tmp_path: Path, capsys) -> None:
        """A resolvable mesh is copied next to the staged URDF."""
        mesh_dir = tmp_path / "meshes"
        mesh_dir.mkdir()
        mesh_file = mesh_dir / "base.stl"
        mesh_file.write_bytes(b"SOLID stub")

        urdf_text = MESH_URDF_TEMPLATE.format(mesh_ref="meshes/base.stl")
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_text(urdf_text)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)

        out_dir = tmp_path / "out"
        project = load_project(spec_path, output=out_dir)
        staged = project.stage()

        # Mesh should be copied to out_dir/meshes/base.stl
        assert (out_dir / "meshes" / "base.stl").exists()
        assert len(staged.asset.local_mesh_files) == 1

    def test_unresolvable_mesh_in_report(self, tmp_path: Path, capsys) -> None:
        """An unresolvable mesh ref ends up in asset.unresolved_meshes."""
        urdf_text = MESH_URDF_TEMPLATE.format(mesh_ref="does/not/exist.stl")
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_text(urdf_text)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)

        out_dir = tmp_path / "out"
        project = load_project(spec_path, output=out_dir)
        staged = project.stage()

        assert "does/not/exist.stl" in staged.asset.unresolved_meshes


class TestStageFatalDiagnostics:
    """Staging a URDF with fatal defects populates diagnostics but succeeds."""

    def test_cycle_urdf_stages_with_fatal_diagnostics(
        self, tmp_path: Path, capsys
    ) -> None:
        """Fatal diagnostics are attached; stage() itself does not raise."""
        urdf_path = tmp_path / "cycle.urdf"
        urdf_path.write_bytes(CYCLE_URDF)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)

        out_dir = tmp_path / "out"
        project = load_project(spec_path, output=out_dir)
        # Stage must succeed regardless of diagnostics (the GUI must open
        # broken URDFs so the user can fix them — decision 17).
        staged = project.stage()

        assert staged.diagnostics.is_fatal
        assert any(f.code == "cycle" for f in staged.diagnostics.fatal)


# ---------------------------------------------------------------------------
# Re-run: overwrite with warning
# ---------------------------------------------------------------------------


class TestReRun:
    """Re-running writes over existing output with a warning."""

    def test_overwrite_issues_warning(self, tmp_path: Path, capsys) -> None:
        """Re-staging an existing output dir emits a UserWarning."""
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)
        out_dir = tmp_path / "out"

        project = load_project(spec_path, output=out_dir)
        project.stage()  # First run

        # Second run must warn, not raise.
        with pytest.warns(UserWarning, match="overwriting"):
            project.stage()

    def test_overwrite_preserves_urdf_content(self, tmp_path: Path, capsys) -> None:
        """After overwrite the staged URDF still has the correct content."""
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)
        out_dir = tmp_path / "out"

        project = load_project(spec_path, output=out_dir)
        project.stage()

        with pytest.warns(UserWarning):
            staged = project.stage()

        assert staged.staged_path.read_bytes() == SIMPLE_URDF

    def test_overwrite_is_reported_through_warnings_only(
        self, tmp_path: Path, capsys
    ) -> None:
        """One condition, one channel.

        Reporting through both `warnings` and stderr would print twice for a
        caller that has not silenced warnings, and the caller could suppress
        only one of them.
        """
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)
        out_dir = tmp_path / "out"

        project = load_project(spec_path, output=out_dir)
        project.stage()
        capsys.readouterr()

        with pytest.warns(UserWarning, match="overwriting"):
            project.stage()

        assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# Same-directory re-run: the second-iteration workflow
# ---------------------------------------------------------------------------


class TestSameDirectoryReRun:
    """Second iteration: source_urdf is already in the output directory."""

    def test_source_in_output_dir_leaves_meshes_in_place(
        self, tmp_path: Path, capsys
    ) -> None:
        """When source dir == output dir, meshes are left untouched.

        Each ref already sits where it would be copied, so the copy is a
        no-op — but it still resolves, so nothing is reported missing.
        """
        mesh_dir = tmp_path / "out" / "meshes"
        mesh_dir.mkdir(parents=True)
        mesh = mesh_dir / "part.stl"
        mesh.write_bytes(b"SOLID x")

        # Create the URDF directly in out/ (simulating prior staging).
        urdf_path = tmp_path / "out" / "robot.source.urdf"
        urdf_text = MESH_URDF_TEMPLATE.format(mesh_ref="meshes/part.stl")
        urdf_path.write_text(urdf_text)

        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)
        out_dir = tmp_path / "out"

        project = load_project(spec_path, output=out_dir)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            staged = project.stage()

        assert mesh.read_bytes() == b"SOLID x"
        assert staged.asset.unresolved_meshes == ()

    def test_restaging_reuses_the_same_filenames(self, tmp_path: Path, capsys) -> None:
        """Staging robot.source.urdf stages back onto robot.source.urdf."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # Simulate a repaired staged copy already named .source.urdf.
        urdf_path = out_dir / "robot.source.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)

        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)

        # The stem drops the trailing `.source` rather than compounding it,
        # so the names are a fixed point across iterations.
        project = load_project(spec_path, output=out_dir)
        assert project.stem == "robot"

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            staged = project.stage()

        assert staged.staged_path.name == "robot.source.urdf"
        assert staged.staged_path.exists()
        assert not (out_dir / "robot.source.source.urdf").exists()

    def test_restaging_onto_itself_preserves_the_file(
        self, tmp_path: Path, capsys
    ) -> None:
        """Staging a file onto its own path must not empty or delete it."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        urdf_path = out_dir / "robot.source.urdf"
        edited = SIMPLE_URDF.replace(b"simple", b"hand_repaired")
        urdf_path.write_bytes(edited)

        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)
        project = load_project(spec_path, output=out_dir)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            staged = project.stage()

        # The staged path IS the source path here; the copy must be a no-op
        # rather than an unlink-then-copy, which would destroy the only copy.
        assert staged.staged_path.samefile(urdf_path)
        assert staged.staged_path.read_bytes() == edited

    def test_data_not_lost_on_same_dir_restage(self, tmp_path: Path, capsys) -> None:
        """Re-staging in same dir does not destroy in-place edits."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        urdf_path = out_dir / "robot.source.urdf"
        # Write "edited" content to simulate in-place repair.
        urdf_path.write_bytes(SIMPLE_URDF.replace(b"simple", b"edited"))

        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)
        project = load_project(spec_path, output=out_dir)

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            staged = project.stage()

        # The staged copy should carry the edited content.
        assert b"edited" in staged.staged_path.read_bytes()

    def test_repeated_iterations_do_not_lose_the_hand_edits(
        self, tmp_path: Path, capsys
    ) -> None:
        """The edit-and-re-run loop is repeatable without destroying work.

        Run once, edit the staged copy, point the spec at it, then re-run
        several times.  The staged copy must keep the hand edits and the
        aligned output must stay a separate file throughout.
        """
        import warnings

        src = tmp_path / "orig" / "robot.urdf"
        src.parent.mkdir()
        src.write_bytes(SIMPLE_URDF)
        out_dir = tmp_path / "out"

        def run(source: Path, tag: str) -> None:
            spec_path = _write_spec(tmp_path / f"spec_{tag}.yaml", source_urdf=source)
            project = load_project(spec_path, output=out_dir)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                staged = project.stage()
            aligned = staged.align(project.spec)
            aligned.write(out_dir, name=project.stem + ".urdf")

        run(src, "1")
        staged_copy = out_dir / "robot.source.urdf"
        # Matched on the link name alone: alignment re-serializes the XML, so
        # the exact tag text is not preserved even though the link is.
        marker = b"hand_added_by_the_user"
        staged_copy.write_bytes(
            staged_copy.read_bytes().replace(
                b"</robot>", b'<link name="' + marker + b'"/></robot>'
            )
        )

        for tag in ("2", "3", "4"):
            run(staged_copy, tag)
            assert marker in staged_copy.read_bytes(), (
                f"iteration {tag} destroyed the user's edit"
            )

        # Exactly two URDFs: the working copy and the aligned result.
        assert sorted(p.name for p in out_dir.glob("*.urdf")) == [
            "robot.source.urdf",
            "robot.urdf",
        ]
        # The aligned output is a distinct file that picked up the edit.
        assert marker in (out_dir / "robot.urdf").read_bytes()


# ---------------------------------------------------------------------------
# StagedRobot.align delegation
# ---------------------------------------------------------------------------


class TestStagedRobotAlign:
    """StagedRobot.align delegates to Robot.align."""

    def test_align_returns_aligned_robot(self, tmp_path: Path, capsys) -> None:
        """StagedRobot.align produces an AlignedRobot."""
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)

        project = load_project(spec_path, output=tmp_path / "out")
        staged = project.stage()

        from uniurdf.robot import AlignedRobot

        aligned = staged.align(project.spec)
        assert isinstance(aligned, AlignedRobot)

    def test_align_normalizes_axes(self, tmp_path: Path, capsys) -> None:
        """Alignment via StagedRobot finds the non-+Z joint in SIMPLE_URDF."""
        urdf_path = tmp_path / "robot.urdf"
        urdf_path.write_bytes(SIMPLE_URDF)
        spec_path = _write_spec(tmp_path / "spec.yaml", source_urdf=urdf_path)

        project = load_project(spec_path, output=tmp_path / "out")
        staged = project.stage()
        aligned = staged.align(project.spec)

        # wrist joint has axis xyz="1 0 0" — should be normalized.
        # shoulder has axis xyz="0 0 1" — already +Z, not normalized.
        # Exact set depends on normalize_actuated_axes; at least one joint
        # should be in the normalized list.
        assert len(aligned.report.normalized_joints) >= 1


# ---------------------------------------------------------------------------
# Shipped-config gate
# ---------------------------------------------------------------------------

CONFIG_DIR = PACKAGE_DIR / "example"


def _config_paths() -> list[Path]:
    return iter_specs()


@pytest.mark.parametrize("config_path", _config_paths(), ids=lambda p: p.stem)
def test_run_gate_all_configs(config_path: Path, tmp_path: Path, capsys) -> None:
    """Every shipped config stages, aligns, verifies, and exits cleanly.

    A spec that ships names a robot this gate is meant to cover, so a URDF
    that cannot be reached fails here. Skipping would leave the gate green
    on a machine where it checked nothing.

    Many configs resolve zero meshes from their source trees, because the
    meshes live under a package:// URI that is not on the current machine.
    That is expected and must not fail this gate.
    """
    import warnings

    project = load_project(config_path, output=tmp_path / "out")

    # Stage: must succeed regardless of diagnostics.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        staged = project.stage()

    # Fatal diagnostics would block `run`, but the test gate only checks
    # that the workflow completes without crashing.  A config whose source
    # URDF is fatally broken would be a real bug; skip if fatal so we do
    # not mask it with a confusing error from the alignment step.
    if staged.diagnostics.is_fatal:
        codes = [f.code for f in staged.diagnostics.fatal]
        pytest.fail(f"{config_path.stem}: staged URDF has fatal diagnostics: {codes}")

    # Align.
    aligned = staged.align(project.spec)
    assert aligned is not None

    # Verify.
    report = aligned.verify()
    assert report.ok, (
        f"{config_path.stem}: verification failed — "
        f"moved_links={report.moved_links[:3]}, "
        f"moved_meshes={report.moved_meshes[:3]}"
    )
