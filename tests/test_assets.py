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

"""Synthetic mesh-copy tests for uniurdf.assets.

All tests are fully hermetic: tmp_path only, no external assets.
The fixtures exercise each code path in ``copy_urdf_asset_tree``:

- Nested mesh sub-directories (standard case).
- Duplicate basenames in different directories (must not collide on copy).
- A sidecar ``.mtl`` file beside an ``.obj`` mesh.
- A ref that is present in the URDF but whose file does not exist (unresolved).
- A ``../`` escaping ref that would land outside the output root.
- A ``package://pkg/rel`` ref that resolves under mesh_root.
- A ``package://`` ref that does NOT resolve (left verbatim).
- Source dir == output dir (each copy is a no-op; an explicit mesh_root
  pointing elsewhere is still honoured).
- A read-only (0444) source mesh (re-run must overwrite cleanly).
- Running the copy twice (idempotent).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from uniurdf.assets import (
    CopiedUrdfAsset,
    copy_urdf_asset_tree,
    in_tree_mesh_path,
    rewrite_resolved_package_uris,
)


def _run(robot_tree: dict) -> CopiedUrdfAsset:
    """Copy a fixture tree with the arguments every test here uses."""
    return copy_urdf_asset_tree(
        source_urdf=robot_tree["source_urdf"],
        target_urdf=robot_tree["output"] / "robot.urdf",
        mesh_root=robot_tree["mesh_root"],
    )


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _write_urdf(directory: Path, refs: list[str]) -> Path:
    """Write a minimal URDF with one mesh element per ref string.

    Args:
        directory: Directory to create the URDF in.
        refs: List of ``filename`` attribute values.

    Returns:
        Path to the written URDF.
    """
    directory.mkdir(parents=True, exist_ok=True)
    meshes_xml = "\n    ".join(f'<mesh filename="{ref}"/>' for ref in refs)
    urdf_text = f"""\
<?xml version="1.0"?>
<robot name="test">
  <link name="base"/>
  <link name="arm">
    <visual>
      <geometry>
        {meshes_xml}
      </geometry>
    </visual>
  </link>
</robot>
"""
    path = directory / "robot.urdf"
    path.write_text(urdf_text)
    return path


def _make_mesh(path: Path, content: bytes = b"STL ") -> Path:
    """Create a minimal mesh file at ``path``.

    Args:
        path: Destination path (created with parents).
        content: File content. Defaults to a 4-byte STL signature.

    Returns:
        The created path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture()
def robot_tree(tmp_path: Path) -> dict:
    """A robot source tree exercising most copy paths.

    Layout::

        source/
          robot.urdf          -- URDF with all refs below
          meshes/
            visual/a.stl      -- nested sub-directory
            collision/b.stl   -- different sub-directory, same level
            left/base.stl     -- basename shared with right/base.stl
            right/base.stl    -- same basename, different directory
            part.obj          -- .obj with a sidecar .mtl
            part.mtl          -- sidecar
          shared_pkg/
            rod.stl           -- package:// ref resolves here
        source_parent/
          meshes/
            escaped.stl       -- mesh ../up from source/

    The URDF references::

        meshes/visual/a.stl           (nested, resolves)
        meshes/collision/b.stl        (nested, resolves)
        meshes/left/base.stl          (resolves; same basename as right/base)
        meshes/right/base.stl         (resolves; different dir)
        meshes/part.obj               (obj + mtl sidecar)
        meshes/ghost.stl              (does NOT exist — unresolved)
        ../escaped.stl                (escaping ref)
        package://shared_pkg/rod.stl  (package://, resolves)
        package://no_such_pkg/x.stl   (package://, does NOT resolve)

    Returns:
        dict with keys:
            ``source_urdf``, ``mesh_root``, ``output``,
            ``expected_resolved_refs`` (refs that should produce copied files).
    """
    source = tmp_path / "source"
    source.mkdir()

    # Create mesh files.
    _make_mesh(source / "meshes" / "visual" / "a.stl")
    _make_mesh(source / "meshes" / "collision" / "b.stl")
    _make_mesh(source / "meshes" / "left" / "base.stl", b"LEFT")
    _make_mesh(source / "meshes" / "right" / "base.stl", b"RIGHT")
    _make_mesh(source / "meshes" / "part.obj", b"OBJ ")
    (source / "meshes" / "part.mtl").write_text("mtl")
    _make_mesh(source / "shared_pkg" / "rod.stl")

    # Escaped mesh: sits at source_parent level.
    escaped_source = tmp_path / "escaped.stl"
    _make_mesh(escaped_source)

    refs = [
        "meshes/visual/a.stl",
        "meshes/collision/b.stl",
        "meshes/left/base.stl",
        "meshes/right/base.stl",
        "meshes/part.obj",
        "meshes/ghost.stl",
        "../escaped.stl",
        "package://shared_pkg/rod.stl",
        "package://no_such_pkg/x.stl",
    ]
    source_urdf = _write_urdf(source, refs)
    output = tmp_path / "output"

    return {
        "source_urdf": source_urdf,
        "mesh_root": source,
        "output": output,
    }


# ---------------------------------------------------------------------------
# Tests: core copy behaviour.
# ---------------------------------------------------------------------------


class TestNestedSubdirectories:
    """Nested mesh subdirectories are copied at their full relative path."""

    def test_visual_subdir_exists(self, robot_tree: dict, tmp_path: Path) -> None:
        result = copy_urdf_asset_tree(
            source_urdf=robot_tree["source_urdf"],
            target_urdf=robot_tree["output"] / "robot.urdf",
            mesh_root=robot_tree["mesh_root"],
        )
        assert (robot_tree["output"] / "meshes" / "visual" / "a.stl").exists()
        assert (
            robot_tree["output"] / "meshes" / "visual" / "a.stl"
        ) in result.copied_files

    def test_collision_subdir_exists(self, robot_tree: dict) -> None:
        copy_urdf_asset_tree(
            source_urdf=robot_tree["source_urdf"],
            target_urdf=robot_tree["output"] / "robot.urdf",
            mesh_root=robot_tree["mesh_root"],
        )
        assert (robot_tree["output"] / "meshes" / "collision" / "b.stl").exists()


class TestDuplicateBasenames:
    """Two refs with the same basename in different dirs must not collide."""

    def test_left_and_right_both_exist(self, robot_tree: dict) -> None:
        copy_urdf_asset_tree(
            source_urdf=robot_tree["source_urdf"],
            target_urdf=robot_tree["output"] / "robot.urdf",
            mesh_root=robot_tree["mesh_root"],
        )
        left = robot_tree["output"] / "meshes" / "left" / "base.stl"
        right = robot_tree["output"] / "meshes" / "right" / "base.stl"
        assert left.exists()
        assert right.exists()

    def test_contents_not_mixed(self, robot_tree: dict) -> None:
        copy_urdf_asset_tree(
            source_urdf=robot_tree["source_urdf"],
            target_urdf=robot_tree["output"] / "robot.urdf",
            mesh_root=robot_tree["mesh_root"],
        )
        left = robot_tree["output"] / "meshes" / "left" / "base.stl"
        right = robot_tree["output"] / "meshes" / "right" / "base.stl"
        assert left.read_bytes() == b"LEFT"
        assert right.read_bytes() == b"RIGHT"


class TestSidecarMtl:
    """An .obj's .mtl sidecar is copied alongside it."""

    def test_mtl_beside_obj(self, robot_tree: dict) -> None:
        copy_urdf_asset_tree(
            source_urdf=robot_tree["source_urdf"],
            target_urdf=robot_tree["output"] / "robot.urdf",
            mesh_root=robot_tree["mesh_root"],
        )
        obj = robot_tree["output"] / "meshes" / "part.obj"
        mtl = robot_tree["output"] / "meshes" / "part.mtl"
        assert obj.exists()
        assert mtl.exists()

    def test_mtl_in_copied_files(self, robot_tree: dict) -> None:
        result = copy_urdf_asset_tree(
            source_urdf=robot_tree["source_urdf"],
            target_urdf=robot_tree["output"] / "robot.urdf",
            mesh_root=robot_tree["mesh_root"],
        )
        mtl = robot_tree["output"] / "meshes" / "part.mtl"
        assert mtl in result.copied_files


# ---------------------------------------------------------------------------
# Tests: unresolved refs.
# ---------------------------------------------------------------------------


class TestUnresolvedRef:
    """A missing mesh is reported verbatim and is not fatal."""

    def test_listed_in_unresolved_meshes(self, robot_tree: dict) -> None:
        result = copy_urdf_asset_tree(
            source_urdf=robot_tree["source_urdf"],
            target_urdf=robot_tree["output"] / "robot.urdf",
            mesh_root=robot_tree["mesh_root"],
        )
        assert "meshes/ghost.stl" in result.unresolved_meshes

    def test_not_created_on_disk(self, robot_tree: dict) -> None:
        copy_urdf_asset_tree(
            source_urdf=robot_tree["source_urdf"],
            target_urdf=robot_tree["output"] / "robot.urdf",
            mesh_root=robot_tree["mesh_root"],
        )
        assert not (robot_tree["output"] / "meshes" / "ghost.stl").exists()

    def test_skipped_meshes_alias(self, robot_tree: dict) -> None:
        """skipped_meshes is a backwards-compat alias for unresolved_meshes."""
        result = copy_urdf_asset_tree(
            source_urdf=robot_tree["source_urdf"],
            target_urdf=robot_tree["output"] / "robot.urdf",
            mesh_root=robot_tree["mesh_root"],
        )
        assert result.skipped_meshes == result.unresolved_meshes

    def test_unresolved_package_ref(self, robot_tree: dict) -> None:
        """A package:// ref that does not resolve is also reported."""
        result = copy_urdf_asset_tree(
            source_urdf=robot_tree["source_urdf"],
            target_urdf=robot_tree["output"] / "robot.urdf",
            mesh_root=robot_tree["mesh_root"],
        )
        assert "package://no_such_pkg/x.stl" in result.unresolved_meshes


# ---------------------------------------------------------------------------
# Tests: escaping refs.
# ---------------------------------------------------------------------------


class TestEscapingRef:
    """A ``../`` ref that would escape the output root is rewritten."""

    def test_listed_in_escaped_meshes(self, robot_tree: dict) -> None:
        result = _run(robot_tree)
        originals = [pair[0] for pair in result.escaped_meshes]
        assert "../escaped.stl" in originals

    def test_escaped_mesh_lands_inside_output(self, robot_tree: dict) -> None:
        _run(robot_tree)
        # The output directory must contain the file.
        output_dir = robot_tree["output"]
        escaped_files = list(output_dir.rglob("escaped.stl"))
        assert len(escaped_files) == 1
        # And it must be under output_dir, not beside it.
        escaped_files[0].relative_to(output_dir)  # raises ValueError if not

    def test_rewritten_ref_uses_up_prefix(self, robot_tree: dict) -> None:
        result = _run(robot_tree)
        rewritten = [pair[1] for pair in result.escaped_meshes]
        # The rewritten path must start with the _up_ prefix.
        assert any(r.startswith("_up_") for r in rewritten)

    def test_staged_urdf_contains_rewritten_ref(self, robot_tree: dict) -> None:
        result = _run(robot_tree)
        staged_text = result.urdf.read_text()
        assert "../escaped.stl" not in staged_text
        # The rewritten ref must appear in the staged copy.
        rewritten = next(
            pair[1] for pair in result.escaped_meshes if pair[0] == "../escaped.stl"
        )
        assert rewritten in staged_text


class TestEscapingPackageRef:
    """``package://`` refs are contained the same way relative ones are.

    The scheme is stripped and the remainder split on its first ``/``, so the
    package name can be ``..`` and address a file above ``mesh_root``. Copying
    such a ref verbatim writes outside the output directory, over a file the
    run was never pointed at.
    """

    @staticmethod
    def _tree(tmp_path: Path) -> tuple[Path, Path, Path]:
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        outside = tmp_path / "outside.stl"
        _make_mesh(outside)

        urdf = source_dir / "robot.urdf"
        urdf.write_bytes(
            b"""<?xml version="1.0"?>
<robot name="r">
  <link name="base">
    <visual>
      <geometry><mesh filename="package://../outside.stl"/></geometry>
    </visual>
  </link>
</robot>"""
        )
        output_dir = tmp_path / "out" / "deep"
        return urdf, source_dir, output_dir

    def test_copy_stays_inside_the_output_directory(self, tmp_path: Path) -> None:
        urdf, source_dir, output_dir = self._tree(tmp_path)
        result = copy_urdf_asset_tree(
            source_urdf=urdf,
            target_urdf=output_dir / "robot.source.urdf",
            mesh_root=source_dir,
        )
        for copied in result.copied_files:
            copied.resolve().relative_to(output_dir.resolve())

    def test_a_file_beside_the_output_directory_survives(self, tmp_path: Path) -> None:
        urdf, source_dir, output_dir = self._tree(tmp_path)
        output_dir.mkdir(parents=True)
        bystander = output_dir.parent / "outside.stl"
        bystander.write_text("PRE-EXISTING")

        copy_urdf_asset_tree(
            source_urdf=urdf,
            target_urdf=output_dir / "robot.source.urdf",
            mesh_root=source_dir,
        )
        assert bystander.read_text() == "PRE-EXISTING"

    def test_reported_as_escaped(self, tmp_path: Path) -> None:
        urdf, source_dir, output_dir = self._tree(tmp_path)
        result = copy_urdf_asset_tree(
            source_urdf=urdf,
            target_urdf=output_dir / "robot.source.urdf",
            mesh_root=source_dir,
        )
        originals = [pair[0] for pair in result.escaped_meshes]
        assert "package://../outside.stl" in originals


# ---------------------------------------------------------------------------
# Tests: package:// refs.
# ---------------------------------------------------------------------------


class TestPackageUri:
    """package:// refs that resolve are copied and rewritten."""

    def test_package_mesh_copied(self, robot_tree: dict) -> None:
        _run(robot_tree)
        assert (robot_tree["output"] / "shared_pkg" / "rod.stl").exists()

    def test_package_rewrite_recorded(self, robot_tree: dict) -> None:
        result = _run(robot_tree)
        assert "package://shared_pkg/rod.stl" in result.package_uri_rewrites
        assert (
            result.package_uri_rewrites["package://shared_pkg/rod.stl"]
            == "shared_pkg/rod.stl"
        )

    def test_rewrite_resolved_package_uris(self, robot_tree: dict) -> None:
        """rewrite_resolved_package_uris rewrites resolved refs only."""
        result = _run(robot_tree)
        aligned_bytes = result.urdf.read_bytes()
        aligned_bytes = rewrite_resolved_package_uris(
            aligned_bytes, result.package_uri_rewrites
        )
        text = aligned_bytes.decode()
        # Resolved ref is rewritten.
        assert 'filename="shared_pkg/rod.stl"' in text
        # Unresolved ref is left verbatim.
        assert 'filename="package://no_such_pkg/x.stl"' in text

    def test_unresolved_package_left_verbatim(self, robot_tree: dict) -> None:
        result = _run(robot_tree)
        staged_text = result.urdf.read_text()
        assert 'filename="package://no_such_pkg/x.stl"' in staged_text


class TestInTreeMeshPath:
    """The copy layout, stated once so readers agree with the writer.

    Anything reading meshes back out of the output tree needs the same
    mapping copying used. The viewer restated it and got `package://` wrong,
    fetching the literal directory `package:/<pkg>/...`.
    """

    def test_package_ref_loses_only_its_scheme(self) -> None:
        assert in_tree_mesh_path("package://shared_pkg/rod.stl") == "shared_pkg/rod.stl"

    def test_relative_ref_is_unchanged(self) -> None:
        assert in_tree_mesh_path("meshes/rod.stl") == "meshes/rod.stl"

    def test_leading_dot_slash_is_dropped(self) -> None:
        assert in_tree_mesh_path("./meshes/rod.stl") == "meshes/rod.stl"

    def test_agrees_with_where_the_copy_landed(self, robot_tree: dict) -> None:
        """The gate that matters: prediction must match the filesystem.

        Checked against the refs as they appear in the *staged* URDF, which
        is what a reader parses. Escaping refs are already rewritten there
        at copy time, so `_up_1/escaped.stl` is what gets read back, never
        the original `../escaped.stl`.
        """
        result = copy_urdf_asset_tree(
            source_urdf=robot_tree["source_urdf"],
            target_urdf=robot_tree["output"] / "robot.urdf",
            mesh_root=robot_tree["mesh_root"],
        )
        output_dir = robot_tree["output"]
        staged_refs = re.findall(
            r'<mesh\b[^>]*?\bfilename\s*=\s*"([^"]+)"',
            result.urdf.read_text(),
        )
        assert staged_refs, "fixture should declare mesh refs"

        resolved = 0
        for reference in staged_refs:
            predicted = output_dir / in_tree_mesh_path(reference)
            if reference in result.unresolved_meshes:
                assert not predicted.exists()
                continue
            assert predicted.is_file(), f"{reference} -> {predicted}"
            resolved += 1
        # Both arms of the branch above must actually be exercised.
        assert resolved and result.unresolved_meshes


# ---------------------------------------------------------------------------
# Tests: source dir == output dir.
# ---------------------------------------------------------------------------


class TestSourceDirEqualsOutputDir:
    """When source and output are the same dir, each copy is a no-op."""

    def test_no_error(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "robot"
        source_dir.mkdir()
        _make_mesh(source_dir / "arm.stl")
        urdf = _write_urdf(source_dir, ["arm.stl"])
        # target_urdf is in the same directory as source_urdf.
        target = source_dir / "robot.source.urdf"
        copy_urdf_asset_tree(source_urdf=urdf, target_urdf=target, mesh_root=source_dir)

    def test_existing_files_not_deleted(self, tmp_path: Path) -> None:
        """Files in the shared directory are not unlinked during the copy."""
        source_dir = tmp_path / "robot"
        source_dir.mkdir()
        mesh = _make_mesh(source_dir / "arm.stl", b"KEEP")
        urdf = _write_urdf(source_dir, ["arm.stl"])
        target = source_dir / "robot.source.urdf"
        copy_urdf_asset_tree(source_urdf=urdf, target_urdf=target, mesh_root=source_dir)
        assert mesh.read_bytes() == b"KEEP"

    def test_report_lists_the_in_place_mesh(self, tmp_path: Path) -> None:
        """A ref already at its target still counts as copied, not missing."""
        source_dir = tmp_path / "robot"
        source_dir.mkdir()
        _make_mesh(source_dir / "arm.stl")
        urdf = _write_urdf(source_dir, ["arm.stl"])
        target = source_dir / "robot.source.urdf"
        result = copy_urdf_asset_tree(
            source_urdf=urdf, target_urdf=target, mesh_root=source_dir
        )
        assert result.unresolved_meshes == ()
        assert source_dir / "arm.stl" in result.local_mesh_files

    def test_missing_mesh_is_reported_not_silently_accepted(
        self, tmp_path: Path
    ) -> None:
        """A ref with no file must be reported even on a same-dir re-run.

        Staying silent here let a run claim every mesh resolved while writing
        a URDF that pointed at files which were never there.
        """
        source_dir = tmp_path / "robot"
        source_dir.mkdir()
        urdf = _write_urdf(source_dir, ["arm.stl"])  # no mesh file created
        target = source_dir / "robot.source.urdf"
        result = copy_urdf_asset_tree(
            source_urdf=urdf, target_urdf=target, mesh_root=source_dir
        )
        assert result.unresolved_meshes == ("arm.stl",)

    def test_an_explicit_mesh_root_still_copies(self, tmp_path: Path) -> None:
        """A mesh_root naming another tree is honoured, not skipped.

        This is how a run whose refs never resolved gets its meshes: the user
        adds the mesh_root and re-runs against the staged copy in place.
        """
        source_dir = tmp_path / "robot"
        source_dir.mkdir()
        elsewhere = tmp_path / "assets"
        (elsewhere / "meshes").mkdir(parents=True)
        _make_mesh(elsewhere / "meshes" / "arm.stl", b"UPSTREAM")
        urdf = _write_urdf(source_dir, ["meshes/arm.stl"])
        target = source_dir / "robot.source.urdf"

        result = copy_urdf_asset_tree(
            source_urdf=urdf, target_urdf=target, mesh_root=elsewhere
        )

        assert result.unresolved_meshes == ()
        assert (source_dir / "meshes" / "arm.stl").read_bytes() == b"UPSTREAM"


# ---------------------------------------------------------------------------
# Tests: read-only source mesh.
# ---------------------------------------------------------------------------


class TestReadOnlySourceMesh:
    """A 0444 source mesh can be overwritten on re-run."""

    def test_readonly_mesh_copies_on_first_run(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        mesh = _make_mesh(source / "arm.stl", b"V1")
        mesh.chmod(0o444)
        urdf = _write_urdf(source, ["arm.stl"])
        output = tmp_path / "output" / "robot.urdf"
        copy_urdf_asset_tree(source_urdf=urdf, target_urdf=output, mesh_root=source)
        assert (tmp_path / "output" / "arm.stl").exists()

    def test_readonly_output_overwritten_on_rerun(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        source.mkdir()
        mesh = _make_mesh(source / "arm.stl", b"V1")
        mesh.chmod(0o444)
        urdf = _write_urdf(source, ["arm.stl"])
        output_urdf = tmp_path / "output" / "robot.urdf"
        # First run.
        copy_urdf_asset_tree(
            source_urdf=urdf, target_urdf=output_urdf, mesh_root=source
        )
        copied = tmp_path / "output" / "arm.stl"
        assert copied.read_bytes() == b"V1"

        # Simulate upstream asset update.
        mesh.chmod(0o644)
        mesh.write_bytes(b"V2")
        mesh.chmod(0o444)

        # Second run must overwrite the read-only copy in the output tree.
        copy_urdf_asset_tree(
            source_urdf=urdf, target_urdf=output_urdf, mesh_root=source
        )
        assert copied.read_bytes() == b"V2"


# ---------------------------------------------------------------------------
# Tests: idempotency.
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Running the copy twice produces the same result."""

    def test_twice_same_files(self, robot_tree: dict) -> None:
        kwargs = {
            "source_urdf": robot_tree["source_urdf"],
            "target_urdf": robot_tree["output"] / "robot.urdf",
            "mesh_root": robot_tree["mesh_root"],
        }
        result1 = copy_urdf_asset_tree(**kwargs)
        result2 = copy_urdf_asset_tree(**kwargs)
        assert set(result1.copied_files) == set(result2.copied_files)
        assert result1.unresolved_meshes == result2.unresolved_meshes

    def test_twice_same_content(self, robot_tree: dict) -> None:
        kwargs = {
            "source_urdf": robot_tree["source_urdf"],
            "target_urdf": robot_tree["output"] / "robot.urdf",
            "mesh_root": robot_tree["mesh_root"],
        }
        copy_urdf_asset_tree(**kwargs)
        snapshot = {
            p: p.read_bytes() for p in robot_tree["output"].rglob("*") if p.is_file()
        }
        copy_urdf_asset_tree(**kwargs)
        for path, content in snapshot.items():
            assert path.read_bytes() == content


# ---------------------------------------------------------------------------
# Tests: report shape.
# ---------------------------------------------------------------------------


class TestReportShape:
    """CopiedUrdfAsset fields have the expected types and contents."""

    def test_urdf_in_copied_files(self, robot_tree: dict) -> None:
        result = _run(robot_tree)
        assert result.urdf in result.copied_files

    def test_local_mesh_files_excludes_urdf(self, robot_tree: dict) -> None:
        result = _run(robot_tree)
        assert result.urdf not in result.local_mesh_files

    def test_all_local_mesh_files_exist(self, robot_tree: dict) -> None:
        result = _run(robot_tree)
        for path in result.local_mesh_files:
            assert path.exists(), f"Expected {path} to exist"

    def test_all_local_files_under_output(self, robot_tree: dict) -> None:
        result = _run(robot_tree)
        output_dir = robot_tree["output"]
        for path in result.local_mesh_files:
            path.relative_to(output_dir)  # raises ValueError if not inside

    def test_unresolved_tuple_of_strings(self, robot_tree: dict) -> None:
        result = _run(robot_tree)
        assert isinstance(result.unresolved_meshes, tuple)
        for ref in result.unresolved_meshes:
            assert isinstance(ref, str)

    def test_escaped_tuple_of_pairs(self, robot_tree: dict) -> None:
        result = _run(robot_tree)
        assert isinstance(result.escaped_meshes, tuple)
        for pair in result.escaped_meshes:
            assert isinstance(pair, tuple) and len(pair) == 2


# ---------------------------------------------------------------------------
# Tests: default mesh_root (= source URDF's parent).
# ---------------------------------------------------------------------------


class TestDefaultMeshRoot:
    """When mesh_root is omitted, it defaults to source_urdf.parent."""

    def test_resolves_mesh_in_source_dir(self, tmp_path: Path) -> None:
        source_dir = tmp_path / "robot"
        source_dir.mkdir()
        _make_mesh(source_dir / "arm.stl")
        urdf = _write_urdf(source_dir, ["arm.stl"])
        output = tmp_path / "out" / "robot.urdf"
        # No mesh_root argument.
        result = copy_urdf_asset_tree(source_urdf=urdf, target_urdf=output)
        assert (tmp_path / "out" / "arm.stl").exists()
        assert result.unresolved_meshes == ()
