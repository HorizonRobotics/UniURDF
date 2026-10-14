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

"""Tests for the Robot / AlignedRobot interface.

This is the seam the CLI, GUI, and everything else go through, so these tests
double as the specification of what alignment guarantees: geometry is
preserved, output is deterministic, and re-running changes nothing.
"""

import dataclasses
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from uniurdf.robot import load_robot
from uniurdf.spec import SpecError, parse_spec

ARM_URDF = b"""<?xml version="1.0"?>
<robot name="arm">
  <link name="base"/><link name="l1"/><link name="l2"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/><axis xyz="0 0 -1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
  </joint>
</robot>"""

TCP_SPEC = {
    "chains": [["j1", "j2"]],
    "tcp": [{"base_link": "l2", "xyz": [0, 0, 0.15], "rpy": [0, 0, 270]}],
}


def _links(urdf: bytes) -> set[str]:
    return {link.get("name", "") for link in ET.fromstring(urdf).findall("link")}


class TestLoadRobot:
    """Reading a URDF."""

    def test_from_bytes(self) -> None:
        robot = load_robot(ARM_URDF)
        assert robot.path is None
        assert robot.joint_names == ("j1", "j2")
        assert robot.link_names == ("base", "l1", "l2")

    def test_from_path(self, tmp_path: Path) -> None:
        path = tmp_path / "arm.urdf"
        path.write_bytes(ARM_URDF)
        assert load_robot(path).path == path

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SpecError, match="cannot read URDF"):
            load_robot(tmp_path / "absent.urdf")

    def test_malformed_urdf(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.urdf"
        path.write_bytes(b"<robot><unclosed>")
        with pytest.raises(SpecError, match="not a usable URDF"):
            load_robot(path)


class TestAlign:
    """The alignment transform itself."""

    def test_normalizes_axes_and_inserts_tcp(self) -> None:
        aligned = load_robot(ARM_URDF).align(parse_spec(TCP_SPEC))
        assert set(aligned.report.normalized_joints) == {"j1", "j2"}
        assert aligned.report.tcp_joints == ("l2_tcp_joint",)
        assert aligned.tcp_links == ("l2_tcp",)

    def test_emits_exactly_one_link_per_tcp(self) -> None:
        """One new link only: base_link is selected, never created."""
        aligned = load_robot(ARM_URDF).align(parse_spec(TCP_SPEC))
        added = _links(aligned.urdf_bytes) - _links(ARM_URDF)
        assert added == {"l2_tcp"}

    def test_does_not_mutate_the_source(self) -> None:
        robot = load_robot(ARM_URDF)
        robot.align(parse_spec(TCP_SPEC))
        assert robot.urdf_bytes == ARM_URDF

    def test_is_deterministic(self) -> None:
        robot = load_robot(ARM_URDF)
        spec = parse_spec(TCP_SPEC)
        assert robot.align(spec).urdf_bytes == robot.align(spec).urdf_bytes

    def test_is_idempotent(self) -> None:
        """Re-aligning aligned output must be a byte-level no-op."""
        spec = parse_spec(TCP_SPEC)
        once = load_robot(ARM_URDF).align(spec)
        twice = load_robot(once.urdf_bytes).align(spec)
        thrice = load_robot(twice.urdf_bytes).align(spec)
        assert twice.urdf_bytes == once.urdf_bytes
        assert thrice.urdf_bytes == once.urdf_bytes

    def test_aligns_all_joints_regardless_of_chains(self) -> None:
        """`chains` scopes what is drawn and verified, never the export."""
        narrow = load_robot(ARM_URDF).align(
            parse_spec({"chains": [["j1"]], "tcp": [None]})
        )
        assert set(narrow.report.normalized_joints) == {"j1", "j2"}

    def test_tcp_pose_composes_offset_and_rotation(self) -> None:
        aligned = load_robot(ARM_URDF).align(parse_spec(TCP_SPEC))
        pose = aligned.fk()["l2_tcp"]
        # l2 sits at (0.3, 0, 0.5); the TCP is 0.15 along l2's local +Z, which
        # after normalization points along the world +Y.
        np.testing.assert_allclose(pose[:3, 3], [0.3, 0.15, 0.5], atol=1e-12)

    def test_tcp_rotation_is_applied(self) -> None:
        straight = load_robot(ARM_URDF).align(
            parse_spec({"tcp": [{"base_link": "l2", "rpy": [0, 0, 0]}]})
        )
        turned = load_robot(ARM_URDF).align(
            parse_spec({"tcp": [{"base_link": "l2", "rpy": [0, 0, 90]}]})
        )
        relative = np.linalg.inv(straight.fk()["l2_tcp"]) @ turned.fk()["l2_tcp"]
        assert math.isclose(
            math.atan2(relative[1, 0], relative[0, 0]),
            math.pi / 2,
            abs_tol=1e-9,
        )

    def test_reports_ambiguous_urdf_structure(self) -> None:
        graph = b"""<?xml version="1.0"?>
        <robot name="graph">
          <link name="base"/><link name="a"/><link name="shared"/>
          <joint name="j" type="revolute">
            <parent link="base"/><child link="a"/><axis xyz="0 0 1"/>
          </joint>
          <joint name="f1" type="fixed">
            <parent link="base"/><child link="shared"/>
          </joint>
          <joint name="f2" type="fixed">
            <parent link="a"/><child link="shared"/>
          </joint>
        </robot>"""
        aligned = load_robot(graph).align(parse_spec({}))
        assert any(
            "more than one joint" in warning for warning in aligned.report.warnings
        )

    def test_rejects_spec_that_contradicts_urdf(self) -> None:
        with pytest.raises(SpecError, match="not a link in the URDF"):
            load_robot(ARM_URDF).align(parse_spec({"tcp": [{"base_link": "ghost"}]}))

    def test_preserves_xml_comments(self) -> None:
        """A URDF's provenance notes must survive alignment.

        ElementTree drops comments unless asked to keep them, which would
        silently delete calibration dates and "do not edit" warnings on the
        first pass. Comments before the <robot> element sit outside the tree
        and are not recoverable.
        """
        commented = b"""<?xml version="1.0"?>
<robot name="c">
  <!-- calibration: measured 2024-03-11, do not edit by hand -->
  <link name="base"/><link name="l1"/>
  <joint name="j" type="revolute">
    <parent link="base"/><child link="l1"/><axis xyz="0 0 -1"/>
  </joint>
</robot>"""
        aligned = load_robot(commented).align(parse_spec({}))
        assert b"calibration: measured 2024-03-11" in aligned.urdf_bytes

    def test_comment_preservation_keeps_output_stable(self) -> None:
        commented = b"""<?xml version="1.0"?>
<robot name="c">
  <!-- a note -->
  <link name="base"/><link name="l1"/>
  <joint name="j" type="revolute">
    <parent link="base"/><child link="l1"/><axis xyz="0 0 -1"/>
  </joint>
</robot>"""
        spec = parse_spec({})
        once = load_robot(commented).align(spec)
        twice = load_robot(once.urdf_bytes).align(spec)
        assert twice.urdf_bytes == once.urdf_bytes

    def test_a_malformed_origin_is_a_spec_error(self) -> None:
        """Numeric attributes are read mid-alignment, and can be junk.

        The value alone does not say which file it came from, so alignment
        reports it as the error this method documents, naming the URDF.
        """
        malformed = b"""<?xml version="1.0"?>
<robot name="m">
  <link name="base"/><link name="l1"/>
  <joint name="j" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="nan_x 0 0" rpy="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>"""
        with pytest.raises(SpecError, match="cannot align"):
            load_robot(malformed).align(parse_spec({}))


class TestVerify:
    """Alignment must preserve geometry, and verify must prove it."""

    def test_reports_ok_for_a_sound_alignment(self) -> None:
        report = load_robot(ARM_URDF).align(parse_spec(TCP_SPEC)).verify()
        assert report.ok
        assert report.max_link_position_error < 1e-12
        assert report.moved_links == ()

    def test_checks_multiple_configurations(self) -> None:
        # A zero pose alone would hide errors in the joint-motion terms.
        report = load_robot(ARM_URDF).align(parse_spec(TCP_SPEC)).verify()
        assert report.checked_configurations >= 3

    def test_exposes_tcp_poses(self) -> None:
        report = load_robot(ARM_URDF).align(parse_spec(TCP_SPEC)).verify()
        assert set(report.tcp_poses) == {"l2_tcp"}
        assert report.tcp_poses["l2_tcp"].shape == (4, 4)

    def test_rotated_links_nonempty_for_off_axis_robot(self) -> None:
        """A robot with off-axis joints must produce rotated_links.

        ARM_URDF has j1 (axis -Z, equivalent to a 180-deg rotation making it
        +Z) and j2 (axis +Y, needs a 90-deg rotation to become +Z). After
        alignment both parent links change orientation, so rotated_links must
        be non-empty. An empty tuple here would mean alignment silently did
        nothing, which is a regression.
        """
        report = load_robot(ARM_URDF).align(parse_spec(TCP_SPEC)).verify()
        assert report.rotated_links, (
            "expected rotated_links to be non-empty for a robot with "
            "off-axis joints, but got an empty tuple"
        )

    def test_mesh_errors_zero_for_sound_alignment(self) -> None:
        """A sound alignment must not move any visual mesh."""
        # ARM_URDF has no meshes, so the mesh error fields should be zero.
        report = load_robot(ARM_URDF).align(parse_spec(TCP_SPEC)).verify()
        assert report.max_mesh_position_error == 0.0
        assert report.max_mesh_rotation_error == 0.0
        assert report.moved_meshes == ()

    def test_verify_checks_all_shared_links_not_only_chains(self) -> None:
        """verify() scopes to all shared links, never narrowed by chains.

        `chains` selects what the viewer draws, not what must hold. Scoping
        verification to it would leave the links that carry meshes unchecked.
        """
        # The spec declares only j1 in chains -- j2's child l2 is not in it,
        # and must still be checked.
        report = (
            load_robot(ARM_URDF)
            .align(parse_spec({"chains": [["j1"]], "tcp": [None]}))
            .verify()
        )
        assert report.ok

    def test_verify_reports_how_much_it_compared(self) -> None:
        """The counts are what make the verdict checkable."""
        report = load_robot(ARM_URDF).align(parse_spec({})).verify()
        assert report.ok
        assert report.compared_links > 0

    def test_verify_fails_when_it_compares_nothing(self) -> None:
        """An empty check has proven nothing, so it must not pass.

        Two URDFs sharing no link name skip every loop body, leaving the
        verdict vacuously true -- which is the state a mis-wired caller
        produces, and the one where a silent pass is most damaging.
        """
        other = b"""<?xml version="1.0"?>
<robot name="other">
  <link name="totally_different_a"/><link name="totally_different_b"/>
  <joint name="jx" type="revolute">
    <parent link="totally_different_a"/><child link="totally_different_b"/>
    <origin xyz="9 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>"""
        aligned = load_robot(ARM_URDF).align(parse_spec({}))
        detached = dataclasses.replace(aligned, urdf_bytes=other, _chain=[])

        report = detached.verify()
        assert report.compared_links == 0
        assert report.ok is False


class TestWrite:
    """The only side-effecting call."""

    def test_writes_urdf_and_creates_directory(self, tmp_path: Path) -> None:
        source = tmp_path / "arm.urdf"
        source.write_bytes(ARM_URDF)
        aligned = load_robot(source).align(parse_spec(TCP_SPEC))

        report = aligned.write(tmp_path / "out")
        assert report.urdf == tmp_path / "out" / "arm.urdf"
        assert report.urdf.read_bytes() == aligned.urdf_bytes

    def test_honours_explicit_name(self, tmp_path: Path) -> None:
        source = tmp_path / "arm.urdf"
        source.write_bytes(ARM_URDF)
        report = (
            load_robot(source)
            .align(parse_spec(TCP_SPEC))
            .write(tmp_path / "out", name="renamed.urdf")
        )
        assert report.urdf.name == "renamed.urdf"

    def test_refuses_to_overwrite_the_source(self, tmp_path: Path) -> None:
        """Writing into the source's own directory must not destroy it.

        The mesh copier unlinks its target before copying, so source ==
        target would delete the user's robot and then fail. Pointing input
        and output at the same folder is an easy mistake to make.
        """
        source = tmp_path / "arm.urdf"
        source.write_bytes(ARM_URDF)
        aligned = load_robot(source).align(parse_spec(TCP_SPEC))

        with pytest.raises(SpecError, match="refusing to overwrite"):
            aligned.write(tmp_path)

        assert source.read_bytes() == ARM_URDF

    def test_refuses_to_overwrite_the_source_with_the_pruned_urdf(
        self, tmp_path: Path
    ) -> None:
        """The pruned URDF is a target too, and a smaller robot.

        Landing it on the source replaces a whole robot with the subset the
        spec selected, so the guard covers both names.
        """
        source = tmp_path / "arm.urdf"
        source.write_bytes(ARM_URDF)
        aligned = load_robot(source).align(parse_spec(TCP_SPEC))

        with pytest.raises(SpecError, match="refusing to overwrite"):
            aligned.write(tmp_path, name="out.urdf", prune_name="arm.urdf")

        assert source.read_bytes() == ARM_URDF

    def test_allows_same_directory_under_a_different_name(self, tmp_path: Path) -> None:
        source = tmp_path / "arm.urdf"
        source.write_bytes(ARM_URDF)
        report = (
            load_robot(source)
            .align(parse_spec(TCP_SPEC))
            .write(tmp_path, name="arm_aligned.urdf")
        )
        assert report.urdf.exists()
        assert source.read_bytes() == ARM_URDF

    def test_writes_pathless_robot(self, tmp_path: Path) -> None:
        report = (
            load_robot(ARM_URDF).align(parse_spec(TCP_SPEC)).write(tmp_path / "out")
        )
        assert report.urdf.name == "aligned.urdf"
        assert report.urdf.exists()

    def test_written_output_reloads_and_is_stable(self, tmp_path: Path) -> None:
        spec = parse_spec(TCP_SPEC)
        aligned = load_robot(ARM_URDF).align(spec)
        report = aligned.write(tmp_path / "out")
        again = load_robot(report.urdf).align(spec)
        assert again.urdf_bytes == aligned.urdf_bytes


class TestWriteBundle:
    """The export path: two URDFs, and an inventory indexing each."""

    def test_writes_all_four_files(self, tmp_path: Path) -> None:
        bundle = (
            load_robot(ARM_URDF).align(parse_spec(TCP_SPEC)).write_bundle(tmp_path, "t")
        )
        assert bundle.write.urdf.name == "t.urdf"
        assert bundle.inventory.name == "t.inventory.json"
        assert bundle.pruned_inventory is not None
        assert bundle.write.pruned_urdf is not None
        for path in (
            bundle.write.urdf,
            bundle.write.pruned_urdf,
            bundle.inventory,
            bundle.pruned_inventory,
        ):
            assert path.exists()

    def test_inventory_describes_the_urdf_beside_it(self, tmp_path: Path) -> None:
        """The inventory is what downstream code indexes against.

        It is built from the spec the alignment was produced with, so it
        cannot describe a robot other than the URDF written next to it.
        """
        aligned = load_robot(ARM_URDF).align(parse_spec(TCP_SPEC))
        bundle = aligned.write_bundle(tmp_path, "t")

        urdf_text = bundle.write.urdf.read_text()
        inventory_text = bundle.inventory.read_text()
        for link in aligned.tcp_links:
            assert link in urdf_text
            assert link in inventory_text

    def test_pruned_inventory_indexes_the_pruned_joint_vector(
        self, tmp_path: Path
    ) -> None:
        """Pruning renumbers, so the two inventories are not interchangeable.

        Indexing a pruned URDF with the full inventory addresses a different
        joint; the robot still moves, just not as intended. Each URDF carries
        its own inventory precisely so that cannot happen.
        """
        import json

        bundle = (
            load_robot(ARM_URDF).align(parse_spec(TCP_SPEC)).write_bundle(tmp_path, "t")
        )
        assert bundle.pruned_inventory is not None
        full = json.loads(bundle.inventory.read_text())["joint_order"]
        pruned = json.loads(bundle.pruned_inventory.read_text())["joint_order"]

        assert full, "inventory must name the joint vector it indexes"
        assert pruned, "the pruned inventory must name its own joint vector"
        # Every joint the pruned tree keeps exists in the full one; the
        # converse need not hold, which is what makes the indices differ.
        assert set(pruned) <= set(full)
