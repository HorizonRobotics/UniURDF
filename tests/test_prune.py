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

"""Pruning an aligned URDF down to the links the spec names.

Two properties carry the weight here. Kept links must not move -- pruning
deletes elements and rewrites none, so anything but exact agreement is a bug,
not a tolerance. And alignment's own emitted links must survive: they hang
*below* their base link, so a plain ancestor closure would have the tool delete
its own output.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from conftest import EXAMPLE_SPEC, iter_specs
from uniurdf.kinematics.chain import build_chain
from uniurdf.prune import (
    PruneError,
    focus_links,
    keep_set,
    prune_urdf,
)
from uniurdf.robot import load_robot
from uniurdf.spec import load_spec, parse_spec

# One driven arm, one arm nobody asked about, and a chassis hanging off the
# root -- the shape that makes pruning worth doing. `cam` sits below the
# wrist, so it is the link that only survives by being named.
FLEET_URDF = b"""<?xml version="1.0"?>
<robot name="fleet">
  <material name="shared"><color rgba="1 0 0 1"/></material>
  <link name="base"/>
  <link name="chassis"/><link name="wheel"/>
  <link name="l_a"/><link name="l_b"/><link name="l_grip"/>
  <link name="cam"/>
  <link name="r_a"/><link name="r_b"/>
  <joint name="chassis_joint" type="fixed">
    <parent link="base"/><child link="chassis"/>
    <origin xyz="0 0 -0.1" rpy="0 0 0"/>
  </joint>
  <joint name="wheel_joint" type="continuous">
    <parent link="chassis"/><child link="wheel"/>
    <origin xyz="0.2 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="l_j1" type="revolute">
    <parent link="base"/><child link="l_a"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="l_j2" type="revolute">
    <parent link="l_a"/><child link="l_b"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="l_grip_joint" type="prismatic">
    <parent link="l_b"/><child link="l_grip"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="0.04" effort="1" velocity="1"/>
  </joint>
  <joint name="cam_joint" type="fixed">
    <parent link="l_b"/><child link="cam"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="r_j1" type="revolute">
    <parent link="base"/><child link="r_a"/>
    <origin xyz="0 0.3 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="r_j2" type="revolute">
    <parent link="r_a"/><child link="r_b"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
</robot>"""

SPEC = {
    "chains": [["l_j1", "l_j2"]],
    "tcp": [
        {
            "base_link": "l_b",
            "xyz": [0, 0, 0.1],
            "joints": ["l_grip_joint"],
        }
    ],
}


def _aligned(spec_dict=None):
    """Align FLEET_URDF and return (aligned, spec)."""
    spec = parse_spec(spec_dict if spec_dict is not None else SPEC)
    return load_robot(FLEET_URDF).align(spec), spec


def _pruned_chain(spec_dict=None):
    """Prune an aligned FLEET_URDF and return its chain."""
    aligned, _ = _aligned(spec_dict)
    return build_chain(aligned.prune().urdf_bytes)


class TestWhatSurvives:
    """The keep-set: named links, their ancestors, and nothing else."""

    def test_keeps_the_named_chain_and_its_ancestors(self) -> None:
        chain = _pruned_chain()
        for link in ("base", "l_a", "l_b"):
            assert link in chain.link_names

    def test_drops_the_arm_the_spec_never_named(self) -> None:
        chain = _pruned_chain()
        assert "r_a" not in chain.link_names
        assert "r_b" not in chain.link_names

    def test_drops_the_chassis_hanging_off_the_root(self) -> None:
        """A sibling subtree of the root goes, but the root itself stays."""
        chain = _pruned_chain()
        assert "chassis" not in chain.link_names
        assert "wheel" not in chain.link_names
        assert chain.root == "base"

    def test_keeps_gripper_links_named_by_tcp_joints(self) -> None:
        assert "l_grip" in _pruned_chain().link_names

    def test_drops_a_wrist_frame_the_spec_did_not_name(self) -> None:
        """The documented cost of the rule.

        `cam` hangs below `l_b`, so nothing but naming it keeps it. This is
        the behaviour the README warns about -- pinned so it stays a decision
        rather than becoming a surprise.
        """
        assert "cam" not in _pruned_chain().link_names

    def test_naming_a_wrist_frame_keeps_it(self) -> None:
        """...and the documented remedy actually works."""
        chain = _pruned_chain(dict(SPEC, camera_references=["cam"]))
        assert "cam" in chain.link_names

    def test_a_rot_z_joint_keeps_its_link(self) -> None:
        """A joint whose roll the user chose is one they are thinking about.

        `r_j2` belongs to no chain here, so only rot_z keeps its link.
        """
        chain = _pruned_chain(dict(SPEC, rot_z={"r_j2": 90.0}))
        assert "r_b" in chain.link_names
        assert "r_a" in chain.link_names  # its ancestor comes along


class TestAlignmentOutputSurvives:
    """The trap: emitted links hang *below* the links that seeded them."""

    def test_tcp_link_is_not_deleted(self) -> None:
        """A plain ancestor closure would drop the tool's own output.

        `l_b_tcp` is a child of `l_b`, so nothing in the spec's link set
        reaches it. It survives only because alignment's emitted links seed
        the keep-set alongside the ones the spec names.
        """
        aligned, _ = _aligned()
        chain = build_chain(aligned.prune().urdf_bytes)
        assert aligned.tcp_links == ("l_b_tcp",)
        assert "l_b_tcp" in chain.link_names

    def test_camera_compat_link_is_not_deleted(self) -> None:
        aligned, _ = _aligned(dict(SPEC, camera_references=["l_a"]))
        compat = aligned.report.camera_compat_links
        assert compat, "expected a compat link for a frame that moved"
        chain = build_chain(aligned.prune().urdf_bytes)
        for link in compat:
            assert link in chain.link_names

    def test_provenance_record_rides_along(self) -> None:
        """Or re-aligning the pruned file would double-apply rot_z."""
        aligned, _ = _aligned(dict(SPEC, rot_z={"l_j2": 90.0}))
        body = aligned.prune().urdf_bytes
        assert b"<uniurdf>" in body
        assert b'joint="l_j2"' in body


class TestGeometryIsUntouched:
    """Kept links must not move. Not approximately -- exactly."""

    def test_kept_link_poses_are_identical(self) -> None:
        aligned, _ = _aligned()
        report = aligned.prune()
        assert report.max_position_error == 0.0

    def test_poses_agree_under_a_non_zero_configuration(self) -> None:
        """A zero pose hides errors in the joint-motion terms."""
        aligned, _ = _aligned()
        pruned = build_chain(aligned.prune().urdf_bytes)
        values = {"l_j1": 0.4, "l_j2": -0.7, "l_grip_joint": 0.02}
        full_poses = aligned.chain.forward_kinematics(values)
        pruned_poses = pruned.forward_kinematics(values)
        for link, pose in pruned_poses.items():
            assert np.abs(pose - full_poses[link]).max() == 0.0

    def test_tcp_world_pose_is_preserved(self) -> None:
        """The pose a policy actually commands."""
        aligned, _ = _aligned()
        pruned = build_chain(aligned.prune().urdf_bytes)
        values = {"l_j1": 0.3, "l_j2": 0.9}
        assert (
            np.abs(
                pruned.forward_kinematics(values)["l_b_tcp"]
                - aligned.chain.forward_kinematics(values)["l_b_tcp"]
            ).max()
            == 0.0
        )

    def test_a_detached_subtree_is_rejected(self) -> None:
        """The self-check has to actually fire.

        Hand a keep-set a URDF whose kept subtree no longer reaches the root,
        and the surviving links land somewhere else -- which must raise rather
        than write a wrong file. Built by deleting the joint that anchors the
        arm, so `l_a` and everything under it detaches.
        """
        aligned, spec = _aligned()
        broken = ET.fromstring(aligned.urdf_bytes)
        for element in list(broken):
            if element.tag == "joint" and element.get("name") == "l_j1":
                element.find("origin").set("xyz", "0 0 99")
        # The joint still exists, so the keep-set is unchanged -- but the
        # source chain passed in disagrees with the bytes, which is exactly
        # the class of mistake the check exists to catch.
        with pytest.raises(PruneError, match="must not happen"):
            prune_urdf(ET.tostring(broken), spec, chain=aligned.chain)


class TestJointVectorRenumbering:
    """The one silent failure mode, so it is reported loudly."""

    def test_dropped_movable_joints_are_reported(self) -> None:
        aligned, _ = _aligned()
        report = aligned.prune()
        assert set(report.dropped_movable_joints) == {
            "wheel_joint",
            "r_j1",
            "r_j2",
        }
        assert report.renumbers_joint_vector

    def test_indices_actually_shift(self) -> None:
        """Proof the warning is not theoretical.

        `l_j1` sits at index 1 in the full vector (behind `wheel_joint`) and
        at 0 in the pruned one. A caller reusing the full index would drive
        the wrong joint, and nothing would appear to fail.
        """
        aligned, _ = _aligned()
        pruned = build_chain(aligned.prune().urdf_bytes)
        assert list(aligned.chain.joint_names).index("l_j1") == 1
        assert list(pruned.joint_names).index("l_j1") == 0

    def test_a_robot_with_nothing_to_drop_does_not_renumber(self) -> None:
        """Six shipped specs are already minimal; that is not a defect."""
        spec = parse_spec(
            {"chains": [["l_j1", "l_j2", "l_grip_joint", "r_j1", "r_j2"]]}
        )
        urdf = FLEET_URDF.replace(
            b'<joint name="wheel_joint" type="continuous">',
            b'<joint name="wheel_joint" type="fixed">',
        )
        report = prune_urdf(load_robot(urdf).align(spec).urdf_bytes, spec)
        assert report.dropped_movable_joints == ()
        assert not report.renumbers_joint_vector


class TestDocumentIntegrity:
    """The pruned file has to be a valid URDF, not just a smaller one."""

    def test_shared_material_definitions_survive(self) -> None:
        """Links reference top-level materials by name.

        Dropping one because no *link* element mentions it directly would
        break the links that did survive.
        """
        aligned, _ = _aligned()
        assert b'<material name="shared">' in aligned.prune().urdf_bytes

    def test_a_joint_whose_parent_was_dropped_is_removed(self) -> None:
        """A link can be the child of more than one joint.

        `shared` is reached from `arm` (kept) and from `junk` (pruned away).
        Keeping the link on account of its surviving parent must still drop
        `from_junk`, or that joint names a link no longer declared. Parsers
        respond to a dangling reference by silently discarding the joint,
        which is how a robot quietly loses an arm.
        """
        urdf = b"""<?xml version="1.0"?>
<robot name="dup">
  <link name="base"/><link name="junk"/>
  <link name="arm"/><link name="shared"/>
  <joint name="junk_joint" type="revolute">
    <parent link="base"/><child link="junk"/>
    <origin xyz="5 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="arm_joint" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="from_junk" type="fixed">
    <parent link="junk"/><child link="shared"/>
    <origin xyz="0 0 9.9" rpy="0 0 0"/></joint>
  <joint name="from_arm" type="fixed">
    <parent link="arm"/><child link="shared"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/></joint>
</robot>"""
        spec = parse_spec({"chains": [["arm_joint"]], "camera_references": ["shared"]})
        aligned = load_robot(urdf).align(spec)
        assert "shared" in aligned.chain.duplicate_child_links
        body = prune_urdf(aligned.urdf_bytes, spec, chain=aligned.chain).urdf_bytes

        root = ET.fromstring(body)
        declared = {el.get("name") for el in root.findall("link")}
        assert "shared" in declared  # kept via its surviving parent
        assert "junk" not in declared  # the other parent went
        names = {joint.get("name") for joint in root.findall("joint")}
        assert "from_junk" not in names
        for joint in root.findall("joint"):
            for end in ("parent", "child"):
                assert joint.find(end).get("link") in declared

    def test_no_joint_references_a_dropped_link(self) -> None:
        """A dangling reference makes parsers silently drop a whole arm."""
        aligned, _ = _aligned()
        root = ET.fromstring(aligned.prune().urdf_bytes)
        declared = {el.get("name") for el in root.findall("link")}
        for joint in root.findall("joint"):
            for end in ("parent", "child"):
                element = joint.find(end)
                assert element is not None
                assert element.get("link") in declared

    def test_a_loop_joint_naming_a_dropped_link_is_removed(self) -> None:
        """A loop joint closes a kinematic linkage, commonly in a gripper.

        Left behind, it names links that no longer exist.
        """
        urdf = FLEET_URDF.replace(
            b"</robot>",
            b'<loop_joint name="closure" type="spherical">'
            b'<link1 link="r_a"/><link2 link="r_b"/></loop_joint></robot>',
        )
        spec = parse_spec(SPEC)
        body = prune_urdf(load_robot(urdf).align(spec).urdf_bytes, spec)
        assert b"loop_joint" not in body.urdf_bytes

    def test_a_loop_joint_between_kept_links_is_left_alone(self) -> None:
        urdf = FLEET_URDF.replace(
            b"</robot>",
            b'<loop_joint name="closure" type="spherical">'
            b'<link1 link="l_a"/><link2 link="l_b"/></loop_joint></robot>',
        )
        spec = parse_spec(SPEC)
        body = prune_urdf(load_robot(urdf).align(spec).urdf_bytes, spec)
        assert b'<loop_joint name="closure"' in body.urdf_bytes


class TestSeedHelpers:
    """focus_links and keep_set, which callers may use directly."""

    def test_focus_links_excludes_ancestors(self) -> None:
        aligned, spec = _aligned()
        seeds = focus_links(aligned.chain, spec)
        assert "l_b" in seeds
        assert "base" not in seeds  # an ancestor, added by keep_set

    def test_keep_set_adds_the_root_even_with_no_seeds(self) -> None:
        """An empty spec must still produce a loadable URDF."""
        chain = build_chain(FLEET_URDF)
        assert keep_set(chain, parse_spec({"chains": [[]]})) == {"base"}

    def test_unknown_extra_links_are_ignored(self) -> None:
        """A caller may pass emitted links a given run did not produce."""
        aligned, spec = _aligned()
        seeds = focus_links(aligned.chain, spec, extra_links=["nope"])
        assert "nope" not in seeds


def test_prunes_every_shipped_spec_without_moving_anything() -> None:
    """Sweep every shipped spec: exact geometry, emitted links intact.

    The reach depends on which specs the branch carries: the packaged example
    alone, or a fleet. The shapes the synthetic fixtures cannot build -- a
    mimic joint whose source lands outside the keep-set, a chassis-rooted
    tree, a 20-link gripper subtree -- arrive with the real robots.
    """
    checked = 0
    problems: list[str] = []
    unreachable: list[str] = []
    for path in iter_specs():
        stem = path.stem
        spec = load_spec(path)
        if spec.source_urdf is None or not spec.source_urdf.exists():
            # A shipped spec names a robot this sweep is supposed to check.
            # Skipping would quietly shrink the corpus to whatever happens to
            # be mounted.
            unreachable.append(f"{stem}: {spec.source_urdf}")
            continue
        aligned = load_robot(spec.source_urdf).align(spec)
        report = aligned.prune()
        checked += 1

        if report.max_position_error != 0.0:
            problems.append(f"{stem}: moved {report.max_position_error:.2e} m")

        pruned = build_chain(report.urdf_bytes)
        names = set(pruned.link_names)
        for emitted in (
            *aligned.tcp_links,
            *aligned.report.camera_compat_links,
        ):
            if emitted not in names:
                problems.append(f"{stem}: lost emitted link {emitted}")

        # A mimic follower whose source was pruned away would silently read
        # zero, which FK reports as a perfectly valid -- and wrong -- pose.
        kept = {joint.name for joint in pruned.joints}
        for joint in pruned.joints:
            if joint.mimic and joint.mimic[0] not in kept:
                problems.append(f"{stem}: {joint.name} mimics missing {joint.mimic[0]}")

    assert not unreachable, "specs name URDFs that are not present:\n" + "\n".join(
        unreachable
    )
    assert checked, "no specs were checked"
    assert not problems, "\n".join(problems[:10])


def test_pruned_urdf_matches_the_oracle() -> None:
    """`pytorch_kinematics` is what consumers load with, so it decides.

    The in-house chain agreeing with itself proves less than the library the
    downstream policy actually uses agreeing with the full URDF.
    """
    pk = pytest.importorskip("pytorch_kinematics")
    torch = pytest.importorskip("torch")

    spec = load_spec(EXAMPLE_SPEC)
    assert spec.source_urdf is not None

    aligned = load_robot(spec.source_urdf).align(spec)
    full = pk.build_chain_from_urdf(aligned.urdf_bytes)
    pruned = pk.build_chain_from_urdf(aligned.prune().urdf_bytes)

    full_names = full.get_joint_parameter_names()
    pruned_names = pruned.get_joint_parameter_names()
    assert len(pruned_names) < len(full_names)

    # Address joints by name: pruning renumbers, so comparing two positional
    # vectors would silently compare different configurations.
    values = {name: 0.13 * (i % 5 - 2) for i, name in enumerate(full_names)}
    full_poses = full.forward_kinematics(
        torch.tensor([[values[n] for n in full_names]], dtype=torch.float32)
    )
    pruned_poses = pruned.forward_kinematics(
        torch.tensor([[values[n] for n in pruned_names]], dtype=torch.float32)
    )
    for link in pruned_poses:
        assert (
            np.abs(
                pruned_poses[link].get_matrix().numpy()
                - full_poses[link].get_matrix().numpy()
            ).max()
            == 0.0
        ), f"{link} moved"
