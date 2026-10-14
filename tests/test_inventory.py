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

"""The link/index inventory an aligned robot exports.

The indices are the load-bearing part: a config that indexes an arm one joint
too late trains on the wrong data and nothing complains. These tests pin them
against the same source of truth the consuming code uses.
"""

from __future__ import annotations

import json

import pytest

from conftest import iter_specs
from uniurdf.inventory import build_inventory
from uniurdf.kinematics.chain import build_chain
from uniurdf.robot import load_robot
from uniurdf.spec import parse_spec

# Two arms with a gripper joint between them, so the second arm's indices are
# NOT contiguous with the first's -- the exact shape that makes hand-derived
# indices go wrong.
DUAL_URDF = b"""<?xml version="1.0"?>
<robot name="dual">
  <link name="base"/>
  <link name="l_a"/><link name="l_b"/><link name="l_grip"/>
  <link name="r_a"/><link name="r_b"/>
  <link name="cam"/>
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
  <joint name="cam_joint" type="fixed">
    <parent link="l_a"/><child link="cam"/>
    <origin xyz="0.05 0 0" rpy="0 0 0"/>
  </joint>
</robot>"""

SPEC = {
    "chains": [["l_j1", "l_j2"], ["r_j1", "r_j2"]],
    "tcp": [
        {"base_link": "l_b", "xyz": [0, 0, 0.1], "joints": ["l_grip_joint"]},
        None,
    ],
}


def _inventory(spec_dict=None):
    spec = parse_spec(spec_dict if spec_dict is not None else SPEC)
    aligned = load_robot(DUAL_URDF).align(spec)
    return build_inventory(aligned.chain, spec, aligned.report.camera_compat_links)


class TestJointIndices:
    """The reason this module exists."""

    def test_indices_match_the_movable_joint_vector(self) -> None:
        inventory = _inventory()
        order = list(build_chain(DUAL_URDF).joint_names)
        for group in inventory["chains"]:
            for joint in group["joints"]:
                assert order[joint["index"]] == joint["name"]

    def test_indices_are_not_assumed_contiguous(self) -> None:
        """The second arm starts at 3, not 2.

        The gripper joint sits between the arms. A reader who assumed two
        consecutive blocks would be wrong here, which is precisely the
        mistake found in a shipped config.
        """
        inventory = _inventory()
        first = [j["index"] for j in inventory["chains"][0]["joints"]]
        second = [j["index"] for j in inventory["chains"][1]["joints"]]
        assert first == [0, 1]
        assert second == [3, 4]

    def test_alignment_does_not_shift_the_indices(self) -> None:
        """Alignment only adds *fixed* joints, which the vector excludes.

        This is what lets an index taken from the aligned robot address the
        same joint in the source URDF.
        """
        spec = parse_spec(SPEC)
        aligned = load_robot(DUAL_URDF).align(spec)
        assert list(aligned.chain.joint_names) == list(
            build_chain(DUAL_URDF).joint_names
        )

    def test_joint_order_is_reported(self) -> None:
        """The full vector, so a reader can check an index without the URDF."""
        inventory = _inventory()
        assert inventory["joint_order"] == list(build_chain(DUAL_URDF).joint_names)


class TestLinks:
    """Names, and which of them alignment created."""

    def test_chain_links_are_the_joints_children(self) -> None:
        inventory = _inventory()
        assert inventory["chains"][0]["links"] == ["l_a", "l_b"]
        assert inventory["chains"][1]["links"] == ["r_a", "r_b"]

    def test_tcp_reports_both_names(self) -> None:
        inventory = _inventory()
        assert inventory["tcp"][0] == {"link": "l_b_tcp", "base_link": "l_b"}

    def test_a_chain_without_a_tcp_keeps_its_slot(self) -> None:
        """The tcp list is paired positionally with chains.

        Dropping the null would renumber every arm after it -- silently, and
        only for robots where one arm has no tool.
        """
        inventory = _inventory()
        assert inventory["tcp"][1] is None
        assert len(inventory["tcp"]) == len(inventory["chains"])

    def test_finger_links_are_the_tcp_joints_children(self) -> None:
        inventory = _inventory()
        assert inventory["finger_links"][0]["links"] == ["l_grip"]

    def test_finger_links_are_empty_without_tcp_joints(self) -> None:
        """7 of the 28 shipped specs declare no tcp.joints.

        An empty list there is correct, not a defect to work around.
        """
        spec = {
            "chains": [["l_j1", "l_j2"]],
            "tcp": [{"base_link": "l_b", "xyz": [0, 0, 0.1]}],
        }
        assert _inventory(spec)["finger_links"][0]["links"] == []


class TestCameraReferences:
    """Which name a calibrated extrinsic should target."""

    def test_compat_link_reported_when_the_frame_moved(self) -> None:
        """A moved frame gets an alias.

        `l_a` hangs off a joint with a +Y axis, so normalisation reorients its
        frame and a compat link is emitted to recover the original.
        """
        spec = dict(SPEC, camera_references=["l_a"])
        inventory = _inventory(spec)
        entry = inventory["camera_references"][0]
        assert entry["link"] == "l_a"
        assert entry["compat_link"] == "l_a_camera_mount_compat"

    def test_compat_link_is_null_when_the_frame_held_still(self) -> None:
        """A frame that held still needs no alias.

        The original name is still the right target. `cam` is a fixed child
        of the very link that rotates, and the S-inverse compensation puts it
        back -- so a naive "its parent moved, so it moved"
        rule would emit an alias that is not needed.
        """
        entry = _inventory(dict(SPEC, camera_references=["cam"]))["camera_references"][
            0
        ]
        assert entry["link"] == "cam"
        assert entry["compat_link"] is None


class TestSerialisation:
    """It has to survive being written to a file."""

    def test_round_trips_through_json(self) -> None:
        inventory = _inventory()
        assert json.loads(json.dumps(inventory)) == inventory

    def test_uses_neutral_field_names(self) -> None:
        """The tool must not bake in any consumer's key names.

        Mapping these onto a particular training config is a human decision;
        emitting `arm_joint_id` here would assert a coupling that does not
        exist.
        """
        inventory = _inventory()
        text = json.dumps(inventory)
        for leaked in (
            "arm_joint_id",
            "arm_link_keys",
            "finger_keys",
            "cam_ref_links",
        ):
            assert leaked not in text


class TestMimicJoints:
    """A follower still occupies a slot in the joint vector."""

    def test_mimic_joint_gets_an_index(self) -> None:
        """`pytorch_kinematics.get_joints()` includes mimic followers.

        Indexing against `independent_joint_names` instead would drop them
        and silently renumber every joint after the first follower -- on the
        Franka Panda that is a 13-entry vector reported as 8.
        """
        urdf = DUAL_URDF.replace(
            b'<joint name="r_j2" type="revolute">',
            b'<joint name="r_j2" type="revolute">'
            b'<mimic joint="r_j1" multiplier="1.0"/>',
        )
        spec = parse_spec({"chains": [["r_j1", "r_j2"]], "tcp": [None]})
        aligned = load_robot(urdf).align(spec)
        inventory = build_inventory(aligned.chain, spec, ())
        order = list(build_chain(urdf).joint_names)
        for joint in inventory["chains"][0]["joints"]:
            assert joint["index"] is not None
            assert order[joint["index"]] == joint["name"]
        assert "r_j2" in aligned.chain.mimic_joint_names


def test_default_chain_is_inventoried_when_the_spec_declares_none() -> None:
    """resolved_chains falls back to every independent joint."""
    inventory = _inventory({})
    assert len(inventory["chains"]) == 1
    names = [j["name"] for j in inventory["chains"][0]["joints"]]
    assert names == list(build_chain(DUAL_URDF).joint_names)


def test_indices_agree_with_pytorch_kinematics_across_the_corpus() -> None:
    """The oracle test: every index must address the same joint pk would.

    `pytorch_kinematics` is what the consuming configs index against, and it
    includes mimic followers. An inventory built from
    `independent_joint_names` satisfies every synthetic test here and still
    misindexes any robot carrying a mimic, so the real library is the only
    thing that settles it. Sweeps every shipped spec.
    """
    pk = pytest.importorskip("pytorch_kinematics")

    from uniurdf.spec import load_spec

    checked = 0
    mismatches: list[str] = []
    unreachable: list[str] = []
    for path in iter_specs():
        stem = path.stem
        spec = load_spec(path)
        if spec.source_urdf is None or not spec.source_urdf.exists():
            unreachable.append(f"{stem}: {spec.source_urdf}")
            continue
        aligned = load_robot(spec.source_urdf).align(spec)
        inventory = build_inventory(
            aligned.chain, spec, aligned.report.camera_compat_links
        )
        oracle = [
            joint.name
            for joint in pk.build_chain_from_urdf(aligned.urdf_bytes).get_joints()
        ]
        if oracle != inventory["joint_order"]:
            mismatches.append(f"{stem}: joint_order disagrees with pk")
            continue
        for group in inventory["chains"]:
            for joint in group["joints"]:
                checked += 1
                index = joint["index"]
                if index is None or oracle[index] != joint["name"]:
                    mismatches.append(
                        f"{stem}: index {index} is {oracle[index]!r}, "
                        f"expected {joint['name']!r}"
                    )
    assert not unreachable, "specs name URDFs that are not present:\n" + "\n".join(
        unreachable
    )
    assert checked, "no joints were checked"
    assert not mismatches, "\n".join(mismatches[:10])
