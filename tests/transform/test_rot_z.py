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

"""Manual rotation about a joint's own ``+Z``.

Normalisation pins Z to the motion axis, which leaves the roll about Z
unconstrained -- today it is whatever the minimum rotation happens to produce,
which differs between robots that are otherwise alike. ``rot_z`` is how a user
picks that roll, and the point of these tests is that picking it changes
*frames only*: every mesh must stay exactly where it was.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from uniurdf.geometry import visual_world_poses
from uniurdf.robot import load_robot
from uniurdf.spec import SpecError, parse_spec

# j1's axis is -Z (needs a rotation), j2's is already +Z (takes the fast
# path). Both are present so these tests cover rot_z on either route.
ARM_URDF = b"""<?xml version="1.0"?>
<robot name="arm">
  <link name="base"/>
  <link name="l1">
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry><mesh filename="part.stl"/></geometry>
    </visual>
  </link>
  <link name="l2">
    <visual>
      <origin xyz="0.02 0 0.05" rpy="0.3 0 0"/>
      <geometry><mesh filename="part.stl"/></geometry>
    </visual>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/><axis xyz="0 0 -1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/>
    <origin xyz="0 0 0.2" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
</robot>"""

CONFIGURATIONS = ({}, {"j1": 0.4, "j2": -0.7}, {"j1": -1.1, "j2": 0.9})

# Rotations round-trip through the URDF as decimal rpy text, so a frame that
# is mathematically unchanged comes back a hair off. This is the same floor
# `VERIFY_MESH_ROT_ATOL_DEG` was set against in robot.py -- far above the
# ~1e-6 deg of observed noise, far below the joint-axis angle a real
# miscompensation would show (typically 90 degrees).
ROT_ATOL_DEG = 1e-4


def _spec(**rot_z: float):
    return parse_spec({"rot_z": rot_z} if rot_z else {})


def _frame_rotation_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


class TestGeometryIsPreserved:
    """The whole premise: R_z about a pinned axis moves nothing physical."""

    @pytest.mark.parametrize("degrees", [17.0, 90.0, 180.0, -45.0, 270.0])
    def test_meshes_and_link_positions_hold_still(self, degrees) -> None:
        robot = load_robot(ARM_URDF)
        plain = robot.align(_spec())
        rotated = robot.align(_spec(j1=degrees, j2=degrees))

        for values in CONFIGURATIONS:
            plain_links = plain.fk(values)
            rotated_links = rotated.fk(values)
            shared = set(plain_links) & set(rotated_links)
            assert shared
            for link in shared:
                position_error = float(
                    np.abs(rotated_links[link][:3, 3] - plain_links[link][:3, 3]).max()
                )
                assert position_error < 1e-12, (
                    f"rot_z={degrees} moved link {link} by {position_error} m"
                )

            plain_meshes = visual_world_poses(plain.urdf_bytes, plain_links)
            rotated_meshes = visual_world_poses(rotated.urdf_bytes, rotated_links)
            assert plain_meshes
            for key, plain_pose in plain_meshes.items():
                rotated_pose = rotated_meshes[key]
                moved = float(np.abs(rotated_pose[:3, 3] - plain_pose[:3, 3]).max())
                turned = _frame_rotation_deg(plain_pose, rotated_pose)
                assert moved < 1e-12, f"mesh {key} moved {moved} m"
                assert turned < ROT_ATOL_DEG, f"mesh {key} turned {turned} deg"

    def test_verify_passes(self) -> None:
        report = load_robot(ARM_URDF).align(_spec(j1=90.0)).verify()
        assert report.ok


class TestFramesDoRotate:
    """Guard against a no-op passing the geometry tests vacuously."""

    @pytest.mark.parametrize("joint,link", [("j1", "l1"), ("j2", "l2")])
    def test_child_frame_turns_by_the_requested_angle(self, joint, link) -> None:
        robot = load_robot(ARM_URDF)
        plain = robot.align(_spec()).fk()
        rotated = robot.align(_spec(**{joint: 61.0})).fk()
        turned = _frame_rotation_deg(plain[link], rotated[link])
        assert turned == pytest.approx(61.0, abs=1e-9)

    def test_applies_to_a_joint_already_on_positive_z(self) -> None:
        """j2's axis is already +Z, so it takes the fast path.

        rot_z must apply on that route too: a joint needing no normalisation
        can still need a chosen roll, and most actuated joints on an aligned
        robot are already +Z.
        """
        robot = load_robot(ARM_URDF)
        plain = robot.align(_spec()).fk()
        rotated = robot.align(_spec(j2=90.0)).fk()
        assert _frame_rotation_deg(plain["l2"], rotated["l2"]) == (
            pytest.approx(90.0, abs=1e-9)
        )


class TestIdempotence:
    """Re-running on the tool's own output is the documented workflow."""

    def test_zero_rot_z_matches_a_spec_without_the_field(self) -> None:
        """An explicit 0 must not perturb anything.

        This is the regression gate for narrowing the +Z fast path: routing
        an already-aligned joint through the identity rewrite reformats its
        origins, which would churn most shipped robots.
        """
        robot = load_robot(ARM_URDF)
        assert (
            robot.align(_spec(j1=0.0, j2=0.0)).urdf_bytes
            == robot.align(_spec()).urdf_bytes
        )

    def test_realigning_with_the_same_angle_is_byte_identical(self) -> None:
        spec = _spec(j1=90.0, j2=30.0)
        once = load_robot(ARM_URDF).align(spec)
        twice = load_robot(once.urdf_bytes).align(spec)
        assert twice.urdf_bytes == once.urdf_bytes

    def test_realigning_a_third_time_still_matches(self) -> None:
        spec = _spec(j1=90.0)
        once = load_robot(ARM_URDF).align(spec)
        twice = load_robot(once.urdf_bytes).align(spec)
        thrice = load_robot(twice.urdf_bytes).align(spec)
        assert thrice.urdf_bytes == once.urdf_bytes

    def test_changing_the_angle_is_geometrically_path_independent(
        self,
    ) -> None:
        """90 then 120 must land where a direct 120 does.

        Geometrically, not textually: the second pass re-writes origins it
        touches, so `0 0 0` becomes `0.0 0.0 0.0`. Asserting bytes here would
        pin cosmetic formatting rather than correctness.
        """
        robot = load_robot(ARM_URDF)
        stepped = load_robot(robot.align(_spec(j1=90.0)).urdf_bytes).align(
            _spec(j1=120.0)
        )
        direct = robot.align(_spec(j1=120.0))
        for values in CONFIGURATIONS:
            stepped_links = stepped.fk(values)
            direct_links = direct.fk(values)
            for link in set(stepped_links) & set(direct_links):
                assert (
                    float(
                        np.abs(
                            stepped_links[link][:3, 3] - direct_links[link][:3, 3]
                        ).max()
                    )
                    < 1e-12
                )
                assert (
                    _frame_rotation_deg(stepped_links[link], direct_links[link])
                    < ROT_ATOL_DEG
                )

    def test_clearing_the_angle_returns_to_the_unrotated_frame(self) -> None:
        """Dropping a joint from rot_z must undo it, not freeze it.

        Without the reverse delta, clearing the field would leave the frame
        rotated with no record of why.
        """
        robot = load_robot(ARM_URDF)
        rotated = robot.align(_spec(j1=90.0))
        cleared = load_robot(rotated.urdf_bytes).align(_spec())
        plain = robot.align(_spec()).fk()
        restored = cleared.fk()
        for link in set(plain) & set(restored):
            assert _frame_rotation_deg(plain[link], restored[link]) < ROT_ATOL_DEG


class TestReporting:
    """What the CLI and GUI show."""

    def test_report_lists_the_rotated_joints_in_degrees(self) -> None:
        report = load_robot(ARM_URDF).align(_spec(j1=90.0, j2=-30.0)).report
        # Degrees round-trip through radians, so -30 comes back as
        # -29.999999999999996. Compare numerically, not exactly.
        assert [name for name, _ in report.rotated_joints] == ["j1", "j2"]
        assert [degrees for _, degrees in report.rotated_joints] == (
            pytest.approx([90.0, -30.0])
        )

    def test_report_is_empty_without_rot_z(self) -> None:
        assert load_robot(ARM_URDF).align(_spec()).report.rotated_joints == ()


class TestSpecValidation:
    """Errors that need the URDF."""

    def test_rejects_an_unknown_joint(self) -> None:
        with pytest.raises(SpecError, match="does not declare as movable"):
            load_robot(ARM_URDF).align(_spec(nope=90.0))

    def test_rejects_a_non_numeric_angle(self) -> None:
        with pytest.raises(SpecError, match="must be a number of degrees"):
            parse_spec({"rot_z": {"j1": "ninety"}})

    def test_rejects_a_boolean_angle(self) -> None:
        # bool is an int subclass, so `true` would otherwise parse as 1 degree.
        with pytest.raises(SpecError, match="must be a number of degrees"):
            parse_spec({"rot_z": {"j1": True}})

    def test_rejects_a_non_mapping(self) -> None:
        with pytest.raises(SpecError, match="rot_z must be a mapping"):
            parse_spec({"rot_z": ["j1", 90.0]})

    def test_parses_degrees_into_radians(self) -> None:
        spec = parse_spec({"rot_z": {"j1": 180.0}})
        assert dict(spec.rot_z)["j1"] == pytest.approx(math.pi)

    def test_key_order_does_not_change_the_spec(self) -> None:
        assert parse_spec({"rot_z": {"b": 1.0, "a": 2.0}}) == parse_spec(
            {"rot_z": {"a": 2.0, "b": 1.0}}
        )

    def test_defaults_to_empty(self) -> None:
        assert parse_spec({}).rot_z == ()

    def test_accepts_a_gripper_joint_outside_every_chain(self) -> None:
        """rot_z is keyed by joint, not by chain membership."""
        spec = dataclasses.replace(
            parse_spec({"chains": [["j1"]], "tcp": [None]}),
            rot_z=(("j2", math.radians(45.0)),),
        )
        rotated = load_robot(ARM_URDF).align(spec).report.rotated_joints
        assert len(rotated) == 1
        assert rotated[0][0] == "j2"
        assert rotated[0][1] == pytest.approx(45.0)
