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

"""Tests for actuated-axis normalization.

The stage rewrites every actuated joint's axis to ``+Z`` and compensates the
surrounding frames so the robot's physical motion is unchanged. That invariant
-- same control values, same physical pose -- is what the FK tests here assert,
since it is the whole reason the rewrite is safe to apply to trained robots.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from uniurdf.kinematics.chain import build_chain, sweep_configuration
from uniurdf.transform.axis import normalize_actuated_axes

# A two-joint arm whose axes point along -Z and +Y, so normalization has to
# compensate both an antipodal and an orthogonal axis.
OFF_AXIS_URDF = """<?xml version="1.0"?>
<robot name="off_axis">
  <link name="base"/>
  <link name="upper">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry><box size="0.1 0.1 0.2"/></geometry>
    </visual>
  </link>
  <link name="tool"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="upper"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 0 -1"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="upper"/><child link="tool"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _axis_of(root: ET.Element, joint_name: str) -> list[float]:
    for joint in root.findall("joint"):
        if joint.get("name") == joint_name:
            axis = joint.find("axis")
            assert axis is not None
            return [float(v) for v in axis.get("xyz", "").split()]
    raise AssertionError(f"joint {joint_name} not found")


class TestNormalizeActuatedAxes:
    """Axis rewriting and the motion-preservation invariant."""

    def test_rewrites_axes_to_positive_z(self) -> None:
        root = ET.fromstring(OFF_AXIS_URDF)
        rewritten = normalize_actuated_axes(root)
        assert set(rewritten) == {"j1", "j2"}
        np.testing.assert_allclose(_axis_of(root, "j1"), [0.0, 0.0, 1.0])
        np.testing.assert_allclose(_axis_of(root, "j2"), [0.0, 0.0, 1.0])

    def test_preserves_link_positions_under_motion(self) -> None:
        """The point of the whole stage: same joint values, same geometry.

        Orientation may change -- that is the coordinate convention being
        aligned -- but a link's position in the base frame must not, or
        previously recorded actions would no longer reach the same place.
        """
        original = build_chain(OFF_AXIS_URDF.encode())
        root = ET.fromstring(OFF_AXIS_URDF)
        normalize_actuated_axes(root)
        aligned = build_chain(ET.tostring(root))

        for config in (
            {},
            sweep_configuration(original.independent_joint_names),
            {"j1": 0.7, "j2": -0.4},
        ):
            before = original.forward_kinematics(config)
            after = aligned.forward_kinematics(config)
            for link in before:
                np.testing.assert_allclose(
                    after[link][:3, 3],
                    before[link][:3, 3],
                    atol=1e-12,
                    err_msg=f"{link} moved for config {config}",
                )

    def test_is_idempotent(self) -> None:
        root = ET.fromstring(OFF_AXIS_URDF)
        normalize_actuated_axes(root)
        once = ET.tostring(root)
        assert normalize_actuated_axes(root) == []
        assert ET.tostring(root) == once

    def test_leaves_fixed_joints_alone(self) -> None:
        urdf = b"""<?xml version="1.0"?>
        <robot name="fixed_only">
          <link name="a"/><link name="b"/>
          <joint name="j" type="fixed">
            <parent link="a"/><child link="b"/>
            <origin xyz="1 0 0" rpy="0 0 0"/>
          </joint>
        </robot>"""
        root = ET.fromstring(urdf)
        assert normalize_actuated_axes(root) == []

    def test_skips_zero_length_axis(self) -> None:
        """``axis xyz="0 0 0"`` appears on placeholder joints in the wild."""
        urdf = b"""<?xml version="1.0"?>
        <robot name="degenerate">
          <link name="a"/><link name="b"/>
          <joint name="j" type="revolute">
            <parent link="a"/><child link="b"/>
            <axis xyz="0 0 0"/>
          </joint>
        </robot>"""
        root = ET.fromstring(urdf)
        assert normalize_actuated_axes(root) == []

    def test_canonicalizes_near_positive_z_without_compensating(self) -> None:
        """A near-+Z axis is snapped, but frames must not be rotated."""
        urdf = b"""<?xml version="1.0"?>
        <robot name="near_z">
          <link name="a"/><link name="b"/>
          <joint name="j" type="revolute">
            <parent link="a"/><child link="b"/>
            <origin xyz="0 0 1" rpy="0 0 0"/>
            <axis xyz="0 0 0.9999999"/>
          </joint>
        </robot>"""
        root = ET.fromstring(urdf)
        normalize_actuated_axes(root)
        np.testing.assert_allclose(_axis_of(root, "j"), [0.0, 0.0, 1.0])


# A link this tool emits, marking a URDF as already aligned.
_ALIGNED_SUFFIXES = ("_tcp", "_camera_mount_compat")


def _is_aligned(root: ET.Element) -> bool:
    return any(
        (link.get("name") or "").endswith(_ALIGNED_SUFFIXES)
        for link in root.findall("link")
    )


class TestRealCorpusIdempotence:
    """The aligned robot assets must already satisfy the stage's invariant.

    Re-normalizing an aligned URDF must be a no-op, so any behavioural drift
    in this stage shows up as a non-empty rewrite.
    """

    @pytest.fixture(scope="class")
    def aligned_urdfs(self, real_urdfs: list[Path]) -> list[Path]:
        """The corpus entries this tool has already aligned.

        A tree holding only unaligned robots is a legitimate corpus that
        cannot answer this question, so the test below skips. Selecting the
        aligned files here is also what keeps that test from iterating an
        empty list and reporting a pass.
        """
        aligned = [path for path in real_urdfs if _is_aligned(ET.parse(path).getroot())]
        if not aligned:
            pytest.skip(f"corpus holds no aligned URDFs ({len(real_urdfs)} seen)")
        return aligned

    def test_aligned_urdfs_are_unchanged(self, aligned_urdfs: list[Path]) -> None:
        drifted = []
        for path in aligned_urdfs:
            rewritten = normalize_actuated_axes(ET.parse(path).getroot())
            if rewritten:
                drifted.append((path.name, len(rewritten)))
        assert not drifted, f"re-normalizing aligned URDFs rewrote axes: {drifted}"


@pytest.mark.parametrize(
    "axis", [[0, 0, -1], [0, 1, 0], [1, 0, 0], [0.577, 0.577, 0.577]]
)
def test_arbitrary_axes_preserve_positions(axis: list[float]) -> None:
    """Motion preservation holds for every axis direction, not just -Z."""
    urdf = f"""<?xml version="1.0"?>
    <robot name="generic">
      <link name="base"/><link name="arm"/><link name="tip"/>
      <joint name="j1" type="revolute">
        <parent link="base"/><child link="arm"/>
        <origin xyz="0 0 0.4" rpy="0.1 0.2 0.3"/>
        <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>
      </joint>
      <joint name="j2" type="fixed">
        <parent link="arm"/><child link="tip"/>
        <origin xyz="0.25 0.1 0" rpy="0 0 0"/>
      </joint>
    </robot>""".encode()

    original = build_chain(urdf)
    root = ET.fromstring(urdf)
    normalize_actuated_axes(root)
    aligned = build_chain(ET.tostring(root))

    for value in (0.0, 0.6, -1.1):
        before = original.forward_kinematics({"j1": value})
        after = aligned.forward_kinematics({"j1": value})
        for link in before:
            np.testing.assert_allclose(
                after[link][:3, 3], before[link][:3, 3], atol=1e-12
            )
