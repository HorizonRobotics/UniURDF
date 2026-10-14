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

"""Unit tests for uniurdf.geometry.

All tests use synthetic URDF strings -- no filesystem access, no network,
no bucket mount.
"""

from __future__ import annotations

import numpy as np

from uniurdf.geometry import parse_float_triple, visual_world_poses
from uniurdf.robot import load_robot

# ---------------------------------------------------------------------------
# Minimal URDFs used across tests
# ---------------------------------------------------------------------------

_SINGLE_LINK_NO_MESH = b"""<?xml version="1.0"?>
<robot name="empty">
  <link name="base"/>
</robot>"""

_SINGLE_LINK_ONE_MESH = b"""<?xml version="1.0"?>
<robot name="one_mesh">
  <link name="base">
    <visual>
      <origin xyz="1 2 3" rpy="0 0 0"/>
      <geometry><mesh filename="box.stl"/></geometry>
    </visual>
  </link>
</robot>"""

_SINGLE_LINK_ROTATED_ORIGIN = b"""<?xml version="1.0"?>
<robot name="rotated">
  <link name="base">
    <visual>
      <origin xyz="0 0 0" rpy="1.5708 0 0"/>
      <geometry><mesh filename="part.stl"/></geometry>
    </visual>
  </link>
</robot>"""

# A link with a <visual> that has no <geometry><mesh> (box primitive).
_PRIMITIVE_ONLY = b"""<?xml version="1.0"?>
<robot name="prim">
  <link name="base">
    <visual>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </visual>
  </link>
</robot>"""

# Two links: one with two meshes, one with none.
_TWO_VISUAL_LINK = b"""<?xml version="1.0"?>
<robot name="multi">
  <link name="base"/>
  <link name="arm">
    <visual>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="arm_vis.stl"/></geometry>
    </visual>
    <visual>
      <origin xyz="0.6 0 0" rpy="0 1.5708 0"/>
      <geometry><mesh filename="arm_dec.stl"/></geometry>
    </visual>
  </link>
  <joint name="j" type="revolute">
    <parent link="base"/><child link="arm"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>"""


# ---------------------------------------------------------------------------
# parse_float_triple
# ---------------------------------------------------------------------------


class TestParseFloatTriple:
    """Float-triple parser tests.

    The float-triple parser is used by both geometry.py and (eventually) the
    GUI. Robustness matters: URDF authors routinely omit attributes.
    """

    def test_normal_triple(self) -> None:
        result = parse_float_triple("1.0 2.0 3.0")
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_none_returns_zeros(self) -> None:
        np.testing.assert_array_equal(parse_float_triple(None), [0.0, 0.0, 0.0])

    def test_empty_string_returns_zeros(self) -> None:
        np.testing.assert_array_equal(parse_float_triple(""), [0.0, 0.0, 0.0])

    def test_malformed_returns_zeros(self) -> None:
        np.testing.assert_array_equal(
            parse_float_triple("abc def ghi"), [0.0, 0.0, 0.0]
        )

    def test_wrong_length_returns_zeros(self) -> None:
        np.testing.assert_array_equal(parse_float_triple("1.0 2.0"), [0.0, 0.0, 0.0])

    def test_scientific_notation(self) -> None:
        result = parse_float_triple("1e-3 2.5e1 -3.0e0")
        np.testing.assert_allclose(result, [0.001, 25.0, -3.0])

    def test_returns_float64(self) -> None:
        assert parse_float_triple("1 2 3").dtype == np.float64


# ---------------------------------------------------------------------------
# visual_world_poses
# ---------------------------------------------------------------------------


class TestVisualWorldPoses:
    """visual_world_poses must compute T_world_link @ T_visual_origin."""

    def test_empty_when_no_meshes(self) -> None:
        poses = {"base": np.eye(4)}
        result = visual_world_poses(_SINGLE_LINK_NO_MESH, poses)
        assert result == {}

    def test_tracks_primitive_geometry(self) -> None:
        """Alignment rewrites a primitive's origin, so it must be checked.

        Skipping non-mesh visuals would leave a robot built from boxes and
        cylinders transformed but unverified.
        """
        poses = {"base": np.eye(4)}
        result = visual_world_poses(_PRIMITIVE_ONLY, poses)
        assert ("base", 0) in result

    def test_single_mesh_identity_link_pose(self) -> None:
        """With an identity link pose the world pose equals the origin."""
        poses = {"base": np.eye(4)}
        result = visual_world_poses(_SINGLE_LINK_ONE_MESH, poses)
        assert ("base", 0) in result
        # origin xyz="1 2 3" rpy="0 0 0" -> pure translation
        np.testing.assert_allclose(
            result[("base", 0)][:3, 3], [1.0, 2.0, 3.0], atol=1e-12
        )
        np.testing.assert_allclose(result[("base", 0)][:3, :3], np.eye(3), atol=1e-12)

    def test_link_pose_composes_with_origin(self) -> None:
        """world_pose = T_link @ T_origin, not just T_origin."""
        t_link = np.eye(4)
        t_link[:3, 3] = [10.0, 0.0, 0.0]
        poses = {"base": t_link}
        result = visual_world_poses(_SINGLE_LINK_ONE_MESH, poses)
        # origin is xyz="1 2 3"; link is at x=10; so world x = 11
        np.testing.assert_allclose(
            result[("base", 0)][:3, 3], [11.0, 2.0, 3.0], atol=1e-12
        )

    def test_rotated_origin_in_world_pose(self) -> None:
        """Visual origin rpy="pi/2 0 0" must appear in the rotation block.

        The URDF uses the decimal approximation `1.5708` rather than the exact
        `math.pi / 2`, so there is a small round-trip error (~3.7e-6) that the
        tolerance must accommodate.
        """
        poses = {"base": np.eye(4)}
        result = visual_world_poses(_SINGLE_LINK_ROTATED_ORIGIN, poses)
        rot = result[("base", 0)][:3, :3]
        # Rx(pi/2): y column -> (0, 0, 1), z column -> (0, -1, 0)
        np.testing.assert_allclose(
            rot @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-5
        )

    def test_two_visuals_get_separate_indices(self) -> None:
        """Two <visual> elements on one link yield indices 0 and 1."""
        chain = load_robot(_TWO_VISUAL_LINK).chain
        poses = chain.forward_kinematics()
        result = visual_world_poses(_TWO_VISUAL_LINK, poses)
        assert ("arm", 0) in result
        assert ("arm", 1) in result
        assert ("arm", 2) not in result

    def test_link_absent_from_poses_is_skipped(self) -> None:
        """If a link has no pose entry it must be silently skipped."""
        result = visual_world_poses(_SINGLE_LINK_ONE_MESH, {})
        assert result == {}

    def test_second_visual_origin_is_distinct(self) -> None:
        """The two visuals on 'arm' have different origins -- must differ."""
        chain = load_robot(_TWO_VISUAL_LINK).chain
        poses = chain.forward_kinematics()
        result = visual_world_poses(_TWO_VISUAL_LINK, poses)
        # They share the same link pose but differ in visual origin.
        assert not np.allclose(result[("arm", 0)], result[("arm", 1)], atol=1e-12)
