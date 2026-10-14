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

"""Tests for the numpy forward-kinematics chain."""

from pathlib import Path

import numpy as np
import pytest

from uniurdf.kinematics.chain import (
    build_chain,
    sweep_configuration,
)

GRAPH_URDF = """<?xml version="1.0"?>
<robot name="graph">
  <link name="base"/>
  <link name="left"/>
  <link name="right"/>
  <link name="shared"/>
  <link name="tip"/>
  <joint name="to_left" type="fixed">
    <parent link="base"/><child link="left"/>
    <origin xyz="0 1 0" rpy="0 0 0"/>
  </joint>
  <joint name="to_right" type="fixed">
    <parent link="base"/><child link="right"/>
    <origin xyz="0 -1 0" rpy="0 0 0"/>
  </joint>
  <joint name="left_to_shared" type="fixed">
    <parent link="left"/><child link="shared"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="shared_to_tip" type="fixed">
    <parent link="shared"/><child link="tip"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="right_to_shared" type="fixed">
    <parent link="right"/><child link="shared"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""


class TestBuildChain:
    """Parsing, rooting, and tree structure."""

    def test_identifies_root_and_links(self, simple_urdf: Path) -> None:
        chain = build_chain(simple_urdf)
        assert chain.root == "base"
        assert chain.link_names == ("base", "upper", "tool")

    def test_joint_names_in_dfs_preorder(self, simple_urdf: Path) -> None:
        assert build_chain(simple_urdf).joint_names == ("shoulder", "wrist")

    def test_accepts_raw_bytes(self, simple_urdf: Path) -> None:
        from_bytes = build_chain(simple_urdf.read_bytes())
        assert from_bytes.joint_names == build_chain(simple_urdf).joint_names

    def test_parses_limits(self, simple_urdf: Path) -> None:
        assert build_chain(simple_urdf).joint_limits() == {
            "shoulder": (-1.5, 1.5),
            "wrist": (0.0, 0.3),
        }

    def test_excludes_orphan_subtrees(self, orphan_urdf: Path) -> None:
        chain = build_chain(orphan_urdf)
        assert chain.link_names == ("base", "attached")
        assert set(chain.orphan_links) == {"floating", "floating_child"}
        # The orphaned joint must not occupy a slot in the joint vector.
        assert chain.joint_names == ()

    def test_child_link_lookup(self, simple_urdf: Path) -> None:
        chain = build_chain(simple_urdf)
        assert chain.child_link_of("shoulder") == "upper"
        assert chain.child_link_of("nope") is None

    def test_rejects_urdf_without_links(self) -> None:
        with pytest.raises(ValueError, match="no links"):
            build_chain(b'<?xml version="1.0"?><robot name="empty"/>')

    def test_rejects_rootless_urdf(self) -> None:
        cyclic = b"""<?xml version="1.0"?>
        <robot name="cyclic">
          <link name="a"/><link name="b"/>
          <joint name="ab" type="fixed">
            <parent link="a"/><child link="b"/>
          </joint>
          <joint name="ba" type="fixed">
            <parent link="b"/><child link="a"/>
          </joint>
        </robot>"""
        with pytest.raises(ValueError, match="no root link"):
            build_chain(cyclic)


class TestDuplicateChildLinks:
    """A link with two parent joints makes the URDF a graph, not a tree."""

    def test_duplicates_are_reported(self) -> None:
        chain = build_chain(GRAPH_URDF.encode())
        assert chain.duplicate_child_links == ("shared",)

    def test_last_joint_wins_for_link_and_subtree(self) -> None:
        """The whole subtree follows the winning branch, not just its root.

        Reproduces pytorch_kinematics, which rebuilds the subtree on each
        visit. Visiting once would leave `shared` on the right branch while
        `tip` stayed on the left -- an internally inconsistent pose.
        """
        poses = build_chain(GRAPH_URDF.encode()).forward_kinematics()
        np.testing.assert_allclose(poses["shared"][:3, 3], [0.0, -1.0, 0.0])
        np.testing.assert_allclose(poses["tip"][:3, 3], [1.0, -1.0, 0.0])

    def test_a_joint_reached_twice_holds_one_slot(self) -> None:
        """`joint_names` is the positional contract for a joint vector.

        Re-descending reaches the joints below a shared link once per branch
        that names it. Letting each visit add a slot would leave one name
        owning several indices, so a caller indexing by name addresses
        whichever the last write happened to be.
        """
        names = build_chain(GRAPH_URDF.encode()).joint_names
        assert len(names) == len(set(names))

    def test_a_diamond_is_not_reported_as_a_cycle(self) -> None:
        """Several paths to one link is legal; a link above itself is not."""
        chain = build_chain(GRAPH_URDF.encode())
        assert "shared" in chain.link_names


class TestCycleDetection:
    """A back edge is refused; re-descending a shared link is not."""

    @staticmethod
    def _ladder(levels: int) -> bytes:
        """A cycle-free graph where each level is reachable two ways."""
        parts = ['<?xml version="1.0"?>', '<robot name="d">', '<link name="L0"/>']
        for i in range(levels):
            parts += [
                f'<link name="A{i}"/>',
                f'<link name="B{i}"/>',
                f'<link name="L{i + 1}"/>',
            ]
            for child, parent in ((f"A{i}", f"L{i}"), (f"B{i}", f"L{i}")):
                parts.append(
                    f'<joint name="j_{child}" type="fixed">'
                    f'<parent link="{parent}"/><child link="{child}"/></joint>'
                )
            for parent in (f"A{i}", f"B{i}"):
                parts.append(
                    f'<joint name="j_{parent}_L{i + 1}" type="fixed">'
                    f'<parent link="{parent}"/>'
                    f'<child link="L{i + 1}"/></joint>'
                )
        parts.append("</robot>")
        return "".join(parts).encode()

    def test_a_back_edge_is_refused(self) -> None:
        urdf = b"""<?xml version="1.0"?>
        <robot name="loop">
          <link name="base"/><link name="a"/><link name="b"/>
          <joint name="j0" type="fixed">
            <parent link="base"/><child link="a"/></joint>
          <joint name="j1" type="fixed">
            <parent link="a"/><child link="b"/></joint>
          <joint name="j2" type="fixed">
            <parent link="b"/><child link="a"/></joint>
        </robot>"""
        with pytest.raises(ValueError, match="has a cycle"):
            build_chain(urdf)

    def test_a_deep_legal_graph_loads(self) -> None:
        """Counting visits cannot tell this from a loop.

        Every level here is reachable two ways, so the number of walks grows
        with depth while the graph stays a valid tree-shaped robot.
        """
        chain = build_chain(self._ladder(14))
        assert chain.root == "L0"
        assert "L14" in chain.link_names


class TestMimicJoints:
    """Mimic joints move but are not independently commandable."""

    def test_included_in_joint_vector(self, mimic_urdf: Path) -> None:
        # pytorch_kinematics gives mimic joints a slot; dropping them here
        # would shift every joint after the first mimic.
        assert build_chain(mimic_urdf).joint_names == ("drive", "follow")

    def test_excluded_from_independent_names(self, mimic_urdf: Path) -> None:
        chain = build_chain(mimic_urdf)
        assert chain.independent_joint_names == ("drive",)
        assert chain.mimic_joint_names == ("follow",)

    def test_value_is_derived_from_source(self, mimic_urdf: Path) -> None:
        resolved = build_chain(mimic_urdf).resolve_joint_values({"drive": 0.5})
        assert resolved["drive"] == pytest.approx(0.5)
        assert resolved["follow"] == pytest.approx(-0.5 + 0.25)

    def test_supplied_mimic_value_is_overridden(self, mimic_urdf: Path) -> None:
        resolved = build_chain(mimic_urdf).resolve_joint_values(
            {"drive": 0.0, "follow": 99.0}
        )
        assert resolved["follow"] == pytest.approx(0.25)


class TestForwardKinematics:
    """Pose composition for each joint type."""

    def test_zero_pose_applies_only_origins(self, simple_urdf: Path) -> None:
        poses = build_chain(simple_urdf).forward_kinematics()
        np.testing.assert_allclose(poses["base"], np.eye(4))
        np.testing.assert_allclose(poses["upper"][:3, 3], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(poses["tool"][:3, 3], [0.5, 0.0, 1.0])

    def test_revolute_joint_rotates_child(self, simple_urdf: Path) -> None:
        poses = build_chain(simple_urdf).forward_kinematics({"shoulder": np.pi / 2})
        # The tool sits 0.5 along the parent's +X; a quarter turn about +Z
        # swings it onto +Y.
        np.testing.assert_allclose(poses["tool"][:3, 3], [0.0, 0.5, 1.0], atol=1e-12)

    def test_prismatic_joint_translates_child(self, simple_urdf: Path) -> None:
        poses = build_chain(simple_urdf).forward_kinematics({"wrist": 0.25})
        np.testing.assert_allclose(poses["tool"][:3, 3], [0.75, 0.0, 1.0])

    def test_missing_joints_default_to_zero(self, simple_urdf: Path) -> None:
        chain = build_chain(simple_urdf)
        np.testing.assert_allclose(
            chain.forward_kinematics({"shoulder": 0.0})["tool"],
            chain.forward_kinematics()["tool"],
        )

    def test_poses_are_rigid_transforms(self, simple_urdf: Path) -> None:
        poses = build_chain(simple_urdf).forward_kinematics(
            {"shoulder": 0.4, "wrist": 0.1}
        )
        for matrix in poses.values():
            rotation = matrix[:3, :3]
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(matrix[3], [0, 0, 0, 1])


class TestSweepConfiguration:
    """The deterministic diagnostic pose."""

    def test_alternates_sign_and_grows(self) -> None:
        result = sweep_configuration(["a", "b", "c"], scale=0.1)
        assert result == pytest.approx({"a": 0.1, "b": -0.2, "c": 0.3})

    def test_empty_input(self) -> None:
        assert sweep_configuration([]) == {}
