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

"""Tests for TCP link attachment.

The stage exists to be re-runnable: an aligned URDF fed back in must come out
byte-identical, while a spec whose numbers changed must take effect. Both
depend on recognising a TCP joint that is already there, so most of these
tests drive that recognition rather than the insertion.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from uniurdf.spec import TcpSpec
from uniurdf.transform.tcp import insert_tcp_links

BASE_URDF = """<?xml version="1.0"?>
<robot name="arm">
  <link name="base"/><link name="wrist"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="wrist"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>"""


def _root() -> ET.Element:
    return ET.fromstring(BASE_URDF)


def _tcp(**overrides) -> TcpSpec:
    fields = {"base_link": "wrist", "xyz": (0.0, 0.0, 0.15), "rpy": (0.0, 0.0, 0.0)}
    fields.update(overrides)
    return TcpSpec(**fields)


def _joint(root: ET.Element, name: str) -> ET.Element | None:
    return next(
        (j for j in root.findall("joint") if j.get("name") == name),
        None,
    )


class TestInsertion:
    """What a first run produces."""

    def test_emits_a_fixed_joint_and_its_link(self) -> None:
        root = _root()
        changed = insert_tcp_links(root, [_tcp()])

        assert changed == ["wrist_tcp_joint"]
        joint = _joint(root, "wrist_tcp_joint")
        assert joint is not None
        assert joint.get("type") == "fixed"
        assert joint.find("parent").get("link") == "wrist"
        assert joint.find("child").get("link") == "wrist_tcp"
        assert any(link.get("name") == "wrist_tcp" for link in root.findall("link"))

    def test_carries_the_declared_origin(self) -> None:
        root = _root()
        insert_tcp_links(root, [_tcp(xyz=(0.1, -0.2, 0.3), rpy=(0.0, 1.5, 0.0))])

        origin = _joint(root, "wrist_tcp_joint").find("origin")
        assert [float(v) for v in origin.get("xyz").split()] == [0.1, -0.2, 0.3]
        assert [float(v) for v in origin.get("rpy").split()] == [0.0, 1.5, 0.0]

    def test_base_link_is_left_untouched(self) -> None:
        """Extrinsics already targeting the base link keep resolving."""
        root = _root()
        insert_tcp_links(root, [_tcp()])

        names = [link.get("name") for link in root.findall("link")]
        assert "wrist" in names
        assert "wrist_tcp_base" not in names

    def test_unknown_base_link_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="is not a link in this"):
            insert_tcp_links(_root(), [_tcp(base_link="ghost")])

    def test_no_tcps_changes_nothing(self) -> None:
        root = _root()
        before = ET.tostring(root)
        assert insert_tcp_links(root, []) == []
        assert ET.tostring(root) == before


class TestReRun:
    """A second run over its own output."""

    def test_is_a_no_op(self) -> None:
        root = _root()
        insert_tcp_links(root, [_tcp()])
        after_first = ET.tostring(root)

        assert insert_tcp_links(root, [_tcp()]) == []
        assert ET.tostring(root) == after_first

    def test_emits_no_duplicate_link(self) -> None:
        root = _root()
        insert_tcp_links(root, [_tcp()])
        insert_tcp_links(root, [_tcp()])

        names = [link.get("name") for link in root.findall("link")]
        assert names.count("wrist_tcp") == 1

    def test_a_changed_origin_is_rewritten_in_place(self) -> None:
        """The spec is the source of truth; stale geometry is replaced."""
        root = _root()
        insert_tcp_links(root, [_tcp(xyz=(0.0, 0.0, 0.15))])
        changed = insert_tcp_links(root, [_tcp(xyz=(0.0, 0.0, 0.25))])

        assert changed == ["wrist_tcp_joint"]
        joints = [
            j for j in root.findall("joint") if j.get("name") == "wrist_tcp_joint"
        ]
        assert len(joints) == 1, "the superseded joint must not survive alongside"
        origin = joints[0].find("origin")
        assert [float(v) for v in origin.get("xyz").split()] == [0.0, 0.0, 0.25]

    def test_a_difference_below_tolerance_is_not_a_change(self) -> None:
        root = _root()
        insert_tcp_links(root, [_tcp(xyz=(0.0, 0.0, 0.15))])
        assert insert_tcp_links(root, [_tcp(xyz=(0.0, 0.0, 0.15 + 1e-12))]) == []


class TestRecognisingAnExistingJoint:
    """Each way a joint at the TCP's name fails to be that TCP.

    Any of these leaves the tool with a name collision it has to resolve by
    rewriting, so a wrong answer here either duplicates a joint or silently
    keeps geometry the spec no longer asks for.
    """

    @staticmethod
    def _with_joint(xml: str) -> ET.Element:
        root = _root()
        root.append(ET.fromstring(xml))
        link = ET.SubElement(root, "link")
        link.set("name", "wrist_tcp")
        return root

    def test_wrong_joint_type(self) -> None:
        root = self._with_joint(
            '<joint name="wrist_tcp_joint" type="revolute">'
            '<parent link="wrist"/><child link="wrist_tcp"/>'
            '<origin rpy="0 0 0" xyz="0 0 0.15"/></joint>'
        )
        assert insert_tcp_links(root, [_tcp()]) == ["wrist_tcp_joint"]
        assert _joint(root, "wrist_tcp_joint").get("type") == "fixed"

    def test_missing_parent_element(self) -> None:
        root = self._with_joint(
            '<joint name="wrist_tcp_joint" type="fixed">'
            '<child link="wrist_tcp"/>'
            '<origin rpy="0 0 0" xyz="0 0 0.15"/></joint>'
        )
        assert insert_tcp_links(root, [_tcp()]) == ["wrist_tcp_joint"]
        assert _joint(root, "wrist_tcp_joint").find("parent") is not None

    def test_missing_child_element(self) -> None:
        root = self._with_joint(
            '<joint name="wrist_tcp_joint" type="fixed">'
            '<parent link="wrist"/>'
            '<origin rpy="0 0 0" xyz="0 0 0.15"/></joint>'
        )
        assert insert_tcp_links(root, [_tcp()]) == ["wrist_tcp_joint"]

    def test_parent_names_a_different_link(self) -> None:
        root = self._with_joint(
            '<joint name="wrist_tcp_joint" type="fixed">'
            '<parent link="base"/><child link="wrist_tcp"/>'
            '<origin rpy="0 0 0" xyz="0 0 0.15"/></joint>'
        )
        assert insert_tcp_links(root, [_tcp()]) == ["wrist_tcp_joint"]
        assert _joint(root, "wrist_tcp_joint").find("parent").get("link") == "wrist"

    def test_child_names_a_different_link(self) -> None:
        root = self._with_joint(
            '<joint name="wrist_tcp_joint" type="fixed">'
            '<parent link="wrist"/><child link="somewhere_else"/>'
            '<origin rpy="0 0 0" xyz="0 0 0.15"/></joint>'
        )
        assert insert_tcp_links(root, [_tcp()]) == ["wrist_tcp_joint"]

    def test_unparseable_origin(self) -> None:
        root = self._with_joint(
            '<joint name="wrist_tcp_joint" type="fixed">'
            '<parent link="wrist"/><child link="wrist_tcp"/>'
            '<origin rpy="0 0 0" xyz="nan_x 0 0.15"/></joint>'
        )
        assert insert_tcp_links(root, [_tcp()]) == ["wrist_tcp_joint"]
        origin = _joint(root, "wrist_tcp_joint").find("origin")
        assert [float(v) for v in origin.get("xyz").split()] == [0.0, 0.0, 0.15]

    def test_origin_with_too_few_components(self) -> None:
        root = self._with_joint(
            '<joint name="wrist_tcp_joint" type="fixed">'
            '<parent link="wrist"/><child link="wrist_tcp"/>'
            '<origin rpy="0 0 0" xyz="0 0.15"/></joint>'
        )
        assert insert_tcp_links(root, [_tcp()]) == ["wrist_tcp_joint"]

    def test_absent_origin_reads_as_the_zero_pose(self) -> None:
        """A joint with no origin is at the zero pose, and matches one."""
        root = self._with_joint(
            '<joint name="wrist_tcp_joint" type="fixed">'
            '<parent link="wrist"/><child link="wrist_tcp"/></joint>'
        )
        assert insert_tcp_links(root, [_tcp(xyz=(0.0, 0.0, 0.0))]) == []


class TestSeveralTcps:
    """One entry per chain, each independent."""

    def test_each_gets_its_own_link(self) -> None:
        root = ET.fromstring(
            """<?xml version="1.0"?>
<robot name="two">
  <link name="base"/><link name="left"/><link name="right"/>
  <joint name="jl" type="fixed">
    <parent link="base"/><child link="left"/></joint>
  <joint name="jr" type="fixed">
    <parent link="base"/><child link="right"/></joint>
</robot>"""
        )
        changed = insert_tcp_links(
            root,
            [_tcp(base_link="left"), _tcp(base_link="right")],
        )
        assert changed == ["left_tcp_joint", "right_tcp_joint"]
        names = {link.get("name") for link in root.findall("link")}
        assert {"left_tcp", "right_tcp"} <= names

    def test_one_unknown_base_link_rejects_the_whole_stage(self) -> None:
        root = _root()
        with pytest.raises(ValueError, match="is not a link in this"):
            insert_tcp_links(root, [_tcp(), _tcp(base_link="ghost")])
