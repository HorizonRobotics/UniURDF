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

"""Tests for camera-frame compatibility links.

Axis normalization reorients link frames. A camera extrinsic calibrated
against one of those frames would silently become wrong, so this stage emits
an alias link holding the original frame. These tests pin two things: an alias
appears exactly when the frame moved, and it reproduces the source frame.
"""

import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from uniurdf.robot import load_robot
from uniurdf.spec import SpecError, parse_spec

# `wrist` hangs off a -Z joint, so normalization rotates its frame; `mount`
# hangs off the root by a fixed joint and cannot move.
MOVING_URDF = b"""<?xml version="1.0"?>
<robot name="cam">
  <link name="base"/><link name="wrist"/>
  <link name="grip"/><link name="mount"/>
  <joint name="j" type="revolute">
    <parent link="base"/><child link="wrist"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/><axis xyz="0 0 -1"/>
  </joint>
  <joint name="gj" type="fixed">
    <parent link="wrist"/><child link="grip"/>
  </joint>
  <joint name="mj" type="fixed">
    <parent link="base"/><child link="mount"/>
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
  </joint>
</robot>"""


def _joint_order(urdf: bytes) -> list[str]:
    return [name.decode() for name in re.findall(rb'joint name="([^"]+)"', urdf)]


class TestCompatLinkEmission:
    """When an alias is emitted, and when it is correctly withheld."""

    def test_emits_alias_when_the_frame_moves(self) -> None:
        aligned = load_robot(MOVING_URDF).align(
            parse_spec({"camera_references": ["wrist"]})
        )
        assert aligned.report.camera_compat_links == ("wrist_camera_mount_compat",)

    def test_alias_reproduces_the_source_frame(self) -> None:
        """The whole point: existing extrinsics keep working."""
        source = load_robot(MOVING_URDF)
        aligned = source.align(parse_spec({"camera_references": ["wrist"]}))

        before = source.fk()["wrist"]
        alias = aligned.fk()["wrist_camera_mount_compat"]
        np.testing.assert_allclose(alias, before, atol=1e-12)

        # Guard against a vacuous pass: the frame must actually have moved,
        # or the alias would trivially match.
        assert not np.allclose(before, aligned.fk()["wrist"], atol=1e-9)

    def test_no_alias_when_the_frame_is_unchanged(self) -> None:
        # An identity alias would be noise, so a frame that survives
        # normalization gets nothing.
        aligned = load_robot(MOVING_URDF).align(
            parse_spec({"camera_references": ["mount"]})
        )
        assert aligned.report.camera_compat_links == ()

    def test_no_references_means_no_work(self) -> None:
        aligned = load_robot(MOVING_URDF).align(parse_spec({}))
        assert aligned.report.camera_compat_links == ()

    def test_is_idempotent(self) -> None:
        spec = parse_spec({"camera_references": ["wrist"]})
        once = load_robot(MOVING_URDF).align(spec)
        twice = load_robot(once.urdf_bytes).align(spec)
        assert twice.urdf_bytes == once.urdf_bytes

    def test_a_re_run_still_reports_the_alias_it_kept(self) -> None:
        """Re-aligning moves nothing, and the alias is reported anyway.

        Pruning and the inventory are seeded from this list, so an alias
        missing from it is dropped from the pruned URDF while the full URDF
        keeps it -- and the inventory then points a consumer at a frame the
        pruned robot does not carry.
        """
        spec = parse_spec({"camera_references": ["wrist"]})
        once = load_robot(MOVING_URDF).align(spec)
        twice = load_robot(once.urdf_bytes).align(spec)
        assert twice.report.camera_compat_links == once.report.camera_compat_links
        assert "wrist_camera_mount_compat" in twice.report.camera_compat_links

    def test_a_re_run_keeps_the_alias_in_the_pruned_urdf(self) -> None:
        spec = parse_spec({"chains": [["j"]], "camera_references": ["wrist"]})
        once = load_robot(MOVING_URDF).align(spec)
        twice = load_robot(once.urdf_bytes).align(spec)
        assert b"wrist_camera_mount_compat" in twice.prune().urdf_bytes, (
            "the alias survived the first prune but not the second"
        )


class TestInsertionOrder:
    """Document order is load-bearing, and it is derived from the TCPs.

    A compat joint is emitted immediately ahead of its link's TCP joint.
    The position is derived from the ``tcp`` block rather than declared, so
    a spec cannot state one that contradicts the TCP it depends on.
    """

    def test_compat_joint_precedes_the_tcp_on_the_same_link(self) -> None:
        """A loader taking the first match must find the camera, not the tool.

        This is the case the deleted ``insert_before`` field existed to
        express, now derived from the TCP declaration itself.
        """
        aligned = load_robot(MOVING_URDF).align(
            parse_spec(
                {
                    "chains": [["j"]],
                    "tcp": [{"base_link": "wrist", "xyz": [0, 0, 0.1]}],
                    "camera_references": ["wrist"],
                }
            )
        )
        order = _joint_order(aligned.urdf_bytes)
        assert order.index("wrist_camera_mount_compat_joint") < order.index(
            "wrist_tcp_joint"
        )

    def test_joint_is_appended_when_the_link_has_no_tcp(self) -> None:
        """With no TCP to sit ahead of, the compat joint goes last."""
        aligned = load_robot(MOVING_URDF).align(
            parse_spec({"camera_references": ["wrist"]})
        )
        order = _joint_order(aligned.urdf_bytes)
        assert order[-1] == "wrist_camera_mount_compat_joint"

    def test_a_tcp_on_another_link_does_not_move_the_joint(self) -> None:
        """Only the reference link's own TCP governs the cut point.

        Without this, any TCP anywhere would drag every compat joint forward.
        """
        aligned = load_robot(MOVING_URDF).align(
            parse_spec(
                {
                    "chains": [["j"]],
                    "tcp": [{"base_link": "mount", "xyz": [0, 0, 0.1]}],
                    "camera_references": ["wrist"],
                }
            )
        )
        order = _joint_order(aligned.urdf_bytes)
        assert order[-1] == "wrist_camera_mount_compat_joint"

    def test_unknown_reference_link_is_rejected(self) -> None:
        with pytest.raises(SpecError, match="not a link in the URDF"):
            load_robot(MOVING_URDF).align(parse_spec({"camera_references": ["ghost"]}))


def test_alias_is_a_fixed_child_of_its_reference() -> None:
    aligned = load_robot(MOVING_URDF).align(
        parse_spec({"camera_references": ["wrist"]})
    )
    root = ET.fromstring(aligned.urdf_bytes)
    joint = next(
        element
        for element in root.findall("joint")
        if element.get("name") == "wrist_camera_mount_compat_joint"
    )
    assert joint.get("type") == "fixed"
    assert joint.find("parent").get("link") == "wrist"
    assert joint.find("child").get("link") == "wrist_camera_mount_compat"
