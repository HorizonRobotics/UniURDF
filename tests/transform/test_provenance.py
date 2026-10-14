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

"""The provenance record that makes ``rot_z`` idempotent.

Without a record of what a previous run rotated, a second pass cannot tell a
joint that was always ``+Z`` from one that was turned 90 degrees to get there,
and would turn it again.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from uniurdf.robot import _serialize
from uniurdf.transform.provenance import (
    PROVENANCE_TAG,
    PROVENANCE_WRAPPER,
    append_provenance,
    read_applied_rot_z,
    strip_provenance,
)

BARE = b"""<?xml version="1.0"?>
<robot name="r">
  <link name="a"/>
</robot>"""

ONE_JOINT = b"""<?xml version="1.0"?>
<robot name="r">
  <link name="a"/>
  <link name="b"/>
  <joint name="j" type="revolute">
    <parent link="a"/>
    <child link="b"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>"""


class TestRoundTrip:
    """What is written must come back."""

    def test_append_then_read(self) -> None:
        root = ET.fromstring(BARE)
        append_provenance(root, {"j1": 90.0, "j2": -37.5})
        assert read_applied_rot_z(root) == {"j1": 90.0, "j2": -37.5}

    def test_survives_serialization(self) -> None:
        """The record must live in the file, not just in memory.

        An XML comment would not: ElementTree's default parser drops comments
        silently, so the next run would see no record and rotate again.
        """
        root = ET.fromstring(BARE)
        append_provenance(root, {"j1": 90.0})
        reparsed = ET.fromstring(_serialize(root))
        assert read_applied_rot_z(reparsed) == {"j1": 90.0}

    def test_entries_are_sorted(self) -> None:
        """Output must not depend on mapping order."""
        first = ET.fromstring(BARE)
        append_provenance(first, {"b": 1.0, "a": 2.0})
        second = ET.fromstring(BARE)
        append_provenance(second, {"a": 2.0, "b": 1.0})
        assert _serialize(first) == _serialize(second)


class TestAbsence:
    """A URDF with no record, and a record with nothing in it."""

    def test_reads_empty_when_absent(self) -> None:
        assert read_applied_rot_z(ET.fromstring(BARE)) == {}

    def test_zero_angles_emit_no_element(self) -> None:
        """A spec with no rotation must produce the same bytes as before.

        Emitting an empty record would change every shipped robot's output
        for no reason.
        """
        root = ET.fromstring(BARE)
        append_provenance(root, {"j1": 0.0})
        assert root.find(PROVENANCE_TAG) is None
        assert _serialize(root) == _serialize(ET.fromstring(BARE))

    def test_empty_mapping_emits_no_element(self) -> None:
        root = ET.fromstring(BARE)
        append_provenance(root, {})
        assert root.find(PROVENANCE_TAG) is None


class TestStrip:
    """Removal, and why it has to happen first."""

    def test_strip_returns_and_removes(self) -> None:
        root = ET.fromstring(BARE)
        append_provenance(root, {"j1": 90.0})
        assert strip_provenance(root) == {"j1": 90.0}
        assert root.find(PROVENANCE_TAG) is None

    def test_strip_of_absent_record_is_harmless(self) -> None:
        root = ET.fromstring(BARE)
        assert strip_provenance(root) == {}
        assert _serialize(root) == _serialize(ET.fromstring(BARE))

    def test_strip_removes_every_copy(self) -> None:
        """A hand-edited file could carry more than one.

        Leaving a copy behind would make the next run read a stale angle.
        """
        root = ET.fromstring(BARE)
        append_provenance(root, {"j1": 90.0})
        append_provenance(root, {"j2": 45.0})
        strip_provenance(root)
        assert root.find(PROVENANCE_TAG) is None


class TestParserCompatibility:
    """The record must not provoke the parsers that read our output."""

    def test_record_is_nested_inside_the_wrapper(self) -> None:
        """A bare child of <robot> is an unknown tag to a URDF parser."""
        root = ET.fromstring(BARE)
        append_provenance(root, {"j1": 90.0})
        assert root.find(PROVENANCE_TAG) is None
        assert root.find(f"{PROVENANCE_WRAPPER}/{PROVENANCE_TAG}") is not None

    def test_pytorch_kinematics_parses_without_warning(self, capfd) -> None:
        """The parser training loads with must stay silent.

        It writes an "Unknown tag" line to stderr for every element ``<robot>``
        does not declare, once per URDF per worker. It declares ``gazebo`` as a
        passthrough, which is why the record lives there.
        """
        pk = pytest.importorskip("pytorch_kinematics")
        root = ET.fromstring(ONE_JOINT)
        append_provenance(root, {"j": 90.0})
        capfd.readouterr()
        pk.build_chain_from_urdf(_serialize(root))
        assert PROVENANCE_TAG not in capfd.readouterr().err

    def test_a_legacy_record_is_still_read(self) -> None:
        """URDFs aligned before the move keep their record.

        Missing it would make the next run apply the angle a second time.
        """
        root = ET.fromstring(BARE)
        block = ET.SubElement(root, PROVENANCE_TAG)
        entry = ET.SubElement(block, "rot_z")
        entry.set("joint", "j1")
        entry.set("degrees", "90.0")
        assert read_applied_rot_z(root) == {"j1": 90.0}

    def test_a_legacy_record_is_stripped(self) -> None:
        """Re-aligning an older file must not leave the old element behind."""
        root = ET.fromstring(BARE)
        block = ET.SubElement(root, PROVENANCE_TAG)
        entry = ET.SubElement(block, "rot_z")
        entry.set("joint", "j1")
        entry.set("degrees", "90.0")
        assert strip_provenance(root) == {"j1": 90.0}
        assert root.find(PROVENANCE_TAG) is None

    def test_an_unrelated_gazebo_block_survives(self) -> None:
        """A wrapper the robot already had is not ours to delete."""
        root = ET.fromstring(BARE)
        existing = ET.SubElement(root, PROVENANCE_WRAPPER)
        ET.SubElement(existing, "plugin").set("name", "controller")
        append_provenance(root, {"j1": 90.0})
        strip_provenance(root)
        wrappers = root.findall(PROVENANCE_WRAPPER)
        assert len(wrappers) == 1
        assert wrappers[0].find("plugin") is not None


class TestCorruption:
    """A damaged record must not move the robot."""

    def test_unparseable_angle_is_ignored(self) -> None:
        """Treating garbage as 0 is safe; guessing at it is not.

        The joint is then handled as unrotated, which at worst re-applies a
        rotation the user asked for.
        """
        root = ET.fromstring(BARE)
        block = ET.SubElement(root, PROVENANCE_TAG)
        entry = ET.SubElement(block, "rot_z")
        entry.set("joint", "j1")
        entry.set("degrees", "not-a-number")
        assert read_applied_rot_z(root) == {}

    def test_entry_without_a_joint_name_is_ignored(self) -> None:
        root = ET.fromstring(BARE)
        block = ET.SubElement(root, PROVENANCE_TAG)
        entry = ET.SubElement(block, "rot_z")
        entry.set("degrees", "90.0")
        assert read_applied_rot_z(root) == {}
