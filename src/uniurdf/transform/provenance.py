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

"""Record which manual Z rotations an aligned URDF already carries.

Without this, ``rot_z`` could not be idempotent. Alignment normalises every
actuated axis to ``+Z``, so a second pass over its own output cannot tell a
joint that was always ``+Z`` from one rotated 90 degrees to get there -- and
would apply the rotation a second time. Re-running the tool on the directory
it just wrote is the documented workflow, so silent drift is not acceptable.

The applied angles are therefore written into the output as a single
``<uniurdf>`` element and read back on the next run, which then applies
only the difference. Records written under an earlier element name are read
too, and rewritten under the current one.

Three details are load-bearing, all established by measurement:

* **Nested inside ``<gazebo>``.** ``pytorch_kinematics`` writes an "Unknown
  tag" line to stderr for any element ``<robot>`` does not declare, and it
  parses every URDF a training job loads. It declares ``gazebo`` as a raw
  passthrough, so nesting the record there keeps it out of the parser's way.
* **One element, not per-joint attributes.** That parser warns once per
  unknown *attribute* too, so tagging joints individually produced a warning
  per joint.
* **An XML comment cannot be used.** ``ElementTree``'s default parser silently
  drops comments, so the record would vanish on the next parse.

The wrapper must also be stripped before alignment runs and appended after
every other stage, or a second pass places it mid-document and the output
differs from the first by block position alone.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping

#: Element holding the provenance record.
PROVENANCE_TAG = "uniurdf"

#: Tags read but never written. A URDF aligned by an earlier release carries
#: the record under a different name; without this its angles would read as
#: absent and a re-align would rotate the joint a second time.
LEGACY_PROVENANCE_TAGS = ("urdf_align",)

#: Wrapper the record sits inside, chosen because URDF parsers pass it
#: through untouched.
PROVENANCE_WRAPPER = "gazebo"

# Angles below this are treated as absent: an entry recording "0 degrees" is
# indistinguishable from no entry at all, and omitting it keeps the output of
# a spec with no rot_z byte-identical to one that never had the field.
_ZERO_EPS = 1e-12


def _find_blocks(root: ET.Element) -> list[tuple[ET.Element, ET.Element]]:
    """Locate provenance records as ``(parent, block)`` pairs.

    Two things vary across releases: the record moved inside ``<gazebo>``,
    and its element was renamed. Every combination is read, so a URDF aligned
    by any earlier version keeps its angles and is not rotated a second time.
    Only :data:`PROVENANCE_TAG` is ever written, so a re-align migrates the
    document and leaves no stale element.
    """
    tags = (PROVENANCE_TAG, *LEGACY_PROVENANCE_TAGS)
    found = [
        (wrapper, block)
        for wrapper in root.findall(PROVENANCE_WRAPPER)
        for tag in tags
        for block in wrapper.findall(tag)
    ]
    found.extend((root, block) for tag in tags for block in root.findall(tag))
    return found


def read_applied_rot_z(root: ET.Element) -> dict[str, float]:
    """Return the Z rotations already applied, in degrees, by joint name.

    Args:
        root (ET.Element): The ``<robot>`` element.

    Returns:
        dict[str, float]: Applied angles in degrees. Empty when the URDF
        carries no record, which is the case for every source URDF.
    """
    applied: dict[str, float] = {}
    for _parent, block in _find_blocks(root):
        for entry in block.findall("rot_z"):
            name = entry.get("joint")
            degrees = entry.get("degrees")
            if not name or degrees is None:
                continue
            try:
                applied[name] = float(degrees)
            except ValueError:
                # A corrupted record is worse than none: applying a bogus
                # delta would move the robot. Skip the entry and let the
                # joint be treated as unrotated.
                continue
    return applied


def strip_provenance(root: ET.Element) -> dict[str, float]:
    """Remove the provenance record and return what it recorded.

    Call this before any transform runs. Leaving the record in place would
    park it ahead of links and joints appended later, so a second pass would
    emit it at a different index and the output would differ from the first
    by position alone.

    Args:
        root (ET.Element): The ``<robot>`` element, modified in place.

    Returns:
        dict[str, float]: Previously applied angles, in degrees.
    """
    applied = read_applied_rot_z(root)
    for parent, block in _find_blocks(root):
        parent.remove(block)
    # A wrapper this tool created holds nothing else; one that was already in
    # the document keeps whatever else it carries.
    for wrapper in root.findall(PROVENANCE_WRAPPER):
        if len(wrapper) == 0 and not wrapper.attrib:
            root.remove(wrapper)
    return applied


def append_provenance(root: ET.Element, rot_z: Mapping[str, float]) -> None:
    """Record the Z rotations this run applied, in degrees.

    Call this last, after every stage that appends to ``<robot>``. Entries are
    sorted by joint name so the output does not depend on mapping order, and
    zero angles are omitted so a spec without ``rot_z`` emits no element.

    Args:
        root (ET.Element): The ``<robot>`` element, modified in place.
        rot_z (Mapping[str, float]): Angles applied, in degrees.
    """
    applied = {
        name: float(degrees)
        for name, degrees in rot_z.items()
        if abs(float(degrees)) > _ZERO_EPS
    }
    if not applied:
        return
    wrapper = ET.SubElement(root, PROVENANCE_WRAPPER)
    block = ET.SubElement(wrapper, PROVENANCE_TAG)
    for name in sorted(applied):
        entry = ET.SubElement(block, "rot_z")
        entry.set("joint", name)
        entry.set("degrees", repr(applied[name]))


__all__ = [
    "LEGACY_PROVENANCE_TAGS",
    "PROVENANCE_TAG",
    "PROVENANCE_WRAPPER",
    "append_provenance",
    "read_applied_rot_z",
    "strip_provenance",
]
