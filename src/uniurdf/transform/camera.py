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

"""Preserve calibrated camera frames across axis normalization.

Normalizing a joint axis to ``+Z`` reorients its child link's frame. That is
the point of alignment -- but a camera extrinsic calibrated against one of
those frames silently becomes wrong. For each declared reference link whose
frame actually moved, this stage emits a ``<link>_camera_mount_compat`` child
holding the transform that recovers the original frame, so existing extrinsics
keep working by targeting the compat link.

A reference whose frame did not move gets no compat link: an identity alias
would be noise.

Insertion order matters. Some downstream loaders resolve extrinsics by walking
a link's children in document order and taking the first match, so a compat
joint emitted after a gripper subtree would never be found. A compat joint
therefore goes immediately ahead of its link's TCP joint when the link carries
one, and is appended otherwise.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence

import numpy as np

from uniurdf.spec import CameraReference
from uniurdf.transform._linalg import OriginTransform, format_vec

# Frames are compared at zero pose; anything above this is a real move rather
# than float noise from re-expressing origins through rotation matrices.
_TOLERANCE = 1e-6


def insert_camera_compat_links(
    root: ET.Element,
    source_poses: Mapping[str, np.ndarray],
    aligned_poses: Mapping[str, np.ndarray],
    references: Sequence[CameraReference],
    tcp_joint_names: Mapping[str, str] | None = None,
) -> list[str]:
    """Emit compat links for reference frames that alignment moved.

    Args:
        root (ET.Element): The ``<robot>`` element, modified in place.
        source_poses (Mapping[str, np.ndarray]): Zero-pose link transforms of
            the source URDF.
        aligned_poses (Mapping[str, np.ndarray]): Zero-pose link transforms of
            the aligned URDF, before this stage runs.
        references (Sequence[CameraReference]): Declared reference links.
        tcp_joint_names (Mapping[str, str], optional): Maps a link name to the
            name of the TCP joint hanging off it. A reference on such a link
            has its compat joint emitted immediately ahead of that TCP joint;
            any other reference is appended. Default is no TCPs.

    Returns:
        list[str]: Names of the compat links emitted, in order.

    Raises:
        ValueError: If a reference link is absent from either URDF.
    """
    tcp_joints = tcp_joint_names or {}
    emitted: list[str] = []
    for reference in references:
        source = source_poses.get(reference.link)
        aligned = aligned_poses.get(reference.link)
        if source is None or aligned is None:
            missing = "source" if source is None else "aligned"
            raise ValueError(
                f"camera reference link '{reference.link}' is not reachable "
                f"in the {missing} URDF"
            )
        if _existing_joint(root, reference) is not None:
            # An earlier run placed this alias, and re-running moves nothing,
            # so the comparison below would read the frame as untouched. The
            # alias is reported anyway: callers seed pruning and the inventory
            # from this list, and a compat link missing from it is dropped
            # from the pruned URDF it belongs in.
            emitted.append(reference.compat_link_name)
            continue
        if np.allclose(source, aligned, atol=_TOLERANCE):
            # The frame survived normalization untouched, so extrinsics
            # targeting it are still correct and need no alias.
            continue

        # The transform that takes the aligned frame back to the original one,
        # expressed in the aligned link's own frame.
        compensator = np.linalg.inv(aligned) @ source
        _insert(root, reference, compensator, tcp_joints.get(reference.link))
        emitted.append(reference.compat_link_name)
    return emitted


def _insert(
    root: ET.Element,
    reference: CameraReference,
    compensator: np.ndarray,
    tcp_joint_name: str | None,
) -> None:
    """Append the compat link and place its joint at the derived position.

    The reference must not already carry a compat joint; calling this for one
    that does gives the URDF two links of the same name.
    """
    link = ET.SubElement(root, "link")
    link.set("name", reference.compat_link_name)

    joint = ET.Element("joint")
    joint.set("name", _joint_name(reference))
    joint.set("type", "fixed")
    origin = ET.SubElement(joint, "origin")
    rpy, xyz = OriginTransform.from_matrix(compensator).to_rpy_xyz()
    origin.set("rpy", format_vec(rpy))
    origin.set("xyz", format_vec(xyz))
    parent = ET.SubElement(joint, "parent")
    parent.set("link", reference.link)
    child = ET.SubElement(joint, "child")
    child.set("link", reference.compat_link_name)

    root.insert(_insertion_index(root, tcp_joint_name), joint)


def _insertion_index(root: ET.Element, tcp_joint_name: str | None) -> int:
    """Where the compat joint goes.

    Ahead of the reference link's TCP joint when it has one, so a loader that
    takes the first matching child finds the calibration frame rather than the
    tool frame. Appended otherwise.

    A TCP joint that is somehow absent falls through to append rather than
    raising: the compat link is still correct, only its document position is
    less useful, and refusing to emit it would be the worse failure.
    """
    if tcp_joint_name is not None:
        for index, element in enumerate(root):
            if element.tag == "joint" and element.get("name") == tcp_joint_name:
                return index
    return len(list(root))


def _existing_joint(root: ET.Element, reference: CameraReference) -> ET.Element | None:
    """Return a previously emitted compat joint for `reference`, if any."""
    name = _joint_name(reference)
    for element in root.findall("joint"):
        if element.get("name") == name:
            return element
    return None


def _joint_name(reference: CameraReference) -> str:
    """Name of the fixed joint attaching a compat link to its reference."""
    return f"{reference.compat_link_name}_joint"


__all__ = ["insert_camera_compat_links"]
