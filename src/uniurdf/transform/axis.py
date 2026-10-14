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

"""Actuated-joint axis normalization for URDF alignment.

For every actuated (`revolute`, `continuous`, `prismatic`) joint whose axis
after unit-normalization does not equal `ẑ`, we rewrite the joint frame so the
axis becomes `0 0 1` while preserving control semantics: the same joint values
must still produce the same physical motion.

The rewrite:

    1. Compute ``S`` that maps ``ẑ`` to the old axis via minimum rotation.
    2. Post-multiply the joint's ``<origin>`` rotation by ``S``.
    3. Re-express visual / collision / inertial ``<origin>`` blocks of the
       immediate child link by ``S⁻¹`` (translation and rpy).
    4. Re-express every outgoing joint's ``<origin>`` from that child link by
       ``S⁻¹``.
    5. Rewrite the joint's ``<axis>`` to ``0 0 1``.

Pinning the axis to ``+Z`` fixes two of three rotational degrees of freedom and
leaves the roll about ``Z`` free -- today it is whatever the minimum rotation
happens to produce, which differs between robots that are otherwise alike. The
optional ``rot_z`` argument is how a caller chooses that roll: ``S`` becomes
``S @ Rz(θ)``, and because the same ``S`` is inverted into the child's origins
in steps 3 and 4, every mesh stays exactly where it was. Only frames move.

Inertia tensor values are left untouched: because the ``<inertial>``
``<origin>`` is also rotated by ``S⁻¹``, the CoM's rotation matrix
relative to the world frame is unchanged, and the tensor stays expressed
in the same physical frame.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping

import numpy as np

from uniurdf.transform._linalg import (
    ACTUATED_JOINT_TYPES,
    OriginTransform,
    format_vec,
    is_unit,
    rotation_from_ez_to,
    rotation_z,
    unit,
)

_LOGGER = logging.getLogger(__name__)

# Below this, a requested Z rotation is treated as absent. Anything smaller
# would not survive the origin round-trip anyway, and skipping it is what
# keeps a zero-valued entry byte-identical to no entry at all.
_ROT_Z_EPS = 1e-12


def normalize_actuated_axes(
    root: ET.Element,
    rot_z: Mapping[str, float] | None = None,
) -> list[str]:
    """Rewrite the URDF `<robot>` element so actuated axes are `0 0 1`.

    Args:
        root (ET.Element): The XML ``<robot>`` element. Modified in place.
        rot_z (Mapping[str, float], optional): Extra rotation about each named
            joint's own ``+Z``, in **radians**, applied after the axis is
            pinned. Sparse -- an absent joint gets none. Default is none.

    Returns:
        list[str]: Ordered names of the joints whose axes were rewritten.

    Raises:
        ValueError: If an actuated joint has no child link, or names one the
            URDF does not declare.
    """
    angles = rot_z or {}

    joints_by_name = {joint.get("name", ""): joint for joint in root.findall("joint")}
    links_by_name = {link.get("name", ""): link for link in root.findall("link")}

    outgoing_by_parent: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        parent_link = _child_get(joint, "parent", "link")
        if parent_link:
            outgoing_by_parent.setdefault(parent_link, []).append(joint)

    rewritten: list[str] = []
    for name, joint in joints_by_name.items():
        joint_type = joint.get("type", "")
        if joint_type not in ACTUATED_JOINT_TYPES:
            continue
        axis = _read_axis(joint)
        if axis is None:
            continue
        theta = float(angles.get(name, 0.0))
        if _is_close_to_positive_z(axis) and abs(theta) <= _ROT_Z_EPS:
            # Nothing to compensate. Taking this path is not an optimisation:
            # sending an already-aligned joint through the identity rewrite
            # below reformats its origins (`0 0 0.123` -> `0.0 0.0 0.123`),
            # which would churn most shipped robots and break byte-level
            # idempotence for no gain.
            _write_axis(joint, np.array([0.0, 0.0, 1.0]))
            continue
        s_matrix = rotation_from_ez_to(axis)
        if abs(theta) > _ROT_Z_EPS:
            # Composed on the right so the axis still lands on +Z: the extra
            # turn happens in the joint's own frame, about the very axis being
            # pinned, which is exactly the freedom normalisation leaves open.
            s_matrix = s_matrix @ rotation_z(theta)
        _post_multiply_joint_origin(joint, s_matrix)
        _write_axis(joint, np.array([0.0, 0.0, 1.0]))
        child_link_name = _child_get(joint, "child", "link")
        if not child_link_name:
            raise ValueError(
                f"joint '{name}' has no <child link=..>; cannot normalize axis"
            )
        child_link = links_by_name.get(child_link_name)
        if child_link is None:
            raise ValueError(
                f"joint '{name}' references missing child link '{child_link_name}'"
            )
        s_inv = s_matrix.T
        _rewrite_link_origins(child_link, s_inv)
        for outgoing in outgoing_by_parent.get(child_link_name, ()):
            _rewrite_element_origin(outgoing, s_inv)
        rewritten.append(name)

    return rewritten


# ---------------------------------------------------------------------------
# Element-level helpers.
# ---------------------------------------------------------------------------


def _read_axis(joint: ET.Element) -> np.ndarray | None:
    axis_el = joint.find("axis")
    if axis_el is None:
        return None
    raw = axis_el.get("xyz")
    if not raw:
        return None
    try:
        values = [float(v) for v in raw.split()]
    except ValueError:
        return None
    if len(values) != 3:
        return None
    vec = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        # `axis xyz="0 0 0"` is a URDF quirk seen on non-actuated placeholder
        # joints; leave those alone.
        return None
    if not is_unit(vec):
        _LOGGER.warning(
            "joint '%s' axis is not unit length (%.6f); normalizing",
            joint.get("name", "<unnamed>"),
            norm,
        )
    return unit(vec)


def _write_axis(joint: ET.Element, axis: Iterable[float]) -> None:
    axis_el = joint.find("axis")
    if axis_el is None:
        axis_el = ET.SubElement(joint, "axis")
    axis_el.set("xyz", format_vec(axis))


def _is_close_to_positive_z(axis: np.ndarray, atol: float = 1e-6) -> bool:
    return bool(np.allclose(unit(axis), [0.0, 0.0, 1.0], atol=atol))


def _post_multiply_joint_origin(
    joint: ET.Element,
    s_matrix: np.ndarray,
) -> None:
    """Set the joint origin's rotation to `R_old @ S`; leave xyz unchanged."""
    origin = _ensure_origin(joint)
    transform = _read_origin(origin)
    new_rot = transform.rot @ s_matrix
    _write_origin(
        origin,
        OriginTransform(xyz=transform.xyz, rot=new_rot),
    )


def _rewrite_link_origins(link: ET.Element, s_inv: np.ndarray) -> None:
    """Re-express visual/collision/inertial `<origin>` by `S⁻¹` in-place."""
    for tag in ("inertial", "visual", "collision"):
        for element in link.findall(tag):
            _rewrite_element_origin(element, s_inv)


def _rewrite_element_origin(element: ET.Element, s_inv: np.ndarray) -> None:
    """Multiply the `<origin>` of `element` by `s_inv` on the left."""
    origin = element.find("origin")
    if origin is None:
        origin = ET.SubElement(element, "origin")
        origin.set("xyz", "0 0 0")
        origin.set("rpy", "0 0 0")
    transform = _read_origin(origin)
    new_rot = s_inv @ transform.rot
    new_xyz = s_inv @ transform.xyz
    _write_origin(origin, OriginTransform(xyz=new_xyz, rot=new_rot))


def _ensure_origin(joint: ET.Element) -> ET.Element:
    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin")
        origin.set("xyz", "0 0 0")
        origin.set("rpy", "0 0 0")
    return origin


def _read_origin(origin: ET.Element) -> OriginTransform:
    raw_rpy = origin.get("rpy")
    raw_xyz = origin.get("xyz")
    rpy = _parse_triple(raw_rpy) if raw_rpy else None
    xyz = _parse_triple(raw_xyz) if raw_xyz else None
    return OriginTransform.from_rpy_xyz(rpy, xyz)


def _write_origin(origin: ET.Element, transform: OriginTransform) -> None:
    rpy, xyz = transform.to_rpy_xyz()
    origin.set("rpy", format_vec(rpy))
    origin.set("xyz", format_vec(xyz))


def _parse_triple(value: str) -> list[float]:
    return [float(v) for v in value.split()]


def _child_get(element: ET.Element, tag: str, attr: str) -> str:
    child = element.find(tag)
    if child is None:
        return ""
    return child.get(attr, "") or ""
