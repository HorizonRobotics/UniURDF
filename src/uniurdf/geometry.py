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

"""Geometry helpers for URDF visual mesh world poses.

Shared by :meth:`AlignedRobot.verify` and the GUI's delta table, which must
report the same numbers for the same alignment.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from uniurdf.transform._linalg import OriginTransform


def rotation_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees between the rotation parts of two 4x4 poses.

    Uses ``arctan2(sin, cos)``, which stays accurate near zero -- where
    alignment residuals sit, and where ``arccos`` loses precision.

    Args:
        a (np.ndarray): A ``(4, 4)`` homogeneous transform.
        b (np.ndarray): A ``(4, 4)`` homogeneous transform.

    Returns:
        float: The absolute rotation between them, in degrees.
    """
    r_rel = a[:3, :3].T @ b[:3, :3]
    cos_theta = float(np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0))
    sin_theta = float(
        np.linalg.norm(
            [
                r_rel[2, 1] - r_rel[1, 2],
                r_rel[0, 2] - r_rel[2, 0],
                r_rel[1, 0] - r_rel[0, 1],
            ]
        )
        / 2.0
    )
    return float(np.degrees(np.arctan2(sin_theta, cos_theta)))


def parse_float_triple(text: str | None) -> np.ndarray:
    """Parse a URDF 3-float attribute (``"x y z"`` or ``"r p y"``).

    Returns the zero vector on missing, empty, or malformed input so that
    callers never need to special-case the common ``<origin/>`` form.

    Args:
        text (str | None): The attribute value string, or ``None``.

    Returns:
        np.ndarray: Shape ``(3,)`` float64 array; zeros on parse failure.
    """
    if not text:
        return np.zeros(3)
    try:
        parts = [float(p) for p in text.split()]
    except ValueError:
        return np.zeros(3)
    if len(parts) != 3:
        return np.zeros(3)
    return np.asarray(parts, dtype=float)


def _origin_matrix(element: ET.Element | None) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a ``<origin>`` element.

    Returns the identity matrix when ``element`` is ``None``, which is the
    correct semantics: an omitted ``<origin>`` is the same as
    ``xyz="0 0 0" rpy="0 0 0"``.

    Args:
        element (ET.Element | None): An ``<origin>`` XML element.

    Returns:
        np.ndarray: Shape ``(4, 4)`` homogeneous transform.
    """
    if element is None:
        return np.eye(4)
    # parse_float_triple tolerates malformed text where OriginTransform's own
    # parser would raise; a viewer must still draw a URDF it cannot fully read.
    xyz = parse_float_triple(element.get("xyz"))
    rpy = parse_float_triple(element.get("rpy"))
    return OriginTransform.from_rpy_xyz(tuple(rpy), tuple(xyz)).as_matrix()


def visual_world_poses(
    urdf_bytes: bytes,
    link_poses: dict[str, np.ndarray],
) -> dict[tuple[str, int], np.ndarray]:
    """World poses of every visual in a URDF.

    For each ``<link>``'s each ``<visual>``, this function computes::

        T_world_visual = T_world_link @ T_visual_origin

    This is the quantity that must be invariant under alignment: alignment
    rotates link frames but compensates the ``<visual><origin>`` so every
    visual stays exactly where it was in the world.

    Every geometry type counts -- mesh, box, cylinder, sphere -- because
    alignment rewrites each one's origin alike.

    Only ``<visual>`` geometry is considered; a paired collision mesh is
    ignored.

    Callers must pass the same URDF and the same FK poses together. Mixing
    source origins with aligned poses leaves every visual rotated by the
    joint-axis correction angle, and the result still renders, so nothing
    catches it but the eye.

    Args:
        urdf_bytes (bytes): The URDF to read visual origins from.
        link_poses (dict[str, np.ndarray]): World poses (4x4 homogeneous
            transforms) for each link, e.g. from
            :meth:`Chain.forward_kinematics`.

    Returns:
        dict[tuple[str, int], np.ndarray]: Mapping from
        ``(link_name, visual_index)`` to a ``(4, 4)`` world pose. Only
        entries whose link name appears in ``link_poses`` are included.
    """
    root = ET.fromstring(urdf_bytes)
    result: dict[tuple[str, int], np.ndarray] = {}
    for link_el in root.findall("link"):
        name = link_el.get("name")
        if not name or name not in link_poses:
            continue
        t_world_link = link_poses[name]
        visual_index = 0
        for visual in link_el.findall("visual"):
            if visual.find("geometry") is None:
                continue
            t_origin = _origin_matrix(visual.find("origin"))
            result[(name, visual_index)] = t_world_link @ t_origin
            visual_index += 1
    return result


__all__ = [
    "parse_float_triple",
    "rotation_between",
    "visual_world_poses",
]
