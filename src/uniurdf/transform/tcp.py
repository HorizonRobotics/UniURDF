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

"""Attach tool-centre-point links to user-selected links.

One stage, one emitted link per TCP: ``<base_link>_tcp``, a fixed child of an
existing link at the spec's full ``xyz`` + ``rpy``.

The base link is selected, never modified: it is not renamed, and no
``_tcp_base`` link is emitted. Anything that already references it -- camera
extrinsics in particular -- keeps working untouched.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence

from uniurdf.spec import TcpSpec
from uniurdf.transform._linalg import format_vec

# Origins are compared at this tolerance when deciding whether a TCP is
# already present. Tight enough that a genuine re-spec is treated as a change,
# loose enough to absorb float round-tripping through the URDF text.
_ORIGIN_TOLERANCE = 1e-9


def insert_tcp_links(root: ET.Element, tcps: Sequence[TcpSpec]) -> list[str]:
    """Insert a fixed TCP child for each spec entry.

    Idempotent: a TCP already present with a matching origin is left alone, so
    re-aligning an aligned URDF is a no-op. A joint at the target name with a
    *different* origin is rewritten in place -- the spec is the source of
    truth, and silently keeping stale geometry would be worse than updating it.

    Args:
        root (ET.Element): The ``<robot>`` element, modified in place.
        tcps (Sequence[TcpSpec]): TCPs to attach.

    Returns:
        list[str]: Names of the joints inserted or updated, in order.

    Raises:
        ValueError: If a `base_link` does not exist in the URDF.
    """
    links_by_name = {link.get("name", ""): link for link in root.findall("link")}
    joints_by_name = {joint.get("name", ""): joint for joint in root.findall("joint")}

    changed: list[str] = []
    for tcp in tcps:
        if tcp.base_link not in links_by_name:
            raise ValueError(
                f"TCP base_link '{tcp.base_link}' is not a link in this "
                f"URDF; it must name an existing link"
            )

        existing = joints_by_name.get(tcp.joint_name)
        if existing is not None and _matches(existing, tcp):
            continue

        if tcp.link_name not in links_by_name:
            link = ET.SubElement(root, "link")
            link.set("name", tcp.link_name)
            links_by_name[tcp.link_name] = link

        if existing is not None:
            root.remove(existing)
        joint = _build_joint(tcp)
        root.append(joint)
        joints_by_name[tcp.joint_name] = joint
        changed.append(tcp.joint_name)

    return changed


def _build_joint(tcp: TcpSpec) -> ET.Element:
    """Construct the fixed joint attaching a TCP to its base link."""
    joint = ET.Element("joint")
    joint.set("name", tcp.joint_name)
    joint.set("type", "fixed")
    origin = ET.SubElement(joint, "origin")
    # format_vec is the serializer every stage shares, so emitted bytes stay
    # canonical and re-running produces no spurious diff.
    origin.set("rpy", format_vec(tcp.rpy))
    origin.set("xyz", format_vec(tcp.xyz))
    parent = ET.SubElement(joint, "parent")
    parent.set("link", tcp.base_link)
    child = ET.SubElement(joint, "child")
    child.set("link", tcp.link_name)
    return joint


def _matches(joint: ET.Element, tcp: TcpSpec) -> bool:
    """Whether `joint` is already exactly the joint `tcp` would emit."""
    if joint.get("type") != "fixed":
        return False
    parent = joint.find("parent")
    child = joint.find("child")
    if parent is None or child is None:
        return False
    if parent.get("link") != tcp.base_link:
        return False
    if child.get("link") != tcp.link_name:
        return False

    origin = joint.find("origin")
    xyz = _floats(origin.get("xyz") if origin is not None else None)
    rpy = _floats(origin.get("rpy") if origin is not None else None)
    if xyz is None or rpy is None:
        return False
    return _close(xyz, tcp.xyz) and _close(rpy, tcp.rpy)


def _floats(text: str | None) -> tuple[float, ...] | None:
    """Parse a 3-component URDF vector attribute, or ``None`` if unusable."""
    if text is None:
        return (0.0, 0.0, 0.0)
    try:
        values = tuple(float(part) for part in text.split())
    except ValueError:
        return None
    return values if len(values) == 3 else None


def _close(actual: Sequence[float], expected: Sequence[float]) -> bool:
    """Component-wise comparison at the origin tolerance."""
    return all(
        abs(a - b) <= _ORIGIN_TOLERANCE for a, b in zip(actual, expected, strict=True)
    )


__all__ = ["insert_tcp_links"]
