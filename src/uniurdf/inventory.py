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

"""What the aligned robot contains, as names and indices.

The output is deliberately neutral -- ``chains``, ``tcp``,
``camera_references``, ``finger_links`` -- rather than named for any consumer's
keys. Mapping the four groups onto whatever a config wants is a human's
decision.

Indices are positions in :attr:`Chain.joint_names`: every movable joint, mimic
followers included, in DFS preorder. That is what ``pytorch_kinematics``
returns from ``get_joints()``, which is the vector a dataset config's joint
indices address. Alignment only ever adds fixed joints -- TCP attachments and
camera-compat aliases -- and ``get_joints(exclude_fixed=True)`` drops those, so
an index taken here addresses the same joint in the source URDF.

This is deliberately not ``independent_joint_names``, which omits mimic
followers: on a robot with a mimicking gripper that would shorten the vector
and silently renumber every joint after the first follower.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from uniurdf.kinematics.chain import Chain
from uniurdf.spec import AlignmentSpec


def build_inventory(
    chain: Chain,
    spec: AlignmentSpec,
    compat_links: Sequence[str] = (),
) -> dict[str, Any]:
    """Describe an aligned robot's links and joint indices.

    Args:
        chain (Chain): The **aligned** robot's kinematic chain. Passing the
            source chain would still produce correct indices -- that is the
            point of the stability argument above -- but the emitted TCP and
            compat links would be missing.
        spec (AlignmentSpec): The spec that produced it.
        compat_links (Sequence[str], optional): Camera-compat links alignment
            actually emitted, from ``AlignmentReport.camera_compat_links``. A
            reference whose frame did not move gets none, and must then be
            targeted by its original name.

    Returns:
        dict[str, Any]: A JSON-serialisable inventory.
    """
    order = {name: i for i, name in enumerate(chain.joint_names)}
    emitted = set(compat_links)

    chains: list[dict[str, Any]] = []
    for group in spec.resolved_chains(chain):
        joints = []
        links = []
        for name in group:
            child = chain.child_link_of(name)
            joints.append(
                {
                    # index is None only for a joint the URDF does not declare
                    # movable, which validation already rejects -- so in
                    # practice it is always an int. Kept nullable rather than
                    # defaulting to 0, which would be a wrong answer.
                    "name": name,
                    "index": order.get(name),
                }
            )
            if child is not None:
                links.append(child)
        chains.append({"joints": joints, "links": links})

    # A null entry is meaningful: tcp is paired positionally with chains, so
    # the hole marks an arm that has no TCP.
    tcp: list[dict[str, Any] | None] = []
    fingers: list[dict[str, Any]] = []
    for entry in spec.tcp:
        if entry is None:
            # Keep the hole: tcp is paired positionally with chains, and
            # dropping it would silently renumber every arm after it.
            tcp.append(None)
            fingers.append({"links": []})
            continue
        tcp.append({"link": entry.link_name, "base_link": entry.base_link})
        links = []
        for joint_name in entry.joints:
            child = chain.child_link_of(joint_name)
            if child is not None and child not in links:
                links.append(child)
        fingers.append({"links": links})

    cameras = [
        {
            "link": reference.link,
            # The compat link is what a calibrated extrinsic must target when
            # alignment moved the frame; null means the frame held still and
            # the original name is still correct.
            "compat_link": (
                reference.compat_link_name
                if reference.compat_link_name in emitted
                else None
            ),
        }
        for reference in spec.camera_references
    ]

    return {
        "root_link": chain.root,
        "joint_order": list(chain.joint_names),
        "chains": chains,
        "tcp": tcp,
        "finger_links": fingers,
        "camera_references": cameras,
    }


__all__ = ["build_inventory"]
