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

"""Cut an aligned URDF down to the links the spec actually asked about.

Kept: the links the spec names -- every chain joint's child, every TCP base and
its declared gripper joints, every camera reference -- plus the TCP and
camera-compat links alignment emitted, plus every ancestor of those up to the
root. A link hanging below a named one is dropped unless it was itself named,
so wrist-mounted sensor frames go too; a frame you still need is one the spec
should name under ``camera_references``.

Pruning renumbers the joint vector. Indices taken against the full URDF address
the wrong joints here and the robot still moves, so a pruned URDF never travels
without its own inventory, written beside it under a matching name.

FK over the kept links is exact: pruning removes whole subtrees and never
rewrites an origin, so every surviving link's pose is computed from the
identical chain of transforms. :func:`prune_urdf` re-checks this on every call
rather than trusting the argument.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from uniurdf.kinematics.chain import Chain, build_chain, parse_urdf
from uniurdf.spec import AlignmentSpec

# Pruning must not perturb geometry at all -- it deletes elements and rewrites
# nothing -- so this is a guard against a coding error, not a tolerance for
# accumulated float error. Anything non-zero here is a bug.
PRUNE_ATOL = 1e-12


class PruneError(RuntimeError):
    """Pruning would have changed the robot's geometry.

    Raised only if the self-check in :func:`prune_urdf` fails, which means the
    keep-set logic dropped something load-bearing. Never expected in practice;
    it exists so the failure is loud rather than a subtly wrong URDF.
    """


@dataclass(frozen=True)
class PruneReport:
    """What pruning removed.

    Attributes:
        urdf_bytes (bytes): The pruned URDF, serialized.
        kept_links (tuple[str, ...]): Links that survived, in the source
            document's order.
        dropped_links (tuple[str, ...]): Links removed.
        dropped_joints (tuple[str, ...]): Joints removed, fixed ones included.
        dropped_movable_joints (tuple[str, ...]): The subset of
            `dropped_joints` that carried a degree of freedom. This is what
            renumbers the joint vector, so it is reported separately.
        max_position_error (float): Largest world-position difference across
            the kept links, in metres. Expected to be exactly zero.
    """

    urdf_bytes: bytes
    kept_links: tuple[str, ...] = ()
    dropped_links: tuple[str, ...] = ()
    dropped_joints: tuple[str, ...] = ()
    dropped_movable_joints: tuple[str, ...] = ()
    max_position_error: float = 0.0

    @property
    def renumbers_joint_vector(self) -> bool:
        """Whether a full-URDF joint index is still valid against this file."""
        return bool(self.dropped_movable_joints)


def focus_links(
    chain: Chain,
    spec: AlignmentSpec,
    extra_links: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return the links the spec asks about, before ancestor closure.

    These are the links a reader of the spec cares about: the chains' moving
    frames, the TCPs and the gripper joints they drive, the camera frames
    that must survive, and any joint given a manual roll.

    Names come back in the order the spec mentions them, first mention
    winning, so a caller rendering labels gets a stable list.

    Args:
        chain (Chain): The chain the spec was validated against.
        spec (AlignmentSpec): The alignment declaration.
        extra_links (Iterable[str], optional): Additional links to seed with,
            such as the TCP and compat links alignment emitted. These hang
            *below* their base link, so ancestor closure alone would drop
            them -- the tool would delete its own output.

    Returns:
        tuple[str, ...]: Seed links in declaration order. Names absent from
        the chain are skipped, so a caller may pass emitted links that a
        given run did not produce.
    """
    known = set(chain.link_names)
    seeds: dict[str, None] = {}

    def add(name: str | None) -> None:
        if name and name in known:
            seeds.setdefault(name, None)

    for group in spec.resolved_chains(chain):
        for joint_name in group:
            add(chain.child_link_of(joint_name))

    for entry in spec.tcp:
        if entry is None:
            continue
        add(entry.base_link)
        for joint_name in entry.joints:
            add(chain.child_link_of(joint_name))

    for reference in spec.camera_references:
        add(reference.link)

    # A joint given a manual rotation is one the user is thinking about, and
    # its frame convention only means something if the frame survives.
    for joint_name, _ in spec.rot_z:
        add(chain.child_link_of(joint_name))

    for name in extra_links:
        add(name)

    return tuple(seeds)


def keep_set(
    chain: Chain,
    spec: AlignmentSpec,
    extra_links: Iterable[str] = (),
) -> set[str]:
    """Return every link the pruned URDF must retain.

    The seeds from :func:`focus_links`, closed upward over parents to the root.
    An intermediate link is kept even when nothing references it, because
    deleting it would detach everything below from the root.

    Args:
        chain (Chain): The chain to prune.
        spec (AlignmentSpec): The alignment declaration.
        extra_links (Iterable[str], optional): Extra seeds; see
            :func:`focus_links`.

    Returns:
        set[str]: Link names to keep, always including the root.
    """
    parent_of = {joint.child: joint.parent for joint in chain.joints}
    keep = {chain.root}
    for link in focus_links(chain, spec, extra_links):
        # Walk to the root, stopping early once the path joins one already
        # kept. `parent_of` comes from a tree, so this terminates.
        current: str | None = link
        while current is not None and current not in keep:
            keep.add(current)
            current = parent_of.get(current)
    return keep


def prune_urdf(
    urdf_bytes: bytes,
    spec: AlignmentSpec,
    extra_links: Iterable[str] = (),
    chain: Chain | None = None,
) -> PruneReport:
    """Remove every link outside the spec's focus, and everything holding it.

    Args:
        urdf_bytes (bytes): The URDF to prune -- normally an aligned one.
        spec (AlignmentSpec): The alignment declaration that scopes the cut.
        extra_links (Iterable[str], optional): Links to keep beyond what the
            spec names, typically alignment's emitted TCP and compat links.
        chain (Chain, optional): The chain for `urdf_bytes`, when the caller
            already built one. Rebuilt from the bytes when omitted.

    Returns:
        PruneReport: The pruned URDF and what it dropped.

    Raises:
        PruneError: If any kept link's world pose moved, which would mean the
            keep-set dropped something the geometry depended on.
    """
    source_chain = chain if chain is not None else build_chain(urdf_bytes)
    keep = keep_set(source_chain, spec, extra_links)

    root = parse_urdf(urdf_bytes)
    dropped_links: list[str] = []
    dropped_joints: list[str] = []
    dropped_movable: list[str] = []

    # Iterate over a snapshot: removing from an element while iterating it
    # skips entries.
    for element in list(root):
        tag = element.tag
        if tag == "link":
            name = element.get("name")
            if name is not None and name not in keep:
                dropped_links.append(name)
                root.remove(element)
        elif tag == "joint":
            child = element.find("child")
            parent = element.find("parent")
            child_link = child.get("link") if child is not None else None
            parent_link = parent.get("link") if parent is not None else None
            # Both endpoints must survive. Testing the child alone is not
            # enough: URDF allows a link to be named as the child of two
            # joints, so it can be kept via one parent while the other
            # parent is dropped, leaving a joint that
            # references a link no longer declared. Parsers respond to that by
            # silently discarding the joint, which is how a robot quietly
            # loses an arm.
            drops_child = child_link is not None and child_link not in keep
            drops_parent = parent_link is not None and parent_link not in keep
            if drops_child or drops_parent:
                name = element.get("name") or child_link or ""
                dropped_joints.append(name)
                joint = source_chain.find_joint(name)
                if joint is not None and joint.is_movable:
                    dropped_movable.append(name)
                root.remove(element)
        elif tag == "loop_joint":
            # A closed-chain constraint naming a link that is gone would leave
            # a dangling reference. These only ever appear inside gripper
            # linkages, so they go with the subtree they constrain.
            bodies = [
                sub.get("link") for sub in element if sub.tag in ("link1", "link2")
            ]
            if any(body is not None and body not in keep for body in bodies):
                root.remove(element)
        # Everything else -- top-level <material> definitions that links
        # reference by name, comments, and <gazebo> blocks including the
        # provenance record -- is left alone. Dropping a shared material would
        # break links that survived, and the record has to keep riding along
        # or a re-align of this file would double-apply rot_z.

    pruned_bytes = _serialize(root)
    kept_links = tuple(
        name
        for name in source_chain.link_names
        if name in keep and name not in dropped_links
    )

    error = _verify_kept_links(source_chain, pruned_bytes)
    if error > PRUNE_ATOL:
        raise PruneError(
            f"pruning moved a kept link by {error:.3e} m, which must not "
            f"happen -- pruning deletes elements and rewrites none. This is a "
            f"bug in the keep-set, not a tolerance to widen."
        )

    return PruneReport(
        urdf_bytes=pruned_bytes,
        kept_links=kept_links,
        dropped_links=tuple(dropped_links),
        dropped_joints=tuple(dropped_joints),
        dropped_movable_joints=tuple(dropped_movable),
        max_position_error=error,
    )


def _verify_kept_links(source_chain: Chain, pruned_bytes: bytes) -> float:
    """Largest world-position drift across links both URDFs share.

    Checked at the zero pose and two opposing sweeps, matching what
    :meth:`AlignedRobot.verify` does -- a zero pose alone would hide an error
    in a joint-motion term, where every joint contributes identity.

    Values are addressed by joint *name*, so the renumbering that pruning
    causes cannot make this compare two different configurations.
    """
    pruned_chain = build_chain(pruned_bytes)
    independent = pruned_chain.independent_joint_names
    configurations: list[dict[str, float]] = [
        {},
        {name: 0.11 * (i % 5 - 2) for i, name in enumerate(independent)},
        {name: -0.17 * (i % 3 - 1) for i, name in enumerate(independent)},
    ]

    worst = 0.0
    for values in configurations:
        source_poses = source_chain.forward_kinematics(values)
        pruned_poses = pruned_chain.forward_kinematics(values)
        for link, pose in pruned_poses.items():
            reference = source_poses.get(link)
            if reference is None:
                continue
            drift = float(np.abs(pose[:3, 3] - reference[:3, 3]).max())
            worst = max(worst, drift)
    return worst


def _serialize(root: ET.Element) -> bytes:
    """Serialize with the same indent rules the aligned URDF uses.

    Removing elements leaves the surrounding whitespace behind, so without
    re-indenting the pruned file would carry the gaps where subtrees used to
    be.
    """
    for element in root.iter():
        if element.text is not None and not element.text.strip():
            element.text = None
        if element.tail is not None and not element.tail.strip():
            element.tail = None
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="utf-8")
    if not body.endswith(b"\n"):
        body += b"\n"
    return body


__all__ = [
    "PRUNE_ATOL",
    "PruneError",
    "PruneReport",
    "focus_links",
    "keep_set",
    "prune_urdf",
]
