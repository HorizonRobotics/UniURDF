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

"""Forward kinematics for URDF robots, in pure numpy.

Replaces ``pytorch_kinematics`` -- and with it torch -- for the small slice of
FK the alignment tooling actually needs: build a chain, list its actuated
joints, and evaluate link poses at a given configuration.

Two conventions are reproduced deliberately, because alignment output is
compared against poses computed by the library this replaces:

* **Root selection.** The URDF is treated as a single-rooted tree whose root is
  the first link, in declaration order, that is never a joint's ``<child>``.
  Links outside that tree are unreachable and are excluded from FK results.
* **Joint ordering.** Depth-first preorder, visiting each link's child joints
  in URDF declaration order. This is what ``pytorch_kinematics`` produces, and
  callers index joint vectors positionally, so any other order silently
  misassigns joint values.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from uniurdf.transform._linalg import (
    ACTUATED_JOINT_TYPES,
    OriginTransform,
)

_DEFAULT_AXIS = np.array([0.0, 0.0, 1.0])
_EPS = 1e-12


@dataclass(frozen=True)
class Joint:
    """A single URDF joint.

    Attributes:
        name (str): Joint name.
        type (str): URDF joint type (``revolute``, ``fixed``, ...).
        parent (str): Parent link name.
        child (str): Child link name.
        origin (OriginTransform): Fixed transform from parent to joint frame.
        axis (np.ndarray): Unit motion axis in the joint frame.
        limits (tuple[float, float] | None): ``(lower, upper)`` when declared.
        mimic (tuple[str, float, float] | None): ``(source, multiplier,
            offset)`` when this joint mimics another.
    """

    name: str
    type: str
    parent: str
    child: str
    origin: OriginTransform
    axis: np.ndarray
    limits: tuple[float, float] | None = None
    mimic: tuple[str, float, float] | None = None

    @property
    def is_movable(self) -> bool:
        """Whether the joint has a degree of freedom, mimic or not."""
        return self.type in ACTUATED_JOINT_TYPES

    @property
    def is_independent(self) -> bool:
        """Whether the joint moves and is independently commandable.

        Mimic joints move, but their value is derived from another joint, so
        they are not something a caller sets directly.
        """
        return self.is_movable and self.mimic is None


@dataclass(frozen=True)
class Chain:
    """A single-rooted kinematic tree ready for forward kinematics.

    Build with :func:`build_chain`. ``joint_names`` is the positional contract
    for every joint vector passed to :meth:`forward_kinematics`.

    Attributes:
        root (str): Root link name.
        joints (tuple[Joint, ...]): Every joint in the tree, DFS preorder.
        link_names (tuple[str, ...]): Reachable links, DFS preorder from root.
        orphan_links (tuple[str, ...]): Declared links outside the tree.
        duplicate_child_links (tuple[str, ...]): Links named as the child of
            more than one joint. A URDF declaring these is malformed -- it is a
            graph, not a tree -- and the pose of such a link depends on which
            joint wins. Surfaced so callers can warn rather than silently
            trusting an ambiguous frame.
    """

    root: str
    joints: tuple[Joint, ...]
    link_names: tuple[str, ...]
    orphan_links: tuple[str, ...] = ()
    duplicate_child_links: tuple[str, ...] = ()
    _by_name: dict[str, Joint] = field(repr=False, default_factory=dict)

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Movable joint names in DFS preorder.

        Matches ``pytorch_kinematics.Chain.get_joint_parameter_names()``, which
        excludes only fixed joints. Mimic joints are included: they occupy a
        slot in that library's joint vector even though their value is derived,
        and callers index positionally, so dropping them would shift every
        joint after the first mimic.

        A joint reached down several branches appears once, at its first
        position. This is the positional contract for every joint vector, and
        one name owning two slots would leave callers indexing whichever the
        last write happened to be.

        Use :attr:`independent_joint_names` for the joints a user actually
        commands -- that is the right list for GUI sliders.
        """
        return _unique(j.name for j in self.joints if j.is_movable)

    @property
    def independent_joint_names(self) -> tuple[str, ...]:
        """Movable joint names excluding mimic joints, in DFS preorder."""
        return _unique(j.name for j in self.joints if j.is_independent)

    @property
    def mimic_joint_names(self) -> tuple[str, ...]:
        """Names of joints whose value is derived from another joint."""
        return _unique(
            j.name for j in self.joints if j.is_movable and j.mimic is not None
        )

    def joint_limits(self) -> dict[str, tuple[float, float]]:
        """Return ``(lower, upper)`` per movable joint that declares limits.

        Continuous joints are unbounded and are omitted, so callers can decide
        their own display range rather than being handed a fabricated one.
        """
        return {
            j.name: j.limits
            for j in self.joints
            if j.is_movable and j.limits is not None
        }

    def resolve_joint_values(
        self, values: Mapping[str, float] | None = None
    ) -> dict[str, float]:
        """Expand a partial configuration into every movable joint's value.

        Unspecified joints default to zero, and mimic joints are derived from
        their source as ``multiplier * source + offset``. A mimic joint's
        source may itself be a mimic, and DFS order does not guarantee a source
        is visited first, so mimics are resolved to a fixed point rather than
        in a single pass.

        An explicitly supplied value for a mimic joint is overwritten by the
        derived one: the URDF says that joint is not independently commandable.

        Args:
            values (Mapping[str, float], optional): Joint values by name.
                Unknown names are ignored. Default is all zeros.

        Returns:
            dict[str, float]: A value for every movable joint.
        """
        supplied = dict(values or {})
        resolved: dict[str, float] = {
            j.name: float(supplied.get(j.name, 0.0))
            for j in self.joints
            if j.is_movable
        }
        mimics = [j for j in self.joints if j.is_movable and j.mimic is not None]
        # Iterate to a fixed point so mimic-of-mimic chains settle regardless
        # of declaration order. Bounded by the chain length, so a cyclic mimic
        # declaration terminates instead of hanging.
        for _ in range(len(mimics)):
            changed = False
            for joint in mimics:
                assert joint.mimic is not None  # guaranteed by the filter
                source, multiplier, offset = joint.mimic
                value = multiplier * resolved.get(source, 0.0) + offset
                if resolved[joint.name] != value:
                    resolved[joint.name] = value
                    changed = True
            if not changed:
                break
        return resolved

    def forward_kinematics(
        self, values: Mapping[str, float] | None = None
    ) -> dict[str, np.ndarray]:
        """Compute world-frame poses for every reachable link.

        Args:
            values (Mapping[str, float], optional): Joint values by name;
                missing joints are treated as zero. Default is the zero pose.

        Returns:
            dict[str, np.ndarray]: A ``(4, 4)`` homogeneous transform per link
            in :attr:`link_names`, expressed in the root link's frame.
        """
        resolved = self.resolve_joint_values(values)
        poses: dict[str, np.ndarray] = {self.root: np.eye(4)}
        for joint in self.joints:
            parent_pose = poses.get(joint.parent)
            if parent_pose is None:
                # Unreachable subtree; build_chain filters these out, so this
                # only guards against a hand-constructed Chain.
                continue
            motion = _joint_motion(joint, resolved.get(joint.name, 0.0))
            local = joint.origin.as_matrix() @ motion
            poses[joint.child] = parent_pose @ local
        return poses

    def find_joint(self, name: str) -> Joint | None:
        """Return the joint called `name`, or ``None`` if absent."""
        return self._by_name.get(name)

    def child_link_of(self, joint_name: str) -> str | None:
        """Return the link a joint drives, or ``None`` if the joint is absent.

        This is the link whose frame a joint controls, which is what the
        alignment spec means when it selects joints for visualisation.
        """
        joint = self._by_name.get(joint_name)
        return None if joint is None else joint.child


def build_chain(source: Path | str | bytes) -> Chain:
    """Parse a URDF into a :class:`Chain`.

    Args:
        source (Path | str | bytes): Path to a ``.urdf`` file, or raw URDF
            bytes.

    Returns:
        Chain: The single-rooted kinematic tree.

    Raises:
        ValueError: If the URDF declares no links, or every link is some
            joint's child so no root exists (a cycle).
    """
    root_el = parse_urdf(source)

    link_names = [name for el in root_el.findall("link") if (name := el.get("name"))]
    if not link_names:
        raise ValueError("URDF declares no links")
    declared = set(link_names)

    parsed: list[Joint] = []
    children_of: dict[str, list[Joint]] = {}
    child_links: set[str] = set()
    duplicated: list[str] = []
    for joint_el in root_el.findall("joint"):
        joint = _parse_joint(joint_el, declared)
        if joint is None:
            continue
        parsed.append(joint)
        children_of.setdefault(joint.parent, []).append(joint)
        if joint.child in child_links and joint.child not in duplicated:
            duplicated.append(joint.child)
        child_links.add(joint.child)

    root = next((n for n in link_names if n not in child_links), None)
    if root is None:
        raise ValueError("URDF has no root link: every link is a joint's child")

    # Iterative DFS preorder. Children are pushed reversed so they pop in
    # declaration order, which is what pytorch_kinematics' recursion produces.
    #
    # A link named as the child of several joints makes the URDF a graph rather
    # than a tree, and its pose becomes ambiguous. pytorch_kinematics rebuilds
    # the whole subtree on each visit, so the last joint reached wins for the
    # link *and everything below it*. That is reproduced here by re-descending
    # on a repeat visit; visiting only once would leave the duplicated link
    # holding one branch's pose while its children held another's. The affected
    # links are reported via ``duplicate_child_links``.
    ordered: list[Joint] = []
    reachable: list[str] = [root]
    visited_links: set[str] = {root}
    # Each pending joint carries the links between the root and itself, so a
    # joint whose child already sits on that path is a back edge and is
    # refused before it is followed. A link named by several branches is
    # re-entered once per branch, which this allows and a loop is not.
    stack: list[tuple[Joint, frozenset[str]]] = [
        (joint, frozenset((root,))) for joint in reversed(children_of.get(root, []))
    ]
    while stack:
        joint, path = stack.pop()
        if joint.child in path:
            raise ValueError(
                f"URDF joint graph has a cycle: joint '{joint.name}' makes "
                f"link '{joint.child}' its own ancestor"
            )
        if joint.child not in visited_links:
            visited_links.add(joint.child)
            reachable.append(joint.child)
        ordered.append(joint)
        child_path = path | {joint.child}
        for descendant in reversed(children_of.get(joint.child, [])):
            stack.append((descendant, child_path))

    return Chain(
        root=root,
        joints=tuple(ordered),
        link_names=tuple(reachable),
        orphan_links=tuple(n for n in link_names if n not in visited_links),
        duplicate_child_links=tuple(
            name for name in duplicated if name in visited_links
        ),
        _by_name={j.name: j for j in ordered},
    )


def parse_urdf(source: Path | str | bytes) -> ET.Element:
    """Parse a URDF into its ``<robot>`` element, preserving comments.

    ``ElementTree`` discards comments by default, which would silently strip a
    URDF's provenance notes -- calibration dates, "do not edit by hand"
    warnings -- on the first alignment pass. An explicit
    :class:`~xml.etree.ElementTree.XMLParser` with comment retention keeps
    them, so an aligned URDF stays as readable as its source.

    Args:
        source (Path | str | bytes): URDF path or raw bytes.

    Returns:
        ET.Element: The ``<robot>`` element.
    """
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
    if isinstance(source, bytes):
        return ET.fromstring(source, parser=parser)
    return ET.parse(Path(source), parser=parser).getroot()


def _parse_joint(element: ET.Element, declared: set[str]) -> Joint | None:
    """Build a :class:`Joint`, or ``None`` if it is unusable.

    A joint is skipped when it is unnamed or when either endpoint is missing or
    names a link the URDF never declares -- a dangling joint cannot contribute
    to FK.
    """
    name = element.get("name")
    parent_el = element.find("parent")
    child_el = element.find("child")
    if not name or parent_el is None or child_el is None:
        return None
    parent = parent_el.get("link", "")
    child = child_el.get("link", "")
    if parent not in declared or child not in declared:
        return None

    origin_el = element.find("origin")
    origin = OriginTransform.from_rpy_xyz(
        _floats(origin_el.get("rpy") if origin_el is not None else None),
        _floats(origin_el.get("xyz") if origin_el is not None else None),
    )

    joint_type = element.get("type", "fixed")
    axis = _parse_axis(element)
    limits = _parse_limits(element, joint_type)
    mimic = _parse_mimic(element)

    return Joint(
        name=name,
        type=joint_type,
        parent=parent,
        child=child,
        origin=origin,
        axis=axis,
        limits=limits,
        mimic=mimic,
    )


def _parse_axis(element: ET.Element) -> np.ndarray:
    """Return the joint's unit motion axis, defaulting to ``+Z`` per URDF.

    A zero-length axis appears on placeholder joints in the wild; it is treated
    as the default rather than producing a division by zero.
    """
    axis_el = element.find("axis")
    values = _floats(axis_el.get("xyz") if axis_el is not None else None)
    if values is None:
        return _DEFAULT_AXIS.copy()
    axis = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < _EPS:
        return _DEFAULT_AXIS.copy()
    return axis / norm


def _parse_limits(element: ET.Element, joint_type: str) -> tuple[float, float] | None:
    """Return ``(lower, upper)`` when the joint declares usable limits.

    Continuous joints revolve without bound, so their limits -- which URDF
    still allows to be present -- are meaningless and are dropped.
    """
    if joint_type == "continuous":
        return None
    limit_el = element.find("limit")
    if limit_el is None:
        return None
    lower = limit_el.get("lower")
    upper = limit_el.get("upper")
    if lower is None or upper is None:
        return None
    try:
        return float(lower), float(upper)
    except ValueError:
        return None


def _parse_mimic(element: ET.Element) -> tuple[str, float, float] | None:
    """Return ``(source, multiplier, offset)`` for a mimic joint."""
    mimic_el = element.find("mimic")
    if mimic_el is None:
        return None
    source = mimic_el.get("joint")
    if not source:
        return None
    return (
        source,
        _float_or(mimic_el.get("multiplier"), 1.0),
        _float_or(mimic_el.get("offset"), 0.0),
    )


def _joint_motion(joint: Joint, value: float) -> np.ndarray:
    """Return the ``(4, 4)`` transform a joint contributes at `value`."""
    if joint.type == "prismatic":
        matrix = np.eye(4)
        matrix[:3, 3] = joint.axis * value
        return matrix
    if joint.type in ("revolute", "continuous"):
        matrix = np.eye(4)
        matrix[:3, :3] = _rotation_about(joint.axis, value)
        return matrix
    return np.eye(4)


def _rotation_about(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' rotation of `theta` radians about a unit `axis`."""
    x, y, z = (float(v) for v in axis)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _floats(text: str | None) -> list[float] | None:
    """Parse a whitespace-separated URDF vector attribute."""
    if text is None:
        return None
    try:
        return [float(part) for part in text.split()]
    except ValueError:
        return None


def _float_or(text: str | None, default: float) -> float:
    """Parse a float attribute, falling back to `default`."""
    if text is None:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _unique(names: Iterable[str]) -> tuple[str, ...]:
    """Names in first-seen order, each appearing once."""
    return tuple(dict.fromkeys(names))


def sweep_configuration(
    joint_names: Sequence[str] | Iterable[str],
    scale: float = 0.08,
) -> dict[str, float]:
    """Build a deterministic non-zero pose for diagnostics.

    Values alternate in sign and grow with position so that every joint moves
    by a distinguishable amount -- a zero pose hides joints whose frames only
    differ once rotated.

    Args:
        joint_names (Sequence[str] | Iterable[str]): Joints to include, in the
            order they should be swept.
        scale (float, optional): Base magnitude per step. Default is ``0.08``.

    Returns:
        dict[str, float]: Joint values by name.
    """
    return {
        name: scale * (index + 1) * (1 if index % 2 == 0 else -1)
        for index, name in enumerate(joint_names)
    }


__all__ = [
    "ACTUATED_JOINT_TYPES",
    "Chain",
    "Joint",
    "build_chain",
    "sweep_configuration",
]
