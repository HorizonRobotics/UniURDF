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

"""Structural diagnostics for URDF files.

:func:`inspect_urdf` examines a URDF and returns a :class:`Diagnostics` report.
It reads nothing and mutates nothing; callers decide what to do with it.

Two severity levels are defined:

* ``"fatal"`` -- the URDF cannot be meaningfully loaded and ``align`` refuses
  to proceed: a cycle, an undeclared link reference, or unparseable XML.
* ``"warning"`` -- structurally loadable, but suspicious. The GUI always opens
  regardless of warnings; a broken URDF is what you need the GUI to fix.

The fatal threshold is deliberately narrow: loading must remain possible so the
tool can be used to diagnose and repair a defective file.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from uniurdf.kinematics.chain import parse_urdf

# Axis vectors shorter than this are treated as zero-length.
_AXIS_EPS = 1e-12
# Tolerance for "is this axis unit-length?" -- well above floating-point noise
# from a round-tripped decimal but below any real misconfiguration.
_UNIT_NORM_TOL = 1e-6

# Patterns that flag unexpanded substitution syntax left in the file.
_XACRO_PATTERNS = (
    re.compile(r"\$\{[^}]+\}"),  # ${expression}
    re.compile(r"\$\(find [^)]+\)"),  # $(find package)
)


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """A single structural defect discovered in a URDF.

    Attributes:
        severity (str): ``"fatal"`` or ``"warning"``.
        code (str): Stable machine-readable identifier, e.g.
            ``"undeclared_link_reference"``.
        message (str): Human-readable description naming the offending element.
        names (tuple[str, ...]): The specific element names involved, for
            programmatic filtering.
    """

    severity: str
    code: str
    message: str
    names: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnostics:
    """All structural findings for a single URDF.

    Build with :func:`inspect_urdf`. The collection is frozen so callers can
    cache results safely.

    Attributes:
        findings (tuple[Finding, ...]): Every finding, in discovery order.
    """

    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def fatal(self) -> tuple[Finding, ...]:
        """Findings with severity ``"fatal"``."""
        return tuple(f for f in self.findings if f.severity == "fatal")

    @property
    def warnings(self) -> tuple[Finding, ...]:
        """Findings with severity ``"warning"``."""
        return tuple(f for f in self.findings if f.severity == "warning")

    @property
    def is_fatal(self) -> bool:
        """True when any finding is fatal."""
        return any(f.severity == "fatal" for f in self.findings)

    def messages(self) -> list[str]:
        """Render each finding as a single ``[SEVERITY] code: message`` line.

        Useful for CLI output and log records. The format is intentionally
        stable so callers can parse it without touching the dataclass.
        """
        return [f"[{f.severity.upper()}] {f.code}: {f.message}" for f in self.findings]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def inspect_urdf(urdf_bytes: bytes) -> Diagnostics:
    """Inspect a URDF and return all structural findings.

    This is a pure function: it reads only `urdf_bytes`, never the filesystem
    or network, and returns the same result for the same input.

    Args:
        urdf_bytes (bytes): Raw URDF content.

    Returns:
        Diagnostics: Every finding. Empty means no problems detected.
    """
    findings: list[Finding] = []

    # Parse once with comment preservation so the full document text is
    # available for xacro-pattern scanning without a second read.
    try:
        root_el = parse_urdf(urdf_bytes)
    except ET.ParseError as error:
        # A parse error is fatal on its own -- nothing else can run.
        return Diagnostics(
            findings=(
                Finding(
                    severity="fatal",
                    code="xml_parse_error",
                    message=f"XML parse error: {error}",
                    names=(),
                ),
            )
        )

    # Collect raw declarations before any filtering.
    link_names_raw = [
        el.get("name", "") for el in root_el.findall("link") if el.get("name")
    ]
    joint_els = root_el.findall("joint")

    findings.extend(_check_duplicate_links(link_names_raw))
    findings.extend(_check_duplicate_joints(joint_els))

    declared_links = set(link_names_raw)
    joint_info = _collect_joint_info(joint_els, declared_links)

    findings.extend(_check_undeclared_link_references(joint_els, declared_links))
    findings.extend(_check_conflicting_parents(joint_info))
    findings.extend(_check_multiple_roots(declared_links, joint_info))
    findings.extend(_check_orphan_links(declared_links, link_names_raw, joint_info))
    findings.extend(_check_cycles(declared_links, joint_info))
    findings.extend(_check_revolute_limits(joint_els))
    findings.extend(_check_mimic_targets(joint_els, declared_links, joint_info))
    findings.extend(_check_axis_vectors(joint_els))
    findings.extend(_check_unexpanded_xacro(urdf_bytes))

    return Diagnostics(findings=tuple(findings))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_duplicate_links(link_names: Sequence[str]) -> list[Finding]:
    """Warn on link names that appear more than once."""
    seen: set[str] = set()
    duplicated: list[str] = []
    for name in link_names:
        if name in seen and name not in duplicated:
            duplicated.append(name)
        seen.add(name)

    return [
        Finding(
            severity="warning",
            code="duplicate_link_name",
            message=f"link '{name}' is declared more than once",
            names=(name,),
        )
        for name in duplicated
    ]


def _check_duplicate_joints(joint_els: list[ET.Element]) -> list[Finding]:
    """Warn on joint names that appear more than once."""
    seen: set[str] = set()
    duplicated: list[str] = []
    for el in joint_els:
        name = el.get("name", "")
        if not name:
            continue
        if name in seen and name not in duplicated:
            duplicated.append(name)
        seen.add(name)

    return [
        Finding(
            severity="warning",
            code="duplicate_joint_name",
            message=f"joint '{name}' is declared more than once",
            names=(name,),
        )
        for name in duplicated
    ]


@dataclass(frozen=True)
class _JointEntry:
    """Lightweight record extracted from a single joint element."""

    name: str
    parent: str
    child: str


def _collect_joint_info(
    joint_els: list[ET.Element], declared: set[str]
) -> list[_JointEntry]:
    """Extract (name, parent, child) for joints with fully declared endpoints.

    Joints whose parent or child is not in `declared` are excluded here because
    the undeclared-reference check emits its own finding; subsequent graph
    checks should not also raise on the same dangling joints.
    """
    entries: list[_JointEntry] = []
    for el in joint_els:
        name = el.get("name", "")
        if not name:
            continue
        parent_el = el.find("parent")
        child_el = el.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link", "")
        child = child_el.get("link", "")
        if parent in declared and child in declared:
            entries.append(_JointEntry(name=name, parent=parent, child=child))
    return entries


def _check_undeclared_link_references(
    joint_els: list[ET.Element], declared: set[str]
) -> list[Finding]:
    """Flag joints that reference a link the URDF never declares.

    ``chain.py:_parse_joint`` silently drops such joints (line 369). This check
    makes that silent drop visible as a fatal finding, so users know they are
    losing kinematics, not just receiving a warning they can safely ignore.
    """
    findings: list[Finding] = []
    for el in joint_els:
        name = el.get("name", "")
        if not name:
            continue
        parent_el = el.find("parent")
        child_el = el.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link", "")
        child = child_el.get("link", "")
        undeclared = [
            endpoint
            for endpoint in (parent, child)
            if endpoint and endpoint not in declared
        ]
        if undeclared:
            offenders = tuple(undeclared)
            findings.append(
                Finding(
                    severity="fatal",
                    code="undeclared_link_reference",
                    message=(
                        f"joint '{name}' references undeclared link(s) "
                        f"{_quote_list(offenders)}"
                    ),
                    names=(name,) + offenders,
                )
            )
    return findings


def _check_conflicting_parents(joint_info: list[_JointEntry]) -> list[Finding]:
    """Warn on links that appear as the child of more than one joint.

    A valid URDF is a tree. A link with multiple parents makes it a DAG or a
    graph, and the link's world pose becomes undefined.
    """
    child_to_joints: dict[str, list[str]] = {}
    for entry in joint_info:
        child_to_joints.setdefault(entry.child, []).append(entry.name)

    findings: list[Finding] = []
    for child, joint_names in child_to_joints.items():
        if len(joint_names) > 1:
            findings.append(
                Finding(
                    severity="warning",
                    code="conflicting_parents",
                    message=(
                        f"link '{child}' is the child of multiple joints: "
                        f"{_quote_list(joint_names)}"
                    ),
                    names=(child,) + tuple(joint_names),
                )
            )
    return findings


def _check_multiple_roots(
    declared: set[str], joint_info: list[_JointEntry]
) -> list[Finding]:
    """Warn when more than one link has no parent joint.

    Exactly one root link is the healthy case. Multiple roots signal a
    disconnected graph -- usually a typo in a joint's parent attribute.
    """
    children = {entry.child for entry in joint_info}
    roots = [name for name in sorted(declared) if name not in children]
    if len(roots) > 1:
        return [
            Finding(
                severity="warning",
                code="multiple_roots",
                message=(
                    f"URDF has {len(roots)} root links (links that are never "
                    f"any joint's child): {_quote_list(roots)}"
                ),
                names=tuple(roots),
            )
        ]
    return []


def _check_orphan_links(
    declared: set[str],
    declared_ordered: list[str],
    joint_info: list[_JointEntry],
) -> list[Finding]:
    """Warn on links that are unreachable from the root of the joint graph.

    An orphan is declared but disconnected: it has a parent in some joint, but
    that parent is itself unreachable, so no configuration can ever place the
    orphan. This is a distinct defect from multiple-roots.
    """
    if not declared:
        return []

    children = {entry.child for entry in joint_info}
    # Candidates in declaration order, matching build_chain's root convention:
    # the first declared link that is never a joint's child.
    candidate_roots = [n for n in declared_ordered if n not in children]
    if not candidate_roots:
        # No roots means a cycle; that check handles it.
        return []

    # Pick the first declared link that is never a child as the root, matching
    # build_chain's convention.
    root = candidate_roots[0]
    reachable = _reachable_links(root, joint_info)
    orphans = sorted(n for n in declared if n not in reachable)
    if not orphans:
        return []

    return [
        Finding(
            severity="warning",
            code="orphan_link",
            message=(
                f"{len(orphans)} link(s) are declared but unreachable from "
                f"root '{root}': {_quote_list(orphans)}"
            ),
            names=tuple(orphans),
        )
    ]


def _check_cycles(declared: set[str], joint_info: list[_JointEntry]) -> list[Finding]:
    """Flag a cycle in the joint graph.

    A cycle means no link can serve as a root (every link is someone's child),
    or a subtree loops back to an ancestor. Either case prevents valid FK and
    makes the URDF fatal.
    """
    children = {entry.child for entry in joint_info}
    candidate_roots = [n for n in declared if n not in children]
    if not declared:
        # A URDF with no links has no graph to loop. `build_chain` reports the
        # empty declaration itself, which is the finding a reader can act on.
        return []
    if not candidate_roots:
        # Every link is a child -- the graph has no root, which is itself a
        # cycle indicator.
        return [
            Finding(
                severity="fatal",
                code="cycle",
                message=(
                    "URDF joint graph has no root: every link is a joint's "
                    "child, indicating a cycle"
                ),
                names=(),
            )
        ]

    # A URDF may declare several disconnected trees. Reachability is
    # computed from all of their roots together, so links belonging to one
    # tree are not counted as unreachable from another's root.
    reachable: set[str] = set()
    for candidate in candidate_roots:
        reachable |= _reachable_links(candidate, joint_info)

    # A back edge -- a joint whose child is also one of its own ancestors --
    # leaves every link reachable, so finding one takes the chain of links
    # the walk is currently inside.
    adjacency: dict[str, list[str]] = {}
    for entry in joint_info:
        adjacency.setdefault(entry.parent, []).append(entry.child)

    # `on_path` holds the links between the root and where the walk is now;
    # `finished` holds those fully explored. A joint reaching a link on the
    # active path closes a loop. A link reachable down several branches is
    # expanded once, so cost follows the graph's size.
    on_cycle: set[str] = set()
    on_path: set[str] = set()
    finished: set[str] = set()
    # Frames are (link, its children, how many of them are done).
    for candidate in candidate_roots:
        if candidate in finished:
            continue
        on_path.add(candidate)
        path: list[str] = [candidate]
        stack: list[tuple[str, list[str], int]] = [
            (candidate, list(adjacency.get(candidate, [])), 0)
        ]
        while stack:
            node, children, index = stack[-1]
            if index == len(children):
                stack.pop()
                on_path.discard(node)
                finished.add(node)
                if path and path[-1] == node:
                    path.pop()
                continue
            stack[-1] = (node, children, index + 1)
            child = children[index]
            if child in on_path:
                # The child is an ancestor of itself; everything from it down
                # the active path forms the loop.
                on_cycle.update(path[path.index(child) :])
                continue
            if child in finished:
                continue
            on_path.add(child)
            path.append(child)
            stack.append((child, list(adjacency.get(child, [])), 0))

    # A link carrying a joint but unreachable from the root sits on a
    # component that loops back on itself. Orphan links with no joint at all
    # are a separate, non-fatal finding.
    jointed = {e.parent for e in joint_info} | {e.child for e in joint_info}
    looping = sorted(on_cycle | (jointed - reachable))
    if looping:
        return [
            Finding(
                severity="fatal",
                code="cycle",
                message=(
                    "URDF joint graph does not terminate: a cycle is present "
                    f"among {_quote_list(looping)}"
                ),
                names=tuple(looping),
            )
        ]

    return []


def _check_revolute_limits(joint_els: list[ET.Element]) -> list[Finding]:
    """Warn on revolute joints whose limits are absent or inverted.

    A revolute joint without a ``<limit>`` element, or one where
    ``lower >= upper``, will produce a zero-range slider in the GUI and is
    likely a URDF authoring mistake.
    """
    findings: list[Finding] = []
    for el in joint_els:
        if el.get("type") != "revolute":
            continue
        name = el.get("name", "")
        if not name:
            continue
        limit_el = el.find("limit")
        if limit_el is None:
            findings.append(
                Finding(
                    severity="warning",
                    code="missing_revolute_limit",
                    message=(f"revolute joint '{name}' has no <limit> element"),
                    names=(name,),
                )
            )
            continue
        lower_str = limit_el.get("lower")
        upper_str = limit_el.get("upper")
        if lower_str is None or upper_str is None:
            findings.append(
                Finding(
                    severity="warning",
                    code="missing_revolute_limit",
                    message=(
                        f"revolute joint '{name}' limit element is missing "
                        f"lower or upper attribute"
                    ),
                    names=(name,),
                )
            )
            continue
        try:
            lower = float(lower_str)
            upper = float(upper_str)
        except ValueError:
            findings.append(
                Finding(
                    severity="warning",
                    code="missing_revolute_limit",
                    message=(
                        f"revolute joint '{name}' limit lower/upper are not "
                        f"parseable numbers"
                    ),
                    names=(name,),
                )
            )
            continue
        if lower >= upper:
            findings.append(
                Finding(
                    severity="warning",
                    code="inverted_revolute_limit",
                    message=(
                        f"revolute joint '{name}' has lower ({lower}) >= "
                        f"upper ({upper}): zero or negative range"
                    ),
                    names=(name,),
                )
            )
    return findings


def _check_mimic_targets(
    joint_els: list[ET.Element],
    declared_links: set[str],
    joint_info: list[_JointEntry],
) -> list[Finding]:
    """Warn on mimic joints whose referenced source joint does not exist.

    A mimic joint that points at a non-existent source is effectively
    un-drivable: the chain's fixed-point solver will silently leave it at zero.
    Named joint names come from elements that have both a name and valid
    endpoints (joint_info), so this check operates on what build_chain actually
    keeps.
    """
    # Use the raw joint names (not only those with declared endpoints) so we
    # catch a mimic reference to a joint that itself was dropped.
    all_joint_names = {el.get("name", "") for el in joint_els if el.get("name")}
    findings: list[Finding] = []
    for el in joint_els:
        mimic_el = el.find("mimic")
        if mimic_el is None:
            continue
        name = el.get("name", "")
        if not name:
            continue
        source = mimic_el.get("joint", "")
        if source and source not in all_joint_names:
            findings.append(
                Finding(
                    severity="warning",
                    code="mimic_missing_source",
                    message=(
                        f"mimic joint '{name}' references source joint "
                        f"'{source}', which is not declared in this URDF"
                    ),
                    names=(name, source),
                )
            )
    return findings


def _check_axis_vectors(joint_els: list[ET.Element]) -> list[Finding]:
    """Warn on joint axis vectors that are zero or not unit-length.

    The URDF spec requires the axis to be a unit vector. A zero axis would
    produce a divide-by-zero in the rotation calculation; a non-unit axis
    silently scales the rotation angle.  ``chain.py:_parse_axis`` normalises
    both cases away, but the source URDF is still wrong and should be fixed.
    """
    findings: list[Finding] = []
    for el in joint_els:
        name = el.get("name", "")
        if not name:
            continue
        # Only actuated joints can have a motion axis that matters.
        joint_type = el.get("type", "fixed")
        if joint_type in ("fixed", "floating"):
            continue
        axis_el = el.find("axis")
        if axis_el is None:
            # Absent axis defaults to +Z, which is unit-length and correct.
            continue
        xyz_str = axis_el.get("xyz")
        if xyz_str is None:
            continue
        try:
            values = [float(v) for v in xyz_str.split()]
        except ValueError:
            findings.append(
                Finding(
                    severity="warning",
                    code="invalid_axis_vector",
                    message=(
                        f"joint '{name}' axis xyz '{xyz_str}' cannot be "
                        f"parsed as three numbers"
                    ),
                    names=(name,),
                )
            )
            continue
        if len(values) != 3:
            findings.append(
                Finding(
                    severity="warning",
                    code="invalid_axis_vector",
                    message=(
                        f"joint '{name}' axis xyz must have 3 components, "
                        f"got {len(values)}"
                    ),
                    names=(name,),
                )
            )
            continue
        norm = float(np.linalg.norm(values))
        if norm < _AXIS_EPS:
            findings.append(
                Finding(
                    severity="warning",
                    code="zero_axis_vector",
                    message=f"joint '{name}' has a zero-length axis vector",
                    names=(name,),
                )
            )
        elif abs(norm - 1.0) > _UNIT_NORM_TOL:
            findings.append(
                Finding(
                    severity="warning",
                    code="non_unit_axis_vector",
                    message=(
                        f"joint '{name}' axis vector has norm {norm:.6f} "
                        f"(|norm-1| > {_UNIT_NORM_TOL})"
                    ),
                    names=(name,),
                )
            )
    return findings


def _check_unexpanded_xacro(urdf_bytes: bytes) -> list[Finding]:
    r"""Warn when the file text contains unexpanded xacro substitution markers.

    Patterns matched:

    * ``${...}`` -- xacro property expansion.
    * ``$(find ...)`` -- ROS package path substitution.

    An aligned file with unexpanded markers almost certainly comes from an
    upstream pipeline that forgot to run ``xacro``; the resulting URDF will
    have literal ``${...}`` in numeric attributes, which produces parse errors
    or silent zeros in link poses.
    """
    try:
        text = urdf_bytes.decode("utf-8", errors="replace")
    except Exception:
        return []

    findings: list[Finding] = []
    for pattern in _XACRO_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            # Report only the first few occurrences to avoid flooding; even
            # one unexpanded expression is actionable and the user can grep.
            preview = matches[:3]
            extra = len(matches) - len(preview)
            suffix = f" (+{extra} more)" if extra > 0 else ""
            findings.append(
                Finding(
                    severity="warning",
                    code="unexpanded_xacro",
                    message=(
                        f"file contains {len(matches)} unexpanded xacro "
                        f"expression(s): "
                        f"{_quote_list(preview)}{suffix}"
                    ),
                    names=tuple(preview),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reachable_links(root: str, joint_info: list[_JointEntry]) -> set[str]:
    """BFS/DFS from `root`, returning all reachable link names."""
    adjacency: dict[str, list[str]] = {}
    for entry in joint_info:
        adjacency.setdefault(entry.parent, []).append(entry.child)

    visited: set[str] = {root}
    stack = [root]
    while stack:
        node = stack.pop()
        for child in adjacency.get(node, []):
            if child not in visited:
                visited.add(child)
                stack.append(child)
    return visited


def _quote_list(names: Sequence[str]) -> str:
    """Render a short list of names as ``'a', 'b', 'c'`` for messages."""
    return ", ".join(f"'{n}'" for n in names)


__all__ = [
    "Diagnostics",
    "Finding",
    "inspect_urdf",
]
