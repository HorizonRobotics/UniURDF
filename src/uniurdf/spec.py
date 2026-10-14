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

"""The alignment spec: what a user declares, and nothing more.

Everything the tool does must be explainable from a spec. No robot, dataset, or
embodiment knowledge lives in this package -- if the output depends on it, it
is declared here.

A spec is YAML:

.. code-block:: yaml

    # Absolute path to the URDF being aligned.
    source_urdf: /path/to/robots/arx5/arx5_description.urdf

    # Base directory that relative mesh references resolve against.
    # Optional; defaults to source_urdf.parent.
    # package://<pkg>/<rel> resolves to <mesh_root>/<pkg>/<rel>.
    mesh_root: /path/to/robots/arx5

    # Where to write the aligned URDF and copied meshes.
    # Optional; defaults to workspace/aligned_urdf/<source stem> beside
    # this file. Relative paths anchor to this file's directory.
    output: ../workspace/aligned_urdf/arx5

    # One entry per kinematic chain (arm). Optional; defaults to a single
    # chain holding every movable joint. Alignment normalises every actuated
    # joint whatever this says, and verification checks every link the two
    # URDFs share. What this selects is what the GUI draws, and which links
    # the pruned URDF keeps -- a joint left out here is dropped from
    # `<stem>.pruned.urdf` along with the links below it.
    chains:
      - [fl_joint1, fl_joint2, fl_joint3, fl_joint4, fl_joint5, fl_joint6]
      - [fr_joint1, fr_joint2, fr_joint3, fr_joint4, fr_joint5, fr_joint6]

    # tcp[i] belongs to chains[i]. Use `null` to give a chain no TCP.
    # `base_link` selects a link that already exists; it is never renamed.
    # One new link `<base_link>_tcp` is added as its fixed child.
    # `joints` lists the gripper joints that hang off this TCP; they are shown
    # as additional sliders in the right panel and their child link triads are
    # drawn in the viewer. Like `chains`, they do not change `<stem>.urdf` --
    # alignment normalises every actuated joint regardless -- but they do keep
    # their links in `<stem>.pruned.urdf`.
    tcp:
      - base_link: fl_link6
        xyz: [0.0, 0.0, 0.155]   # metres
        rpy: [0.0, 0.0, 270.0]   # DEGREES
        joints: [fl_gripper_left_joint, fl_gripper_right_joint]
      - base_link: fr_link6
        xyz: [0.0, 0.0, 0.155]
        rpy: [0.0, 0.0, 270.0]
        joints: [fr_gripper_left_joint, fr_gripper_right_joint]

    # (optional) Extra rotation about each joint's own +Z, in DEGREES.
    # Alignment pins Z to the motion axis, which leaves the roll about Z
    # unconstrained; this is how you choose it. Sparse -- an absent joint
    # means 0. Any movable joint may appear, including gripper joints.
    rot_z:
      fl_joint6: 270.0

    # Links whose calibrated extrinsics must survive axis normalisation.
    # Where the compat joint lands in the document is derived, not declared:
    # ahead of the link's TCP joint when it has one, else appended.
    camera_references:
      - link: fl_link6

Angles are **degrees** in YAML and radians in :class:`TcpSpec`; the conversion
happens once, at parse time.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from uniurdf.kinematics.chain import Chain

TCP_LINK_SUFFIX = "_tcp"


class SpecError(ValueError):
    """A spec is malformed, or contradicts the URDF it targets.

    Carries a message aimed at whoever wrote the YAML: what was wrong, where,
    and -- where there is a plausible fix -- what to write instead.
    """


@dataclass(frozen=True)
class TcpSpec:
    """A tool centre point: a fixed child of an existing link.

    Attributes:
        base_link (str): An existing link, selected by the user. Never renamed
            and never emitted -- only referenced.
        xyz (tuple[float, float, float]): Offset from `base_link`, in metres.
        rpy (tuple[float, float, float]): Orientation relative to `base_link`,
            in **radians** (the YAML declares degrees).
        joints (tuple[str, ...]): Gripper joints that belong to this TCP's
            end-effector. They give the GUI its sliders and triads, and they
            keep their links in the pruned URDF. Alignment normalises every
            actuated joint regardless of this list, so ``<stem>.urdf`` is the
            same either way.
    """

    base_link: str
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joints: tuple[str, ...] = ()

    @property
    def link_name(self) -> str:
        """Name of the link this TCP emits."""
        return f"{self.base_link}{TCP_LINK_SUFFIX}"

    @property
    def joint_name(self) -> str:
        """Name of the fixed joint attaching the TCP to its base link."""
        return f"{self.base_link}{TCP_LINK_SUFFIX}_joint"


@dataclass(frozen=True)
class CameraReference:
    """A link whose world pose must survive axis normalization.

    Attributes:
        link (str): The existing link a calibrated extrinsic targets.
    """

    link: str

    @property
    def compat_link_name(self) -> str:
        """Name of the compatibility link emitted when the frame moves."""
        return f"{self.link}_camera_mount_compat"


@dataclass(frozen=True)
class AlignmentSpec:
    """A complete, validated alignment declaration.

    Attributes:
        source_urdf (Path | None): URDF this spec targets, when the spec names
            one. A caller may supply the URDF separately instead.
        mesh_root (Path | None): Base directory that relative mesh references
            resolve against. ``package://<pkg>/<rel>`` resolves to
            ``mesh_root / <pkg> / <rel>``. When ``None``, defaults to
            ``source_urdf.parent``; use :attr:`resolved_mesh_root` to get the
            effective value.
        output (Path | None): Where to write the aligned URDF and copied
            meshes, relative to the spec file's directory when not absolute.
            When ``None``, defaults to
            ``workspace/aligned_urdf/<source_urdf.stem>`` beside the spec.
            :func:`~uniurdf.project.load_project` resolves it.
        chains (tuple[tuple[str, ...], ...]): Joint names per kinematic chain.
            Empty means "every movable joint, as one chain", resolved against a
            URDF by :meth:`resolved_chains`.
        tcp (tuple[TcpSpec | None, ...]): Positionally paired with `chains`.
        camera_references (tuple[CameraReference, ...]): Frames to preserve.
        rot_z (tuple[tuple[str, float], ...]): Extra rotation about each named
            joint's own ``+Z``, in **radians** (the YAML declares degrees).
            Sparse: a joint absent from this list gets no rotation. Stored as
            pairs rather than a dict so the spec stays hashable.
    """

    source_urdf: Path | None = None
    mesh_root: Path | None = None
    output: Path | None = None
    chains: tuple[tuple[str, ...], ...] = ()
    tcp: tuple[TcpSpec | None, ...] = ()
    camera_references: tuple[CameraReference, ...] = ()
    rot_z: tuple[tuple[str, float], ...] = ()

    @property
    def resolved_mesh_root(self) -> Path | None:
        """Effective mesh root: explicit value, or source_urdf.parent.

        Returns ``None`` only when neither ``mesh_root`` nor ``source_urdf``
        is set (i.e. the spec is anonymous -- the caller supplies the URDF
        separately and must also supply the mesh root).
        """
        if self.mesh_root is not None:
            return self.mesh_root
        if self.source_urdf is not None:
            return self.source_urdf.parent
        return None

    @property
    def tcps(self) -> tuple[TcpSpec, ...]:
        """Every declared TCP, skipping chains that declare none."""
        return tuple(entry for entry in self.tcp if entry is not None)

    def resolved_chains(self, chain: Chain) -> tuple[tuple[str, ...], ...]:
        """Return the chains, defaulting to every movable joint.

        Args:
            chain (Chain): The kinematic chain of the target URDF, used to
                supply the default.

        Returns:
            tuple[tuple[str, ...], ...]: One tuple of joint names per chain.
        """
        if self.chains:
            return self.chains
        return (tuple(chain.independent_joint_names),)

    def validate_against(self, chain: Chain) -> None:
        """Check the spec is coherent with the URDF it targets.

        Structural validation happens at parse time; this catches the errors
        that need the URDF -- names that do not exist, joints that are fixed,
        and emitted link names that would collide with existing ones.

        Args:
            chain (Chain): The kinematic chain built from the target URDF.

        Raises:
            SpecError: On the first problem found, naming the offending entry.
        """
        joint_names = set(chain.joint_names)
        orphan_links = set(chain.orphan_links)
        link_names = set(chain.link_names) | orphan_links

        seen: dict[str, int] = {}
        for index, joints in enumerate(self.chains):
            for name in joints:
                if name not in joint_names:
                    raise SpecError(
                        f"chains[{index}] names joint '{name}', which the "
                        f"URDF does not declare as movable. Movable joints "
                        f"are: {_preview(sorted(joint_names))}"
                    )
                if name in seen:
                    raise SpecError(
                        f"chains[{index}] repeats joint '{name}', already "
                        f"claimed by chains[{seen[name]}]. Each joint belongs "
                        f"to exactly one chain."
                    )
                seen[name] = index

        for index, entry in enumerate(self.tcp):
            if entry is None:
                continue
            if entry.base_link not in link_names:
                raise SpecError(
                    f"tcp[{index}].base_link '{entry.base_link}' is not a "
                    f"link in the URDF. It must name an existing link; the "
                    f"tool adds '{entry.link_name}' as its child."
                )
            if entry.base_link in orphan_links:
                # Forward kinematics reaches only links below the root, so a
                # TCP here is never placed: it contributes no comparison and
                # leaves verification with nothing to check but a clean
                # verdict to report.
                raise SpecError(
                    f"tcp[{index}].base_link '{entry.base_link}' is declared "
                    f"but unreachable from root '{chain.root}', so its pose "
                    f"is undefined and the TCP would never be placed. Join "
                    f"'{entry.base_link}' to the robot, or point base_link at "
                    f"a link that already is."
                )
            if entry.link_name in link_names and not _is_own_tcp_link(chain, entry):
                raise SpecError(
                    f"tcp[{index}] would emit link '{entry.link_name}', but "
                    f"the URDF already declares an unrelated link with that "
                    f"name. Rename the existing link, or point base_link "
                    f"elsewhere."
                )
            # Validate tcp.joints: each named joint must be movable and must
            # not already appear in a chain entry. A joint driven from two
            # places (a chain and a tcp.joints list) would be ambiguous about
            # which group "owns" it, so we reject overlap for the same reason
            # chains themselves may not share a joint.
            for joint_name in entry.joints:
                if joint_name not in joint_names:
                    raise SpecError(
                        f"tcp[{index}].joints names joint '{joint_name}', "
                        f"which the URDF does not declare as movable. Movable "
                        f"joints are: {_preview(sorted(joint_names))}"
                    )
                if joint_name in seen:
                    raise SpecError(
                        f"tcp[{index}].joints names joint '{joint_name}', "
                        f"already claimed by chains[{seen[joint_name]}]. "
                        f"Each joint belongs to exactly one chain or one "
                        f"tcp.joints list."
                    )

        for index, reference in enumerate(self.camera_references):
            if reference.link not in link_names:
                raise SpecError(
                    f"camera_references[{index}].link '{reference.link}' is "
                    f"not a link in the URDF."
                )

        # rot_z may name any movable joint, including mimic followers and
        # gripper joints that belong to no chain. Rotating a follower's frame
        # is an ordinary geometric change, and FK still recomputes its value
        # from its reference.
        for name, _ in self.rot_z:
            if name not in joint_names:
                raise SpecError(
                    f"rot_z names joint '{name}', which the URDF does not "
                    f"declare as movable. Movable joints are: "
                    f"{_preview(sorted(joint_names))}"
                )


def load_spec(path: Path | str) -> AlignmentSpec:
    """Read and validate a spec file.

    Args:
        path (Path | str): Path to the YAML spec.

    Returns:
        AlignmentSpec: The parsed spec, structurally validated. Call
        :meth:`AlignmentSpec.validate_against` to check it against a URDF.

    Raises:
        SpecError: If the file is unreadable or malformed.
    """
    spec_path = Path(path)
    try:
        raw = yaml.safe_load(spec_path.read_text())
    except FileNotFoundError as error:
        raise SpecError(f"spec file not found: {spec_path}") from error
    except IsADirectoryError as error:
        raise SpecError(f"spec path is a directory, not a file: {spec_path}") from error
    except UnicodeDecodeError as error:
        raise SpecError(f"{spec_path} is not text; a spec is a YAML file") from error
    except OSError as error:
        raise SpecError(f"could not read {spec_path}: {error}") from error
    except yaml.YAMLError as error:
        raise SpecError(f"{spec_path} is not valid YAML: {error}") from error

    spec = parse_spec(raw or {}, origin=str(spec_path))
    source = spec.source_urdf
    if source is not None and not source.is_absolute():
        # Relative paths are anchored to the spec file, so a spec and its
        # robot can be moved together.
        source = (spec_path.parent / source).resolve()
    mesh_root = spec.mesh_root
    if mesh_root is not None and not mesh_root.is_absolute():
        # mesh_root is also spec-relative so the pair can be moved together.
        mesh_root = (spec_path.parent / mesh_root).resolve()
    # output stays verbatim: load_project resolves it, because the --output
    # argument may override it before any directory is chosen.
    return AlignmentSpec(
        source_urdf=source,
        mesh_root=mesh_root,
        output=spec.output,
        chains=spec.chains,
        tcp=spec.tcp,
        camera_references=spec.camera_references,
        rot_z=spec.rot_z,
    )


def parse_spec(data: Mapping[str, Any], origin: str = "<spec>") -> AlignmentSpec:
    """Build an :class:`AlignmentSpec` from a plain mapping.

    Args:
        data (Mapping[str, Any]): Decoded YAML.
        origin (str, optional): Label used in error messages. Default is
            ``"<spec>"``.

    Returns:
        AlignmentSpec: The parsed spec.

    Raises:
        SpecError: If a field is missing, mistyped, or unrecognised.
    """
    if not isinstance(data, Mapping):
        raise SpecError(f"{origin}: expected a mapping at the top level")

    known = {
        "source_urdf",
        "mesh_root",
        "output",
        "chains",
        "tcp",
        "camera_references",
        "rot_z",
    }
    unknown = sorted(set(data) - known)
    if unknown:
        # Silently ignoring a typo would produce a valid-looking but wrong
        # URDF, which is far worse than failing here.
        raise SpecError(
            f"{origin}: unknown field(s) {unknown}. Valid fields are: {sorted(known)}"
        )

    source_urdf = data.get("source_urdf")
    if source_urdf is not None and not isinstance(source_urdf, str):
        raise SpecError(f"{origin}: source_urdf must be a string path")

    mesh_root = data.get("mesh_root")
    if mesh_root is not None and not isinstance(mesh_root, str):
        raise SpecError(f"{origin}: mesh_root must be a string path")

    output = data.get("output")
    if output is not None and not isinstance(output, str):
        raise SpecError(f"{origin}: output must be a string path")

    chains = _parse_chains(data.get("chains"), origin)
    tcp = _parse_tcp(data.get("tcp"), origin)
    if chains and tcp and len(tcp) != len(chains):
        raise SpecError(
            f"{origin}: tcp has {len(tcp)} entries but chains has "
            f"{len(chains)}. They are paired positionally -- use `null` for a "
            f"chain with no TCP."
        )

    return AlignmentSpec(
        source_urdf=Path(source_urdf) if source_urdf else None,
        mesh_root=Path(mesh_root) if mesh_root else None,
        output=Path(output) if output else None,
        chains=chains,
        tcp=tcp,
        camera_references=_parse_camera_references(
            data.get("camera_references"), origin
        ),
        rot_z=_parse_rot_z(data.get("rot_z"), origin),
    )


def _parse_rot_z(value: Any, origin: str) -> tuple[tuple[str, float], ...]:
    """Parse ``rot_z`` as a joint-name -> degrees mapping.

    Returned as pairs in **radians**, sorted by joint name so a spec's
    identity does not depend on YAML key order.
    """
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise SpecError(
            f"{origin}: rot_z must be a mapping of joint name to degrees, "
            f"e.g. {{fl_joint6: 270.0}}"
        )
    out: list[tuple[str, float]] = []
    for name, degrees in value.items():
        if not isinstance(name, str) or not name:
            raise SpecError(f"{origin}: rot_z keys must be joint names")
        # bool is an int subclass, and `true` here is always a mistake.
        if isinstance(degrees, bool) or not isinstance(degrees, (int, float)):
            raise SpecError(f"{origin}: rot_z['{name}'] must be a number of degrees")
        out.append((name, math.radians(float(degrees))))
    return tuple(sorted(out))


def _parse_chains(value: Any, origin: str) -> tuple[tuple[str, ...], ...]:
    """Parse ``chains`` as a list of joint-name lists."""
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpecError(
            f"{origin}: chains must be a list of joint-name lists, one per "
            f"arm, e.g. [[j1, j2], [k1, k2]]"
        )
    chains: list[tuple[str, ...]] = []
    for index, entry in enumerate(value):
        if isinstance(entry, str) or not isinstance(entry, Sequence):
            raise SpecError(
                f"{origin}: chains[{index}] must be a list of joint names. "
                f"A single arm is still written as a list of lists."
            )
        names = []
        for position, name in enumerate(entry):
            if not isinstance(name, str) or not name:
                raise SpecError(
                    f"{origin}: chains[{index}][{position}] must be a "
                    f"non-empty joint name"
                )
            names.append(name)
        chains.append(tuple(names))
    return tuple(chains)


def _parse_tcp(value: Any, origin: str) -> tuple[TcpSpec | None, ...]:
    """Parse ``tcp`` as a positional list of optional TCP declarations."""
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpecError(f"{origin}: tcp must be a list")
    entries: list[TcpSpec | None] = []
    for index, entry in enumerate(value):
        if entry is None:
            entries.append(None)
            continue
        label = f"{origin}: tcp[{index}]"
        if not isinstance(entry, Mapping):
            raise SpecError(f"{label} must be a mapping or null")
        known = {"base_link", "xyz", "rpy", "joints"}
        unknown = sorted(set(entry) - known)
        if unknown:
            raise SpecError(
                f"{label} has unknown field(s) {unknown}. "
                f"Valid fields are: {sorted(known)}"
            )
        base_link = entry.get("base_link")
        if not isinstance(base_link, str) or not base_link:
            raise SpecError(
                f"{label} requires `base_link`: the existing link the TCP attaches to"
            )
        entries.append(
            TcpSpec(
                base_link=base_link,
                xyz=_vec3(entry.get("xyz"), f"{label}.xyz"),
                rpy=_radians3(_vec3(entry.get("rpy"), f"{label}.rpy")),
                joints=_joint_list(entry.get("joints"), label),
            )
        )
    return tuple(entries)


def _parse_camera_references(value: Any, origin: str) -> tuple[CameraReference, ...]:
    """Parse ``camera_references``, accepting bare link names as shorthand."""
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise SpecError(f"{origin}: camera_references must be a list")
    references: list[CameraReference] = []
    for index, entry in enumerate(value):
        label = f"{origin}: camera_references[{index}]"
        if isinstance(entry, str):
            if not entry:
                raise SpecError(f"{label} must be a non-empty link name")
            references.append(CameraReference(link=entry))
            continue
        if not isinstance(entry, Mapping):
            raise SpecError(f"{label} must be a link name or a mapping")
        known = {"link"}
        unknown = sorted(set(entry) - known)
        if unknown:
            raise SpecError(
                f"{label} has unknown field(s) {unknown}. "
                f"Valid fields are: {sorted(known)}"
            )
        link = entry.get("link")
        if not isinstance(link, str) or not link:
            raise SpecError(f"{label} requires `link`")
        references.append(CameraReference(link=link))
    return tuple(references)


def _joint_list(value: Any, label: str) -> tuple[str, ...]:
    """Parse an optional list of joint names, defaulting to empty."""
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpecError(f"{label}.joints must be a list of joint names")
    out = []
    for position, name in enumerate(value):
        if not isinstance(name, str) or not name:
            raise SpecError(
                f"{label}.joints[{position}] must be a non-empty joint name"
            )
        out.append(name)
    return tuple(out)


def _radians3(degrees: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert a 3-vector from degrees to radians, keeping its length known."""
    return (
        math.radians(degrees[0]),
        math.radians(degrees[1]),
        math.radians(degrees[2]),
    )


def _vec3(value: Any, label: str) -> tuple[float, float, float]:
    """Parse an optional 3-vector, defaulting to zeros."""
    if value is None:
        return (0.0, 0.0, 0.0)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise SpecError(f"{label} must be a list of 3 numbers")
    if len(value) != 3:
        raise SpecError(f"{label} must have exactly 3 entries, got {len(value)}")
    out = []
    for component in value:
        # bool is an int subclass, and `true` here is always a mistake.
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise SpecError(f"{label} must contain numbers")
        out.append(float(component))
    return (out[0], out[1], out[2])


def _is_own_tcp_link(chain: Chain, tcp: TcpSpec) -> bool:
    """Whether an existing link is this spec's own TCP from an earlier run.

    Re-aligning an already-aligned URDF must be possible -- it is how the tool
    stays idempotent -- so a link that this very TCP would emit is not a
    collision. The test is structural rather than name-based: the link must
    hang off the declared base link by a fixed joint, which is exactly what a
    previous run would have produced. A same-named link attached anywhere else
    is a genuine clash and still raises.
    """
    joint = chain.find_joint(tcp.joint_name)
    if joint is None:
        return False
    return (
        joint.type == "fixed"
        and joint.parent == tcp.base_link
        and joint.child == tcp.link_name
    )


def _preview(names: Sequence[str], limit: int = 8) -> str:
    """Render a bounded name list for an error message."""
    if len(names) <= limit:
        return ", ".join(names)
    return f"{', '.join(names[:limit])}, ... (+{len(names) - limit} more)"


__all__ = [
    "TCP_LINK_SUFFIX",
    "AlignmentSpec",
    "CameraReference",
    "SpecError",
    "TcpSpec",
    "load_spec",
    "parse_spec",
]
