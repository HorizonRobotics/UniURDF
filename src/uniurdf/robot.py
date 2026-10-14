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

"""The package interface: load a robot, align it, inspect it, write it out.

.. code-block:: python

    robot = load_robot("robots/aloha.urdf")
    aligned = robot.align(spec)
    aligned.fk({"fl_joint1": 0.3})  # link poses, for a viewer
    aligned.verify()  # did alignment preserve the geometry?
    aligned.write("out/")  # URDF plus meshes

The CLI, the GUI, and the tests are all thin adapters over these five calls.
That is deliberate: the GUI exports through exactly the code path it renders,
so what you see is what you get.

Alignment is pure -- :meth:`Robot.align` returns a new object and never
touches the filesystem. :meth:`AlignedRobot.write` is the only side effect.

What alignment preserves:
    * Link world *position* -- enforced.
    * Visual world *position* and *orientation* -- enforced, for every
      geometry type.
    * Link world *orientation* is free to change (that is the whole point) --
      reported, never enforced.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from uniurdf.assets import (
    copy_urdf_asset_tree,
    rewrite_resolved_package_uris,
    same_file,
)
from uniurdf.geometry import rotation_between, visual_world_poses
from uniurdf.inventory import build_inventory
from uniurdf.kinematics.chain import (
    Chain,
    build_chain,
    parse_urdf,
    sweep_configuration,
)
from uniurdf.prune import PruneReport, prune_urdf
from uniurdf.spec import AlignmentSpec, SpecError
from uniurdf.transform.axis import normalize_actuated_axes
from uniurdf.transform.camera import insert_camera_compat_links
from uniurdf.transform.provenance import (
    append_provenance,
    strip_provenance,
)
from uniurdf.transform.tcp import insert_tcp_links

# Alignment may reorient a link's frame -- that is its purpose -- but it must
# never move one. This is the tolerance for "did not move", in metres.
VERIFY_ATOL = 1e-9

# Alignment must not move any mesh in the world frame. 1e-4 degrees is
# comfortably above the 2.09e-06 deg decimal round-trip noise observed when
# the URDF source writes `1.5708` rather than the exact pi/2 value, and far
# below any real miscompensation error (which would appear as the full
# joint-axis rotation, typically 90 degrees).
VERIFY_MESH_ROT_ATOL_DEG = 1e-4


@dataclass(frozen=True)
class AlignmentReport:
    """What alignment changed.

    Attributes:
        normalized_joints (tuple[str, ...]): Joints whose axis became ``+Z``.
        tcp_joints (tuple[str, ...]): TCP joints inserted or updated.
        camera_compat_links (tuple[str, ...]): Compat links emitted for
            reference frames that normalization moved.
        warnings (tuple[str, ...]): Non-fatal problems worth surfacing, such as
            a URDF whose structure is ambiguous.
        rotated_joints (tuple[tuple[str, float], ...]): Joints given a manual
            rotation about their own ``+Z``, as ``(name, degrees)`` pairs.
    """

    normalized_joints: tuple[str, ...] = ()
    tcp_joints: tuple[str, ...] = ()
    camera_compat_links: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    rotated_joints: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class VerifyReport:
    """Whether alignment preserved the robot's geometry.

    What is enforced:
        * ``max_link_position_error`` -- every shared link's world position
          must not move.
        * ``max_mesh_position_error`` -- every visual's world position must
          not move.
        * ``max_mesh_rotation_error`` -- every visual's world orientation
          must not rotate.

    Visuals of every geometry type are checked, primitives included, since
    alignment rewrites all of their origins alike.

    What is reported only (and expected to be non-empty for any robot that
    needed alignment):
        * ``rotated_links`` -- links whose world frame orientation changed.
          This is the whole point of alignment: axis normalization rotates
          frames so that joint axes point +Z.

    Attributes:
        ok (bool): True when all three enforced checks passed, and something
            was actually compared.
        max_link_position_error (float): Largest link positional drift, in
            metres, across all shared links and all checked configurations.
        max_mesh_position_error (float): Largest visual positional drift,
            in metres, across all visual-bearing links and configurations.
        max_mesh_rotation_error (float): Largest visual orientation
            change, in degrees.
        moved_links (tuple[str, ...]): Links that violated position tolerance.
        moved_meshes (tuple[str, ...]): Link names whose visual moved.
        rotated_links (tuple[tuple[str, float], ...]): Per-link pairs of
            ``(link_name, angle_degrees)`` for every link whose world
            orientation changed (at the zero configuration). Reported only --
            this is alignment working as intended.
        tcp_poses (dict[str, np.ndarray]): World pose of each emitted TCP at
            the zero configuration, for inspection.
        checked_configurations (int): How many joint configurations were tried.
        compared_links (int): Link comparisons performed, summed over
            configurations. Zero means the two URDFs shared no link, which
            fails: an empty check proves nothing.
        compared_visuals (int): Visual comparisons performed, summed over
            configurations. Zero is normal for a robot with no visuals.
    """

    ok: bool
    max_link_position_error: float
    max_mesh_position_error: float
    max_mesh_rotation_error: float
    moved_links: tuple[str, ...] = ()
    moved_meshes: tuple[str, ...] = ()
    rotated_links: tuple[tuple[str, float], ...] = ()
    tcp_poses: dict[str, np.ndarray] = field(default_factory=dict)
    checked_configurations: int = 0
    compared_links: int = 0
    compared_visuals: int = 0


@dataclass(frozen=True)
class WriteReport:
    """Where the aligned robot landed.

    Attributes:
        urdf (Path): The written URDF.
        copied_meshes (tuple[str, ...]): Mesh files copied beside it.
        missing_meshes (tuple[str, ...]): References that could not be
            resolved; these stay verbatim in the URDF so a live ROS
            environment can still find them.
        pruned_urdf (Path | None): The pruned URDF, when one was requested.
        prune (PruneReport | None): What pruning dropped, when it ran.
    """

    urdf: Path
    copied_meshes: tuple[str, ...] = ()
    missing_meshes: tuple[str, ...] = ()
    pruned_urdf: Path | None = None
    prune: PruneReport | None = None


@dataclass(frozen=True)
class Robot:
    """A parsed URDF, ready to align.

    Attributes:
        path (Path | None): Where the URDF was read from, when it came from
            disk. Needed to resolve relative mesh references on write.
        urdf_bytes (bytes): The source URDF, unmodified.
        chain (Chain): Its kinematic chain.
    """

    path: Path | None
    urdf_bytes: bytes
    chain: Chain

    @property
    def name(self) -> str:
        """The URDF's declared robot name, or ``"robot"`` when it has none."""
        return ET.fromstring(self.urdf_bytes).get("name") or "robot"

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Movable joint names, in the canonical positional order."""
        return self.chain.joint_names

    @property
    def link_names(self) -> tuple[str, ...]:
        """Reachable link names."""
        return self.chain.link_names

    def fk(self, values: Mapping[str, float] | None = None) -> dict[str, np.ndarray]:
        """Link poses at a configuration.

        See :meth:`Chain.forward_kinematics`.
        """
        return self.chain.forward_kinematics(values)

    def align(self, spec: AlignmentSpec) -> AlignedRobot:
        """Apply the alignment described by `spec`.

        Normalizes every actuated joint axis to ``+Z`` -- regardless of which
        joints the spec's `chains` select, since those scope what is drawn and
        what the pruned URDF keeps -- then attaches the declared TCP links.

        Args:
            spec (AlignmentSpec): The alignment declaration.

        Returns:
            AlignedRobot: The aligned robot. The receiver is unchanged.

        Raises:
            SpecError: If the spec contradicts this URDF.
        """
        spec.validate_against(self.chain)

        root = parse_urdf(self.urdf_bytes)
        # Read and remove any record of a previous run's manual rotations
        # before anything else touches the tree, so this run can apply the
        # difference rather than the whole angle again. Removing it first also
        # keeps it from sitting ahead of elements appended below.
        applied_rot_z = strip_provenance(root)
        wanted_rot_z = {name: math.degrees(radians) for name, radians in spec.rot_z}
        delta_rot_z = {
            name: math.radians(degrees - applied_rot_z.get(name, 0.0))
            for name, degrees in wanted_rot_z.items()
        }
        # A joint rotated by a previous run but dropped from the spec has to
        # be turned back, or clearing rot_z would silently leave it rotated.
        for name, degrees in applied_rot_z.items():
            if name not in wanted_rot_z:
                delta_rot_z[name] = math.radians(-degrees)

        # The stages below read numeric attributes -- axis, origin xyz/rpy --
        # straight out of the URDF. A malformed one raises where it is parsed,
        # naming the value but not the file it came from, so it is re-raised
        # as the spec error this method documents.
        try:
            normalized = normalize_actuated_axes(root, rot_z=delta_rot_z)
            tcp_joints = insert_tcp_links(root, spec.tcps)

            compat_links: list[str] = []
            if spec.camera_references:
                # Compare frames at the zero pose: a compat link is a fixed
                # transform, and every stage above is a frame-only rewrite, so
                # one configuration settles whether a reference moved.
                compat_links = insert_camera_compat_links(
                    root,
                    source_poses=self.chain.forward_kinematics(),
                    aligned_poses=build_chain(
                        ET.tostring(root, encoding="utf-8")
                    ).forward_kinematics(),
                    references=spec.camera_references,
                    tcp_joint_names={
                        tcp.base_link: tcp.joint_name for tcp in spec.tcps
                    },
                )
        except SpecError:
            raise
        except ValueError as error:
            source = self.path or "<bytes>"
            raise SpecError(f"cannot align {source}: {error}") from error
        warnings: list[str] = []
        if self.chain.duplicate_child_links:
            warnings.append(
                "URDF declares these links as the child of more than one "
                "joint, so their pose is ambiguous: "
                f"{', '.join(self.chain.duplicate_child_links)}"
            )

        # Last, so the record sits after every element the stages above
        # appended and a re-run reproduces this file byte for byte.
        append_provenance(root, wanted_rot_z)

        return AlignedRobot(
            source=self,
            spec=spec,
            urdf_bytes=_serialize(root),
            report=AlignmentReport(
                normalized_joints=tuple(normalized),
                tcp_joints=tuple(tcp_joints),
                camera_compat_links=tuple(compat_links),
                warnings=tuple(warnings),
                rotated_joints=tuple(sorted(wanted_rot_z.items())),
            ),
        )


@dataclass(frozen=True)
class AlignedRobot:
    """The result of aligning a :class:`Robot`.

    Attributes:
        source (Robot): The robot this was aligned from.
        spec (AlignmentSpec): The spec that produced it.
        urdf_bytes (bytes): The aligned URDF, serialized.
        report (AlignmentReport): What changed.
    """

    source: Robot
    spec: AlignmentSpec
    urdf_bytes: bytes
    report: AlignmentReport
    _chain: list[Chain] = field(default_factory=list, repr=False, compare=False)

    @property
    def chain(self) -> Chain:
        """The aligned robot's kinematic chain, built on first use."""
        if not self._chain:
            self._chain.append(build_chain(self.urdf_bytes))
        return self._chain[0]

    @property
    def tcp_links(self) -> tuple[str, ...]:
        """Names of the emitted TCP links."""
        return tuple(tcp.link_name for tcp in self.spec.tcps)

    def fk(self, values: Mapping[str, float] | None = None) -> dict[str, np.ndarray]:
        """Link poses at a configuration, including the emitted TCP links."""
        return self.chain.forward_kinematics(values)

    def verify(self) -> VerifyReport:
        """Check that alignment preserved the robot's geometry.

        Alignment rewrites joint axes and compensates the surrounding frames.
        Link world orientation may change -- that is the coordinate convention
        being aligned. But a link's *position* must not move, and every
        visual mesh must stay at the same world position and orientation, or
        previously recorded actions would no longer reach the same place.

        Scope: every link the source and aligned URDFs share, not just those
        named in the spec's ``chains``. The ``chains`` field scopes what the
        viewer draws; narrowing verification to chains would leave a hole
        exactly where meshes live.

        Links are checked at the zero configuration and at two opposing
        sweeps; a zero pose alone would hide errors in the joint-motion terms,
        where every joint contributes identity.

        Returns:
            VerifyReport: The verdict, with the worst offenders named.
        """
        source_chain = self.source.chain
        aligned_chain = self.chain
        independent = source_chain.independent_joint_names
        configurations: list[dict[str, float]] = [
            {},
            sweep_configuration(independent),
            sweep_configuration(independent, scale=-0.15),
        ]

        # All links both URDFs share -- no narrowing to chains.
        shared_links = set(source_chain.link_names) & set(aligned_chain.link_names)

        worst_link_pos = 0.0
        worst_mesh_pos = 0.0
        worst_mesh_rot = 0.0
        moved: list[str] = []
        moved_meshes: list[str] = []
        compared_links = 0
        compared_visuals = 0

        for values in configurations:
            src_poses = source_chain.forward_kinematics(values)
            aln_poses = aligned_chain.forward_kinematics(values)

            # --- link position check ---
            for link in shared_links:
                if link not in src_poses or link not in aln_poses:
                    continue
                compared_links += 1
                error = float(
                    np.abs(aln_poses[link][:3, 3] - src_poses[link][:3, 3]).max()
                )
                if error > worst_link_pos:
                    worst_link_pos = error
                if error > VERIFY_ATOL and link not in moved:
                    moved.append(link)

            # --- mesh world pose checks ---
            src_mesh_poses = visual_world_poses(self.source.urdf_bytes, src_poses)
            aln_mesh_poses = visual_world_poses(self.urdf_bytes, aln_poses)
            for key in src_mesh_poses:
                if key not in aln_mesh_poses:
                    continue
                compared_visuals += 1
                t_src = src_mesh_poses[key]
                t_aln = aln_mesh_poses[key]
                link_name = key[0]

                pos_error = float(np.abs(t_aln[:3, 3] - t_src[:3, 3]).max())
                if pos_error > worst_mesh_pos:
                    worst_mesh_pos = pos_error
                if pos_error > VERIFY_ATOL and link_name not in moved_meshes:
                    moved_meshes.append(link_name)

                rot_deg = rotation_between(t_src, t_aln)
                if rot_deg > worst_mesh_rot:
                    worst_mesh_rot = rot_deg
                if rot_deg > VERIFY_MESH_ROT_ATOL_DEG and link_name not in moved_meshes:
                    moved_meshes.append(link_name)

        # --- rotated links: measure at zero configuration only ---
        # A compat link inserted by this run is part of the aligned URDF but
        # not the source, so we skip it gracefully via `shared_links`.
        zero_src = source_chain.forward_kinematics()
        zero_aln = aligned_chain.forward_kinematics()
        rotated: list[tuple[str, float]] = []
        for link in sorted(shared_links):
            if link not in zero_src or link not in zero_aln:
                continue
            angle_deg = rotation_between(zero_src[link], zero_aln[link])
            if angle_deg > VERIFY_MESH_ROT_ATOL_DEG:
                rotated.append((link, angle_deg))

        # A check that compared nothing has proven nothing. Two URDFs sharing
        # no link name skip every comparison, so the count is what separates
        # "clean" from "never looked".
        ok = bool(compared_links) and not moved and not moved_meshes
        zero_pose = aligned_chain.forward_kinematics()
        return VerifyReport(
            ok=ok,
            max_link_position_error=worst_link_pos,
            max_mesh_position_error=worst_mesh_pos,
            max_mesh_rotation_error=worst_mesh_rot,
            moved_links=tuple(moved),
            moved_meshes=tuple(moved_meshes),
            rotated_links=tuple(rotated),
            tcp_poses={
                name: zero_pose[name] for name in self.tcp_links if name in zero_pose
            },
            checked_configurations=len(configurations),
            compared_links=compared_links,
            compared_visuals=compared_visuals,
        )

    def prune(self) -> PruneReport:
        """Cut the aligned URDF down to the links the spec asks about.

        Alignment's own emitted links -- TCPs and camera-compat aliases --
        hang *below* their base link, so they are passed as explicit seeds.
        Without that the tool would delete its own output.

        Returns:
            PruneReport: The pruned URDF and what it dropped.
        """
        return prune_urdf(
            self.urdf_bytes,
            self.spec,
            extra_links=(
                *self.tcp_links,
                *self.report.camera_compat_links,
            ),
            chain=self.chain,
        )

    def write(
        self,
        output_dir: Path | str,
        name: str | None = None,
        prune_name: str | None = None,
    ) -> WriteReport:
        """Write the aligned URDF and its meshes into `output_dir`.

        The directory is created if absent. Meshes are copied beside the URDF
        and ``package://`` references rewritten to relative paths, so the
        result is self-contained and directly servable.

        Args:
            output_dir (Path | str): Destination directory.
            name (str, optional): Output filename. Default is the source
                URDF's name, or ``aligned.urdf`` when the source had no path.
            prune_name (str, optional): When given, also write a pruned URDF
                under this name. It shares the meshes copied for the full one
                -- pruning only removes elements, so every mesh it still
                references was already resolved and rewritten by the same map.

        Returns:
            WriteReport: Where things landed, and what could not be resolved.

        Raises:
            SpecError: If the target would overwrite the source URDF.
            PruneError: If pruning would have moved a kept link.
        """
        destination = Path(output_dir)
        source_path = self.source.path
        filename = name or (source_path.name if source_path else "aligned.urdf")
        target = destination / filename

        if source_path is not None:
            # Writing either output over the input would leave no copy of what
            # was aligned, and the staged source is the file the user edits
            # between runs. Both targets are checked: the pruned URDF is a
            # smaller robot, so landing it on the source destroys more than it
            # replaces.
            for candidate, argument in (
                (target, "`name`"),
                (
                    destination / prune_name if prune_name is not None else None,
                    "`prune_name`",
                ),
            ):
                if candidate is not None and same_file(source_path, candidate):
                    raise SpecError(
                        f"refusing to overwrite the source URDF at "
                        f"{source_path}. Choose a different output directory, "
                        f"or pass a different {argument}."
                    )

        destination.mkdir(parents=True, exist_ok=True)

        # Prune before writing anything: a PruneError must leave the output
        # directory as it was, not half-written.
        pruned = self.prune() if prune_name is not None else None

        if source_path is None:
            # Nothing to resolve meshes against; write the URDF alone.
            target.write_bytes(self.urdf_bytes)
            pruned_path = _write_pruned(destination, prune_name, pruned, rewrites=None)
            return WriteReport(urdf=target, pruned_urdf=pruned_path, prune=pruned)

        copied = copy_urdf_asset_tree(source_urdf=source_path, target_urdf=target)
        urdf_bytes = rewrite_resolved_package_uris(
            self.urdf_bytes, copied.package_uri_rewrites
        )
        target.write_bytes(urdf_bytes)
        pruned_path = _write_pruned(
            destination,
            prune_name,
            pruned,
            rewrites=copied.package_uri_rewrites,
        )
        return WriteReport(
            urdf=target,
            copied_meshes=tuple(str(path) for path in copied.local_mesh_files),
            missing_meshes=tuple(copied.skipped_meshes),
            pruned_urdf=pruned_path,
            prune=pruned,
        )

    def write_bundle(
        self,
        destination: Path,
        stem: str,
    ) -> BundleReport:
        """Write the aligned URDF, its pruned twin, and an inventory for each.

        An inventory names joint indices into the URDF beside it, and pruning
        renumbers that vector, so each URDF carries its own: an index from the
        full inventory addresses a different joint in the pruned tree.

        The inventory is built from :attr:`spec`, so it always describes the
        URDF written beside it.

        Args:
            destination (Path): Directory to write into.
            stem (str): Base name; files are ``<stem>.urdf``,
                ``<stem>.pruned.urdf``, and ``<stem>[.pruned].inventory.json``.

        Returns:
            BundleReport: The write report plus the inventory paths.
        """
        written = self.write(
            destination,
            name=stem + ".urdf",
            prune_name=stem + ".pruned.urdf",
        )
        compat = self.report.camera_compat_links

        def dump(path: Path, chain: Chain) -> Path:
            path.write_text(
                json.dumps(build_inventory(chain, self.spec, compat), indent=2) + "\n"
            )
            return path

        inventory = dump(destination / (stem + ".inventory.json"), self.chain)
        pruned_inventory = None
        if written.pruned_urdf is not None and written.prune is not None:
            pruned_inventory = dump(
                destination / (stem + ".pruned.inventory.json"),
                build_chain(written.prune.urdf_bytes),
            )
        return BundleReport(
            write=written,
            inventory=inventory,
            pruned_inventory=pruned_inventory,
        )


@dataclass(frozen=True)
class BundleReport:
    """Every file one export produced.

    Attributes:
        write (WriteReport): Where the URDFs landed.
        inventory (Path): Inventory for the full URDF.
        pruned_inventory (Path | None): Inventory for the pruned URDF, when
            one was written.
    """

    write: WriteReport
    inventory: Path
    pruned_inventory: Path | None = None


def _write_pruned(
    destination: Path,
    prune_name: str | None,
    pruned: PruneReport | None,
    rewrites: Mapping[str, str] | None,
) -> Path | None:
    """Write the pruned URDF, applying the same mesh rewrites as the full one.

    Pruning only deletes elements, so every mesh reference that survives was
    already resolved when the full URDF's meshes were copied. Reusing that
    same map is what keeps the two files pointing at one shared ``meshes/``
    tree instead of copying it twice.
    """
    if prune_name is None or pruned is None:
        return None
    target = destination / prune_name
    body = pruned.urdf_bytes
    if rewrites:
        body = rewrite_resolved_package_uris(body, rewrites)
    target.write_bytes(body)
    return target


def load_robot(source: Path | str | bytes) -> Robot:
    """Read a URDF into a :class:`Robot`.

    Args:
        source (Path | str | bytes): Path to a ``.urdf`` file, or raw bytes.

    Returns:
        Robot: The parsed robot.

    Raises:
        SpecError: If the URDF cannot be read or parsed.
    """
    if isinstance(source, bytes):
        return Robot(path=None, urdf_bytes=source, chain=build_chain(source))

    path = Path(source)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise SpecError(f"cannot read URDF {path}: {error}") from error
    try:
        chain = build_chain(data)
    except (ET.ParseError, ValueError) as error:
        raise SpecError(f"{path} is not a usable URDF: {error}") from error
    return Robot(path=path, urdf_bytes=data, chain=chain)


def _serialize(root: ET.Element) -> bytes:
    """Serialize with a stable 2-space indent and trailing newline.

    Upstream URDFs arrive with wildly inconsistent whitespace, so without
    this every re-align would produce a churny diff even when nothing changed
    semantically. Normalizing here is what makes re-running a byte-level no-op.
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
    "AlignedRobot",
    "AlignmentReport",
    "BundleReport",
    "Robot",
    "VerifyReport",
    "WriteReport",
    "load_robot",
]
