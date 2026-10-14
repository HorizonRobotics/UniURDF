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

"""The alignment viewer: author a spec, watch it, export it.

Forward kinematics runs here, in numpy, not in the browser. A second FK
implementation in JavaScript would be untestable and would drift from the one
that produces the exported URDF -- and the whole point of the viewer is that
what you see is what you get. The client sends joint values and draws the 4x4
transforms that come back.

The server is a **three-stage workspace**:

Stage 1 (source) -- on startup the project is staged: the source URDF and its
    resolvable meshes are copied into the output directory. The viewer shows
    the *staged* URDF's frames and meshes, plus structural diagnostics. No
    alignment has run. The user may edit the staged ``.source.urdf`` on disk
    to fix ``package://`` refs / conflicting parents / xacro vars, then
    re-run.
Stage 2 (aligned) -- pressing "Run alignment" aligns, verifies, and returns
    per-link deltas. ``/api/pose`` with ``stage="aligned"`` drives the
    before/after toggle.
Export -- two separate buttons write the updated config YAML and the aligned
    URDF.  They only write files; they never recompute.

Endpoints:

===========================  ==========================================
``GET  /``                   the page
``GET  /api/model``          staged robot: links, edges, chains,
                             meshes, diagnostics, mesh copy report
``POST /api/pose``           ``{joints, spec, stage, seq}`` → transforms
``POST /api/align``          align + verify + per-link delta table
``POST /api/export/config``  write the spec as YAML
``POST /api/export/urdf``    write the aligned URDF
``GET  /meshes/<path>``      mesh files, sandboxed to staged tree
===========================  ==========================================
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from flask import Flask, Response, jsonify, request, send_file

from uniurdf.assets import in_tree_mesh_path
from uniurdf.geometry import (
    parse_float_triple,
    rotation_between,
    visual_world_poses,
)
from uniurdf.kinematics.chain import Chain
from uniurdf.project import (
    AlignmentStatus,
    Project,
    StagedRobot,
    run_alignment,
)
from uniurdf.prune import focus_links
from uniurdf.robot import AlignedRobot, Robot, VerifyReport
from uniurdf.spec import AlignmentSpec, SpecError, parse_spec

_STATIC = Path(__file__).resolve().parent / "static"

# Continuous joints declare no limits; give sliders something usable.
_DEFAULT_LIMIT = (-np.pi, np.pi)

#: Mesh formats the viewer serves, sent to the page in ``/api/model`` so it
#: knows which loaders to dispatch. A format served without a loader gives a
#: mesh that never appears, reported only in the browser console, so this
#: list and the page's loaders must hold the same formats.
_MESH_SUFFIXES = ("stl", "dae", "obj", "glb", "gltf")

#: Longest export filename accepted, in bytes. Common filesystems stop at 255
#: per component, so a longer name cannot be written whatever it says.
_MAX_NAME_BYTES = 255


@dataclass(frozen=True)
class Session:
    """What the server was started with, built from a Project.

    Attributes:
        project (Project): The fully resolved alignment project.
        staged (StagedRobot): The staged copy of the source URDF in the
            output directory. ``staged.robot`` may be ``None`` when the URDF
            has fatal structural defects; every endpoint degrades gracefully
            in that case rather than raising a 500.
    """

    project: Project
    staged: StagedRobot

    @property
    def robot(self) -> Robot | None:
        """The staged robot, or None when the URDF was unloadable."""
        return self.staged.robot

    @property
    def spec(self) -> AlignmentSpec:
        """The opening alignment spec."""
        return self.project.spec

    @property
    def output_root(self) -> Path:
        """The directory all exports must write into."""
        return self.project.output

    @property
    def mesh_root(self) -> Path:
        """The staged output directory that meshes are served from."""
        return self.project.output


def create_app(session: Session) -> Flask:
    """Build the Flask application.

    Args:
        session (Session): The staged robot, spec, and write sandbox.

    Returns:
        Flask: The configured application.
    """
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index() -> Response:
        response = Response((_STATIC / "index.html").read_text(), mimetype="text/html")
        # The page embeds no state, so caching it would only serve a stale
        # bundle after an upgrade.
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/static/<path:name>")
    def static_file(name: str):
        target = _resolve_within(_STATIC, name)
        if target is None or not target.is_file():
            return jsonify({"error": f"not found: {name}"}), 404
        return send_file(target)

    @app.get("/api/model")
    def model():
        return jsonify(describe_model(session))

    @app.post("/api/pose")
    def pose():
        payload = request.get_json(silent=True) or {}
        joints = payload.get("joints") or {}
        if not isinstance(joints, dict):
            return jsonify({"error": "`joints` must be an object"}), 400

        values = {
            name: float(value)
            for name, value in joints.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

        stage = payload.get("stage", "source")
        if stage not in ("source", "aligned"):
            return jsonify({"error": "`stage` must be 'source' or 'aligned'"}), 400

        if session.robot is None:
            # The URDF was unloadable; tell the client why rather than 500.
            return jsonify(
                {
                    "seq": payload.get("seq"),
                    "error": ("URDF has fatal structural defects; no FK available"),
                    "diagnostics": _diagnostics_payload(session),
                }
            ), 422

        if stage == "source":
            transforms = _flatten(session.robot.fk(values))
            return jsonify(
                {
                    "seq": payload.get("seq"),
                    "transforms": transforms,
                    "tcp_links": [],
                    "compat_links": [],
                }
            )

        # stage == "aligned"
        try:
            aligned = session.robot.align(_spec_from(payload, session))
        except SpecError as error:
            return jsonify({"error": str(error)}), 400

        return jsonify(
            {
                "seq": payload.get("seq"),
                "transforms": _flatten(aligned.fk(values)),
                "tcp_links": list(aligned.tcp_links),
                "compat_links": list(aligned.report.camera_compat_links),
            }
        )

    @app.post("/api/align")
    def align():
        payload = request.get_json(silent=True) or {}

        if session.robot is None:
            return jsonify(
                {
                    "error": (
                        "URDF has fatal structural defects; cannot align. "
                        "Fix the issues in the staged copy and reload."
                    ),
                    "diagnostics": _diagnostics_payload(session),
                }
            ), 422

        try:
            spec = _spec_from(payload, session)
            aligned = session.robot.align(spec)
        except SpecError as error:
            return jsonify({"error": str(error)}), 400

        verify = aligned.verify()
        delta_table = _delta_table(session.robot, aligned)

        return jsonify(
            {
                "alignment": {
                    "normalized_joints": list(aligned.report.normalized_joints),
                    "tcp_links": list(aligned.tcp_links),
                    "compat_links": list(aligned.report.camera_compat_links),
                    "warnings": list(aligned.report.warnings),
                    "rotated_joints": [
                        [name, degrees]
                        for name, degrees in aligned.report.rotated_joints
                    ],
                },
                "verify": _verify_summary(verify),
                "deltas": delta_table,
                # The aligned robot carries links the source does not -- the
                # emitted `_tcp` frames, and any camera-compat links. The page
                # builds its link groups from the source `model.links`, so
                # without these it has nowhere to hang those transforms and
                # they are silently dropped: the pose reply carries more
                # entries than there are groups, and the TCP triads never
                # appear.
                "links": list(aligned.chain.link_names),
                "edges": _edges(aligned.chain),
                "focus_links": _focus_links_aligned(session, aligned),
                # Alignment compensates the S-inverse into each rotated link's
                # visual origins, so the aligned URDF's origins differ from
                # the source's on every link whose axis moved -- by as much as
                # 180 degrees.
                # Stage 2 draws aligned FK poses, so it must pair them with
                # aligned origins: keeping the source's would apply the
                # rotation once instead of twice and draw those meshes turned,
                # which is exactly the invariant alignment exists to preserve.
                "meshes": _mesh_payload_from_urdf(session, aligned.urdf_bytes),
            }
        )

    @app.post("/api/export/config")
    def export_config():
        payload = request.get_json(silent=True) or {}
        try:
            spec = _spec_from(payload, session)
        except SpecError as error:
            return jsonify({"error": str(error)}), 400

        # A preview renders the spec without writing it, so the pane shows
        # byte-for-byte what an export would land on disk.
        if payload.get("preview"):
            return jsonify({"yaml": _spec_to_yaml(spec)})

        name = payload.get("name")
        if name is None:
            name = session.project.stem + ".yaml"
        if not _is_safe_name(name):
            return jsonify({"error": f"unsafe filename: {name}"}), 400

        target = session.output_root / name
        if not _within_sandbox(session.output_root, target):
            return jsonify({"error": "export path escapes output root"}), 400
        if target.suffix.lower() not in (".yaml", ".yml"):
            # A config is YAML. Any other extension in the output directory
            # names something a run produced or the user hand-edited -- the
            # staged .source.urdf above all -- and writing YAML there would
            # replace it with a file of a different kind.
            return jsonify(
                {"error": (f"config exports must end in .yaml or .yml: {name}")}
            ), 400

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(_spec_to_yaml(spec), encoding="utf-8")
        except (OSError, ValueError) as error:
            # `strerror` alone: the exception's text carries the full server
            # path, which the browser has no use for and should not receive.
            reason = getattr(error, "strerror", None) or type(error).__name__
            return jsonify({"error": f"could not write {name}: {reason}"}), 500

        return jsonify({"config": str(target)})

    @app.post("/api/export/urdf")
    def export_urdf():
        payload = request.get_json(silent=True) or {}

        if session.robot is None:
            return jsonify(
                {
                    "error": (
                        "URDF has fatal structural defects; cannot export. "
                        "Fix the issues in the staged copy and reload."
                    ),
                    "diagnostics": _diagnostics_payload(session),
                }
            ), 422

        try:
            spec = _spec_from(payload, session)
        except SpecError as error:
            return jsonify({"error": str(error)}), 400

        name = payload.get("name")
        if name is None:
            # Default to <stem>.urdf so the export never collides with the
            # staged source copy (<stem>.source.urdf).
            name = session.project.stem + ".urdf"
        if not _is_safe_name(name):
            return jsonify({"error": f"unsafe filename: {name}"}), 400
        if not _within_sandbox(session.output_root, session.output_root / name):
            return jsonify({"error": "export path escapes output root"}), 400

        # The spec being edited and the requested filename override what
        # the project was loaded with; nothing else changes. Going through
        # run_alignment applies the same refusals the command line does.
        outcome = run_alignment(
            replace(session.project, spec=spec, stem=Path(name).stem),
            force=bool(payload.get("force")),
            staged=session.staged,
        )

        if outcome.status is AlignmentStatus.SPEC_ERROR:
            return jsonify({"error": outcome.error}), 400
        if outcome.status is AlignmentStatus.FATAL_DIAGNOSTICS:
            return jsonify(
                {
                    "error": outcome.error,
                    "diagnostics": _diagnostics_payload(session),
                }
            ), 422
        if outcome.status is AlignmentStatus.WRITE_FAILED:
            return jsonify({"error": outcome.error}), 500
        if outcome.bundle is None:
            # Verification failed, so nothing was written. 409 rather than
            # 500: the request was well formed and the tool worked, it just
            # will not vouch for the result. Retry with `force` to write it.
            return jsonify(
                {
                    "error": outcome.error,
                    "verify": _verify_summary(outcome.verify),
                }
            ), 409

        aligned = outcome.aligned
        assert aligned is not None  # a bundle implies an alignment
        written = outcome.bundle.write
        inventory_path = outcome.bundle.inventory
        pruned_inventory = outcome.bundle.pruned_inventory

        prune = written.prune
        assert prune is not None  # write_bundle always prunes
        return jsonify(
            {
                "urdf": str(written.urdf),
                "inventory": str(inventory_path),
                "pruned_urdf": str(written.pruned_urdf),
                "pruned_inventory": str(pruned_inventory),
                "pruned_links": len(prune.kept_links),
                "dropped_links": len(prune.dropped_links),
                # The client warns on this: a renumbered joint vector is the
                # one way to misuse the pruned file silently.
                "pruned_renumbers": prune.renumbers_joint_vector,
                "dropped_movable_joints": list(prune.dropped_movable_joints),
                "copied_meshes": len(written.copied_meshes),
                "missing_meshes": list(written.missing_meshes),
                "normalized_joints": list(aligned.report.normalized_joints),
                "tcp_links": list(aligned.tcp_links),
                "compat_links": list(aligned.report.camera_compat_links),
                "warnings": list(aligned.report.warnings),
                "forced": outcome.forced,
            }
        )

    @app.get("/meshes/<path:name>")
    def mesh(name: str):
        # Meshes are served from the staged output tree: that is where
        # copy_urdf_asset_tree placed them during staging.
        target = _resolve_within(session.mesh_root, name)
        if target is None or not target.is_file():
            return jsonify({"error": f"not found: {name}"}), 404
        if target.suffix.lower().lstrip(".") not in _MESH_SUFFIXES:
            # Serving arbitrary files from the robot's directory would leak
            # whatever else lives there.
            return jsonify({"error": f"not a mesh: {name}"}), 403
        return send_file(target)

    return app


def describe_model(session: Session) -> dict[str, Any]:
    """Everything the viewer needs to draw the robot at stage 1 (source).

    Describes the **staged** robot: the copy in the output directory that
    the user may edit in place to repair broken references. This is what
    ``/api/pose?stage=source`` returns transforms for, so the mesh origins
    served here must come from the same URDF.

    When ``staged.robot`` is ``None`` (fatal diagnostics prevent loading),
    the response still returns all structural information -- links, edges,
    spec, diagnostics -- just without movable joints or meshes.

    Args:
        session (Session): The active session.

    Returns:
        dict[str, Any]: Links, joint edges, slider ranges, chain grouping,
        the opening spec, per-link mesh references, structural diagnostics,
        and the mesh copy report.
    """
    robot = session.robot
    if robot is None:
        return _describe_unloadable(session)

    chain = robot.chain
    links = list(chain.link_names)
    index = {name: position for position, name in enumerate(links)}
    limits = chain.joint_limits()

    joints = []
    edges = []
    for joint in chain.joints:
        if joint.parent in index and joint.child in index:
            edges.append([index[joint.parent], index[joint.child]])
        if not joint.is_movable:
            continue
        lower, upper = limits.get(joint.name, _DEFAULT_LIMIT)
        joints.append(
            {
                "name": joint.name,
                "type": joint.type,
                "child": joint.child,
                "lower": float(lower),
                "upper": float(upper),
                "mimic": joint.mimic[0] if joint.mimic else None,
            }
        )

    # Source-URDF mesh origins alongside source poses -- same URDF, same FK.
    meshes = _mesh_payload_from_urdf(session, robot.urdf_bytes)

    return {
        "robot": robot.name,
        "source_urdf": str(session.staged.staged_path),
        "output_root": str(session.output_root),
        "mesh_root": str(session.mesh_root),
        "links": links,
        "edges": edges,
        "joints": joints,
        "chains": [list(group) for group in session.spec.resolved_chains(chain)],
        "focus_links": _focus_links_source(session, chain),
        "spec": _spec_payload(session.spec),
        # tcp_base_links and tcp_joints let the client highlight TCP origins
        # and group gripper sliders without re-deriving the spec.
        "tcp_base_links": [entry.base_link for entry in session.spec.tcps],
        "tcp_joints": [
            list(entry.joints) for entry in session.spec.tcp if entry is not None
        ],
        "meshes": meshes,
        # The formats the server will serve. The page dispatches its loaders
        # off this, so a format it cannot draw is never offered.
        "mesh_formats": list(_MESH_SUFFIXES),
        "diagnostics": _diagnostics_payload(session),
        "asset": _asset_payload(session),
        "warnings": list(_model_warnings(chain)),
        "notes": list(_model_notes(session, chain, joints, robot.urdf_bytes)),
    }


def _describe_unloadable(session: Session) -> dict[str, Any]:
    """Minimal payload when the staged URDF could not be loaded.

    The GUI must open a broken URDF so the user can diagnose and fix it.
    An unloadable robot has no chain, no joints, and no
    meshes -- but it still has diagnostics and a staged path to edit.
    """
    return {
        "robot": session.project.stem,
        "source_urdf": str(session.staged.staged_path),
        "output_root": str(session.output_root),
        "mesh_root": str(session.mesh_root),
        "links": [],
        "edges": [],
        "joints": [],
        "chains": [],
        "focus_links": [],
        "spec": _spec_payload(session.spec),
        "tcp_base_links": [],
        "tcp_joints": [],
        "meshes": {},
        "diagnostics": _diagnostics_payload(session),
        "asset": _asset_payload(session),
        "warnings": [],
        "notes": [
            "URDF has fatal structural defects; see diagnostics. "
            f"Edit {session.staged.staged_path} to fix them, then reload."
        ],
    }


def _edges(chain: Chain) -> list[list[int]]:
    """Parent/child link index pairs, for drawing the skeleton.

    Indices are positions in ``chain.link_names``, which is the same order the
    payload's ``links`` list uses.
    """
    index = {name: position for position, name in enumerate(chain.link_names)}
    return [
        [index[joint.parent], index[joint.child]]
        for joint in chain.joints
        if joint.parent in index and joint.child in index
    ]


def _focus_links_aligned(session: Session, aligned: AlignedRobot) -> list[str]:
    """The frames worth labelling on the *aligned* robot.

    Everything the spec asks about, plus the links alignment itself created:
    the emitted ``_tcp`` frames and any camera-compat links. Those cannot
    come from the spec alone because they do not exist until alignment runs.

    Returns:
        list[str]: Link names, in declaration order, without duplicates.
    """
    return list(
        focus_links(
            aligned.chain,
            session.spec,
            extra_links=(
                *aligned.tcp_links,
                *aligned.report.camera_compat_links,
            ),
        )
    )


def _focus_links_source(session: Session, chain: Chain) -> list[str]:
    """The links the spec is about, resolved against the source chain.

    A robot has far more links than an alignment concerns itself with, often
    by a factor of two, and labelling every one buries the frames under
    review. :func:`~uniurdf.prune.focus_links` makes the selection, so the
    frames drawn here are the ones the pruned URDF keeps.

    Returns:
        list[str]: Link names, in declaration order, without duplicates.
    """
    return list(focus_links(chain, session.spec))


def _model_notes(
    session: Session,
    chain: Chain,
    joints: list[dict[str, Any]],
    urdf_bytes: bytes | None = None,
) -> list[str]:
    """Plain statements of what the viewer is about to show.

    Surfaced so an empty or sparse view explains itself. A silent blank canvas
    is indistinguishable from a broken server, and leaves the user with no way
    to tell a correct empty result from a failure.
    """
    notes = [
        f"{len(chain.link_names)} links, {len(joints)} movable joints",
    ]

    mesh_entries = _mesh_payload_from_urdf(session, urdf_bytes)
    total = sum(len(entries) for entries in mesh_entries.values())
    if not total:
        notes.append("no visual meshes declared; showing frames only")
        return notes

    missing = _unresolved_mesh_paths(session, mesh_entries)
    if not missing:
        notes.append(f"{total} visual meshes, all resolved")
    else:
        notes.append(
            f"{len(missing)} of {total} mesh references not found under "
            f"{session.mesh_root} -- those links show frames only. "
            f"Set `mesh_root` in the config if the meshes live elsewhere."
        )
        notes.extend(f"missing: {name}" for name in missing[:5])
        if len(missing) > 5:
            notes.append(f"... and {len(missing) - 5} more")
    return notes


def _unresolved_mesh_paths(
    session: Session,
    mesh_entries: dict[str, list[dict[str, Any]]] | None = None,
) -> list[str]:
    """Mesh references that do not resolve to a file under the mesh root."""
    entries = _mesh_payload_from_urdf(session) if mesh_entries is None else mesh_entries
    missing: list[str] = []
    for group in entries.values():
        for entry in group:
            reference = in_tree_mesh_path(entry["filename"])
            target = _resolve_within(session.mesh_root, reference)
            if (target is None or not target.is_file()) and (
                entry["filename"] not in missing
            ):
                missing.append(entry["filename"])
    return missing


def _model_warnings(chain: Chain) -> list[str]:
    """Structural problems the viewer should show rather than hide."""
    warnings = []
    if chain.duplicate_child_links:
        warnings.append(
            "ambiguous structure: "
            f"{', '.join(chain.duplicate_child_links)} "
            "each have more than one parent joint"
        )
    if chain.orphan_links:
        warnings.append(
            f"{len(chain.orphan_links)} link(s) are not connected to the "
            "root and are not drawn"
        )
    return warnings


def _verify_summary(verify: VerifyReport | None) -> dict[str, Any] | None:
    """Render a verification report for the browser.

    Shared by the align and export routes so a refusal to write reports the
    same numbers the page already showed for that alignment.

    Args:
        verify (VerifyReport | None): The report, or ``None`` when the run
            stopped before verifying.

    Returns:
        dict[str, Any] | None: JSON-ready summary, or ``None``.
    """
    if verify is None:
        return None
    return {
        "ok": verify.ok,
        "checked_configurations": verify.checked_configurations,
        "compared_links": verify.compared_links,
        "compared_visuals": verify.compared_visuals,
        "max_link_position_error": verify.max_link_position_error,
        "max_mesh_position_error": verify.max_mesh_position_error,
        "max_mesh_rotation_error": verify.max_mesh_rotation_error,
        "moved_links": list(verify.moved_links),
        "moved_meshes": list(verify.moved_meshes),
        "rotated_links": [
            {"link": name, "angle_deg": deg} for name, deg in verify.rotated_links
        ],
    }


def _diagnostics_payload(session: Session) -> list[dict[str, Any]]:
    """All structural findings, serialized for the client.

    Each finding carries severity, code, and message so the client can
    colour-code and filter them.
    """
    return [
        {
            "severity": f.severity,
            "code": f.code,
            "message": f.message,
        }
        for f in session.staged.diagnostics.findings
    ]


def _asset_payload(session: Session) -> dict[str, Any]:
    """Mesh copy report: what resolved, what did not, what was rewritten."""
    asset = session.staged.asset
    return {
        "staged_urdf": str(asset.urdf),
        "copied_mesh_count": len(asset.local_mesh_files),
        "unresolved_meshes": list(asset.unresolved_meshes),
        "escaped_meshes": [
            {"original": orig, "rewritten": rew} for orig, rew in asset.escaped_meshes
        ],
    }


def _spec_payload(spec: AlignmentSpec) -> dict[str, Any]:
    """Serialize a spec for the authoring panel, angles back in degrees."""
    return {
        "source_urdf": str(spec.source_urdf) if spec.source_urdf else "",
        "mesh_root": str(spec.mesh_root) if spec.mesh_root else "",
        "output": str(spec.output) if spec.output else "",
        "chains": [list(group) for group in spec.chains],
        "tcp": [
            None
            if entry is None
            else {
                "base_link": entry.base_link,
                "xyz": list(entry.xyz),
                "rpy": [float(np.degrees(value)) for value in entry.rpy],
                # joints round-trips as a list so the client can group sliders
                # and so that the spec payload is fully self-describing.
                "joints": list(entry.joints),
            }
            for entry in spec.tcp
        ],
        "camera_references": [
            {"link": reference.link} for reference in spec.camera_references
        ],
        # Sparse map, degrees, so the editor can show a value per joint row
        # without caring which chain the joint belongs to.
        "rot_z": {name: float(np.degrees(radians)) for name, radians in spec.rot_z},
    }


class _InlineList(list):
    """A list rendered on one line, as ``[a, b, c]``."""


def _represent_inline_list(dumper: yaml.Dumper, data: _InlineList):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.add_representer(_InlineList, _represent_inline_list)


def _spec_to_yaml(spec: AlignmentSpec) -> str:
    """Serialize a spec to YAML, angles in degrees (as the user declared).

    The dataclass stores rpy in radians; the YAML format uses degrees.
    This mirrors the ``_spec_payload`` convention exactly.

    Coordinate triples and joint-name lists render inline; everything else is
    block style, matching ``example/example_arm.yaml``. Keys keep the order
    below rather than being sorted, so a spec reads source, then structure.
    """
    data: dict[str, Any] = {}
    if spec.source_urdf is not None:
        data["source_urdf"] = str(spec.source_urdf)
    if spec.mesh_root is not None:
        data["mesh_root"] = str(spec.mesh_root)
    if spec.output is not None:
        data["output"] = str(spec.output)
    if spec.chains:
        data["chains"] = [_InlineList(group) for group in spec.chains]
    if spec.tcp:
        tcp_list = []
        for entry in spec.tcp:
            if entry is None:
                tcp_list.append(None)
            else:
                tcp_entry: dict[str, Any] = {
                    "base_link": entry.base_link,
                    "xyz": _InlineList(entry.xyz),
                    "rpy": _InlineList(
                        round(float(np.degrees(v)), 6) for v in entry.rpy
                    ),
                }
                if entry.joints:
                    tcp_entry["joints"] = _InlineList(entry.joints)
                tcp_list.append(tcp_entry)
        data["tcp"] = tcp_list
    if spec.camera_references:
        data["camera_references"] = [
            {"link": ref.link} for ref in spec.camera_references
        ]
    if spec.rot_z:
        data["rot_z"] = {
            name: round(float(np.degrees(radians)), 6) for name, radians in spec.rot_z
        }
    return yaml.dump(
        data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )


def _spec_from(payload: dict[str, Any], session: Session) -> AlignmentSpec:
    """Build the spec for a request, falling back to the session's.

    The page edits structure -- chains, TCPs, camera references, rotations --
    and posts only those fields. Where the URDF lives is a property of the
    run, so the session's paths are carried onto the result and an exported
    config names the URDF it was edited against.

    Raises:
        SpecError: If the supplied spec is malformed.
    """
    raw = payload.get("spec")
    if raw is None:
        return session.spec
    if not isinstance(raw, dict):
        raise SpecError("`spec` must be an object")
    edited = parse_spec(raw, origin="<request>")
    return replace(
        edited,
        source_urdf=edited.source_urdf or session.spec.source_urdf,
        mesh_root=edited.mesh_root or session.spec.mesh_root,
        output=edited.output or session.spec.output,
    )


def _mesh_payload_from_urdf(
    session: Session, urdf_bytes: bytes | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Per-link visual mesh references, relative to the mesh route.

    Only ``<visual>`` geometry is reported: URDFs routinely declare a detailed
    visual mesh and a cruder collision mesh for the same link, and drawing
    both would double the geometry for no benefit.

    The ``urdf_bytes`` argument controls which URDF's origins are read.
    **Callers must pass the same URDF whose FK poses they will use.** Mixing
    source origins with aligned poses (or vice versa) is the exact bug this
    function was designed to prevent: such a mismatch leaves every compensated
    mesh visibly rotated.

    Args:
        session (Session): The active session.
        urdf_bytes (bytes, optional): Which URDF to read origins from.
            Defaults to the staged source URDF.

    Returns:
        dict[str, list[dict[str, Any]]]: Per-link mesh entries.
    """
    robot = session.robot
    if robot is None:
        return {}
    bytes_to_parse = urdf_bytes if urdf_bytes is not None else robot.urdf_bytes
    root = ET.fromstring(bytes_to_parse)
    meshes: dict[str, list[dict[str, Any]]] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if not name:
            continue
        entries = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.get("filename")
            if not filename:
                continue
            entries.append(
                {
                    "filename": filename,
                    # Where the copy actually landed. `filename` stays the
                    # verbatim ref because that is what a warning must name
                    # for the user to find it in the URDF; the client should
                    # not have to re-derive the copy layout to fetch a file.
                    "path": in_tree_mesh_path(filename),
                    "scale": _scale(mesh.get("scale")),
                    "origin": _origin(visual.find("origin")),
                }
            )
        if entries:
            meshes[name] = entries
    return meshes


def _delta_table(robot: Robot, aligned: AlignedRobot) -> list[dict[str, Any]]:
    """Per-link position and rotation deltas between source and aligned.

    This is the key diagnostic output of ``/api/align``: it tells the user
    which link frames changed orientation (that is alignment working) and
    confirms that no link moved (that is the invariant).

    For links with visual meshes, mesh world pose deltas are also included.
    Both sides use :func:`visual_world_poses` with their own URDF and their
    own FK poses -- pairing source origins with aligned frames silently
    rotates every mesh, so one helper serves both sides.

    Args:
        robot (Robot): The staged source robot.
        aligned (AlignedRobot): The aligned result.

    Returns:
        list[dict[str, Any]]: One entry per shared link, sorted by link name.
        Each entry has:
            link: link name
            position_delta: max absolute position change (m) at zero config
            rotation_delta: frame orientation change (deg) at zero config
            mesh_position_delta: max mesh world position change (m), or null
            mesh_rotation_delta: max mesh world rotation change (deg), or null
    """
    src_poses = robot.fk()
    aln_poses = aligned.fk()

    # Source and aligned share all links the source declared.
    shared = sorted(set(src_poses.keys()) & set(aln_poses.keys()))

    # Build mesh-world-pose maps once, each from its own URDF and its own FK.
    src_mesh = visual_world_poses(robot.urdf_bytes, src_poses)
    aln_mesh = visual_world_poses(aligned.urdf_bytes, aln_poses)

    # Group mesh keys by link name for easy lookup.
    src_by_link: dict[str, list[tuple[str, int]]] = {}
    for key in src_mesh:
        src_by_link.setdefault(key[0], []).append(key)

    rows: list[dict[str, Any]] = []
    for link in shared:
        t_src = src_poses[link]
        t_aln = aln_poses[link]

        # --- link position delta ---
        pos_delta = float(np.abs(t_aln[:3, 3] - t_src[:3, 3]).max())

        # --- link frame rotation delta ---
        rot_delta = rotation_between(t_src, t_aln)

        # --- mesh deltas (optional: only for links with visual meshes) ---
        mesh_pos_delta: float | None = None
        mesh_rot_delta: float | None = None
        keys_for_link = src_by_link.get(link, [])
        for key in keys_for_link:
            if key not in aln_mesh:
                continue
            t_ms = src_mesh[key]
            t_ma = aln_mesh[key]
            mp = float(np.abs(t_ma[:3, 3] - t_ms[:3, 3]).max())
            mr = rotation_between(t_ms, t_ma)
            if mesh_pos_delta is None or mp > mesh_pos_delta:
                mesh_pos_delta = mp
            if mesh_rot_delta is None or mr > mesh_rot_delta:
                mesh_rot_delta = mr

        rows.append(
            {
                "link": link,
                "position_delta": pos_delta,
                "rotation_delta": rot_delta,
                "mesh_position_delta": mesh_pos_delta,
                "mesh_rotation_delta": mesh_rot_delta,
            }
        )
    return rows


def _origin(element: ET.Element | None) -> dict[str, list[float]]:
    """Read an ``<origin>`` into plain lists."""
    if element is None:
        return {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}
    return {
        "xyz": _triple(element.get("xyz")),
        "rpy": _triple(element.get("rpy")),
    }


def _triple(text: str | None) -> list[float]:
    """Parse a 3-vector attribute as JSON-ready floats.

    Args:
        text (str | None): The attribute value.

    Returns:
        list[float]: Three floats; zeros when absent or malformed.
    """
    return parse_float_triple(text).tolist()


def _scale(text: str | None) -> list[float]:
    """Parse a mesh scale attribute, defaulting to unity.

    A missing ``scale`` means 1, where a missing ``xyz`` means 0, so the
    absent case cannot share :func:`_triple`'s default.

    Args:
        text (str | None): The attribute value.

    Returns:
        list[float]: Three floats; ones when absent or malformed.
    """
    if not text:
        return [1.0, 1.0, 1.0]
    values = parse_float_triple(text)
    return [1.0, 1.0, 1.0] if not values.any() else values.tolist()


def _flatten(poses: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """Row-major 4x4 transforms as flat lists, ready for JSON."""
    return {
        name: [float(value) for value in matrix.reshape(-1)]
        for name, matrix in poses.items()
    }


def _is_safe_name(name: Any) -> bool:
    """Whether `name` is a plain filename a filesystem will accept.

    Rejects a path component, a NUL or control character, and a name longer
    than :data:`_MAX_NAME_BYTES`.
    """
    if not isinstance(name, str) or not name:
        return False
    if len(name.encode("utf-8")) > _MAX_NAME_BYTES:
        return False
    if any(character < " " or character == "\x7f" for character in name):
        return False
    return name == Path(name).name and name not in (".", "..")


def _resolve_within(root: Path, relative: str) -> Path | None:
    """Resolve `relative` under `root`, or ``None`` if it escapes.

    Guards the mesh and static routes against ``../`` traversal and against
    symlinks pointing outside the sandbox.
    """
    try:
        target = (root / relative).resolve()
        target.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return target


def _within_sandbox(output_root: Path, target: Path) -> bool:
    """Return True when ``target`` is under ``output_root``."""
    try:
        target.relative_to(output_root.resolve())
        return True
    except ValueError:
        return False


def serve(
    project: Project,
    staged: StagedRobot | None = None,
    host: str = "127.0.0.1",
    port: int = 7890,
) -> None:
    """Run the viewer until interrupted.

    Args:
        project (Project): The resolved alignment project. The output
            directory is created by staging if not already present.
        staged (StagedRobot, optional): Pre-staged robot. When ``None``,
            staging is performed here (creates the output directory and copies
            the source URDF + meshes).
        host (str, optional): Interface to bind. Default is ``127.0.0.1``.
        port (int, optional): Port to bind. Default is ``7890``.
    """
    if staged is None:
        staged = project.stage()

    session = Session(project=project, staged=staged)
    app = create_app(session)

    print(f"uniurdf viewer: http://{host}:{port}")
    print(f"  staged  {staged.staged_path}")
    print(f"  exports {session.output_root}")
    missing = _unresolved_mesh_paths(session)
    note = f" ({len(missing)} reference(s) not found)" if missing else ""
    print(f"  meshes  {session.mesh_root}{note}")
    # Name them. A bare count says the viewer will be missing something but
    # not what, and the refs are the whole diagnostic: they show whether the
    # path is wrong, the mesh_root is wrong, or the file is genuinely absent.
    # `run` has always listed them; there is no reason `gui` should not.
    for reference in missing[:8]:
        print(f"    {reference}")
    if len(missing) > 8:
        print(f"    ... and {len(missing) - 8} more")
    if missing:
        print(
            "  those links draw as frames only — set `mesh_root`, or fix "
            "the refs in the staged copy and re-run"
        )

    try:
        from gevent.pywsgi import WSGIServer
    except ImportError:
        # gevent is the production server, but the dev server is enough to
        # drive a single browser tab.
        app.run(host=host, port=port)
        return
    WSGIServer((host, port), app).serve_forever()


__all__ = ["Session", "create_app", "describe_model", "serve"]
