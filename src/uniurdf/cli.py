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

"""Command line interface.

Two verbs, one config file each::

    uniurdf align <config.yaml> [--output DIR] [--force]
    uniurdf view  <config.yaml> [--output DIR] [--host H] [--port P]

``align`` performs the complete workflow in one shot:

1. Resolve paths and announce the output directory.
2. Stage: copy the source URDF + meshes, inspect for structural defects.
3. Report all diagnostics. Refuse to continue on fatal findings.
4. Align: normalize joint axes, attach TCP links, insert camera-compat links.
5. Report what changed (joints normalized, TCPs attached, camera links).
6. Verify: enforce mesh-world position/orientation and link position.
7. Report the three error magnitudes. Nothing is written when verification
   fails, so the output directory never holds geometry the tool could not
   vouch for. ``--force`` writes anyway, for inspecting a bad result.
8. Write ``<output>/<stem>.urdf`` and ``<output>/<stem>.inventory.json``.

``view`` performs steps 1-3, then serves the interactive viewer. Fatal
diagnostics stop ``align`` but never the viewer, which is the tool someone
reaches for when the URDF is broken.

Nothing in this module does alignment math, and it holds no policy of its
own: :func:`~uniurdf.project.run_alignment` owns the sequence and its gates.
This parses arguments, renders the outcome, and maps it to an exit code.
"""

from __future__ import annotations

import argparse
import sys
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

from uniurdf.project import (
    AlignmentStatus,
    StagedRobot,
    load_project,
    run_alignment,
)
from uniurdf.robot import AlignedRobot, VerifyReport, WriteReport
from uniurdf.spec import SpecError

EXIT_OK = 0
EXIT_ERROR = 1
# `argparse` exits 2 for a malformed command line, so verification failure
# takes the next code. A script testing for 2 would otherwise read a typo'd
# flag as geometry the tool could not vouch for.
EXIT_VERIFY_FAILED = 3


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser.

    Returns:
        argparse.ArgumentParser: The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="uniurdf",
        description=(
            "Align robot URDFs onto a shared coordinate convention: actuated "
            "joint axes normalized to +Z, with tool-centre-point links "
            "attached where the spec declares."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    align_p = subparsers.add_parser(
        "align",
        help="stage, align, verify, and write the aligned URDF in one shot",
    )
    align_p.add_argument(
        "config",
        type=Path,
        help="YAML spec file. Must declare `source_urdf`.",
    )
    align_p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "output directory (overrides spec `output`; resolved against the "
            "CWD). No per-robot subdirectory is appended."
        ),
    )
    align_p.add_argument(
        "--force",
        action="store_true",
        help=(
            "write the URDF even when verification fails. The exit status "
            "still reports the failure."
        ),
    )

    view_p = subparsers.add_parser(
        "view",
        help="serve the interactive alignment viewer",
    )
    view_p.add_argument(
        "config",
        type=Path,
        help="YAML spec file. Must declare `source_urdf`.",
    )
    view_p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "output directory (overrides spec `output`; resolved against the "
            "CWD). No per-robot subdirectory is appended."
        ),
    )
    view_p.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "interface to bind (default: 127.0.0.1). The viewer writes "
            "files, so exposing it beyond localhost is an explicit choice."
        ),
    )
    view_p.add_argument("--port", type=int, default=7890, help="port to bind")

    return parser


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _print_diagnostics(staged: StagedRobot) -> None:
    """Print all diagnostic findings, labelled by severity.

    Args:
        staged (StagedRobot): The staged robot whose diagnostics to render.
    """
    diag = staged.diagnostics
    if not diag.findings:
        print("diagnostics: none")
        return
    for line in diag.messages():
        print(f"  {line}")

    # A finding says which element is malformed. This says which joints the
    # robot ended up without, which is what a reader needs to judge how far
    # the damage reaches.
    dropped = _dropped_joints(staged)
    if dropped:
        listed = ", ".join(dropped)
        print(
            f"  => {len(dropped)} declared joint(s) could not be placed and "
            f"are absent from the robot: {listed}"
        )


def _dropped_joints(staged: StagedRobot) -> tuple[str, ...]:
    """Joints the URDF declares that the loaded robot does not carry.

    Args:
        staged (StagedRobot): The staged robot to compare against its source.

    Returns:
        tuple[str, ...]: Missing joint names, in declaration order. Empty
        when the URDF did not load at all, where every joint is missing and
        saying so adds nothing.
    """
    if staged.robot is None:
        return ()
    declared = [
        name
        for element in ET.fromstring(staged.robot.urdf_bytes).findall("joint")
        if (name := element.get("name"))
    ]
    loaded = set(staged.robot.joint_names)
    return tuple(name for name in declared if name not in loaded)


def _print_asset_report(staged: StagedRobot) -> None:
    """Print the mesh copy report: what resolved, what did not.

    Args:
        staged (StagedRobot): The staged robot whose asset report to render.
    """
    asset = staged.asset
    n_copied = len(asset.local_mesh_files)
    if n_copied:
        print(f"staged {n_copied} mesh file(s) beside {staged.staged_path}")
    else:
        print(f"staged URDF: {staged.staged_path}")

    if asset.unresolved_meshes:
        n = len(asset.unresolved_meshes)
        print(
            f"warning: {n} mesh reference(s) could not be resolved and "
            f"were left verbatim — edit the staged copy to fix them:"
        )
        for ref in asset.unresolved_meshes[:8]:
            print(f"  {ref}")
        if n > 8:
            print(f"  ... and {n - 8} more")

    if asset.escaped_meshes:
        n = len(asset.escaped_meshes)
        print(
            f"warning: {n} mesh reference(s) escaped the output root and "
            f"were rewritten:"
        )
        for original, rewritten in asset.escaped_meshes[:5]:
            print(f"  {original!r} -> {rewritten!r}")
        if n > 5:
            print(f"  ... and {n - 5} more")


def _print_alignment_report(aligned: AlignedRobot) -> None:
    """Print what alignment changed.

    Args:
        aligned (AlignedRobot): The aligned robot.
    """
    report = aligned.report
    print(f"normalized {len(report.normalized_joints)} joint axes to +Z")
    if report.rotated_joints:
        rotations = ", ".join(
            f"{name}={degrees:g}deg" for name, degrees in report.rotated_joints
        )
        print(
            f"applied manual rot_z to {len(report.rotated_joints)} "
            f"joint(s): {rotations}"
        )
    if report.tcp_joints:
        print(
            f"attached {len(report.tcp_joints)} TCP link(s): "
            f"{', '.join(aligned.tcp_links)}"
        )
    if report.camera_compat_links:
        print(
            f"emitted {len(report.camera_compat_links)} camera compat "
            f"link(s): {', '.join(report.camera_compat_links)}"
        )
    for warning in report.warnings:
        print(f"warning: {warning}", file=sys.stderr)


def _print_prune_report(write_report: WriteReport) -> None:
    """Print what the pruned URDF dropped, and warn about renumbering.

    Args:
        write_report (WriteReport): The write result carrying a prune report.
    """
    report = write_report.prune
    if report is None or write_report.pruned_urdf is None:
        return
    print(
        f"wrote {write_report.pruned_urdf} "
        f"({len(report.kept_links)} link(s) kept, "
        f"{len(report.dropped_links)} dropped)"
    )
    if not report.renumbers_joint_vector:
        return
    # This is the one way the pruned URDF can be used wrongly without
    # anything appearing to fail: the robot still moves, just not the joints
    # the caller meant. Say it plainly and say which file to index against.
    n = len(report.dropped_movable_joints)
    print(
        f"  dropping {n} movable joint(s) renumbers the joint vector — "
        f"index against the .pruned.inventory.json, not the full one"
    )


def _print_verify_report(report: VerifyReport) -> None:
    """Print the verification verdict with all three error magnitudes.

    Args:
        report (VerifyReport): The verification result.
    """
    print(
        f"verified {report.checked_configurations} configuration(s), "
        f"{report.compared_links} link and {report.compared_visuals} visual "
        f"comparison(s): "
        f"link pos {report.max_link_position_error:.2e} m, "
        f"mesh pos {report.max_mesh_position_error:.2e} m, "
        f"mesh rot {report.max_mesh_rotation_error:.2e} deg"
    )

    if report.rotated_links:
        n = len(report.rotated_links)
        print(
            f"  {n} link frame(s) reoriented (expected — "
            f"this is alignment working correctly):"
        )
        for link, deg in report.rotated_links[:6]:
            print(f"    {link}: {deg:.3f} deg")
        if n > 6:
            print(f"    ... and {n - 6} more")

    if not report.ok:
        if not report.compared_links:
            print(
                "FAIL: nothing was compared -- the aligned URDF shares no "
                "link with its source, so the check proves nothing",
                file=sys.stderr,
            )
        if report.moved_links:
            print(
                f"FAIL: {len(report.moved_links)} link(s) moved:",
                file=sys.stderr,
            )
            for name in report.moved_links[:5]:
                print(f"  {name}", file=sys.stderr)
        if report.moved_meshes:
            print(
                f"FAIL: {len(report.moved_meshes)} mesh(es) moved or rotated:",
                file=sys.stderr,
            )
            for name in report.moved_meshes[:5]:
                print(f"  {name}", file=sys.stderr)
    else:
        print("verify: OK")


# ---------------------------------------------------------------------------
# Verb implementations
# ---------------------------------------------------------------------------


def _run_align(args: argparse.Namespace) -> int:
    """Implement the ``align`` verb.

    Performs the complete workflow: load → stage → diagnose → align → verify
    → write.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Exit status.
    """
    try:
        project = load_project(args.config, output=args.output)
    except SpecError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR

    # The library resolves this but does not announce it, so a misread path
    # is visible before any work is done.
    print(f"output: {project.output}")

    # The library reports an overwrite as a warning; a terminal user wants it
    # on stderr whatever the interpreter's warning filter is set to.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        outcome = run_alignment(project, force=args.force)
    for entry in caught:
        print(f"warning: {entry.message}", file=sys.stderr)

    if outcome.staged is not None:
        _print_diagnostics(outcome.staged)
    if outcome.status is AlignmentStatus.FATAL_DIAGNOSTICS:
        print(f"error: {outcome.error}", file=sys.stderr)
        return EXIT_ERROR
    if outcome.staged is not None:
        _print_asset_report(outcome.staged)

    if outcome.aligned is not None:
        _print_alignment_report(outcome.aligned)
    if outcome.verify is not None:
        _print_verify_report(outcome.verify)

    if outcome.status is AlignmentStatus.SPEC_ERROR:
        print(f"error: {outcome.error}", file=sys.stderr)
        return EXIT_ERROR
    if outcome.status is AlignmentStatus.WRITE_FAILED:
        print(f"error: {outcome.error}", file=sys.stderr)
        return EXIT_ERROR

    if outcome.bundle is None:
        print(f"error: {outcome.error}", file=sys.stderr)
        return EXIT_VERIFY_FAILED

    bundle = outcome.bundle
    print(f"wrote {bundle.write.urdf}")
    _print_prune_report(bundle.write)
    print(f"wrote {bundle.inventory}")
    if bundle.pruned_inventory is not None:
        print(f"wrote {bundle.pruned_inventory}")

    if outcome.forced:
        print(
            "warning: wrote unverified geometry because --force was given",
            file=sys.stderr,
        )
        return EXIT_VERIFY_FAILED
    return EXIT_OK


def _run_view(args: argparse.Namespace) -> int:
    """Implement the ``view`` verb.

    Stages the project, reports what staging found, then hands off to the
    viewer. Fatal diagnostics are reported but do not stop it: a broken URDF
    is precisely when someone opens the viewer.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        int: Exit status. Does not return until the server is stopped.
    """
    try:
        project = load_project(args.config, output=args.output)
    except SpecError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR

    print(f"output: {project.output}")

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            staged = project.stage()
        for entry in caught:
            print(f"warning: {entry.message}", file=sys.stderr)
    except (OSError, SpecError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_ERROR

    _print_diagnostics(staged)
    if staged.diagnostics.is_fatal:
        print(
            f"warning: URDF has fatal structural defects; the viewer will "
            f"open, but alignment will not run until they are fixed. Edit "
            f"{staged.staged_path}.",
            file=sys.stderr,
        )
    _print_asset_report(staged)

    # Imported here so `align` stays usable without Flask installed.
    from uniurdf.gui.app import serve

    # The staged robot is passed along rather than re-staged, so the viewer
    # shows the diagnostics just printed and the meshes are copied once.
    serve(project=project, staged=staged, host=args.host, port=args.port)
    return EXIT_OK


_COMMANDS = {
    "align": _run_align,
    "view": _run_view,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv (list[str], optional): Arguments to parse. Default is
            ``sys.argv[1:]``.

    Returns:
        int: Process exit status. ``0`` succeeded, ``1`` is a usage or input
        error, ``2`` comes from ``argparse`` for a malformed command line,
        and ``3`` means the geometry check failed.
    """
    args = build_parser().parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
