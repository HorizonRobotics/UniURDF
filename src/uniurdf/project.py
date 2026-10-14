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

"""Path resolution, staging, and diagnostics for a single alignment run.

:func:`load_project` is the entry point. It resolves every path against the
spec file's directory, then returns a :class:`Project` holding the
fully-resolved configuration. From there:

.. code-block:: python

    project = load_project(
        "example/example_arm.yaml", output=Path("/tmp/aligned")
    )
    staged = project.stage()  # copies URDF + meshes, inspects
    aligned = staged.robot.align(project.spec)
    report = aligned.verify()
    written = aligned.write(project.output, name=project.stem + ".urdf")

The output directory holds ``<stem>.source.urdf`` (a verbatim copy of the
input), ``<stem>.urdf`` (the aligned result), and one mesh tree shared by both.

``output`` comes from the spec unless an explicit argument overrides it, and
the override is used flat -- no per-robot subdirectory is appended. Re-running
overwrites with a warning; the output directory is never deleted. Re-running
against a staged ``<stem>.source.urdf``, the workflow for repairing a broken
URDF by hand, reuses the same two names rather than accumulating a suffix. See
:func:`_output_stem`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from uniurdf.assets import CopiedUrdfAsset, copy_urdf_asset_tree
from uniurdf.diagnostics import Diagnostics, inspect_urdf
from uniurdf.prune import PruneError
from uniurdf.robot import (
    AlignedRobot,
    BundleReport,
    Robot,
    VerifyReport,
    load_robot,
)
from uniurdf.spec import AlignmentSpec, SpecError, load_spec


@dataclass(frozen=True)
class Project:
    """A self-sufficient alignment configuration with resolved absolute paths.

    All paths are absolute and have been validated to exist at construction
    time.  Call :meth:`stage` to copy the source URDF and its meshes into the
    output directory.

    Attributes:
        spec (AlignmentSpec): The parsed spec, with ``source_urdf`` and
            ``mesh_root`` already resolved to absolute paths.
        source_urdf (Path): Absolute path to the source URDF.
        mesh_root (Path): Absolute base directory for mesh resolution.
        output (Path): Absolute output directory.
        stem (str): The source URDF's stem, used to derive output filenames.
    """

    spec: AlignmentSpec
    source_urdf: Path
    mesh_root: Path
    output: Path
    stem: str

    def stage(self) -> StagedRobot:
        """Copy the source URDF into the output directory and diagnose it.

        Delegates to :func:`stage`; see that function for the full contract.

        Returns:
            StagedRobot: The staged robot with diagnostics attached.
        """
        return stage(self)


@dataclass(frozen=True)
class StagedRobot:
    """The result of :meth:`Project.stage`: the URDF copied into the output.

    The staged ``.source.urdf`` is what the user edits in place to fix
    ``package://`` refs, conflicting parents, or unexpanded xacro vars.  Once
    repaired, they point ``source_urdf`` at it and re-run.

    Attributes:
        staged_path (Path): Where the ``.source.urdf`` landed.
        robot (Robot | None): The loaded robot from ``staged_path``, or
            ``None`` when the URDF was structurally unloadable (e.g. a cycle).
            The GUI opens regardless; ``align()`` raises if called with
            ``robot is None``.
        diagnostics (Diagnostics): Structural findings for the staged URDF.
        asset (CopiedUrdfAsset): Mesh copy report (unresolved refs,
            escaped refs, package-URI rewrites).
    """

    staged_path: Path
    robot: Robot | None
    diagnostics: Diagnostics
    asset: CopiedUrdfAsset

    def align(self, spec: AlignmentSpec) -> AlignedRobot:
        """Align this robot with the given spec.

        Thin delegation to :meth:`~uniurdf.robot.Robot.align` so callers
        can work entirely through :class:`StagedRobot` without importing
        ``robot.py``.

        Args:
            spec (AlignmentSpec): The alignment declaration to apply.

        Returns:
            AlignedRobot: The aligned robot; the source robot is unchanged.

        Raises:
            SpecError: If the URDF was unloadable (fatal diagnostics), or if
                the spec is incoherent with this URDF.
        """
        if self.robot is None:
            raise SpecError(
                f"cannot align {self.staged_path}: the URDF has fatal "
                f"structural defects that prevent loading.  Fix the issues "
                f"reported in diagnostics and re-run."
            )
        return self.robot.align(spec)


def load_project(
    config: Path | str,
    output: Path | None = None,
) -> Project:
    """Load a config YAML and resolve every path for an alignment run.

    Every relative path in a spec -- ``source_urdf``, ``mesh_root``,
    ``output`` -- anchors to the directory holding the spec file, so a spec
    and the tree it names move together and a run gives the same result from
    any working directory. The ``--output`` CLI argument is the exception: it
    is a run-time choice, so it resolves against the CWD.

    When ``output`` is absent the default is ``workspace/aligned_urdf/<stem>``
    beside the spec.

    The resolved output path is carried on the returned :class:`Project`
    rather than reported, so a caller decides whether and how to surface it.

    Args:
        config (Path | str): Path to the YAML spec.
        output (Path | None): Override for the output directory, resolved
            against the CWD. When given, this is used exactly as stated (no
            per-robot suffix is appended).

    Returns:
        Project: Fully resolved project configuration.

    Raises:
        SpecError: If the spec is missing, malformed, or names a URDF that
            does not exist.
    """
    spec = load_spec(config)
    spec_dir = Path(config).parent

    source_urdf = spec.source_urdf
    if source_urdf is None:
        raise SpecError(
            f"{config}: spec must declare `source_urdf` — the path to the "
            f"URDF file being aligned"
        )
    if not source_urdf.exists():
        raise SpecError(f"source URDF does not exist: {source_urdf}")

    # mesh_root falls back to source_urdf.parent through resolved_mesh_root,
    # which returns a path whenever source_urdf is set.
    mesh_root = spec.resolved_mesh_root
    if mesh_root is None:
        raise SpecError(
            f"{config}: cannot determine a mesh root; declare `mesh_root` or "
            f"`source_urdf`"
        )
    if not mesh_root.is_dir():
        raise SpecError(
            f"mesh_root is not a directory: {mesh_root}. Mesh references "
            f"resolve against it, so nothing would be found."
        )

    # Explicit argument > spec field > default.
    if output is not None:
        resolved_output = Path(output).resolve()
    elif spec.output is not None:
        resolved_output = (spec_dir / spec.output).resolve()
    else:
        # The same stem the output files use, so re-running against a staged
        # copy defaults back to the directory that copy already lives in.
        stem = _output_stem(source_urdf)
        resolved_output = (spec_dir / "workspace" / "aligned_urdf" / stem).resolve()

    return Project(
        spec=spec,
        source_urdf=source_urdf,
        mesh_root=mesh_root,
        output=resolved_output,
        stem=_output_stem(source_urdf),
    )


def _output_stem(source_urdf: Path) -> str:
    """Return the stem that output filenames derive from.

    A trailing ``.source`` is stripped so that re-running against a staged
    copy reuses the names of the run that produced it.  Without this, a
    second run over ``robot.source.urdf`` would take ``robot.source`` as its
    stem and write its aligned output to ``robot.source.urdf`` — on top of
    the very file the user had just repaired by hand.

    Stripping makes the names a fixed point: every iteration stages to
    ``<stem>.source.urdf`` and writes ``<stem>.urdf``, so the workflow can
    be repeated indefinitely without the names growing or colliding.
    """
    stem = source_urdf.stem
    suffix = ".source"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return stem


def stage(project: Project) -> StagedRobot:
    """Stage a project: create the output directory and copy the URDF.

    This is the implementation of :meth:`Project.stage`, exposed as a
    free function so the GUI can call it directly.

    Steps:
    1. Create the output directory (``exist_ok=True``).
    2. Warn if the output directory already contains files: a re-run
       overwrites rather than hard-erroring.
    3. Copy the source URDF to ``<output>/<stem>.source.urdf`` and mirror its
       resolvable meshes via :func:`~uniurdf.assets.copy_urdf_asset_tree`.
    4. Run :func:`~uniurdf.diagnostics.inspect_urdf` on the staged bytes
       and attach the :class:`~uniurdf.diagnostics.Diagnostics`.

    When ``source_urdf`` already lives in the output directory — the user has
    repaired a staged copy and pointed the spec back at it — the mesh copy is
    skipped by ``copy_urdf_asset_tree`` and the staged path resolves to that
    same file, which :func:`_output_stem` keeps stable across iterations.

    Args:
        project (Project): The resolved project configuration.

    Returns:
        StagedRobot: The staged robot with diagnostics attached.
    """
    output_dir = project.output
    if output_dir.exists() and not output_dir.is_dir():
        raise SpecError(
            f"output path is not a directory: {output_dir}. Point --output "
            f"or the spec's `output` at a directory."
        )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SpecError(
            f"could not create the output directory {output_dir}: "
            f"{error.strerror or error}"
        ) from error

    # A re-run overwrites rather than erroring: the output directory is the
    # user's workspace across iterations, so it is written into, never wiped.
    existing = [p for p in output_dir.iterdir() if p != output_dir]
    if existing:
        warnings.warn(
            f"output directory already exists and contains "
            f"{len(existing)} item(s); overwriting: {output_dir}",
            UserWarning,
            stacklevel=2,
        )

    # Re-running against a staged copy resolves to that same file: the stem
    # has no trailing `.source` (see `_output_stem`).
    staged_name = project.stem + ".source.urdf"
    staged_path = output_dir / staged_name

    try:
        asset = copy_urdf_asset_tree(
            source_urdf=project.source_urdf,
            target_urdf=staged_path,
            mesh_root=project.mesh_root,
        )
    except OSError as error:
        raise SpecError(
            f"could not stage {project.source_urdf} into {output_dir}: "
            f"{error.strerror or error}"
        ) from error
    staged_bytes = staged_path.read_bytes()
    diagnostics = inspect_urdf(staged_bytes)

    # load_robot calls build_chain which raises ValueError for cycles/no-root.
    # We must not propagate this: the GUI has to open a broken URDF so the
    # user can diagnose and repair it.  Store None and let diagnostics carry
    # the fatal finding; callers check is_fatal before calling .align().
    try:
        robot: Robot | None = load_robot(staged_path)
    except (SpecError, ValueError):
        robot = None

    return StagedRobot(
        staged_path=staged_path,
        robot=robot,
        diagnostics=diagnostics,
        asset=asset,
    )


class AlignmentStatus(str, Enum):
    """How an alignment run ended.

    Attributes:
        OK: Aligned, verified, and written.
        FATAL_DIAGNOSTICS: The URDF carries a fatal structural finding, so
            nothing was aligned or written.
        SPEC_ERROR: The spec is missing, malformed, or disagrees with the
            URDF it names.
        VERIFY_FAILED: Alignment moved geometry, or compared nothing. Files
            are written only when ``force`` was given.
        WRITE_FAILED: Alignment succeeded but the output could not be
            written.
    """

    OK = "ok"
    FATAL_DIAGNOSTICS = "fatal_diagnostics"
    SPEC_ERROR = "spec_error"
    VERIFY_FAILED = "verify_failed"
    WRITE_FAILED = "write_failed"


@dataclass(frozen=True)
class AlignmentOutcome:
    """Everything one alignment run produced, including why it stopped.

    Carries reports rather than text: a caller renders them for a terminal,
    a browser, or a log, and the decisions that produced them stay here.

    Attributes:
        status (AlignmentStatus): How the run ended.
        project (Project): The resolved configuration it ran with.
        staged (StagedRobot | None): The staged copy, absent only when
            staging itself failed.
        aligned (AlignedRobot | None): The aligned robot, absent when the
            run stopped before aligning.
        verify (VerifyReport | None): The geometry check, absent when the
            run stopped before verifying.
        bundle (BundleReport | None): The files written, absent when nothing
            was written.
        error (str): Why the run stopped, empty when it did not.
        forced (bool): Whether files were written despite a failed check.
    """

    status: AlignmentStatus
    project: Project
    staged: StagedRobot | None = None
    aligned: AlignedRobot | None = None
    verify: VerifyReport | None = None
    bundle: BundleReport | None = None
    error: str = ""
    forced: bool = False

    @property
    def ok(self) -> bool:
        """Whether the run completed and wrote verified output."""
        return self.status is AlignmentStatus.OK


def run_alignment(
    project: Project,
    *,
    force: bool = False,
    staged: StagedRobot | None = None,
) -> AlignmentOutcome:
    """Stage, align, verify, and write, stopping at the first refusal.

    This is the whole alignment workflow, gates included, so every caller
    enforces the same rules:

    * A URDF with a fatal structural finding is not aligned. Parsing skips
      what it cannot resolve, so such a URDF often still loads -- as a robot
      missing the joints it could not read. Its aligned output would carry a
      joint vector that matches no robot anyone inspected.
    * Geometry that failed its check is not written. Output the tool could
      not vouch for is indistinguishable from output it could, and nothing
      downstream re-checks it.

    ``force`` writes despite a failed check, for inspecting a bad result.
    The status still reports the failure.

    Nothing here writes to stdout or stderr. The returned outcome carries
    the reports; rendering them is the caller's job.

    Args:
        project (Project): The resolved configuration to run.
        force (bool): Write even when verification fails.
        staged (StagedRobot | None): An already-staged robot, to avoid a
            second mesh copy. Staged here when absent.

    Returns:
        AlignmentOutcome: What happened, and everything it produced.
    """
    try:
        staged = staged if staged is not None else stage(project)
    except (OSError, SpecError) as error:
        return AlignmentOutcome(
            status=AlignmentStatus.SPEC_ERROR,
            project=project,
            error=str(error),
        )

    # Parsing drops any joint it cannot place, so a URDF carrying a fatal
    # finding often still loads -- with fewer joints than it declares. The
    # inventory written beside its aligned URDF would index that shorter
    # vector while naming the full robot, so the run stops here.
    if staged.diagnostics.is_fatal:
        return AlignmentOutcome(
            status=AlignmentStatus.FATAL_DIAGNOSTICS,
            project=project,
            staged=staged,
            error=(
                f"URDF has fatal structural defects; edit "
                f"{staged.staged_path} to fix them, then re-run"
            ),
        )

    try:
        aligned = staged.align(project.spec)
    except SpecError as error:
        return AlignmentOutcome(
            status=AlignmentStatus.SPEC_ERROR,
            project=project,
            staged=staged,
            error=str(error),
        )

    verify = aligned.verify()
    if not verify.ok and not force:
        return AlignmentOutcome(
            status=AlignmentStatus.VERIFY_FAILED,
            project=project,
            staged=staged,
            aligned=aligned,
            verify=verify,
            error=(
                "refusing to write unverified geometry; pass force to write it anyway"
            ),
        )

    try:
        bundle = aligned.write_bundle(project.output, project.stem)
    except (OSError, SpecError, PruneError) as error:
        return AlignmentOutcome(
            status=AlignmentStatus.WRITE_FAILED,
            project=project,
            staged=staged,
            aligned=aligned,
            verify=verify,
            error=str(error),
        )

    return AlignmentOutcome(
        status=(AlignmentStatus.OK if verify.ok else AlignmentStatus.VERIFY_FAILED),
        project=project,
        staged=staged,
        aligned=aligned,
        verify=verify,
        bundle=bundle,
        forced=not verify.ok,
    )


__all__ = [
    "AlignmentOutcome",
    "AlignmentStatus",
    "Project",
    "StagedRobot",
    "load_project",
    "run_alignment",
    "stage",
]
