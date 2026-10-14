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

"""Copy a source URDF (and its mesh assets) into the output tree.

Mesh references appear in three shapes; each resolves against a single,
explicit ``mesh_root`` — no guessing, no searching multiple bases:

1. ``filename="rel/path/mesh.stl"`` — resolved as ``mesh_root / rel``.
   Default ``mesh_root`` is the source URDF's parent directory.
2. ``filename="package://<pkg>/<rel>"`` — ROS package URI. Resolved as
   ``mesh_root / <pkg> / <rel>`` (strip the scheme, that is the whole rule).
   When it resolves, the **aligned** URDF is rewritten to use the relative
   path ``<pkg>/<rel>`` so the output tree is self-contained; the staged
   ``.source.urdf`` keeps the original string verbatim so it stays a
   faithful copy, unless the ref escapes the output root (see below).
3. Anything that does not resolve is preserved verbatim in
   :attr:`CopiedUrdfAsset.unresolved_meshes` — the user fixes the URDF in
   the staged output copy and re-runs.

Copy layout mirrors the relative sub-path: ``output/<ref>`` is a copy of
``mesh_root/<ref>``, so refs need no rewriting and
``diff <staged>.source.urdf <original>`` shows only real changes.

The sole rewrite applied to the staged copy is a ref that would *escape*
the output root after mirroring — a ref stepping above ``mesh_root``, whether
through leading ``../`` segments or a ``package://../`` package name, both
common enough in real URDFs to be worth handling.
Escaping segments are replaced with a ``_up_N/`` prefix (``N`` = the number
of directory levels that would be left), so the copy lands inside the output
root in a predictable, documented location and the user can see what happened.
Every copy is confined to the output tree: a ref resolving outside it is
reported unresolved rather than followed.

When the source directory and the output directory are the same (re-running
against a staged copy the user has edited), every resolvable ref already sits
at the path it would be copied to, so each copy is a no-op and nothing is
touched.  Declaring a ``mesh_root`` that points elsewhere is the exception:
it is how a run whose meshes never resolved gets them, and it overwrites
in-tree copies with the ones it names.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

# Textual rewriter for `<mesh filename="..."/>`. We rewrite textually rather
# than parse-and-re-serialize so every byte the transformer did not touch
# stays identical to the source — this keeps `diff` output reviewable.
_MESH_FILENAME_ATTR_RE = re.compile(
    r'(<mesh\b[^>]*?\bfilename\s*=\s*")([^"]+)(")',
)

# A ref is a "rel-escape" when, after resolving against mesh_root, its
# normalized PurePosixPath relative to mesh_root starts with "..". We detect
# this by calling Path.relative_to on the resolved absolute path and catching
# the ValueError it raises when the resolved path is outside mesh_root.


@dataclass(frozen=True)
class CopiedUrdfAsset:
    """Result of a URDF + mesh copy pass.

    Attributes:
        urdf: The copied URDF path in the output tree.
        copied_files: URDF plus every mesh/sidecar file copied under the
            output tree.
        local_mesh_files: ``copied_files`` minus the URDF itself.
        unresolved_meshes: Verbatim ``filename`` strings that were not
            copied: the reference was not found under ``mesh_root``, or it
            resolved to a path outside the output tree. They are left as-is
            in the URDF; the user fixes them in the staged copy and re-runs.
        escaped_meshes: ``(original_ref, rewritten_ref)`` pairs for refs that
            would have escaped the output root. Both the staged copy and the
            aligned URDF carry the rewritten ref; a warning is emitted per
            entry.
        package_uri_rewrites: Mapping from ``package://<pkg>/<rel>`` refs
            that resolved successfully to their output-tree-relative path
            ``<pkg>/<rel>``. Apply this to the aligned URDF bytes with
            :func:`rewrite_resolved_package_uris`.
    """

    urdf: Path
    copied_files: tuple[Path, ...] = ()
    local_mesh_files: tuple[Path, ...] = ()
    unresolved_meshes: tuple[str, ...] = ()
    escaped_meshes: tuple[tuple[str, str], ...] = ()
    package_uri_rewrites: dict[str, str] = field(default_factory=dict)

    @property
    def skipped_meshes(self) -> tuple[str, ...]:
        """Unresolved refs; backwards-compat alias for WriteReport."""
        return self.unresolved_meshes


def copy_urdf_asset_tree(
    source_urdf: Path,
    target_urdf: Path,
    mesh_root: Path | None = None,
) -> CopiedUrdfAsset:
    """Copy a URDF and its resolvable mesh files to the output tree.

    The URDF itself is copied byte-for-byte so alignment edits happen on the
    staged copy while the original is never touched.  Meshes are resolved
    through ``mesh_root`` only (no multi-base searching); every
    ref that resolves is copied and every ref that does not is returned in
    :attr:`CopiedUrdfAsset.unresolved_meshes` so the caller can surface an
    actionable checklist.

    Re-running against the staged copy is the second-iteration workflow: the
    user edits it, points ``source_urdf`` at it, and runs again.  Meshes are
    resolved the same way on every run, so with the default ``mesh_root`` each
    ref already sits where it would be copied and the copy is a no-op.  A
    ``mesh_root`` naming a different tree is honoured rather than skipped —
    without that, a first run whose refs never resolved could never be
    repaired by supplying one.

    Args:
        source_urdf: Origin URDF path.
        target_urdf: Destination for the staged (``.source.urdf``) copy.
        mesh_root: Base directory that relative mesh references resolve
            against.  ``package://<pkg>/<rel>`` resolves to
            ``mesh_root / <pkg> / <rel>``.  Defaults to
            ``source_urdf.parent`` when ``None``.

    Returns:
        CopiedUrdfAsset: The copy result, including which meshes were copied,
        which were unresolvable, and which were rewritten to avoid escaping
        the output root.
    """
    effective_mesh_root = mesh_root if mesh_root is not None else source_urdf.parent

    target_urdf.parent.mkdir(parents=True, exist_ok=True)
    _overwrite_copy(source_urdf, target_urdf)
    # `shutil.copy2` preserves source mode bits; some upstream meshes are
    # shipped read-only (0444).  Add owner-write so the aligner can rewrite
    # the staged copy it just created.
    _ensure_owner_writable(target_urdf)

    copied_files: list[Path] = [target_urdf]
    unresolved_meshes: list[str] = []
    escaped_meshes: list[tuple[str, str]] = []
    package_uri_rewrites: dict[str, str] = {}
    # Rewrites applied to the staged .source.urdf (escape corrections only).
    source_rewrites: dict[str, str] = {}

    for original_ref in _iter_mesh_filenames(source_urdf):
        result = _resolve_ref(original_ref, effective_mesh_root)
        if result is None:
            unresolved_meshes.append(original_ref)
            continue

        source_mesh, output_relative, was_escaped, rewritten_ref = result

        if was_escaped:
            escaped_meshes.append((original_ref, rewritten_ref))
            source_rewrites[original_ref] = rewritten_ref

        target_mesh = target_urdf.parent / output_relative
        if not _is_within(target_mesh, target_urdf.parent):
            # Every mesh copy lands inside the output tree. A ref resolving
            # outside it would overwrite a file the run was never pointed at,
            # so it is reported unresolved and left for the user to edit.
            unresolved_meshes.append(original_ref)
            continue
        copied_files.extend(_copy_mesh_with_sidecars(source_mesh, target_mesh))

        if original_ref.startswith("package://"):
            # The aligned URDF must not reference package://; use the
            # output-tree-relative path so the result is self-contained.
            package_uri_rewrites[original_ref] = str(output_relative)
        elif was_escaped:
            # A non-package ref that escaped: the aligned URDF also gets the
            # rewritten path (the original would still escape on re-resolve).
            package_uri_rewrites[original_ref] = str(output_relative)

    # Apply escape rewrites to the staged copy so it too uses the in-tree
    # paths; this keeps `diff source.urdf aligned.urdf` clean.
    if source_rewrites:
        staged_bytes = target_urdf.read_bytes()
        staged_bytes = rewrite_resolved_package_uris(staged_bytes, source_rewrites)
        target_urdf.write_bytes(staged_bytes)

    return CopiedUrdfAsset(
        urdf=target_urdf,
        copied_files=tuple(copied_files),
        local_mesh_files=tuple(copied_files[1:]),
        unresolved_meshes=tuple(unresolved_meshes),
        escaped_meshes=tuple(escaped_meshes),
        package_uri_rewrites=package_uri_rewrites,
    )


def in_tree_mesh_path(filename: str) -> str:
    """The output-tree-relative path a mesh ref was copied to.

    Copying mirrors a ref at its original relative path, except that
    ``package://<pkg>/<rel>`` loses only its scheme and lands at
    ``<pkg>/<rel>``. Anything reading meshes out of the output tree needs the
    same mapping, so it lives here rather than being restated.

    Maps a ref to where the copy would be; it does not check that the file
    exists. A ref carrying ``../`` segments is returned with them intact,
    because the copy stage rewrites such refs in the staged URDF and what is
    read back is already sanitised. Callers resolving the result against a
    directory must still confine it to that directory.

    Args:
        filename: A ``<mesh>`` ``filename`` attribute value.

    Returns:
        str: The path relative to the output tree.
    """
    if filename.startswith("package://"):
        return filename[len("package://") :]
    # Prefix-wise, so a leading dot survives: ".hidden/a.stl" keeps its dot.
    while filename.startswith("./"):
        filename = filename[2:]
    return filename


def rewrite_resolved_package_uris(
    urdf_bytes: bytes,
    package_uri_rewrites: Mapping[str, str],
) -> bytes:
    """Textually rewrite ``<mesh filename="...">`` refs in URDF bytes.

    Only refs listed in ``package_uri_rewrites`` are rewritten; every other
    filename stays as-is.  The textual (non-parsing) approach keeps the rest
    of the URDF byte-stable so diffs stay reviewable.

    Args:
        urdf_bytes: Raw URDF bytes to rewrite.
        package_uri_rewrites: Mapping from original ``filename`` attribute
            value to its replacement.

    Returns:
        bytes: Rewritten URDF bytes, or the original bytes unchanged if the
        mapping is empty.
    """
    if not package_uri_rewrites:
        return urdf_bytes
    text = urdf_bytes.decode("utf-8")

    def _sub(match: re.Match[str]) -> str:
        prefix, filename, suffix = (
            match.group(1),
            match.group(2),
            match.group(3),
        )
        replacement = package_uri_rewrites.get(filename)
        if replacement is None:
            return match.group(0)
        return f"{prefix}{replacement}{suffix}"

    return _MESH_FILENAME_ATTR_RE.sub(_sub, text).encode("utf-8")


# ---------------------------------------------------------------------------
# Internal resolution logic.
# ---------------------------------------------------------------------------


def _resolve_ref(
    filename: str,
    mesh_root: Path,
) -> tuple[Path, Path, bool, str] | None:
    """Resolve one mesh filename to its source path and output-relative path.

    Returns a 4-tuple ``(source_mesh, output_relative, was_escaped,
    rewritten_ref)`` or ``None`` if the ref could not be found.

    ``was_escaped`` is True when the ref contained leading ``../`` segments
    that would have placed the copy outside ``mesh_root``. In that case
    ``output_relative`` and ``rewritten_ref`` name the sanitised path (the
    leading escaping segments encoded as ``_up_N/``).

    Args:
        filename: The ``filename`` attribute from a ``<mesh>`` element.
        mesh_root: Base directory for resolution, and the tree a ref must not
            escape.

    Returns:
        Resolved (source_mesh, output_relative, was_escaped, rewritten_ref)
        or None.
    """
    if filename.startswith("package://"):
        return _resolve_package_ref(filename, mesh_root)
    return _resolve_relative_ref(filename, mesh_root)


def _resolve_relative_ref(
    filename: str,
    mesh_root: Path,
) -> tuple[Path, Path, bool, str] | None:
    """Resolve a plain relative ref under mesh_root.

    Absolute paths are environment-specific and are treated as unresolvable
    (the user's staged copy can be edited if needed).
    """
    mesh_path = Path(filename)
    if mesh_path.is_absolute():
        return None
    candidate = (mesh_root / mesh_path).resolve()
    if not candidate.exists():
        return None
    return _apply_escape_sanitize(filename, candidate, mesh_root)


def _resolve_package_ref(
    filename: str,
    mesh_root: Path,
) -> tuple[Path, Path, bool, str] | None:
    """Resolve ``package://<pkg>/<rel>`` under ``mesh_root``.

    Strips the scheme and resolves ``<pkg>/<rel>`` under ``mesh_root``.
    No multi-base search.

    The remainder is split on its first ``/`` only, so ``<pkg>`` can be
    ``..`` and name a file above ``mesh_root``. Such a ref is sanitised into
    an ``_up_N/`` prefix by the same rule that covers plain relative refs,
    which keeps the copy inside the output tree.
    """
    without_scheme = filename[len("package://") :]
    parts = without_scheme.split("/", 1)
    if len(parts) != 2:
        return None
    pkg, rel = parts
    if not pkg or not rel:
        return None
    candidate = (mesh_root / pkg / rel).resolve()
    if not candidate.exists():
        return None
    return _apply_escape_sanitize(without_scheme, candidate, mesh_root)


def _apply_escape_sanitize(
    original_ref: str,
    resolved_absolute: Path,
    mesh_root: Path,
) -> tuple[Path, Path, bool, str]:
    """Compute output_relative, detecting and sanitizing escaping refs.

    A ref escapes when ``resolved_absolute`` lies outside ``mesh_root``.
    This happens when the original ref contains ``../`` components that
    step above ``mesh_root``.

    Escape sanitization rule: count the number of ``..``
    components that would take the path above ``mesh_root``, then encode
    them as a ``_up_N/`` prefix.  For example::

        mesh_root = /urdf/robot/
        filename  = ../../shared/base.stl
        resolved  = /urdf/shared/base.stl

    Here ``resolved`` is 1 level above ``mesh_root.parent`` → 2 levels above
    ``mesh_root``.  The canonical filename inside the output tree becomes
    ``_up_2/shared/base.stl``.

    The ``_up_N`` prefix is deterministic and conveys the topology to the
    user without collisions (two refs escaping by different amounts land in
    different sub-trees).

    Args:
        original_ref: The verbatim ``filename`` string from the URDF.
        resolved_absolute: ``(mesh_root / original_ref).resolve()``.
        mesh_root: Base directory.

    Returns:
        ``(source_path, output_relative, was_escaped, rewritten_ref)``
    """
    mesh_root_resolved = mesh_root.resolve()
    try:
        output_relative = resolved_absolute.relative_to(mesh_root_resolved)
        # Path is inside mesh_root — no escape.
        return resolved_absolute, output_relative, False, str(output_relative)
    except ValueError:
        pass

    # The ref escapes.  Count how many ``..`` steps are required to reach
    # ``resolved_absolute`` from ``mesh_root_resolved``, then build the
    # sanitised path.
    #
    # Strategy: use PurePosixPath parts to walk up from mesh_root until we
    # find a common ancestor, then prefix with ``_up_N``.
    root_parts = mesh_root_resolved.parts
    abs_parts = resolved_absolute.parts

    # Find common prefix length.
    common_len = 0
    for rp, ap in zip(root_parts, abs_parts, strict=False):
        if rp != ap:
            break
        common_len += 1

    # Steps up from mesh_root to common ancestor.
    up_count = len(root_parts) - common_len
    # Remainder path below common ancestor.
    tail = abs_parts[common_len:]
    remainder = Path(*tail) if tail else Path()

    sanitised_rel = Path(f"_up_{up_count}") / remainder
    return resolved_absolute, sanitised_rel, True, str(sanitised_rel)


# ---------------------------------------------------------------------------
# Copy helpers.
# ---------------------------------------------------------------------------


def _copy_mesh_with_sidecars(
    source_mesh: Path,
    target_mesh: Path,
) -> list[Path]:
    """Copy ``source_mesh`` and any recognised sidecars to ``target_mesh``.

    Sidecars (``*.mtl``, ``*.png``, ``*.jpg``, ``*.jpeg``) sit beside the
    mesh and are referenced by local relative name inside the mesh file, so
    preserving that layout is what keeps .obj files renderable.
    """
    copied_files: list[Path] = []
    for source_path in [source_mesh, *_iter_sidecars(source_mesh)]:
        relative_path = source_path.relative_to(source_mesh.parent)
        target_path = target_mesh.parent / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        _overwrite_copy(source_path, target_path)
        copied_files.append(target_path)
    return copied_files


def _iter_sidecars(source_mesh: Path) -> list[Path]:
    sidecars: list[Path] = []
    for suffix in (".mtl", ".png", ".jpg", ".jpeg"):
        candidate = source_mesh.with_suffix(suffix)
        if candidate.exists():
            sidecars.append(candidate)
    return sidecars


def _iter_mesh_filenames(urdf_path: Path) -> list[str]:
    """Mesh refs declared in a URDF, in first-seen order.

    Unparseable XML yields no refs: the URDF is still copied verbatim so the
    user can repair it in place, and ``inspect_urdf`` reports the parse error
    as the fatal finding that stops the run.
    """
    try:
        root = ElementTree.parse(urdf_path).getroot()
    except ElementTree.ParseError:
        return []
    filenames: list[str] = []
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if filename and filename not in filenames:
            filenames.append(filename)
    return filenames


def _ensure_owner_writable(path: Path) -> None:
    """Ensure ``path`` has S_IWUSR set, tolerating missing chmod support."""
    try:
        current = path.stat().st_mode
        os.chmod(path, current | stat.S_IWUSR)
    except (OSError, NotImplementedError):
        # Some network filesystems do not support chmod; leave the mode as-is
        # and let a subsequent write surface a clearer error.
        return


def _overwrite_copy(source: Path, target: Path) -> None:
    """`shutil.copy2` that overwrites read-only destinations cleanly.

    Prior align runs can leave read-only mesh copies in the output tree
    (``shutil.copy2`` preserves source mode, and upstream meshes can be
    0444).  Unlinking before copying makes re-runs a clean overwrite
    without a ``PermissionError``.

    Copying a file onto itself is a no-op rather than an unlink: re-running
    against a staged copy stages it to its own path, and the unlink below
    would otherwise delete the user's only copy before it could be read.
    """
    if same_file(source, target):
        return
    if target.exists() or target.is_symlink():
        try:
            target.unlink()
        except PermissionError:
            # Parent directory itself may be read-only; chmod it and retry.
            _ensure_owner_writable(target.parent)
            target.unlink()
    shutil.copy2(source, target)


def same_file(a: Path, b: Path) -> bool:
    """Whether two paths denote the same file, existing or not.

    Compares inodes so hard links and paths differing only by symlinks or
    ``..`` segments are recognised.  ``os.path.samefile`` needs both paths to
    exist, which is not true of a target that has yet to be written, so fall
    back to comparing their resolved forms.
    """
    try:
        return os.path.samefile(a, b)
    except (OSError, ValueError):
        return a.resolve() == b.resolve()


def _is_within(target: Path, root: Path) -> bool:
    """Whether ``target`` lies inside ``root``.

    Both sides are resolved first, so the comparison sees where ``..``
    segments and symlinks actually lead.
    """
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
