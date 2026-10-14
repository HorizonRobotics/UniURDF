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

"""Download the browser viewer's third-party JavaScript.

The viewer renders with three.js, which is too large to keep in the
repository. This fetches the pinned release into ``gui/static/`` and
checks every file against a recorded SHA-256, so a tampered or truncated
download fails loudly instead of producing a viewer that half works.

Run it once after cloning::

    python scripts/fetch_vendor.py

Files already present with the right hash are left alone, so re-running is
cheap. Pass ``--force`` to re-download regardless.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

#: Pinned three.js release. Changing this means regenerating every hash
#: below; ``--print-hashes`` does that.
THREE_VERSION = "0.160.0"

_BASE = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}"

#: Destination path, relative to ``static/`` -> (source URL, SHA-256).
#:
#: The layout is load-bearing. ``index.html``'s import map points
#: ``three/addons/`` at ``vendor/``, so the files it names directly live
#: there. But two of those import each other by relative path --
#: ``ColladaLoader`` reaches for ``../loaders/TGALoader.js`` and
#: ``GLTFLoader`` for ``../utils/BufferGeometryUtils.js`` -- so those two sit
#: beside ``vendor/``, not inside it. ES modules fail as a unit: one 404 here
#: leaves the viewer stuck with no error on the page.
VENDOR_FILES: dict[str, tuple[str, str]] = {
    "vendor/three.module.min.js": (
        f"{_BASE}/build/three.module.min.js",
        "3e690ac7d180b0aadf0891bea39eec643e29e2d3e75c99b18689518665f69ba6",
    ),
    "vendor/OrbitControls.js": (
        f"{_BASE}/examples/jsm/controls/OrbitControls.js",
        "5a44a9e86a2a0fb11933eed69bc2cd33c76a496854c1aed6ed776efa87d7b064",
    ),
    "vendor/GLTFLoader.js": (
        f"{_BASE}/examples/jsm/loaders/GLTFLoader.js",
        "d073b438e6a07e1359741dd5d6c76c953420cc0d4fd84eb1bdde94315540e6a3",
    ),
    "vendor/ColladaLoader.js": (
        f"{_BASE}/examples/jsm/loaders/ColladaLoader.js",
        "f188ae4f2891b9c689ab9b7c5ece2722e149f5d69b8ec6d43a474e510b3c88b1",
    ),
    "vendor/OBJLoader.js": (
        f"{_BASE}/examples/jsm/loaders/OBJLoader.js",
        "022e0334f837c60506276e136faf3e54b21b20aea672eed4a0c50651d4fc0a5d",
    ),
    "vendor/STLLoader.js": (
        f"{_BASE}/examples/jsm/loaders/STLLoader.js",
        "896d006a48b8f125385a485ccae154dadee801a953f0b45ceffe7ddd8a29ca93",
    ),
    "loaders/TGALoader.js": (
        f"{_BASE}/examples/jsm/loaders/TGALoader.js",
        "ce938459ed99d5234f9665270cca8552464d0d1703a094e49760631e6b14c7a3",
    ),
    "utils/BufferGeometryUtils.js": (
        f"{_BASE}/examples/jsm/utils/BufferGeometryUtils.js",
        "9be041e96308775d00e2695cc607645b9a9b64fd7c0e759dd8f7c00a8d92becb",
    ),
}

STATIC_DIR = (
    Path(__file__).resolve().parent.parent / "src" / "uniurdf" / "gui" / "static"
)


def _fetch(url: str) -> bytes:
    """Return the bytes at ``url``.

    Args:
        url (str): The URL to read.

    Returns:
        bytes: The response body.
    """
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        return response.read()


def _digest(payload: bytes) -> str:
    """Return the SHA-256 hex digest of ``payload``.

    Args:
        payload (bytes): Bytes to hash.

    Returns:
        str: Lowercase hex digest.
    """
    return hashlib.sha256(payload).hexdigest()


def print_hashes() -> int:
    """Download every file and print a pastable hash table.

    Use this when bumping :data:`THREE_VERSION`. Read the diff before
    committing the result -- the point of pinning is lost if the hashes are
    refreshed without anyone looking at what changed.

    Returns:
        int: Process exit status.
    """
    for name, (url, _) in VENDOR_FILES.items():
        print(
            f'    "{name}": (\n        f"{url}",\n'
            f'        "{_digest(_fetch(url))}",\n    ),'
        )
    return 0


def fetch_all(force: bool = False) -> int:
    """Download every vendored file, verifying its hash.

    Args:
        force (bool): Re-download even when the local copy already matches.

    Returns:
        int: ``0`` on success, ``1`` if any file failed verification.
    """
    failed = False

    for name, (url, expected) in VENDOR_FILES.items():
        target = STATIC_DIR / name
        target.parent.mkdir(parents=True, exist_ok=True)

        if not force and target.exists():
            if not expected or _digest(target.read_bytes()) == expected:
                print(f"ok       {name} (cached)")
                continue

        try:
            payload = _fetch(url)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            # A proxy, a DNS failure, or a timeout is an ordinary condition
            # here; the message names the file that did not arrive.
            print(f"FAIL     {name}\n           {error}", file=sys.stderr)
            failed = True
            continue
        actual = _digest(payload)

        if expected and actual != expected:
            print(
                f"FAIL     {name}\n"
                f"           expected {expected}\n"
                f"           got      {actual}",
                file=sys.stderr,
            )
            failed = True
            continue

        target.write_bytes(payload)
        note = "" if expected else "  (unpinned: run --print-hashes)"
        print(f"fetched  {name}  {len(payload):>8,} bytes{note}")

    if failed:
        print(
            "\nOne or more files did not match their recorded hash. "
            "Nothing was written for those files.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv (list[str], optional): Arguments to parse.

    Returns:
        int: Process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even when the local copy already matches",
    )
    parser.add_argument(
        "--print-hashes",
        action="store_true",
        help="download and print a hash table, for bumping the pinned version",
    )
    args = parser.parse_args(argv)

    if args.print_hashes:
        return print_hashes()
    return fetch_all(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
