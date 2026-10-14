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

"""Check the built wheel the way a user meets it.

The test suite imports from the checkout, where a file can be present and
still be missing from the wheel. This unpacks the built wheel somewhere
unrelated, puts only that on the path, and runs the CLI and the viewer
against it -- so package data that never made it into the build fails here
rather than for whoever installs it.

Run via ``make wheel``. Exits non-zero, naming what was missing.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Run inside the unpacked wheel: align the packaged example robot, then
#: serve every viewer asset. Kept as source so it runs in a clean
#: interpreter whose only import path is the wheel.
_CHECK = r"""
import pathlib, sys, tempfile, warnings

import uniurdf
from uniurdf.project import load_project, run_alignment

spec = pathlib.Path(uniurdf.__file__).parent / "example" / "example_arm.yaml"
if not spec.is_file():
    print(f"FAIL package data missing: {spec}", file=sys.stderr)
    raise SystemExit(1)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    outcome = run_alignment(
        load_project(spec, output=pathlib.Path(tempfile.mkdtemp()))
    )
if not outcome.ok:
    print(f"FAIL alignment: {outcome.status.value}: {outcome.error}",
          file=sys.stderr)
    raise SystemExit(1)
print(f"ok   aligned the packaged example ({outcome.verify.compared_links} "
      f"link comparisons)")

try:
    import flask  # noqa: F401
except ImportError:
    # Serving the assets is the half of this check that pytest cannot do, so
    # skipping it would report a verified wheel on the strength of the half
    # that ran.
    print(
        "FAIL viewer assets unverified: flask is not installed. Install the "
        "extra with `pip install -e \".[gui]\"`, then re-run.",
        file=sys.stderr,
    )
    raise SystemExit(1)

from uniurdf.gui.app import Session, create_app

REQUIRED = [
    "vendor/three.module.min.js", "vendor/OrbitControls.js",
    "vendor/GLTFLoader.js", "vendor/ColladaLoader.js",
    "vendor/OBJLoader.js", "vendor/STLLoader.js",
    "loaders/TGALoader.js", "utils/BufferGeometryUtils.js",
]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    project = load_project(spec, output=pathlib.Path(tempfile.mkdtemp()))
    staged = project.stage()
client = create_app(Session(project=project, staged=staged)).test_client()

failures = [f"GET / -> {r.status_code}"
            for r in [client.get("/")] if r.status_code != 200]
for asset in REQUIRED:
    status = client.get(f"/static/{asset}").status_code
    if status != 200:
        failures.append(f"GET /static/{asset} -> {status}")

if failures:
    print(file=sys.stderr)
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    print("\nAn ES module graph fails as a unit, so the page would render "
          "nothing.", file=sys.stderr)
    raise SystemExit(1)
print(f"ok   viewer serves the page and all {len(REQUIRED)} assets")
"""


def main() -> int:
    """Unpack the newest built wheel and exercise it.

    Returns:
        int: ``0`` when the wheel is usable, ``1`` otherwise.
    """
    wheels = sorted(glob.glob(str(REPO / "build" / "wheel" / "*.whl")))
    if not wheels:
        print("FAIL no wheel under build/wheel; run `make wheel`", file=sys.stderr)
        return 1

    wheel = wheels[-1]
    unpacked = tempfile.mkdtemp(prefix="uniurdf-wheel-")
    zipfile.ZipFile(wheel).extractall(unpacked)
    print(f"     {Path(wheel).name}")

    # Run outside the repository: from the checkout, the source tree would
    # satisfy imports the wheel does not.
    result = subprocess.run(
        [sys.executable, "-c", _CHECK],
        cwd=tempfile.gettempdir(),
        env={**os.environ, "PYTHONPATH": unpacked},
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
