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

"""The viewer's static assets must be present and internally consistent.

An ES module graph fails as a unit: one missing file leaves the page blank
with nothing on it to say why. These check the graph closes, against the
assets the installed package serves rather than a copy in the source tree.

These skip until ``scripts/fetch_vendor.py`` has downloaded the third-party
JavaScript.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("flask")

from uniurdf.gui import app as app_module  # noqa: E402

STATIC = Path(app_module.__file__).resolve().parent / "static"
INDEX = STATIC / "index.html"

#: Every asset the page needs, at the path it needs it. Listed here rather
#: than read from scripts/fetch_vendor.py, which would make this agree with
#: that table however wrong the table became.
EXPECTED_ASSETS = (
    "vendor/three.module.min.js",
    "vendor/OrbitControls.js",
    "vendor/GLTFLoader.js",
    "vendor/ColladaLoader.js",
    "vendor/OBJLoader.js",
    "vendor/STLLoader.js",
    "loaders/TGALoader.js",
    "utils/BufferGeometryUtils.js",
)

_vendored = pytest.mark.skipif(
    not (STATIC / "vendor" / "three.module.min.js").is_file(),
    reason="vendored JavaScript absent; run scripts/fetch_vendor.py",
)


def test_the_page_ships() -> None:
    """index.html is first-party, so it is always present."""
    assert INDEX.is_file()


@_vendored
@pytest.mark.parametrize("asset", EXPECTED_ASSETS)
def test_every_expected_asset_is_present(asset: str) -> None:
    assert (STATIC / asset).is_file(), f"missing static asset: {asset}"


@_vendored
def test_every_relative_import_resolves() -> None:
    """The trap: a loader importing a sibling the page never names.

    ColladaLoader and GLTFLoader reach for `../loaders/` and `../utils/`,
    which the import map does not cover, so those two files are reachable
    only by sitting at exactly the right path.
    """
    unresolved = []
    for source in sorted(STATIC.rglob("*.js")):
        text = source.read_text(errors="ignore")
        for specifier in re.findall(r"""from\s+['"](\.\.?/[^'"]+)['"]""", text):
            if not (source.parent / specifier).resolve().is_file():
                unresolved.append(f"{source.name} -> {specifier}")
    assert not unresolved, f"unresolved relative imports: {unresolved}"


@_vendored
def test_the_import_map_targets_exist() -> None:
    """Every bare specifier the page remaps must land on a real file."""
    page = INDEX.read_text()
    block = re.search(r'<script type="importmap">(.*?)</script>', page, re.DOTALL)
    assert block is not None, "index.html declares no import map"

    for target in re.findall(r'"(/static/[^"]*)"', block.group(1)):
        relative = target[len("/static/") :]
        # A trailing slash is a prefix mapping, so the directory must exist.
        expected = STATIC / relative
        if target.endswith("/"):
            assert expected.is_dir(), f"import map prefix missing: {target}"
        else:
            assert expected.is_file(), f"import map target missing: {target}"


def test_the_page_fetches_no_remote_assets() -> None:
    """The viewer must work offline and behind a proxy."""
    page = INDEX.read_text()
    assert 'src="http' not in page
    assert 'href="http' not in page
    assert "cdn." not in page
