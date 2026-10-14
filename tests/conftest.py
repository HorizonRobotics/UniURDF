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

"""Shared fixtures for uniurdf tests.

Synthetic URDFs keep the unit tests hermetic. The example robot shipped in
``config/`` is built from primitive geometry, so tests that need a complete
robot use it rather than depending on an asset mount.
"""

import os
from pathlib import Path

import pytest

import uniurdf

# The package under test, located through the import rather than by walking
# up from this file, so the tests exercise whatever is installed. Several
# tests glob ``PACKAGE_DIR / "example"``, and a glob over a missing directory
# yields nothing and passes vacuously, so a stale path would silently retire
# whole test classes. The check below fails collection instead.
PACKAGE_DIR = Path(uniurdf.__file__).resolve().parent

if not (PACKAGE_DIR / "example").is_dir():
    raise RuntimeError(
        f"uniurdf example directory not found under {PACKAGE_DIR}. "
        "The package data is missing from the installation."
    )

#: The runnable example robot, used wherever a test needs a complete robot
#: with visuals rather than a synthetic stub.
EXAMPLE_SPEC = PACKAGE_DIR / "example" / "example_arm.yaml"

#: Repo-root ``config/``, holding the robot specs a branch adds beyond the
#: packaged example. Absent on a branch that ships only the example.
_REPO_CONFIG = Path(__file__).resolve().parent.parent / "config"

#: Every directory the corpus sweeps read: the packaged example, and the
#: specs a branch carries. This is what decides their reach -- one robot
#: where only the example ships, the full fleet on a branch with `config/`.
SPEC_DIRS = tuple(
    directory
    for directory in (PACKAGE_DIR / "example", _REPO_CONFIG)
    if directory.is_dir()
)


def iter_specs() -> list[Path]:
    """Every spec the corpus sweeps should check, in a stable order.

    ``example.yaml`` and ``debug.yaml`` are placeholders naming no real
    robot, and are skipped wherever they appear.
    """
    found: list[Path] = []
    for directory in SPEC_DIRS:
        for path in sorted(directory.glob("*.yaml")):
            if path.stem in {"example", "debug"}:
                continue
            found.append(path)
    return found


SIMPLE_URDF = """<?xml version="1.0"?>
<robot name="simple">
  <link name="base"/>
  <link name="upper"/>
  <link name="tool"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="upper"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="1"/>
  </joint>
  <joint name="wrist" type="prismatic">
    <parent link="upper"/>
    <child link="tool"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.3" effort="10" velocity="1"/>
  </joint>
</robot>
"""

MIMIC_URDF = """<?xml version="1.0"?>
<robot name="mimic">
  <link name="base"/>
  <link name="finger_a"/>
  <link name="finger_b"/>
  <joint name="drive" type="revolute">
    <parent link="base"/>
    <child link="finger_a"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="follow" type="revolute">
    <parent link="base"/>
    <child link="finger_b"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <mimic joint="drive" multiplier="-1.0" offset="0.25"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>
"""

ORPHAN_URDF = """<?xml version="1.0"?>
<robot name="orphan">
  <link name="base"/>
  <link name="attached"/>
  <link name="floating"/>
  <link name="floating_child"/>
  <joint name="real" type="fixed">
    <parent link="base"/>
    <child link="attached"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="detached" type="revolute">
    <parent link="floating"/>
    <child link="floating_child"/>
    <origin xyz="0 1 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
"""


@pytest.fixture()
def simple_urdf(tmp_path: Path) -> Path:
    """A two-joint arm: one revolute, one prismatic."""
    path = tmp_path / "simple.urdf"
    path.write_text(SIMPLE_URDF)
    return path


@pytest.fixture()
def mimic_urdf(tmp_path: Path) -> Path:
    """A gripper whose second finger mimics the first."""
    path = tmp_path / "mimic.urdf"
    path.write_text(MIMIC_URDF)
    return path


@pytest.fixture()
def orphan_urdf(tmp_path: Path) -> Path:
    """A URDF with a subtree disconnected from the root."""
    path = tmp_path / "orphan.urdf"
    path.write_text(ORPHAN_URDF)
    return path


@pytest.fixture(scope="session")
def urdf_root() -> Path:
    """Root of an external robot asset tree, or skip when unavailable.

    Only tests that need real robots use this. Set ``UNIURDF_TEST_URDF_DIR``
    to point at a tree of ``.urdf`` files; without it these tests skip, so the
    suite stays green on a machine that has no such tree.

    Callers must take the root from here rather than deriving it from a URDF's
    parents: assets sit at varying depths, so any fixed number of `.parent`
    hops is wrong for some robot and silently resolves to a directory that
    contains nothing -- which turns every test that uses it into a skip.
    """
    configured = os.environ.get("UNIURDF_TEST_URDF_DIR")
    if not configured:
        pytest.skip("UNIURDF_TEST_URDF_DIR is not set")
    root = Path(configured)
    if not root.is_dir():
        pytest.skip(f"URDF asset directory not available: {root}")
    return root


@pytest.fixture(scope="session")
def real_urdfs(urdf_root: Path) -> list[Path]:
    """Every real robot URDF, or skip when the asset tree is unavailable.

    The tree is genuinely absent in most environments, so skipping is correct
    here rather than a way of hiding a failure.
    """
    paths = sorted(urdf_root.rglob("*.urdf"))
    if not paths:
        pytest.skip(f"no URDF files under {urdf_root}")
    return paths
