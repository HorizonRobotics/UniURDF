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

"""The package's own examples must run.

A docstring example is the first code a reader tries, and one that raises
teaches the wrong interface. These execute the documented flows against the
shipped example robot rather than asserting on their text, so an interface
change that invalidates an example fails here.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import uniurdf
from conftest import EXAMPLE_SPEC
from uniurdf import load_project, load_robot, run_alignment


def test_the_headline_example_runs(tmp_path: Path) -> None:
    """The first example in ``uniurdf.__doc__``, executed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outcome = run_alignment(load_project(EXAMPLE_SPEC, output=tmp_path))

    assert outcome.ok, f"{outcome.status.value}: {outcome.error}"
    assert outcome.bundle is not None
    assert outcome.bundle.write.urdf.exists()


def test_the_staged_pipeline_example_runs(tmp_path: Path) -> None:
    """The second example in ``uniurdf.__doc__``, executed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        project = load_project(EXAMPLE_SPEC, output=tmp_path)
        staged = project.stage()
        aligned = staged.align(project.spec)
        assert aligned.verify().ok
        bundle = aligned.write_bundle(project.output, project.stem)

    assert bundle.write.urdf.exists()
    assert bundle.inventory.exists()


def test_the_direct_load_example_runs() -> None:
    """The second example in ``uniurdf.__doc__``, executed."""
    robot = load_robot(EXAMPLE_SPEC.parent / "example_arm.urdf")
    poses = robot.fk({"left_shoulder": 0.5})
    assert poses
    assert robot.joint_names
    assert robot.link_names


def test_every_exported_name_resolves() -> None:
    """``__all__`` must not promise a name the package does not carry."""
    missing = [name for name in uniurdf.__all__ if not hasattr(uniurdf, name)]
    assert not missing, f"exported but absent: {missing}"
