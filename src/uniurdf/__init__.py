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

"""Align robot URDFs onto a shared coordinate convention.

Two things happen to a robot: every actuated joint axis is rotated to ``+Z``,
and a tool centre point link is attached where the spec says. Everything else
is preserved -- most importantly, the robot does not move. Meshes stay exactly
where they were in world space; only the frames turn.

Everything the tool does is declared in one YAML file. No robot, dataset, or
embodiment knowledge lives in this package, so the output URDF is predictable
from the spec alone.

The entry point is :func:`run_alignment`, which resolves the spec, stages
the URDF, aligns it, checks the geometry, and writes -- refusing at the first
step that says no::

    from uniurdf import load_project, run_alignment

    outcome = run_alignment(load_project("robot.yaml"))
    if outcome.ok:
        print(outcome.bundle.write.urdf)
    else:
        print(outcome.status.value, outcome.error)

Failures arrive as a status on the returned outcome, and nothing is
written to stdout or stderr, so a command line, a server, or a batch job
can each render the result their own way.

The stages are also available individually, for a caller that needs to
interleave its own work. They apply no refusals: a URDF with fatal findings
will align, and geometry that fails its check will be written::

    project = load_project("robot.yaml")
    staged = project.stage()
    aligned = staged.align(project.spec)
    if aligned.verify().ok:
        aligned.write_bundle(project.output, project.stem)

A single URDF can also be loaded directly, without a spec or a staging
directory, which is the smallest useful unit::

    from uniurdf import load_robot

    robot = load_robot("robot.urdf")
    poses = robot.fk({"joint1": 0.5})

The command line (:mod:`uniurdf.cli`) and the viewer are adapters over these
same calls, so anything they do is reachable from Python.
"""

from uniurdf.diagnostics import Diagnostics, Finding, inspect_urdf
from uniurdf.kinematics.chain import Chain, Joint, build_chain
from uniurdf.project import (
    AlignmentOutcome,
    AlignmentStatus,
    Project,
    StagedRobot,
    load_project,
    run_alignment,
    stage,
)
from uniurdf.prune import PruneError, PruneReport
from uniurdf.robot import (
    AlignedRobot,
    AlignmentReport,
    BundleReport,
    Robot,
    VerifyReport,
    WriteReport,
    load_robot,
)
from uniurdf.spec import (
    TCP_LINK_SUFFIX,
    AlignmentSpec,
    CameraReference,
    SpecError,
    TcpSpec,
    load_spec,
    parse_spec,
)

__version__ = "0.1.0"

__all__ = [
    "TCP_LINK_SUFFIX",
    "AlignedRobot",
    "AlignmentOutcome",
    "AlignmentReport",
    "AlignmentSpec",
    "AlignmentStatus",
    "BundleReport",
    "CameraReference",
    "Chain",
    "Diagnostics",
    "Finding",
    "Joint",
    "Project",
    "PruneError",
    "PruneReport",
    "Robot",
    "SpecError",
    "StagedRobot",
    "TcpSpec",
    "VerifyReport",
    "WriteReport",
    "__version__",
    "build_chain",
    "inspect_urdf",
    "load_project",
    "load_robot",
    "load_spec",
    "parse_spec",
    "run_alignment",
    "stage",
]
