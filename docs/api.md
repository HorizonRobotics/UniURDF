# Python API

Everything the CLI does is reachable from Python; the CLI parses arguments
and formats results, and holds no alignment logic of its own.

## The whole workflow

`run_alignment` is the path the CLI and the viewer both take. It carries the
two refusals with it, so every caller enforces the same rules:

```python
from uniurdf import load_project, run_alignment

outcome = run_alignment(load_project("robot.yaml"))
if outcome.ok:
    print(outcome.bundle.write.urdf)
else:
    print(outcome.status.value, outcome.error)
```

Failures arrive as a status on the returned `AlignmentOutcome`, and nothing
is written to stdout or stderr, so a server can call it as readily as a
terminal.

| `AlignmentStatus` | Meaning |
|---|---|
| `OK` | Aligned, verified, written. |
| `FATAL_DIAGNOSTICS` | The URDF has a fatal structural finding. Nothing aligned. |
| `SPEC_ERROR` | The spec is malformed or disagrees with the URDF. |
| `VERIFY_FAILED` | Geometry moved, or nothing was compared. Written only with `force`. |
| `WRITE_FAILED` | Alignment succeeded; the output could not be written. |

Pass `force=True` to write despite a failed check. The status still reports
the failure, and `outcome.forced` is `True`.

## The stages, individually

Each step is available on its own, for a caller that needs to interleave its
own work — inspecting diagnostics before aligning, say. **These apply no
refusals**: a URDF with fatal findings will align, and geometry that fails
its check will be written.

```python
from uniurdf import load_project

project = load_project("robot.yaml", output=None)
staged = project.stage()
aligned = staged.align(project.spec)
report = aligned.verify()
if report.ok:
    bundle = aligned.write_bundle(project.output, project.stem)
```

| Step | Returns | Does |
|---|---|---|
| `load_project` | `Project` | Resolves the spec's paths. |
| `.stage()` | `StagedRobot` | Copies the URDF and its meshes into the output tree, then runs structural diagnostics. |
| `.align(spec)` | `AlignedRobot` | Normalises axes, attaches TCPs, emits compat links. |
| `.verify()` | `VerifyReport` | Measures whether any geometry moved. |
| `.write_bundle(...)` | `BundleReport` | Writes the URDF, the pruned URDF, and both inventories. |

`StagedRobot` carries `.diagnostics`, `.asset`, and `.staged_path`. Staging
before aligning is what makes the two-pass workflow possible: when a URDF has
unresolved mesh references or structural defects, edit the staged copy, point
`source_urdf` at it, and run again.

## Loading a single URDF

When there is no spec and no staging directory — for forward kinematics, or
to inspect a file — load the robot directly:

```python
from uniurdf import load_robot

robot = load_robot("robot.urdf")  # also accepts bytes
poses = robot.fk({"joint1": 0.5})  # link name -> 4x4 world transform
robot.joint_names
robot.link_names
```

## Diagnostics without alignment

```python
from uniurdf import inspect_urdf

diagnostics = inspect_urdf(open("robot.urdf", "rb").read())
for message in diagnostics.messages():
    print(message)
if diagnostics.is_fatal:
    ...
```

## Errors

`SpecError` covers a malformed or inconsistent spec, including one that names
a joint or link the URDF does not contain — fix the spec and re-run.

`PruneError` reports that pruning moved a link it kept. Pruning deletes
elements and rewrites none, so this cannot happen for any input: it means the
keep-set is wrong. It is a bug in this tool, not something to fix in your URDF
or widen a tolerance for — please report it with the spec that triggered it.

Both are exported at the top level.
