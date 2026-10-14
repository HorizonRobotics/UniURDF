# uniurdf

Align robot URDFs onto a shared coordinate convention, and see what changed.

Two things happen to a robot: every actuated joint axis is rotated to **+Z**,
and a **tool centre point** link is attached where you say. Everything else is
preserved — most importantly, the robot does not move. Visuals stay exactly
where they were in world space; only the frames turn.

Everything the tool does is declared in one YAML file. No robot, dataset, or
embodiment knowledge lives in the code, so you can predict the output URDF
from the spec alone.

## Install

```bash
pip install -e .                    # core
pip install -e ".[gui]"             # plus the browser viewer
python scripts/fetch_vendor.py      # the viewer's three.js, pinned
```

The core is **numpy and PyYAML**. No torch, no ROS, no simulator. The viewer
adds Flask and needs its JavaScript downloaded once, by the third command.

## Try it

A complete example robot ships with the package, built from primitive
geometry so it needs no mesh files:

```bash
uniurdf align src/uniurdf/example/example_arm.yaml --output /tmp/example
```

That robot covers the cases alignment has to get right: one joint spins about
`+X`, one about `-Z` (anti-parallel, where a naive rotation fails), one is
already `+Z` and must not move, one mimics another, and a camera frame has to
survive. Read `src/uniurdf/example/example_arm.yaml`: every spec field
is documented there.

## What you get

One run writes five files into the output directory, alongside a copy of any
mesh tree the URDF references:

| File | What it is |
|---|---|
| `<stem>.source.urdf` | Verbatim copy of the input. Edit this to repair a broken URDF, point `source_urdf` at it, and re-run. |
| `<stem>.urdf` | The aligned robot: every actuated joint axis on `+Z`, TCP and camera-compat links attached. |
| `<stem>.pruned.urdf` | The same robot cut down to the links `chains` and `tcp` name, plus their ancestors. |
| `<stem>.inventory.json` | Joint order and indices for `<stem>.urdf`, with the chains, TCPs and camera frames resolved to real link names. |
| `<stem>.pruned.inventory.json` | The same, for the pruned URDF. |

The two inventories differ whenever pruning drops a movable joint, because
dropping one renumbers everything after it. **Index against the inventory
that matches the URDF you loaded** — the CLI says so explicitly when the
numbering diverges.

`inventory.json` is a JSON object with `root_link`, `joint_order` (the
positional joint vector), `chains` (each with `joints[].name`/`.index` and
`links`), `tcp` (`link`, `base_link`), `finger_links`, and
`camera_references` (`link`, `compat_link` — `null` when the frame did not
move and needs no alias).

## What is preserved

Alignment rotates link frames and compensates every `<visual>` origin, so
each visual keeps its world pose. `verify()` measures exactly that, and the
CLI **refuses to write** when it fails:

```
verified 3 configuration(s): link pos 0.00e+00 m, mesh pos 0.00e+00 m,
                             mesh rot 9.92e-15 deg
verify: OK
```

Pass `--force` to write anyway when you want to inspect a bad result; the
exit status still reports the failure.

Exit codes: `0` success, `1` bad input or config, `2` malformed command line
(from `argparse`), `3` verification failed.

## Seeing it

```bash
uniurdf view src/uniurdf/example/example_arm.yaml
```

Opens a viewer at <http://127.0.0.1:7890> showing the robot before and after
alignment, with a slider per joint and a per-link delta table. Edit the spec
in the page and export it when it looks right.

Forward kinematics runs on the server, so the viewer draws the same numbers
the exported URDF was built from. Working over SSH? Forward the port:

```bash
ssh -L 7890:127.0.0.1:7890 <host>
```

It binds `127.0.0.1` by default. The viewer writes files anywhere under the
output directory, so `--host 0.0.0.0` gives anyone who can reach the port
that access.

## Using it as a library

The CLI is a thin adapter over the Python API, so anything it does is
reachable from code:

```python
from uniurdf import load_project, run_alignment

outcome = run_alignment(load_project("robot.yaml"))
if outcome.ok:
    print(outcome.bundle.write.urdf)
else:
    print(outcome.status.value, outcome.error)
```

This is what `uniurdf align` runs, refusals included. The individual stages
are available too — see [docs/api.md](docs/api.md) — and they apply no
refusals of their own.

A single URDF can also be loaded directly, with no spec and no staging:

```python
from uniurdf import load_robot

robot = load_robot("robot.urdf")
poses = robot.fk({"joint1": 0.5})
```

## Re-running is safe

Alignment is idempotent. The angles it applied are recorded inside the output
URDF, so running the tool on its own result reaches a fixed point rather than
rotating a second time. Re-aligning a file produced by an earlier release
reads its record and migrates it.

## Mesh paths are yours to fix

When a mesh reference cannot be resolved, the tool reports it and leaves it
verbatim. It does not guess, search, or rewrite paths — correcting them in the
source URDF is the user's job. The staged copy under the output directory
exists to be edited: fix it, point `source_urdf` at the corrected file, and
re-run.

## Development

Install the tools the checks need:

```bash
pip install -e ".[gui]"
pip install --group dev              # pytest, ruff, pyright, hatchling (pip 25.1+)
```

Then:

```bash
make check             # lint, types, tests, and the wheel
```

Or one at a time: `make lint`, `make types`, `make test`, `make wheel`.

`make wheel` is the one `pytest` cannot replace. Package data is found
relative to its module at run time, so a file present in the checkout can
still be missing from the built wheel while every test passes. It builds the
wheel, unpacks it somewhere unrelated, and asks the installed package to
align the example robot and serve every viewer asset.

The suite is hermetic. Tests that want real robots read
`UNIURDF_TEST_URDF_DIR` and skip without it; the viewer's asset tests skip
until `make vendor` has run.

Three tests check that the joint indices in `inventory.json` point at the
joints another URDF library would find at those positions. They need
`pytorch_kinematics`, which is a large install because it brings PyTorch, so
it is optional and those three skip without it:

```bash
pip install --group crosscheck
```

Run them before publishing an aligned URDF anyone else will index against.

## License

Apache-2.0. See [LICENSE](LICENSE).
