# Spec reference

One YAML file declares everything the tool does to a robot. The shipped
[`example_arm.yaml`](../src/uniurdf/example/example_arm.yaml) is a working
spec with every field annotated; this page is the same information as
reference rather than tutorial.

## Path resolution

Relative paths resolve from **the directory containing the spec file**, never
from the working directory, so a run gives the same result from anywhere and
a spec moves with the tree it names.

`--output` is the one exception: it is a run-time choice, so it resolves
against the current working directory.

## Fields

### `source_urdf` (required)

Path to the URDF being aligned.

### `mesh_root` (optional)

Base directory that mesh references resolve against:

- relative ref → `mesh_root / ref`
- `package://<pkg>/<rel>` → `mesh_root / <pkg> / <rel>`

Defaults to the directory containing `source_urdf`. Set it only when the URDF
lives in a subdirectory but its mesh paths are written relative to a parent.

### `output` (optional)

Where to write the aligned URDF and the copied mesh tree. Defaults to
`workspace/aligned_urdf/<source_urdf stem>` beside the spec file.

### `chains` (optional)

One inner list per arm, each naming the movable joints from root to tip.
This groups the viewer's sliders and selects which frames get labelled.

The export **always aligns every actuated joint**, whether or not it appears
here — that is `<stem>.urdf`. `<stem>.pruned.urdf` is the exception: it keeps
the links these joints drive plus their ancestors and drops the rest, so a
joint left out here is absent from the pruned export. Omit the field to treat
all movable joints as a single chain.

### `tcp` (optional)

Positionally paired with `chains`; use `null` for an arm that needs no tool
centre point. Each entry adds a fixed child link `<base_link>_tcp`.

| Key | Meaning |
|---|---|
| `base_link` | An existing link to attach to. Never renamed. |
| `xyz` | Offset from `base_link`, in **metres**. |
| `rpy` | Orientation relative to `base_link`, in **degrees**. |
| `joints` | Gripper joints this TCP moves, for the viewer. |

Note the unit split: `xyz` is metres, `rpy` is degrees.

### `camera_references` (optional)

Links whose world pose must survive axis normalisation. Accepts either a bare
link name or a mapping with a `link` key.

A `<link>_camera_mount_compat` child is emitted **only where alignment
rotated the frame**. This is per-robot: a link below a normalised joint gets
one; a link hanging off fixed joints does not.

### `rot_z` (optional)

Extra rotation about a joint's own `+Z`, applied after normalisation, in
**degrees**, keyed by joint name.

Normalisation fixes the axis *direction* but leaves the roll about that axis
free. Use this where a robot needs a particular roll. Applied angles are
recorded in the output, so re-running does not apply them twice.
