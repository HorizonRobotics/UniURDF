# What alignment does

## What changes

Every **actuated joint axis** is rotated to point along `+Z`. A joint whose
axis already points that way is left untouched, byte for byte.

Rotating a joint's axis changes the orientation of its child link's frame, so
that rotation is conjugated through the subtree: the child joint origins,
and the `<visual>`, `<collision>` and `<inertial>` origins of the links, are
all re-expressed in the new frame.

Two kinds of link are added:

- `<base_link>_tcp` — a tool centre point, wherever the spec declares one.
- `<link>_camera_mount_compat` — a compatibility frame, emitted **only** where
  alignment actually rotated a declared camera reference. A calibrated
  extrinsic that pointed at the original link keeps pointing at the right
  place. Where alignment did not move the frame, no compat link is created and
  the plain link name stays correct.

## What is preserved

**The robot does not move.** Concretely, across every checked configuration:

| Quantity | Requirement |
|---|---|
| Link world position | unchanged |
| Visual world position | unchanged |
| Visual world orientation | unchanged |
| Link world *orientation* | **may change** — this is the point |

The last row is the one to internalise. Alignment exists to turn frames, so
link orientations changing is the tool working, not failing. What must not
change is where any geometry actually sits in the world.

`verify()` checks all visuals — mesh, box, cylinder, sphere. Alignment
rewrites each one's origin alike, so checking only meshes would leave a robot
built from primitives transformed but unverified.

## Idempotence

Alignment normalises every actuated axis to `+Z`, which means a second pass
over its own output cannot tell a joint that was always `+Z` from one that was
rotated 90° to get there. Applying the rotation again would silently move the
robot.

The applied angles are therefore recorded inside the output URDF, nested in a
`<gazebo>` wrapper so URDF parsers pass them through without warning. On the
next run they are read back and only the difference is applied, so re-running
the tool on the directory it just wrote reaches a fixed point.

Records written by earlier releases, under a different element name or as a
direct child of `<robot>`, are read too and rewritten in the current form.

## Pruning

The pruned URDF keeps the links a spec names plus their ancestors — and the
links alignment itself emitted, which hang *below* their base link and would
otherwise be deleted by a plain ancestor closure.

Dropping movable joints **renumbers the joint vector**. An index taken against
the full URDF addresses a different joint in the pruned one; the robot still
moves, just not as intended. Index against `.pruned.inventory.json`, which is
written alongside for that purpose.
