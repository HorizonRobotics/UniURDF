# Third-party JavaScript

These files are **not** part of uniurdf. Download them with:

```bash
python scripts/fetch_vendor.py
```

That script pins the version and checks every file against a recorded
SHA-256, so what lands here is exactly what was reviewed.

## What is fetched

All eight files are [three.js](https://threejs.org/) **r160** (npm
`three@0.160.0`), retrieved from `cdn.jsdelivr.net`.

Copyright © 2010-2023 three.js authors. Licensed under the
[MIT License](https://github.com/mrdoob/three.js/blob/dev/LICENSE).

| File | Source path in the three.js package |
|---|---|
| `vendor/three.module.min.js` | `build/three.module.min.js` |
| `vendor/OrbitControls.js` | `examples/jsm/controls/OrbitControls.js` |
| `vendor/GLTFLoader.js` | `examples/jsm/loaders/GLTFLoader.js` |
| `vendor/ColladaLoader.js` | `examples/jsm/loaders/ColladaLoader.js` |
| `vendor/OBJLoader.js` | `examples/jsm/loaders/OBJLoader.js` |
| `vendor/STLLoader.js` | `examples/jsm/loaders/STLLoader.js` |
| `loaders/TGALoader.js` | `examples/jsm/loaders/TGALoader.js` |
| `utils/BufferGeometryUtils.js` | `examples/jsm/utils/BufferGeometryUtils.js` |

Only `three.module.min.js` carries an inline licence header; this file is the
attribution for the rest.

## Why two files sit outside `vendor/`

`index.html`'s import map points `three/addons/` at `vendor/`, so everything
it imports by name lives there. But two of those import a sibling by relative
path:

- `ColladaLoader.js` → `../loaders/TGALoader.js`
- `GLTFLoader.js` → `../utils/BufferGeometryUtils.js`

so those two must sit *beside* `vendor/`, not inside it. ES modules fail as a
unit: one missing file here leaves the viewer blank with nothing on the page
to say why.

## Upgrading

Change `THREE_VERSION` in `scripts/fetch_vendor.py`, run it with
`--print-hashes`, and paste the result back. Read that diff — pinning buys
nothing if the hashes are refreshed without anyone looking at what moved.
