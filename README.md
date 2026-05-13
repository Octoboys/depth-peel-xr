# Depth Peel XR

A Blender 5.x add-on that bakes a 3D environment into a small set of **depth-peeled layered impostors** optimised for XR viewing through a bounded portal from a fixed head-plane. Exports USD for consumption by external real-time engines.

## What it does

Given a Blender scene and a viewing rig (portal + head-plane), the add-on:

1. Probes scene depth through the portal to derive N depth-layer boundaries (user-count-driven or histogram-driven).
2. Renders an NxN grid of views across the head-plane for each layer, one clipped depth slab at a time.
3. Consolidates multi-view captures into one canonical RGBA + depth per layer, filling disocclusions visible from head-plane extremes.
4. Reconstructs each layer as an **adaptive quadtree mesh** on the portal-frustum plane — coarse quads in flat interior regions, finer quads along cut edges where the layer's depth slab ends.
5. Splits each layer into two geom subsets — opaque interior + alpha-tested border — bound to separate materials to minimise runtime overdraw.
6. Culls quads never visible from any head-plane viewpoint (PVS optimisation).
7. Exports a USD stage with `UsdPreviewSurface` materials, consumable by any Hydra-based engine.

The result: a full 3D environment collapses to a handful of textured displaced quads that preserve parallax for limited head motion at a fraction of the runtime cost.

## Status

This repo is in early bootstrap. Implementation is broken into phases with explicit sign-off between each:

| Phase | Subject | Status |
|---|---|---|
| 0 | Dev environment bootstrap | pending |
| 1 | Portal Rig + Infrastructure | pending |
| 2 | Layer Placement | pending |
| 3 | Multi-view Capture | pending |
| 4 | Reconstruction + Preview | pending |
| 5 | Occlusion Culling | pending |
| 6 | USD Export (user-activated) | pending |

Full scope, acceptance criteria, and architecture for every phase live in [`docs/implementation_plan.md`](docs/implementation_plan.md).

## Development setup

Primary workflow: **VS Code + Jacques Lucke's "Blender Development" extension**, for hot-reload on save without restarting Blender.

```bash
# 1. Clone
git clone git@github.pie.apple.com:oliver-shaw/depth-peel-xr.git
cd depth-peel-xr

# 2. Dev venv (for pytest + autocomplete; Blender uses its own bundled Python)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt   # created in Phase 0

# 3. Symlink the add-on into Blender's extensions dir
./tools/install_dev_symlink.sh        # created in Phase 0

# 4. Install VS Code + the Blender Development extension, then:
#    Cmd+Shift+P → "Blender: Start"
```

See the **Development workflow** section of the implementation plan for the full setup, the manual `dev_reload.py` fallback, and the `core/` vs `operators/` package rules that keep hot-reload clean.

## Repository layout

```
depth-peel-xr/
├── addon/                            # the add-on package (symlinked into Blender's extensions dir)
├── tools/                            # install_dev_symlink.sh, run_core_tests.sh
├── docs/
│   ├── implementation_plan.md        # authoritative phase-by-phase plan
│   └── reference/                    # user's existing depth-layer-placement Python logic (ported into addon/core/)
├── .vscode/                          # launch.json, settings.json (for the Blender Development extension)
├── requirements-dev.txt              # fake-bpy-module-latest, pytest, numpy, usd-core
├── CLAUDE.md                         # project context for Claude Code
└── README.md                         # this file
```

Addon-package sub-tree and module responsibilities are documented in the implementation plan.

## Design principles

- **`addon/core/` never imports `bpy`.** All pure-math modules (depth pooling, CDF sampling, quadtree building, multi-view reprojection, occlusion culling, USD authoring) are testable in plain Python via `pytest`.
- **Every Blender-mutating operator has a matching cleanup operator**, and a master `Clean up all` runs them in sequence. Running the pipeline leaves no orphan datablocks on reset.
- **USD written directly via `pxr.Usd`** (not Blender's built-in exporter), for precise control over prim hierarchy, geom subsets, and material binding. `pxr` is not bundled with Blender — the add-on detects it and surfaces an install prompt if missing.
- **Layer placement is a pluggable strategy.** V1 ships two strategies (user-count + fixed distribution; histogram-driven hybrid CDF + uniform); the `LayerPlacementStrategy` ABC leaves room for budget-driven optimisation and manual gizmos without touching the operator layer.

## Reference material

- [`docs/reference/hybrid_depth_layers.py`](docs/reference/hybrid_depth_layers.py) — original hybrid importance + uniform NDC sampling with empty-layer removal. Being ported into `addon/core/layer_placement/histogram_hybrid.py`.
- [`docs/reference/importance_depth_layers.py`](docs/reference/importance_depth_layers.py) — pure importance-sampling variant; CDF-logic reference.

The reusable pieces — NDC transform, histogram → CDF inverse sampling, empty-layer detection, and gap-closing — port directly; only the depth-loading front-end changes (Blender render result → numpy, instead of OIIO EXR read).
