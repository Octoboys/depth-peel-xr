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
| 0 | Dev environment bootstrap | complete |
| 1 | Portal Rig + Infrastructure | complete |
| 2 | Layer Placement | complete |
| 3 | Multi-view Capture | pending |
| 4 | Reconstruction + Preview | pending |
| 5 | Occlusion Culling | pending |
| 6 | USD Export (user-activated) | pending |

Full scope, acceptance criteria, and architecture for every phase live in [`docs/implementation_plan.md`](docs/implementation_plan.md).

## Development setup

Primary workflow: **VS Code + Jacques Lucke's "Blender Development" extension**, for hot-reload on save without restarting Blender.

### First-time setup (per machine)

```bash
# 1. Clone
git clone git@github.pie.apple.com:oliver-shaw/depth-peel-xr.git
cd depth-peel-xr

# 2. Dev venv (for pytest + autocomplete; Blender uses its own bundled Python)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# 3. Install the Blender Development VS Code extension
code --install-extension jacqueslucke.blender-development
```

### VS Code configuration (required for hot-reload)

After opening the repo in VS Code, three one-off steps in the UI. The JSON settings live in `.vscode/settings.json` but some of them only take effect after being confirmed manually.

1. **Trust the workspace.** If the bottom-left status bar says "Restricted Mode", workspace-contributed extension settings are partially ignored and hot-reload will not fire. `Cmd+Shift+P → Workspaces: Manage Workspace Trust → Trust`.
2. **Confirm `reloadOnSave` is applied at workspace scope.** `Cmd+,` → search `blender.addon.reloadOnSave` → click the **Workspace** tab directly below the search bar → the checkbox must be ticked. If it's empty despite being `true` in `.vscode/settings.json`, tick it in the UI once — the first-time workspace load sometimes doesn't apply the JSON value until it's touched through the UI. Without this, `Blender: Reload Addons` still works manually but saves won't trigger reload.
3. **Point the extension at the right Blender.** First time you run `Cmd+Shift+P → Blender: Start`, it prompts for a Blender binary. The path is typically `/Applications/Blender.app/Contents/MacOS/blender` on a standard install. If you have multiple Blender versions coexisting (e.g. `Blender_44.app` for 4.4, `Blender_5.app` for 5.x), adjust the app name accordingly. If you later need to re-point it, edit `blender.executables` via `Cmd+Shift+P → Preferences: Open User Settings (JSON)`.

4. **`blender.addon.loadDirectory` must be a relative path.** The value in `.vscode/settings.json` is already set to `"depth_peel_xr"` (relative). Do **not** change it to `"${workspaceFolder}/depth_peel_xr"` — the Blender Development extension v0.0.31 constructs the absolute path itself via `path.join(workspaceFolder, rawValue)`, then does variable substitution afterwards. Using an absolute or `${workspaceFolder}`-prefixed value causes the path to be doubled at runtime and the add-on fails to load.

### First-time Blender launch

1. `Cmd+Shift+P → Blender: Start` — the first launch registers Blender's `vscode_development` extensions repository in Blender prefs and creates the directory on disk. It then attempts to symlink our add-on into it.
2. If the add-on panel ("Depth Peel XR" tab in the N-panel) appears immediately, the auto-link succeeded — skip to step 5.
3. If the panel is absent (VS Code's auto-link step is known to silently no-op in some Blender versions / fresh installs), quit Blender and run the fallback:
   ```bash
   ./tools/install_dev_symlink.sh
   ```
   This creates the symlink at the exact path VS Code's save-reload handler expects (`extensions/vscode_development/depth_peel_xr`). **`Blender: Start` must have run at least once before this script** — it creates the `vscode_development/` directory that the script targets.
4. `Blender: Start` again. In Blender, **Edit → Preferences → Get Extensions**, find "Depth Peel XR" and tick the enable checkbox. If any stale `bl_ext.user_default.depth_peel_xr` ghost entries show up broken, remove them first.
5. **Hamburger menu → Save Preferences** — locks in auto-enable for every future launch so step 4 is one-off.

### Daily loop

- `Cmd+Shift+P → Blender: Start` — launches Blender with the add-on already enabled (thanks to saved prefs).
- Save any file under `depth_peel_xr/` — triggers a live add-on reload in the running Blender (no restart).
- `Cmd+Shift+P → Blender: Reload Addons` — manual reload fallback if auto-reload misfires.
- `pytest tests/` in the terminal — runs the `core/` unit tests without touching Blender.

See the **Development workflow** section of the implementation plan for full detail, the manual `dev_reload.py` fallback, and the `core/` vs `operators/` package rules that keep hot-reload clean.

## Repository layout

```
depth-peel-xr/
├── depth_peel_xr/                            # the add-on package (symlinked into Blender's extensions dir)
├── tests/                            # pytest suite for depth_peel_xr/core/ (outside depth_peel_xr/ so pytest
│                                     # collection isn't blocked by depth_peel_xr/__init__.py's bpy import)
├── tools/                            # install_dev_symlink.sh, run_core_tests.sh
├── docs/
│   ├── implementation_plan.md        # authoritative phase-by-phase plan
│   └── reference/                    # user's existing depth-layer-placement Python logic (ported into depth_peel_xr/core/)
├── .vscode/                          # launch.json, settings.json (for the Blender Development extension)
├── pyproject.toml                    # pytest config (testpaths, pythonpath, import mode)
├── requirements-dev.txt              # fake-bpy-module-latest, pytest, numpy, usd-core
├── CLAUDE.md                         # project context for Claude Code
└── README.md                         # this file
```

Addon-package sub-tree and module responsibilities are documented in the implementation plan.

## Design principles

- **`depth_peel_xr/core/` never imports `bpy`.** All pure-math modules (depth pooling, CDF sampling, quadtree building, multi-view reprojection, occlusion culling, USD authoring) are testable in plain Python via `pytest`.
- **Every Blender-mutating operator has a matching cleanup operator**, and a master `Clean up all` runs them in sequence. Running the pipeline leaves no orphan datablocks on reset.
- **USD written directly via `pxr.Usd`** (not Blender's built-in exporter), for precise control over prim hierarchy, geom subsets, and material binding. `pxr` is not bundled with Blender — the add-on detects it and surfaces an install prompt if missing.
- **Layer placement is a pluggable strategy.** V1 ships two strategies (user-count + fixed distribution; histogram-driven hybrid CDF + uniform); the `LayerPlacementStrategy` ABC leaves room for budget-driven optimisation and manual gizmos without touching the operator layer.

## Reference material

- [`docs/reference/hybrid_depth_layers.py`](docs/reference/hybrid_depth_layers.py) — original hybrid importance + uniform NDC sampling with empty-layer removal. Shipped at `depth_peel_xr/core/layer_placement/histogram_hybrid.py` with intentional divergence on the empty-layer collapse semantics (the reference fused every contiguous non-empty pair into one layer; the port keeps them separate).
- [`docs/reference/importance_depth_layers.py`](docs/reference/importance_depth_layers.py) — pure importance-sampling variant; CDF-logic reference.

The reusable pieces — NDC transform, histogram → CDF inverse sampling, empty-layer detection, and gap-closing — port directly; only the depth-loading front-end changes (Blender render result → numpy, instead of OIIO EXR read).
