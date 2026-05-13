# Depth Peel XR — Blender Add-on Implementation Plan

## Context

We're building a Blender 5.x add-on that optimises a 3D environment for XR viewing from a bounded **head-plane** in front of a rectangular **portal** (a "window into the scene"). The add-on re-interprets the scene as a small number of depth-peeled layers — each an adaptive quad mesh displaced to real scene depth with a baked colour texture — so the runtime (an external engine consuming USD) pays only the cost of a handful of textured, displaced quads while preserving parallax for limited head motion.

Problem it solves: full 3D environments are too expensive to render at XR framerates on mobile/standalone headsets, but the viewer's movement is limited to a small head-plane in front of a portal, so most of the scene's geometric complexity is redundant from every reachable viewpoint. By baking the scene into depth-peeled layered impostors once, we get near-photoreal visuals at trivial runtime cost, with just enough parallax for a comfortable XR experience.

Intended outcome: the artist opens a Blender scene, adds a **Portal Rig** (portal + head-plane), chooses a layer count and quad budget, clicks `Bake` to generate the preview, inspects and iterates, and — when satisfied — explicitly exports a USD stage consumable by the target engine.

The user already has reference Python logic for layer placement (hybrid CDF + uniform sampling with empty-layer removal) at:
- `/Users/ollieshaw/Desktop/hybrid_depth_layers.py`
- `/Users/ollieshaw/Desktop/importance_depth_layers.py`

The reusable pieces — NDC transform, histogram → CDF inverse sampling, empty-layer dropout, gap-closing — port directly. Only the depth-loading front-end changes: instead of reading pre-rendered EXR files, we feed numpy depth arrays straight out of a Blender render result.

## Locked-in decisions

| Area | Decision |
|---|---|
| Target | Blender 5.x, Depth Peel XR add-on. |
| Output | USD authored directly via `pxr.Usd`. Preview always built in Blender; USD export is a separate, user-activated action. |
| Scene scope (V1) | Bounded indoor + outdoor with distant backdrop. No thin/alpha geometry, no semi-transparency. |
| Capture geometry | Regular planar 3D rendering through the portal from perspective cameras on the head-plane. |
| Portal Rig | Single **Portal** plane + `headplane_offset` property that draws a second, unselectable **head-plane** behind it. Head-plane rectangle is co-planar and parallel to the portal. |
| Multi-view grid | NxN perspective cameras on the head-plane (2 axes, no depth sampling). Default 3×3 = 9. User-tunable N. |
| Depth probe for layer placement | **Single** camera at head-plane centre, FoV expanded to encompass the portal as seen from every head-plane extreme. Sufficient for depth-distribution analysis; multi-view is only needed for disocclusion fill during capture. |
| Layer placement | Two strategies: (a) user-count + fixed distribution (uniform-Z / 1-over-Z / log); (b) histogram-driven hybrid CDF + uniform (port of existing script). Pluggable interface so budget-driven and manual-gizmo strategies can slot in later. |
| Viewing resolution | User-configurable render/reconstruction resolution that **matches the portal's display aspect ratio** — it's a single detail-level control governing texel density relative to the viewer. Lower values speed up iteration; higher values stress-test output quality. |
| Layer reconstruction | Per-layer adaptive **quadtree** of quads on the portal-frustum plane. Two user-tunable sizes: `core_quad_size` (baseline for smooth interior) and `min_quad_size` (finest allowed at cut edges). Vertices displaced to true 3D position (`ray_dir * depth`, *not* naive Z-only displacement — critical for wide-FoV correctness). |
| Per-layer submeshes | Two geom subsets per layer: `interior` (opaque, front-to-back, writes Z) and `border` (alpha-tested/blended, sparse). Each binds its own material to minimise runtime overdraw. |
| Occlusion culling | Post-reconstruction stage: cull quads that are never visible from **any** of the NxN head-plane viewpoints (occluded by closer layers from every position). Toggleable. |
| Texture per layer | Single consolidated RGBA texture (RGB = colour, A = coverage mask for the border material). Multi-view captures used during baking **only** to fill disocclusions visible from head-plane extremes into the canonical (head-plane-centre) projection. No view-dependent textures in V1. |
| Render engine | User-selectable Cycles / EEVEE Next per bake. |
| Sky / backdrop | User-configurable: merge into last depth layer, or dedicated backdrop layer. |
| Materials | User chooses **either** `UsdPreviewSurface` **or** MaterialX at export time — not both. V1 ships `UsdPreviewSurface` only; MaterialX deferred. In-Blender preview uses Principled BSDF built manually at reconstruction time. |
| UI | Blender 5.x N-panel sidebar (`Depth Peel XR` tab) with staged sections. Every section that adds entities to the scene includes a cleanup operator. |
| Scene organisation | Three dedicated collections, auto-created: `DepthPeelXR_Rig` (portal + multi-view cameras), `DepthPeelXR_Source` (original scene objects, moved there on preview build), `DepthPeelXR_Preview` (reconstructed layer meshes). A single `Toggle Preview` operator hides one and shows the other. |
| Inspection tools | Dropdown to look through each multi-view camera (sets active camera + enters camera view). Works for both original scene inspection and preview inspection. |
| Project folder | All temporary files, captures, and USD output default to `<blend_file_dir>/depth_peel_xr/<scene_name>/` with subfolders `temp/`, `captures/`, `usd/`. Overridable per-scene via a property. Warn if the .blend is unsaved. |
| Cleanup | Every operator that modifies the scene (adds rig, creates collections, builds preview) has a matching cleanup operator that removes added entities and resets parameter changes. A master `Clean up all` operator runs them all. |
| Dependency | `pxr` (usd-core) is not bundled. Add-on detects it on registration and shows an actionable install instruction. Export is disabled until `pxr` is importable. |

## Architecture overview

Top-level repository layout:

```
depth_peel_blender/                   # repo root (= current working dir)
├── depth_peel_xr/                            # the add-on package (symlinked into Blender's extensions dir)
│   └── ...                           # see tree below
├── tests/                            # pytest suite for depth_peel_xr/core/ (kept outside depth_peel_xr/ so
│                                     # pytest doesn't trip on depth_peel_xr/__init__.py's `import bpy`)
├── tools/
│   ├── install_dev_symlink.sh        # symlinks ./depth_peel_xr/ into Blender's extensions dir
│   └── run_core_tests.sh             # pytest wrapper for core/ modules
├── .vscode/
│   ├── launch.json                   # Blender Development extension debug config
│   └── settings.json                 # blender.addon.loadDirectory → ./depth_peel_xr/
├── pyproject.toml                    # pytest config (testpaths, pythonpath, import mode)
├── requirements-dev.txt              # fake-bpy-module-latest, pytest, numpy, usd-core
└── README.md
```

Add-on package (`depth_peel_xr/`):

```
depth_peel_xr/
├── __init__.py                       # bl_info + register()/unregister()
├── dev_reload.py                     # manual reload fallback (unregister + purge sys.modules + re-register)
├── preferences.py                    # add-on prefs (pxr install status, global defaults)
├── properties.py                     # PropertyGroups on Scene:
│                                     #   PortalRigProps, LayerPlacementProps,
│                                     #   CaptureProps, ReconstructionProps,
│                                     #   OcclusionProps, ExportProps, PathsProps
├── ui/
│   ├── n_panel.py                    # staged sections in DepthPeelXR tab
│   └── overlays.py                   # viewport draw handlers (head-plane, last-layer far)
├── operators/
│   ├── portal_rig.py                 # add / cleanup portal rig + cameras
│   ├── place_layers.py               # run layer-placement strategy + cleanup
│   ├── capture.py                    # NxN multi-view RGBA+depth render per layer + cleanup
│   ├── reconstruct.py                # quadtree + displacement + preview build + cleanup
│   ├── occlusion_cull.py             # post-reconstruction quad culling
│   ├── inspect.py                    # look-through-camera, toggle original/preview
│   ├── export_usd.py                 # user-activated USD stage export
│   └── reset.py                      # master "Clean up all"
├── core/                             # engine-agnostic (numpy + pxr only, no bpy)
│   ├── depth.py                      # NDC transform, pooled depth analysis
│   ├── layer_placement/
│   │   ├── base.py                   # LayerPlacementStrategy ABC
│   │   ├── fixed_distribution.py     # uniform-Z / 1-over-Z / log
│   │   └── histogram_hybrid.py       # port of hybrid_depth_layers.py
│   ├── quadtree.py                   # adaptive quad subdivision with core/min size
│   ├── reconstruction.py             # depth → displaced vertex positions → interior/border submeshes
│   ├── multi_view.py                 # canonical-view disocclusion fill
│   ├── occlusion.py                  # multi-view PVS culling of occluded quads
│   └── usd_author.py                 # pxr-based USD stage authoring (UsdPreviewSurface)
├── paths.py                          # project-folder resolution helpers
└── (tests live at repo-root tests/, not inside this package — see top-level tree)
```

Design principle: **core/ has no `bpy` imports.** Anything pure math (depth pooling, CDF sampling, quadtree building, multi-view merging, occlusion culling, USD authoring) is testable in a plain Python environment with numpy + usd-core. `operators/` is the thin bpy-binding feeding scene data into core and writing results back.

## Data flow

```
Scene + Portal + Head-plane + Viewing Resolution
        │
        ▼
[Depth probe: single centre camera, super-frustum FoV, Z pass only]
        │  depth buffer (camera-axis Z, at viewing resolution)
        ▼
[Layer Placement: NDC transform → CDF/uniform/hybrid → (near,far) list]
        │  layers = [(n0,f0), (n1,f1), ...]
        │  viewport overlay: far edge of last layer only
        ▼
[Capture: NxN grid × N_layers, RGBA+depth]
        │  depth-slab via camera.data.clip_start / clip_end (works in Cycles + EEVEE Next)
        │  written to <project>/captures/layer_NN/view_MM_{color,depth}.exr
        ▼
[Multi-view consolidation, per layer]
        │  canonical RGBA (centre view) + disocclusion fill from siblings
        │  canonical depth map + coverage mask
        ▼
[Reconstruction, per layer]
        │  adaptive quadtree (core_quad_size → min_quad_size at cut edges)
        │  vertex positions = inverse_projection(ndc, depth)  — true 3D
        │  classify leaves: interior (opaque) / border (alpha)
        │  build Blender Mesh datablocks + Principled BSDF materials
        ▼
[Preview collection build]
        │  move source objects into DepthPeelXR_Source
        │  populate DepthPeelXR_Preview with per-layer meshes
        ▼  (automatic)
[Occlusion Culling (optional toggle)]
        │  for each farther-layer quad: test visibility against every head-plane view
        │  drop quads occluded from every viewpoint
        ▼
   Preview visible in viewport
        │
        ▼  (user-activated)
[USD Export via pxr.Usd]
        │  one UsdGeomMesh per layer with two GeomSubsets (interior/border)
        │  UsdPreviewSurface per material
        │  textures written to <project>/usd/textures/
        ▼
   <project>/usd/scene.usda
```

## Phased roadmap

Each phase is independently shippable with sign-off demo. We pause for discussion between phases.

### Phase 0 — Dev environment bootstrap

**Goal**: hot-reload dev loop working end-to-end before any feature code gets written.

**Scope**:
- Repo scaffolding: `depth_peel_xr/__init__.py` (minimal `bl_info` + register/unregister stubs), `depth_peel_xr/dev_reload.py`, `tools/install_dev_symlink.sh`, `tools/run_core_tests.sh`, `.vscode/launch.json`, `.vscode/settings.json`, `requirements-dev.txt`, `README.md` (onboarding instructions).
- `depth_peel_xr/__init__.py` registers a trivial N-panel with a single "Hello" label — enough to prove the add-on loads and the panel shows up in the `Depth Peel XR` tab.
- Empty `depth_peel_xr/core/` and `tests/` (at repo root, not under `depth_peel_xr/` — see Development workflow note) with one smoke test (`def test_numpy_available(): import numpy` etc.).
- Confirm VS Code `Blender: Start` launches Blender with the add-on registered and the panel visible.
- Confirm editing a string in `__init__.py` and saving triggers a live reload in Blender (panel label updates).
- Confirm `pytest` in the terminal runs the smoke test.

**Acceptance**:
- Fresh clone → run `install_dev_symlink.sh` → open in VS Code → `Blender: Start` → panel visible in Blender.
- Edit + save picks up without restart.
- `tools/run_core_tests.sh` passes.

**Critical files**: `depth_peel_xr/__init__.py`, `depth_peel_xr/dev_reload.py`, `tools/install_dev_symlink.sh`, `tools/run_core_tests.sh`, `.vscode/launch.json`, `.vscode/settings.json`, `requirements-dev.txt`.

**Verification**: manual — run through the onboarding sequence from a clean shell; demonstrate hot reload.

---

### Phase 1 — Portal Rig + Infrastructure

**Goal**: artist can add a portal rig, inspect multi-view cameras, toggle preview visibility. Project folder scaffolding in place.

**Scope**:
- `PortalRigProps` PropertyGroup (portal object ref, `headplane_offset`, `headplane_width`, `headplane_height`, grid `N`, `viewing_resolution` — a single scalar driving render/reconstruction texel density; the other dimension is derived from the portal's aspect ratio).
- `PathsProps` PropertyGroup (project root, auto-derived from .blend location; user-overridable).
- `DEPTHPEELXR_OT_add_portal_rig` operator:
  - Creates collection `DepthPeelXR_Rig`.
  - Adds a Portal Mesh plane into it.
  - Generates NxN perspective cameras on the head-plane (centre + grid), all aiming through the portal with matching frustums.
- `DEPTHPEELXR_OT_cleanup_portal_rig` operator: removes the rig collection and all contained entities.
- `DEPTHPEELXR_OT_look_through_view_camera` operator: dropdown of the NxN cameras; selecting one sets active camera + enters camera view.
- `DEPTHPEELXR_OT_toggle_preview` operator: swaps visibility between `DepthPeelXR_Source` and `DepthPeelXR_Preview` collections (creates them if absent).
- Overlay: head-plane wireframe drawn unselectably behind portal + NxN grid sample points.
- N-panel section `Portal Rig`: add/cleanup buttons, property fields, camera-inspection dropdown, preview toggle, project-folder field.

**Acceptance**:
- Adding a rig creates exactly one collection with portal + NxN cameras.
- Head-plane + grid overlay update live with property changes.
- Camera dropdown works: selecting `view_05` enters that camera's view.
- Toggle preview works even before Phase 4 exists (just hides/shows collections).
- Cleanup removes everything the operator added; no orphan datablocks.
- Save+reopen preserves rig and property state.
- Project folder auto-resolves to `<blend_dir>/depth_peel_xr/<scene_name>/`; warn if .blend is unsaved.

**Critical files**: `depth_peel_xr/__init__.py`, `depth_peel_xr/properties.py`, `depth_peel_xr/paths.py`, `depth_peel_xr/operators/portal_rig.py`, `depth_peel_xr/operators/inspect.py`, `depth_peel_xr/ui/n_panel.py`, `depth_peel_xr/ui/overlays.py`.

**Verification**: manual — add rig, tweak properties, inspect cameras, toggle collections, save/load.

---

### Phase 2 — Layer Placement

**Goal**: produce a list of `(near, far)` layer boundaries from a single-camera depth probe.

**Scope**:
- `core/depth.py` — NDC transform and pooled-depth utilities (adapted from existing scripts to take numpy arrays directly).
- `core/layer_placement/base.py` — `LayerPlacementStrategy` ABC.
- `core/layer_placement/fixed_distribution.py` — user-count + uniform-Z / 1-over-Z / log.
- `core/layer_placement/histogram_hybrid.py` — port of `hybrid_depth_layers.py` logic.
- `DEPTHPEELXR_OT_place_layers` operator:
  - Configures centre camera FoV to super-frustum through portal.
  - Renders Z pass only at viewing resolution (fast).
  - Runs selected strategy, stores `layers` list on scene props.
  - Updates viewport overlay to show the **far edge of the last layer only** (single plane).
- `DEPTHPEELXR_OT_cleanup_layers`: clears layer list and overlay; reverts camera-config changes.
- N-panel section `Layer Placement`: strategy dropdown, per-strategy params (layer count, distribution), `Analyse & Place`, cleanup button.

**Acceptance**:
- Single centre-camera depth probe is sufficient (verified on a synthetic scene — no need for multi-view at this stage).
- Running `Analyse & Place` on a test scene matches the standalone `hybrid_depth_layers.py` output (same depth input → same boundaries), verified by unit test.
- Far-edge-of-last-layer plane appears and updates correctly.
- Changing strategy or params changes the result.
- Cleanup removes the overlay and any camera mods.

**Verification**: numpy unit tests against existing script outputs; manual visual check of far-edge plane in a test scene.

---

### Phase 3 — Multi-view Capture

**Goal**: for each placed layer, capture NxN RGBA + depth passes with the layer's depth slab enforced.

**Scope**:
- `DEPTHPEELXR_OT_capture` operator:
  - For each of the NxN cameras × each layer: set `camera.data.clip_start = layer.near`, `clip_end = layer.far`, render RGBA + Z pass, write to `<project>/captures/layer_NN/view_MM_{color,depth}.exr`.
  - Engine selector (Cycles / EEVEE Next).
  - Resolution driven by `viewing_resolution` from PortalRigProps, with the secondary dimension derived from portal aspect.
  - Modal operator with progress.
- `DEPTHPEELXR_OT_cleanup_captures`: deletes the captures subfolder, clears any cached image references.
- `core/multi_view.py` — reproject each sibling view's RGBA into the canonical centre view via `(ndc, depth) → world → centre-ndc` and composite where the centre view's coverage is missing. Produces `(rgba_canonical, depth_canonical, coverage_mask)` per layer.
- N-panel section `Capture`: engine dropdown, output path display, `Run Capture`, cleanup button.

**Acceptance**:
- N=3, L=4 produces 36 RGBA + 36 depth EXRs on disk in the expected layout.
- Depth-slab clipping works in both engines (inspect outermost layer: nothing nearer than `near`, nothing further than `far`).
- Consolidated canonical RGBA per layer covers pixels visible from head-plane extremes but occluded from centre view.
- Cleanup removes the captures directory.

**Gotchas (from Blender API recon)**:
- Cycles and EEVEE Next both respect `camera.data.clip_start/clip_end`.
- Depth-pass access differs between engines (`ViewLayer.cycles` vs `ViewLayer.eevee`); wrapped behind a single helper.
- Both engines return camera-axis Z; inverse projection uses that directly.

**Verification**: manual render of a known scene; compare depth pass to analytic expectation; visually inspect consolidated textures.

---

### Phase 4 — Reconstruction + Preview

**Goal**: build per-layer displaced quad meshes with interior/border subsets, populate `DepthPeelXR_Preview` collection, preview is immediately visible.

**Scope**:
- `core/quadtree.py` — adaptive quadtree:
  - Root = portal-frustum quad.
  - Split rule: if quad straddles coverage-mask edge **and** quad size > `min_quad_size`, subdivide into 4; also subdivide if depth variance within the quad exceeds `depth_gradient_tol`, down to `core_quad_size`.
  - Leaves classified `interior` (fully covered) or `border` (straddles).
- `core/reconstruction.py`:
  - Each leaf vertex's world position via `inverse_projection(ndc_x, ndc_y, depth)` — avoids spurious curvature at wide FoV.
  - Build interior and border face arrays.
  - UVs: screen-space passthrough.
  - Border quads: rasterise coverage mask into RGBA texture's alpha channel for that quad's UV region.
- `DEPTHPEELXR_OT_reconstruct` operator:
  - For each layer: create `bpy.types.Mesh` with two geom-subset-equivalent vertex groups, build Principled BSDF materials (opaque for interior, alpha-blend for border), spawn as Objects in `DepthPeelXR_Preview` collection.
  - Move original scene objects into `DepthPeelXR_Source` collection.
  - Auto-switch visibility to Preview on completion.
- `DEPTHPEELXR_OT_cleanup_reconstruction`: removes Preview collection contents, restores source visibility.
- N-panel section `Reconstruction`: `core_quad_size`, `min_quad_size`, `depth_gradient_tol`, `Reconstruct`, cleanup button.

**Acceptance**:
- Flat wall: interior quads coalesce to coarsest, border quads only at silhouette, all ≥ `min_quad_size`.
- Detailed layer: depth-gradient subdivision produces finer interior quads in high-variance regions.
- Head-plane-corner viewpoint shows correct parallax; no z-fighting.
- Wide-FoV flat-wall test: reconstructed mesh stays planar (verifies inverse-projection maths).
- Source moves into `DepthPeelXR_Source`; preview appears in `DepthPeelXR_Preview`; toggle operator swaps them.
- Cleanup restores scene to pre-reconstruction state.

**Verification**: numpy unit test on synthetic depth maps with known cut edges; manual visual test in Blender.

---

### Phase 5 — Occlusion Culling

**Goal**: cull quads invisible from every head-plane viewpoint to reduce final poly count and overdraw.

**Scope**:
- `core/occlusion.py`:
  - For each layer's quads (walk far-to-near), test visibility by rasterising closer-layer interior meshes into each of the NxN views' depth buffers.
  - A quad is culled iff all NxN views report it fully occluded (within an epsilon).
  - Vectorised numpy implementation; no bpy dependency.
- `DEPTHPEELXR_OT_occlusion_cull` operator: runs the cull on the existing preview meshes, prints stats (quads removed per layer, % reduction).
- `DEPTHPEELXR_OT_undo_cull`: rebuilds from reconstruction cache (so user can re-enable culled quads without a full rebake).
- N-panel section `Optimisation`: `Cull Occluded Quads` toggle + button, `Undo Cull` button.

**Acceptance**:
- On a scene with a closer wall fully covering a farther wall: the farther layer's covered region is culled.
- On a scene with a closer wall partially covering: only the blocked quads drop.
- No quads culled that are visible from any head-plane extreme (sanity-checked by eye + automated test).
- Undo restores the pre-cull mesh without re-running capture.

**Verification**: synthetic-scene unit test + manual visual inspection.

---

### Phase 6 — USD Export (user-activated)

**Goal**: explicit-action export of the preview as a USD stage consumable by downstream engines.

**Scope**:
- `core/usd_author.py`:
  - Imports `pxr.Usd`, `pxr.UsdGeom`, `pxr.UsdShade`, `pxr.Sdf` (guarded).
  - Per layer: one `UsdGeomMesh` with two `UsdGeomSubset`s (interior / border).
  - Each subset binds a `UsdShadeMaterial` with a `UsdPreviewSurface` shader network.
  - Texture assets copied/referenced relative to the .usda file.
  - Sky/backdrop: either fused into last layer or authored as its own prim (user choice).
- `DEPTHPEELXR_OT_export_usd` operator: collects preview data, calls `core.usd_author.write_stage`, writes to `<project>/usd/scene.usda`.
- N-panel section `Export`: output-path display, `Export USD` button, `pxr` availability banner.
- `preferences.py`: `pxr` detection on registration + `Install usd-core into Blender Python` button.

**Acceptance**:
- `Export USD` only runs on explicit click — never auto-triggered.
- Exported `.usda` opens in `usdview` or any Hydra-based renderer with visuals matching Blender preview.
- `UsdPreviewSurface` network present on every material.
- Missing `pxr` produces an actionable banner, never a traceback.

**Verification**:
- Full pipeline end-to-end on a small test scene.
- USD re-imported back into Blender via built-in importer matches preview.
- Unit tests for `usd_author.py` with synthetic mesh data.

---

## Cross-cutting concerns

- **Project folder**: all generated files live under `<blend_dir>/depth_peel_xr/<scene_name>/{temp,captures,usd}/`. Overridable. Add-on warns on unsaved .blend.
- **Collections**: `DepthPeelXR_Rig`, `DepthPeelXR_Source`, `DepthPeelXR_Preview` — auto-created when first needed, never touched otherwise.
- **Cleanup discipline**: every operator that adds entities provides a matching cleanup operator. A master `Clean up all` runs them all.
- **`pxr` dependency**: detected on registration; export disabled until satisfied; banner + install button in preferences.
- **Quad size units**: NDC-space by default (artists think in screen-space pixels); optional world-unit-at-reference-depth mode for uniform physical sizing.
- **Curvature correctness**: every depth → 3D conversion uses `world_pos = inverse_projection(ndc_x, ndc_y, depth)`. Unit-tested on wide-FoV scenes.
- **Layer-placement pluggability**: `LayerPlacementStrategy` ABC leaves room for budget-driven optimisation and manual-gizmo boundaries without touching operator layer.
- **No VCS integration** anywhere.
- **MaterialX**: deferred to V2.

## Development workflow

Primary setup: **VS Code + Jacques Lucke's "Blender Development" extension**. This is the de-facto standard for Blender add-on work — it launches Blender with our add-on pre-registered, auto-reloads on every file save via Blender's internal RPC, and supports `debugpy` breakpoints natively.

### First-time setup

1. Clone/open the repo at `<working_dir>`.
2. Create a Python venv for dev tooling (autocomplete + tests only; Blender uses its own bundled Python): `python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt`. Contents: `fake-bpy-module-latest`, `pytest`, `numpy`, `usd-core`.
3. Run `tools/install_dev_symlink.sh` to symlink `<working_dir>/depth_peel_xr/` into Blender's extensions directory (`~/Library/Application Support/Blender/5.0/extensions/user_default/depth_peel_xr` on macOS; the script detects the platform).
4. Install VS Code + the "Blender Development" extension.
5. Open the repo in VS Code. The shipped `.vscode/settings.json` points `blender.addon.loadDirectory` at `./depth_peel_xr/`.
6. `Cmd+Shift+P → Blender: Start` launches Blender with the add-on registered.

### Daily cycle

- **Blender-side changes** (operators, UI, properties, overlays): edit a file in VS Code → save → the running Blender reloads the add-on automatically. No restart, no manual trigger.
- **Core/math changes** (anything in `core/`): edit → save → run pytest in VS Code's terminal (`pytest tests/`) for sub-second feedback. Optionally scope to one test with `-k`. Never touches Blender.
- **Debugging Blender code**: `Cmd+Shift+P → Blender: Start (Debug)` — VS Code attaches debugpy; set breakpoints normally; hit them next time the operator runs.
- **Debugging core code**: standard VS Code pytest debug config.

### Package-structure rules (required for reload to work cleanly)

- All intra-add-on imports use **relative form** (`from .core import depth`, not absolute). This is what lets reload purge `sys.modules` entries cleanly.
- No add-on code caches bpy types at module-load time (property definitions register at `register()` call, not at import).
- `core/` never imports `bpy`. Enforced by a one-line lint check in `run_core_tests.sh`.

### Fallback — no VS Code available

- Open Blender's Text Editor, load `depth_peel_xr/dev_reload.py`, click Run Script. It unregisters, purges all `depth_peel_xr.*` entries from `sys.modules`, re-imports, and re-registers. Binds cleanly to a custom keymap for one-keystroke reload.

### Autocomplete

- `fake-bpy-module-latest` (dev-dependency only) provides stubs for `bpy`, `bmesh`, `gpu`, `mathutils`. VS Code's Python extension picks them up from the `.venv/` interpreter automatically.

### Tooling scripts

- `tools/install_dev_symlink.sh` — creates (or refreshes) the symlink from Blender's extensions dir to `./depth_peel_xr/`. Idempotent. Detects Blender version dir on macOS/Linux/Windows.
- `tools/run_core_tests.sh` — `pytest -q tests/` plus a one-line check that `grep -r "^import bpy\|^from bpy" depth_peel_xr/core/` returns nothing (enforces core's bpy-free contract). Tests live at repo-root `tests/` rather than `depth_peel_xr/tests/` because `depth_peel_xr/__init__.py` imports bpy at module load, which breaks pytest collection outside Blender.

## Open items deferred to later iterations

- Thin / alpha-tested geometry (foliage, fences).
- Semi-transparent surfaces (glass, water).
- View-dependent shading (specular, reflections).
- Budget-driven layer-placement optimisation.
- Manual layer-boundary gizmos.
- Animated scenes.
- MaterialX material authoring.

## End-to-end verification plan

Once all feature phases (1–6) are implemented on top of the Phase 0 dev environment:

1. **Synthetic flat-wall test**: single flat wall → one layer, no border submesh, no curvature artefacts.
2. **Occlusion test**: closer wall in front of farther wall → two layers, disocclusion filled from sibling views, farther-layer covered region culled.
3. **Wide-FoV curvature test**: flat wall with 120° portal FoV → reconstructed mesh stays planar.
4. **Round-trip test**: Blender preview vs. USD reload matches.
5. **Target-engine handshake**: export USD, load in user's engine, verify visual parity.
6. **Cleanup test**: run all stages, hit `Clean up all`, confirm scene is fully restored to original state with no orphan datablocks.
