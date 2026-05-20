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
| Texture per layer | Single consolidated RGBA texture (RGB = colour, A = coverage mask for the border material). Multi-view captures used during baking **only** to fill disocclusions visible from head-plane extremes into the canonical projection. No view-dependent textures in V1. Canonical is **super-frustum-sized** (Phase 3 slice 3) so off-centre XR viewers see no angular gaps along the portal interior — centre-view pixels occupy the central sub-rectangle, sibling fills populate the surrounding annulus. |
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
- `core/depth.py` — NDC perspective transform (`ndc_perspective` / `inverse_ndc_perspective`) and `pool_valid_depths`.
- `core/layer_placement/base.py` — `LayerPlacementStrategy` ABC.
- `core/layer_placement/fixed_distribution.py` — user-count + uniform-Z / 1-over-Z / log.
- `core/layer_placement/histogram_hybrid.py` — two histogram-driven strategies sharing pipeline (NDC sampling, CDF, empty-collapse, contiguity pin):
  - `HistogramHybrid` — N/2+1 importance + N/2+1 uniform NDC boundaries merged. Uniform half guarantees baseline coverage. Differs from `docs/reference/hybrid_depth_layers.py` in two intentional ways:
    - The reference's empty-layer-collapse loop fused every contiguous non-empty pair into one layer, so any depth distribution without explicit empty bands collapsed to a single layer. Our port keeps non-empty layers separate; only empty-induced gaps trigger gap-close.
    - Importance-half endpoints are snapped to `ndc_min` / `ndc_max` so dedupe with the uniform-half endpoints fires cleanly (no near-zero-thickness tail layers from float32 imprint).
    - Uniform-half count is bumped by 1 to compensate for endpoint dedupe (otherwise the result has `n_layers - 1` layers).
    - Near-duplicate interior boundaries (within `1/(4·n_layers)` of each other in NDC) are merged.
  - `HistogramImportance` — N+1 boundaries from CDF inverse only. Equal-population layers; no uniform safety net. Skips the close-boundary dedupe (intentional clustering in dense NDC regions is the whole point).
- Rig depth-scope properties on `PortalRigProps`:
  - `min_depth` (metres past the portal; default 0). Pushes the first layer's near boundary into the scene to skip empty near-portal volume.
  - `max_depth` (metres past the portal; default 20). Defines the deepest plane.
  - Layer placement subdivides `[headplane_offset + min_depth, headplane_offset + max_depth]`.
- `DEPTHPEELXR_OT_place_layers` operator:
  - For HYBRID: spawns a temp centre-camera (symmetric super-frustum FoV at head-plane centre), with `clip_start = ho + min_depth` and `clip_end = ho + max_depth`. Renders an EEVEE Z-pass at 256 px (no AA). Compositor uses a temp `Render Layers → Viewer` graph; depth is read directly from `bpy.data.images['Viewer Node']` via `pixels.foreach_get` — no disk I/O. All scene state (engine, resolution, view-layer pass enable, compositor node group, scene camera) saved before the render and restored via `try/finally` on every exit path.
  - HYBRID refuses to place layers if the probe is essentially empty (< 100 valid samples in scope) or essentially flat (depth spread < 1% of scope) — depth-peeling adds no value in either case.
  - Layers are cleared up front on every attempt; failures leave the overlay clean rather than showing a stale preview.
  - Live recompute: changing any of `min_depth`, `max_depth`, `distribution`, `layer_count`, or `strategy` re-runs the strategy. FIXED runs synchronously; HYBRID is debounced ~150 ms via `bpy.app.timers` so slider drags trigger one render on release. Auto-recompute is gated on a hidden `auto_recompute` flag (set by `Analyse & Place`, cleared by `Clean Up Layers`) — independent of layer count, so transient failures don't break live updates.
  - Updates viewport overlay to show **all `N+1` unique layer boundaries** as partially-transparent rectangles, each scaled to match the centre camera's super-frustum projected to that depth. The deepest plane (rig depth scope) is **pink**; all interior planes are **green**. Each plane has a near-black 1.5 px wireframe outline for visibility.
- `DEPTHPEELXR_OT_cleanup_layers`: clears layer list and overlay, clears `auto_recompute` flag.
- `DEPTHPEELXR_OT_auto_fit_depth_scope`: renders a wide-open probe (clip 0.001–1000 m), filters out background sentinels, and snaps `min_depth` / `max_depth` to the actual scene-content extents past the portal. On empty scene, errors out and clears any previous layers.
- N-panel section `Layer Placement`: strategy dropdown, per-strategy params (layer count, spread), `min_depth` / `max_depth` sliders, `Auto-fit to Scene`, `Analyse & Place`, cleanup button.

**Status**: complete.

- **Slice 1 (shipped)** — fixed-distribution end-to-end. `core/layer_placement/{base,fixed_distribution}.py`, scene props, place/cleanup operators, N+1 boundary-plane overlay, N-panel UI, live recompute on param change. 14 pytest cases for the three distributions.
- **Slice 2 (shipped)** — `core/depth.py`, `core/layer_placement/histogram_hybrid.py`, probe operator (EEVEE Z-pass via Viewer Node), debounced HYBRID recompute, auto-fit operator, plus the empty/flat-probe safety thresholds and clear-on-failure model. Diagnostic tooling in `tools/debug_probe.py` and `tools/introspect_compositor_api.py` for headless API spelunking.

**Acceptance**:
- HYBRID on a non-trivial scene produces visibly distinct layer boundaries reflecting the depth distribution; `Auto-fit to Scene` snaps `min_depth` / `max_depth` to the actual content extents.
- Empty / flat / nearly-flat scenes produce zero layers with a clear error report; previous layers are cleared rather than left as a stale preview.
- All `N+1` boundary planes appear, sized to the super-frustum at their depth, deepest pink and the rest green, with dark wireframe outlines.
- Changing strategy or params changes the result; HYBRID slider drags fire at most one render on release.
- Pytest covers `core/depth` round-trip + invalid inputs and `HistogramHybrid` shape, contiguity, dense/bimodal distributions, edge cases, and the gap-close semantics.

**Verification**: numpy unit tests; manual visual check of overlay against scene geometry.

---

### Phase 3 — Multi-view Capture

**Goal**: for each placed layer, capture NxN RGBA + depth passes with the layer's depth slab enforced.

**Scope**:
- `DEPTHPEELXR_OT_capture` modal operator:
  - For each of the NxN cameras × each layer: set `camera.data.clip_start = layer.near`, `clip_end = layer.far`, run **one render** that produces both products via a compositor with two outputs in the same node group: `RL.Image → NodeGroupOutput.Image` (drives Render Result, saved by `write_still=True` to `<project>/captures/layer_NN/view_MM_color.exr` as flat OPEN_EXR) and `RL.Depth → Viewer` (`Viewer Node` pixels drained into numpy, saved as `view_MM_depth.exr` via a throw-away `bpy.data.images.new + save`). Earlier iterations split this into two renders because a non-empty 5.x compositor leaves Render Result unpopulated without a Composite-equivalent output node — `NodeGroupOutput` plus an interface output socket is that replacement, and probing showed it co-exists cleanly with a `Viewer` node in the same group. Multilayer File Output is still avoided (its EXRs aren't readable per-pass via `Image.pixels`).
  - Engine selector (Cycles / EEVEE), default EEVEE for fast iteration.
  - Resolution driven by `viewing_resolution` from PortalRigProps, with the secondary dimension derived from portal aspect.
  - Auto-wipes `captures/` only when the existing layout (existing files outside the expected `layer_NN/view_MM_{color,depth}.exr` set) doesn't match the upcoming run — protects Phase 4 from picking up stale orphans after `grid_n` or `layer_count` changes. Layout-matching content is overwritten cleanly without wipe; the `consolidated/` subtree is treated as owned by `consolidate` and excluded from the orphan scan.
  - State save/restore: render engine, samples, resolution, view-layer Z-pass, film transparency, scene compositor node group, scene.camera, scene render filepath / image-format / color-depth / color-mode, every rig camera's clip range. `_restore_state` is idempotent and runs from FINISHED, CANCELLED, exception, and Blender's `cancel()` callback.
  - ESC sets a deferred-cancel flag — `bpy.ops.render.render` is fully synchronous, so cancellation actually triggers at the next view boundary (i.e., after the current view's color+depth pair completes). Header text shows `(cancelling…)` between request and effect.
- `DEPTHPEELXR_OT_cleanup_captures`: deletes the captures subfolder (which transitively removes `consolidated/`).
- `core/multi_view.py` — reproject each sibling view's RGBA into the canonical centre view via `(ndc, depth) → world → centre-ndc` and composite where the centre view's coverage is missing. Produces `(rgba_canonical, depth_canonical, coverage_mask, source_mask)` per layer.
- N-panel section `Capture`: engine dropdown, "X queued / N done" status, `Run Capture`, `Consolidate Views`, `Clean Up Captures`.

**Status**: slice 1 shipped, slice 2 shipped, slice 3 shipped.

- **Slice 1 (shipped)** — capture operator + cleanup + N-panel + auto-wipe-on-layout-change. Per (layer, view), two flat-OPEN_EXR files: color via standard render path (`bpy.ops.render.render(write_still=True)`, no compositor active) and depth via RL→Viewer compositor drained into numpy and written through a throw-away `bpy.data.images.new`. Rendered with `film_transparent=True` so `Combined.Alpha` encodes per-pixel coverage. Modal with deferred-ESC cancel (cancellation lands between views). Plus an early-staged **Reconstruction** section (Phase 4's section, populated ahead of schedule with slice-1's flat-cutout preview): per-layer alpha-cut plane at `layer.near`, sized to the centre camera's frustum at that depth, textured from the centre-view EXR (Combined.RGB → unlit emission, Combined.Alpha → alpha clip). Auto-invalidates whenever any rig / layer-placement param changes, the layers collection is rewritten, capture re-runs, captures are cleaned, or the rig is torn down — the preview only ever reflects state matching the captures on disk. Building the preview also forces the Source/Preview toggle into "preview view" (Source hidden, Preview shown); invalidation flips it back. Layer-boundary gizmos draw if the Source collection is visible, so they hide as a side-effect of the toggle. Shared math in `core/geometry.py` (super- and centre-frustum extents).
- **Slice 2 (shipped)** — `core/multi_view.py` consolidation (pure numpy, no bpy): for each layer, reproject sibling views into a canonical NDC via per-pixel unproject+reproject through the cameras' projection matrices (clip-range-invariant). Closest-to-centre wins for sibling overlap; nearest-neighbour forward splat with depth-buffered tie-break inside each sibling's scatter. Outputs `consolidated/layer_NN_{color,depth,source}.exr` per layer (flat OpenEXRs, written via `bpy.data.images.new + save`). `DEPTHPEELXR_OT_consolidate_views` is the post-capture operator (separate button under Capture; iteration-friendly — re-run consolidation without re-rendering). Reads `view_MM_{color,depth}.exr` flat EXR pairs (`Image.pixels.foreach_get`) and per-camera projection matrices built manually from intrinsics (Blender 5.x dropped `Camera.calc_matrix_camera`). Slice 2 shipped with centre-NDC canonical sizing; slice 3 widened the canonical to super-frustum.
- **Slice 3 (shipped)** — **super-frustum canonical**. `core/geometry.py` gained `super_frustum_projection` (virtual perspective apex at `z = -ho/2` in centre-cam space, framing the union of NxN frustums) and `super_frustum_canonical_shape` (per-layer buffer dims via the super/centre half-extents ratio at `layer.near`, caps at 2× as `layer.near → ∞`). `core/multi_view.py` generalised: `reproject_view_to_centre` → `reproject_view_to_canonical`; `consolidate_layer` now takes one `views_sorted` list (centre first by convention, then siblings closest-to-centre first) + canonical shape + canonical matrices. Centre stopped being a special-cased seed copy and goes through the same forward-splat path. `operators/consolidate.py` builds the canonical projection once from rig geometry, computes per-layer canonical buffer dims, and feeds the unified view list. `operators/reconstruct.py` (`Build Quadtree Planes`) swapped `centre_frustum_half_extents` → `super_frustum_half_extents` for plane sizing. `core/reconstruction.py`'s `build_layer_mesh` kwargs renamed from `half_w_centre_frustum` / `half_h_centre_frustum` → generic `half_w_plane` / `half_h_plane`. New `tests/test_geometry.py` (10 cases) covers the projection round-trip; `tests/test_multi_view.py` extended with an annulus-fill test from an off-axis sibling and a centre-region-landing test. `Build Simple Planes` deliberately unchanged — stays centre-only, centre-frustum-sized, as the pre-consolidation capture sanity-check (locked-in per the reconstruction-fidelity hierarchy). Known artefacts of the virtual-apex framing: (a) centre-view content captured past `layer.near` within each layer appears slightly compressed toward image centre in `Build Quadtree Planes` vs `Build Simple Planes` (depth-dependent NDC mismatch intrinsic to flat impostors with a forward-shifted apex), (b) corner annulus regions look stretched because off-axis siblings forward-splat sparsely at the canonical edges. Both are addressed by Phase 4 slice 2 displacement (per-vertex depth removes the depth-collapse and lets the quadtree route content to its true world position). Other variants explored during slice-3 polish — centre-cam apex, angular canonical, per-layer angular extents, bilinear forward splat, centre-EXR-only rollback — were measurably worse in their own ways; see *Multi-view consolidation: explored alternatives* under Open items.

**Acceptance**:
- N=3, L=4 produces 72 flat OPEN_EXRs at `<project>/captures/layer_NN/view_MM_color.exr` and `view_MM_depth.exr`. Each is `OPEN_EXR` (not multilayer), float32, scene-linear. Color is RGBA Combined; depth packs the Z pass into the R channel.
- Consolidation produces 3 flat EXRs per layer at `<project>/captures/consolidated/layer_NN_{color,depth,source}.exr`. Closest-to-centre sibling wins for pixels the centre view doesn't cover; uncovered pixels get alpha=0 and depth=0.
- Depth-slab clipping works in both engines (inspect outermost layer: nothing nearer than `near`, nothing further than `far`).
- Cleanup removes the captures directory (transitively wiping `consolidated/`).
- ESC during capture leaves the scene in a sensible state (cameras / engine / compositor / image-settings restored, partial outputs on disk OK; cancellation lands between view boundaries since each view is a two-render unit).
- Flat-cutout preview shows L alpha-cut planes stacked at each layer's near depth, silhouettes match captured content, depth-sort correct between layers, unlit (visible regardless of scene lighting). Tweaking any rig or layer-placement param while preview is visible → planes disappear immediately, no error spam. Cleanup leaves no orphan datablocks. Re-running preview without cleanup first refreshes in place.

**Gotchas (Blender 5.x API recon)**:
- Compositor API renamed from 4.x: scenes hold `compositing_node_group` (a NodeGroup asset) — `scene.use_nodes` / `scene.node_tree` are gone. File Output node uses `directory` / `file_name` (was `base_path`) and `file_output_items.new(socket_type, name)` (was `file_slots.new`); only `OPEN_EXR_MULTILAYER` format is exposed (use socket_type `RGBA` for color, `FLOAT` for single-channel passes).
- `CompositorNodeComposite` was removed in 5.x. The replacement when the compositor is a NodeGroup is `NodeGroupOutput` plus an interface output socket. A non-empty compositor without one of these doesn't populate `Render Result` — `bpy.ops.render.render` still returns `FINISHED`, but `Render Result.pixels` stays empty.
- **Multilayer EXR → Python is one-way**. `Image.pixels.foreach_get` on any `type='MULTILAYER'` image returns "low-quality preview pixels" (empty in practice — `size=(0,0)`, `channels=0`). Even single-slot multilayer files written by File Output are unreadable. The only Python-friendly EXR format is flat `OPEN_EXR`, which the compositor can't write — so for any pass you need to read back, route it through a Viewer node and drain `bpy.data.images['Viewer Node']` into numpy.
- Mutating live render-pipeline images breaks subsequent renders: `viewer.save()` sets the Viewer image's `filepath_raw` and marks it saved-to-disk, after which subsequent renders silently don't refresh it. `result.save_render()` has similar side effects on `Render Result`. Always copy pixels out first via `pixels.foreach_get`, then write through a throw-away `bpy.data.images.new`.
- `Camera.calc_matrix_camera(...)` removed in 5.x. Build the projection matrix manually from intrinsics (lens, sensor_width, shift_x/y, clip range) as an off-axis perspective frustum.
- Image data block has no `image_user` attribute. That's a property of consumers (ShaderNodeTexImage, CompositorNodeImage, etc.). Single-layer or multilayer, there's no way to "select which pass" via the Image directly.
- `view_layer.use_pass_z = True` must be set before `rl.outputs["Depth"]` is name-lookable — the socket only appears when the pass is active.
- Workbench engine does not populate the Viewer node; stick to BLENDER_EEVEE for compositor-based depth reads.
- Render engine enum is `BLENDER_EEVEE` in 5.x (not `BLENDER_EEVEE_NEXT`).
- `bpy.ops.render.render(write_still=False)` is fully synchronous and uninterruptible from Python — modal operators can only cancel between renders.

**Verification**: manual render of a known scene; compare depth pass to analytic expectation; visually inspect consolidated textures.

---

### Phase 4 — Reconstruction + Preview

**Goal**: build per-layer adaptive quad meshes with interior/border subsets, populate `DepthPeelXR_Preview` collection, preview is immediately visible.

**Reconstruction-fidelity hierarchy** — three tiers, each strictly supersedes the previous as a quality choice but the previous remains as a faster/cheaper diagnostic. All consume Phase 3 outputs; only the centre-view sanity-check tier skips slice-2 consolidation.

1. **Flat cutouts** (Phase 3 slice-1 polish — shipped). Four verts per layer at `layer.near`, full-rect alpha-test from the raw centre-view EXR. No consolidation needed. Stays as the cheapest capture sanity-check; not a reconstruction option.
2. **Flat impostors with quadtree** (Phase 4 slice 1 — new). Quadtree-subdivided plane per layer at `layer.near` with proper silhouette-cut borders, no per-vertex depth. Coverage-mask drives subdivision + border/interior classification; vertices stay coplanar. Cheapest *shippable* reconstruction — useful when downstream doesn't need (or can't afford) displacement.
3. **Displaced quadtree meshes** (Phase 4 slice 2). Same quadtree from slice 1, with per-vertex depth displacement on top via inverse projection. Full parallax inside each layer. The original Phase 4 goal.

**Scope**:
- `core/quadtree.py` (slice 1) — adaptive quadtree:
  - Root = portal-frustum quad sized to the centre camera's frustum cross-section at `layer.near`.
  - Split rule: if quad straddles coverage-mask edge **and** quad size > `min_quad_size`, subdivide into 4; (slice 2 adds:) also subdivide if depth variance within the quad exceeds `depth_gradient_tol`, down to `core_quad_size`.
  - Leaves classified `interior` (fully covered) or `border` (straddles).
- `core/reconstruction.py`:
  - Slice 1: flat vertex placement at `layer.near` in portal-local coords.
  - Slice 2: each leaf vertex's world position via `inverse_projection(ndc_x, ndc_y, depth_canonical)` — avoids spurious curvature at wide FoV.
  - Build interior and border face arrays.
  - UVs: screen-space passthrough from centre NDC.
  - Border quads: rasterise coverage mask into RGBA texture's alpha channel for that quad's UV region.
- `DEPTHPEELXR_OT_reconstruct_flat` (slice 1), `DEPTHPEELXR_OT_reconstruct_displaced` (slice 2):
  - For each layer: create `bpy.types.Mesh` with two geom-subset-equivalent vertex groups, build emission + alpha-clip materials (interior + border share the canonical RGBA, border has the per-quad coverage rasterised), spawn as Objects in `DepthPeelXR_Preview` collection.
  - Move original scene objects into `DepthPeelXR_Source` collection (extends slice-1 polish behaviour to also reparent geometry, not just toggle visibility).
  - Auto-switch visibility to Preview on completion (existing slice-1 mechanism).
- `Clean Up Geo` operator (already forward-named) extends to cover the new mesh kinds — name prefix scheme generalised from `dpxr_preview_layer_*` to `dpxr_recon_layer_*` once Phase 4 lands, so cleanup walks both prefix families.
- N-panel `Reconstruction` section gains: `Quadtree params` group (`core_quad_size`, `min_quad_size`, slice-2 adds `depth_gradient_tol`), `Reconstruct Flat`, `Reconstruct Displaced`, all behind the same `Clean Up Geo` and source-toggle plumbing.

**Status**: slice 1 shipped, slice 2 shipped. Depends on Phase 3 slice 2 (`consolidated/layer_NN_*.exr`) and slice 3 (super-frustum canonical).

**Slice 1 (shipped)** — flat-quadtree reconstruction. `core/quadtree.py` runs the silhouette-driven subdivision (no depth-variance splits yet); top-level tile loop divides the canonical into MxM square tiles before the 4-way subdivision runs within each, so leaves come out square (the previous single-root layout inherited the canonical's portal-aspect ratio → rectangular world quads). `core/reconstruction.py` builds mesh arrays with non-shared vertices per leaf (no manifold needed at slice-1 flatness). `DEPTHPEELXR_OT_reconstruct_flat` (UI label `Build Quadtree Planes`) loads the canonical `consolidated/layer_NN_color.exr` per layer, derives coverage from its alpha, runs quadtree + mesh-build, spawns `dpxr_recon_layer_NN` objects into `DepthPeelXR_Preview` with interior/border vertex groups and a shared emission+alpha-clip material (reused from `Build Simple Planes`). `Clean Up Geo` walks both `dpxr_preview_layer_*` and `dpxr_recon_layer_*` prefixes — the two geo flavours are mutually exclusive. N-panel `Reconstruction` section: `Min Quad Size` and `Max Quad Size` (both `IntProperty` with power-of-2 snap — ±1 slider-arrow deltas step pow2, arbitrary jumps snap to nearest), `Build Simple Planes`, `Build Quadtree Planes`, `Clean Up Geo`; greying separates "simple ready" (captures present) from "quadtree ready" (consolidated present). Capture's `viewing_resolution` snaps to pow2 too (EnumProperty: 512/1024/2048/4096) and intentionally does NOT invalidate the reconstruction on change (only affects future captures' scene render res; existing captures and the on-screen geo stay valid). `Consolidate Views` invalidates the on-screen reconstruction at run start (re-consolidation writes new canonical EXRs → existing geo stale). 14 pytest cases cover the quadtree split rules and the mesh-builder coordinate mapping. Plane sized to the super-frustum cross-section at `layer.near` (super_frustum_half_extents) — `build_layer_mesh` takes generic `half_w_plane` / `half_h_plane` kwargs.

**Slice 2 (shipped)** — displaced quadtree reconstruction with shared verts + bridge triangles. New operator `DEPTHPEELXR_OT_reconstruct_displaced` (UI label `Build Displaced Quadtree`, RNDCURVE icon). End-to-end:

- `core/geometry.py` gained `inverse_canonical_projection(u, v, d, half_w_p, half_h_p, ho)` — canonical apex at portal-local `(0, ho/2, 0)`, near plane at the portal; output is portal-local `(x, y, z)` for canonical NDC at canonical-cam-frame depth `d` (= the value stored in `consolidated/layer_NN_depth.exr`).
- `core/quadtree.py` extended with optional `depth` + `depth_tol` args. When both are passed, fully-covered leaves whose `max - min` depth across the leaf exceeds `depth_tol` subdivide further (down to `min_quad_size`). Slice-1 silhouette-only path preserved exactly when either arg is None.
- `core/reconstruction.py` added `build_displaced_layer_mesh` — pass-1 hashes leaf corners by canonical pixel coord (vectorised via `np.unique(corners, axis=0, return_inverse=True)`; per-row/per-column lookup tables built by `np.lexsort` group-by). Pass-2 fan-triangulates each leaf's polygon with mid-edge verts owned by smaller neighbours inserted in pixel-CCW order. Fast-path leaves (no T-junctions, detected via cumsum range queries on a `has_vert` grid) skip the Python polygon walk and emit two tris per leaf in one vectorised `np.empty + assignment`; T-junction leaves keep the per-leaf Python polygon walk. Per-corner depth via MIN over the 2×2 covered stencil, dilating to 4×4 if the inner stencil is empty; invalid corners fall back to the leaf's own valid-corner mean (averaged across owning leaves for shared invalid corners), then layer-mean as ultimate fallback. Final pass: every border-classified leaf snaps all 4 corners to the leaf's MIN over its own covered cells — silhouette cards stay co-planar in 3D regardless of stencil noise across corners. Mesh winding is pixel-CCW so face normals point portal-local +Y (toward the head-plane viewer); also corrected the slice-1 quadtree builder and `Build Simple Planes` to the same convention so all three reconstructions face the viewer.
- `core/multi_view.py` `consolidate_layer` runs `_remove_single_pixel_anomalies` as a final cleanup pass: (1) source-disagreement specks (covered pixel whose source view differs from all 4 covered neighbours) → 3×3 median replace; (2) morphological opening (3×3 box, OOB-treated-as-covered for erosion) drops covered blobs ≤ 2×2 in negative space; (3) morphological closing fills 1-pixel holes / thin lines (typically depth-plane cuts through scene geometry). Synthesised pixels marked `source = 255` sentinel so `layer_NN_source.exr` can be inspected to see which pixels were cleaned up.
- `properties.py` adds `depth_gradient_tol` FloatProperty (metres, default 0.1, unit=LENGTH) on `ReconstructionProps`.
- `operators/reconstruct.py` ships the new operator alongside `reconstruct_flat`. Mesh handoff to bpy uses bulk APIs: `mesh.from_pydata(verts.tolist(), [], tris.tolist())` then `uv_layer.data.foreach_set("uv", flat_uvs)` instead of per-loop assignment. Operator runs as a per-layer modal — one layer per timer tick, `wm.progress_*` cursor, header text `"Depth Peel XR — building displaced layer N/M — ESC to cancel"`, deferred-cancel between layers. Profile log written to `<project>/temp/displaced_profile.log` (per-layer load / quadtree / core_mesh / bpy_handoff timings + total).
- `operators/consolidate.py` converted to modal so the cleanup pass doesn't block the UI: per-layer timer ticks, `wm.progress_*` cursor, header text `"Depth Peel XR — consolidating layer N/M — ESC to cancel"`, deferred-cancel.
- `ui/n_panel.py` adds `Depth Tol` slider + `Build Displaced Quadtree` button under Reconstruction (gated on consolidated/ presence).

Performance: at the point of slice-2 ship, end-to-end build takes ~12s for a 12-layer scene with ~250k total leaves on a portal-resolution 4K canvas; ~2× of that is the core-mesh phase. The cumulative perf path was: vectorised pass-1 corner hashing (np.unique), bulk UV write via foreach_set, then fast-path triangulation for the no-T-junction leaves, which together brought core_mesh down ~3× from the initial Python-list-append implementation.

Open quality items deferred (still tunable via `Min Quad Size` / `Max Quad Size` / `Depth Tol`): residual mid-edge depth discontinuities at T-junctions where the small leaf's mid-edge vertex spans a sharp depth jump; widening morphological closing to a 5×5 SE for 2-pixel-wide depth-plane cut gaps. Border-flatness pass is the remaining Python loop in core_mesh and is the next perf target if needed.

**Acceptance**:
- Slice 1: silhouettes cleanly follow scene edges (alpha-cut border quads); interior of a flat wall coalesces to coarsest; head-plane-corner view shows correct *occlusion silhouettes* (no parallax expected — flat).
- Slice 2: depth-gradient subdivision produces finer interior quads in high-variance regions; head-plane-corner view shows correct *parallax* inside each layer; wide-FoV flat-wall test stays planar (verifies inverse-projection maths).
- Both: source moves into `DepthPeelXR_Source`; preview appears in `DepthPeelXR_Preview`; toggle operator swaps them; `Clean Up Geo` restores scene to pre-reconstruction state.

**Verification**: numpy unit test on synthetic coverage masks + depth maps with known cut edges; manual visual test in Blender.

---

### Phase 5 — Occlusion Culling

**Goal**: cull quads invisible from every head-plane viewpoint to reduce final poly count and overdraw.

**Scope**:
- `core/occlusion.py`:
  - For each layer's quads (walk far-to-near), test visibility by rasterising closer-layer interior meshes into each of the NxN views' depth buffers.
  - A quad is culled if all NxN views report it fully occluded (within an epsilon).
  - Vectorised numpy implementation; no bpy dependency.
- `DEPTHPEELXR_OT_occlusion_cull` operator: runs the cull on the existing preview meshes, prints stats (quads removed per layer, % reduction).
- `DEPTHPEELXR_OT_undo_cull`: rebuilds from reconstruction cache (so user can re-enable culled quads without a full rebake).
- N-panel section `Optimisation`: `Cull Occluded Quads` toggle + button, `Undo Cull` button.

**Status**: slice 1 in working tree, sign-off pending. Not yet committed.

**Slice 1 (working tree, sign-off pending)** — per-vertex cull with opaque-mesh closer-surface (cheap, conservative).

*Reconstruction split.* Both `Build Quadtree Planes` and `Build Displaced Quadtree` now emit two mesh objects per layer: `dpxr_recon_layer_NN_opaque` (interior leaves, emission-only material — early-z friendly, runtime-perf win) and `dpxr_recon_layer_NN_alpha` (border leaves, emission + alpha-clip). `core/reconstruction.py` gained `split_layer_mesh` / `split_displaced_layer_mesh` to partition mesh arrays by leaf class, with vert compaction (`split_displaced_layer_mesh` reuses `filter_displaced_mesh`). Verts shared between an interior and border leaf appear in both halves at coincident world positions — border-flatness already snapped the shared corner to the foreground depth, so there's no visual seam. The slice-1-shipped vertex groups (`interior` / `border`) are gone — selection by object name in the Outliner replaces them.

*Cull pipeline.* `core/occlusion.py` (`cull_layer`) — pure-numpy per-vert visibility test against per-cam closer-surface depth buffers; unchanged from the initial slice-1 working-tree state. `core/raster.py` (new) — pure-numpy software depth rasteriser (`rasterise_depth`, perspective-correct via 1/w interpolation; `fan_triangulate_polygons` for converting variable-length DisplacedLayerMesh polygons to tris). Triangles whose verts straddle the near plane are skipped (conservative under-cull at slab boundaries, fine for the cull goal). 11 pytest cases.

*Cache.* `operators/recon_cache.py` is a module-level `dict[base_name → LayerCachePair(opaque, alpha)]`, populated by both `Build Quadtree Planes` (slice-1 `LayerMesh` halves repackaged via `layer_mesh_to_displaced`) and `Build Displaced Quadtree`, cleared by `Clean Up Geo`.

*Operator.* `operators/occlusion_cull.py` walks layers near→far. At each layer L > 0 it culls **both** halves against the accumulated `closer_per_cam`, then rasterises **only** layer L's opaque half from each cam and per-pixel-MIN merges into `closer_per_cam` for downstream layers. Captured-EXR alpha-thresholded closer-surface (the original slice-1-state path) is gone entirely — the opaque half *is* the geometry that owns pixels at runtime, so its silhouette is the correct occluder by construction. Cull works on both reconstruction tiers via the unified `DisplacedLayerMesh` cache; on flat-quadtree the closer-surface sits at each layer's `near` depth (rather than actual scene depth), so the cull is conservative — it under-culls vs. displaced but still drops fully-occluded layers, useful for the cheaper recon tier. Undo Cull rebuilds both halves from the cached arrays via the polygon-based mesh path (geometrically equivalent to the original flat-quadtree build). The cull runs as a single-tick modal — invoke sets the header `"Depth Peel XR — culling occluded quads…"`, returns RUNNING_MODAL so Blender redraws once, then the first TIMER tick runs the entire cull synchronously in one block. The single-tick form avoids the per-layer modal overhead (per-tick redraws would be ~25-40% of total time on a fast cull) while still surfacing what's running during the freeze. UI: `Optimisation` section under Reconstruction, gated on the cache being populated. Cache lives in module memory only — reload Blender → rebuild before culling.

Per-vert was picked over per-pixel for slice-1 simplicity. Conservative: misses the rare case where a quad's 4 corners are all hidden but its interior is visible (closer-surface bumps non-monotonically inside the quad's projection).

Cull's closer-surface depth buffer runs at `_CLOSER_SURFACE_SCALE` (currently 0.25) of render resolution to keep the per-cam, per-layer rasterise cheap. Slight under-cull at silhouette edges in exchange for a smaller buffer; the bulk of dropped polys is unaffected. Camera projection matrices are unchanged — they encode aspect, preserved when both axes scale equally.

**Cull rasteriser perf — future optimisation.** The rasteriser in `core/raster.py` is a Python `for tri in triangles` loop with numpy vectorisation only *within* each tri's bbox. Lowering the closer-surface resolution helped only modestly, which means the per-tri Python iteration overhead dominates the pixel-fill cost. The right next step (when cull wall time becomes a problem) is to vectorise across tris — chunked or padded per-tri bbox arrays + masked fill, so numpy handles the whole tri batch in one pass. Pure-numpy, stays in `core/`, no quality regression. Estimated 5-10× over the current per-tri loop.

**Slice 2 (deferred)** — per-pixel rasterizer for the cull's visibility test (catches the corner-hidden-but-interior-visible case). The architecture (rasterised opaque-half closer-surface) extends naturally: the cull would rasterise each layer's full mesh (opaque + alpha) at projected vert depths and z-test the result.

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

### Quality refinements (post-slice-3, prioritised — feathered slabs first)

- **Feathered slab overlap (inter-layer seam fix).** Capture is strict per-slab (`clip_start=layer.near`, `clip_end=layer.far`), so geometry straddling a layer boundary has no overlap between the two captures' silhouettes. Even with slice-2 displacement, hairline gaps can appear at extreme portal viewpoints where the two edges don't agree. Fix: capture with a small `epsilon` past each slab's `clip_end` and feather the canonical alpha across the overlap so adjacent layers cross-fade. Touches capture (extra clip headroom + alpha falloff bake) and reconstruction (border quads honour the feather).
- **Corner annulus stretching (intra-layer sampling fix).** `core.multi_view.consolidate_layer` forward-splats one source pixel per canonical pixel; off-axis siblings projecting into the canonical's outer annulus end up sparse, which the quadtree alpha-thresholds into blocky/stretched border quads at the diagonals. Fix candidates ranked: (1) backward (gather) resample — invert the loop direction, per canonical pixel pick the best-fitting view and bilinearly sample it; (2) bilinear forward kernel — each source pixel splats with weights into 2×2 neighbours, needs a weighted accumulator instead of overwrite; (3) higher sibling capture resolution (brute force, linear cost); (4) morphological dilation of the canonical alpha before the quadtree (cheap band-aid, degrades silhouettes). (1) is the principled fix. (2) was prototyped during the angular-canonical experiment and closed pixel-level holes but did not eliminate the within-layer geometric distortion that variant suffered — see *explored alternatives* below.
- **Whole-frame texture quality (sharpness + aliasing).** Same root cause as the annulus stretching above — NN forward-splat point-samples the source, so silhouettes alias and interiors look softer than the source actually is. The principled fix is the same as candidate (1) above: a two-pass reverse-mapping consolidation. Pass 1 keeps today's forward-splat for *depth only*; pass 2 walks each canonical pixel, uses that depth to backproject to world, projects into every contributing source view, and **bilinear-samples the source colour** (closest-to-centre wins for source priority, same as today). Bilinear at the source = sharp silhouettes (no NN stairsteps) + proper sub-pixel anti-aliasing on flat surfaces; reverse direction also fills divergent-mapping holes the forward splat leaves. Cost: one extra numpy pass at canonical resolution, comparable to current consolidation. Risk: depth errors from forward-splat propagate into colour misalignment in pass 2 — single-pixel-anomaly cleanup mitigates, extreme depth discontinuities to verify on a real scene.
- **Robust depth-variance metric (quadtree fix).** `core/quadtree.py:_depth_within_tol` currently uses `max - min` to decide whether a fully-covered quad subdivides further. That's the most extreme metric possible — single-pixel depth-EXR outliers (common at silhouette boundaries from forward-splat consolidation) trip the variance check on regions that are visually flat. Fix: swap `max - min` for `percentile(95) - percentile(5)` (or similar — `np.percentile` on the region's depth slice). One-line change, absorbs outliers without losing real depth-jump detection, no new tunables. Principled-but-bigger alternative: fit a plane to the depth and measure residual — correctly identifies tilted-but-flat surfaces a `max - min` metric calls "high variance". Probably overkill until the percentile fix proves insufficient.

### Multi-view consolidation: explored alternatives

Variants prototyped after slice 3 shipped, all measurably worse than the virtual-apex super-frustum baseline and discarded. Recording here so we don't relearn the same lessons.

- **Centre-cam apex with widened FoV.** Move the canonical's projection apex back to the centre camera origin and widen the FoV so NDC ±1 still frames the super-frustum at `layer.near`. Goal: eliminate the slice-3 centre-cam shift artefact (centre pixel → constant canonical NDC regardless of depth). Outcome: shift gone, but sibling content past `(super_w_at_near / layer_near) · d` at `d != layer_near` gets clipped, leaving more uncovered annulus pixels than the virtual-apex variant. The quadtree alpha-threshold then carves *more* border quads around the clipped edges. User-visible verdict: "tearing is increased" — annulus holes were more disruptive than the centre-cam shift they fixed.
- **Equirectangular angular canonical.** Replace the perspective canonical projection with an angular `(θ, φ)` reprojection (each canonical pixel = a centre-cam angular direction). Goal: angular space has no FoV cap, so no clipping; centre-cam pixels project to constant angular position regardless of depth. Forward-splat math: `θ = atan2(x, −z)`, `φ = atan2(y, √(x²+z²))`, map linearly to canonical pixels. Outcomes: (a) initial implementation had a missing `1/cos(θ)` factor in the inverse `(θ, φ) → flat plane` mapping (yaw-then-pitch is horizontal-symmetric but vertical-asymmetric) — produced visible barrel distortion top/bottom + pinch left/right in the centre cam view; (b) after the `/cos(θ)` fix, within-leaf distortion remained: the rasterizer linearly interpolates vertex positions across each quadtree leaf, but `tan(θ)·near` and `tan(φ)·near/cos(θ)` are nonlinear in pixel position, so default-size leaves at canonical edges showed a few-percent world-position drift per fragment; (c) centre-cam pixel density (perspective) doesn't match angular pixel density (equirectangular) — forward-splat NN leaves holes at the centre cam's optical axis where canonical density > source density. User-visible verdict: "very distorted quads" and "alpha holes in the layers". The natural fix for (b) would be a curved (spherical) reconstruction surface where each leaf sits on the angular sphere itself — recorded under *future approaches* below.
- **Per-layer angular extents.** Refinement on the angular variant: instead of the asymptotic angular bound `atan(2·half_w_p/ho)`, use the per-layer reach `atan(half_w_p · (1/ho + 1/layer_near))`. Goal: tighten the canonical to what siblings actually reach at each layer's depth, eliminating empty corners. Outcome: fewer empty corners, but the underlying flat-plane + equirectangular geometric mismatch was unchanged. Did not improve the within-leaf distortion or density-mismatch holes.
- **Bilinear forward splat.** Replace NN forward splat in `reproject_view_to_canonical` with bilinear scatter (each source pixel writes weighted contributions to 4 canonical neighbours via `np.bincount`). Goal: close single-pixel alpha holes that NN leaves at scale mismatches. Outcome: holes mostly gone, fewer phantom border quads in the quadtree, ~2–4× slower consolidation. But the within-leaf distortion (angular variant) and depth-collapse (any flat-impostor) remained. Bilinear stays available as a candidate for the *Corner annulus stretching* fix above (cheaper than backward resampling, blurrier).
- **Centre-EXR-only rollback.** Strip consolidation entirely; route `Build Quadtree Planes` to read centre-view EXRs directly (same source as `Build Simple Planes`) with adaptive quadtree subdivision based on captured alpha. Outcome: centre-cam match is pixel-perfect (it IS the centre data), zero consolidation artefacts, much simpler codebase. Cost: no sibling fill at all — off-axis XR viewers see the same coverage envelope as Simple Planes. Briefly committed (commit dropped via `git reset` because the slice-3 baseline ended up the preferred working state).

### Future reconstruction approaches considered

- **Curved (spherical) reconstruction surface paired with angular canonical.** Each layer's leaves placed on a sphere of radius `layer.near` at the canonical pixel's `(θ, φ)`. Would eliminate the within-leaf distortion of the flat-plane + angular pairing because the chord-quads approximate the sphere naturally. Doesn't fix inter-layer parallax tearing (still flat-impostor per layer in radial terms); also doesn't fix the centre-cam/angular density mismatch. Worth revisiting only alongside slice-2 displacement, where it becomes radial-displacement-on-the-sphere and the artefacts converge with the planar variant.
- **Backward (gather) resample at consolidation.** Per canonical pixel, find the best-fitting source view and bilinearly sample. Eliminates all forward-splat artefacts (density mismatch, sparse splat in annulus, hole patterns). Listed under *Corner annulus stretching* above as the principled fix. Heavier to implement than bilinear forward splat but cleaner output.

## End-to-end verification plan

Once all feature phases (1–6) are implemented on top of the Phase 0 dev environment:

1. **Synthetic flat-wall test**: single flat wall → one layer, no border submesh, no curvature artefacts.
2. **Occlusion test**: closer wall in front of farther wall → two layers, disocclusion filled from sibling views, farther-layer covered region culled.
3. **Wide-FoV curvature test**: flat wall with 120° portal FoV → reconstructed mesh stays planar.
4. **Round-trip test**: Blender preview vs. USD reload matches.
5. **Target-engine handshake**: export USD, load in user's engine, verify visual parity.
6. **Cleanup test**: run all stages, hit `Clean up all`, confirm scene is fully restored to original state with no orphan datablocks.
