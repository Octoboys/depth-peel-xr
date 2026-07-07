"""Reconstruction operators.

`DEPTHPEELXR_OT_reconstruct_flat` ("Build Quadtree Planes"): flat-
quadtree planes per layer at `layer.near`. Silhouette-driven
subdivision; no displacement.

`DEPTHPEELXR_OT_reconstruct_displaced` ("Build Displaced Quadtree"):
the same quadtree augmented with depth-variance subdivision, plus per-
vertex displacement via inverse super-frustum projection. Shared corner
verts with bridge-triangle T-junction handling — adjacent leaves at
different subdivision levels stitch through fan triangulation so the
mesh stays watertight at depth discontinuities.

Both consume `consolidated/layer_NN_color.exr` (RGB texture + alpha
coverage); the displaced operator also reads `consolidated/layer_NN_depth.exr`
(canonical-cam-frame depth). Mutually exclusive: each operator wipes
existing reconstruction objects before building. The shared
`Clean Up Geo` operator (`_remove_preview_objects`) covers both because
they both spawn under `dpxr_recon_layer_*`.

Each layer emits two mesh objects: `<base>_opaque` (interior leaves,
emission-only, early-z friendly) and `<base>_alpha` (border leaves,
emission + alpha-clip). The opaque half acts as the cull's closer-
surface source and saves alpha-test cost on the runtime for the bulk of
each layer's coverage.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import bpy
import numpy as np
from bpy.types import Operator
from mathutils import Matrix

from .. import paths as paths_mod
from ..core.geometry import super_frustum_half_extents
from ..core.quadtree import build_quadtree
from ..core.reconstruction import (
    DisplacedLayerMesh,
    LayerMesh,
    build_displaced_layer_mesh,
    build_layer_mesh,
    layer_mesh_to_displaced,
    split_displaced_layer_mesh,
    split_layer_mesh,
)
from .inspect import PREVIEW_COLLECTION, SOURCE_COLLECTION
from .portal_rig import ensure_collection, link_unique
from .preview import (
    _build_opaque_material,
    _build_preview_material,
    _load_layer_image,
    _remove_preview_objects,
)
from .recon_cache import set_layer_cache


_RECON_PREFIX = "dpxr_recon_layer_"
_OPAQUE_SUFFIX = "_opaque"
_ALPHA_SUFFIX = "_alpha"

# Vertex group holding the verts safe for a Decimate-style modifier to
# collapse — everything EXCEPT the interior/border shared edge. Named
# from the group's semantic ("this is the area to decimate") so a user
# who adds Decimate manually and picks the group by name gets correct
# pinning of the seam without also needing to toggle Invert Vertex Group.
DECIMATE_AREA_GROUP = "dpxr_decimate_area"
_RECON_TIMER_INTERVAL = 0.05
# Canonical-alpha threshold above which a pixel counts as covered for
# quadtree subdivision. Consolidation packs anti-aliased silhouette alpha
# at edges, so we sit comfortably above the 0/1 boundary.
_COVERAGE_THRESHOLD = 0.5


def _consolidated_dir(context):
    captures = paths_mod.captures_dir(context)
    return captures / "consolidated" if captures else None


def _layer_color_exr(consolidated: Path, layer_idx: int) -> Path:
    return consolidated / f"layer_{layer_idx:02d}_color.exr"


def _layer_depth_exr(consolidated: Path, layer_idx: int) -> Path:
    return consolidated / f"layer_{layer_idx:02d}_depth.exr"


def consolidated_present(context) -> bool:
    """True if at least the first layer's consolidated colour EXR exists.
    Consolidation writes layers atomically, so the first file is a
    reliable proxy for the rest."""
    consolidated = _consolidated_dir(context)
    if consolidated is None or not consolidated.exists():
        return False
    scene = context.scene
    lp = scene.dpx_layer_placement
    if not lp.layers:
        return False
    return _layer_color_exr(consolidated, 0).exists()


def _force_preview_view():
    ensure_collection(SOURCE_COLLECTION).hide_viewport = True
    ensure_collection(PREVIEW_COLLECTION).hide_viewport = False


def _load_canonical_image_and_pixels(
    exr_path: Path, image_name: str
) -> tuple[bpy.types.Image, np.ndarray, int, int]:
    """Load canonical-colour EXR for use as both shader texture and
    coverage-mask source. Returns the kept Image, an (H, W, 4) float32
    top-down RGBA array, and (W, H)."""
    img = _load_layer_image(exr_path, image_name)
    W, H = img.size
    if W == 0 or H == 0:
        raise RuntimeError(
            f"Canonical image '{exr_path.name}' loaded with no data — "
            "re-run capture / consolidate"
        )
    buf = np.empty(H * W * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    rgba_bottomup = buf.reshape(H, W, 4)
    rgba = np.flipud(rgba_bottomup).copy()
    return img, rgba, W, H


def _load_canonical_depth(exr_path: Path) -> np.ndarray:
    """Load `layer_NN_depth.exr` and return (H, W) float32 depth — the
    R-channel of the flat RGBA image written by `consolidate_views`."""
    name = f"dpxr_recon_depth_{exr_path.stem}"
    existing = bpy.data.images.get(name)
    if existing is not None:
        bpy.data.images.remove(existing)
    img = bpy.data.images.load(str(exr_path), check_existing=False)
    img.name = name
    img.colorspace_settings.name = "Non-Color"
    try:
        W, H = img.size
        if W == 0 or H == 0:
            raise RuntimeError(
                f"Depth image '{exr_path.name}' loaded with no data"
            )
        buf = np.empty(H * W * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        rgba_bottomup = buf.reshape(H, W, 4)
        depth = np.flipud(rgba_bottomup[..., 0]).copy()
        return depth
    finally:
        bpy.data.images.remove(img)


def _apply_decimate_area_vertex_group(
    obj: bpy.types.Object, boundary_verts: np.ndarray
) -> None:
    """Tag every vert on `obj` that is NOT part of the interior/border
    shared edge into `DECIMATE_AREA_GROUP` with weight 1.0.

    Consumers (users adding a Decimate modifier) set
    `modifier.vertex_group = DECIMATE_AREA_GROUP` and leave Invert off —
    group weight scales collapse influence, so weight-1 verts decimate
    normally while the unlisted boundary verts stay pinned at weight 0.
    Empty `boundary_verts` skips group creation (nothing to protect ⇒
    the modifier can just decimate the whole mesh)."""
    if boundary_verts.size == 0:
        return
    n_verts = len(obj.data.vertices)
    if n_verts == 0:
        return
    mask = np.ones(n_verts, dtype=bool)
    mask[boundary_verts] = False
    non_boundary = np.nonzero(mask)[0]
    if non_boundary.size == 0:
        return
    vg = obj.vertex_groups.new(name=DECIMATE_AREA_GROUP)
    vg.add(non_boundary.tolist(), 1.0, "REPLACE")


def _build_recon_mesh_object(
    name: str,
    mesh_arrays: LayerMesh,
    portal,
    material,
    coll,
) -> bpy.types.Object | None:
    """Hand a flat-quadtree `LayerMesh` half to bpy: build mesh, write
    UVs, attach material, parent to portal. Returns None when the half
    has no faces."""
    if mesh_arrays.faces.shape[0] == 0:
        return None
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(mesh_arrays.vertices.tolist(), [], mesh_arrays.faces.tolist())
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="UVMap")
    # Loops in mesh.polygons are stored poly-then-corner (mesh.from_pydata
    # preserves input ordering), so flattening (M, 4, 2) → (M·4·2,) lines
    # up with foreach_set's expected layout. Bulk write avoids the
    # Python→bpy round-trip cost of writing UVs one loop at a time.
    flat_uvs = np.ascontiguousarray(
        mesh_arrays.uvs.reshape(-1), dtype=np.float32
    )
    uv_layer.data.foreach_set("uv", flat_uvs)

    mesh.materials.append(material)

    obj = bpy.data.objects.new(name, mesh)
    obj.parent = portal
    obj.matrix_parent_inverse = Matrix.Identity(4)
    _apply_decimate_area_vertex_group(obj, mesh_arrays.boundary_verts)
    link_unique(obj, coll)
    return obj


def _build_displaced_recon_mesh_object(
    name: str,
    mesh_arrays: DisplacedLayerMesh,
    portal,
    material,
    coll,
) -> bpy.types.Object | None:
    """Hand a `DisplacedLayerMesh` half to bpy. Polygons may be quads
    (fast-path leaves) or n-gons (T-junction leaves with mid-edge verts
    on the perimeter). Returns None for empty halves so callers don't
    spawn empty objects."""
    if not mesh_arrays.polygons:
        return None
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(
        mesh_arrays.vertices.tolist(), [], mesh_arrays.polygons
    )
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="UVMap")
    flat_uvs = np.ascontiguousarray(
        mesh_arrays.poly_uvs.reshape(-1), dtype=np.float32
    )
    uv_layer.data.foreach_set("uv", flat_uvs)

    mesh.materials.append(material)

    obj = bpy.data.objects.new(name, mesh)
    obj.parent = portal
    obj.matrix_parent_inverse = Matrix.Identity(4)
    _apply_decimate_area_vertex_group(obj, mesh_arrays.boundary_verts)
    link_unique(obj, coll)
    return obj


class DEPTHPEELXR_OT_reconstruct_flat(Operator):
    bl_idname = "depthpeelxr.reconstruct_flat"
    bl_label = "Build Quadtree Planes"
    bl_description = "Per-layer planes with reduced overdraw"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        rig = getattr(scene, "dpx_portal_rig", None)
        lp = getattr(scene, "dpx_layer_placement", None)
        return (
            rig is not None
            and rig.portal_object is not None
            and lp is not None
            and len(lp.layers) > 0
            and consolidated_present(context)
        )

    def execute(self, context):
        scene = context.scene
        rig = scene.dpx_portal_rig
        lp = scene.dpx_layer_placement
        recon_props = scene.dpx_reconstruction
        portal = rig.portal_object
        min_quad_size = int(recon_props.min_quad_size)
        max_quad_size = int(recon_props.max_quad_size)

        consolidated = _consolidated_dir(context)

        _remove_preview_objects()

        coll = ensure_collection(PREVIEW_COLLECTION)
        ho = rig.headplane_offset
        half_w_p = rig.portal_width * 0.5
        half_h_p = rig.portal_height * 0.5

        built = 0
        total_quads = 0
        for li, layer in enumerate(lp.layers):
            exr_path = _layer_color_exr(consolidated, li)
            if not exr_path.exists():
                continue
            base_name = f"{_RECON_PREFIX}{li:02d}"
            try:
                image, rgba, W, H = _load_canonical_image_and_pixels(
                    exr_path, f"{base_name}_img"
                )
            except RuntimeError as exc:
                self.report({"ERROR"}, f"Layer {li}: {exc}")
                return {"CANCELLED"}

            coverage_mask = rgba[..., 3] > _COVERAGE_THRESHOLD
            leaves = build_quadtree(
                coverage_mask,
                min_quad_size=min_quad_size,
                tile_size=max_quad_size,
            )
            if not leaves:
                continue  # nothing covered in this layer

            d = layer.near
            half_w_plane, half_h_plane = super_frustum_half_extents(
                d, half_w_p, half_h_p, ho
            )
            mesh_arrays = build_layer_mesh(
                leaves=leaves,
                image_width=W, image_height=H,
                half_w_plane=half_w_plane,
                half_h_plane=half_h_plane,
                headplane_offset=ho,
                layer_near=d,
            )
            interior_half, border_half = split_layer_mesh(mesh_arrays)
            opaque_mat = _build_opaque_material(
                f"{base_name}{_OPAQUE_SUFFIX}_mat", image,
                portal, ho, half_w_p, half_h_p,
            )
            alpha_mat = _build_preview_material(
                f"{base_name}{_ALPHA_SUFFIX}_mat", image
            )
            _build_recon_mesh_object(
                base_name + _OPAQUE_SUFFIX,
                interior_half, portal, opaque_mat, coll,
            )
            _build_recon_mesh_object(
                base_name + _ALPHA_SUFFIX,
                border_half, portal, alpha_mat, coll,
            )
            # Cache flat halves in DisplacedLayerMesh shape so the cull
            # operator can treat flat + displaced uniformly. Cull on flat
            # is conservative (closer-surface sits at layer.near rather
            # than actual scene depth) but still drops fully-occluded
            # layers — useful for the cheaper recon tier.
            set_layer_cache(
                base_name,
                layer_mesh_to_displaced(interior_half),
                layer_mesh_to_displaced(border_half),
            )
            built += 1
            total_quads += len(leaves)

        if built == 0:
            self.report({"ERROR"}, "No consolidated layers found")
            return {"CANCELLED"}

        _force_preview_view()
        self.report(
            {"INFO"},
            f"Built {built} reconstruction layer(s), {total_quads} quad(s) total",
        )
        return {"FINISHED"}


class DEPTHPEELXR_OT_reconstruct_displaced(Operator):
    bl_idname = "depthpeelxr.reconstruct_displaced"
    bl_label = "Build Displaced Quadtree"
    bl_description = "Per-layer quadtree meshes displaced to scene depth"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        rig = getattr(scene, "dpx_portal_rig", None)
        lp = getattr(scene, "dpx_layer_placement", None)
        return (
            rig is not None
            and rig.portal_object is not None
            and lp is not None
            and len(lp.layers) > 0
            and consolidated_present(context)
        )

    def invoke(self, context, event):
        scene = context.scene
        rig = scene.dpx_portal_rig
        lp = scene.dpx_layer_placement
        recon_props = scene.dpx_reconstruction
        consolidated = _consolidated_dir(context)

        # Filter layers up-front so the progress count is honest. Skip
        # any layer missing its consolidated EXR pair.
        plan: list[int] = []
        for li, _layer in enumerate(lp.layers):
            color_path = _layer_color_exr(consolidated, li)
            depth_path = _layer_depth_exr(consolidated, li)
            if color_path.exists() and depth_path.exists():
                plan.append(li)
        if not plan:
            self.report({"ERROR"}, "No consolidated layers found")
            return {"CANCELLED"}

        # Profile log path: <project>/temp/displaced_profile.log. Cleared
        # at the start of each run so we capture only this build's timings.
        profile_log: Path | None = None
        temp = paths_mod.temp_dir(context)
        if temp is not None:
            temp.mkdir(parents=True, exist_ok=True)
            profile_log = temp / "displaced_profile.log"
            try:
                profile_log.write_text("")
            except OSError:
                profile_log = None

        self._plan = plan
        self._cursor = 0
        self._total = len(plan)
        self._completed = 0
        self._built = 0
        self._total_polys = 0
        self._should_cancel = False
        self._profile_log = profile_log
        self._consolidated = consolidated
        self._portal = rig.portal_object
        self._coll = ensure_collection(PREVIEW_COLLECTION)
        self._ho = rig.headplane_offset
        self._half_w_p = rig.portal_width * 0.5
        self._half_h_p = rig.portal_height * 0.5
        self._min_quad_size = int(recon_props.min_quad_size)
        self._max_quad_size = int(recon_props.max_quad_size)
        self._depth_tol = float(recon_props.depth_gradient_tol)
        self._lp = lp
        self._run_t0 = perf_counter()

        self._log(
            f"# Build Displaced Quadtree profile — "
            f"min={self._min_quad_size}, max={self._max_quad_size}, "
            f"tol={self._depth_tol:.3f}"
        )

        _remove_preview_objects()

        wm = context.window_manager
        wm.progress_begin(0, self._total)
        self._timer = wm.event_timer_add(_RECON_TIMER_INTERVAL, window=context.window)
        wm.modal_handler_add(self)
        self._update_header(context)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self._should_cancel = True
            self._update_header(context, "cancelling…")
            return {"RUNNING_MODAL"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        if self._should_cancel:
            return self._abort(context, "Cancelled by user")

        if self._cursor >= self._total:
            return self._finish(context)

        try:
            self._build_one_layer(context)
        except Exception as exc:
            li = self._plan[self._cursor]
            return self._abort(context, f"Build failed at layer {li}: {exc}")

        self._cursor += 1
        self._completed += 1
        context.window_manager.progress_update(self._completed)
        self._update_header(context)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self._teardown_modal(context)
        return None

    def _build_one_layer(self, context):
        li = self._plan[self._cursor]
        color_path = _layer_color_exr(self._consolidated, li)
        depth_path = _layer_depth_exr(self._consolidated, li)
        base_name = f"{_RECON_PREFIX}{li:02d}"

        t0 = perf_counter()
        image, rgba, W, H = _load_canonical_image_and_pixels(
            color_path, f"{base_name}_img"
        )
        depth = _load_canonical_depth(depth_path)
        t_load = perf_counter() - t0
        if depth.shape != (H, W):
            raise RuntimeError(
                f"depth/colour size mismatch ({depth.shape} vs {(H, W)})"
            )

        t0 = perf_counter()
        coverage_mask = rgba[..., 3] > _COVERAGE_THRESHOLD
        leaves = build_quadtree(
            coverage_mask,
            min_quad_size=self._min_quad_size,
            tile_size=self._max_quad_size,
            depth=depth,
            depth_tol=self._depth_tol,
        )
        t_quadtree = perf_counter() - t0
        if not leaves:
            return

        t0 = perf_counter()
        mesh_arrays = build_displaced_layer_mesh(
            leaves=leaves,
            canonical_depth=depth,
            canonical_coverage=coverage_mask,
            half_w_p=self._half_w_p,
            half_h_p=self._half_h_p,
            headplane_offset=self._ho,
        )
        interior_half, border_half = split_displaced_layer_mesh(mesh_arrays)
        t_meshcore = perf_counter() - t0

        t0 = perf_counter()
        opaque_mat = _build_opaque_material(
            f"{base_name}{_OPAQUE_SUFFIX}_mat", image,
            self._portal, self._ho, self._half_w_p, self._half_h_p,
        )
        alpha_mat = _build_preview_material(
            f"{base_name}{_ALPHA_SUFFIX}_mat", image
        )
        _build_displaced_recon_mesh_object(
            base_name + _OPAQUE_SUFFIX,
            interior_half, self._portal, opaque_mat, self._coll,
        )
        _build_displaced_recon_mesh_object(
            base_name + _ALPHA_SUFFIX,
            border_half, self._portal, alpha_mat, self._coll,
        )
        # Cache both halves so the cull operator can rasterise the
        # opaque half for closer-surface and rebuild either half
        # post-filter; Undo Cull restores from the cache.
        set_layer_cache(base_name, interior_half, border_half)
        t_bpy = perf_counter() - t0

        n_polys = len(mesh_arrays.polygons)
        self._log(
            f"layer {li:02d}  canonical={W}x{H}  "
            f"leaves={len(leaves)}  verts={int(mesh_arrays.vertices.shape[0])}  "
            f"polys={n_polys} (opaque={len(interior_half.polygons)}, "
            f"alpha={len(border_half.polygons)})  | "
            f"load={t_load:.3f}s  quadtree={t_quadtree:.3f}s  "
            f"core_mesh={t_meshcore:.3f}s  bpy_handoff={t_bpy:.3f}s"
        )

        self._built += 1
        self._total_polys += n_polys

    def _finish(self, context):
        self._teardown_modal(context)
        if self._built == 0:
            self.report({"WARNING"}, "Nothing built — every layer's quadtree was empty")
            return {"CANCELLED"}
        _force_preview_view()
        wall = perf_counter() - self._run_t0
        self._log(f"# total wall time: {wall:.3f}s for {self._built} layer(s)")
        info = (
            f"Built {self._built} displaced layer(s), {self._total_polys} "
            f"polygon(s) total in {wall:.2f}s"
        )
        if self._profile_log is not None:
            info = f"{info} — log: {self._profile_log}"
        self.report({"INFO"}, info)
        return {"FINISHED"}

    def _abort(self, context, reason: str):
        self._teardown_modal(context)
        self.report(
            {"WARNING"},
            f"{reason} ({self._completed}/{self._total} layers built)",
        )
        return {"CANCELLED"}

    def _teardown_modal(self, context):
        wm = context.window_manager
        wm.progress_end()
        if getattr(self, "_timer", None) is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.header_text_set(None)

    def _update_header(self, context, suffix: str = ""):
        text = f"Depth Peel XR — building displaced layer {self._completed}/{self._total}"
        if suffix:
            text = f"{text} ({suffix})"
        text = f"{text} — ESC to cancel"
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.header_text_set(text)

    def _log(self, msg: str) -> None:
        if self._profile_log is None:
            return
        with self._profile_log.open("a") as fh:
            fh.write(msg + "\n")


CLASSES = (
    DEPTHPEELXR_OT_reconstruct_flat,
    DEPTHPEELXR_OT_reconstruct_displaced,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
