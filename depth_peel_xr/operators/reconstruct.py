"""Phase 4 reconstruction operators.

Slice 1 (`DEPTHPEELXR_OT_reconstruct_flat`, "Build Quadtree Planes"):
flat-quadtree planes per layer at `layer.near`. Silhouette-driven
subdivision; no displacement.

Slice 2 (`DEPTHPEELXR_OT_reconstruct_displaced`, "Build Displaced
Planes"): the same quadtree augmented with depth-variance subdivision,
plus per-vertex displacement via inverse super-frustum projection.
Shared corner verts with bridge-triangle T-junction handling — adjacent
leaves at different subdivision levels stitch through fan triangulation
so the mesh stays watertight at depth discontinuities.

Both consume `consolidated/layer_NN_color.exr` (RGB texture + alpha
coverage) and slice-2 also reads `consolidated/layer_NN_depth.exr`
(canonical-cam-frame depth). Mutually exclusive: each operator wipes
existing reconstruction objects before building. The shared
`Clean Up Geo` operator (`_remove_preview_objects`) covers both because
they both spawn under `dpxr_recon_layer_*`.
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
    build_displaced_layer_mesh,
    build_layer_mesh,
)
from .inspect import PREVIEW_COLLECTION, SOURCE_COLLECTION
from .portal_rig import ensure_collection, link_unique
from .preview import (
    _build_preview_material,
    _load_layer_image,
    _remove_preview_objects,
)


_RECON_PREFIX = "dpxr_recon_layer_"
# Canonical-alpha threshold above which a pixel counts as covered for
# quadtree subdivision. Slice 2 consolidation packs anti-aliased silhouette
# alpha at edges, so we sit comfortably above the 0/1 boundary.
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


def _build_recon_mesh_object(
    name: str,
    mesh_arrays,
    portal,
    material,
    coll,
) -> bpy.types.Object:
    """Hand mesh arrays to bpy: build mesh, write UVs and interior/border
    vertex groups, attach material, parent to portal."""
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
    link_unique(obj, coll)

    interior_group = obj.vertex_groups.new(name="interior")
    border_group = obj.vertex_groups.new(name="border")
    # Vertices aren't shared between faces (slice-1 design), so each
    # vertex belongs to exactly one class. Assign in face order.
    interior_verts, border_verts = [], []
    for fi, cls in enumerate(mesh_arrays.face_classes):
        bucket = interior_verts if cls == "interior" else border_verts
        bucket.extend(int(i) for i in mesh_arrays.faces[fi])
    if interior_verts:
        interior_group.add(interior_verts, 1.0, "REPLACE")
    if border_verts:
        border_group.add(border_verts, 1.0, "REPLACE")

    return obj


def _build_displaced_recon_mesh_object(
    name: str,
    mesh_arrays,
    portal,
    material,
    coll,
) -> bpy.types.Object:
    """Slice-2 counterpart of `_build_recon_mesh_object` for the
    polygon-based `DisplacedLayerMesh` (one quad per fast-path leaf,
    one n-gon per slow-path / T-junction leaf). Verts are shared across
    polygons so a single vert can belong to both interior and border
    vertex groups — both groups are assigned independently."""
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
    link_unique(obj, coll)

    interior_group = obj.vertex_groups.new(name="interior")
    border_group = obj.vertex_groups.new(name="border")
    interior_verts: set[int] = set()
    border_verts: set[int] = set()
    for poly, cls in zip(mesh_arrays.polygons, mesh_arrays.poly_classes):
        bucket = interior_verts if cls == "interior" else border_verts
        for vi in poly:
            bucket.add(int(vi))
    if interior_verts:
        interior_group.add(list(interior_verts), 1.0, "REPLACE")
    if border_verts:
        border_group.add(list(border_verts), 1.0, "REPLACE")

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
            material = _build_preview_material(f"{base_name}_mat", image)
            _build_recon_mesh_object(
                base_name, mesh_arrays, portal, material, coll
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

    def execute(self, context):
        scene = context.scene
        rig = scene.dpx_portal_rig
        lp = scene.dpx_layer_placement
        recon_props = scene.dpx_reconstruction
        portal = rig.portal_object
        min_quad_size = int(recon_props.min_quad_size)
        max_quad_size = int(recon_props.max_quad_size)
        depth_tol = float(recon_props.depth_gradient_tol)

        consolidated = _consolidated_dir(context)

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

        def _log(msg: str) -> None:
            if profile_log is None:
                return
            with profile_log.open("a") as fh:
                fh.write(msg + "\n")

        run_t0 = perf_counter()
        _log(
            f"# Build Displaced Quadtree profile — "
            f"min={min_quad_size}, max={max_quad_size}, tol={depth_tol:.3f}"
        )

        _remove_preview_objects()

        coll = ensure_collection(PREVIEW_COLLECTION)
        ho = rig.headplane_offset
        half_w_p = rig.portal_width * 0.5
        half_h_p = rig.portal_height * 0.5

        built = 0
        total_polys = 0
        for li, layer in enumerate(lp.layers):
            color_path = _layer_color_exr(consolidated, li)
            depth_path = _layer_depth_exr(consolidated, li)
            if not color_path.exists() or not depth_path.exists():
                continue
            base_name = f"{_RECON_PREFIX}{li:02d}"
            t0 = perf_counter()
            try:
                image, rgba, W, H = _load_canonical_image_and_pixels(
                    color_path, f"{base_name}_img"
                )
                depth = _load_canonical_depth(depth_path)
            except RuntimeError as exc:
                self.report({"ERROR"}, f"Layer {li}: {exc}")
                return {"CANCELLED"}
            t_load = perf_counter() - t0
            if depth.shape != (H, W):
                self.report(
                    {"ERROR"},
                    f"Layer {li}: depth/colour size mismatch "
                    f"({depth.shape} vs {(H, W)})",
                )
                return {"CANCELLED"}

            t0 = perf_counter()
            coverage_mask = rgba[..., 3] > _COVERAGE_THRESHOLD
            leaves = build_quadtree(
                coverage_mask,
                min_quad_size=min_quad_size,
                tile_size=max_quad_size,
                depth=depth,
                depth_tol=depth_tol,
            )
            t_quadtree = perf_counter() - t0
            if not leaves:
                continue

            t0 = perf_counter()
            mesh_arrays = build_displaced_layer_mesh(
                leaves=leaves,
                canonical_depth=depth,
                canonical_coverage=coverage_mask,
                half_w_p=half_w_p,
                half_h_p=half_h_p,
                headplane_offset=ho,
            )
            t_meshcore = perf_counter() - t0

            t0 = perf_counter()
            material = _build_preview_material(f"{base_name}_mat", image)
            _build_displaced_recon_mesh_object(
                base_name, mesh_arrays, portal, material, coll
            )
            t_bpy = perf_counter() - t0

            n_leaves = len(leaves)
            n_verts = int(mesh_arrays.vertices.shape[0])
            n_polys = len(mesh_arrays.polygons)
            _log(
                f"layer {li:02d}  canonical={W}x{H}  "
                f"leaves={n_leaves}  verts={n_verts}  polys={n_polys}  | "
                f"load={t_load:.3f}s  quadtree={t_quadtree:.3f}s  "
                f"core_mesh={t_meshcore:.3f}s  bpy_handoff={t_bpy:.3f}s"
            )

            built += 1
            total_polys += n_polys

        if built == 0:
            self.report({"ERROR"}, "No consolidated layers found")
            return {"CANCELLED"}

        _force_preview_view()
        run_total = perf_counter() - run_t0
        _log(f"# total wall time: {run_total:.3f}s for {built} layer(s)")
        info = (
            f"Built {built} displaced layer(s), {total_polys} polygon(s) "
            f"total in {run_total:.2f}s"
        )
        if profile_log is not None:
            info = f"{info} — log: {profile_log}"
        self.report({"INFO"}, info)
        return {"FINISHED"}


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
