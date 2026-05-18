"""Phase 4 slice 1: flat-quadtree reconstruction.

Consumes the consolidated centre-NDC outputs (`consolidated/layer_NN_color.exr`):
derives coverage from the canonical alpha, runs the adaptive quadtree,
emits a per-layer mesh whose vertices are coplanar at `layer.near`. No
displacement (that lands in Phase 4 slice 2).

Cheapest *shippable* reconstruction — silhouette-cut interior/border
quads, alpha-clip emission material. Mutually exclusive with Build
Simple Planes; both live in `DepthPeelXR_Preview` and are wiped by the
shared `Clean Up Geo` operator.
"""

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np
from bpy.types import Operator
from mathutils import Matrix

from .. import paths as paths_mod
from ..core.geometry import centre_frustum_half_extents
from ..core.quadtree import build_quadtree
from ..core.reconstruction import build_layer_mesh
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
    verts = [tuple(v) for v in mesh_arrays.vertices]
    faces = [tuple(int(i) for i in f) for f in mesh_arrays.faces]
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for corner in range(poly.loop_total):
            loop_idx = poly.loop_start + corner
            uv_layer.data[loop_idx].uv = tuple(
                mesh_arrays.uvs[poly.index, corner]
            )

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


class DEPTHPEELXR_OT_reconstruct_flat(Operator):
    bl_idname = "depthpeelxr.reconstruct_flat"
    bl_label = "Build Flat Quadtree"
    bl_description = (
        "Per layer, adaptive quadtree drives silhouette-cut interior/border "
        "quads at the layer's near depth. No displacement — vertices stay "
        "coplanar. Cheapest shippable reconstruction tier."
    )
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
            leaves = build_quadtree(coverage_mask, min_quad_size=min_quad_size)
            if not leaves:
                continue  # nothing covered in this layer

            d = layer.near
            half_w_cf, half_h_cf = centre_frustum_half_extents(
                d, half_w_p, half_h_p, ho
            )
            mesh_arrays = build_layer_mesh(
                leaves=leaves,
                image_width=W, image_height=H,
                half_w_centre_frustum=half_w_cf,
                half_h_centre_frustum=half_h_cf,
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


CLASSES = (DEPTHPEELXR_OT_reconstruct_flat,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
