"""Occlusion culling for the layered reconstruction.

Cull Occluded Quads walks the cached layer meshes near→far. At each
layer L > 0 it culls both the opaque and alpha halves against the
accumulated per-cam closer-surface depth, then merges layer L's *opaque*
half (rasterised per cam) into the closer-surface for downstream layers.

Critically, only the OPAQUE half contributes to closer-surface — border
(alpha-cut) halves never claim depth ownership of the pixels they only
sparsely cover (e.g., fences, foliage). The opaque half IS the geometry
that owns pixels at runtime, so its silhouette is the correct occluder
by construction.

Per-vert visibility test only — `cull_layer`'s docstring discusses the
conservative "all 4 corners hidden, interior visible" failure mode that
a future per-pixel rasterizer upgrade would catch.

Cache holds (opaque, alpha) `DisplacedLayerMesh` halves per layer,
keyed by base_name (e.g. `dpxr_recon_layer_03`); spawned objects are
`<base>_opaque` and `<base>_alpha`. Undo Cull rebuilds both halves from
the cached arrays.
"""

from __future__ import annotations

from time import perf_counter

import bpy
import numpy as np
from bpy.types import Operator
from mathutils import Matrix

from .. import paths as paths_mod
from ..core.occlusion import cull_layer
from ..core.raster import fan_triangulate_polygons, rasterise_depth
from ..core.reconstruction import DisplacedLayerMesh, filter_displaced_mesh
from .consolidate import _build_projection_matrix
from .inspect import PREVIEW_COLLECTION, get_view_cameras_sorted
from .portal_rig import ensure_collection
from .preview import _build_opaque_material, _build_preview_material
from .recon_cache import LayerCachePair, all_cached_names, get_layer_cache
from .reconstruct import (
    _ALPHA_SUFFIX,
    _OPAQUE_SUFFIX,
    _RECON_PREFIX,
    _build_displaced_recon_mesh_object,
)


# Closer-surface depth buffers (and the per-vert pixel-coord lookups
# that probe them) run at this fraction of render resolution. Lower =
# faster rasterise (16× fewer pixels at 0.25), more conservative cull
# at silhouette edges (slight under-cull). Camera projection matrices
# are unaffected — they encode aspect, which is preserved when both
# axes scale equally.
_CLOSER_SURFACE_SCALE = 0.25


def _portal_world_matrix(rig) -> np.ndarray:
    return np.array(rig.portal_object.matrix_world, dtype=np.float64)


def _world_verts(local_verts: np.ndarray, portal_world: np.ndarray) -> np.ndarray:
    V = local_verts.shape[0]
    if V == 0:
        return np.empty((0, 3), dtype=np.float64)
    homog = np.concatenate(
        [local_verts, np.ones((V, 1), dtype=local_verts.dtype)], axis=-1
    )
    return (portal_world @ homog.T).T[:, :3]


def _gather_cam_matrices(
    cams: list, render_x: int, render_y: int, pa_x: float, pa_y: float
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int]]]:
    world_from_cams: list[np.ndarray] = []
    projs: list[np.ndarray] = []
    resolutions: list[tuple[int, int]] = []
    for cam in cams:
        world_from_cams.append(np.array(cam.matrix_world, dtype=np.float64))
        projs.append(_build_projection_matrix(cam.data, render_x, render_y, pa_x, pa_y))
        resolutions.append((render_x, render_y))
    return world_from_cams, projs, resolutions


def _accumulate_opaque_closer_surface(
    closer_per_cam: list[np.ndarray],
    opaque_half: DisplacedLayerMesh,
    portal_world: np.ndarray,
    cam_world_from_cams: list[np.ndarray],
    cam_projs: list[np.ndarray],
    cam_resolutions: list[tuple[int, int]],
) -> None:
    """Rasterise `opaque_half` from each cam and per-pixel-MIN merge into
    `closer_per_cam`. Border halves never call this — only opaque-classed
    leaves contribute to the closer-surface."""
    if not opaque_half.polygons:
        return
    verts_world = _world_verts(opaque_half.vertices, portal_world)
    tris = fan_triangulate_polygons(opaque_half.polygons)
    for c, world_from_cam in enumerate(cam_world_from_cams):
        W, H = cam_resolutions[c]
        layer_depth = rasterise_depth(
            verts_world,
            tris,
            world_from_cam,
            cam_projs[c],
            W, H,
        )
        np.minimum(closer_per_cam[c], layer_depth, out=closer_per_cam[c])


def _rebuild_object_from_mesh_arrays(
    name: str,
    mesh_arrays: DisplacedLayerMesh,
    rig,
    coll,
    *,
    opaque: bool,
) -> None:
    """Replace the existing `<name>` object with a freshly built mesh
    from `mesh_arrays`. Reuses the existing texture image so we don't
    repay the EXR-load cost. `opaque` selects the material flavour
    (emission-only vs emission+alpha-clip)."""
    portal = rig.portal_object
    base_name = name[: -len(_OPAQUE_SUFFIX)] if opaque else name[: -len(_ALPHA_SUFFIX)]
    image_name = f"{base_name}_img"
    image = bpy.data.images.get(image_name)

    obj = bpy.data.objects.get(name)
    if obj is not None:
        old_mesh = obj.data if isinstance(obj.data, bpy.types.Mesh) else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if old_mesh is not None and old_mesh.users == 0:
            bpy.data.meshes.remove(old_mesh)
    mat_name = f"{name}_mat"
    old_mat = bpy.data.materials.get(mat_name)
    if old_mat is not None and old_mat.users == 0:
        bpy.data.materials.remove(old_mat)

    if image is None or not mesh_arrays.polygons:
        # Image got cleaned up, or the half is empty — nothing to rebuild.
        return
    if opaque:
        material = _build_opaque_material(
            mat_name, image, portal,
            rig.headplane_offset,
            rig.portal_width * 0.5,
            rig.portal_height * 0.5,
        )
    else:
        material = _build_preview_material(mat_name, image)
    _build_displaced_recon_mesh_object(
        name, mesh_arrays, portal, material, coll
    )


class DEPTHPEELXR_OT_cull_occluded(Operator):
    bl_idname = "depthpeelxr.cull_occluded"
    bl_label = "Cull Occluded Quads"
    bl_description = "Drop polygons hidden behind closer layers from every head-plane camera"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        rig = getattr(scene, "dpx_portal_rig", None)
        lp = getattr(scene, "dpx_layer_placement", None)
        if rig is None or rig.portal_object is None or lp is None or not lp.layers:
            return False
        # Need at least one cached mesh — i.e., the user has run a recon
        # build this session that populated the cache.
        return bool(all_cached_names())

    def invoke(self, context, event):
        # Single-tick modal — invoke sets the header, returns RUNNING_MODAL,
        # Blender's event loop runs once (header repaints), TIMER fires,
        # modal runs the entire cull synchronously in that tick. Without
        # the yield, the header would never become visible because the
        # cull would block Blender's main thread before any redraw.
        cams = get_view_cameras_sorted()
        if not cams:
            self.report({"ERROR"}, "No rig cameras found")
            return {"CANCELLED"}
        self._cams = cams
        self._set_header(context, "Depth Peel XR — culling occluded quads…")
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.001, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        try:
            kept, dropped, n_layers_culled, wall = self._run_cull(context)
        except Exception as exc:
            return self._abort(context, f"Cull failed: {exc}")
        self._teardown(context)
        if dropped == 0:
            self.report(
                {"INFO"},
                f"Nothing to cull — {kept} polygons kept ({wall:.1f}s)",
            )
        else:
            pct = 100.0 * dropped / max(1, kept + dropped)
            self.report(
                {"INFO"},
                f"Culled {dropped} polygon(s) ({pct:.1f}%), kept {kept} "
                f"across {n_layers_culled} layer(s) in {wall:.1f}s",
            )
        return {"FINISHED"}

    def cancel(self, context):
        self._teardown(context)
        return None

    def _run_cull(self, context) -> tuple[int, int, int, float]:
        scene = context.scene
        rig = scene.dpx_portal_rig
        lp = scene.dpx_layer_placement
        cams = self._cams
        n_cams = len(cams)

        coll = ensure_collection(PREVIEW_COLLECTION)
        portal_world = _portal_world_matrix(rig)

        render_x = scene.render.resolution_x
        render_y = scene.render.resolution_y
        pa_x = scene.render.pixel_aspect_x
        pa_y = scene.render.pixel_aspect_y
        cam_world_from_cams, cam_projs, _ = _gather_cam_matrices(
            cams, render_x, render_y, pa_x, pa_y
        )
        scaled_x = max(1, int(round(render_x * _CLOSER_SURFACE_SCALE)))
        scaled_y = max(1, int(round(render_y * _CLOSER_SURFACE_SCALE)))
        cam_resolutions = [(scaled_x, scaled_y) for _ in cams]

        # Closer-surface depth buffer per cam — incrementally built as
        # we walk layers near→far. Starts at +inf (nothing closer).
        closer_per_cam: list[np.ndarray] = [
            np.full((scaled_y, scaled_x), np.inf, dtype=np.float32)
            for _ in range(n_cams)
        ]

        t_run = perf_counter()
        n_layers_culled = 0
        total_kept = 0
        total_dropped = 0

        for li, _layer in enumerate(lp.layers):
            base_name = f"{_RECON_PREFIX}{li:02d}"
            pair: LayerCachePair | None = get_layer_cache(base_name)
            if pair is None:
                # No cache for this layer — recon skipped it (no
                # consolidated EXR) or pre-cached state was cleared.
                continue

            if li > 0:
                for half_name, half in (
                    (base_name + _OPAQUE_SUFFIX, pair.opaque),
                    (base_name + _ALPHA_SUFFIX, pair.alpha),
                ):
                    if not half.polygons:
                        continue
                    verts_world = _world_verts(half.vertices, portal_world)
                    keep = cull_layer(
                        polygons=half.polygons,
                        vertices_world=verts_world,
                        cam_world_from_cams=cam_world_from_cams,
                        cam_projs=cam_projs,
                        cam_resolutions=cam_resolutions,
                        closer_surface_depths_per_cam=closer_per_cam,
                    )
                    n_keep = int(keep.sum())
                    n_drop = int((~keep).sum())
                    total_kept += n_keep
                    total_dropped += n_drop
                    if n_drop > 0:
                        filtered = filter_displaced_mesh(half, keep)
                        _rebuild_object_from_mesh_arrays(
                            half_name,
                            filtered,
                            rig,
                            coll,
                            opaque=half_name.endswith(_OPAQUE_SUFFIX),
                        )
                n_layers_culled += 1
            else:
                total_kept += len(pair.opaque.polygons) + len(pair.alpha.polygons)

            # Opaque half rasterised into closer_per_cam — alpha half
            # never contributes (border quads can't claim closer-surface
            # ownership of the pixels they sparsely cover).
            _accumulate_opaque_closer_surface(
                closer_per_cam,
                pair.opaque,
                portal_world,
                cam_world_from_cams,
                cam_projs,
                cam_resolutions,
            )

        wall = perf_counter() - t_run
        return total_kept, total_dropped, n_layers_culled, wall

    def _abort(self, context, reason: str):
        self._teardown(context)
        self.report({"WARNING"}, reason)
        return {"CANCELLED"}

    def _teardown(self, context):
        wm = context.window_manager
        if getattr(self, "_timer", None) is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.header_text_set(None)

    def _set_header(self, context, text: str) -> None:
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.header_text_set(text)


class DEPTHPEELXR_OT_undo_cull(Operator):
    bl_idname = "depthpeelxr.undo_cull"
    bl_label = "Undo Cull"
    bl_description = "Restore the pre-cull displaced meshes from the cache"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(all_cached_names())

    def execute(self, context):
        scene = context.scene
        rig = getattr(scene, "dpx_portal_rig", None)
        if rig is None or rig.portal_object is None:
            self.report({"ERROR"}, "No portal rig")
            return {"CANCELLED"}

        coll = ensure_collection(PREVIEW_COLLECTION)
        n_restored = 0
        for base_name in all_cached_names():
            pair = get_layer_cache(base_name)
            if pair is None:
                continue
            _rebuild_object_from_mesh_arrays(
                base_name + _OPAQUE_SUFFIX, pair.opaque, rig, coll, opaque=True
            )
            _rebuild_object_from_mesh_arrays(
                base_name + _ALPHA_SUFFIX, pair.alpha, rig, coll, opaque=False
            )
            n_restored += 1
        self.report({"INFO"}, f"Restored {n_restored} layer(s) from cache")
        return {"FINISHED"}


CLASSES = (
    DEPTHPEELXR_OT_cull_occluded,
    DEPTHPEELXR_OT_undo_cull,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
