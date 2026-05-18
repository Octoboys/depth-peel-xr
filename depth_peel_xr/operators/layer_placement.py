"""Layer-placement operators: analyse depth scope and emit layer boundaries.

Phase 2 — both slices wired:
  - FIXED  : analytic spacing (uniform-Z / 1-over-Z / log). Pure math, live.
  - HYBRID : centre-camera Z-pass probe + histogram-driven CDF placement.
             Live but debounced (~150ms) so slider drags trigger one render
             on release rather than per-tick.
"""

from __future__ import annotations

import bpy
import numpy as np
from bpy.types import Operator
from mathutils import Matrix, Vector

from ..core.layer_placement import (
    Distribution,
    FixedDistribution,
    HistogramHybrid,
    HistogramImportance,
)
from .portal_rig import camera_local_rotation_4x4
from .preview import invalidate_flat_preview


_PROBE_CAM_NAME = "DPX_Probe_temp"
_PROBE_NODE_TREE_NAME = "DPX_ProbeCompositor_temp"
_VIEWER_IMAGE_NAME = "Viewer Node"
_PROBE_RESOLUTION_LONG = 256
_HYBRID_DEBOUNCE_SECS = 0.15
# Minimum probe samples in [near, far] for HYBRID to attempt placement.
# Below this we refuse rather than emit one mega-layer from a few stray
# pixels.
_HYBRID_MIN_VALID_SAMPLES = 100
# Minimum depth-spread fraction of the placement scope. Below this the
# probed depth is essentially flat (single plane / single density mode);
# layer placement adds no value, so refuse.
_HYBRID_MIN_DEPTH_SPREAD_FRACTION = 0.01

_HYBRID_PENDING_GEN = 0


def tag_view3d_redraw() -> None:
    wm = bpy.data.window_managers
    if not wm:
        return
    for window in wm[0].windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def _populate_layers(props, layers: list[tuple[float, float]]) -> None:
    props.layers.clear()
    for near, far in layers:
        entry = props.layers.add()
        entry.near = float(near)
        entry.far = float(far)


def _create_probe_camera(scene, rig, clip_start: float, clip_end: float) -> bpy.types.Object:
    portal = rig.portal_object
    cam_data = bpy.data.cameras.new(_PROBE_CAM_NAME)
    cam_obj = bpy.data.objects.new(_PROBE_CAM_NAME, cam_data)
    scene.collection.objects.link(cam_obj)

    rot = camera_local_rotation_4x4()
    cam_obj.matrix_world = (
        portal.matrix_world
        @ Matrix.Translation(Vector((0.0, rig.headplane_offset, 0.0)))
        @ rot
    )

    cam_data.type = "PERSP"
    cam_data.lens_unit = "MILLIMETERS"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.lens = 36.0 * rig.headplane_offset / rig.portal_width
    cam_data.shift_x = 0.0
    cam_data.shift_y = 0.0
    cam_data.clip_start = clip_start
    cam_data.clip_end = clip_end
    return cam_obj


def _delete_probe_camera(cam_obj: bpy.types.Object) -> None:
    if cam_obj is None:
        return
    data = cam_obj.data
    bpy.data.objects.remove(cam_obj, do_unlink=True)
    if isinstance(data, bpy.types.Camera) and data.users == 0:
        bpy.data.cameras.remove(data)


def _resolution_xy(rig) -> tuple[int, int]:
    long_res = _PROBE_RESOLUTION_LONG
    if rig.portal_width >= rig.portal_height:
        return long_res, max(1, round(long_res * rig.portal_height / rig.portal_width))
    return max(1, round(long_res * rig.portal_width / rig.portal_height)), long_res


def _build_probe_compositor(scene):
    """Replace the scene's compositor node group with a Render Layers →
    Viewer graph. The Viewer node populates `bpy.data.images['Viewer Node']`
    in memory during render — no file I/O, simplest API surface.

    Blender 5.x: scenes hold a `compositing_node_group` (a NodeGroup
    asset) instead of the older `scene.use_nodes` / `scene.node_tree`.
    """
    saved_tree = scene.compositing_node_group

    tree = bpy.data.node_groups.new(_PROBE_NODE_TREE_NAME, "CompositorNodeTree")
    scene.compositing_node_group = tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.scene = scene
    viewer = tree.nodes.new("CompositorNodeViewer")

    tree.links.new(rl.outputs["Depth"], viewer.inputs["Image"])
    return saved_tree


def _restore_compositor(scene, saved_tree) -> None:
    tree = scene.compositing_node_group
    scene.compositing_node_group = saved_tree
    if tree is not None and tree.name == _PROBE_NODE_TREE_NAME:
        bpy.data.node_groups.remove(tree)


def _read_viewer_depth(width: int, height: int) -> np.ndarray:
    """Read camera-space depth from the Viewer Node image, populated by the
    compositor during the last render. Returns a (height, width) float32
    array, top-down.
    """
    img = bpy.data.images.get(_VIEWER_IMAGE_NAME)
    if img is None:
        raise RuntimeError("'Viewer Node' image missing — compositor render did not run")
    if img.channels == 0 or img.size[0] == 0:
        raise RuntimeError("'Viewer Node' image is empty after render")
    n = img.size[0] * img.size[1] * img.channels
    buf = np.empty(n, dtype=np.float32)
    img.pixels.foreach_get(buf)
    depth = buf.reshape(img.size[1], img.size[0], img.channels)[:, :, 0]
    return np.flipud(depth).copy()


def _render_depth_probe(
    context,
    *,
    clip_start: float | None = None,
    clip_end: float | None = None,
) -> np.ndarray:
    """Render a low-res Z pass through the centre of the head-plane and
    return camera-space depth as a 2D numpy array.

    By default the probe's clip range matches the layer-placement scope
    (`[ho + min_depth, ho + max_depth]`), so depths outside the scope are
    clipped — keeping the histogram free of background sentinels. Pass
    `clip_start` / `clip_end` to override this (used by the auto-fit
    operator, which wants to see *all* scene depth before deciding the
    scope).

    All scene state (engine, resolution, passes, compositor, camera) is
    saved before the render and restored on every exit path.
    """
    scene = context.scene
    rig = scene.dpx_portal_rig
    view_layer = context.view_layer

    width, height = _resolution_xy(rig)
    if clip_start is None:
        clip_start = rig.headplane_offset + rig.min_depth
    if clip_end is None:
        clip_end = rig.headplane_offset + rig.max_depth

    saved = {
        "engine": scene.render.engine,
        "res_x": scene.render.resolution_x,
        "res_y": scene.render.resolution_y,
        "res_pct": scene.render.resolution_percentage,
        "use_pass_z": view_layer.use_pass_z,
        "camera": scene.camera,
    }
    saved_tree = probe_cam = None
    try:
        scene.render.engine = "BLENDER_EEVEE"
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.resolution_percentage = 100
        view_layer.use_pass_z = True

        probe_cam = _create_probe_camera(scene, rig, clip_start, clip_end)
        scene.camera = probe_cam

        saved_tree = _build_probe_compositor(scene)

        bpy.ops.render.render(write_still=False)

        return _read_viewer_depth(width, height)
    finally:
        if saved_tree is not None or scene.compositing_node_group is not None:
            _restore_compositor(scene, saved_tree)
        if probe_cam is not None:
            _delete_probe_camera(probe_cam)
        scene.render.engine = saved["engine"]
        scene.render.resolution_x = saved["res_x"]
        scene.render.resolution_y = saved["res_y"]
        scene.render.resolution_percentage = saved["res_pct"]
        view_layer.use_pass_z = saved["use_pass_z"]
        scene.camera = saved["camera"]


def compute_and_store_layers(context) -> tuple[bool, str]:
    """Run the selected strategy and write its boundaries to scene props.

    Returns (ok, message). Layers are cleared up front, so any failure
    leaves the overlay empty rather than showing a stale preview that
    doesn't match the current params. Caller is responsible for tagging
    the viewport redraw.
    """
    scene = context.scene
    rig = scene.dpx_portal_rig
    lp = scene.dpx_layer_placement

    lp.layers.clear()
    invalidate_flat_preview(context)

    near = float(rig.headplane_offset) + float(rig.min_depth)
    far = float(rig.headplane_offset) + float(rig.max_depth)
    if not (far > near):
        return False, (
            f"Invalid depth scope: min_depth ({rig.min_depth}) must be < "
            f"max_depth ({rig.max_depth})"
        )

    if lp.strategy == "FIXED":
        try:
            strategy = FixedDistribution(Distribution(lp.distribution))
            layers = strategy.place(near, far, int(lp.layer_count))
        except ValueError as exc:
            return False, str(exc)
        _populate_layers(lp, layers)
        return True, f"Placed {len(layers)} layers in [{near:.2f}, {far:.2f}] m"

    if lp.strategy in ("HYBRID", "IMPORTANCE"):
        try:
            depth = _render_depth_probe(context)
        except RuntimeError as exc:
            return False, str(exc)

        valid = depth[(depth > near) & (depth < far)]
        if valid.size < _HYBRID_MIN_VALID_SAMPLES:
            return False, (
                f"Probe is essentially empty: only {valid.size} valid "
                f"samples in [{near:.2f}, {far:.2f}] m "
                f"(need {_HYBRID_MIN_VALID_SAMPLES}+). Adjust depth scope "
                "or check scene content."
            )

        spread = float(valid.max() - valid.min())
        scope = far - near
        if spread < _HYBRID_MIN_DEPTH_SPREAD_FRACTION * scope:
            return False, (
                f"Probed depth is essentially flat: spread {spread:.3f} m "
                f"({100 * spread / scope:.2f}% of scope). Layer placement "
                "adds no value — use a single plane instead, or widen scope."
            )

        strategy = (
            HistogramHybrid() if lp.strategy == "HYBRID" else HistogramImportance()
        )
        try:
            layers = strategy.place(near, far, int(lp.layer_count), depth=depth)
        except ValueError as exc:
            return False, str(exc)
        _populate_layers(lp, layers)
        requested = int(lp.layer_count)
        suffix = f" (requested {requested}, empty layers collapsed)" if len(layers) < requested else ""
        return True, f"Placed {len(layers)} layers in [{near:.2f}, {far:.2f}] m{suffix}"

    return False, f"Unknown strategy {lp.strategy!r}"


def repopulate_layers_if_active(context) -> None:
    """Auto-trigger recompute on param change. Gated on `auto_recompute`,
    not on layer count — that way a transient compute failure (which
    clears layers) doesn't break live updates on subsequent param
    changes."""
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    lp = getattr(scene, "dpx_layer_placement", None)
    if lp is None or not lp.auto_recompute:
        return

    if lp.strategy == "FIXED":
        compute_and_store_layers(context)
        tag_view3d_redraw()
        return

    if lp.strategy in ("HYBRID", "IMPORTANCE"):
        _schedule_hybrid_recompute()


def _schedule_hybrid_recompute() -> None:
    """Debounce HYBRID recomputes so slider drags trigger one render on
    release. Uses a generation counter — every call invalidates pending
    fires."""
    global _HYBRID_PENDING_GEN
    _HYBRID_PENDING_GEN += 1
    gen = _HYBRID_PENDING_GEN

    def _fire():
        if gen != _HYBRID_PENDING_GEN:
            return None  # superseded by a later schedule
        ctx = bpy.context
        if getattr(ctx, "scene", None) is None:
            return None
        lp = getattr(ctx.scene, "dpx_layer_placement", None)
        if lp is None or not lp.auto_recompute or lp.strategy not in ("HYBRID", "IMPORTANCE"):
            return None
        compute_and_store_layers(ctx)
        tag_view3d_redraw()
        return None

    bpy.app.timers.register(_fire, first_interval=_HYBRID_DEBOUNCE_SECS)


class DEPTHPEELXR_OT_place_layers(Operator):
    bl_idname = "depthpeelxr.place_layers"
    bl_label = "Analyse & Place"
    bl_description = "Compute depth-layer boundaries for the rig's depth scope"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return (
            getattr(scene, "dpx_portal_rig", None) is not None
            and scene.dpx_portal_rig.portal_object is not None
        )

    def execute(self, context):
        context.scene.dpx_layer_placement.auto_recompute = True
        ok, msg = compute_and_store_layers(context)
        tag_view3d_redraw()
        if not ok:
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class DEPTHPEELXR_OT_cleanup_layers(Operator):
    bl_idname = "depthpeelxr.cleanup_layers"
    bl_label = "Clean Up Layers"
    bl_description = "Clear the placed layer boundaries"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        lp = context.scene.dpx_layer_placement
        lp.layers.clear()
        lp.auto_recompute = False
        invalidate_flat_preview(context)
        tag_view3d_redraw()
        return {"FINISHED"}


class DEPTHPEELXR_OT_auto_fit_depth_scope(Operator):
    bl_idname = "depthpeelxr.auto_fit_depth_scope"
    bl_label = "Auto-fit Depth Scope"
    bl_description = (
        "Render a wide-open depth probe and snap min_depth / max_depth to "
        "the actual scene-content extents past the Portal"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        return (
            getattr(scene, "dpx_portal_rig", None) is not None
            and scene.dpx_portal_rig.portal_object is not None
        )

    def execute(self, context):
        scene = context.scene
        rig = scene.dpx_portal_rig
        lp = scene.dpx_layer_placement

        def fail(msg: str) -> set:
            lp.layers.clear()
            invalidate_flat_preview(context)
            tag_view3d_redraw()
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}

        clip_start = 0.001
        clip_end = 1000.0
        try:
            depth = _render_depth_probe(
                context, clip_start=clip_start, clip_end=clip_end
            )
        except RuntimeError as exc:
            return fail(str(exc))

        # Pixels with no geometry hit read as ~clip_end (background
        # sentinel). Exclude them from the extents calculation, otherwise
        # an empty scene would set min/max to ~clip_end.
        bg_threshold = clip_end * 0.999
        real = depth[
            np.isfinite(depth)
            & (depth > rig.headplane_offset)
            & (depth < bg_threshold)
        ]
        if real.size == 0:
            return fail("No scene geometry visible past the portal")

        min_z = float(real.min())
        max_z = float(real.max())
        # Pad the bounds slightly so scene extrema don't sit on the
        # clip-plane boundary — the low-res probe (256 along long axis)
        # can miss exact corners by a fraction of a pixel, and float
        # precision at clip_end clips geometry that's mathematically
        # in range. 1% of scene depth, floor 1cm.
        pad = max(0.01, (max_z - min_z) * 0.01)
        new_min = max(0.0, min_z - rig.headplane_offset - pad)
        new_max = max(new_min + 0.01, max_z - rig.headplane_offset + pad)

        rig.min_depth = new_min
        rig.max_depth = new_max
        self.report(
            {"INFO"},
            f"Depth scope set to [{new_min:.2f}, {new_max:.2f}] m past portal",
        )
        return {"FINISHED"}


CLASSES = (
    DEPTHPEELXR_OT_place_layers,
    DEPTHPEELXR_OT_cleanup_layers,
    DEPTHPEELXR_OT_auto_fit_depth_scope,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
