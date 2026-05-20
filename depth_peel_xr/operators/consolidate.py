"""Multi-view consolidation: post-capture pass that fuses each layer's
NxN views into a canonical super-frustum-NDC RGBA + depth + source-mask.

Phase 3 slice 3. Reads the per-view flat EXRs written by
`DEPTHPEELXR_OT_capture`, reprojects every view (centre + siblings) into a
virtual super-frustum perspective camera that exactly frames the union of
all rig cameras' frustums, and writes one set of canonical outputs per
layer under `<project>/captures/consolidated/`.

The canonical's pixel grid grows per layer with the super/centre frustum-
half-extent ratio at `layer.near` — deeper layers get bigger canonicals,
capped at 2× centre as layer.near → ∞. Centre-view pixels seed the central
region; the surrounding annulus is filled by whichever sibling sees it
(closest-to-centre wins for sibling overlap).

Consumed downstream by Phase 4's reconstruction operators, which size
their per-layer planes to `super_frustum_half_extents(layer.near)` so the
canonical pixel grid maps 1:1 onto the reconstructed plane.
"""

from __future__ import annotations

from pathlib import Path

import bpy
import numpy as np
from bpy.types import Operator
from mathutils import Vector

from .. import paths as paths_mod
from ..core.geometry import (
    super_frustum_canonical_shape,
    super_frustum_projection,
)
from ..core.multi_view import consolidate_layer
from .inspect import get_view_cameras_sorted
from .preview import invalidate_flat_preview


_LOAD_IMAGE_PREFIX = "dpxr_consolidate_tmp_"
_SAVE_IMAGE_PREFIX = "dpxr_consolidate_out_"


def _captures_dir(context) -> Path | None:
    return paths_mod.captures_dir(context)


def _consolidated_dir(context) -> Path | None:
    captures = _captures_dir(context)
    return captures / "consolidated" if captures else None


def _color_path(captures: Path, layer_idx: int, view_idx: int) -> Path:
    return captures / f"layer_{layer_idx:02d}" / f"view_{view_idx:02d}_color.exr"


def _depth_path(captures: Path, layer_idx: int, view_idx: int) -> Path:
    return captures / f"layer_{layer_idx:02d}" / f"view_{view_idx:02d}_depth.exr"


def _load_flat_exr_rgba(exr_path: Path) -> tuple[np.ndarray, int, int]:
    """Load a flat OpenEXR as a top-down (H, W, 4) float32 array.
    Blender stores image pixels bottom-up; we flip on the way out.
    """
    name = f"{_LOAD_IMAGE_PREFIX}{exr_path.stem}"
    existing = bpy.data.images.get(name)
    if existing is not None:
        bpy.data.images.remove(existing)
    img = bpy.data.images.load(str(exr_path), check_existing=False)
    img.name = name
    img.colorspace_settings.name = "Non-Color"
    try:
        W, H = img.size
        buf = np.empty(H * W * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        arr = buf.reshape(H, W, 4)
        return np.flipud(arr).copy(), W, H
    finally:
        bpy.data.images.remove(img)


def _load_view_passes(
    captures: Path, layer_idx: int, view_idx: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load the (color, depth) pair for a given (layer, view). Returns
    (rgba (H, W, 4), depth (H, W), W, H). Depth is taken from the R
    channel — capture writes the Z pass as an RGBA image with R holding
    the depth."""
    color_p = _color_path(captures, layer_idx, view_idx)
    depth_p = _depth_path(captures, layer_idx, view_idx)
    rgba, W, H = _load_flat_exr_rgba(color_p)
    depth_rgba, _, _ = _load_flat_exr_rgba(depth_p)
    depth = depth_rgba[..., 0]
    return rgba, depth, W, H


def _build_projection_matrix(
    cam_data: bpy.types.Camera,
    render_x: int,
    render_y: int,
    pixel_aspect_x: float = 1.0,
    pixel_aspect_y: float = 1.0,
) -> np.ndarray:
    """Build a 4x4 OpenGL-style perspective projection matrix from a
    Blender perspective camera. Handles off-axis frustum via shift_x/y.

    Replaces `Camera.calc_matrix_camera`, which was removed in Blender
    5.x. core.multi_view is invariant to the camera's clip range, so it
    doesn't matter that the post-capture defaults differ from whatever
    per-layer ranges were used during capture.
    """
    near = cam_data.clip_start
    far = cam_data.clip_end
    lens = cam_data.lens
    sensor_w = cam_data.sensor_width
    sensor_h = cam_data.sensor_height
    shift_x = cam_data.shift_x
    shift_y = cam_data.shift_y

    aspect_image = (render_x * pixel_aspect_x) / (render_y * pixel_aspect_y)

    # half-extents at unit camera-space distance, in scene units.
    if cam_data.sensor_fit == "VERTICAL":
        half_h_norm = sensor_h / (2.0 * lens)
        half_w_norm = half_h_norm * aspect_image
    else:  # HORIZONTAL or AUTO (rig uses HORIZONTAL)
        half_w_norm = sensor_w / (2.0 * lens)
        half_h_norm = half_w_norm / aspect_image

    # Blender shift is in fractions of sensor_width regardless of fit.
    shift_x_norm = shift_x * sensor_w / lens
    shift_y_norm = shift_y * sensor_w / lens

    left = (-half_w_norm + shift_x_norm) * near
    right = (half_w_norm + shift_x_norm) * near
    bottom = (-half_h_norm + shift_y_norm) * near
    top = (half_h_norm + shift_y_norm) * near

    M = np.zeros((4, 4), dtype=np.float64)
    M[0, 0] = 2.0 * near / (right - left)
    M[1, 1] = 2.0 * near / (top - bottom)
    M[0, 2] = (right + left) / (right - left)
    M[1, 2] = (top + bottom) / (top - bottom)
    M[2, 2] = -(far + near) / (far - near)
    M[2, 3] = -2.0 * far * near / (far - near)
    M[3, 2] = -1.0
    return M


def _camera_matrices(
    cam_obj: bpy.types.Object,
    render_x: int,
    render_y: int,
    pixel_aspect_x: float,
    pixel_aspect_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (world_from_cam, proj) as (4,4) numpy arrays."""
    world_from_cam = np.array(cam_obj.matrix_world, dtype=np.float64)
    proj = _build_projection_matrix(
        cam_obj.data, render_x, render_y, pixel_aspect_x, pixel_aspect_y
    )
    return world_from_cam, proj


def _sibling_order_closest_first(cams: list, centre_idx: int) -> list[int]:
    """Return sibling indices into `cams`, sorted by Euclidean distance
    of the camera from the centre camera on the head-plane (cam local
    XZ). Centre's own index excluded."""
    centre_pos = Vector(cams[centre_idx].matrix_local.translation)
    distances = []
    for i, cam in enumerate(cams):
        if i == centre_idx:
            continue
        d = (Vector(cam.matrix_local.translation) - centre_pos).length
        distances.append((d, i))
    distances.sort()
    return [i for _d, i in distances]


def _centre_view_index(n_views: int) -> int:
    return n_views // 2


def _write_canonical_outputs(
    consolidated_dir: Path,
    layer_idx: int,
    rgba: np.ndarray,
    depth: np.ndarray,
    source: np.ndarray,
) -> None:
    """Write the three per-layer outputs as flat OpenEXR files. depth
    and source are single-channel-meaningful but stored as RGBA (R holds
    the value, others zeroed) so Blender's image-save path can write
    them as float EXR without compositor gymnastics.
    """
    consolidated_dir.mkdir(parents=True, exist_ok=True)
    H, W, _ = rgba.shape

    _save_flat_exr(consolidated_dir / f"layer_{layer_idx:02d}_color.exr", rgba)

    depth_rgba = np.zeros((H, W, 4), dtype=np.float32)
    depth_rgba[..., 0] = depth
    depth_rgba[..., 3] = 1.0
    _save_flat_exr(consolidated_dir / f"layer_{layer_idx:02d}_depth.exr", depth_rgba)

    src_rgba = np.zeros((H, W, 4), dtype=np.float32)
    src_rgba[..., 0] = source.astype(np.float32)
    src_rgba[..., 3] = 1.0
    _save_flat_exr(consolidated_dir / f"layer_{layer_idx:02d}_source.exr", src_rgba)


def _save_flat_exr(path: Path, rgba: np.ndarray) -> None:
    """Save an (H, W, 4) float32 array as a flat OpenEXR. Pixels are
    fed to Blender bottom-up to match its storage order."""
    H, W, _ = rgba.shape
    name = f"{_SAVE_IMAGE_PREFIX}{path.stem}"
    existing = bpy.data.images.get(name)
    if existing is not None:
        bpy.data.images.remove(existing)
    img = bpy.data.images.new(name, width=W, height=H, alpha=True, float_buffer=True)
    img.colorspace_settings.name = "Non-Color"
    try:
        flipped = np.flipud(rgba).astype(np.float32, copy=False).ravel()
        img.pixels.foreach_set(flipped)
        img.filepath_raw = str(path)
        img.file_format = "OPEN_EXR"
        img.save()
    finally:
        bpy.data.images.remove(img)


_CONSOLIDATE_TIMER_INTERVAL = 0.05  # seconds — same cadence as capture


class DEPTHPEELXR_OT_consolidate_views(Operator):
    bl_idname = "depthpeelxr.consolidate_views"
    bl_label = "Consolidate Views"
    bl_description = "Fuse the per-view captures of each layer into one canonical RGBA + depth + source-mask"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        rig = getattr(scene, "dpx_portal_rig", None)
        lp = getattr(scene, "dpx_layer_placement", None)
        captures = _captures_dir(context)
        if rig is None or rig.portal_object is None or lp is None or not lp.layers:
            return False
        if captures is None or not captures.exists():
            return False
        return True

    def invoke(self, context, event):
        scene = context.scene
        rig = scene.dpx_portal_rig
        lp = scene.dpx_layer_placement
        cams = get_view_cameras_sorted()
        if not cams:
            self.report({"ERROR"}, "No rig cameras found")
            return {"CANCELLED"}

        captures = _captures_dir(context)
        consolidated = _consolidated_dir(context)
        if captures is None or consolidated is None:
            self.report({"ERROR"}, "Project root unresolved")
            return {"CANCELLED"}

        # Re-consolidation writes new canonical EXRs; any on-screen
        # reconstruction was built from the previous canonical and is
        # now stale.
        invalidate_flat_preview(context)

        centre_idx = _centre_view_index(len(cams))
        sibling_order = _sibling_order_closest_first(cams, centre_idx)

        render_x = scene.render.resolution_x
        render_y = scene.render.resolution_y
        pixel_aspect_x = scene.render.pixel_aspect_x
        pixel_aspect_y = scene.render.pixel_aspect_y

        ho = rig.headplane_offset
        half_w_p = rig.portal_width * 0.5
        half_h_p = rig.portal_height * 0.5

        # Canonical's virtual cam is fixed by the rig; only the buffer
        # shape varies per layer.
        canonical_from_centre, canonical_proj = super_frustum_projection(
            half_w_p, half_h_p, ho
        )
        centre_cam = cams[centre_idx]
        centre_world_from_cam, centre_proj = _camera_matrices(
            centre_cam, render_x, render_y, pixel_aspect_x, pixel_aspect_y
        )
        canonical_cam_from_world = (
            canonical_from_centre @ np.linalg.inv(centre_world_from_cam)
        )

        # Sibling rig cameras don't change between layers either.
        sibling_matrices: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for sib_idx in sibling_order:
            sibling_matrices[sib_idx] = _camera_matrices(
                cams[sib_idx], render_x, render_y, pixel_aspect_x, pixel_aspect_y
            )

        # Stash everything modal() needs.
        self._captures = captures
        self._consolidated = consolidated
        self._layers = list(lp.layers)
        self._centre_idx = centre_idx
        self._sibling_order = sibling_order
        self._centre_world_from_cam = centre_world_from_cam
        self._centre_proj = centre_proj
        self._canonical_cam_from_world = canonical_cam_from_world
        self._canonical_proj = canonical_proj
        self._sibling_matrices = sibling_matrices
        self._half_w_p = half_w_p
        self._half_h_p = half_h_p
        self._ho = ho
        self._cursor = 0
        self._total = len(self._layers)
        self._completed = 0
        self._written = 0
        self._should_cancel = False

        wm = context.window_manager
        wm.progress_begin(0, max(1, self._total))
        self._timer = wm.event_timer_add(
            _CONSOLIDATE_TIMER_INTERVAL, window=context.window
        )
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
            self._consolidate_one_layer(context)
        except Exception as exc:
            return self._abort(
                context,
                f"Consolidation failed at layer {self._cursor}: {exc}",
            )

        self._cursor += 1
        self._completed += 1
        context.window_manager.progress_update(self._completed)
        self._update_header(context)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self._teardown_modal(context)
        return None

    def _consolidate_one_layer(self, context):
        li = self._cursor
        layer = self._layers[li]
        centre_color = _color_path(self._captures, li, self._centre_idx)
        if not centre_color.exists():
            self.report({"WARNING"}, f"Layer {li}: centre EXR missing, skipped")
            return

        centre_rgba, centre_depth, centre_W, centre_H = _load_view_passes(
            self._captures, li, self._centre_idx
        )
        views_sorted = [{
            "rgba": centre_rgba,
            "depth": centre_depth,
            "world_from_cam": self._centre_world_from_cam,
            "proj": self._centre_proj,
        }]
        for sib_idx in self._sibling_order:
            sib_color = _color_path(self._captures, li, sib_idx)
            if not sib_color.exists():
                continue
            try:
                sib_rgba, sib_depth, _, _ = _load_view_passes(
                    self._captures, li, sib_idx
                )
            except Exception as exc:
                self.report(
                    {"WARNING"},
                    f"Layer {li} view {sib_idx}: read failed: {exc}",
                )
                continue
            sib_world_from_cam, sib_proj = self._sibling_matrices[sib_idx]
            views_sorted.append({
                "rgba": sib_rgba,
                "depth": sib_depth,
                "world_from_cam": sib_world_from_cam,
                "proj": sib_proj,
            })

        canonical_W, canonical_H = super_frustum_canonical_shape(
            layer.near, centre_W, centre_H,
            self._half_w_p, self._half_h_p, self._ho,
        )

        rgba, depth, _coverage, source = consolidate_layer(
            views_sorted=views_sorted,
            canonical_shape=(canonical_H, canonical_W),
            canonical_cam_from_world=self._canonical_cam_from_world,
            canonical_proj=self._canonical_proj,
        )

        _write_canonical_outputs(self._consolidated, li, rgba, depth, source)
        self._written += 1

    def _finish(self, context):
        self._teardown_modal(context)
        if self._written == 0:
            self.report({"ERROR"}, "No layers consolidated (check captures)")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            f"Consolidated {self._written} layer(s) → {self._consolidated}",
        )
        return {"FINISHED"}

    def _abort(self, context, reason: str):
        self._teardown_modal(context)
        self.report(
            {"WARNING"},
            f"{reason} ({self._completed}/{self._total} done)",
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
        text = (
            f"Depth Peel XR — consolidating layer "
            f"{min(self._cursor + 1, self._total)}/{self._total}"
        )
        if suffix:
            text = f"{text} ({suffix})"
        text = f"{text} — ESC to cancel"
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.header_text_set(text)


CLASSES = (DEPTHPEELXR_OT_consolidate_views,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
