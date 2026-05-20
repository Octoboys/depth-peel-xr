"""Multi-view capture: per-layer NxN renders writing flat OpenEXR
pairs.

Output layout:
    <project>/captures/layer_NN/view_MM_color.exr   (RGBA float32, scene-linear)
    <project>/captures/layer_NN/view_MM_depth.exr   (depth pass, RGBA float32 with R = z)

Two renders per (layer, view):
  - Color: no compositor, standard render path writes Combined directly
    to scene.render.filepath as flat OPEN_EXR.
  - Depth: RL→Viewer compositor, drain Viewer to numpy, save as flat
    OPEN_EXR via a throw-away `bpy.data.images.new`.

5.x compositor notes:
  - File Output writes multilayer-only EXRs, which Python `Image.pixels`
    can't pick individual passes from — so we never use it.
  - A non-empty compositor without a Composite-equivalent output node
    (CompositorNodeComposite was removed in 5.x; NodeGroupOutput is the
    replacement) doesn't populate `Render Result`. Hence the color pass
    runs with no compositor at all.
  - Mutating live `Viewer Node` / `Render Result` images (e.g. via
    `viewer.save()` or `result.save_render()`) corrupts subsequent
    renders — we always drain pixels into numpy first and save via a
    fresh image.

Modal: timer-driven loop, ESC cancels at the next view boundary
(`bpy.ops.render.render` is fully synchronous and uninterruptible from
Python, so cancellation lands between renders).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import bpy
import numpy as np
from bpy.types import Operator

from .. import paths as paths_mod
from .inspect import get_view_cameras_sorted
from .preview import invalidate_flat_preview


_CAPTURE_NODE_TREE_NAME = "DPX_CaptureCompositor_temp"
_CAPTURE_TIMER_INTERVAL = 0.05


def _captures_dir(context) -> Path | None:
    root = paths_mod.project_root(context)
    return root / "captures" if root else None


def _layer_dir(captures: Path, layer_idx: int) -> Path:
    return captures / f"layer_{layer_idx:02d}"


def _color_path(layer_dir: Path, view_idx: int) -> Path:
    return layer_dir / f"view_{view_idx:02d}_color.exr"


def _depth_path(layer_dir: Path, view_idx: int) -> Path:
    return layer_dir / f"view_{view_idx:02d}_depth.exr"


def _build_depth_compositor(scene):
    """RL.Depth → Viewer. Same pattern as the layer-placement depth
    probe — Viewer reliably populates after `bpy.ops.render.render`.
    No Composite/Output: the depth pass is consumed via Viewer drain,
    not the standard render output.

    NOTE: caller must enable `view_layer.use_pass_z = True` first —
    `rl.outputs["Depth"]` only exists when the pass is active.
    """
    saved_tree = scene.compositing_node_group

    tree = bpy.data.node_groups.new(_CAPTURE_NODE_TREE_NAME, "CompositorNodeTree")
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
    if tree is not None and tree.name == _CAPTURE_NODE_TREE_NAME:
        bpy.data.node_groups.remove(tree)


_SAVE_TMP_IMAGE_NAME = "_dpx_capture_save_tmp"


def _save_rgba_as_flat_exr(target: Path, rgba: np.ndarray) -> None:
    """Write an (H, W, 4) float32 RGBA array to disk as a flat OPEN_EXR.
    Uses a fresh, throw-away `bpy.data.images` entry so existing images
    (like 'Viewer Node' and 'Render Result') aren't mutated as a
    side-effect — that breaks subsequent renders in 5.x."""
    H, W, _ = rgba.shape
    existing = bpy.data.images.get(_SAVE_TMP_IMAGE_NAME)
    if existing is not None:
        bpy.data.images.remove(existing)
    img = bpy.data.images.new(
        _SAVE_TMP_IMAGE_NAME, width=W, height=H, alpha=True, float_buffer=True
    )
    img.colorspace_settings.name = "Non-Color"
    try:
        img.pixels.foreach_set(rgba.astype(np.float32, copy=False).ravel())
        img.filepath_raw = str(target)
        img.file_format = "OPEN_EXR"
        img.save()
    finally:
        bpy.data.images.remove(img)


def _save_viewer_as_depth_exr(target: Path) -> None:
    """Drain the Depth pass from Viewer Node into numpy, write as flat
    OPEN_EXR via a throw-away image. Never calls `viewer.save()` —
    that mutates the Viewer image's state (sets filepath_raw, marks
    saved-to-disk) and subsequent renders silently don't refresh it."""
    viewer = bpy.data.images.get("Viewer Node")
    if viewer is None:
        raise RuntimeError("'Viewer Node' image missing — compositor did not run")
    W, H = viewer.size
    if W == 0 or H == 0 or viewer.channels == 0:
        raise RuntimeError("'Viewer Node' image has no data after render")
    buf = np.empty(H * W * viewer.channels, dtype=np.float32)
    viewer.pixels.foreach_get(buf)
    arr = buf.reshape(H, W, viewer.channels)
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    # Depth pass is single-channel; Blender replicates into RGBA so any
    # channel works — pick R for clarity downstream (consolidate reads R).
    rgba[..., 0] = arr[..., 0]
    rgba[..., 3] = 1.0
    _save_rgba_as_flat_exr(target, rgba)


def _engine_samples(scene, engine: str) -> tuple[str, int] | None:
    """Returns (attr_path, current_value) for the engine's sample knob,
    or None if unknown."""
    if engine == "CYCLES" and hasattr(scene, "cycles"):
        return ("cycles.samples", scene.cycles.samples)
    if engine == "BLENDER_EEVEE" and hasattr(scene, "eevee"):
        return ("eevee.taa_render_samples", scene.eevee.taa_render_samples)
    return None


def _captures_have_layout_orphans(captures: Path, n_layers: int, n_views: int) -> bool:
    """Return True if any file under `captures` would be orphaned by a
    fresh run with the current `(n_layers, n_views)` layout.

    Layout-matching content (same dirs, same filenames) is fine — the
    new run overwrites those cleanly. Orphans (e.g., a `view_08.exr`
    left over after grid_n was reduced from 3 to 2) would mislead Phase
    4 consumers that walk the captures dir.
    """
    if not captures.exists():
        return False
    expected = {
        f"layer_{l:02d}/view_{v:02d}_color.exr"
        for l in range(n_layers)
        for v in range(n_views)
    } | {
        f"layer_{l:02d}/view_{v:02d}_depth.exr"
        for l in range(n_layers)
        for v in range(n_views)
    }
    for p in captures.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(captures).as_posix()
        # Allow the consolidated/ subtree — it's owned by a downstream op.
        if rel.startswith("consolidated/"):
            continue
        if rel not in expected:
            return True
    return False


class DEPTHPEELXR_OT_capture(Operator):
    bl_idname = "depthpeelxr.capture"
    bl_label = "Run Capture"
    bl_description = "Render every rig camera × every placed layer to a (color, depth) EXR pair (ESC to cancel)"
    bl_options = {"REGISTER"}

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
            and paths_mod.project_root(context) is not None
        )

    def invoke(self, context, event):
        scene = context.scene
        view_layer = context.view_layer
        capture = scene.dpx_capture
        lp = scene.dpx_layer_placement

        cams = get_view_cameras_sorted()
        if not cams:
            self.report({"ERROR"}, "No rig cameras found")
            return {"CANCELLED"}

        captures = _captures_dir(context)
        if captures is None:
            self.report({"ERROR"}, "Project root unresolved (save the .blend first)")
            return {"CANCELLED"}

        invalidate_flat_preview(context)

        if _captures_have_layout_orphans(captures, len(lp.layers), len(cams)):
            try:
                shutil.rmtree(captures)
            except OSError as exc:
                self.report({"ERROR"}, f"Failed to wipe stale captures: {exc}")
                return {"CANCELLED"}
            self.report({"INFO"}, "Wiped stale captures from a previous run with a different layout")

        self._plan = [
            (li, vi, float(layer.near), float(layer.far), cam)
            for li, layer in enumerate(lp.layers)
            for vi, cam in enumerate(cams)
        ]
        self._cursor = 0
        self._total = len(self._plan)
        self._completed = 0
        self._captures_dir = captures
        self._should_cancel = False

        self._saved = {
            "engine": scene.render.engine,
            "res_x": scene.render.resolution_x,
            "res_y": scene.render.resolution_y,
            "res_pct": scene.render.resolution_percentage,
            "use_pass_z": view_layer.use_pass_z,
            "film_transparent": scene.render.film_transparent,
            "camera": scene.camera,
            "filepath": scene.render.filepath,
            "file_format": scene.render.image_settings.file_format,
            "color_depth": scene.render.image_settings.color_depth,
            "color_mode": scene.render.image_settings.color_mode,
        }
        self._saved_samples = _engine_samples(scene, capture.engine)
        self._saved_tree = scene.compositing_node_group
        self._saved_cam_clips = {
            cam.name: (cam.data.clip_start, cam.data.clip_end) for cam in cams
        }

        try:
            scene.render.engine = capture.engine
            view_layer.use_pass_z = True
            scene.render.film_transparent = True
            scene.render.resolution_percentage = 100
            scene.render.image_settings.file_format = "OPEN_EXR"
            scene.render.image_settings.color_depth = "32"
            scene.render.image_settings.color_mode = "RGBA"
        except TypeError as exc:
            self._restore_state(context)
            self.report({"ERROR"}, f"Engine setup failed: {exc}")
            return {"CANCELLED"}

        wm = context.window_manager
        wm.progress_begin(0, self._total)
        self._timer = wm.event_timer_add(_CAPTURE_TIMER_INTERVAL, window=context.window)
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
            self._render_one_view(context)
        except Exception as exc:
            li, vi, *_ = self._plan[self._cursor]
            return self._abort(context, f"Render failed at layer {li} view {vi}: {exc}")

        self._cursor += 1
        self._completed += 1
        context.window_manager.progress_update(self._completed)
        self._update_header(context)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        # Blender invokes this on file load / scene change / undo through invoke.
        self._restore_state(context)
        return None

    def _render_one_view(self, context):
        """Two renders for one (layer, view): color (no compositor,
        standard render path) and depth (RL→Viewer, drain Viewer)."""
        scene = context.scene
        li, vi, near, far, cam = self._plan[self._cursor]

        layer_dir = _layer_dir(self._captures_dir, li)
        layer_dir.mkdir(parents=True, exist_ok=True)
        cam.data.clip_start = near
        cam.data.clip_end = far
        scene.camera = cam

        color_target = _color_path(layer_dir, vi)
        depth_target = _depth_path(layer_dir, vi)

        scene.compositing_node_group = None
        scene.render.filepath = str(color_target.with_suffix(""))
        result = bpy.ops.render.render(write_still=True)
        if "FINISHED" not in result:
            raise RuntimeError(f"color render returned {result} (check system console)")

        depth_tree = _build_depth_compositor(scene)
        try:
            result = bpy.ops.render.render(write_still=False)
            if "FINISHED" not in result:
                raise RuntimeError(f"depth render returned {result} (check system console)")
            _save_viewer_as_depth_exr(depth_target)
        finally:
            _restore_compositor(scene, depth_tree)

    def _finish(self, context):
        self._restore_state(context)
        self._teardown_modal(context)
        self.report(
            {"INFO"},
            f"Captured {self._completed} renders to {self._captures_dir}",
        )
        return {"FINISHED"}

    def _abort(self, context, reason: str):
        self._restore_state(context)
        self._teardown_modal(context)
        self.report({"WARNING"}, f"{reason} ({self._completed}/{self._total} done)")
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
        text = f"Depth Peel XR — capturing {self._completed}/{self._total}"
        if suffix:
            text = f"{text} ({suffix})"
        text = f"{text} — ESC to cancel"
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.header_text_set(text)

    def _restore_state(self, context):
        if not getattr(self, "_saved", None):
            return
        _restore_state(
            context.scene,
            context.view_layer,
            self._saved,
            self._saved_samples,
            self._saved_tree,
            self._saved_cam_clips,
        )
        self._saved = None  # idempotent on repeat calls


def _restore_state(scene, view_layer, saved, saved_samples, saved_tree, saved_cam_clips):
    # Restore compositor — handles the case where a render died mid-flight
    # without cleaning up its temp tree.
    if scene.compositing_node_group is not None and (
        scene.compositing_node_group.name == _CAPTURE_NODE_TREE_NAME
    ):
        _restore_compositor(scene, saved_tree)
    else:
        scene.compositing_node_group = saved_tree

    scene.render.engine = saved["engine"]
    scene.render.resolution_x = saved["res_x"]
    scene.render.resolution_y = saved["res_y"]
    scene.render.resolution_percentage = saved["res_pct"]
    view_layer.use_pass_z = saved["use_pass_z"]
    scene.render.film_transparent = saved["film_transparent"]
    scene.render.filepath = saved["filepath"]
    scene.render.image_settings.file_format = saved["file_format"]
    scene.render.image_settings.color_depth = saved["color_depth"]
    scene.render.image_settings.color_mode = saved["color_mode"]
    scene.camera = saved["camera"]

    if saved_samples is not None:
        attr_path, value = saved_samples
        obj = scene
        parts = attr_path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if obj is not None:
            try:
                setattr(obj, parts[-1], value)
            except (AttributeError, TypeError):
                pass

    for cam_name, (cs, ce) in saved_cam_clips.items():
        cam = bpy.data.objects.get(cam_name)
        if cam is not None and cam.type == "CAMERA":
            cam.data.clip_start = cs
            cam.data.clip_end = ce


class DEPTHPEELXR_OT_cleanup_captures(Operator):
    bl_idname = "depthpeelxr.cleanup_captures"
    bl_label = "Clean Up Captures"
    bl_description = "Delete the captures subfolder"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        captures = _captures_dir(context)
        if captures is None:
            self.report({"ERROR"}, "Project root unresolved")
            return {"CANCELLED"}
        invalidate_flat_preview(context)
        if not captures.exists():
            self.report({"INFO"}, "No captures directory to clean")
            return {"FINISHED"}
        try:
            shutil.rmtree(captures)
        except OSError as exc:
            self.report({"ERROR"}, f"Cleanup failed: {exc}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Removed {captures}")
        return {"FINISHED"}


CLASSES = (
    DEPTHPEELXR_OT_capture,
    DEPTHPEELXR_OT_cleanup_captures,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
