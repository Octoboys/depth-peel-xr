"""Multi-view capture: per-layer NxN renders writing multilayer EXRs.

Phase 3 slice 1.

Output layout:
    <project>/captures/layer_NN/view_MM.exr   (multilayer: Combined + Depth)

The operator is modal so the UI stays responsive and ESC cancels at the
next render boundary (`bpy.ops.render.render` is fully synchronous so
mid-render cancel isn't possible — see CLAUDE.md / implementation_plan).

State save/restore covers everything we touch: render engine, samples,
resolution, view-layer Z-pass enable, scene compositor node group,
scene.camera, and per-camera clip ranges. `_restore_state` is idempotent
and called from every exit path (FINISHED, CANCELLED, exception, Blender
cancel callback).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import bpy
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


def _view_filename(view_idx: int) -> str:
    return f"view_{view_idx:02d}"


def _build_capture_compositor(scene, layer_dir: Path, view_name: str):
    """Build a Render Layers → File Output graph writing multilayer EXR
    with `Combined` (RGBA) and `Depth` (FLOAT) channels.

    NOTE: caller must enable `view_layer.use_pass_z = True` before
    invoking — otherwise `rl.outputs["Depth"]` lookup fails (the socket
    only exists when the pass is active).
    """
    saved_tree = scene.compositing_node_group

    tree = bpy.data.node_groups.new(_CAPTURE_NODE_TREE_NAME, "CompositorNodeTree")
    scene.compositing_node_group = tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.scene = scene

    out = tree.nodes.new("CompositorNodeOutputFile")
    out.directory = str(layer_dir)
    out.file_name = view_name
    out.format.file_format = "OPEN_EXR_MULTILAYER"
    out.format.color_depth = "32"
    # Write scene-linear data (no view transform) — this output will be
    # consumed by Phase 4 reconstruction, not displayed directly.
    out.save_as_render = False

    out.file_output_items.clear()
    out.file_output_items.new("RGBA", "Combined")
    out.file_output_items.new("FLOAT", "Depth")

    tree.links.new(rl.outputs["Image"], out.inputs["Combined"])
    tree.links.new(rl.outputs["Depth"], out.inputs["Depth"])
    return saved_tree, out


def _restore_compositor(scene, saved_tree) -> None:
    tree = scene.compositing_node_group
    scene.compositing_node_group = saved_tree
    if tree is not None and tree.name == _CAPTURE_NODE_TREE_NAME:
        bpy.data.node_groups.remove(tree)


def _strip_frame_suffix(written_path: Path, target_path: Path) -> None:
    """File Output appends `<frame:04d>` to the filename. Rename to the
    spec'd `view_MM.exr` for predictable consumer paths."""
    if written_path.exists():
        if target_path.exists():
            target_path.unlink()
        written_path.rename(target_path)


def _layer_count(scene) -> int:
    return len(scene.dpx_layer_placement.layers)


def _view_count() -> int:
    return len(get_view_cameras_sorted())


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
        f"layer_{l:02d}/view_{v:02d}.exr"
        for l in range(n_layers)
        for v in range(n_views)
    }
    for p in captures.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(captures).as_posix()
        if rel not in expected:
            return True
    return False


class DEPTHPEELXR_OT_capture(Operator):
    bl_idname = "depthpeelxr.capture"
    bl_label = "Run Capture"
    bl_description = (
        "Render every rig camera × every placed layer to a multilayer EXR. "
        "Modal — ESC cancels at the next render boundary."
    )
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
        rig = scene.dpx_portal_rig
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

        # New captures invalidate any flat preview (images will be replaced
        # on disk; cached image data in Blender would otherwise display stale).
        invalidate_flat_preview(context)

        # Wipe captures only when the existing layout doesn't match what
        # this run will write — protects Phase 4 from picking up stale
        # files left by a previous run with different grid_n / layer_count.
        if _captures_have_layout_orphans(captures, len(lp.layers), len(cams)):
            try:
                shutil.rmtree(captures)
            except OSError as exc:
                self.report({"ERROR"}, f"Failed to wipe stale captures: {exc}")
                return {"CANCELLED"}
            self.report({"INFO"}, "Wiped stale captures from a previous run with a different layout")

        # Build iteration plan: (layer_idx, view_idx, layer_near, layer_far, cam).
        self._plan = []
        for li, layer in enumerate(lp.layers):
            for vi, cam in enumerate(cams):
                self._plan.append((li, vi, float(layer.near), float(layer.far), cam))
        self._cursor = 0
        self._total = len(self._plan)
        self._completed = 0
        self._captures_dir = captures
        self._should_cancel = False

        # Save state.
        self._saved = {
            "engine": scene.render.engine,
            "res_x": scene.render.resolution_x,
            "res_y": scene.render.resolution_y,
            "res_pct": scene.render.resolution_percentage,
            "use_pass_z": view_layer.use_pass_z,
            "film_transparent": scene.render.film_transparent,
            "camera": scene.camera,
        }
        self._saved_samples = _engine_samples(scene, capture.engine)
        self._saved_tree = scene.compositing_node_group
        self._saved_cam_clips = {
            cam.name: (cam.data.clip_start, cam.data.clip_end) for cam in cams
        }

        # Apply baseline render config.
        try:
            scene.render.engine = capture.engine
            view_layer.use_pass_z = True
            # Film transparency so Combined.Alpha encodes per-pixel coverage.
            # Without this, "miss" pixels (out-of-slab / no hit) take the
            # world background colour with alpha=1, polluting downstream
            # layer compositing and the flat-cutout preview.
            scene.render.film_transparent = True
            # Resolution comes from the rig's `viewing_resolution` already
            # baked into scene.render via _apply_portal_render_resolution
            # (called by the rig's resync) — no extra mutation needed.
            scene.render.resolution_percentage = 100
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
            self._render_one(context)
        except Exception as exc:
            return self._abort(context, f"Render failed: {exc}")

        self._cursor += 1
        self._completed += 1
        context.window_manager.progress_update(self._completed)
        self._update_header(context)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        # Blender invokes this on file load / scene change / undo through invoke.
        self._restore_state(context)
        return None

    def _render_one(self, context):
        scene = context.scene
        layer_idx, view_idx, near, far, cam = self._plan[self._cursor]

        layer_dir = _layer_dir(self._captures_dir, layer_idx)
        layer_dir.mkdir(parents=True, exist_ok=True)
        view_name = _view_filename(view_idx)

        cam.data.clip_start = near
        cam.data.clip_end = far
        scene.camera = cam

        saved_tree, _out = _build_capture_compositor(scene, layer_dir, view_name)
        try:
            bpy.ops.render.render(write_still=False)
        finally:
            _restore_compositor(scene, saved_tree)

        # File Output appends frame number; rename to the spec'd path.
        written = layer_dir / f"{view_name}{scene.frame_current:04d}.exr"
        target = layer_dir / f"{view_name}.exr"
        _strip_frame_suffix(written, target)

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
        # Clear header text.
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
        scene = context.scene
        view_layer = context.view_layer

        # Restore compositor (in case _render_one died mid-flight without
        # cleaning up its temp tree).
        if scene.compositing_node_group is not None and (
            scene.compositing_node_group.name == _CAPTURE_NODE_TREE_NAME
        ):
            _restore_compositor(scene, self._saved_tree)
        else:
            scene.compositing_node_group = self._saved_tree

        scene.render.engine = self._saved["engine"]
        scene.render.resolution_x = self._saved["res_x"]
        scene.render.resolution_y = self._saved["res_y"]
        scene.render.resolution_percentage = self._saved["res_pct"]
        view_layer.use_pass_z = self._saved["use_pass_z"]
        scene.render.film_transparent = self._saved["film_transparent"]
        scene.camera = self._saved["camera"]

        if self._saved_samples is not None:
            attr_path, value = self._saved_samples
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

        for cam_name, (cs, ce) in self._saved_cam_clips.items():
            cam = bpy.data.objects.get(cam_name)
            if cam is not None and cam.type == "CAMERA":
                cam.data.clip_start = cs
                cam.data.clip_end = ce

        self._saved = None  # mark restored — idempotent on repeat calls


class DEPTHPEELXR_OT_cleanup_captures(Operator):
    bl_idname = "depthpeelxr.cleanup_captures"
    bl_label = "Clean Up Captures"
    bl_description = "Delete the captures subfolder under the project root"
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
