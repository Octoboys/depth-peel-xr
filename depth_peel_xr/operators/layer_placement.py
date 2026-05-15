"""Layer-placement operators: analyse depth scope and emit layer boundaries.

Slice 1 (Phase 2): only the FIXED strategy is wired. The depth probe and
HYBRID strategy land in slice 2.
"""

import bpy
from bpy.types import Operator

from ..core.layer_placement import Distribution, FixedDistribution


def tag_view3d_redraw() -> None:
    """Force every VIEW_3D area to redraw — needed because the boundary-plane
    overlay only repaints when the viewport invalidates."""
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


def compute_and_store_layers(scene) -> tuple[bool, str]:
    """Run the selected strategy and write its boundaries to scene props.

    Returns (ok, message). Pure helper — no UI side effects, no redraw tag.
    """
    rig = scene.dpx_portal_rig
    lp = scene.dpx_layer_placement

    near = float(rig.headplane_offset) + float(rig.min_depth)
    far = float(rig.headplane_offset) + float(rig.max_depth)
    if not (far > near):
        return False, (
            f"Invalid depth scope: min_depth ({rig.min_depth}) must be < "
            f"max_depth ({rig.max_depth})"
        )

    if lp.strategy != "FIXED":
        return False, f"Strategy {lp.strategy!r} not implemented yet"

    try:
        strategy = FixedDistribution(Distribution(lp.distribution))
        layers = strategy.place(near, far, int(lp.layer_count))
    except ValueError as exc:
        return False, str(exc)

    _populate_layers(lp, layers)
    return True, f"Placed {len(layers)} layers in [{near:.2f}, {far:.2f}] m"


def repopulate_layers_if_active(context) -> None:
    """Re-run placement as a side effect of param changes — but only when
    FIXED is selected and the user has already explicitly placed layers
    once. Avoids surprising the user with auto-populated overlays before
    they've asked for any."""
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    lp = getattr(scene, "dpx_layer_placement", None)
    if lp is None or len(lp.layers) == 0 or lp.strategy != "FIXED":
        return
    compute_and_store_layers(scene)
    tag_view3d_redraw()


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
        ok, msg = compute_and_store_layers(context.scene)
        if not ok:
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        self.report({"INFO"}, msg)
        tag_view3d_redraw()
        return {"FINISHED"}


class DEPTHPEELXR_OT_cleanup_layers(Operator):
    bl_idname = "depthpeelxr.cleanup_layers"
    bl_label = "Clean Up Layers"
    bl_description = "Clear the placed layer boundaries"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        context.scene.dpx_layer_placement.layers.clear()
        tag_view3d_redraw()
        return {"FINISHED"}


CLASSES = (
    DEPTHPEELXR_OT_place_layers,
    DEPTHPEELXR_OT_cleanup_layers,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
