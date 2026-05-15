"""N-panel UI: `Depth Peel XR` tab in the 3D viewport sidebar."""

import bpy
from bpy.types import Panel

from .. import paths
from ..operators.inspect import get_view_cameras_sorted
from ..operators.portal_rig import CAM_PREFIX


class DEPTHPEELXR_PT_main(Panel):
    bl_label = "Depth Peel XR"
    bl_idname = "DEPTHPEELXR_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Depth Peel XR"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.dpx_portal_rig
        path_props = scene.dpx_paths

        rig_box = layout.box()
        rig_box.label(text="Portal Rig", icon="OUTLINER_OB_CAMERA")
        if props.portal_object is None:
            rig_box.operator("depthpeelxr.add_portal_rig", icon="ADD")
        else:
            col = rig_box.column(align=True)
            col.prop(props, "portal_width")
            col.prop(props, "portal_height")
            col.prop(props, "headplane_offset")
            col.prop(props, "grid_n")
            col.separator()
            col.prop(props, "viewing_resolution")

            rig_box.separator()
            rig_box.operator(
                "depthpeelxr.cleanup_portal_rig",
                text="Clean Up Portal Rig",
                icon="TRASH",
            )

        if props.portal_object is not None:
            inspect_box = layout.box()
            inspect_box.label(text="Inspect", icon="HIDE_OFF")

            cams = get_view_cameras_sorted()
            current = scene.camera
            current_in_rig = (
                current is not None
                and current.type == "CAMERA"
                and current.name.startswith(CAM_PREFIX)
            )
            if current_in_rig and cams:
                idx = next((i for i, c in enumerate(cams) if c == current), -1)
                label_text = (
                    f"{current.name}  ({idx + 1}/{len(cams)})"
                    if idx >= 0
                    else f"{current.name}"
                )
            else:
                label_text = f"— / {len(cams)}"

            row = inspect_box.row(align=True)
            outer = row.split(factor=0.2, align=True)
            outer.operator(
                "depthpeelxr.view_camera_prev", text="", icon="TRIA_LEFT"
            )
            inner = outer.split(factor=0.75, align=True)
            mid = inner.row(align=True)
            mid.alignment = "CENTER"
            mid.label(text=label_text)
            inner.operator(
                "depthpeelxr.view_camera_next", text="", icon="TRIA_RIGHT"
            )
            inspect_box.operator(
                "depthpeelxr.exit_camera_view", icon="VIEW_PERSPECTIVE"
            )
            inspect_box.operator(
                "depthpeelxr.toggle_preview", icon="RESTRICT_VIEW_OFF"
            )

        project_box = layout.box()
        project_box.label(text="Project", icon="FILE_FOLDER")
        unsaved = not paths.is_blend_saved() and not path_props.project_root
        if unsaved:
            project_box.label(text="Save the .blend to enable project paths", icon="ERROR")
        project_box.prop(path_props, "project_root", text="")


CLASSES = (DEPTHPEELXR_PT_main,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
