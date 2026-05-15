"""Inspection operators: cycle through rig cameras, toggle Source/Preview."""

import bpy
from bpy.types import Object, Operator

from .portal_rig import CAM_PREFIX, ensure_collection


SOURCE_COLLECTION = "DepthPeelXR_Source"
PREVIEW_COLLECTION = "DepthPeelXR_Preview"


def get_view_cameras_sorted() -> list[Object]:
    return sorted(
        (
            obj for obj in bpy.data.objects
            if obj.type == "CAMERA" and obj.name.startswith(CAM_PREFIX)
        ),
        key=lambda o: o.name,
    )


def look_through_camera(context, cam_obj: Object) -> None:
    if cam_obj is None or cam_obj.type != "CAMERA":
        return
    context.scene.camera = cam_obj
    screen = context.screen
    if screen is None:
        return
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        for region in area.regions:
            if region.type != "WINDOW":
                continue
            r3d = area.spaces.active.region_3d
            if r3d.view_perspective != "CAMERA":
                with context.temp_override(area=area, region=region):
                    bpy.ops.view3d.view_camera()
            break


def _cycle_view_camera(context, direction: int) -> Object | None:
    cams = get_view_cameras_sorted()
    if not cams:
        return None
    current = context.scene.camera
    try:
        idx = cams.index(current)
    except ValueError:
        idx = -1 if direction > 0 else 0
    new_cam = cams[(idx + direction) % len(cams)]
    look_through_camera(context, new_cam)
    return new_cam


class DEPTHPEELXR_OT_view_camera_next(Operator):
    bl_idname = "depthpeelxr.view_camera_next"
    bl_label = "Next View"
    bl_description = "Look through the next rig camera"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(get_view_cameras_sorted())

    def execute(self, context):
        return {"FINISHED"} if _cycle_view_camera(context, +1) else {"CANCELLED"}


class DEPTHPEELXR_OT_view_camera_prev(Operator):
    bl_idname = "depthpeelxr.view_camera_prev"
    bl_label = "Previous View"
    bl_description = "Look through the previous rig camera"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(get_view_cameras_sorted())

    def execute(self, context):
        return {"FINISHED"} if _cycle_view_camera(context, -1) else {"CANCELLED"}


class DEPTHPEELXR_OT_exit_camera_view(Operator):
    bl_idname = "depthpeelxr.exit_camera_view"
    bl_label = "Exit Camera View"
    bl_description = "Restore the previous user-perspective view"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        screen = context.screen
        if screen is None:
            return {"CANCELLED"}
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            for region in area.regions:
                if region.type != "WINDOW":
                    continue
                r3d = area.spaces.active.region_3d
                if r3d.view_perspective == "CAMERA":
                    with context.temp_override(area=area, region=region):
                        bpy.ops.view3d.view_camera()
                break
        return {"FINISHED"}


class DEPTHPEELXR_OT_toggle_preview(Operator):
    bl_idname = "depthpeelxr.toggle_preview"
    bl_label = "Toggle Source / Preview"
    bl_description = "Swap viewport visibility between Source and Preview collections"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        source = ensure_collection(SOURCE_COLLECTION)
        preview = ensure_collection(PREVIEW_COLLECTION)
        if source.hide_viewport:
            source.hide_viewport = False
            preview.hide_viewport = True
        else:
            source.hide_viewport = True
            preview.hide_viewport = False
        return {"FINISHED"}


CLASSES = (
    DEPTHPEELXR_OT_view_camera_next,
    DEPTHPEELXR_OT_view_camera_prev,
    DEPTHPEELXR_OT_exit_camera_view,
    DEPTHPEELXR_OT_toggle_preview,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
