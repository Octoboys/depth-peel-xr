import bpy
from bpy.props import (
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Object, PropertyGroup


def _on_rig_param_change(self, context):
    from .operators.portal_rig import resync_rig_from_props

    resync_rig_from_props(context)


class PortalRigProps(PropertyGroup):
    portal_object: PointerProperty(
        name="Portal",
        description="Portal mesh plane defining the viewing aperture",
        type=Object,
    )
    portal_width: FloatProperty(
        name="Portal Width",
        description="Width of the Portal rectangle (and head-plane) in metres",
        default=4.0,
        min=0.01,
        soft_max=20.0,
        precision=2,
        step=10,
        update=_on_rig_param_change,
    )
    portal_height: FloatProperty(
        name="Portal Height",
        description="Height of the Portal rectangle (and head-plane) in metres",
        default=2.45,
        min=0.01,
        soft_max=20.0,
        precision=2,
        step=10,
        update=_on_rig_param_change,
    )
    headplane_offset: FloatProperty(
        name="Head-plane Offset",
        description="Distance from Portal to head-plane along Portal +Y normal, in metres",
        default=1.5,
        min=0.0,
        soft_max=10.0,
        precision=2,
        step=10,
        update=_on_rig_param_change,
    )
    grid_n: IntProperty(
        name="Grid N",
        description="N×N camera grid on the head-plane (corners-inclusive)",
        default=3,
        min=1,
        soft_max=9,
        update=_on_rig_param_change,
    )
    viewing_resolution: IntProperty(
        name="Viewing Resolution",
        description=(
            "Render resolution along the portal's longer axis; the other "
            "dimension is derived from the portal's aspect ratio"
        ),
        default=1024,
        min=64,
        soft_max=4096,
        subtype="PIXEL",
        update=_on_rig_param_change,
    )


class PathsProps(PropertyGroup):
    project_root: StringProperty(
        name="Project Root",
        description="Folder under which generated artefacts (captures, USD, etc.) are written",
        default="",
        subtype="DIR_PATH",
    )


CLASSES = (PortalRigProps, PathsProps)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.dpx_portal_rig = PointerProperty(type=PortalRigProps)
    bpy.types.Scene.dpx_paths = PointerProperty(type=PathsProps)


def unregister():
    del bpy.types.Scene.dpx_paths
    del bpy.types.Scene.dpx_portal_rig
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
