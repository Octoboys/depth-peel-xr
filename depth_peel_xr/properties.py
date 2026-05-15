import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Object, PropertyGroup


def _on_rig_param_change(self, context):
    from .operators.portal_rig import resync_rig_from_props
    from .operators.layer_placement import repopulate_layers_if_active

    resync_rig_from_props(context)
    repopulate_layers_if_active(context)


def _on_layer_param_change(self, context):
    from .operators.layer_placement import repopulate_layers_if_active

    repopulate_layers_if_active(context)


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
    max_depth: FloatProperty(
        name="Max Depth",
        description=(
            "Maximum scene depth (metres past the Portal) the rig captures. "
            "Layer placement subdivides [headplane_offset + min_depth, "
            "headplane_offset + max_depth]"
        ),
        default=20.0,
        min=0.01,
        soft_max=200.0,
        precision=2,
        step=10,
        update=_on_layer_param_change,
    )
    min_depth: FloatProperty(
        name="Min Depth",
        description=(
            "Offset of the first layer past the Portal, in metres. Push back "
            "the near boundary to skip empty space between the Portal and the "
            "first scene content"
        ),
        default=0.0,
        min=0.0,
        soft_max=50.0,
        precision=2,
        step=10,
        update=_on_layer_param_change,
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


class LayerEntry(PropertyGroup):
    near: FloatProperty(name="Near", description="Layer near depth (metres, camera-space)")
    far: FloatProperty(name="Far", description="Layer far depth (metres, camera-space)")


def _strategy_items(self, context):
    return (
        ("FIXED", "Fixed Distribution", "Analytic spacing — uniform-Z, 1/Z, or log"),
        ("HYBRID", "Histogram Hybrid", "Importance + uniform NDC sampling from a centre-camera depth probe"),
    )


_DISTRIBUTION_ITEMS = (
    ("UNIFORM_Z", "Uniform Z", "Even spacing in linear depth"),
    ("ONE_OVER_Z", "1 / Z", "Even spacing in inverse depth (perceptually uniform near-field)"),
    ("LOG", "Log Z", "Even spacing in log depth"),
)


class LayerPlacementProps(PropertyGroup):
    strategy: EnumProperty(
        name="Strategy",
        description="Layer-placement algorithm",
        items=_strategy_items,
        default=0,
        update=_on_layer_param_change,
    )
    distribution: EnumProperty(
        name="Spread",
        description="Spacing curve for fixed-distribution placement",
        items=_DISTRIBUTION_ITEMS,
        default="LOG",
        update=_on_layer_param_change,
    )
    layer_count: IntProperty(
        name="Layers",
        description="Number of depth layers to place",
        default=4,
        min=1,
        soft_max=128,
        update=_on_layer_param_change,
    )
    layers: CollectionProperty(type=LayerEntry)
    auto_recompute: BoolProperty(
        name="Auto-recompute",
        description=(
            "Internal flag — True once the user has requested placement, "
            "until they hit Clean Up Layers. Drives whether param changes "
            "auto-trigger recomputes."
        ),
        default=False,
    )


CLASSES = (PortalRigProps, PathsProps, LayerEntry, LayerPlacementProps)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.dpx_portal_rig = PointerProperty(type=PortalRigProps)
    bpy.types.Scene.dpx_paths = PointerProperty(type=PathsProps)
    bpy.types.Scene.dpx_layer_placement = PointerProperty(type=LayerPlacementProps)


def unregister():
    del bpy.types.Scene.dpx_layer_placement
    del bpy.types.Scene.dpx_paths
    del bpy.types.Scene.dpx_portal_rig
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
