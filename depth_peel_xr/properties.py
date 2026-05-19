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
    from .operators.preview import invalidate_flat_preview

    resync_rig_from_props(context)
    repopulate_layers_if_active(context)
    invalidate_flat_preview(context)


def _on_viewing_resolution_change(self, context):
    """viewing_resolution only affects FUTURE captures (scene render
    resolution); it doesn't invalidate existing captures, consolidated
    outputs, or the on-screen reconstruction (which is built from
    portal dimensions, not viewing_resolution). Resync the rig so
    scene.render.resolution_x/y updates for the next capture, then
    stop — no preview invalidation."""
    from .operators.portal_rig import resync_rig_from_props

    resync_rig_from_props(context)


def _on_layer_param_change(self, context):
    from .operators.layer_placement import repopulate_layers_if_active
    from .operators.preview import invalidate_flat_preview

    repopulate_layers_if_active(context)
    invalidate_flat_preview(context)


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
        description="Maximum scene depth past the Portal that the rig captures (metres)",
        default=20.0,
        min=0.01,
        soft_max=200.0,
        precision=2,
        step=10,
        update=_on_layer_param_change,
    )
    min_depth: FloatProperty(
        name="Min Depth",
        description="Skip empty space between the Portal and the first scene content (metres)",
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
    viewing_resolution: EnumProperty(
        name="Viewing Resolution",
        description="Render resolution along the portal's longer axis (other dimension follows portal aspect)",
        items=(
            ("512", "512 (low)", ""),
            ("1024", "1024 (default)", ""),
            ("2048", "2048 (high)", ""),
            ("4096", "4096 (max)", ""),
        ),
        default="1024",
        update=_on_viewing_resolution_change,
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
        ("IMPORTANCE", "Histogram Importance", "Pure CDF inverse sampling — equal-population layers, no uniform safety net"),
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


_CAPTURE_ENGINE_ITEMS = (
    ("BLENDER_EEVEE", "EEVEE", "Fast raster engine — best for iteration"),
    ("CYCLES", "Cycles", "Physically-based path tracer — best for final output"),
)


class CaptureProps(PropertyGroup):
    engine: EnumProperty(
        name="Engine",
        description="Render engine used for capture",
        items=_CAPTURE_ENGINE_ITEMS,
        default="BLENDER_EEVEE",
    )


def _snap_to_pow2(value: int) -> int:
    """Round to the nearest power of 2 (>= 1). Used by the reconstruction
    quad-size sliders so values like 7 / 9 / 100 snap to 8 / 8 / 128."""
    if value < 1:
        return 1
    lower = 1 << (value.bit_length() - 1)
    upper = lower << 1
    return upper if (upper - value) < (value - lower) else lower


def _next_pow2(value: int) -> int:
    """Smallest power of 2 strictly greater than `value`."""
    if value < 1:
        return 1
    return 1 << value.bit_length()


def _prev_pow2(value: int) -> int:
    """Largest power of 2 strictly less than `value` (>= 1)."""
    if value <= 1:
        return 1
    if value & (value - 1) == 0:  # value is itself a power of 2
        return value >> 1
    return 1 << (value.bit_length() - 1)


# Per-prop last-value cache so update callbacks can detect ±1 deltas
# (slider increment arrows) vs. arbitrary jumps (drag / typed value).
# Single-scene assumption; multi-scene users see at most one off-by-one snap.
_last_quad_size = {"min": 8, "max": 128}


def _on_quad_size_change(prop_name: str, attr: str, value: int) -> int:
    """Map raw input to the snapped value. ±1 deltas (arrow clicks) step
    to next/prev power of 2; everything else snaps to nearest."""
    old = _last_quad_size[prop_name]
    if value == old:
        return value
    if abs(value - old) == 1:
        return _next_pow2(old) if value > old else _prev_pow2(old)
    return _snap_to_pow2(value)


def _on_min_quad_size_change(self, context):
    snapped = _on_quad_size_change("min", "min_quad_size", self.min_quad_size)
    if snapped != self.min_quad_size:
        self.min_quad_size = snapped  # idempotent re-entry, no recursion
    _last_quad_size["min"] = self.min_quad_size


def _on_max_quad_size_change(self, context):
    snapped = _on_quad_size_change("max", "max_quad_size", self.max_quad_size)
    if snapped != self.max_quad_size:
        self.max_quad_size = snapped
    _last_quad_size["max"] = self.max_quad_size


class ReconstructionProps(PropertyGroup):
    max_quad_size: IntProperty(
        name="Max Quad Size",
        description="Largest quadtree leaf size (canonical pixels, snaps to power of 2)",
        default=128,
        min=8,
        soft_max=512,
        subtype="PIXEL",
        update=_on_max_quad_size_change,
    )
    min_quad_size: IntProperty(
        name="Min Quad Size",
        description="Smallest quadtree leaf size (canonical pixels, snaps to power of 2) — lower = finer silhouettes",
        default=8,
        min=1,
        soft_max=64,
        subtype="PIXEL",
        update=_on_min_quad_size_change,
    )


CLASSES = (PortalRigProps, PathsProps, LayerEntry, LayerPlacementProps, CaptureProps, ReconstructionProps)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.dpx_portal_rig = PointerProperty(type=PortalRigProps)
    bpy.types.Scene.dpx_paths = PointerProperty(type=PathsProps)
    bpy.types.Scene.dpx_layer_placement = PointerProperty(type=LayerPlacementProps)
    bpy.types.Scene.dpx_capture = PointerProperty(type=CaptureProps)
    bpy.types.Scene.dpx_reconstruction = PointerProperty(type=ReconstructionProps)


def unregister():
    del bpy.types.Scene.dpx_reconstruction
    del bpy.types.Scene.dpx_capture
    del bpy.types.Scene.dpx_layer_placement
    del bpy.types.Scene.dpx_paths
    del bpy.types.Scene.dpx_portal_rig
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
