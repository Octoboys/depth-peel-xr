"""Viewport overlays: head-plane wireframe + NxN camera-grid dots, plus
N+1 layer-boundary planes once layers are placed.

Drawn via a SpaceView3D POST_VIEW handler so the head-plane stays unselectable
and adds zero datablocks to the scene. Reads PortalRigProps + Portal world
transform each frame, so it tracks property and transform changes live.

The boundary-plane gizmos draw if the DepthPeelXR_Source collection is
visible (the conceptual home of the captured scene). Hiding Source via
Toggle Source / Preview hides them as a side-effect; head-plane wireframe
+ grid dots keep drawing throughout.
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from ..core.geometry import super_frustum_half_extents
from ..operators.inspect import SOURCE_COLLECTION


_HANDLE = None

_RECT_COLOR = (0.4, 0.8, 1.0, 1.0)
_GRID_COLOR = (1.0, 0.85, 0.2, 1.0)
_CENTRE_ANCHOR_COLOR = (1.0, 1.0, 1.0, 1.0)
_BOUNDARY_INTERIOR_COLOR = (0.3, 1.0, 0.4, 0.18)
_BOUNDARY_INTERIOR_EDGE_COLOR = (0.04, 0.18, 0.06, 1.0)
_BOUNDARY_DEEPEST_COLOR = (1.0, 0.4, 0.7, 0.22)
_BOUNDARY_DEEPEST_EDGE_COLOR = (0.22, 0.05, 0.12, 1.0)


def _boundary_quad_corners(M, d: float, half_w_p: float, half_h_p: float, ho: float):
    half_w, half_h = super_frustum_half_extents(d, half_w_p, half_h_p, ho)
    y = ho - d
    return [
        M @ Vector((-half_w, y, -half_h)),
        M @ Vector(( half_w, y, -half_h)),
        M @ Vector(( half_w, y,  half_h)),
        M @ Vector((-half_w, y,  half_h)),
    ]


def _boundary_depths(scene) -> list[float]:
    lp = getattr(scene, "dpx_layer_placement", None)
    if lp is None or len(lp.layers) == 0:
        return []
    layers = list(lp.layers)
    return [layers[0].near] + [layer.far for layer in layers]


def _source_collection_hidden() -> bool:
    """Layer-boundary gizmos draw if the Source collection is visible
    (the conceptual home of the scene the rig is capturing). When the
    user toggles to Preview, Source is hidden and so are the gizmos.
    No explicit Source collection yet → not hidden → gizmos draw.
    """
    coll = bpy.data.collections.get(SOURCE_COLLECTION)
    return coll is not None and coll.hide_viewport


def _draw():
    context = bpy.context
    scene = getattr(context, "scene", None)
    if scene is None or not hasattr(scene, "dpx_portal_rig"):
        return
    props = scene.dpx_portal_rig
    portal = props.portal_object
    if portal is None or portal.name not in bpy.data.objects:
        return
    if not portal.visible_get():
        return

    n = max(1, props.grid_n)
    w = props.portal_width
    h = props.portal_height
    ho = props.headplane_offset
    half_w, half_h = w * 0.5, h * 0.5
    M = portal.matrix_world

    corners = [
        M @ Vector((-half_w, ho, -half_h)),
        M @ Vector(( half_w, ho, -half_h)),
        M @ Vector(( half_w, ho,  half_h)),
        M @ Vector((-half_w, ho,  half_h)),
    ]
    rect_lines = []
    for i in range(4):
        rect_lines.append(corners[i])
        rect_lines.append(corners[(i + 1) % 4])

    grid_points = []
    if n == 1:
        grid_points.append(M @ Vector((0.0, ho, 0.0)))
    else:
        for r in range(n):
            z = half_h - (r / (n - 1)) * h
            for c in range(n):
                x = -half_w + (c / (n - 1)) * w
                grid_points.append(M @ Vector((x, ho, z)))

    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    shader.bind()

    boundary_depths = _boundary_depths(scene)
    if boundary_depths and not _source_collection_hidden():
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.line_width_set(1.5)
        for i, d in enumerate(boundary_depths):
            quad = _boundary_quad_corners(M, d, half_w, half_h, ho)
            tris = [quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]]
            edges = [
                quad[0], quad[1],
                quad[1], quad[2],
                quad[2], quad[3],
                quad[3], quad[0],
            ]
            is_deepest = i == len(boundary_depths) - 1
            fill = _BOUNDARY_DEEPEST_COLOR if is_deepest else _BOUNDARY_INTERIOR_COLOR
            edge = _BOUNDARY_DEEPEST_EDGE_COLOR if is_deepest else _BOUNDARY_INTERIOR_EDGE_COLOR
            shader.uniform_float("color", fill)
            batch_for_shader(shader, "TRIS", {"pos": tris}).draw(shader)
            shader.uniform_float("color", edge)
            batch_for_shader(shader, "LINES", {"pos": edges}).draw(shader)
        gpu.state.line_width_set(1.0)
        gpu.state.blend_set("NONE")
        gpu.state.depth_test_set("NONE")

    gpu.state.line_width_set(2.0)
    shader.uniform_float("color", _RECT_COLOR)
    batch_for_shader(shader, "LINES", {"pos": rect_lines}).draw(shader)

    gpu.state.point_size_set(8.0)
    shader.uniform_float("color", _GRID_COLOR)
    batch_for_shader(shader, "POINTS", {"pos": grid_points}).draw(shader)

    # Even N has no grid entry at the head-plane centre — the rig appends
    # one extra cam there (see `portal_rig.centre_view_index`). Draw it
    # in white so it's visibly distinct from the yellow grid samples.
    if n >= 2 and n % 2 == 0:
        centre_anchor = [M @ Vector((0.0, ho, 0.0))]
        shader.uniform_float("color", _CENTRE_ANCHOR_COLOR)
        batch_for_shader(shader, "POINTS", {"pos": centre_anchor}).draw(shader)

    gpu.state.line_width_set(1.0)
    gpu.state.point_size_set(1.0)


def register():
    global _HANDLE
    if _HANDLE is not None:
        return
    _HANDLE = bpy.types.SpaceView3D.draw_handler_add(_draw, (), "WINDOW", "POST_VIEW")


def unregister():
    global _HANDLE
    if _HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_HANDLE, "WINDOW")
        _HANDLE = None
