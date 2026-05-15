"""Viewport overlays: head-plane wireframe + NxN camera-grid dots.

Drawn via a SpaceView3D POST_VIEW handler so the head-plane stays unselectable
and adds zero datablocks to the scene. Reads PortalRigProps + Portal world
transform each frame, so it tracks property and transform changes live.
"""

import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector


_HANDLE = None

_RECT_COLOR = (0.4, 0.8, 1.0, 1.0)
_GRID_COLOR = (1.0, 0.85, 0.2, 1.0)


def _draw():
    context = bpy.context
    scene = getattr(context, "scene", None)
    if scene is None or not hasattr(scene, "dpx_portal_rig"):
        return
    props = scene.dpx_portal_rig
    portal = props.portal_object
    if portal is None or portal.name not in bpy.data.objects:
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

    gpu.state.line_width_set(2.0)
    shader.uniform_float("color", _RECT_COLOR)
    batch_for_shader(shader, "LINES", {"pos": rect_lines}).draw(shader)

    gpu.state.point_size_set(8.0)
    shader.uniform_float("color", _GRID_COLOR)
    batch_for_shader(shader, "POINTS", {"pos": grid_points}).draw(shader)

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
