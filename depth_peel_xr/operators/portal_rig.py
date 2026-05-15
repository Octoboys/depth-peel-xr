"""Portal Rig: add / cleanup / live resync.

Hierarchy built by `Add Portal Rig`:

    DepthPeelXR_Rig                         (collection)
      ├─ DPX_Portal                         (mesh: 2×2 in local XZ, +Y normal)
          └─ DPX_HeadplaneAnchor            (Empty, +Y offset = headplane_offset)
              └─ DPX_View_NN                (N² perspective cameras, off-axis)

All children inherit Portal's transform via Blender parenting, so moving /
rotating the Portal moves the whole rig. Cameras use asymmetric (off-axis)
frustums so every view's image frames the same Portal rectangle.
"""

import bpy
from bpy.types import Object, Operator
from mathutils import Matrix, Vector


RIG_COLLECTION = "DepthPeelXR_Rig"
PORTAL_NAME = "DPX_Portal"
ANCHOR_NAME = "DPX_HeadplaneAnchor"
CAM_PREFIX = "DPX_View_"


def _camera_local_rotation_4x4() -> Matrix:
    # Map cam axes (in image space) into Anchor-local axes:
    #   cam +X = anchor -X   (image right is anchor left, expected for a
    #                         camera facing back toward -Y from +Y)
    #   cam +Y = anchor +Z   (image up = world up)
    #   cam +Z = anchor +Y   (cam back = away from scene)
    return Matrix((
        (-1.0, 0.0, 0.0, 0.0),
        ( 0.0, 0.0, 1.0, 0.0),
        ( 0.0, 1.0, 0.0, 0.0),
        ( 0.0, 0.0, 0.0, 1.0),
    ))


def _camera_grid_positions(n: int, headplane_w: float, headplane_h: float):
    if n <= 0:
        return
    if n == 1:
        yield 0, 0.0, 0.0
        return
    half_w = headplane_w / 2.0
    half_h = headplane_h / 2.0
    for r in range(n):
        z = half_h - (r / (n - 1)) * headplane_h
        for c in range(n):
            x = -half_w + (c / (n - 1)) * headplane_w
            yield r * n + c, x, z


def ensure_collection(name: str) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def _link_unique(obj: Object, coll: bpy.types.Collection) -> None:
    if obj.name not in coll.objects:
        coll.objects.link(obj)


def _set_portal_mesh_size(portal: Object, width: float, height: float) -> None:
    """Build the portal as a hollow rectangular frame surrounding the aperture.

    The inner aperture matches the (width × height) portal bounds — the same
    rectangle the head-plane and camera frustums frame. The frame extends
    outwards by `thickness = 15% of the larger portal dimension`, so the
    aperture stays clean and unobscured by geometry.
    """
    half_w, half_h = width * 0.5, height * 0.5
    thickness = max(width, height) * 0.15
    ow, oh = half_w + thickness, half_h + thickness

    mesh = portal.data
    mesh.clear_geometry()
    mesh.from_pydata(
        [
            (    -ow, 0.0,    -oh),   # 0: outer BL
            (     ow, 0.0,    -oh),   # 1: outer BR
            (     ow, 0.0,     oh),   # 2: outer TR
            (    -ow, 0.0,     oh),   # 3: outer TL
            (-half_w, 0.0, -half_h),  # 4: inner (aperture) BL
            ( half_w, 0.0, -half_h),  # 5: inner BR
            ( half_w, 0.0,  half_h),  # 6: inner TR
            (-half_w, 0.0,  half_h),  # 7: inner TL
        ],
        [],
        [
            (0, 1, 5, 4),  # bottom strip
            (1, 2, 6, 5),  # right strip
            (2, 3, 7, 6),  # top strip
            (3, 0, 4, 7),  # left strip
        ],
    )
    mesh.update()


def _build_portal(coll: bpy.types.Collection, width: float, height: float) -> Object:
    mesh = bpy.data.meshes.new(PORTAL_NAME)
    obj = bpy.data.objects.new(PORTAL_NAME, mesh)
    obj.show_wire = True
    _link_unique(obj, coll)
    _set_portal_mesh_size(obj, width, height)
    return obj


def _build_anchor(coll: bpy.types.Collection, parent: Object, offset: float) -> Object:
    anchor = bpy.data.objects.get(ANCHOR_NAME)
    if anchor is None or anchor.type != 'EMPTY':
        anchor = bpy.data.objects.new(ANCHOR_NAME, None)
        anchor.empty_display_type = 'PLAIN_AXES'
        anchor.empty_display_size = 0.2
    _link_unique(anchor, coll)
    anchor.parent = parent
    anchor.matrix_parent_inverse = Matrix.Identity(4)
    anchor.location = (0.0, offset, 0.0)
    anchor.rotation_euler = (0.0, 0.0, 0.0)
    anchor.scale = (1.0, 1.0, 1.0)
    return anchor


def _configure_camera(
    cam_obj: Object,
    cx: float,
    cz: float,
    ho: float,
    portal_w: float,
    portal_h: float,
) -> None:
    rot = _camera_local_rotation_4x4()
    cam_obj.matrix_local = Matrix.Translation(Vector((cx, 0.0, cz))) @ rot

    cam = cam_obj.data
    cam.type = 'PERSP'
    cam.lens_unit = 'MILLIMETERS'
    cam.sensor_fit = 'HORIZONTAL'
    cam.sensor_width = 36.0
    # Symmetric horizontal FoV that frames portal width Wp at depth ho.
    cam.lens = 36.0 * ho / portal_w
    # Off-axis shift so the principal point coincides with portal centre.
    cam.shift_x = cx / portal_w
    cam.shift_y = -cz / portal_w
    cam.clip_start = max(0.001, ho * 0.01)
    cam.clip_end = max(100.0, ho * 100.0)


def _spawn_camera(
    idx: int,
    coll: bpy.types.Collection,
    anchor: Object,
    cx: float,
    cz: float,
    ho: float,
    portal_w: float,
    portal_h: float,
) -> Object:
    name = f"{CAM_PREFIX}{idx:02d}"
    cam_obj = bpy.data.objects.get(name)
    if cam_obj is None or cam_obj.type != 'CAMERA':
        cam_data = bpy.data.cameras.new(name)
        cam_obj = bpy.data.objects.new(name, cam_data)
    _link_unique(cam_obj, coll)
    cam_obj.parent = anchor
    cam_obj.matrix_parent_inverse = Matrix.Identity(4)
    _configure_camera(cam_obj, cx, cz, ho, portal_w, portal_h)
    return cam_obj


def _existing_view_cameras() -> list[Object]:
    return [
        obj for obj in bpy.data.objects
        if obj.name.startswith(CAM_PREFIX) and obj.type == 'CAMERA'
    ]


def _delete_object(obj: Object) -> None:
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if data is None:
        return
    if data.users != 0:
        return
    if isinstance(data, bpy.types.Mesh):
        bpy.data.meshes.remove(data)
    elif isinstance(data, bpy.types.Camera):
        bpy.data.cameras.remove(data)


def _apply_portal_render_resolution(context, portal_w: float, portal_h: float) -> None:
    """Set scene render resolution so its aspect matches the portal aperture.

    Without this, the camera's vertical FoV (derived from render aspect under
    sensor_fit='HORIZONTAL') doesn't match the portal's actual height — the
    frustum gizmo and the rendered image both letterbox / underflow the
    aperture vertically.
    """
    props = context.scene.dpx_portal_rig
    long_res = max(1, int(props.viewing_resolution))
    if portal_w >= portal_h:
        res_x = long_res
        res_y = max(1, round(long_res * portal_h / portal_w))
    else:
        res_y = long_res
        res_x = max(1, round(long_res * portal_w / portal_h))
    render = context.scene.render
    render.resolution_x = res_x
    render.resolution_y = res_y
    render.pixel_aspect_x = 1.0
    render.pixel_aspect_y = 1.0


def _build_or_resync_rig(context, *, create_portal_if_missing: bool) -> bool:
    props = context.scene.dpx_portal_rig
    coll = ensure_collection(RIG_COLLECTION)

    portal_w = props.portal_width
    portal_h = props.portal_height
    if portal_w <= 0.0 or portal_h <= 0.0:
        return False

    portal = props.portal_object
    if portal is None or portal.name not in bpy.data.objects:
        if not create_portal_if_missing:
            return False
        portal = _build_portal(coll, portal_w, portal_h)
        props.portal_object = portal
    else:
        _set_portal_mesh_size(portal, portal_w, portal_h)
    _link_unique(portal, coll)

    anchor = _build_anchor(coll, portal, props.headplane_offset)

    for cam in _existing_view_cameras():
        _delete_object(cam)

    for idx, cx, cz in _camera_grid_positions(
        props.grid_n, portal_w, portal_h
    ):
        _spawn_camera(idx, coll, anchor, cx, cz, props.headplane_offset, portal_w, portal_h)

    _apply_portal_render_resolution(context, portal_w, portal_h)

    return True


def resync_rig_from_props(context) -> None:
    if context is None or not hasattr(context, "scene"):
        return
    props = context.scene.dpx_portal_rig
    if props.portal_object is None:
        return
    _build_or_resync_rig(context, create_portal_if_missing=False)


class DEPTHPEELXR_OT_add_portal_rig(Operator):
    bl_idname = "depthpeelxr.add_portal_rig"
    bl_label = "Add Portal Rig"
    bl_description = "Create the Portal, head-plane Anchor, and N×N camera grid"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene.dpx_portal_rig.portal_object is None

    def execute(self, context):
        props = context.scene.dpx_portal_rig
        for name in (
            "portal_width",
            "portal_height",
            "headplane_offset",
            "grid_n",
            "viewing_resolution",
        ):
            props.property_unset(name)
        ok = _build_or_resync_rig(context, create_portal_if_missing=True)
        if not ok:
            self.report({"ERROR"}, "Failed to build Portal Rig")
            return {"CANCELLED"}
        from .. import paths as paths_mod

        paths_props = context.scene.dpx_paths
        if not paths_props.project_root:
            derived = paths_mod.derived_project_root(context)
            if derived is not None:
                paths_props.project_root = str(derived)
        return {"FINISHED"}


class DEPTHPEELXR_OT_cleanup_portal_rig(Operator):
    bl_idname = "depthpeelxr.cleanup_portal_rig"
    bl_label = "Clean Up Portal Rig"
    bl_description = "Remove the rig collection and all contained entities"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        coll = bpy.data.collections.get(RIG_COLLECTION)
        if coll is not None:
            for obj in list(coll.objects):
                _delete_object(obj)
            bpy.data.collections.remove(coll)

        portal = bpy.data.objects.get(PORTAL_NAME)
        if portal is not None:
            _delete_object(portal)
        anchor = bpy.data.objects.get(ANCHOR_NAME)
        if anchor is not None:
            _delete_object(anchor)
        for cam in _existing_view_cameras():
            _delete_object(cam)

        context.scene.dpx_portal_rig.portal_object = None
        return {"FINISHED"}


CLASSES = (
    DEPTHPEELXR_OT_add_portal_rig,
    DEPTHPEELXR_OT_cleanup_portal_rig,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
