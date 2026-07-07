"""Flat-cutout preview of captured layers.

Builds one alpha-cut plane per layer at its `near` depth, textured from the
centre camera's flat EXR. Combined.RGB drives an unlit emission;
Combined.Alpha (populated by film_transparent=True during capture) drives
the alpha cutout. Mutually-exclusive preview that lives in DepthPeelXR_Preview
so the user can toggle vs. source via the existing Inspect toggle.

The plane is sized to the **centre camera's** frustum cross-section at
the layer's near depth (not the super-frustum) — centre-only by design,
acting as the pre-consolidation sanity check. The full reconstructions
(Build Quadtree Planes / Build Displaced Quadtree) size to the super-
frustum cross-section.

`invalidate_flat_preview` is invoked from properties.py update callbacks
and from operators that touch captures/rig state, so the preview only ever
reflects state matching the captures on disk.
"""

from __future__ import annotations

from pathlib import Path

import bpy
from bpy.types import Operator
from mathutils import Matrix

from .. import paths as paths_mod
from ..core.geometry import centre_frustum_half_extents
from .inspect import PREVIEW_COLLECTION, SOURCE_COLLECTION, get_view_cameras_sorted
from .portal_rig import centre_view_index, ensure_collection, link_unique


_PREVIEW_PREFIX = "dpxr_preview_layer_"
# Reconstruction-mesh objects share the cleanup path — mutually
# exclusive with simple-plane preview, both live in DepthPeelXR_Preview.
_RECON_PREFIX = "dpxr_recon_layer_"
_GEO_PREFIXES = (_PREVIEW_PREFIX, _RECON_PREFIX)


def _layer_centre_exr(captures: Path, layer_idx: int, view_idx: int) -> Path:
    return captures / f"layer_{layer_idx:02d}" / f"view_{view_idx:02d}_color.exr"


def expected_captures_present(context) -> bool:
    """True if the captures dir holds at least the first layer's centre-view
    EXR. Capture writes layers atomically and cleanup wipes the whole tree,
    so the first file's presence is a reliable proxy for the rest. Called
    from operator poll() and the N-panel — both hot, so it stays cheap."""
    captures = paths_mod.captures_dir(context)
    if captures is None:
        return False
    scene = context.scene
    lp = scene.dpx_layer_placement
    rig = getattr(scene, "dpx_portal_rig", None)
    cams = get_view_cameras_sorted()
    if not lp.layers or not cams or rig is None:
        return False
    centre = centre_view_index(rig.grid_n)
    return _layer_centre_exr(captures, 0, centre).exists()


def _build_preview_mesh(name: str, half_w: float, half_h: float, y: float):
    mesh = bpy.data.meshes.new(name)
    verts = [
        (-half_w, y, -half_h),
        ( half_w, y, -half_h),
        ( half_w, y,  half_h),
        (-half_w, y,  half_h),
    ]
    # Reverse winding so the face normal points portal-local +Y (toward
    # head-plane / viewer). The mesh is parented to the portal with
    # identity parent_inverse, so mesh-local axes coincide with portal-
    # local — rotating the portal rotates the normal with it.
    faces = [(0, 3, 2, 1)]
    mesh.from_pydata(verts, [], faces)
    # Centre camera: image right = portal-local -X (per camera_local_rotation_4x4),
    # image up = +Z. EXR pixel (0,0) is bottom-left of image = (+X, -Z) vertex.
    uvs = [(1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for loop in mesh.loops:
        uv_layer.data[loop.index].uv = uvs[loop.vertex_index]
    mesh.update()
    return mesh


def _load_layer_image(exr_path: Path, image_name: str) -> bpy.types.Image:
    existing = bpy.data.images.get(image_name)
    if existing is not None:
        bpy.data.images.remove(existing)
    img = bpy.data.images.load(str(exr_path), check_existing=False)
    img.name = image_name
    # Captured with save_as_render=False — pixels are scene-linear, no view
    # transform. Tell Blender to treat them as raw data.
    img.colorspace_settings.name = "Non-Color"
    return img


def _build_preview_material(
    name: str, image: bpy.types.Image
) -> bpy.types.Material:
    """Emission + alpha-clip — for border (silhouette) reconstruction
    halves and the simple-plane preview. Coverage drives alpha cutout."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    # DITHERED gives sharp cutout with depth writes only where opaque —
    # Eevee Next's stand-in for the old "alpha clip" blend mode.
    mat.surface_render_method = "DITHERED"

    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")

    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image

    emission = nt.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0

    transparent = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")

    nt.links.new(tex.outputs["Color"], emission.inputs["Color"])
    # Alpha encodes coverage (capture renders with film_transparent=True).
    nt.links.new(tex.outputs["Alpha"], mix.inputs["Fac"])
    nt.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    nt.links.new(emission.outputs["Emission"], mix.inputs[2])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    return mat


def _build_world_projected_uv_chain(
    nt: bpy.types.NodeTree,
    portal: bpy.types.Object,
    ho: float,
    half_w_p: float,
    half_h_p: float,
) -> bpy.types.NodeSocket:
    """Node chain: Texture Coordinate(→object=portal) → portal-local (x,y,z)
    → canonical UV via the forward of `inverse_canonical_projection`.

    Result socket carries UV as a Vector (u, v, 0). Feed into an Image
    Texture node's Vector input to replace the mesh's stored UVs — texels
    then track world position, so vert moves (e.g. from Decimate) don't
    slide texture across the surface.
    """
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.object = portal

    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["Object"], sep.inputs["Vector"])

    ho_half = 0.5 * ho
    # u = 0.5 − x·(ho / (4·hw))·(1/d) — precompute the negative
    # sign into the multiplier so a single Multiply-Add resolves u.
    ku = -ho / (4.0 * half_w_p)
    kv = ho / (4.0 * half_h_p)

    # d = ho/2 − y_portal
    d_node = nt.nodes.new("ShaderNodeMath")
    d_node.operation = "SUBTRACT"
    d_node.inputs[0].default_value = ho_half
    nt.links.new(sep.outputs["Y"], d_node.inputs[1])

    # 1/d
    inv_d = nt.nodes.new("ShaderNodeMath")
    inv_d.operation = "DIVIDE"
    inv_d.inputs[0].default_value = 1.0
    nt.links.new(d_node.outputs["Value"], inv_d.inputs[1])

    # x · ku
    x_scaled = nt.nodes.new("ShaderNodeMath")
    x_scaled.operation = "MULTIPLY"
    nt.links.new(sep.outputs["X"], x_scaled.inputs[0])
    x_scaled.inputs[1].default_value = ku

    # u = 0.5 + (x · ku) · (1/d)
    u_node = nt.nodes.new("ShaderNodeMath")
    u_node.operation = "MULTIPLY_ADD"
    nt.links.new(x_scaled.outputs["Value"], u_node.inputs[0])
    nt.links.new(inv_d.outputs["Value"], u_node.inputs[1])
    u_node.inputs[2].default_value = 0.5

    # z · kv
    z_scaled = nt.nodes.new("ShaderNodeMath")
    z_scaled.operation = "MULTIPLY"
    nt.links.new(sep.outputs["Z"], z_scaled.inputs[0])
    z_scaled.inputs[1].default_value = kv

    # v = 0.5 + (z · kv) · (1/d)
    v_node = nt.nodes.new("ShaderNodeMath")
    v_node.operation = "MULTIPLY_ADD"
    nt.links.new(z_scaled.outputs["Value"], v_node.inputs[0])
    nt.links.new(inv_d.outputs["Value"], v_node.inputs[1])
    v_node.inputs[2].default_value = 0.5

    combine = nt.nodes.new("ShaderNodeCombineXYZ")
    nt.links.new(u_node.outputs["Value"], combine.inputs["X"])
    nt.links.new(v_node.outputs["Value"], combine.inputs["Y"])
    return combine.outputs["Vector"]


def _build_opaque_material(
    name: str,
    image: bpy.types.Image,
    portal: bpy.types.Object,
    ho: float,
    half_w_p: float,
    half_h_p: float,
) -> bpy.types.Material:
    """Emission only — for interior halves. No alpha test, no transparent
    BSDF; early-z friendly for the runtime, and acts as a clean occluder
    for the cull's rasterised closer-surface.

    UVs are computed in-shader from world-space vertex position via the
    portal's local frame (see `_build_world_projected_uv_chain`) rather
    than sampled from the mesh's stored UV layer. This decouples texel
    flow from mesh topology so aggressive Decimate on the opaque half
    won't cause texture swimming — the stored UVs stay on the mesh but
    the material ignores them."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.surface_render_method = "DITHERED"

    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    uv_socket = _build_world_projected_uv_chain(
        nt, portal, ho, half_w_p, half_h_p
    )
    nt.links.new(uv_socket, tex.inputs["Vector"])
    emission = nt.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0

    nt.links.new(tex.outputs["Color"], emission.inputs["Color"])
    nt.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


def _remove_preview_objects() -> int:
    """Wipe both simple-plane (`dpxr_preview_layer_*`) and reconstruction
    (`dpxr_recon_layer_*`) entities. They share the cleanup path because
    they're mutually exclusive — only one geo flavour at a time. Also
    drops the displaced-mesh pre-cull cache, since it's keyed by the
    recon mesh names we're about to remove."""
    from .recon_cache import clear_cache

    def starts_with_any(name: str) -> bool:
        return any(name.startswith(p) for p in _GEO_PREFIXES)

    removed = 0
    for obj in [o for o in bpy.data.objects if starts_with_any(o.name)]:
        mesh = obj.data if isinstance(obj.data, bpy.types.Mesh) else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
        removed += 1
    for mesh in [m for m in bpy.data.meshes if starts_with_any(m.name)]:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for mat in [m for m in bpy.data.materials if starts_with_any(m.name)]:
        if mat.users == 0:
            bpy.data.materials.remove(mat)
    for img in [i for i in bpy.data.images if starts_with_any(i.name)]:
        if img.users == 0:
            bpy.data.images.remove(img)
    clear_cache()
    return removed


def _restore_source_view() -> None:
    """Force Source visible / Preview hidden — the inverse of what
    preview_flat sets. Used whenever the preview is torn down so the user
    isn't left staring at an empty Preview collection."""
    source = bpy.data.collections.get(SOURCE_COLLECTION)
    if source is not None:
        source.hide_viewport = False
    preview = bpy.data.collections.get(PREVIEW_COLLECTION)
    if preview is not None:
        preview.hide_viewport = True


def invalidate_flat_preview(context) -> None:
    """Drop all reconstruction geo (simple planes AND quadtree meshes) and
    flip back to source view. Cheap no-op when nothing exists. Called from
    properties.py update callbacks whenever a rig/layer param changes and
    the on-screen geo would become stale, plus from operators that
    invalidate captures."""
    coll = bpy.data.collections.get(PREVIEW_COLLECTION)
    if coll is None or not coll.objects:
        return
    _remove_preview_objects()
    _restore_source_view()


class DEPTHPEELXR_OT_preview_flat(Operator):
    bl_idname = "depthpeelxr.preview_flat"
    bl_label = "Build Simple Planes"
    bl_description = "Per-layer flat planes with alpha cutouts (preview only)"
    bl_options = {"REGISTER", "UNDO"}

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
            and expected_captures_present(context)
        )

    def execute(self, context):
        scene = context.scene
        rig = scene.dpx_portal_rig
        lp = scene.dpx_layer_placement
        portal = rig.portal_object

        captures = paths_mod.captures_dir(context)
        centre_idx = centre_view_index(rig.grid_n)

        _remove_preview_objects()

        coll = ensure_collection(PREVIEW_COLLECTION)
        ho = rig.headplane_offset
        half_w_p = rig.portal_width * 0.5
        half_h_p = rig.portal_height * 0.5

        built = 0
        for li, layer in enumerate(lp.layers):
            exr = _layer_centre_exr(captures, li, centre_idx)
            if not exr.exists():
                continue

            d = layer.near
            half_w, half_h = centre_frustum_half_extents(d, half_w_p, half_h_p, ho)
            y = ho - d

            base_name = f"{_PREVIEW_PREFIX}{li:02d}"
            mesh = _build_preview_mesh(f"{base_name}_mesh", half_w, half_h, y)
            image = _load_layer_image(exr, f"{base_name}_img")
            mat = _build_preview_material(f"{base_name}_mat", image)
            mesh.materials.append(mat)

            obj = bpy.data.objects.new(base_name, mesh)
            obj.parent = portal
            obj.matrix_parent_inverse = Matrix.Identity(4)
            link_unique(obj, coll)
            built += 1

        if built == 0:
            self.report({"ERROR"}, "No layer captures found")
            return {"CANCELLED"}

        # Force into "preview view": Source hidden, Preview shown. Same end
        # state as one click of Toggle Source / Preview, no-op if already
        # there. Layer-boundary gizmos key off SOURCE collection visibility
        # so they hide as a side-effect.
        ensure_collection(SOURCE_COLLECTION).hide_viewport = True
        coll.hide_viewport = False

        self.report({"INFO"}, f"Built {built} flat-cutout preview plane(s)")
        return {"FINISHED"}


class DEPTHPEELXR_OT_cleanup_preview(Operator):
    bl_idname = "depthpeelxr.cleanup_preview"
    bl_label = "Clean Up Geo"
    bl_description = "Remove all reconstruction geometry"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        removed = _remove_preview_objects()
        # Unconditional restore — explicit cleanup means "I want source view"
        # regardless of whether preview existed.
        _restore_source_view()
        if removed == 0:
            self.report({"INFO"}, "No preview entities to remove")
        else:
            self.report({"INFO"}, f"Removed {removed} preview object(s)")
        return {"FINISHED"}


CLASSES = (
    DEPTHPEELXR_OT_preview_flat,
    DEPTHPEELXR_OT_cleanup_preview,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
