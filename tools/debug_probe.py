"""Headless probe-debug script.

Build a tiny test scene, run the depth probe (matching what
DEPTHPEELXR_OT_place_layers does for HYBRID), and dump diagnostics so we
can see whether the Viewer Node depth read is sane.

Run with:
   /Applications/Blender_5.app/Contents/MacOS/blender --background --python tools/debug_probe.py
"""

import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "depth_peel_xr"))

# Pull in the standalone helpers without registering the whole add-on.
# (The probe doesn't depend on operator registration.)
from operators.portal_rig import camera_local_rotation_4x4  # noqa: E402


PROBE_NODE_TREE_NAME = "DPX_ProbeCompositor_temp"
VIEWER_IMAGE_NAME = "Viewer Node"


def build_test_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # Three cubes at known depths along -Y from origin.
    for i, y in enumerate((-3.0, -8.0, -18.0)):
        bpy.ops.mesh.primitive_cube_add(size=1.5, location=(0.0, y, 0.0))

    return scene


def build_probe_compositor(scene):
    saved_tree = scene.compositing_node_group
    tree = bpy.data.node_groups.new(PROBE_NODE_TREE_NAME, "CompositorNodeTree")
    scene.compositing_node_group = tree
    tree.nodes.clear()

    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.scene = scene
    viewer = tree.nodes.new("CompositorNodeViewer")
    tree.links.new(rl.outputs["Depth"], viewer.inputs["Image"])
    return saved_tree


def restore_compositor(scene, saved_tree):
    tree = scene.compositing_node_group
    scene.compositing_node_group = saved_tree
    if tree is not None and tree.name == PROBE_NODE_TREE_NAME:
        bpy.data.node_groups.remove(tree)


def create_probe_camera(scene, headplane_offset, portal_width):
    cam_data = bpy.data.cameras.new("DPX_Probe_temp")
    cam_obj = bpy.data.objects.new("DPX_Probe_temp", cam_data)
    scene.collection.objects.link(cam_obj)

    rot = camera_local_rotation_4x4()
    cam_obj.matrix_world = Matrix.Translation(Vector((0.0, headplane_offset, 0.0))) @ rot

    cam_data.type = "PERSP"
    cam_data.lens_unit = "MILLIMETERS"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0
    cam_data.lens = 36.0 * headplane_offset / portal_width
    cam_data.shift_x = 0.0
    cam_data.shift_y = 0.0
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0
    return cam_obj


def read_viewer_depth(width, height):
    img = bpy.data.images.get(VIEWER_IMAGE_NAME)
    if img is None:
        return None, "Viewer Node image missing"
    if img.channels == 0 or img.size[0] == 0:
        return None, f"empty: channels={img.channels}, size={tuple(img.size)}"
    n = img.size[0] * img.size[1] * img.channels
    buf = np.empty(n, dtype=np.float32)
    img.pixels.foreach_get(buf)
    arr = buf.reshape(img.size[1], img.size[0], img.channels)
    return arr, None


def run_probe(engine):
    print(f"\n=== Engine: {engine} ===")
    scene = build_test_scene()
    view_layer = scene.view_layers[0]

    width, height = 256, 256

    saved = {
        "engine": scene.render.engine,
        "res_x": scene.render.resolution_x,
        "res_y": scene.render.resolution_y,
        "use_pass_z": view_layer.use_pass_z,
        "camera": scene.camera,
    }
    saved_tree = probe_cam = None
    try:
        scene.render.engine = engine
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.resolution_percentage = 100
        view_layer.use_pass_z = True

        probe_cam = create_probe_camera(scene, headplane_offset=1.5, portal_width=4.0)
        scene.camera = probe_cam

        saved_tree = build_probe_compositor(scene)

        bpy.ops.render.render(write_still=False)

        print(f"  bpy.data.images after render:")
        for img in bpy.data.images:
            print(f"    - {img.name!r}: type={img.type} size={tuple(img.size)} channels={img.channels}")

        arr, err = read_viewer_depth(width, height)
        if arr is None:
            print(f"  read failed: {err}")
            return

        print(f"  full RGBA shape: {arr.shape} dtype: {arr.dtype}")
        for ch in range(arr.shape[2]):
            c = arr[:, :, ch]
            finite = c[np.isfinite(c)]
            if finite.size:
                print(
                    f"  channel {ch}: min={finite.min():.4g} "
                    f"max={finite.max():.4g} mean={finite.mean():.4g} "
                    f"unique-vals={len(np.unique(finite))}"
                )
            else:
                print(f"  channel {ch}: all non-finite")

        depth = arr[:, :, 0]
        print(
            f"  centre pixel R: {depth[height // 2, width // 2]:.4f}"
        )
        print(
            f"  in-range (0.5..50): "
            f"{int(((depth > 0.5) & (depth < 50.0)).sum())}/{depth.size}"
        )
    finally:
        if saved_tree is not None or scene.compositing_node_group is not None:
            restore_compositor(scene, saved_tree)
        if probe_cam is not None:
            data = probe_cam.data
            bpy.data.objects.remove(probe_cam, do_unlink=True)
            if isinstance(data, bpy.types.Camera) and data.users == 0:
                bpy.data.cameras.remove(data)
        scene.render.engine = saved["engine"]
        scene.render.resolution_x = saved["res_x"]
        scene.render.resolution_y = saved["res_y"]
        view_layer.use_pass_z = saved["use_pass_z"]
        scene.camera = saved["camera"]


for engine in ("BLENDER_EEVEE", "BLENDER_WORKBENCH"):
    try:
        run_probe(engine)
    except Exception as e:
        print(f"  exception: {e!r}")
