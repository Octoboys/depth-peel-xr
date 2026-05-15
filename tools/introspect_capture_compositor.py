"""Verify file_output_items socket types for RGBA + depth output.

Run via:
   /Applications/Blender_5.app/Contents/MacOS/blender --background --python tools/introspect_capture_compositor.py
"""

import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

ng = bpy.data.node_groups.new("DPX_capture_check", "CompositorNodeTree")
out = ng.nodes.new("CompositorNodeOutputFile")

# Try common socket_type values; print which succeed.
for socket_type in ("COLOR", "RGBA", "VECTOR", "FLOAT", "IMAGE"):
    out.file_output_items.clear()
    try:
        item = out.file_output_items.new(socket_type, "Test")
        # Verify the input was created and what type it is.
        if "Test" in out.inputs:
            inp = out.inputs["Test"]
            print(f"  {socket_type!r:10}: OK   input type={inp.type!r}")
        else:
            print(f"  {socket_type!r:10}: OK but no 'Test' input found")
    except Exception as e:
        print(f"  {socket_type!r:10}: ERR  {e!r}")

# Now build a real Render Layers + multilayer File Output and try connecting.
print("\n=== Building Render Layers → multilayer File Output ===")
rl = ng.nodes.new("CompositorNodeRLayers")
rl.scene = scene
print(f"Render Layers outputs: {[o.name + '(' + o.type + ')' for o in rl.outputs]}")

out2 = ng.nodes.new("CompositorNodeOutputFile")
out2.format.file_format = "OPEN_EXR_MULTILAYER"
out2.format.color_depth = "32"
out2.file_output_items.clear()

for st_name, item_name in [("RGBA", "Combined"), ("FLOAT", "Depth")]:
    try:
        out2.file_output_items.new(st_name, item_name)
        print(f"  Added item {item_name!r} type={st_name!r}")
    except Exception as e:
        print(f"  Failed {item_name!r} type={st_name!r}: {e!r}")

print(f"out2 inputs after items: {[i.name + '(' + i.type + ')' for i in out2.inputs]}")

# Diagnose the input lookup.
print("\n=== Input identifiers vs names ===")
for i, inp in enumerate(out2.inputs):
    print(f"  [{i}] name={inp.name!r}, identifier={inp.identifier!r}, type={inp.type!r}")
print(f"  out2.inputs.get('Depth'): {out2.inputs.get('Depth')!r}")
print(f"  out2.inputs[1]: {out2.inputs[1]!r}")
print(f"  Direct subscript out2.inputs['Depth']:")
try:
    inp = out2.inputs["Depth"]
    print(f"    OK -> {inp!r}")
except Exception as e:
    print(f"    ERR {e!r}")
print(f"  Direct subscript out2.inputs['Combined']:")
try:
    inp = out2.inputs["Combined"]
    print(f"    OK -> {inp!r}")
except Exception as e:
    print(f"    ERR {e!r}")

try:
    # Test: link with both sides as already-resolved socket objects.
    print("\n=== Linking with pre-resolved sockets ===")
    print("  about to fetch img_out", flush=True)
    img_out = rl.outputs["Image"]
    print(f"  img_out: {img_out!r}", flush=True)
    depth_out = rl.outputs["Depth"]
    print(f"  depth_out: {depth_out!r}", flush=True)
    combined_in = out2.inputs[0]
    print(f"  combined_in: {combined_in!r}", flush=True)
    depth_in = out2.inputs[1]
    print(f"  depth_in: {depth_in!r}", flush=True)
    for from_sock, to_sock, label in [
        (img_out, combined_in, "img→combined"),
        (depth_out, depth_in, "depth→depth"),
    ]:
        try:
            link = ng.links.new(from_sock, to_sock)
            print(f"  {label}: OK (link={link!r})", flush=True)
        except Exception as e:
            print(f"  {label}: ERR {e!r}", flush=True)
except Exception as outer_e:
    import traceback
    print(f"  OUTER EXC: {outer_e!r}", flush=True)
    traceback.print_exc()
