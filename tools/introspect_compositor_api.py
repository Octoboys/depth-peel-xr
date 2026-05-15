"""Introspect Blender 5.x compositor API. Run via:
   blender --background --python tools/introspect_compositor_api.py
"""
import bpy

scene = bpy.context.scene

print("\n=== Scene compositor-related attributes ===")
for name in sorted(dir(scene)):
    if any(k in name.lower() for k in ("comp", "node")):
        try:
            val = getattr(scene, name)
            print(f"  scene.{name} = {type(val).__name__} :: {val!r}")
        except Exception as e:
            print(f"  scene.{name} <err: {e}>")

print("\n=== Try creating a CompositorNodeTree node group + add OutputFile ===")
ng = bpy.data.node_groups.new("DPX_introspect", "CompositorNodeTree")
print(f"NodeGroup type: {type(ng).__name__}, bl_idname={ng.bl_idname}")
print(f"NodeGroup attrs (compositor-related):")
for n in sorted(dir(ng)):
    if not n.startswith("_") and not callable(getattr(ng, n, None)):
        if any(k in n.lower() for k in ("path", "compos", "output")):
            print(f"  ng.{n}")

print(f"\n=== Adding CompositorNodeOutputFile ===")
out = ng.nodes.new("CompositorNodeOutputFile")
print(f"Node type: {type(out).__name__}")
print("\nAll non-private attrs on CompositorNodeOutputFile:")
for n in sorted(dir(out)):
    if not n.startswith("_"):
        v = getattr(out, n, None)
        if not callable(v):
            try:
                vr = repr(v)
                if len(vr) > 80:
                    vr = vr[:77] + "..."
                print(f"  {n} = {type(v).__name__} :: {vr}")
            except Exception as e:
                print(f"  {n} <err: {e}>")

print("\n=== file_output_items ===")
items = out.file_output_items
print(f"  type: {type(items).__name__}, len: {len(items)}")
print("  collection methods/attrs:")
for n in sorted(dir(items)):
    if not n.startswith("_"):
        print(f"    {n}")
if len(items) > 0:
    s = items[0]
    print(f"  items[0] type: {type(s).__name__}")
    print(f"  items[0] attrs:")
    for n in sorted(dir(s)):
        if not n.startswith("_") and not callable(getattr(s, n, None)):
            print(f"    {n} = {getattr(s, n, '?')!r}")
print(f"\n  out.inputs len: {len(out.inputs)}")
for i, inp in enumerate(out.inputs):
    print(f"    inputs[{i}]: name={inp.name!r}, type={inp.type!r}")

print("\n=== format attrs ===")
fmt = out.format
print(f"  format type: {type(fmt).__name__}")
for n in sorted(dir(fmt)):
    if not n.startswith("_") and not callable(getattr(fmt, n, None)):
        print(f"    {n}")

print("\n=== Setting compositing_node_group on scene ===")
try:
    scene.compositing_node_group = ng
    print(f"  scene.compositing_node_group set OK -> {scene.compositing_node_group!r}")
except AttributeError as e:
    print(f"  AttributeError: {e}")

print("\nDONE")
