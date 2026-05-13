import bpy


class DEPTHPEELXR_PT_main(bpy.types.Panel):
    bl_label = "Depth Peel XR"
    bl_idname = "DEPTHPEELXR_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Depth Peel XR"

    def draw(self, context):
        self.layout.label(text="Hello from Depth Peel XR (Phase 0)")


classes = (DEPTHPEELXR_PT_main,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
