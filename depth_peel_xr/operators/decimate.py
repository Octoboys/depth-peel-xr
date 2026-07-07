"""Convenience Decimate for every reconstruction half.

`Decimate All` adds a single Decimate modifier — named
`DECIMATE_MODIFIER_NAME` — to every `dpxr_recon_layer_*` object.
Two modes depending on suffix:

- `_opaque` halves get COLLAPSE at `scene.dpx_reconstruction.decimate_ratio`,
  wired to the `dpxr_decimate_area` vertex group so boundary verts
  (seam + any silhouette) stay pinned via weight-scaled collapse.
- `_alpha` halves get PLANAR DISSOLVE at
  `scene.dpx_reconstruction.decimate_angle_limit`. Alpha halves are
  usually one-quad-wide silhouette strips where every vert is
  topologically boundary — COLLAPSE stalls at 0% reduction. DISSOLVE
  merges near-coplanar adjacent border quads into n-gons without
  touching boundary edges (single-owner edges can't be dissolved), so
  seam + silhouette shape are preserved automatically.

Idempotent: re-running just updates the params on existing modifiers.

`Undo Decimate` removes only the modifiers this operator added
(matched by name) — user-added Decimate modifiers are left untouched.
"""

from __future__ import annotations

import bpy
from bpy.types import Operator

from .preview import _RECON_PREFIX
from .reconstruct import DECIMATE_AREA_GROUP, _ALPHA_SUFFIX, _OPAQUE_SUFFIX


# Well-known modifier name so `Undo Decimate` can target exactly what
# `Decimate All` added, without clobbering any Decimate the user
# stacked manually on the same object.
DECIMATE_MODIFIER_NAME = "dpxr_decimate"


def _recon_objects() -> list[bpy.types.Object]:
    return [
        obj for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name.startswith(_RECON_PREFIX)
    ]


def apply_ratio_to_existing_opaque_modifiers(ratio: float) -> None:
    """Live-update: push a new COLLAPSE ratio to every already-stamped
    Decimate modifier on `_opaque` halves. No-op if `Decimate All` hasn't
    been run yet, or if the modifier got its type swapped out manually."""
    for obj in _recon_objects():
        if not obj.name.endswith(_OPAQUE_SUFFIX):
            continue
        mod = obj.modifiers.get(DECIMATE_MODIFIER_NAME)
        if mod is not None and mod.decimate_type == "COLLAPSE":
            mod.ratio = ratio


def apply_angle_to_existing_alpha_modifiers(angle: float) -> None:
    """Live-update counterpart for `_alpha` halves' DISSOLVE angle_limit.
    Same no-op semantics."""
    for obj in _recon_objects():
        if not obj.name.endswith(_ALPHA_SUFFIX):
            continue
        mod = obj.modifiers.get(DECIMATE_MODIFIER_NAME)
        if mod is not None and mod.decimate_type == "DISSOLVE":
            mod.angle_limit = angle


class DEPTHPEELXR_OT_decimate_all(Operator):
    bl_idname = "depthpeelxr.decimate_all"
    bl_label = "Decimate Layers"
    bl_description = (
        "Add or update a Decimate modifier on every reconstruction half. "
        "Opaque halves use COLLAPSE + vertex-group pin (seam preserved); "
        "alpha halves use PLANAR DISSOLVE (silhouette shape preserved "
        "automatically via boundary-edge protection)."
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(
            obj.name.startswith(_RECON_PREFIX) and obj.type == "MESH"
            for obj in bpy.data.objects
        )

    def execute(self, context):
        recon = getattr(context.scene, "dpx_reconstruction", None)
        if recon is None:
            self.report({"ERROR"}, "Reconstruction properties missing")
            return {"CANCELLED"}
        ratio = float(recon.decimate_ratio)
        angle = float(recon.decimate_angle_limit)

        objs = _recon_objects()
        if not objs:
            self.report({"WARNING"}, "No reconstruction meshes found")
            return {"CANCELLED"}

        added = updated = 0
        for obj in objs:
            existing = obj.modifiers.get(DECIMATE_MODIFIER_NAME)
            if existing is None:
                mod = obj.modifiers.new(name=DECIMATE_MODIFIER_NAME, type="DECIMATE")
                added += 1
            else:
                mod = existing
                updated += 1

            if obj.name.endswith(_ALPHA_SUFFIX):
                # PLANAR DISSOLVE — boundary edges (seam + silhouette)
                # are single-owner and thus can't be dissolved, so shape
                # is preserved without needing the vertex group. Clear
                # any prior vertex_group binding for a clean state.
                mod.decimate_type = "DISSOLVE"
                mod.angle_limit = angle
                mod.delimit = set()
                mod.use_dissolve_boundaries = False
                mod.vertex_group = ""
                mod.invert_vertex_group = False
            else:
                # Default (opaque or any other suffix) → COLLAPSE +
                # weight-scaled seam pin via the build-time vertex group.
                mod.decimate_type = "COLLAPSE"
                mod.ratio = ratio
                mod.use_collapse_triangulate = True
                if DECIMATE_AREA_GROUP in obj.vertex_groups:
                    mod.vertex_group = DECIMATE_AREA_GROUP
                    mod.invert_vertex_group = False
                else:
                    # Half had no boundary (all-interior or all-border
                    # layer) — nothing to pin, so no group and the whole
                    # mesh decimates.
                    mod.vertex_group = ""
                    mod.invert_vertex_group = False

        self.report(
            {"INFO"},
            f"Decimate: opaque ratio={ratio:.2f}, alpha angle={angle:.3f} rad — "
            f"added {added}, updated {updated}",
        )
        return {"FINISHED"}


class DEPTHPEELXR_OT_undo_decimate(Operator):
    bl_idname = "depthpeelxr.undo_decimate"
    bl_label = "Undo Decimate"
    bl_description = (
        "Remove the Decimate modifier added by Decimate All from every "
        "reconstruction half. Leaves user-added modifiers untouched."
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(
            obj.type == "MESH"
            and obj.name.startswith(_RECON_PREFIX)
            and DECIMATE_MODIFIER_NAME in obj.modifiers
            for obj in bpy.data.objects
        )

    def execute(self, context):
        removed = 0
        for obj in _recon_objects():
            mod = obj.modifiers.get(DECIMATE_MODIFIER_NAME)
            if mod is not None:
                obj.modifiers.remove(mod)
                removed += 1
        self.report({"INFO"}, f"Removed Decimate from {removed} recon mesh(es)")
        return {"FINISHED"}


CLASSES = (
    DEPTHPEELXR_OT_decimate_all,
    DEPTHPEELXR_OT_undo_decimate,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
