"""Project folder resolution.

All generated artefacts live under <blend_dir>/depth_peel_xr/<scene_name>/
with temp/ captures/ usd/ subfolders, overridable via PathsProps.
"""

from pathlib import Path

import bpy


def is_blend_saved() -> bool:
    return bool(bpy.data.filepath)


def derived_project_root(context) -> Path | None:
    """Default project root derived from the .blend location and scene name."""
    if not is_blend_saved():
        return None
    return Path(bpy.data.filepath).parent / "depth_peel_xr" / context.scene.name


def project_root(context) -> Path | None:
    """The active project root.

    Returns the user-configured `project_root` field if set, otherwise the
    default derived from the .blend location. None if the .blend is unsaved
    and no value is configured.
    """
    paths = context.scene.dpx_paths
    if paths.project_root:
        return Path(bpy.path.abspath(paths.project_root))
    return derived_project_root(context)


def temp_dir(context) -> Path | None:
    root = project_root(context)
    return root / "temp" if root else None


def captures_dir(context) -> Path | None:
    root = project_root(context)
    return root / "captures" if root else None


def usd_dir(context) -> Path | None:
    root = project_root(context)
    return root / "usd" if root else None
