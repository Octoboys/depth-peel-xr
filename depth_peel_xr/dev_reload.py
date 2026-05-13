"""Manual hot-reload fallback for depth_peel_xr.

Primary dev loop is VS Code + Jacques Lucke's "Blender Development"
extension, which reloads the add-on on save. This file is the escape
hatch for running Blender outside VS Code: open it in Blender's Text
Editor and click Run Script (or bind to a custom keymap).

It unregisters the add-on, purges its module tree from sys.modules, and
re-registers from disk — picking up any edits made since the last load.
"""

from __future__ import annotations

import importlib
import sys

# Both install paths — tools/install_dev_symlink.sh and Jacques Lucke's
# VS Code extension — mount the add-on under this name, because the repo
# folder name (depth_peel_xr) matches the manifest id.
_ROOT = "bl_ext.user_default.depth_peel_xr"
_VSCODE_ROOT = "bl_ext.vscode_development.depth_peel_xr"


def reload() -> None:
    reloaded: list[str] = []

    for root in (_ROOT, _VSCODE_ROOT):
        if root not in sys.modules:
            continue

        mod = sys.modules[root]
        if hasattr(mod, "unregister"):
            try:
                mod.unregister()
            except Exception as e:
                print(f"[dev_reload] unregister({root}) raised: {e}")

        for name in [n for n in sys.modules if n == root or n.startswith(root + ".")]:
            del sys.modules[name]

        mod = importlib.import_module(root)
        if hasattr(mod, "register"):
            mod.register()
        reloaded.append(root)

    if reloaded:
        print(f"[dev_reload] reloaded: {', '.join(reloaded)}")
    else:
        print("[dev_reload] no live install found; enable the add-on in Preferences first")


if __name__ == "__main__":
    reload()
