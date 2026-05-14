#!/usr/bin/env bash
# Symlink ./depth_peel_xr/ into Blender's VS Code extensions repo so the
# add-on loads cleanly and save-on-reload works with Jacques Lucke's
# "Blender Development" VS Code extension.
#
# Why `vscode_development/` and not `user_default/`:
#   The VS Code extension's save-reload handler hardcodes the module path
#   as `bl_ext.vscode_development.<manifest_id>`. If the add-on is
#   installed under any other extensions repo (e.g. user_default), the
#   initial load works but save-on-reload fails with
#   "No module named 'bl_ext.vscode_development.depth_peel_xr'".
#
# Prerequisite:
#   Run VS Code `Cmd+Shift+P → Blender: Start` **once** before using this
#   script — that's what registers the `vscode_development` extension repo
#   with Blender. After that, this script is safe to re-run any time
#   (it's idempotent) to reinstate the symlink if VS Code's own link-
#   creation step silently fails.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ADDON_SRC="$REPO_ROOT/depth_peel_xr"
ADDON_ID="depth_peel_xr"
REPO_NAME="vscode_development"

case "$(uname -s)" in
  Darwin) BLENDER_USER_ROOT="$HOME/Library/Application Support/Blender" ;;
  Linux)  BLENDER_USER_ROOT="${XDG_CONFIG_HOME:-$HOME/.config}/blender" ;;
  MINGW*|MSYS*|CYGWIN*) BLENDER_USER_ROOT="$APPDATA/Blender Foundation/Blender" ;;
  *) echo "Unsupported platform: $(uname -s)" >&2; exit 1 ;;
esac

if [[ ! -d "$BLENDER_USER_ROOT" ]]; then
  echo "No Blender user directory at: $BLENDER_USER_ROOT" >&2
  echo "Launch Blender at least once before re-running." >&2
  exit 1
fi

# Pick the highest-versioned user dir present (e.g. 5.0 over 4.5).
LATEST_VERSION="$(
  ls "$BLENDER_USER_ROOT" 2>/dev/null \
    | grep -E '^[0-9]+\.[0-9]+$' \
    | sort -V \
    | tail -1
)"

if [[ -z "$LATEST_VERSION" ]]; then
  echo "No versioned Blender directories under $BLENDER_USER_ROOT" >&2
  exit 1
fi

EXT_DIR="$BLENDER_USER_ROOT/$LATEST_VERSION/extensions/$REPO_NAME"
TARGET="$EXT_DIR/$ADDON_ID"
OLD_USER_DEFAULT="$BLENDER_USER_ROOT/$LATEST_VERSION/extensions/user_default/$ADDON_ID"

if [[ ! -d "$EXT_DIR" ]]; then
  echo "ERROR: $EXT_DIR does not exist yet." >&2
  echo "  Run VS Code 'Cmd+Shift+P → Blender: Start' once to register the" >&2
  echo "  vscode_development extensions repo, then re-run this script." >&2
  exit 1
fi

# Clean up a stale link in user_default pointing at the same source —
# having the add-on in two repos at once double-registers it in Blender.
if [[ -L "$OLD_USER_DEFAULT" ]]; then
  existing="$(readlink "$OLD_USER_DEFAULT")"
  if [[ "$existing" == "$ADDON_SRC" ]]; then
    echo "Removing duplicate link in user_default: $OLD_USER_DEFAULT"
    rm "$OLD_USER_DEFAULT"
  fi
fi

if [[ -L "$TARGET" ]]; then
  existing="$(readlink "$TARGET")"
  if [[ "$existing" == "$ADDON_SRC" ]]; then
    echo "Already linked: $TARGET -> $existing"
    echo "Blender version: $LATEST_VERSION"
    exit 0
  fi
  echo "Removing stale symlink: $TARGET -> $existing"
  rm "$TARGET"
elif [[ -e "$TARGET" ]]; then
  echo "ERROR: $TARGET exists and is not a symlink. Remove it manually first." >&2
  exit 1
fi

ln -s "$ADDON_SRC" "$TARGET"
echo "Linked: $TARGET -> $ADDON_SRC"
echo "Blender version: $LATEST_VERSION"
echo
echo "Next steps:"
echo "  1. In Blender, Edit > Preferences > Get Extensions — enable 'Depth Peel XR'."
echo "     (If an older 'bl_ext.user_default.depth_peel_xr' ghost entry appears broken, remove it.)"
echo "  2. Hamburger menu > Save Preferences — locks in auto-enable on future launches."
