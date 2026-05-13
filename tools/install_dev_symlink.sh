#!/usr/bin/env bash
# Symlink ./depth_peel_xr/ into Blender's user-extensions directory so the
# add-on loads when Blender is launched normally (outside VS Code).
#
# Idempotent: re-running is safe.
#
# Note: if you also use VS Code + Jacques Lucke's "Blender Development"
# extension, that creates its own separate symlink under a
# `vscode_development` extensions repo. Having both present at once means
# Blender registers the add-on twice under different ids. Pick one path
# per Blender session — either run this script and launch Blender
# directly, or use the VS Code `Blender: Start` flow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ADDON_SRC="$REPO_ROOT/depth_peel_xr"
ADDON_ID="depth_peel_xr"

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

EXT_DIR="$BLENDER_USER_ROOT/$LATEST_VERSION/extensions/user_default"
TARGET="$EXT_DIR/$ADDON_ID"

mkdir -p "$EXT_DIR"

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
echo "  - Launch Blender and enable 'Depth Peel XR' under Preferences > Get Extensions > User."
