#!/usr/bin/env bash
# Run pytest over tests/ and enforce the core/ bpy-free contract.
#
# - `depth_peel_xr/core/` must not import `bpy` under any circumstances —
#   that's the module boundary that keeps math logic testable without Blender.
# - Tests live in `tests/` (at the repo root, not inside the addon package,
#   because depth_peel_xr/__init__.py imports bpy and would break pytest
#   collection).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Prefer the repo's dev venv so behaviour is reproducible across shells.
if [[ -x ".venv/bin/pytest" ]]; then
  PYTEST=".venv/bin/pytest"
else
  PYTEST="pytest"
fi

# Enforce: nothing under depth_peel_xr/core/ is allowed to import bpy.
if [[ -d "depth_peel_xr/core" ]]; then
  if grep -rnE '^[[:space:]]*(import[[:space:]]+bpy|from[[:space:]]+bpy)' depth_peel_xr/core/; then
    echo "ERROR: depth_peel_xr/core/ must not import bpy — keep it pure numpy/pxr." >&2
    exit 1
  fi
fi

exec "$PYTEST" -q tests/
