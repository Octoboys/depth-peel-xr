"""Adaptive quadtree over a 2D coverage mask.

Phase 4 slice 1: silhouette-driven subdivision only (no depth-gradient
splits — those land in slice 2). Pixels on the canonical centre-NDC
image; quad sizes measured in pixels.

Top-level structure is a **tile grid** — the canonical is divided into
TxT square tiles before the 4-way subdivision runs within each. Without
this the leaves would inherit the canonical's aspect ratio (typically
non-square because portal aspect != 1:1), giving rectangular world quads.
Edge tiles where the canonical W or H isn't a multiple of `tile_size`
become rectangular border tiles — acceptable since they're at the
canonical periphery.

Split rule inside a tile:
  - Quad fully covered → `interior` leaf.
  - Quad fully uncovered → no leaf (skipped; represents background).
  - Quad mixed AND size > min_quad_size → subdivide into 4.
  - Quad mixed AND size <= min_quad_size → `border` leaf.

Tiles and splits are anchored at integer pixel coords. The 4-way splits
produce roughly-equal halves; right/bottom halves may be one pixel larger
when sizes don't divide evenly.

Pure numpy, no bpy — testable standalone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


LeafClass = Literal["interior", "border"]


@dataclass(frozen=True)
class QuadtreeLeaf:
    """Axis-aligned leaf on the canonical image grid.

    x, y: top-left pixel coords.
    w, h: size in pixels.
    cls: 'interior' (fully covered) or 'border' (straddles silhouette).
    """
    x: int
    y: int
    w: int
    h: int
    cls: LeafClass


def _coverage_state(mask: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    """Inspect a sub-region of the coverage mask.
    Returns 'full', 'empty', or 'mixed'."""
    region = mask[y : y + h, x : x + w]
    covered = region.sum()
    total = region.size
    if covered == 0:
        return "empty"
    if covered == total:
        return "full"
    return "mixed"


def _subdivide_tile(
    mask: np.ndarray, x: int, y: int, w: int, h: int, min_quad_size: int
) -> list[QuadtreeLeaf]:
    """Run the 4-way quadtree on one tile. Returns the leaves it produces."""
    leaves: list[QuadtreeLeaf] = []
    stack: list[tuple[int, int, int, int]] = [(x, y, w, h)]
    while stack:
        qx, qy, qw, qh = stack.pop()
        state = _coverage_state(mask, qx, qy, qw, qh)
        if state == "empty":
            continue
        if state == "full":
            leaves.append(QuadtreeLeaf(qx, qy, qw, qh, "interior"))
            continue
        if max(qw, qh) <= min_quad_size:
            leaves.append(QuadtreeLeaf(qx, qy, qw, qh, "border"))
            continue
        w_left = qw // 2
        h_top = qh // 2
        w_right = qw - w_left
        h_bot = qh - h_top
        stack.append((qx,          qy,         w_left,  h_top))
        stack.append((qx + w_left, qy,         w_right, h_top))
        stack.append((qx,          qy + h_top, w_left,  h_bot))
        stack.append((qx + w_left, qy + h_top, w_right, h_bot))
    return leaves


def build_quadtree(
    coverage_mask: np.ndarray,
    min_quad_size: int = 8,
    tile_size: int = 128,
) -> list[QuadtreeLeaf]:
    """Build the quadtree leaf list for a coverage mask.

    `coverage_mask` is a (H, W) bool array — True where the pixel is
    covered by any captured view, False otherwise. `min_quad_size` is
    the smallest square edge (in pixels) a border quad can have before
    we stop subdividing. `tile_size` is the size of the top-level
    square tiles that the canonical is divided into; should be a
    multiple of `min_quad_size` for clean subdivision.
    """
    if coverage_mask.ndim != 2:
        raise ValueError(f"coverage_mask must be 2D, got shape {coverage_mask.shape}")
    if coverage_mask.dtype != np.bool_:
        coverage_mask = coverage_mask.astype(bool)
    if min_quad_size < 1:
        raise ValueError(f"min_quad_size must be >= 1, got {min_quad_size}")
    if tile_size < min_quad_size:
        raise ValueError(
            f"tile_size ({tile_size}) must be >= min_quad_size ({min_quad_size})"
        )

    H, W = coverage_mask.shape
    if H == 0 or W == 0:
        return []

    leaves: list[QuadtreeLeaf] = []
    for ty in range(0, H, tile_size):
        for tx in range(0, W, tile_size):
            tw = min(tile_size, W - tx)
            th = min(tile_size, H - ty)
            leaves.extend(_subdivide_tile(coverage_mask, tx, ty, tw, th, min_quad_size))
    return leaves

