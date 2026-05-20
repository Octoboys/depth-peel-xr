"""Adaptive quadtree over a 2D coverage mask.

Silhouette split rule:
  - Quad fully covered → `interior` leaf.
  - Quad fully uncovered → no leaf (skipped; represents background).
  - Quad mixed AND size > min_quad_size → subdivide into 4.
  - Quad mixed AND size <= min_quad_size → `border` leaf.

Optional depth-variance subdivision for `interior` leaves: when a
canonical depth array and tolerance are passed in, a fully-covered quad
whose `max - min` depth exceeds the tolerance subdivides further (down
to min_quad_size). Tile-size still caps the largest interior leaf in
low-variance regions; the variance check only refines from there. When
`depth` or `depth_tol` is None, only the silhouette split applies.

Top-level structure is a **tile grid** — the canonical is divided into
TxT square tiles before the 4-way subdivision runs within each. Without
this the leaves would inherit the canonical's aspect ratio (typically
non-square because portal aspect != 1:1), giving rectangular world quads.
Edge tiles where the canonical W or H isn't a multiple of `tile_size`
become rectangular border tiles — acceptable since they're at the
canonical periphery.

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


def _depth_within_tol(
    depth: np.ndarray, x: int, y: int, w: int, h: int, tol: float
) -> bool:
    """True if max-min depth across the region is <= tol. Region is assumed
    fully covered (caller checks coverage before invoking)."""
    region = depth[y : y + h, x : x + w]
    return float(region.max() - region.min()) <= tol


def _subdivide_tile(
    mask: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    min_quad_size: int,
    depth: np.ndarray | None,
    depth_tol: float | None,
) -> list[QuadtreeLeaf]:
    """Run the 4-way quadtree on one tile. Returns the leaves it produces."""
    leaves: list[QuadtreeLeaf] = []
    stack: list[tuple[int, int, int, int]] = [(x, y, w, h)]
    use_variance = depth is not None and depth_tol is not None
    while stack:
        qx, qy, qw, qh = stack.pop()
        state = _coverage_state(mask, qx, qy, qw, qh)
        if state == "empty":
            continue
        at_floor = max(qw, qh) <= min_quad_size
        if at_floor:
            cls = "interior" if state == "full" else "border"
            leaves.append(QuadtreeLeaf(qx, qy, qw, qh, cls))
            continue
        if state == "full":
            # Without depth-variance subdivision, fully-covered quads
            # become leaves here. With it, subdivide further when depth
            # variance across the region exceeds tolerance.
            if not use_variance or _depth_within_tol(depth, qx, qy, qw, qh, depth_tol):
                leaves.append(QuadtreeLeaf(qx, qy, qw, qh, "interior"))
                continue
        # state == "mixed" or (state == "full" with high variance) → subdivide.
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
    depth: np.ndarray | None = None,
    depth_tol: float | None = None,
) -> list[QuadtreeLeaf]:
    """Build the quadtree leaf list for a coverage mask.

    `coverage_mask` is a (H, W) bool array — True where the pixel is
    covered by any captured view, False otherwise. `min_quad_size` is
    the smallest square edge (in pixels) a quad can have before we stop
    subdividing. `tile_size` is the size of the top-level square tiles
    that the canonical is divided into; should be a multiple of
    `min_quad_size` for clean subdivision.

    Variance subdivision: if both `depth` (a (H, W) float array aligned
    with `coverage_mask`) and `depth_tol` are passed, fully-covered
    quads whose `max - min` depth exceeds `depth_tol` subdivide further
    until variance is within tolerance or `min_quad_size` is reached.
    Pass either as None to disable variance refinement (silhouette-only
    splits).
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
    if depth is not None:
        if depth.shape != coverage_mask.shape:
            raise ValueError(
                f"depth shape {depth.shape} must match coverage_mask {coverage_mask.shape}"
            )
        if depth_tol is not None and depth_tol < 0.0:
            raise ValueError(f"depth_tol must be >= 0, got {depth_tol}")

    H, W = coverage_mask.shape
    if H == 0 or W == 0:
        return []

    leaves: list[QuadtreeLeaf] = []
    for ty in range(0, H, tile_size):
        for tx in range(0, W, tile_size):
            tw = min(tile_size, W - tx)
            th = min(tile_size, H - ty)
            leaves.extend(
                _subdivide_tile(
                    coverage_mask, tx, ty, tw, th, min_quad_size, depth, depth_tol
                )
            )
    return leaves

