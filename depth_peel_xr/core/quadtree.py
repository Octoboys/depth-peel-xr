"""Adaptive quadtree over a 2D coverage mask.

Phase 4 slice 1: silhouette-driven subdivision only (no depth-gradient
splits — those land in slice 2). Pixels on the canonical centre-NDC
image; quad sizes measured in pixels.

Split rule:
  - Quad fully covered → `interior` leaf.
  - Quad fully uncovered → no leaf (skipped; represents background).
  - Quad mixed AND size > min_quad_size → subdivide into 4.
  - Quad mixed AND size <= min_quad_size → `border` leaf.

Quads are square, anchored at integer pixel coords. Splits produce
roughly-equal halves; the right/bottom halves may be one pixel larger
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


def build_quadtree(
    coverage_mask: np.ndarray, min_quad_size: int = 8
) -> list[QuadtreeLeaf]:
    """Build the quadtree leaf list for a coverage mask.

    `coverage_mask` is a (H, W) bool array — True where the pixel is
    covered by any captured view, False otherwise. `min_quad_size` is
    the smallest square edge (in pixels) a border quad can have before
    we stop subdividing.
    """
    if coverage_mask.ndim != 2:
        raise ValueError(f"coverage_mask must be 2D, got shape {coverage_mask.shape}")
    if coverage_mask.dtype != np.bool_:
        coverage_mask = coverage_mask.astype(bool)
    if min_quad_size < 1:
        raise ValueError(f"min_quad_size must be >= 1, got {min_quad_size}")

    H, W = coverage_mask.shape
    if H == 0 or W == 0:
        return []

    leaves: list[QuadtreeLeaf] = []
    # Stack-based recursion: (x, y, w, h).
    stack: list[tuple[int, int, int, int]] = [(0, 0, W, H)]
    while stack:
        x, y, w, h = stack.pop()
        state = _coverage_state(coverage_mask, x, y, w, h)
        if state == "empty":
            continue
        if state == "full":
            leaves.append(QuadtreeLeaf(x, y, w, h, "interior"))
            continue
        # Mixed.
        if max(w, h) <= min_quad_size:
            leaves.append(QuadtreeLeaf(x, y, w, h, "border"))
            continue
        # Split into 4 children. Right/bottom take any odd-pixel remainder.
        w_left = w // 2
        h_top = h // 2
        w_right = w - w_left
        h_bot = h - h_top
        stack.append((x,           y,         w_left,  h_top))
        stack.append((x + w_left,  y,         w_right, h_top))
        stack.append((x,           y + h_top, w_left,  h_bot))
        stack.append((x + w_left,  y + h_top, w_right, h_bot))

    return leaves
