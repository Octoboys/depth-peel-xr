"""Tests for core.quadtree adaptive subdivision."""

from __future__ import annotations

import numpy as np
import pytest

from core.quadtree import QuadtreeLeaf, build_quadtree


def test_empty_mask_produces_no_leaves():
    mask = np.zeros((64, 64), dtype=bool)
    leaves = build_quadtree(mask, min_quad_size=4)
    assert leaves == []


def test_fully_covered_produces_single_interior_leaf_at_root_scale():
    mask = np.ones((64, 64), dtype=bool)
    leaves = build_quadtree(mask, min_quad_size=4)
    assert len(leaves) == 1
    assert leaves[0] == QuadtreeLeaf(0, 0, 64, 64, "interior")


def test_silhouette_subdivides_to_min_quad_size_at_border_only():
    # Left half covered, right half empty — vertical silhouette down the middle.
    mask = np.zeros((64, 64), dtype=bool)
    mask[:, :32] = True
    leaves = build_quadtree(mask, min_quad_size=4)

    interiors = [l for l in leaves if l.cls == "interior"]
    borders = [l for l in leaves if l.cls == "border"]

    # No border quads — silhouette falls on the 32-pixel split, so the
    # two left quadrants come out fully covered (each 32x32 interior).
    assert not borders
    assert len(interiors) == 2
    interior_regions = {(l.x, l.y, l.w, l.h) for l in interiors}
    assert interior_regions == {(0, 0, 32, 32), (0, 32, 32, 32)}


def test_off_grid_silhouette_produces_min_size_border_quads():
    # Diagonal silhouette guarantees the boundary doesn't align with any
    # power-of-two split — must produce border leaves at min_quad_size.
    H = W = 64
    yy, xx = np.indices((H, W))
    mask = yy < xx  # triangular coverage
    leaves = build_quadtree(mask, min_quad_size=4)

    borders = [l for l in leaves if l.cls == "border"]
    assert borders, "diagonal silhouette must produce border leaves"
    # All border leaves should be ≤ min_quad_size (mixed leaves don't
    # subdivide further once they hit the floor).
    for l in borders:
        assert max(l.w, l.h) <= 4


def test_all_leaves_within_image_bounds():
    H, W = 50, 70  # non-square, non-power-of-two
    yy, xx = np.indices((H, W))
    mask = (xx - 35) ** 2 + (yy - 25) ** 2 < 20 ** 2
    leaves = build_quadtree(mask, min_quad_size=4)

    for l in leaves:
        assert l.x >= 0
        assert l.y >= 0
        assert l.x + l.w <= W
        assert l.y + l.h <= H


def test_leaves_tile_their_class_regions_without_gaps_or_overlap():
    # The union of interior+border leaves should equal the coverage mask
    # (no covered pixel goes unaccounted for; no pixel is double-claimed).
    H = W = 32
    yy, xx = np.indices((H, W))
    mask = (xx > 5) & (xx < 25) & (yy > 8) & (yy < 22)
    leaves = build_quadtree(mask, min_quad_size=2)

    reconstructed = np.zeros((H, W), dtype=np.int32)
    for l in leaves:
        reconstructed[l.y : l.y + l.h, l.x : l.x + l.w] += 1

    # Every covered pixel covered exactly once; every non-covered pixel
    # may be covered 0 or 1 times (border quads include some non-covered
    # pixels, but interior quads do not extend into empty regions).
    assert (reconstructed[mask] >= 1).all()
    assert (reconstructed <= 1).all(), "leaves overlap"


def test_min_quad_size_floor_validated():
    with pytest.raises(ValueError):
        build_quadtree(np.ones((8, 8), dtype=bool), min_quad_size=0)


def test_rejects_non_2d_mask():
    with pytest.raises(ValueError):
        build_quadtree(np.ones((8, 8, 4), dtype=bool), min_quad_size=4)


def test_non_bool_mask_is_coerced():
    # Float coverage (e.g., alpha buffer) should still work after coercion.
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[:8, :8] = 1.0
    leaves = build_quadtree(mask, min_quad_size=2)
    interiors = [l for l in leaves if l.cls == "interior"]
    assert any(
        l.x == 0 and l.y == 0 and l.w == 8 and l.h == 8
        for l in interiors
    )
