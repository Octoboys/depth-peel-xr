"""Tests for core.quadtree adaptive subdivision."""

from __future__ import annotations

import numpy as np
import pytest

from core.quadtree import QuadtreeLeaf, build_quadtree


def test_empty_mask_produces_no_leaves():
    mask = np.zeros((64, 64), dtype=bool)
    leaves = build_quadtree(mask, min_quad_size=4)
    assert leaves == []


def test_fully_covered_produces_tile_grid_of_interior_leaves():
    """Top-level tile loop divides the canonical into TxT squares before
    the 4-way subdivision runs. A fully-covered 64x64 mask with
    tile_size=32 should produce a 2x2 grid of 32x32 interior leaves."""
    mask = np.ones((64, 64), dtype=bool)
    leaves = build_quadtree(mask, min_quad_size=4, tile_size=32)
    assert len(leaves) == 4
    interior_regions = {(l.x, l.y, l.w, l.h) for l in leaves if l.cls == "interior"}
    assert interior_regions == {
        (0, 0, 32, 32), (32, 0, 32, 32),
        (0, 32, 32, 32), (32, 32, 32, 32),
    }


def test_tile_size_equal_to_image_size_acts_as_single_root():
    """When tile_size >= image dimensions, the whole image is one tile."""
    mask = np.ones((64, 64), dtype=bool)
    leaves = build_quadtree(mask, min_quad_size=4, tile_size=128)
    assert len(leaves) == 1
    assert leaves[0] == QuadtreeLeaf(0, 0, 64, 64, "interior")


def test_silhouette_subdivides_to_min_quad_size_at_border_only():
    # Left half covered, right half empty — vertical silhouette down the middle.
    # tile_size = 64 = image width → single root; behaves like the
    # pre-tile-grid implementation.
    mask = np.zeros((64, 64), dtype=bool)
    mask[:, :32] = True
    leaves = build_quadtree(mask, min_quad_size=4, tile_size=64)

    interiors = [l for l in leaves if l.cls == "interior"]
    borders = [l for l in leaves if l.cls == "border"]

    # No border quads — silhouette falls on the 32-pixel split, so the
    # two left quadrants come out fully covered (each 32x32 interior).
    assert not borders
    assert len(interiors) == 2
    interior_regions = {(l.x, l.y, l.w, l.h) for l in interiors}
    assert interior_regions == {(0, 0, 32, 32), (0, 32, 32, 32)}


def test_edge_tiles_when_image_not_multiple_of_tile_size():
    """Bottom/right edge tiles get the remainder when W or H isn't a
    multiple of tile_size — rectangular at the periphery."""
    mask = np.ones((50, 70), dtype=bool)  # not a multiple of 32
    leaves = build_quadtree(mask, min_quad_size=4, tile_size=32)
    # Tiles: x=[0,32,64], y=[0,32]. Widths 32, 32, 6; heights 32, 18.
    expected = {
        (0, 0, 32, 32), (32, 0, 32, 32), (64, 0, 6, 32),
        (0, 32, 32, 18), (32, 32, 32, 18), (64, 32, 6, 18),
    }
    assert {(l.x, l.y, l.w, l.h) for l in leaves} == expected


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


def test_tile_size_below_min_quad_size_rejected():
    with pytest.raises(ValueError):
        build_quadtree(np.ones((8, 8), dtype=bool), min_quad_size=8, tile_size=4)


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


# ---- depth-variance subdivision (slice 2) -----------------------------------

def test_constant_depth_behaves_like_slice_1():
    """Fully-covered region with zero depth variance should produce the
    same single interior leaf as slice 1, regardless of tol."""
    mask = np.ones((64, 64), dtype=bool)
    depth = np.full((64, 64), 5.0, dtype=np.float32)
    leaves = build_quadtree(
        mask, min_quad_size=4, tile_size=64, depth=depth, depth_tol=0.001
    )
    assert len(leaves) == 1
    assert leaves[0] == QuadtreeLeaf(0, 0, 64, 64, "interior")


def test_high_variance_subdivides_to_min_size():
    """Linear depth ramp with tol smaller than the per-leaf range should
    subdivide all the way down to min_quad_size."""
    H = W = 32
    mask = np.ones((H, W), dtype=bool)
    # Depth ranges 0..1 across the canonical — at min_quad_size=4 the per-
    # leaf range is 4/32 = 0.125, so tol = 0.05 forces subdivision past
    # that level entirely (every leaf at min size).
    depth = np.tile(np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1))
    leaves = build_quadtree(
        mask, min_quad_size=4, tile_size=W, depth=depth, depth_tol=0.05
    )
    # Floor leaves are 4-px squares → expect (32/4)^2 = 64 of them.
    assert len(leaves) == 64
    assert all(l.cls == "interior" and l.w == 4 and l.h == 4 for l in leaves)


def test_tol_threshold_controls_subdivision_density():
    """Increasing tol reduces leaf count monotonically on the same data."""
    H = W = 32
    mask = np.ones((H, W), dtype=bool)
    depth = np.tile(np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1))
    counts = []
    for tol in (0.05, 0.2, 0.6, 2.0):
        leaves = build_quadtree(
            mask, min_quad_size=4, tile_size=W, depth=depth, depth_tol=tol
        )
        counts.append(len(leaves))
    # Strictly non-increasing as tol grows.
    for a, b in zip(counts, counts[1:]):
        assert a >= b
    # tol = 2.0 well above ramp range (= 1.0) → single coarse leaf.
    assert counts[-1] == 1


def test_variance_only_applies_to_full_quads():
    """Mixed (silhouette) quads still subdivide on coverage rules even when
    depth tol could in principle be satisfied — variance subdivision is
    additive, not a replacement."""
    H = W = 16
    yy, xx = np.indices((H, W))
    mask = (xx + yy) < 16  # diagonal silhouette
    depth = np.zeros((H, W), dtype=np.float32)
    leaves = build_quadtree(
        mask, min_quad_size=2, tile_size=W, depth=depth, depth_tol=10.0
    )
    # With huge tol, full quads stay coarse; mixed still subdivide → border
    # leaves at min size.
    assert any(l.cls == "border" for l in leaves)
    assert any(l.cls == "interior" for l in leaves)


def test_min_size_floor_caps_variance_subdivision():
    """Even with tiny tol, a leaf at min_quad_size stops subdividing —
    becomes interior with whatever variance is left over."""
    mask = np.ones((4, 4), dtype=bool)
    depth = np.array(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        dtype=np.float32,
    )
    leaves = build_quadtree(
        mask, min_quad_size=4, tile_size=4, depth=depth, depth_tol=0.01
    )
    # 4x4 IS min_quad_size → emit one interior leaf at floor regardless of
    # variance.
    assert len(leaves) == 1
    assert leaves[0].cls == "interior"


def test_depth_tol_negative_rejected():
    with pytest.raises(ValueError):
        build_quadtree(
            np.ones((8, 8), dtype=bool),
            min_quad_size=4,
            depth=np.zeros((8, 8), dtype=np.float32),
            depth_tol=-0.1,
        )


def test_depth_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        build_quadtree(
            np.ones((8, 8), dtype=bool),
            min_quad_size=4,
            depth=np.zeros((4, 4), dtype=np.float32),
            depth_tol=0.1,
        )


def test_depth_without_tol_falls_back_to_slice_1():
    """Passing depth alone (no tol) must not change slice-1 behaviour."""
    mask = np.ones((32, 32), dtype=bool)
    depth = np.tile(np.linspace(0.0, 100.0, 32, dtype=np.float32), (32, 1))
    leaves_no_variance = build_quadtree(mask, min_quad_size=4, tile_size=32)
    leaves_depth_only = build_quadtree(
        mask, min_quad_size=4, tile_size=32, depth=depth, depth_tol=None
    )
    assert leaves_no_variance == leaves_depth_only
