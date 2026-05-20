"""Tests for core.raster — software depth rasteriser.

Synthetic camera + simple triangle setups verify projection, perspective-
correct depth interpolation, near-plane handling, and tri winding
independence (CW and CCW screen-space tris both fill).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.raster import fan_triangulate_polygons, rasterise_depth


def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _identity_cam() -> tuple[np.ndarray, np.ndarray]:
    """Cam at origin looking down -Z, 90° vertical FoV, 1:1 aspect."""
    return np.eye(4, dtype=np.float64), _perspective(90.0, 1.0, 0.1, 100.0)


# ---- fan_triangulate_polygons ----------------------------------------------

def test_fan_triangulate_quad_emits_two_tris():
    tris = fan_triangulate_polygons([(0, 1, 2, 3)])
    assert tris.shape == (2, 3)
    assert tuple(tris[0]) == (0, 1, 2)
    assert tuple(tris[1]) == (0, 2, 3)


def test_fan_triangulate_ngon_emits_n_minus_2_tris():
    tris = fan_triangulate_polygons([(10, 11, 12, 13, 14)])
    assert tris.shape == (3, 3)


def test_fan_triangulate_skips_under_3_verts():
    tris = fan_triangulate_polygons([(0, 1)])
    assert tris.shape == (0, 3)


def test_fan_triangulate_empty_input():
    tris = fan_triangulate_polygons([])
    assert tris.shape == (0, 3)


# ---- rasterise_depth -------------------------------------------------------

def test_rasterise_empty_mesh_returns_inf_buffer():
    world_from_cam, proj = _identity_cam()
    buf = rasterise_depth(
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.int64),
        world_from_cam,
        proj,
        16, 16,
    )
    assert buf.shape == (16, 16)
    assert np.isinf(buf).all()


def test_rasterise_centred_quad_at_constant_depth():
    """Quad at z=-5, spanning NDC x ∈ [-0.4, 0.4], y ∈ [-0.4, 0.4] →
    pixels in [10, 90] × [10, 90] (on a 100×100 image with 90° FoV)
    should hold depth 5; rest +inf."""
    world_from_cam, proj = _identity_cam()
    verts = np.array(
        [
            [-2.0, -2.0, -5.0],
            [+2.0, -2.0, -5.0],
            [+2.0, +2.0, -5.0],
            [-2.0, +2.0, -5.0],
        ],
        dtype=np.float64,
    )
    tris = fan_triangulate_polygons([(0, 1, 2, 3)])
    buf = rasterise_depth(verts, tris, world_from_cam, proj, 100, 100)
    # Centre pixel covered, depth ≈ 5.
    assert buf[50, 50] == pytest.approx(5.0, abs=1e-3)
    # Far corner (1, 1) — way outside the quad's projected bbox — stays +inf.
    assert np.isinf(buf[1, 1])
    # Quad covers the full image (verts at ±2, depth 5 → NDC ±2/5/aspect_term =
    # ±0.4 with the 90° FoV — wait, 90° FoV means f=1, so ndc = x/(-z) = 2/5 = 0.4).
    # Pixel coverage = roughly central 80% of image. Pixel (10, 10) should be
    # outside; pixel (50, 50) is inside.
    assert np.isinf(buf[5, 5])


def test_rasterise_full_screen_quad_covers_all_pixels():
    """Quad at z=-1 with verts ±1 → NDC ±1 → covers full image."""
    world_from_cam, proj = _identity_cam()
    verts = np.array(
        [
            [-1.0, -1.0, -1.0],
            [+1.0, -1.0, -1.0],
            [+1.0, +1.0, -1.0],
            [-1.0, +1.0, -1.0],
        ],
        dtype=np.float64,
    )
    tris = fan_triangulate_polygons([(0, 1, 2, 3)])
    buf = rasterise_depth(verts, tris, world_from_cam, proj, 50, 50)
    assert np.all(np.isfinite(buf))
    # Constant-depth quad → every pixel == 1.
    np.testing.assert_allclose(buf, 1.0, atol=1e-3)


def test_rasterise_perspective_correct_depth_on_tilted_quad():
    """Tilted quad: front edge at z=-2, back edge at z=-10. Depth
    interpolation across the image should be hyperbolic in screen
    space, NOT linear. At the screen midpoint (y=mid), the depth
    should NOT equal the arithmetic mean (6); it should be the
    harmonic mean weighted by 1/z, which is closer to 2 than 10."""
    world_from_cam, proj = _identity_cam()
    # Bottom edge (y=-1) at z=-2 (close); top edge (y=+1) at z=-10 (far).
    # X spans ±5 so the projected quad covers the full image at all depths.
    verts = np.array(
        [
            [-5.0, -1.0, -2.0],   # bottom-left, near
            [+5.0, -1.0, -2.0],   # bottom-right, near
            [+50.0, +5.0, -10.0], # top-right, far (widened so projection still covers)
            [-50.0, +5.0, -10.0], # top-left, far
        ],
        dtype=np.float64,
    )
    tris = fan_triangulate_polygons([(0, 1, 2, 3)])
    buf = rasterise_depth(verts, tris, world_from_cam, proj, 100, 100)
    # Mid-Y pixel: perspective-correct depth ≠ arithmetic mean (which would be 6).
    mid_depth = buf[50, 50]
    assert np.isfinite(mid_depth)
    arithmetic_mean = (2.0 + 10.0) / 2.0
    # Perspective-correct depth at v=0 (screen mid) → 1/d = (1/2 + 1/10)/2 = 0.3 → d ≈ 3.33.
    assert mid_depth < arithmetic_mean - 0.5  # comfortably below 6
    assert mid_depth == pytest.approx(10.0 / 3.0, rel=0.05)


def test_rasterise_z_buffer_keeps_nearest():
    """Two overlapping quads at different depths — the nearer should win."""
    world_from_cam, proj = _identity_cam()
    near_quad = np.array(
        [
            [-1.0, -1.0, -2.0],
            [+1.0, -1.0, -2.0],
            [+1.0, +1.0, -2.0],
            [-1.0, +1.0, -2.0],
        ],
        dtype=np.float64,
    )
    far_quad = np.array(
        [
            [-2.0, -2.0, -5.0],
            [+2.0, -2.0, -5.0],
            [+2.0, +2.0, -5.0],
            [-2.0, +2.0, -5.0],
        ],
        dtype=np.float64,
    )
    verts = np.concatenate([near_quad, far_quad], axis=0)
    tris = fan_triangulate_polygons([(0, 1, 2, 3), (4, 5, 6, 7)])
    buf = rasterise_depth(verts, tris, world_from_cam, proj, 50, 50)
    # Centre — both cover. Near wins.
    assert buf[25, 25] == pytest.approx(2.0, abs=1e-3)


def test_rasterise_skips_tri_behind_camera():
    """Tri entirely behind cam (z > 0) contributes no pixels."""
    world_from_cam, proj = _identity_cam()
    verts = np.array(
        [
            [-1.0, -1.0, +1.0],
            [+1.0, -1.0, +1.0],
            [+1.0, +1.0, +1.0],
        ],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2]], dtype=np.int64)
    buf = rasterise_depth(verts, tris, world_from_cam, proj, 16, 16)
    assert np.isinf(buf).all()


def test_rasterise_handles_both_winding_orders():
    """Same triangle described CCW and CW (screen-space) should both
    fill — bary sign cancels with area2 sign."""
    world_from_cam, proj = _identity_cam()
    verts = np.array(
        [
            [-1.0, -1.0, -2.0],
            [+1.0, -1.0, -2.0],
            [0.0,  +1.0, -2.0],
        ],
        dtype=np.float64,
    )
    tris_ccw = np.array([[0, 1, 2]], dtype=np.int64)
    tris_cw = np.array([[0, 2, 1]], dtype=np.int64)
    buf_ccw = rasterise_depth(verts, tris_ccw, world_from_cam, proj, 50, 50)
    buf_cw = rasterise_depth(verts, tris_cw, world_from_cam, proj, 50, 50)
    np.testing.assert_allclose(buf_ccw, buf_cw, equal_nan=True)
    # Centre pixel of the triangle covered with depth 2.
    assert np.isfinite(buf_ccw[25, 25])
    assert buf_ccw[25, 25] == pytest.approx(2.0, abs=1e-3)
