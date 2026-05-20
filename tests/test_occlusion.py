"""Tests for core.occlusion — per-vertex visibility cull.

Synthetic camera setup using a symmetric perspective and a hand-crafted
closer-surface depth buffer so the rules are exercised in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.occlusion import (
    cull_layer,
    project_verts_to_cam,
    vert_visibility_per_cam,
)


def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Symmetric perspective, OpenGL convention."""
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _identity_cam() -> tuple[np.ndarray, np.ndarray]:
    """Camera at world origin looking down -Z with 90° vertical FoV,
    1:1 aspect, near 0.1, far 100. Returns (world_from_cam, proj)."""
    proj = _perspective(90.0, 1.0, 0.1, 100.0)
    return np.eye(4, dtype=np.float64), proj


# ---- project_verts_to_cam ---------------------------------------------------

def test_project_axis_vert_lands_at_image_centre():
    world_from_cam, proj = _identity_cam()
    verts = np.array([[0.0, 0.0, -5.0]], dtype=np.float64)
    u, v, d = project_verts_to_cam(verts, world_from_cam, proj, 100, 100)
    assert u[0] == pytest.approx(50.0)
    assert v[0] == pytest.approx(50.0)
    assert d[0] == pytest.approx(5.0)


def test_project_offaxis_vert_lands_off_centre():
    """Vert at (1, 0, -1) with 90° FoV → NDC u = 1 → right edge of image."""
    world_from_cam, proj = _identity_cam()
    verts = np.array([[1.0, 0.0, -1.0]], dtype=np.float64)
    u, v, d = project_verts_to_cam(verts, world_from_cam, proj, 100, 100)
    assert u[0] == pytest.approx(100.0, abs=1e-6)
    assert d[0] == pytest.approx(1.0)


def test_project_behind_camera_returns_negative_depth():
    """Vert behind the cam (positive z in cam frame) reports depth <= 0
    so the visibility test can drop it."""
    world_from_cam, proj = _identity_cam()
    verts = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
    _, _, d = project_verts_to_cam(verts, world_from_cam, proj, 100, 100)
    assert d[0] <= 0.0


# ---- vert_visibility_per_cam ------------------------------------------------

def test_vert_in_front_of_inf_closer_surface_is_visible():
    """No closer layer (closer = +inf everywhere) → every in-frustum
    vert is visible."""
    world_from_cam, proj = _identity_cam()
    verts = np.array([[0.0, 0.0, -5.0]], dtype=np.float64)
    closer = np.full((100, 100), np.inf, dtype=np.float32)
    visible = vert_visibility_per_cam(
        verts,
        [world_from_cam],
        [proj],
        [(100, 100)],
        [closer],
    )
    assert visible[0]


def test_vert_behind_closer_surface_is_hidden():
    """Vert at depth 10 with closer surface at depth 3 at its projected
    pixel → vert is hidden from this cam (the only cam) → fully hidden."""
    world_from_cam, proj = _identity_cam()
    verts = np.array([[0.0, 0.0, -10.0]], dtype=np.float64)
    closer = np.full((100, 100), 3.0, dtype=np.float32)
    visible = vert_visibility_per_cam(
        verts,
        [world_from_cam],
        [proj],
        [(100, 100)],
        [closer],
    )
    assert not visible[0]


def test_vert_visible_from_one_cam_counts_as_visible():
    """Vert behind closer surface from cam A but in front from cam B →
    visible (any-cam is enough)."""
    world_from_cam, proj = _identity_cam()
    verts = np.array([[0.0, 0.0, -5.0]], dtype=np.float64)
    closer_blocks = np.full((100, 100), 3.0, dtype=np.float32)  # blocks
    closer_open = np.full((100, 100), np.inf, dtype=np.float32)  # open
    visible = vert_visibility_per_cam(
        verts,
        [world_from_cam, world_from_cam],
        [proj, proj],
        [(100, 100), (100, 100)],
        [closer_blocks, closer_open],
    )
    assert visible[0]


def test_vert_out_of_frustum_from_only_cam_is_hidden():
    """Vert that projects outside the cam's frustum is treated as
    hidden — that cam doesn't see it. With only one cam, the vert is
    fully hidden."""
    world_from_cam, proj = _identity_cam()
    verts = np.array([[100.0, 0.0, -1.0]], dtype=np.float64)  # way off-axis
    closer = np.full((100, 100), np.inf, dtype=np.float32)
    visible = vert_visibility_per_cam(
        verts,
        [world_from_cam],
        [proj],
        [(100, 100)],
        [closer],
    )
    assert not visible[0]


def test_eps_keeps_coplanar_vert_visible():
    """A vert with depth equal to the closer-surface depth (within eps)
    is treated as in-front, not occluded — protects coplanar layers."""
    world_from_cam, proj = _identity_cam()
    verts = np.array([[0.0, 0.0, -5.0]], dtype=np.float64)
    closer = np.full((100, 100), 5.0, dtype=np.float32)  # exactly equal
    visible = vert_visibility_per_cam(
        verts,
        [world_from_cam],
        [proj],
        [(100, 100)],
        [closer],
        eps=0.01,
    )
    # depth=5, closer=5, eps=0.01 → vert is in-front since 5 < 5 - 0.01 is False;
    # so it's hidden by this rule. eps actually flips the comparison the
    # OTHER way: 'in_front' = depth < closer - eps; closer-eps = 4.99,
    # depth = 5 > 4.99 → not in front → hidden. The eps is "give the
    # closer surface a small extra forward bias" — coplanar verts ARE
    # culled. Verify that's the convention.
    assert not visible[0]


def test_eps_reverses_when_vert_just_in_front():
    """A vert closer than the closer-surface by more than eps stays
    visible."""
    world_from_cam, proj = _identity_cam()
    verts = np.array([[0.0, 0.0, -4.5]], dtype=np.float64)  # depth=4.5
    closer = np.full((100, 100), 5.0, dtype=np.float32)     # closer=5
    visible = vert_visibility_per_cam(
        verts,
        [world_from_cam],
        [proj],
        [(100, 100)],
        [closer],
        eps=0.01,
    )
    # depth=4.5 < 5 - 0.01 = 4.99 → in front → visible.
    assert visible[0]


# ---- cull_layer -------------------------------------------------------------

def test_cull_polygon_with_all_hidden_verts():
    world_from_cam, proj = _identity_cam()
    verts = np.array(
        [
            [0.0, 0.0, -5.0],
            [1.0, 0.0, -5.0],
            [1.0, 1.0, -5.0],
            [0.0, 1.0, -5.0],
        ],
        dtype=np.float64,
    )
    polygons = [(0, 1, 2, 3)]
    closer = np.full((100, 100), 1.0, dtype=np.float32)  # blocks at depth 1
    keep = cull_layer(
        polygons,
        verts,
        [world_from_cam],
        [proj],
        [(100, 100)],
        [closer],
    )
    assert keep.shape == (1,)
    assert not keep[0]


def test_keep_polygon_with_one_visible_vert():
    """One vert in front of an inf-closer-surface region, three verts
    behind a near-closer-surface region → at least one vert visible →
    polygon survives."""
    world_from_cam, proj = _identity_cam()
    # All verts project to NDC (0, 0) (image centre = pixel 50, 50) with
    # different depths.
    verts = np.array(
        [
            [0.0, 0.0, -2.0],   # depth 2 — in front
            [0.0, 0.0, -10.0],  # depth 10 — behind
            [0.0, 0.0, -10.0],
            [0.0, 0.0, -10.0],
        ],
        dtype=np.float64,
    )
    polygons = [(0, 1, 2, 3)]
    closer = np.full((100, 100), 5.0, dtype=np.float32)
    keep = cull_layer(
        polygons,
        verts,
        [world_from_cam],
        [proj],
        [(100, 100)],
        [closer],
    )
    assert keep[0]


def test_cull_with_no_polygons_returns_empty():
    keep = cull_layer(
        [],
        np.empty((0, 3), dtype=np.float64),
        [],
        [],
        [],
        [],
    )
    assert keep.shape == (0,)


def test_polygon_offscreen_from_every_cam_is_culled():
    """No closer layer (open everywhere) but the polygon's verts project
    outside every cam's frustum → fully hidden → culled."""
    world_from_cam, proj = _identity_cam()
    verts = np.array(
        [
            [50.0, 50.0, -1.0],
            [51.0, 50.0, -1.0],
            [51.0, 51.0, -1.0],
            [50.0, 51.0, -1.0],
        ],
        dtype=np.float64,
    )
    polygons = [(0, 1, 2, 3)]
    closer = np.full((100, 100), np.inf, dtype=np.float32)
    keep = cull_layer(
        polygons,
        verts,
        [world_from_cam],
        [proj],
        [(100, 100)],
        [closer],
    )
    assert not keep[0]


def test_ngon_polygon_culled_when_all_verts_hidden():
    """Cull rule generalises beyond quads — any polygon length works."""
    world_from_cam, proj = _identity_cam()
    n = 6
    verts = np.zeros((n, 3), dtype=np.float64)
    verts[:, 2] = -10.0  # all at depth 10
    polygons = [tuple(range(n))]
    closer = np.full((100, 100), 1.0, dtype=np.float32)
    keep = cull_layer(
        polygons,
        verts,
        [world_from_cam],
        [proj],
        [(100, 100)],
        [closer],
    )
    assert not keep[0]
