"""Tests for core.geometry — frustum extents + super-frustum projection."""

from __future__ import annotations

import numpy as np
import pytest

from core.geometry import (
    centre_frustum_half_extents,
    inverse_canonical_projection,
    super_frustum_canonical_shape,
    super_frustum_half_extents,
    super_frustum_projection,
)


# Portal: 4 wide × 2 tall at headplane_offset = 2.
HALF_W_P = 2.0
HALF_H_P = 1.0
HO = 2.0


def _project(canonical_proj: np.ndarray, p_canonical: np.ndarray) -> np.ndarray:
    """Project a 3D point in canonical-cam space → NDC (x, y, z), drop w."""
    p4 = np.array([p_canonical[0], p_canonical[1], p_canonical[2], 1.0])
    clip = canonical_proj @ p4
    return clip[:3] / clip[3]


def _centre_to_canonical(canonical_from_centre: np.ndarray, p_centre: np.ndarray) -> np.ndarray:
    p4 = np.array([p_centre[0], p_centre[1], p_centre[2], 1.0])
    out = canonical_from_centre @ p4
    return out[:3] / out[3]


# ---- super_frustum_half_extents / centre_frustum_half_extents ---------------

def test_super_and_centre_match_at_portal():
    sw, sh = super_frustum_half_extents(HO, HALF_W_P, HALF_H_P, HO)
    cw, ch = centre_frustum_half_extents(HO, HALF_W_P, HALF_H_P, HO)
    assert (sw, sh) == pytest.approx((cw, ch))
    assert (sw, sh) == pytest.approx((HALF_W_P, HALF_H_P))


def test_super_grows_twice_as_fast_past_portal():
    # At d = 2·ho: super = 3·half_w_p, centre = 2·half_w_p.
    sw, _ = super_frustum_half_extents(2 * HO, HALF_W_P, HALF_H_P, HO)
    cw, _ = centre_frustum_half_extents(2 * HO, HALF_W_P, HALF_H_P, HO)
    assert sw == pytest.approx(3 * HALF_W_P)
    assert cw == pytest.approx(2 * HALF_W_P)


# ---- super_frustum_projection -----------------------------------------------

def test_portal_corner_projects_to_ndc_unit():
    """A portal-rectangle corner (centre-cam space) projects to NDC ±1."""
    cfc, proj = super_frustum_projection(HALF_W_P, HALF_H_P, HO)
    # +x +y top-right portal corner at z = -ho.
    p_centre = np.array([HALF_W_P, HALF_H_P, -HO])
    p_can = _centre_to_canonical(cfc, p_centre)
    ndc = _project(proj, p_can)
    assert ndc[0] == pytest.approx(1.0, abs=1e-9)
    assert ndc[1] == pytest.approx(1.0, abs=1e-9)
    # -x -y bottom-left, NDC -1, -1.
    p_centre = np.array([-HALF_W_P, -HALF_H_P, -HO])
    p_can = _centre_to_canonical(cfc, p_centre)
    ndc = _project(proj, p_can)
    assert ndc[0] == pytest.approx(-1.0, abs=1e-9)
    assert ndc[1] == pytest.approx(-1.0, abs=1e-9)


def test_super_frustum_corner_at_doubled_depth_projects_to_ndc_unit():
    """At d = 2·ho, the super-frustum corner sits at ±3·half_w_p and still
    lands on the canonical NDC boundary."""
    cfc, proj = super_frustum_projection(HALF_W_P, HALF_H_P, HO)
    p_centre = np.array([3 * HALF_W_P, 3 * HALF_H_P, -2 * HO])
    p_can = _centre_to_canonical(cfc, p_centre)
    ndc = _project(proj, p_can)
    assert ndc[0] == pytest.approx(1.0, abs=1e-9)
    assert ndc[1] == pytest.approx(1.0, abs=1e-9)


def test_centre_cam_axis_point_projects_to_ndc_origin():
    """Points on the centre cam's optical axis project to canonical NDC (0, 0)
    regardless of depth — the virtual apex sits on that axis."""
    cfc, proj = super_frustum_projection(HALF_W_P, HALF_H_P, HO)
    for d in (HO, 2 * HO, 5 * HO):
        p_centre = np.array([0.0, 0.0, -d])
        p_can = _centre_to_canonical(cfc, p_centre)
        ndc = _project(proj, p_can)
        assert ndc[0] == pytest.approx(0.0, abs=1e-9)
        assert ndc[1] == pytest.approx(0.0, abs=1e-9)


def test_canonical_near_plane_at_portal_depth():
    """The virtual cam's near plane lives at the portal — a point at the
    portal centre projects to NDC z = -1 (near clip)."""
    cfc, proj = super_frustum_projection(HALF_W_P, HALF_H_P, HO)
    p_centre = np.array([0.0, 0.0, -HO])
    p_can = _centre_to_canonical(cfc, p_centre)
    ndc = _project(proj, p_can)
    assert ndc[2] == pytest.approx(-1.0, abs=1e-9)


def test_super_frustum_projection_rejects_bad_inputs():
    with pytest.raises(ValueError):
        super_frustum_projection(HALF_W_P, HALF_H_P, 0.0)
    with pytest.raises(ValueError):
        super_frustum_projection(0.0, HALF_H_P, HO)
    with pytest.raises(ValueError):
        super_frustum_projection(HALF_W_P, -1.0, HO)


# ---- super_frustum_canonical_shape ------------------------------------------

def test_canonical_shape_equals_centre_at_portal_depth():
    w, h = super_frustum_canonical_shape(HO, 1024, 512, HALF_W_P, HALF_H_P, HO)
    assert (w, h) == (1024, 512)


def test_canonical_shape_scales_at_double_depth():
    # super/centre = 3/2 at d = 2·ho.
    w, h = super_frustum_canonical_shape(2 * HO, 1024, 512, HALF_W_P, HALF_H_P, HO)
    assert (w, h) == (1536, 768)


def test_canonical_shape_caps_below_double():
    # Ratio → 2 as d → ∞; nothing should ever exceed 2·centre.
    w, h = super_frustum_canonical_shape(1000 * HO, 1024, 512, HALF_W_P, HALF_H_P, HO)
    assert w <= 2048 and h <= 1024
    assert w > 2040 and h > 1020  # very close to 2x


# ---- inverse_canonical_projection -------------------------------------------

def test_inverse_canonical_round_trip_against_super_proj():
    """Canonical NDC → portal-local → centre-cam → canonical NDC must be
    identity for points the canonical can frame (d_can >= near)."""
    cfc, proj = super_frustum_projection(HALF_W_P, HALF_H_P, HO)
    centre_to_canonical = cfc  # cfc IS canonical_from_centre
    for u, v, d in [(0.0, 0.0, HO), (0.5, -0.3, 1.5 * HO), (-1.0, 1.0, 2 * HO)]:
        x, y, z = inverse_canonical_projection(u, v, d, HALF_W_P, HALF_H_P, HO)
        # Portal-local → centre-cam (centre cam at y=ho looking -Y; cam +X
        # = portal -X; cam +Y = portal +Z; cam +Z = portal +Y).
        cam_x = -x
        cam_y =  z
        cam_z =  y - HO  # portal-local y minus centre-cam height
        p_can = _centre_to_canonical(centre_to_canonical, np.array([cam_x, cam_y, cam_z]))
        ndc = _project(proj, p_can)
        assert ndc[0] == pytest.approx(u, abs=1e-9)
        assert ndc[1] == pytest.approx(v, abs=1e-9)


def test_inverse_canonical_near_plane_is_portal():
    """At d = ho/2 (canonical near), inverse projection lands on the portal
    plane (y_portal = 0); NDC ±1 reaches portal half-extents."""
    x, y, z = inverse_canonical_projection(1.0, 1.0, HO * 0.5, HALF_W_P, HALF_H_P, HO)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert x == pytest.approx(-HALF_W_P, abs=1e-9)
    assert z == pytest.approx(+HALF_H_P, abs=1e-9)


def test_inverse_canonical_apex_at_zero_depth():
    """At d = 0 the inverse should land on the apex (origin x, z; y = ho/2)
    regardless of NDC."""
    for u, v in [(0.0, 0.0), (1.0, -1.0), (-0.5, 0.7)]:
        x, y, z = inverse_canonical_projection(u, v, 0.0, HALF_W_P, HALF_H_P, HO)
        assert x == pytest.approx(0.0, abs=1e-9)
        assert y == pytest.approx(0.5 * HO, abs=1e-9)
        assert z == pytest.approx(0.0, abs=1e-9)


def test_inverse_canonical_vectorised():
    """Numpy inputs of matching shape produce same-shape (x, y, z) outputs."""
    u = np.array([[-1.0, 0.0, 1.0]])
    v = np.array([[1.0, 0.0, -1.0]])
    d = np.array([[HO, HO, HO]])
    x, y, z = inverse_canonical_projection(u, v, d, HALF_W_P, HALF_H_P, HO)
    assert x.shape == (1, 3) and y.shape == (1, 3) and z.shape == (1, 3)
    np.testing.assert_allclose(y, [[0.5 * HO - HO, 0.5 * HO - HO, 0.5 * HO - HO]])


def test_inverse_canonical_rejects_zero_ho():
    with pytest.raises(ValueError):
        inverse_canonical_projection(0.0, 0.0, 1.0, HALF_W_P, HALF_H_P, 0.0)
