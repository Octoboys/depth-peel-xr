"""Tests for core.multi_view reprojection + consolidation math.

Synthetic two-camera setup with hand-crafted projection matrices, so
results are deterministic and don't depend on Blender's API.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.multi_view import (
    COVERAGE_ALPHA_THRESHOLD,
    consolidate_layer,
    reproject_view_to_centre,
)


def _perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Symmetric perspective projection matrix, OpenGL convention
    (camera looks down -Z; clip space x,y,z ∈ [-1, 1])."""
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _translation(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def _identity_proj_and_world() -> tuple[np.ndarray, np.ndarray]:
    proj = _perspective(fovy_deg=60.0, aspect=1.0, near=0.1, far=100.0)
    world_from_cam = np.eye(4, dtype=np.float64)
    return proj, world_from_cam


def test_unproject_then_project_round_trips_centre_pixel():
    """A pixel rendered from centre, reprojected into centre, lands in
    the same pixel."""
    proj, world_from_cam = _identity_proj_and_world()
    H, W = 64, 64

    # Single covered pixel in the middle of the image, depth 5 m.
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[32, 32] = [0.5, 0.7, 0.9, 1.0]
    depth = np.zeros((H, W), dtype=np.float32)
    depth[32, 32] = 5.0

    contrib_rgba, contrib_depth, contrib_valid = reproject_view_to_centre(
        sibling_rgba=rgba,
        sibling_depth=depth,
        sib_world_from_cam=world_from_cam,
        sib_proj=proj,
        centre_cam_from_world=np.linalg.inv(world_from_cam),
        centre_proj=proj,
        centre_shape=(H, W),
    )

    # Should reproject exactly to (32, 32).
    assert contrib_valid[32, 32]
    assert contrib_depth[32, 32] == pytest.approx(5.0, abs=1e-4)
    np.testing.assert_allclose(contrib_rgba[32, 32], [0.5, 0.7, 0.9, 1.0], atol=1e-5)
    # Nowhere else.
    contrib_valid[32, 32] = False
    assert not contrib_valid.any()


def test_sibling_offset_shifts_reprojected_pixel():
    """A sibling camera translated +X by 1 m looking at a point at depth
    5 m sees that point shifted -X in its image. Reprojecting into the
    centre puts it back to the centre's image-centre pixel."""
    proj, _ = _identity_proj_and_world()
    H, W = 64, 64

    centre_world_from_cam = np.eye(4, dtype=np.float64)
    sib_world_from_cam = _translation(1.0, 0.0, 0.0)

    # World point at (0, 0, -5) — straight ahead of centre at depth 5.
    # In sibling cam space (translated +1 in x), the point is at (-1, 0, -5).
    # Sibling NDC: x_ndc = proj * (-1, 0, -5, 1)
    f_aspect = proj[0, 0]
    f = proj[1, 1]
    # clip_x = f_aspect * -1, clip_w = 5 ⇒ ndc_x = -f_aspect / 5
    expected_sib_ndc_x = -f_aspect / 5.0
    expected_sib_pixel_u = int(round((expected_sib_ndc_x + 1.0) * 0.5 * W - 0.5))

    rgba = np.zeros((H, W, 4), dtype=np.float32)
    depth = np.zeros((H, W), dtype=np.float32)
    rgba[32, expected_sib_pixel_u] = [1.0, 0.0, 0.0, 1.0]
    depth[32, expected_sib_pixel_u] = 5.0

    contrib_rgba, contrib_depth, contrib_valid = reproject_view_to_centre(
        sibling_rgba=rgba,
        sibling_depth=depth,
        sib_world_from_cam=sib_world_from_cam,
        sib_proj=proj,
        centre_cam_from_world=np.linalg.inv(centre_world_from_cam),
        centre_proj=proj,
        centre_shape=(H, W),
    )

    # Should reproject back to centre image (around pixel 32,32 ± 1 due to rounding).
    where = np.argwhere(contrib_valid)
    assert where.shape[0] == 1
    v, u = where[0]
    assert v == 32
    assert abs(u - 32) <= 1
    assert contrib_depth[v, u] == pytest.approx(5.0, abs=1e-3)


def test_invalid_sibling_pixels_dont_splat():
    """Pixels with alpha below threshold or non-finite depth contribute
    nothing."""
    proj, world_from_cam = _identity_proj_and_world()
    H, W = 32, 32

    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[16, 16] = [1.0, 1.0, 1.0, 0.0]  # below threshold
    rgba[10, 10] = [1.0, 1.0, 1.0, 1.0]  # valid alpha but ...
    depth = np.zeros((H, W), dtype=np.float32)
    depth[16, 16] = 5.0
    depth[10, 10] = np.inf  # non-finite depth

    _, _, valid = reproject_view_to_centre(
        sibling_rgba=rgba,
        sibling_depth=depth,
        sib_world_from_cam=world_from_cam,
        sib_proj=proj,
        centre_cam_from_world=np.linalg.inv(world_from_cam),
        centre_proj=proj,
        centre_shape=(H, W),
    )
    assert not valid.any()


def test_consolidate_preserves_centre_coverage():
    """Centre pixels with coverage win; siblings only fill empty centre
    pixels."""
    proj, world_from_cam = _identity_proj_and_world()
    H, W = 32, 32

    # Centre covers half the image (left half).
    centre_rgba = np.zeros((H, W, 4), dtype=np.float32)
    centre_rgba[:, :W // 2] = [0.1, 0.2, 0.3, 1.0]
    centre_depth = np.full((H, W), 0.0, dtype=np.float32)
    centre_depth[:, :W // 2] = 5.0

    # Sibling at the same pose, covers the full image with a different colour.
    sib_rgba = np.full((H, W, 4), [0.9, 0.9, 0.9, 1.0], dtype=np.float32)
    sib_depth = np.full((H, W), 5.0, dtype=np.float32)

    siblings = [{
        "rgba": sib_rgba,
        "depth": sib_depth,
        "world_from_cam": world_from_cam,
        "proj": proj,
    }]

    rgba, depth, coverage, source = consolidate_layer(
        centre_rgba=centre_rgba,
        centre_depth=centre_depth,
        siblings_sorted=siblings,
        centre_world_from_cam=world_from_cam,
        centre_proj=proj,
    )

    # Left half: centre colour preserved, source = 0.
    np.testing.assert_allclose(rgba[:, :W // 2, :3],
                                np.broadcast_to([0.1, 0.2, 0.3], (H, W // 2, 3)),
                                atol=1e-6)
    assert (source[:, :W // 2] == 0).all()

    # Right half: sibling fill, source = 1.
    np.testing.assert_allclose(rgba[:, W // 2:, :3],
                                np.broadcast_to([0.9, 0.9, 0.9], (H, W - W // 2, 3)),
                                atol=1e-6)
    assert (source[:, W // 2:] == 1).all()

    assert coverage.all()
    np.testing.assert_allclose(depth, 5.0, atol=1e-3)


def test_consolidate_closest_to_centre_wins():
    """Two siblings with overlapping fills; the first-iterated (closest)
    wins for shared pixels."""
    proj, world_from_cam = _identity_proj_and_world()
    H, W = 16, 16

    # Empty centre.
    centre_rgba = np.zeros((H, W, 4), dtype=np.float32)
    centre_depth = np.zeros((H, W), dtype=np.float32)

    near_sib = {
        "rgba": np.full((H, W, 4), [1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "depth": np.full((H, W), 5.0, dtype=np.float32),
        "world_from_cam": world_from_cam,
        "proj": proj,
    }
    far_sib = {
        "rgba": np.full((H, W, 4), [0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "depth": np.full((H, W), 5.0, dtype=np.float32),
        "world_from_cam": world_from_cam,
        "proj": proj,
    }

    rgba, _, _, source = consolidate_layer(
        centre_rgba=centre_rgba,
        centre_depth=centre_depth,
        siblings_sorted=[near_sib, far_sib],  # closest-to-centre first
        centre_world_from_cam=world_from_cam,
        centre_proj=proj,
    )

    # Near sibling (red) wins; source = 1 throughout.
    np.testing.assert_allclose(rgba[..., :3],
                                np.broadcast_to([1.0, 0.0, 0.0], (H, W, 3)),
                                atol=1e-6)
    assert (source == 1).all()


def test_consolidate_uncovered_pixel_alpha_zeroed():
    """Pixels neither centre nor any sibling covered end up with alpha=0
    and depth=0 in the canonical output."""
    proj, world_from_cam = _identity_proj_and_world()
    H, W = 8, 8

    centre_rgba = np.zeros((H, W, 4), dtype=np.float32)
    centre_depth = np.zeros((H, W), dtype=np.float32)

    rgba, depth, coverage, source = consolidate_layer(
        centre_rgba=centre_rgba,
        centre_depth=centre_depth,
        siblings_sorted=[],
        centre_world_from_cam=world_from_cam,
        centre_proj=proj,
    )

    assert not coverage.any()
    assert (rgba[..., 3] == 0.0).all()
    assert (depth == 0.0).all()
    assert (source == 0).all()


def test_alpha_threshold_constant_is_within_unit_range():
    assert 0.0 < COVERAGE_ALPHA_THRESHOLD < 1.0
