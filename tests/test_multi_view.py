"""Tests for core.multi_view reprojection + consolidation math.

Synthetic camera setup with hand-crafted projection matrices, so
results are deterministic and don't depend on Blender's API.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.geometry import super_frustum_projection
from core.multi_view import (
    COVERAGE_ALPHA_THRESHOLD,
    DEFAULT_REVERSE_DEPTH_REL_TOL,
    REVERSE_FALLBACK_SOURCE,
    _remove_single_pixel_anomalies,
    consolidate_layer_forward,
    consolidate_layer_reverse,
    reproject_view_to_canonical,
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

    contrib_rgba, contrib_depth, contrib_valid = reproject_view_to_canonical(
        src_rgba=rgba,
        src_depth=depth,
        src_world_from_cam=world_from_cam,
        src_proj=proj,
        canonical_cam_from_world=np.linalg.inv(world_from_cam),
        canonical_proj=proj,
        canonical_shape=(H, W),
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
    # clip_x = f_aspect * -1, clip_w = 5 ⇒ ndc_x = -f_aspect / 5
    expected_sib_ndc_x = -f_aspect / 5.0
    expected_sib_pixel_u = int(round((expected_sib_ndc_x + 1.0) * 0.5 * W - 0.5))

    rgba = np.zeros((H, W, 4), dtype=np.float32)
    depth = np.zeros((H, W), dtype=np.float32)
    rgba[32, expected_sib_pixel_u] = [1.0, 0.0, 0.0, 1.0]
    depth[32, expected_sib_pixel_u] = 5.0

    contrib_rgba, contrib_depth, contrib_valid = reproject_view_to_canonical(
        src_rgba=rgba,
        src_depth=depth,
        src_world_from_cam=sib_world_from_cam,
        src_proj=proj,
        canonical_cam_from_world=np.linalg.inv(centre_world_from_cam),
        canonical_proj=proj,
        canonical_shape=(H, W),
    )

    # Should reproject back to centre image (around pixel 32,32 ± 1 due to rounding).
    where = np.argwhere(contrib_valid)
    assert where.shape[0] == 1
    v, u = where[0]
    assert v == 32
    assert abs(u - 32) <= 1
    assert contrib_depth[v, u] == pytest.approx(5.0, abs=1e-3)


def test_invalid_source_pixels_dont_splat():
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

    _, _, valid = reproject_view_to_canonical(
        src_rgba=rgba,
        src_depth=depth,
        src_world_from_cam=world_from_cam,
        src_proj=proj,
        canonical_cam_from_world=np.linalg.inv(world_from_cam),
        canonical_proj=proj,
        canonical_shape=(H, W),
    )
    assert not valid.any()


def test_consolidate_preserves_centre_priority():
    """Centre is iterated first; siblings only fill where centre missed."""
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

    views = [
        {"rgba": centre_rgba, "depth": centre_depth,
         "world_from_cam": world_from_cam, "proj": proj},
        {"rgba": sib_rgba, "depth": sib_depth,
         "world_from_cam": world_from_cam, "proj": proj},
    ]

    rgba, depth, coverage, source = consolidate_layer_forward(
        views_sorted=views,
        canonical_shape=(H, W),
        canonical_cam_from_world=np.linalg.inv(world_from_cam),
        canonical_proj=proj,
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
    """First-iterated sibling wins for shared pixels (caller sorts
    closest-to-centre first)."""
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
    centre = {
        "rgba": centre_rgba, "depth": centre_depth,
        "world_from_cam": world_from_cam, "proj": proj,
    }

    rgba, _, _, source = consolidate_layer_forward(
        views_sorted=[centre, near_sib, far_sib],
        canonical_shape=(H, W),
        canonical_cam_from_world=np.linalg.inv(world_from_cam),
        canonical_proj=proj,
    )

    # Near sibling (red) wins; source = 1 (centre is 0 but covered nothing).
    np.testing.assert_allclose(rgba[..., :3],
                                np.broadcast_to([1.0, 0.0, 0.0], (H, W, 3)),
                                atol=1e-6)
    assert (source == 1).all()


def test_consolidate_uncovered_pixel_alpha_zeroed():
    """Pixels no view covered end up with alpha=0 and depth=0."""
    proj, world_from_cam = _identity_proj_and_world()
    H, W = 8, 8

    centre = {
        "rgba": np.zeros((H, W, 4), dtype=np.float32),
        "depth": np.zeros((H, W), dtype=np.float32),
        "world_from_cam": world_from_cam, "proj": proj,
    }

    rgba, depth, coverage, source = consolidate_layer_forward(
        views_sorted=[centre],
        canonical_shape=(H, W),
        canonical_cam_from_world=np.linalg.inv(world_from_cam),
        canonical_proj=proj,
    )

    assert not coverage.any()
    assert (rgba[..., 3] == 0.0).all()
    assert (depth == 0.0).all()
    assert (source == 0).all()


def test_alpha_threshold_constant_is_within_unit_range():
    assert 0.0 < COVERAGE_ALPHA_THRESHOLD < 1.0


# ---- single-pixel anomaly removal ------------------------------------------

def _make_uniform_canonical(H: int, W: int, depth_val: float, source_val: int):
    """Build a fully-covered canonical with uniform colour, depth and source."""
    rgba = np.tile([0.5, 0.5, 0.5, 1.0], (H, W, 1)).astype(np.float32)
    depth = np.full((H, W), depth_val, dtype=np.float32)
    coverage = np.ones((H, W), dtype=bool)
    source = np.full((H, W), source_val, dtype=np.uint8)
    return rgba, depth, coverage, source


def test_anomaly_pass_replaces_source_speck():
    """A covered pixel whose source view differs from all 4 neighbours
    is treated as a 1-pixel forward-splat anomaly and replaced by the
    median of covered neighbours."""
    rgba, depth, coverage, source = _make_uniform_canonical(8, 8, 3.0, source_val=0)
    # Speck at (3, 3): different colour, depth, and source view.
    rgba[3, 3] = [1.0, 0.0, 0.0, 1.0]
    depth[3, 3] = 99.0
    source[3, 3] = 5  # sibling

    new_rgba, new_depth, new_cov, new_src = _remove_single_pixel_anomalies(
        rgba, depth, coverage, source
    )

    # Speck depth and colour replaced by neighbour median (3.0 / grey).
    assert new_depth[3, 3] == pytest.approx(3.0)
    np.testing.assert_allclose(new_rgba[3, 3, :3], [0.5, 0.5, 0.5], atol=1e-6)
    # Stays covered, source marked synthesised (255).
    assert new_cov[3, 3]
    assert new_src[3, 3] == 255
    # Untouched neighbour stays at original values.
    assert new_depth[2, 2] == pytest.approx(3.0)
    assert new_src[2, 2] == 0


def test_anomaly_pass_fills_hole():
    """An uncovered pixel with all 4 neighbours covered gets median-
    filled and marked covered."""
    rgba, depth, coverage, source = _make_uniform_canonical(8, 8, 3.0, source_val=0)
    coverage[4, 4] = False
    rgba[4, 4] = 0.0
    depth[4, 4] = 0.0

    new_rgba, new_depth, new_cov, _ = _remove_single_pixel_anomalies(
        rgba, depth, coverage, source
    )

    assert new_cov[4, 4]
    assert new_depth[4, 4] == pytest.approx(3.0)
    np.testing.assert_allclose(new_rgba[4, 4], [0.5, 0.5, 0.5, 1.0], atol=1e-6)


def test_anomaly_pass_drops_stray():
    """A covered pixel with no covered 4-neighbours is dropped."""
    H = W = 8
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    depth = np.zeros((H, W), dtype=np.float32)
    coverage = np.zeros((H, W), dtype=bool)
    source = np.zeros((H, W), dtype=np.uint8)
    # Single covered stray in the negative space.
    coverage[2, 5] = True
    rgba[2, 5] = [0.7, 0.2, 0.1, 1.0]
    depth[2, 5] = 7.0
    source[2, 5] = 3

    new_rgba, new_depth, new_cov, new_src = _remove_single_pixel_anomalies(
        rgba, depth, coverage, source
    )

    assert not new_cov[2, 5]
    assert new_depth[2, 5] == 0.0
    assert new_rgba[2, 5, 3] == 0.0
    assert new_src[2, 5] == 0


def test_anomaly_pass_leaves_real_silhouette_alone():
    """A clean covered/uncovered boundary (not single-pixel anomalies)
    must not be touched."""
    H = W = 8
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    depth = np.zeros((H, W), dtype=np.float32)
    coverage = np.zeros((H, W), dtype=bool)
    source = np.zeros((H, W), dtype=np.uint8)
    # Left half covered with depth 4, source 0.
    coverage[:, :4] = True
    rgba[:, :4] = [0.3, 0.6, 0.9, 1.0]
    depth[:, :4] = 4.0
    source[:, :4] = 0

    new_rgba, new_depth, new_cov, new_src = _remove_single_pixel_anomalies(
        rgba, depth, coverage, source
    )

    np.testing.assert_array_equal(new_cov, coverage)
    np.testing.assert_allclose(new_depth, depth)
    np.testing.assert_allclose(new_rgba, rgba)
    np.testing.assert_array_equal(new_src, source)


def test_anomaly_pass_leaves_uniform_source_pixel_alone():
    """A covered pixel sharing its source view with neighbours stays
    untouched even if its colour or depth differs slightly — the pass
    keys off source-mask agreement, not value comparison."""
    rgba, depth, coverage, source = _make_uniform_canonical(8, 8, 3.0, source_val=0)
    # Different colour but same source as neighbours → not a speck.
    rgba[3, 3] = [0.9, 0.1, 0.2, 1.0]
    new_rgba, new_depth, _, _ = _remove_single_pixel_anomalies(
        rgba, depth, coverage, source
    )
    np.testing.assert_allclose(new_rgba[3, 3], [0.9, 0.1, 0.2, 1.0])


def test_anomaly_pass_drops_two_pixel_stray():
    """Morphological opening drops covered blobs up to 2×2 — a 1×2
    stray pair in the negative space gets cleared."""
    H = W = 8
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    depth = np.zeros((H, W), dtype=np.float32)
    coverage = np.zeros((H, W), dtype=bool)
    source = np.zeros((H, W), dtype=np.uint8)
    coverage[3, 4] = True
    coverage[3, 5] = True
    rgba[3, 4:6] = [0.7, 0.2, 0.1, 1.0]
    depth[3, 4:6] = 7.0

    new_rgba, new_depth, new_cov, _ = _remove_single_pixel_anomalies(
        rgba, depth, coverage, source
    )

    assert not new_cov[3, 4]
    assert not new_cov[3, 5]


def test_anomaly_pass_fills_one_pixel_wide_line_hole():
    """Morphological closing fills 1-pixel-wide line holes that depth-
    plane cuts can carve through scene geometry. A vertical line of
    uncovered pixels with covered cells on both sides fills in."""
    H = W = 12
    coverage = np.ones((H, W), dtype=bool)
    rgba = np.tile([0.5, 0.5, 0.5, 1.0], (H, W, 1)).astype(np.float32)
    depth = np.full((H, W), 4.0, dtype=np.float32)
    source = np.zeros((H, W), dtype=np.uint8)
    # 1-pixel-wide vertical hole in the middle.
    coverage[2:H - 2, W // 2] = False
    depth[2:H - 2, W // 2] = 0.0
    rgba[2:H - 2, W // 2] = 0.0

    new_rgba, new_depth, new_cov, new_src = _remove_single_pixel_anomalies(
        rgba, depth, coverage, source
    )

    assert new_cov[2:H - 2, W // 2].all()
    np.testing.assert_allclose(new_depth[2:H - 2, W // 2], 4.0)
    np.testing.assert_allclose(new_rgba[2:H - 2, W // 2, :3], 0.5, atol=1e-6)
    # Synthesised pixels marked with sentinel source.
    assert (new_src[2:H - 2, W // 2] == 255).all()


def test_anomaly_pass_preserves_3x3_blob():
    """Morphological opening preserves 3×3 covered blobs untouched."""
    H = W = 12
    coverage = np.zeros((H, W), dtype=bool)
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    depth = np.zeros((H, W), dtype=np.float32)
    source = np.zeros((H, W), dtype=np.uint8)
    coverage[4:7, 4:7] = True
    rgba[4:7, 4:7] = [0.3, 0.6, 0.9, 1.0]
    depth[4:7, 4:7] = 5.0

    new_rgba, new_depth, new_cov, _ = _remove_single_pixel_anomalies(
        rgba, depth, coverage, source
    )

    np.testing.assert_array_equal(new_cov, coverage)
    np.testing.assert_allclose(new_depth, depth)


# ---- super-frustum canonical: sibling-only annulus pixels --------------------

def test_super_frustum_canonical_fills_annulus_from_sibling():
    """A foreground occluder visible only to an off-centre sibling lands
    in the canonical's annulus — outside the centre view's clip but inside
    the super-frustum.

    Geometry: portal half-extents (2.0, 1.0) at ho=2, head-plane offset 2.
    Centre cam at origin; sibling shifted +1 in x (toward the right edge of
    the head-plane). A bright pixel on the sibling at its right edge falls
    just outside the centre frustum at that depth, but is captured by the
    canonical (super-frustum extends past the centre frustum past d = ho).
    """
    half_w_p, half_h_p, ho = 2.0, 1.0, 2.0
    Hs = Ws = 64
    # Layer near = 3·ho so super/centre ratio = (2·3/1 - 1) / 3 = 5/3.
    layer_near = 3.0 * ho
    canonical_W = int(round(Ws * 5 / 3))
    canonical_H = int(round(Hs * 5 / 3))

    # Centre cam projection: symmetric perspective from origin with
    # frustum bounds (±half_w_p, ±half_h_p) at z=-ho.
    n_centre, f_centre = ho, 100.0
    centre_proj = np.array([
        [n_centre / half_w_p, 0, 0, 0],
        [0, n_centre / half_h_p, 0, 0],
        [0, 0, -(f_centre + n_centre) / (f_centre - n_centre),
         -2 * f_centre * n_centre / (f_centre - n_centre)],
        [0, 0, -1, 0],
    ], dtype=np.float64)
    centre_world_from_cam = np.eye(4, dtype=np.float64)

    # Sibling cam: same intrinsics, translated +x by head-plane-half.
    head_half = 1.0
    sib_world_from_cam = _translation(head_half, 0.0, 0.0)
    sib_proj = centre_proj.copy()

    # Place a covered pixel in the sibling near its right edge (NDC u ≈ +0.9)
    # at depth = layer_near, with bright red. In world space:
    #   sib NDC u=0.9 at depth d=6 → sibling-cam x = 0.9 * half_w_p * d/ho
    #     = 0.9 * 2 * 3 = 5.4
    #   world x = sib_x + head_half = 6.4
    # Centre NDC for this world point at depth 6: u = 6.4 / (centre_w(6)) = 6.4/6 = 1.067
    # That's outside the centre frustum (>|1|).
    sib_rgba = np.zeros((Hs, Ws, 4), dtype=np.float32)
    sib_depth = np.zeros((Hs, Ws), dtype=np.float32)
    # NDC u=0.9, v=0 → pixel (Hs/2, ~Ws*0.95). Use a 5×5 source patch so
    # the consolidated coverage produces ≥3×3 in the canonical (the post-
    # consolidate morphological opening drops blobs of 2×2 and smaller).
    sib_pixel_u = int((0.9 + 1.0) * 0.5 * Ws)
    sib_pixel_v = Hs // 2
    for du in range(-2, 3):
        for dv in range(-2, 3):
            sib_rgba[sib_pixel_v + dv, sib_pixel_u + du] = [1.0, 0.0, 0.0, 1.0]
            sib_depth[sib_pixel_v + dv, sib_pixel_u + du] = layer_near

    # Centre view is empty.
    centre = {
        "rgba": np.zeros((Hs, Ws, 4), dtype=np.float32),
        "depth": np.zeros((Hs, Ws), dtype=np.float32),
        "world_from_cam": centre_world_from_cam,
        "proj": centre_proj,
    }
    sibling = {
        "rgba": sib_rgba, "depth": sib_depth,
        "world_from_cam": sib_world_from_cam, "proj": sib_proj,
    }

    # Canonical super-frustum projection.
    canonical_from_centre, canonical_proj = super_frustum_projection(
        half_w_p, half_h_p, ho
    )
    canonical_cam_from_world = canonical_from_centre @ np.linalg.inv(centre_world_from_cam)

    rgba, depth, coverage, source = consolidate_layer_forward(
        views_sorted=[centre, sibling],
        canonical_shape=(canonical_H, canonical_W),
        canonical_cam_from_world=canonical_cam_from_world,
        canonical_proj=canonical_proj,
    )

    # The sibling region should land inside the canonical buffer (no out-of-
    # frustum drop) and outside the central centre-view sub-rect.
    where = np.argwhere(coverage)
    assert where.shape[0] >= 1, "expected sibling patch to land in the canonical"
    # Pick a covered pixel near the centre of the sibling's source patch.
    v, u = where[where.shape[0] // 2]
    # Centre's sub-rect spans canonical pixels [(Wc - Ws)/2, (Wc + Ws)/2]
    # (approximately, since centre/super ratio at portal is 1 but here we're
    # at depth 3·ho so sub-rect is centre_w/super_w = 3/5 of the canonical
    # width). The covered pixel must land past that boundary on the right.
    centre_subrect_right = (canonical_W + canonical_W * 3 // 5) // 2
    assert u >= centre_subrect_right, (
        f"sibling pixel at canonical u={u}, expected >= {centre_subrect_right}"
    )
    # And the colour is the sibling red. Canonical depth is measured from
    # the virtual apex (forward-shifted ho/2 from the centre cam) → world
    # depth `layer_near` reads as `layer_near - ho/2` in the canonical.
    np.testing.assert_allclose(rgba[v, u, :3], [1.0, 0.0, 0.0], atol=1e-5)
    assert depth[v, u] == pytest.approx(layer_near - ho / 2.0, rel=1e-2)
    assert source[v, u] == 1


def test_centre_view_lands_in_canonical_centre_region():
    """A centre-view pixel splats into the canonical's central region (not
    the full canonical, because canonical_W > centre_W)."""
    half_w_p, half_h_p, ho = 2.0, 1.0, 2.0
    Hs = Ws = 64
    layer_near = 3.0 * ho  # super/centre = 5/3

    n_centre, f_centre = ho, 100.0
    centre_proj = np.array([
        [n_centre / half_w_p, 0, 0, 0],
        [0, n_centre / half_h_p, 0, 0],
        [0, 0, -(f_centre + n_centre) / (f_centre - n_centre),
         -2 * f_centre * n_centre / (f_centre - n_centre)],
        [0, 0, -1, 0],
    ], dtype=np.float64)
    centre_world_from_cam = np.eye(4, dtype=np.float64)

    # Centre right-edge pixel at depth layer_near. Use a 5×5 patch so the
    # post-consolidate morphological opening doesn't drop the result.
    centre_rgba = np.zeros((Hs, Ws, 4), dtype=np.float32)
    centre_depth = np.zeros((Hs, Ws), dtype=np.float32)
    cv, cu = Hs // 2, Ws - 1
    for du in range(-4, 1):  # rightmost five columns
        for dv in range(-2, 3):
            centre_rgba[cv + dv, cu + du] = [0.0, 1.0, 0.0, 1.0]
            centre_depth[cv + dv, cu + du] = layer_near

    centre = {
        "rgba": centre_rgba, "depth": centre_depth,
        "world_from_cam": centre_world_from_cam, "proj": centre_proj,
    }

    canonical_W = int(round(Ws * 5 / 3))
    canonical_H = int(round(Hs * 5 / 3))
    canonical_from_centre, canonical_proj = super_frustum_projection(
        half_w_p, half_h_p, ho
    )
    canonical_cam_from_world = canonical_from_centre @ np.linalg.inv(centre_world_from_cam)

    _, _, coverage, _ = consolidate_layer_forward(
        views_sorted=[centre],
        canonical_shape=(canonical_H, canonical_W),
        canonical_cam_from_world=canonical_cam_from_world,
        canonical_proj=canonical_proj,
    )

    where = np.argwhere(coverage)
    assert where.shape[0] >= 1
    # Pick a covered pixel near the middle of the splatted region.
    _, u = where[where.shape[0] // 2]
    # Right edge at NDC ~0.6 maps to canonical pixel ≈ 0.8 * canonical_W.
    expected_u = int(0.8 * canonical_W)
    assert abs(u - expected_u) <= 3


# ---- reverse-mapping consolidation -----------------------------------------

def test_reverse_mapping_yields_bilinear_sub_pixel_colour():
    """Pass-2 reverse-maps each canonical pixel to a fractional source
    position and bilinear-samples — so a colour gradient survives at sub-
    pixel precision. The forward splat alone NN-quantises to a single
    source pixel's value; this test asserts the reverse result is the
    bilinear-interpolated value, not the NN value."""
    proj, world_from_cam = _identity_proj_and_world()
    Hs, Ws = 16, 16
    Hc, Wc = 8, 8  # canonical SMALLER than source so every canonical pixel
                   # gets ≥1 source splat (full coverage, morph opening
                   # leaves it untouched) while canonical → source still
                   # maps to fractional pixel positions for bilinear.

    # Linear red gradient across columns of the source. Alpha=1 everywhere.
    centre_rgba = np.zeros((Hs, Ws, 4), dtype=np.float32)
    centre_rgba[..., 0] = np.linspace(0.0, 1.0, Ws, dtype=np.float32)[None, :]
    centre_rgba[..., 3] = 1.0
    centre_depth = np.full((Hs, Ws), 5.0, dtype=np.float32)

    # Two siblings — same pose; centre is priority 1 so reverse-mapping
    # picks centre everywhere and sibling values shouldn't appear in the
    # output. Their presence just satisfies the "synthetic 3-view scene".
    sib_rgba = np.full((Hs, Ws, 4), [0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    sib_depth = np.full((Hs, Ws), 5.0, dtype=np.float32)
    centre = {
        "rgba": centre_rgba, "depth": centre_depth,
        "world_from_cam": world_from_cam, "proj": proj,
    }
    sib1 = {
        "rgba": sib_rgba.copy(), "depth": sib_depth.copy(),
        "world_from_cam": world_from_cam, "proj": proj,
    }
    sib2 = {
        "rgba": sib_rgba.copy(), "depth": sib_depth.copy(),
        "world_from_cam": world_from_cam, "proj": proj,
    }

    rgba_rev, depth_rev, coverage, source = consolidate_layer_reverse(
        views_sorted=[centre, sib1, sib2],
        canonical_shape=(Hc, Wc),
        canonical_cam_from_world=np.linalg.inv(world_from_cam),
        canonical_proj=proj,
    )

    assert coverage.all()
    # Centre wins reverse-mapping at every covered pixel (passes both
    # gates first in priority order).
    assert (source == 0).all()

    # Canonical pixel (4, 4): NDC (0.125, -0.125) → source u_pix = v_pix = 8.5,
    # bilinear of source cols 8 and 9 with equal weights.
    cu, cv = Wc // 2, Hc // 2
    canonical_ndc_u = (cu + 0.5) / Wc * 2.0 - 1.0
    source_u_frac = (canonical_ndc_u + 1.0) * 0.5 * Ws - 0.5
    u0 = int(np.floor(source_u_frac))
    fu = source_u_frac - u0
    red_ramp = np.linspace(0.0, 1.0, Ws)
    expected_red = (1.0 - fu) * red_ramp[u0] + fu * red_ramp[u0 + 1]
    np.testing.assert_allclose(rgba_rev[cv, cu, 0], expected_red, atol=2e-3)

    # Sanity: not equal to either NN-quantised neighbour value (so the
    # test isn't accidentally passing because forward and reverse coincide).
    assert abs(rgba_rev[cv, cu, 0] - red_ramp[u0]) > 1e-2
    assert abs(rgba_rev[cv, cu, 0] - red_ramp[u0 + 1]) > 1e-2

    # Pass-1 (forward) owns depth, untouched by pass-2.
    np.testing.assert_allclose(depth_rev, 5.0, atol=1e-3)


def test_reverse_mapping_disocclusion_rejects_closer_occluder():
    """Closer-occluder sibling is offered first in pass-2's priority
    order but fails the depth-consistency gate; the next-priority sibling
    that actually sees the surface at pass-1's recorded depth wins.

    Geometry: world point W = (0, 0, -5).
      - centre cam (origin): no coverage at W's pixel.
      - sib1 (priority 1, world translation +X 0.5): a closer occluder at
        depth 3 in the pixel where W projects (so projecting W back into
        sib1's image lands on a real, alpha=1 pixel — but with a closer
        depth than W's true depth). Disocclusion → depth gate must fail.
      - sib2 (priority 2, world translation -X 0.5): the far surface at
        depth 5 in the pixel where W projects. Both gates pass.
    """
    proj, _ = _identity_proj_and_world()
    Hs, Ws = 64, 64
    f_aspect = proj[0, 0]  # = 1.732 for fovy=60°, aspect=1

    centre_world_from_cam = np.eye(4, dtype=np.float64)
    sib1_world_from_cam = _translation(+0.5, 0.0, 0.0)
    sib2_world_from_cam = _translation(-0.5, 0.0, 0.0)

    # Where W projects in each sibling's image (Y centred). For W = (0,0,-5)
    # in a sibling at world (±0.5, 0, 0), sibling-cam-x = ∓0.5 at z=-5,
    # NDC u = sibling-cam-x / 5 * f_aspect. ⇒ sib1 ≈ -0.173, sib2 ≈ +0.173.
    u_sib1_ndc = -0.5 / 5.0 * f_aspect
    u_sib2_ndc = +0.5 / 5.0 * f_aspect
    u_sib1_pix = int(round((u_sib1_ndc + 1.0) * 0.5 * Ws))
    u_sib2_pix = int(round((u_sib2_ndc + 1.0) * 0.5 * Ws))
    v_centre = Hs // 2

    # 5×5 patches so the canonical splat survives morphological opening
    # in pass-1 cleanup and ends up with ≥1 fully-covered pixel at the
    # patch centre.
    sib1_rgba = np.zeros((Hs, Ws, 4), dtype=np.float32)
    sib1_depth = np.zeros((Hs, Ws), dtype=np.float32)
    for du in range(-2, 3):
        for dv in range(-2, 3):
            sib1_rgba[v_centre + dv, u_sib1_pix + du] = [1.0, 0.0, 0.0, 1.0]
            sib1_depth[v_centre + dv, u_sib1_pix + du] = 3.0  # closer occluder

    sib2_rgba = np.zeros((Hs, Ws, 4), dtype=np.float32)
    sib2_depth = np.zeros((Hs, Ws), dtype=np.float32)
    for du in range(-2, 3):
        for dv in range(-2, 3):
            sib2_rgba[v_centre + dv, u_sib2_pix + du] = [0.0, 1.0, 0.0, 1.0]
            sib2_depth[v_centre + dv, u_sib2_pix + du] = 5.0  # far surface

    centre = {
        "rgba": np.zeros((Hs, Ws, 4), dtype=np.float32),
        "depth": np.zeros((Hs, Ws), dtype=np.float32),
        "world_from_cam": centre_world_from_cam, "proj": proj,
    }
    sib1 = {
        "rgba": sib1_rgba, "depth": sib1_depth,
        "world_from_cam": sib1_world_from_cam, "proj": proj,
    }
    sib2 = {
        "rgba": sib2_rgba, "depth": sib2_depth,
        "world_from_cam": sib2_world_from_cam, "proj": proj,
    }

    # Identity canonical = centre cam → W projects to canonical NDC=0,
    # i.e. canonical pixel (Hs/2, Ws/2). Use the source resolution as
    # the canonical too (no super-frustum scaling — we want a clean
    # geometric match for the test).
    rgba, _depth, coverage, source = consolidate_layer_reverse(
        views_sorted=[centre, sib1, sib2],
        canonical_shape=(Hs, Ws),
        canonical_cam_from_world=np.linalg.inv(centre_world_from_cam),
        canonical_proj=proj,
        rel_tol=DEFAULT_REVERSE_DEPTH_REL_TOL,
    )

    # The canonical pixel where W lands (sib2's far-surface splat — sib1
    # splatted to a different canonical column because its closer-depth
    # changed the world-point reconstruction).
    cv, cu = Hs // 2, Ws // 2
    assert coverage[cv, cu], "sib2 should have splatted W into canonical centre"
    np.testing.assert_allclose(rgba[cv, cu, :3], [0.0, 1.0, 0.0], atol=1e-5)
    assert source[cv, cu] == 2, (
        f"expected sib2 (idx 2) after sib1 fails depth gate, got {source[cv, cu]}"
    )


def test_reverse_mapping_falls_back_to_pass1_when_no_view_passes():
    """If every view fails both gates at a covered canonical pixel, the
    reverse path leaves pass-1's RGBA untouched and writes the fallback
    sentinel into source_mask."""
    proj, world_from_cam = _identity_proj_and_world()
    Hs, Ws = 16, 16

    # Centre fully covered with magenta. Pass-1 records depth=5 +
    # colour=magenta everywhere.
    centre_rgba = np.full((Hs, Ws, 4), [1.0, 0.0, 1.0, 1.0], dtype=np.float32)
    centre_depth = np.full((Hs, Ws), 5.0, dtype=np.float32)
    centre = {
        "rgba": centre_rgba, "depth": centre_depth,
        "world_from_cam": world_from_cam, "proj": proj,
    }

    # Forcing rel_tol=0 makes the depth-consistency gate impossible to
    # satisfy at float precision, so pass-2 falls back to pass-1's colour
    # at every covered pixel.
    rgba, _depth, coverage, source = consolidate_layer_reverse(
        views_sorted=[centre],
        canonical_shape=(Hs, Ws),
        canonical_cam_from_world=np.linalg.inv(world_from_cam),
        canonical_proj=proj,
        rel_tol=0.0,
    )

    assert coverage.all()
    np.testing.assert_allclose(
        rgba[..., :3],
        np.broadcast_to([1.0, 0.0, 1.0], (Hs, Ws, 3)),
        atol=1e-5,
    )
    assert (source == REVERSE_FALLBACK_SOURCE).all()
