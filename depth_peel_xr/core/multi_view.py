"""Multi-view consolidation: reproject NxN per-layer captures into one
canonical centre-NDC view per layer.

Pure numpy — no bpy. Tests run standalone.

Pipeline per layer:
  1. Centre view's RGBA + depth seeds the canonical buffer; anything with
     `alpha > coverage_threshold` is taken as the authoritative pixel.
  2. Sibling views are processed in order of head-plane distance to centre
     (closest first). For each, we forward-splat (nearest-neighbour, with
     z-buffer guarding) into centre NDC; pixels that the canonical buffer
     doesn't yet cover get filled, marked source=sibling. Closest-to-centre
     wins because we iterate sorted and skip already-covered pixels.

Outputs:
  - canonical_rgba (H, W, 4) — centre + sibling fills, scene-linear.
  - canonical_depth (H, W)   — distance from centre camera along -Z.
  - coverage_mask (H, W) bool — True for any pixel a view contributed to.
  - source_mask (H, W) uint8 — 0 = centre, 1..N = sibling index (for debug).

Coordinate conventions:
  - Camera-space: Blender's, camera looks down -Z, +Y up, +X right.
  - NDC: x, y in [-1, 1], z in [-1, 1] (Blender's clip-space convention).
  - Image pixel (u, v): u increases rightward, v increases downward
    (matches Blender's pixel storage when extracted top-down).
  - Depth (from EEVEE/Cycles Z pass): positive scene-linear distance from
    camera along its -Z axis.

The reprojection bypasses NDC-z entirely — it builds a camera-space ray
from the projection matrix's near/far intercepts, then scales the ray to
land at `cam_z == -depth`. This makes the math invariant to whatever
clip range was set when the projection matrix was built (captures use
per-layer clip ranges, but consolidation runs after they've been
restored).
"""

from __future__ import annotations

import numpy as np


# Alpha threshold above which a pixel is considered "covered" by a view.
# Captures use film_transparent=True so geometry pixels have alpha=1 and
# miss pixels alpha=0; the threshold sits comfortably above the bilinear
# anti-aliased boundary.
COVERAGE_ALPHA_THRESHOLD = 0.5


def _ndc_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pixel NDC (u, v) grids, both shaped (H, W).
    u in [-1, 1] going right, v in [-1, 1] going up (Y flipped from
    image-pixel convention)."""
    us = (np.arange(width, dtype=np.float64) + 0.5)
    vs = (np.arange(height, dtype=np.float64) + 0.5)
    u_ndc = 2.0 * us / width - 1.0
    v_ndc = 1.0 - 2.0 * vs / height
    u_grid, v_grid = np.meshgrid(u_ndc, v_ndc)
    return u_grid, v_grid


def _unproject_to_camera_space(
    u_ndc: np.ndarray,
    v_ndc: np.ndarray,
    depth: np.ndarray,
    proj_inv: np.ndarray,
) -> np.ndarray:
    """For each pixel, find the camera-space 3D point that projects to
    (u_ndc, v_ndc) at the given scene-linear depth.

    Returns (H, W, 3) of camera-space (x, y, z).

    Builds two NDC points along the same view ray (at the near and far
    clip planes), transforms both to camera space, then scales along the
    ray to reach the requested depth. Independent of the clip range that
    built proj_inv."""
    H, W = u_ndc.shape
    ones = np.ones_like(u_ndc)
    ndc_near = np.stack([u_ndc, v_ndc, -ones,  ones], axis=-1)  # (H, W, 4)
    ndc_far  = np.stack([u_ndc, v_ndc,  ones,  ones], axis=-1)

    cam_near = ndc_near @ proj_inv.T
    cam_near = cam_near / cam_near[..., 3:4]
    cam_far = ndc_far @ proj_inv.T
    cam_far = cam_far / cam_far[..., 3:4]

    ray = cam_far[..., :3] - cam_near[..., :3]
    # Ray's z-component is negative (points away from camera origin).
    # We want cam_point.z == -depth, so:
    #   cam_point.z = cam_near.z + ray.z * t
    #   t = (-depth - cam_near.z) / ray.z
    denom = ray[..., 2]
    safe = denom != 0
    t = np.where(safe, (-depth - cam_near[..., 2]) / np.where(safe, denom, 1.0), 0.0)
    cam_point = cam_near[..., :3] + ray * t[..., None]
    return cam_point


def reproject_view_to_centre(
    sibling_rgba: np.ndarray,
    sibling_depth: np.ndarray,
    sib_world_from_cam: np.ndarray,
    sib_proj: np.ndarray,
    centre_cam_from_world: np.ndarray,
    centre_proj: np.ndarray,
    centre_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-splat sibling view's covered pixels into centre NDC.

    Returns (contrib_rgba, contrib_depth, contrib_valid):
      - contrib_rgba (Hc, Wc, 4)
      - contrib_depth (Hc, Wc), inf where no contribution
      - contrib_valid (Hc, Wc) bool

    Nearest-neighbour splat. Multiple sibling pixels landing in the same
    centre pixel are resolved by z-buffer (closest wins inside the
    sibling's own scatter). Pixels outside the centre frustum drop.
    """
    Hs, Ws, _ = sibling_rgba.shape
    Hc, Wc = centre_shape

    u_ndc_s, v_ndc_s = _ndc_grid(Ws, Hs)
    cam_pt_s = _unproject_to_camera_space(
        u_ndc_s, v_ndc_s, sibling_depth, np.linalg.inv(sib_proj)
    )  # (Hs, Ws, 3) in sibling cam space

    # Mask sibling pixels we should consider at all.
    alpha = sibling_rgba[..., 3]
    valid_src = (alpha > COVERAGE_ALPHA_THRESHOLD) & np.isfinite(sibling_depth) & (sibling_depth > 0)

    # Sibling cam → world → centre cam.
    cam_pt_4d = np.concatenate(
        [cam_pt_s, np.ones((Hs, Ws, 1), dtype=cam_pt_s.dtype)], axis=-1
    )
    world_pt = cam_pt_4d @ sib_world_from_cam.T  # (Hs, Ws, 4)
    centre_cam_pt = world_pt @ centre_cam_from_world.T  # (Hs, Ws, 4)

    # Project to centre NDC.
    centre_clip = centre_cam_pt @ centre_proj.T  # (Hs, Ws, 4)
    w = centre_clip[..., 3]
    w_safe = w != 0
    centre_ndc = np.where(
        w_safe[..., None],
        centre_clip / np.where(w_safe[..., None], centre_clip[..., 3:4], 1.0),
        0.0,
    )

    # NDC → centre pixel.
    u_c = (centre_ndc[..., 0] + 1.0) * 0.5 * Wc
    v_c = (1.0 - centre_ndc[..., 1]) * 0.5 * Hc
    d_c = -centre_cam_pt[..., 2]  # positive distance from centre camera

    in_frustum = (
        valid_src
        & w_safe
        & (u_c >= 0) & (u_c < Wc)
        & (v_c >= 0) & (v_c < Hc)
        & (d_c > 0)
        & np.isfinite(d_c)
    )

    # Gather contributing pixels.
    flat_idx = np.flatnonzero(in_frustum)
    if flat_idx.size == 0:
        return (
            np.zeros((Hc, Wc, 4), dtype=np.float32),
            np.full((Hc, Wc), np.inf, dtype=np.float32),
            np.zeros((Hc, Wc), dtype=bool),
        )

    src_rgba = sibling_rgba.reshape(-1, 4)[flat_idx]
    src_depth = d_c.ravel()[flat_idx]
    src_u = np.clip(np.floor(u_c.ravel()[flat_idx]).astype(np.int64), 0, Wc - 1)
    src_v = np.clip(np.floor(v_c.ravel()[flat_idx]).astype(np.int64), 0, Hc - 1)

    # z-buffer splat: keep the closest contribution per centre pixel.
    contrib_depth = np.full((Hc, Wc), np.inf, dtype=np.float32)
    contrib_rgba = np.zeros((Hc, Wc, 4), dtype=np.float32)
    contrib_valid = np.zeros((Hc, Wc), dtype=bool)

    # Sort sources by depth descending so later writes (closer) overwrite.
    order = np.argsort(-src_depth, kind="stable")
    src_u = src_u[order]
    src_v = src_v[order]
    src_depth = src_depth[order]
    src_rgba = src_rgba[order]

    contrib_depth[src_v, src_u] = src_depth
    contrib_rgba[src_v, src_u] = src_rgba
    contrib_valid[src_v, src_u] = True

    return contrib_rgba, contrib_depth, contrib_valid


def consolidate_layer(
    centre_rgba: np.ndarray,
    centre_depth: np.ndarray,
    siblings_sorted: list[dict],
    centre_world_from_cam: np.ndarray,
    centre_proj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Consolidate one layer's centre + sibling captures into canonical
    centre-NDC buffers. Siblings must already be sorted closest-to-centre
    first.

    Each sibling dict:
      { 'rgba': (H, W, 4), 'depth': (H, W),
        'world_from_cam': (4, 4), 'proj': (4, 4) }

    Returns (rgba, depth, coverage_mask, source_mask).
    """
    H, W, _ = centre_rgba.shape
    canonical_rgba = centre_rgba.astype(np.float32, copy=True)
    canonical_depth = centre_depth.astype(np.float32, copy=True)

    coverage_mask = canonical_rgba[..., 3] > COVERAGE_ALPHA_THRESHOLD
    source_mask = np.zeros((H, W), dtype=np.uint8)

    centre_cam_from_world = np.linalg.inv(centre_world_from_cam)

    for sib_idx, sib in enumerate(siblings_sorted, start=1):
        contrib_rgba, contrib_depth, contrib_valid = reproject_view_to_centre(
            sib["rgba"],
            sib["depth"],
            sib["world_from_cam"],
            sib["proj"],
            centre_cam_from_world,
            centre_proj,
            centre_shape=(H, W),
        )
        fill = (~coverage_mask) & contrib_valid
        canonical_rgba[fill] = contrib_rgba[fill]
        canonical_depth[fill] = contrib_depth[fill]
        source_mask[fill] = sib_idx
        coverage_mask |= fill

    # Where no view contributed, depth stays at whatever centre_depth held
    # (might be the background sentinel); set to 0 for cleanliness.
    canonical_depth = np.where(coverage_mask, canonical_depth, 0.0)
    # Same for RGBA's alpha — force to 0 where uncovered.
    canonical_rgba[..., 3] = np.where(coverage_mask, canonical_rgba[..., 3], 0.0)

    return canonical_rgba, canonical_depth, coverage_mask, source_mask
