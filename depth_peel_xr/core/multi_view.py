"""Multi-view consolidation: reproject NxN per-layer captures into one
canonical super-frustum-NDC view per layer.

Pure numpy — no bpy. Tests run standalone.

Pipeline per layer:
  1. Views are processed in priority order (centre first, then siblings
     closest-to-centre first). Each pixel above `COVERAGE_ALPHA_THRESHOLD`
     forward-splats into canonical NDC via per-pixel unproject (source cam)
     + reproject (canonical cam). Pixels that the canonical doesn't yet
     cover get filled.
  2. The canonical's virtual cam is a forward-shifted perspective that frames
     the super-frustum exactly — see core.geometry.super_frustum_projection.
     Centre-view pixels naturally land in a central region of the canonical
     buffer; the surrounding annulus is filled by whichever sibling can see
     it (geometrically nearest sibling wins because we iterate sorted).

Outputs:
  - canonical_rgba (H, W, 4) — scene-linear, alpha = coverage.
  - canonical_depth (H, W)   — distance along the canonical cam's -Z axis.
  - coverage_mask (H, W) bool — True for any pixel a view contributed to.
  - source_mask (H, W) uint8 — index into the input views_sorted list (so
    centre = 0 by convention when the caller puts it first).

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


def reproject_view_to_canonical(
    src_rgba: np.ndarray,
    src_depth: np.ndarray,
    src_world_from_cam: np.ndarray,
    src_proj: np.ndarray,
    canonical_cam_from_world: np.ndarray,
    canonical_proj: np.ndarray,
    canonical_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-splat a source view's covered pixels into canonical NDC.

    Returns (contrib_rgba, contrib_depth, contrib_valid):
      - contrib_rgba (Hc, Wc, 4)
      - contrib_depth (Hc, Wc), inf where no contribution
      - contrib_valid (Hc, Wc) bool

    Nearest-neighbour splat. Multiple source pixels landing in the same
    canonical pixel are resolved by z-buffer (closest wins inside the
    source's own scatter). Pixels outside the canonical frustum drop.
    """
    Hs, Ws, _ = src_rgba.shape
    Hc, Wc = canonical_shape

    u_ndc_s, v_ndc_s = _ndc_grid(Ws, Hs)
    cam_pt_s = _unproject_to_camera_space(
        u_ndc_s, v_ndc_s, src_depth, np.linalg.inv(src_proj)
    )  # (Hs, Ws, 3) in source cam space

    # Mask source pixels we should consider at all.
    alpha = src_rgba[..., 3]
    valid_src = (alpha > COVERAGE_ALPHA_THRESHOLD) & np.isfinite(src_depth) & (src_depth > 0)

    # Source cam → world → canonical cam.
    cam_pt_4d = np.concatenate(
        [cam_pt_s, np.ones((Hs, Ws, 1), dtype=cam_pt_s.dtype)], axis=-1
    )
    world_pt = cam_pt_4d @ src_world_from_cam.T  # (Hs, Ws, 4)
    canonical_cam_pt = world_pt @ canonical_cam_from_world.T  # (Hs, Ws, 4)

    # Project to canonical NDC.
    canonical_clip = canonical_cam_pt @ canonical_proj.T  # (Hs, Ws, 4)
    w = canonical_clip[..., 3]
    w_safe = w != 0
    canonical_ndc = np.where(
        w_safe[..., None],
        canonical_clip / np.where(w_safe[..., None], canonical_clip[..., 3:4], 1.0),
        0.0,
    )

    # NDC → canonical pixel.
    u_c = (canonical_ndc[..., 0] + 1.0) * 0.5 * Wc
    v_c = (1.0 - canonical_ndc[..., 1]) * 0.5 * Hc
    d_c = -canonical_cam_pt[..., 2]  # positive distance from canonical apex

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

    src_rgba_flat = src_rgba.reshape(-1, 4)[flat_idx]
    src_depth_c = d_c.ravel()[flat_idx]
    src_u = np.clip(np.floor(u_c.ravel()[flat_idx]).astype(np.int64), 0, Wc - 1)
    src_v = np.clip(np.floor(v_c.ravel()[flat_idx]).astype(np.int64), 0, Hc - 1)

    # z-buffer splat: keep the closest contribution per canonical pixel.
    contrib_depth = np.full((Hc, Wc), np.inf, dtype=np.float32)
    contrib_rgba = np.zeros((Hc, Wc, 4), dtype=np.float32)
    contrib_valid = np.zeros((Hc, Wc), dtype=bool)

    # Sort sources by depth descending so later writes (closer) overwrite.
    order = np.argsort(-src_depth_c, kind="stable")
    src_u = src_u[order]
    src_v = src_v[order]
    src_depth_c = src_depth_c[order]
    src_rgba_flat = src_rgba_flat[order]

    contrib_depth[src_v, src_u] = src_depth_c
    contrib_rgba[src_v, src_u] = src_rgba_flat
    contrib_valid[src_v, src_u] = True

    return contrib_rgba, contrib_depth, contrib_valid


def consolidate_layer(
    views_sorted: list[dict],
    canonical_shape: tuple[int, int],
    canonical_cam_from_world: np.ndarray,
    canonical_proj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Consolidate one layer's views into super-frustum canonical buffers.

    Views are processed in the order given. The caller's convention is
    centre first, then siblings sorted closest-to-centre first — so the
    centre view's pixels seed the canonical's central region and each
    annulus pixel goes to the sibling geometrically nearest the position
    from which it's visible.

    Each view dict:
      { 'rgba': (H, W, 4), 'depth': (H, W),
        'world_from_cam': (4, 4), 'proj': (4, 4) }

    Returns (rgba, depth, coverage_mask, source_mask). `source_mask[y, x]`
    is the index into views_sorted that filled that pixel (only meaningful
    where coverage_mask is True; 0 elsewhere).
    """
    Hc, Wc = canonical_shape
    canonical_rgba = np.zeros((Hc, Wc, 4), dtype=np.float32)
    canonical_depth = np.zeros((Hc, Wc), dtype=np.float32)
    coverage_mask = np.zeros((Hc, Wc), dtype=bool)
    source_mask = np.zeros((Hc, Wc), dtype=np.uint8)

    for view_idx, view in enumerate(views_sorted):
        contrib_rgba, contrib_depth, contrib_valid = reproject_view_to_canonical(
            view["rgba"],
            view["depth"],
            view["world_from_cam"],
            view["proj"],
            canonical_cam_from_world,
            canonical_proj,
            canonical_shape=(Hc, Wc),
        )
        fill = (~coverage_mask) & contrib_valid
        canonical_rgba[fill] = contrib_rgba[fill]
        canonical_depth[fill] = contrib_depth[fill]
        source_mask[fill] = view_idx
        coverage_mask |= fill

    # Force alpha=0 where uncovered for downstream coverage reads.
    canonical_rgba[..., 3] = np.where(coverage_mask, canonical_rgba[..., 3], 0.0)

    return canonical_rgba, canonical_depth, coverage_mask, source_mask
