"""Per-vertex occlusion-culling visibility test for displaced layer meshes.

For each polygon walked far→near, decide whether to cull it based on
whether every one of its vertices is hidden from every head-plane camera
by some closer-layer surface. The closer-surface depth buffer is
supplied by the caller (per cam, with +inf where no closer layer covers).

Conservative: a polygon is dropped only if every one of its verts is
hidden from every cam — the moment any vert is visible (in-frustum AND
in-front-of-closer-surface) from any cam, the polygon survives. Misses
the rare "all corners hidden, interior visible" case where a closer
surface bumps non-monotonically inside the projected polygon — a future
per-pixel rasterizer upgrade would catch that.

Pure numpy — no bpy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def project_verts_to_cam(
    vertices_world: np.ndarray,
    world_from_cam: np.ndarray,
    proj: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world-space verts into a camera's screen space.

    Returns (u_pix, v_pix, depth) — all shape (V,). u/v are pixel coords
    (top-left origin); depth is positive distance from the cam along its
    -Z axis. Verts behind the camera have depth <= 0.
    """
    V = vertices_world.shape[0]
    homog = np.concatenate(
        [vertices_world, np.ones((V, 1), dtype=vertices_world.dtype)],
        axis=-1,
    )
    cam_from_world = np.linalg.inv(world_from_cam)
    cam_pos = (cam_from_world @ homog.T).T  # (V, 4)
    clip = (proj @ cam_pos.T).T  # (V, 4)
    w = clip[:, 3]
    safe_w = np.where(np.abs(w) > 1e-12, w, 1.0)
    ndc_x = clip[:, 0] / safe_w
    ndc_y = clip[:, 1] / safe_w
    u_pix = (ndc_x + 1.0) * 0.5 * width
    v_pix = (1.0 - ndc_y) * 0.5 * height
    depth = -cam_pos[:, 2]
    return u_pix, v_pix, depth


def vert_visibility_per_cam(
    vertices_world: np.ndarray,
    cam_world_from_cams: Sequence[np.ndarray],
    cam_projs: Sequence[np.ndarray],
    cam_resolutions: Sequence[tuple[int, int]],
    closer_surface_depths_per_cam: Sequence[np.ndarray],
    eps: float = 1e-3,
) -> np.ndarray:
    """Return a (V,) bool array: True for verts visible from at least
    one cam (in-frustum AND in-front-of-closer-surface), False for verts
    fully hidden (every cam either has them out-of-frustum or behind a
    closer surface).
    """
    V = vertices_world.shape[0]
    if V == 0:
        return np.empty(0, dtype=bool)

    visible_any = np.zeros(V, dtype=bool)
    for c in range(len(cam_world_from_cams)):
        W, H = cam_resolutions[c]
        u_pix, v_pix, depth = project_verts_to_cam(
            vertices_world,
            cam_world_from_cams[c],
            cam_projs[c],
            W,
            H,
        )
        in_frustum = (
            (u_pix >= 0)
            & (u_pix < W)
            & (v_pix >= 0)
            & (v_pix < H)
            & (depth > 0)
        )
        # Out-of-frustum verts can't be "visible from this cam" — they're
        # outside the cam's view. We sample closer-surface depth at
        # clamped pixel coords just to keep the array op valid; the
        # sampled value is masked out by `in_frustum` below so it doesn't
        # leak into the result.
        u_idx = np.clip(np.floor(u_pix).astype(np.int64), 0, W - 1)
        v_idx = np.clip(np.floor(v_pix).astype(np.int64), 0, H - 1)
        closer = closer_surface_depths_per_cam[c][v_idx, u_idx]
        in_front = depth < closer - eps
        visible_any |= in_frustum & in_front
    return visible_any


def cull_layer(
    polygons: Sequence[Sequence[int]],
    vertices_world: np.ndarray,
    cam_world_from_cams: Sequence[np.ndarray],
    cam_projs: Sequence[np.ndarray],
    cam_resolutions: Sequence[tuple[int, int]],
    closer_surface_depths_per_cam: Sequence[np.ndarray],
    eps: float = 1e-3,
) -> np.ndarray:
    """Per-polygon keep mask. Cull polygon p if EVERY vertex of p is
    fully hidden (not visible from any cam).

    `closer_surface_depths_per_cam[c]` is the per-cam (H, W) depth
    buffer of the closest covered surface across layers 0..N-1; should
    be +inf where no closer layer has coverage at that pixel.
    """
    if not polygons:
        return np.empty(0, dtype=bool)
    visible_any = vert_visibility_per_cam(
        vertices_world,
        cam_world_from_cams,
        cam_projs,
        cam_resolutions,
        closer_surface_depths_per_cam,
        eps,
    )
    fully_hidden = ~visible_any
    keep = np.empty(len(polygons), dtype=bool)
    for p, poly in enumerate(polygons):
        idx = np.fromiter(poly, dtype=np.int64, count=len(poly))
        # Polygon survives if at least one of its verts is not fully hidden.
        keep[p] = not bool(fully_hidden[idx].all())
    return keep
