"""Software depth rasteriser for occlusion-cull closer-surfaces.

Produces a per-camera, per-pixel cam-space depth buffer for a triangle
mesh. Used by occlusion culling to derive each closer layer's
contribution to the closer-surface from its opaque (interior-leaf) mesh
half. The opaque half excludes border quads by construction, so its
silhouette is the correct occluder — alpha-cut silhouettes (fences,
foliage) can't claim depth ownership of pixels they only sparsely cover.

Perspective-correct depth via 1/w interpolation. Triangles whose verts
straddle the near plane are skipped (no pixels claimed → +inf →
conservative under-cull at the slab boundary, fine for the cull goal).

Pure numpy / Python — no bpy. The operator that calls this is
responsible for assembling per-cam matrices and merging buffers across
closer layers.
"""

from __future__ import annotations

import numpy as np


def fan_triangulate_polygons(
    polygons: list[tuple[int, ...]],
) -> np.ndarray:
    """Fan-triangulate variable-length polygons into a (T, 3) int64
    array. Convex polygons (which leaf-derived quads and T-junction
    n-gons always are) tessellate correctly with a fan from vertex 0.
    """
    tris: list[tuple[int, int, int]] = []
    for poly in polygons:
        n = len(poly)
        if n < 3:
            continue
        v0 = int(poly[0])
        for i in range(1, n - 1):
            tris.append((v0, int(poly[i]), int(poly[i + 1])))
    if not tris:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(tris, dtype=np.int64)


def rasterise_depth(
    vertices_world: np.ndarray,
    triangles: np.ndarray,
    world_from_cam: np.ndarray,
    proj: np.ndarray,
    width: int,
    height: int,
    near_eps: float = 1e-4,
) -> np.ndarray:
    """Rasterise `triangles` (indices into `vertices_world`) from a
    single camera and return a (height, width) float32 depth buffer.

    Pixels with no covering triangle stay at +inf. Covered pixels hold
    cam-space depth (positive distance from cam along its -Z axis) of
    the nearest covering triangle — same convention as
    `core.occlusion.project_verts_to_cam` so the cull's per-vert
    comparisons line up.
    """
    depth_buf = np.full((height, width), np.inf, dtype=np.float32)
    if vertices_world.shape[0] == 0 or triangles.shape[0] == 0:
        return depth_buf

    V = vertices_world.shape[0]
    homog = np.concatenate(
        [vertices_world, np.ones((V, 1), dtype=np.float64)],
        axis=-1,
    )
    cam_from_world = np.linalg.inv(world_from_cam)
    cam_pos = (cam_from_world @ homog.T).T
    clip = (proj @ cam_pos.T).T
    w_clip = clip[:, 3]  # = -cam_z (positive for verts in front of cam)
    valid_w = w_clip > near_eps
    safe_w = np.where(valid_w, w_clip, 1.0)
    ndc_x = clip[:, 0] / safe_w
    ndc_y = clip[:, 1] / safe_w
    sx = (ndc_x + 1.0) * 0.5 * width
    sy = (1.0 - ndc_y) * 0.5 * height
    inv_w = np.where(valid_w, 1.0 / safe_w, 0.0)

    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        if not (valid_w[a] and valid_w[b] and valid_w[c]):
            continue
        x0, y0 = sx[a], sy[a]
        x1, y1 = sx[b], sy[b]
        x2, y2 = sx[c], sy[c]

        x_min_f = min(x0, x1, x2)
        x_max_f = max(x0, x1, x2)
        y_min_f = min(y0, y1, y2)
        y_max_f = max(y0, y1, y2)
        x_min = max(0, int(np.floor(x_min_f)))
        x_max = min(width - 1, int(np.ceil(x_max_f)) - 1)
        y_min = max(0, int(np.floor(y_min_f)))
        y_max = min(height - 1, int(np.ceil(y_max_f)) - 1)
        if x_min > x_max or y_min > y_max:
            continue

        # Doubled signed area; sign distinguishes screen-space CW vs CCW
        # winding. Bary = ratio of edge-function over area, so the sign
        # cancels and `w_i >= 0` selects the inside regardless of winding.
        area2 = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if abs(area2) < 1e-12:
            continue

        ys = (np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5).reshape(-1, 1)
        xs = (np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5).reshape(1, -1)

        w0 = ((x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)) / area2
        w1 = ((x0 - x2) * (ys - y2) - (y0 - y2) * (xs - x2)) / area2
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        if not inside.any():
            continue

        # Perspective-correct depth: 1/w_clip is linear in screen space,
        # so interpolate that, then invert. w_clip == cam-space depth.
        pix_inv_w = w0 * inv_w[a] + w1 * inv_w[b] + w2 * inv_w[c]
        safe_inv = np.where(pix_inv_w > 0, pix_inv_w, 1.0)
        pix_depth = np.where(pix_inv_w > 0, 1.0 / safe_inv, np.inf)

        sub = depth_buf[y_min : y_max + 1, x_min : x_max + 1]
        update = inside & (pix_depth < sub)
        sub_new = np.where(update, pix_depth.astype(np.float32), sub)
        depth_buf[y_min : y_max + 1, x_min : x_max + 1] = sub_new

    return depth_buf
