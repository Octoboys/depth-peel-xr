"""Build per-layer mesh arrays from quadtree leaves.

Slice 1 (`build_layer_mesh`): coplanar verts at `layer.near` in portal-
local space, non-shared verts (4 fresh per leaf), no displacement.

Slice 2 (`build_displaced_layer_mesh`): per-vertex inverse-projected world
coords driven by canonical depth, shared verts hashed by canonical pixel
coord, T-junctions handled via fan triangulation that includes mid-edge
verts owned by smaller neighbours. Output is triangles (variable count
per leaf, pre-triangulated) instead of quads.

Pure numpy / Python — no bpy. The operator that calls this is
responsible for hand-off to `bpy.types.Mesh`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import inverse_canonical_projection
from .quadtree import QuadtreeLeaf


@dataclass(frozen=True)
class LayerMesh:
    """Plain-data mesh arrays for one layer's reconstruction.

    vertices: (N, 3) float64, portal-local coords. The operator parents
      the resulting object to the portal with identity parent_inverse,
      so portal-local = mesh-local — rotating the portal in the scene
      transforms the mesh with it.
    faces: (M, 4) int64, vertex indices per quad in TL→BL→BR→TR order
      (pixel-CCW), so the face normal points portal-local +Y — the rig
      always places the head-plane at portal-local +Y, so this points
      at the viewer regardless of portal world orientation.
    uvs: (M, 4, 2) float64, per-loop UV. Screen-space passthrough from
      the canonical centre image.
    face_classes: (M,) object array of 'interior' / 'border' strings.

    Vertices are NOT shared between adjacent leaves (4 fresh verts per
    quad). Slice-1 flat impostors don't need a manifold mesh — sharing
    would force per-vertex class disambiguation when adjacent leaves
    differ in interior/border.
    """
    vertices: np.ndarray
    faces: np.ndarray
    uvs: np.ndarray
    face_classes: np.ndarray


@dataclass(frozen=True)
class DisplacedLayerMesh:
    """Plain-data mesh arrays for slice-2 displaced reconstruction.

    vertices: (N, 3) float64, portal-local. One row per unique canonical
      pixel-corner across all leaves (verts shared between adjacent
      leaves and T-junction-mid-edges).
    triangles: (M, 3) int64, per-tri vertex indices. Variable count per
      leaf because fan triangulation through mid-edge verts produces a
      different number of tris depending on T-junction density.
    triangle_uvs: (M, 3, 2) float64, per-loop UV in canonical pixel-
      coord space.
    triangle_classes: (M,) object array of 'interior' / 'border'. Per-
      triangle, inherited from the leaf that emitted the tri.
    """
    vertices: np.ndarray
    triangles: np.ndarray
    triangle_uvs: np.ndarray
    triangle_classes: np.ndarray


def build_layer_mesh(
    leaves: Sequence[QuadtreeLeaf],
    image_width: int,
    image_height: int,
    half_w_plane: float,
    half_h_plane: float,
    headplane_offset: float,
    layer_near: float,
) -> LayerMesh:
    """Emit mesh arrays for a single layer.

    `half_w_plane` / `half_h_plane` are the world-space half-extents of the
    reconstructed plane at `layer_near` depth. Slice 1 sizes the plane to
    the super-frustum cross-section at that depth (see
    `super_frustum_half_extents` in core.geometry) — wide enough that off-
    centre XR viewers see no angular gap along the portal interior.
    """
    n_leaves = len(leaves)
    if n_leaves == 0:
        return LayerMesh(
            vertices=np.empty((0, 3), dtype=np.float64),
            faces=np.empty((0, 4), dtype=np.int64),
            uvs=np.empty((0, 4, 2), dtype=np.float64),
            face_classes=np.empty(0, dtype=object),
        )

    inv_w = 1.0 / image_width
    inv_h = 1.0 / image_height
    y_local = headplane_offset - layer_near

    vertices = np.empty((n_leaves * 4, 3), dtype=np.float64)
    faces = np.empty((n_leaves, 4), dtype=np.int64)
    uvs = np.empty((n_leaves, 4, 2), dtype=np.float64)
    face_classes = np.empty(n_leaves, dtype=object)

    for fi, leaf in enumerate(leaves):
        x0, y0 = leaf.x, leaf.y
        x1, y1 = leaf.x + leaf.w, leaf.y + leaf.h
        # Pixel corners in TL, BL, BR, TR order (pixel-CCW) so the face
        # normal points portal-local +Y — the rig places the head-plane
        # at portal-local +Y, so this faces the viewer regardless of the
        # portal's world orientation.
        corners = ((x0, y0), (x0, y1), (x1, y1), (x1, y0))
        for ci, (px, py) in enumerate(corners):
            # Pixel → portal-local. Image right = portal -X; image top = portal +Z.
            wx = half_w_plane * (1.0 - 2.0 * px * inv_w)
            wz = half_h_plane * (1.0 - 2.0 * py * inv_h)
            vi = fi * 4 + ci
            vertices[vi] = (wx, y_local, wz)
            faces[fi, ci] = vi
            uvs[fi, ci] = (px * inv_w, 1.0 - py * inv_h)
        face_classes[fi] = leaf.cls

    return LayerMesh(
        vertices=vertices, faces=faces, uvs=uvs, face_classes=face_classes
    )


def _compute_corner_depths(
    canonical_depth: np.ndarray, canonical_coverage: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (corner_depth, corner_valid), both shape (H+1, W+1).

    Each corner takes the MIN depth over the up-to-4 covered cells in
    its 2×2 stencil — Z-buffer convention. MIN beats MEAN at silhouettes
    and within-coverage depth jumps because mean averaging blends fg
    and bg, leaving the corner stranded at a fictitious mid-depth where
    bridge tris fold the wrong way.

    Dilate-once fallback: if a corner has no covered cell in its 2×2
    stencil (rare but possible on thin features or at AA bands the
    coverage threshold dropped), expand to a 4×4 stencil before giving
    up. Only fully-uncovered 4×4 windows mark `corner_valid = False`,
    in which case the caller falls back to the layer-mean depth.
    """
    H, W = canonical_depth.shape
    pad_d = np.zeros((H + 4, W + 4), dtype=np.float64)
    pad_c = np.zeros((H + 4, W + 4), dtype=bool)
    pad_d[2 : H + 2, 2 : W + 2] = canonical_depth
    pad_c[2 : H + 2, 2 : W + 2] = canonical_coverage
    # Uncovered cells masked to +inf so MIN ignores them.
    masked = np.where(pad_c, pad_d, np.inf)

    # 2×2 stencil: cells at (r-1+s, c-1+t) for s, t in {0, 1}; in padded
    # coords with offset +2 that's pad_d[1+s : 1+s+H+1, 1+t : 1+t+W+1].
    inner = np.minimum.reduce([
        masked[1 + s : 1 + s + H + 1, 1 + t : 1 + t + W + 1]
        for s in range(2)
        for t in range(2)
    ])

    if not np.isfinite(inner).all():
        # 4×4 stencil: cells at (r-2+s, c-2+t) for s, t in {0..3}.
        outer = np.minimum.reduce([
            masked[s : s + H + 1, t : t + W + 1]
            for s in range(4)
            for t in range(4)
        ])
        corner_depth = np.where(np.isfinite(inner), inner, outer)
    else:
        corner_depth = inner

    corner_valid = np.isfinite(corner_depth)
    # Replace any remaining +inf with 0 so the caller's `np.where(valid,
    # depth, layer_mean)` doesn't propagate inf.
    corner_depth = np.where(corner_valid, corner_depth, 0.0)
    return corner_depth, corner_valid


def build_displaced_layer_mesh(
    leaves: Sequence[QuadtreeLeaf],
    canonical_depth: np.ndarray,
    canonical_coverage: np.ndarray,
    half_w_p: float,
    half_h_p: float,
    headplane_offset: float,
) -> DisplacedLayerMesh:
    """Slice-2 mesh builder. Produces a manifold mesh with shared corner
    verts; T-junctions between leaves of different sizes are stitched by
    fan-triangulating each leaf's polygon through any mid-edge verts owned
    by smaller neighbours.

    `canonical_depth` is the consolidated depth (canonical-cam-space, =
    distance from canonical apex along its -Z axis); `canonical_coverage`
    is the boolean mask of pixels actually filled by capture+consolidate.

    Vertex world position is computed by `inverse_canonical_projection`
    using a per-corner depth that takes MIN over covered cells in a 2×2
    stencil (Z-buffer convention — closer surface wins at silhouettes
    and within-coverage depth jumps). Dilates to 4×4 if the inner
    stencil has no covered neighbour.

    Three-tier fallback for corners that still come out invalid:
      1. Average of the leaf's *own* valid corner depths (per the user's
         "match the rest of the quad" rule). For shared corners owned by
         multiple leaves, average those leaves' per-leaf averages.
      2. Layer-mean depth across all covered pixels — only when every
         leaf owning the invalid corner has all-invalid corners itself.
      3. Zero (apex) — only when the entire layer has no coverage at all.

    Final pass — **border-quad flatness**: every border-classified
    leaf's 4 corners snap to the leaf's MIN over its own covered cells
    (the closest visible foreground surface in the leaf's pixel region).
    Multiple border leaves owning a shared corner resolve via MIN. This
    keeps silhouette cards co-planar in 3D regardless of stencil noise
    across corners (which is otherwise common in the sparse super-
    frustum annulus where each canonical pixel comes from a different
    sibling forward-splat). Corners shared between border and interior
    leaves get pulled to the border's foreground depth too — small loss
    of interior stencil detail right at the silhouette edge in exchange
    for visually flat silhouette quads from any XR viewpoint.
    """
    if canonical_depth.shape != canonical_coverage.shape:
        raise ValueError(
            f"depth / coverage shape mismatch: "
            f"{canonical_depth.shape} vs {canonical_coverage.shape}"
        )
    H, W = canonical_depth.shape
    if not leaves:
        return DisplacedLayerMesh(
            vertices=np.empty((0, 3), dtype=np.float64),
            triangles=np.empty((0, 3), dtype=np.int64),
            triangle_uvs=np.empty((0, 3, 2), dtype=np.float64),
            triangle_classes=np.empty(0, dtype=object),
        )

    corner_depth, corner_valid = _compute_corner_depths(
        canonical_depth, canonical_coverage
    )

    # Pass 1: hash leaf corners + remember each leaf's 4 corner indices
    # so we can do per-leaf-local fallback later. Vectorised via
    # np.unique to avoid the per-leaf Python dict cost on large layers.
    n_leaves = len(leaves)
    leaf_x = np.fromiter((leaf.x for leaf in leaves), dtype=np.int64, count=n_leaves)
    leaf_y = np.fromiter((leaf.y for leaf in leaves), dtype=np.int64, count=n_leaves)
    leaf_w = np.fromiter((leaf.w for leaf in leaves), dtype=np.int64, count=n_leaves)
    leaf_h = np.fromiter((leaf.h for leaf in leaves), dtype=np.int64, count=n_leaves)
    leaf_x1 = leaf_x + leaf_w
    leaf_y1 = leaf_y + leaf_h

    # (n_leaves, 4, 2) corner pixel coords in TL, BL, BR, TR order.
    corners = np.empty((n_leaves, 4, 2), dtype=np.int64)
    corners[:, 0, 0] = leaf_x;  corners[:, 0, 1] = leaf_y       # TL
    corners[:, 1, 0] = leaf_x;  corners[:, 1, 1] = leaf_y1      # BL
    corners[:, 2, 0] = leaf_x1; corners[:, 2, 1] = leaf_y1      # BR
    corners[:, 3, 0] = leaf_x1; corners[:, 3, 1] = leaf_y       # TR

    flat_corners = corners.reshape(-1, 2)
    unique_corners, inverse = np.unique(
        flat_corners, axis=0, return_inverse=True
    )
    pixel_coords = unique_corners  # (n_unique, 2) — same role as before
    # leaf_corner_indices[li] = (idx_TL, idx_BL, idx_BR, idx_TR).
    leaf_corner_indices_arr = inverse.reshape(n_leaves, 4)
    leaf_corner_indices: list[tuple[int, int, int, int]] = [
        tuple(int(i) for i in row) for row in leaf_corner_indices_arr
    ]

    # corner_idx maps (px, py) → unique vert idx. Pass 2 uses it for
    # mid-edge T-junction lookups.
    corner_idx: dict[tuple[int, int], int] = {
        (int(px), int(py)): i
        for i, (px, py) in enumerate(unique_corners)
    }

    # xs_at_y / ys_at_x: sorted lists of x-coords at each y row (and
    # vice-versa). Built by group-by on the unique-corner array.
    unique_xs = unique_corners[:, 0]
    unique_ys = unique_corners[:, 1]

    xs_at_y: dict[int, list[int]] = defaultdict(list)
    sort_xy = np.lexsort((unique_xs, unique_ys))
    sorted_x_by_y = unique_xs[sort_xy]
    sorted_y = unique_ys[sort_xy]
    if sorted_y.size:
        y_vals, y_starts = np.unique(sorted_y, return_index=True)
        for i, y_val in enumerate(y_vals):
            end = y_starts[i + 1] if i + 1 < len(y_vals) else len(sorted_x_by_y)
            xs_at_y[int(y_val)] = sorted_x_by_y[y_starts[i] : end].tolist()

    ys_at_x: dict[int, list[int]] = defaultdict(list)
    sort_yx = np.lexsort((unique_ys, unique_xs))
    sorted_y_by_x = unique_ys[sort_yx]
    sorted_x = unique_xs[sort_yx]
    if sorted_x.size:
        x_vals, x_starts = np.unique(sorted_x, return_index=True)
        for i, x_val in enumerate(x_vals):
            end = x_starts[i + 1] if i + 1 < len(x_vals) else len(sorted_y_by_x)
            ys_at_x[int(x_val)] = sorted_y_by_x[x_starts[i] : end].tolist()

    # Per-vert depth from the canonical's corner grid.
    pix = np.asarray(pixel_coords, dtype=np.int64)
    pxs = pix[:, 0]
    pys = pix[:, 1]
    py_idx = np.clip(pys, 0, H)
    px_idx = np.clip(pxs, 0, W)
    vert_depth = corner_depth[py_idx, px_idx].astype(np.float64, copy=True)
    vert_valid = corner_valid[py_idx, px_idx].copy()

    # Per-leaf-local fallback for invalid verts.
    if not vert_valid.all():
        if canonical_coverage.any():
            layer_mean_depth = float(canonical_depth[canonical_coverage].mean())
        else:
            layer_mean_depth = 0.0

        # Per-leaf mean over its own valid corners (NaN if leaf has none).
        n_leaves = len(leaves)
        leaf_means = np.full(n_leaves, np.nan, dtype=np.float64)
        for li, indices in enumerate(leaf_corner_indices):
            valid_d = [vert_depth[i] for i in indices if vert_valid[i]]
            if valid_d:
                leaf_means[li] = sum(valid_d) / len(valid_d)

        # Inverse map: for each vert, which leaves own it.
        vert_to_leaves: dict[int, list[int]] = defaultdict(list)
        for li, indices in enumerate(leaf_corner_indices):
            for vi in indices:
                vert_to_leaves[vi].append(li)

        for vi in np.flatnonzero(~vert_valid):
            owner_means = [
                leaf_means[li]
                for li in vert_to_leaves[int(vi)]
                if not np.isnan(leaf_means[li])
            ]
            if owner_means:
                vert_depth[vi] = sum(owner_means) / len(owner_means)
            else:
                vert_depth[vi] = layer_mean_depth

    # Border-quad flatness pass: collect per-border-leaf MIN-over-covered
    # depths first, then resolve shared corners via MIN across owning
    # border leaves before overriding `vert_depth` unconditionally. The
    # unconditional override means a border-interior shared corner snaps
    # to the border's foreground depth — which is what keeps silhouette
    # cards co-planar.
    border_target: dict[int, float] = {}
    for li, leaf in enumerate(leaves):
        if leaf.cls != "border":
            continue
        region_d = canonical_depth[leaf.y : leaf.y + leaf.h, leaf.x : leaf.x + leaf.w]
        region_c = canonical_coverage[leaf.y : leaf.y + leaf.h, leaf.x : leaf.x + leaf.w]
        if not region_c.any():
            continue
        flat_d = float(region_d[region_c].min())
        for vi in leaf_corner_indices[li]:
            existing = border_target.get(vi)
            border_target[vi] = flat_d if existing is None else min(existing, flat_d)
    for vi, d in border_target.items():
        vert_depth[vi] = d

    # Pixel-corner NDC (no +0.5 — corners sit at cell boundaries):
    # u = 2·px/W − 1 (left edge -1, right +1); v = 1 − 2·py/H (top +1, bottom -1).
    u = 2.0 * pxs / W - 1.0
    v = 1.0 - 2.0 * pys / H
    x_p, y_p, z_p = inverse_canonical_projection(
        u, v, vert_depth, half_w_p, half_h_p, headplane_offset
    )
    vertices = np.stack([x_p, y_p, z_p], axis=-1).astype(np.float64)

    # Pass 2: triangulate. Split into a fast path (no T-junctions) that
    # runs fully vectorised in numpy, and a slow path (T-junction leaves)
    # that keeps the per-leaf Python polygon walk.
    inv_w = 1.0 / W
    inv_h = 1.0 / H
    # Per-vert UV table — indexed once and reused for every triangle.
    vert_uvs = np.empty((pixel_coords.shape[0], 2), dtype=np.float64)
    vert_uvs[:, 0] = pixel_coords[:, 0] * inv_w
    vert_uvs[:, 1] = 1.0 - pixel_coords[:, 1] * inv_h

    # T-junction detection via cumulative-sum range queries on the
    # has-vert grid. has_vert[y, x] = True where any leaf has a corner.
    has_vert = np.zeros((H + 1, W + 1), dtype=bool)
    has_vert[unique_ys, unique_xs] = True
    cumsum_x = np.cumsum(has_vert, axis=1, dtype=np.int32)
    cumsum_y = np.cumsum(has_vert, axis=0, dtype=np.int32)
    # Mid-edge vert count on each leaf edge — cumsum from x0 (exclusive)
    # to x1 (exclusive) along its row, etc. Endpoint corners always
    # contribute, so subtract them out of the range.
    top_extras = (
        cumsum_x[leaf_y, leaf_x1 - 1] - cumsum_x[leaf_y, leaf_x]
    )
    bot_extras = (
        cumsum_x[leaf_y1, leaf_x1 - 1] - cumsum_x[leaf_y1, leaf_x]
    )
    left_extras = (
        cumsum_y[leaf_y1 - 1, leaf_x] - cumsum_y[leaf_y, leaf_x]
    )
    right_extras = (
        cumsum_y[leaf_y1 - 1, leaf_x1] - cumsum_y[leaf_y, leaf_x1]
    )
    has_tj = (top_extras + bot_extras + left_extras + right_extras) > 0

    # ---- Fast path: leaves without T-junctions emit exactly 2 tris each.
    leaf_classes_arr = np.array(
        [leaf.cls for leaf in leaves], dtype=object
    )
    fast_idx = np.flatnonzero(~has_tj)
    n_fast = fast_idx.size
    if n_fast:
        fast_corners = leaf_corner_indices_arr[fast_idx]  # (n_fast, 4) [TL, BL, BR, TR]
        fast_tris = np.empty((n_fast * 2, 3), dtype=np.int64)
        # Tri 0 per leaf: TL → BL → BR.
        fast_tris[0::2, 0] = fast_corners[:, 0]
        fast_tris[0::2, 1] = fast_corners[:, 1]
        fast_tris[0::2, 2] = fast_corners[:, 2]
        # Tri 1 per leaf: TL → BR → TR.
        fast_tris[1::2, 0] = fast_corners[:, 0]
        fast_tris[1::2, 1] = fast_corners[:, 2]
        fast_tris[1::2, 2] = fast_corners[:, 3]
        fast_tri_uvs = vert_uvs[fast_tris]  # (n_fast * 2, 3, 2)
        fast_classes = np.repeat(leaf_classes_arr[fast_idx], 2)
    else:
        fast_tris = np.empty((0, 3), dtype=np.int64)
        fast_tri_uvs = np.empty((0, 3, 2), dtype=np.float64)
        fast_classes = np.empty(0, dtype=object)

    # ---- Slow path: leaves with T-junctions walk the polygon and fan-tri.
    slow_tris: list[tuple[int, int, int]] = []
    slow_uvs: list[
        tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    ] = []
    slow_classes: list[str] = []
    slow_idx = np.flatnonzero(has_tj)
    for li in slow_idx:
        leaf = leaves[int(li)]
        x0, y0 = leaf.x, leaf.y
        x1, y1 = leaf.x + leaf.w, leaf.y + leaf.h
        # Polygon vert indices in pixel-CCW order: TL → left-extras (T→B)
        # → BL → bottom-extras (L→R) → BR → right-extras (B→T) → TR →
        # top-extras (R→L). Pixel-CCW makes the fan normal point portal-
        # local +Y.
        poly_idx: list[int] = [int(leaf_corner_indices_arr[li, 0])]  # TL
        for py in ys_at_x.get(x0, ()):
            if y0 < py < y1:
                poly_idx.append(corner_idx[(x0, int(py))])
        poly_idx.append(int(leaf_corner_indices_arr[li, 1]))  # BL
        for px in xs_at_y.get(y1, ()):
            if x0 < px < x1:
                poly_idx.append(corner_idx[(int(px), y1)])
        poly_idx.append(int(leaf_corner_indices_arr[li, 2]))  # BR
        for py in reversed(ys_at_x.get(x1, ())):
            if y0 < py < y1:
                poly_idx.append(corner_idx[(x1, int(py))])
        poly_idx.append(int(leaf_corner_indices_arr[li, 3]))  # TR
        for px in reversed(xs_at_y.get(y0, ())):
            if x0 < px < x1:
                poly_idx.append(corner_idx[(int(px), y0)])

        v0 = poly_idx[0]
        v0_uv = (float(vert_uvs[v0, 0]), float(vert_uvs[v0, 1]))
        cls = leaf.cls
        for i in range(1, len(poly_idx) - 1):
            v1 = poly_idx[i]
            v2 = poly_idx[i + 1]
            slow_tris.append((v0, v1, v2))
            slow_uvs.append((
                v0_uv,
                (float(vert_uvs[v1, 0]), float(vert_uvs[v1, 1])),
                (float(vert_uvs[v2, 0]), float(vert_uvs[v2, 1])),
            ))
            slow_classes.append(cls)

    if slow_tris:
        slow_tris_arr = np.asarray(slow_tris, dtype=np.int64)
        slow_uvs_arr = np.asarray(slow_uvs, dtype=np.float64)
        slow_classes_arr = np.asarray(slow_classes, dtype=object)
        triangles = np.concatenate([fast_tris, slow_tris_arr])
        triangle_uvs = np.concatenate([fast_tri_uvs, slow_uvs_arr])
        triangle_classes = np.concatenate([fast_classes, slow_classes_arr])
    else:
        triangles = fast_tris
        triangle_uvs = fast_tri_uvs
        triangle_classes = fast_classes

    return DisplacedLayerMesh(
        vertices=vertices,
        triangles=triangles,
        triangle_uvs=triangle_uvs,
        triangle_classes=triangle_classes,
    )
