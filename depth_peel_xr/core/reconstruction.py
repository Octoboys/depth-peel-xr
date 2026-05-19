"""Build per-layer flat-impostor mesh arrays from quadtree leaves.

Phase 4 slice 1: vertices coplanar at `layer.near` depth in portal-local
space. Slice 2 will swap the flat positions for per-vertex inverse-
projected world coords driven by the canonical depth map.

Pure numpy / Python — no bpy. The operator that calls this is
responsible for hand-off to `bpy.types.Mesh`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .quadtree import QuadtreeLeaf


@dataclass(frozen=True)
class LayerMesh:
    """Plain-data mesh arrays for one layer's reconstruction.

    vertices: (N, 3) float64, portal-local coords.
    faces: (M, 4) int64, vertex indices per quad in TL→TR→BR→BL order
      (matches the centre-image pixel-corner traversal — same winding as
      slice-1 `Build Simple Planes`, normal pointing away from the head-
      plane cameras).
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
        # Pixel corners in TL, TR, BR, BL order (matches Build Simple Planes
        # winding so normals come out consistent).
        corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
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
