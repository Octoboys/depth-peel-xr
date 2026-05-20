"""Tests for core.reconstruction mesh builder."""

from __future__ import annotations

import numpy as np
import pytest

from core.geometry import inverse_canonical_projection
from core.quadtree import QuadtreeLeaf
from core.reconstruction import (
    build_displaced_layer_mesh,
    build_layer_mesh,
    filter_displaced_mesh,
    layer_mesh_to_displaced,
    split_displaced_layer_mesh,
    split_layer_mesh,
)


# Portal: 4 wide × 2 tall at headplane_offset = 2.
HALF_W_P = 2.0
HALF_H_P = 1.0
HO = 2.0


def test_empty_leaf_list_returns_empty_arrays():
    mesh = build_layer_mesh(
        leaves=[],
        image_width=64, image_height=64,
        half_w_plane=2.0, half_h_plane=1.0,
        headplane_offset=1.5, layer_near=3.5,
    )
    assert mesh.vertices.shape == (0, 3)
    assert mesh.faces.shape == (0, 4)
    assert mesh.uvs.shape == (0, 4, 2)


def test_single_full_image_leaf_spans_plane():
    # One interior leaf covering the whole 64x64 image.
    leaves = [QuadtreeLeaf(0, 0, 64, 64, "interior")]
    mesh = build_layer_mesh(
        leaves=leaves,
        image_width=64, image_height=64,
        half_w_plane=2.0, half_h_plane=1.0,
        headplane_offset=1.5, layer_near=3.5,
    )
    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.shape == (1, 4)

    # Y coord should be ho - near = 1.5 - 3.5 = -2.0 for every vertex.
    assert np.allclose(mesh.vertices[:, 1], -2.0)

    # Vertex positions in TL, BL, BR, TR order (pixel-CCW for +Y normal).
    # TL pixel (0, 0)   → portal (+half_w, y, +half_h)
    # BL pixel (0, 64)  → portal (+half_w, y, -half_h)
    # BR pixel (64, 64) → portal (-half_w, y, -half_h)
    # TR pixel (64, 0)  → portal (-half_w, y, +half_h)
    np.testing.assert_allclose(mesh.vertices[0], [+2.0, -2.0, +1.0])
    np.testing.assert_allclose(mesh.vertices[1], [+2.0, -2.0, -1.0])
    np.testing.assert_allclose(mesh.vertices[2], [-2.0, -2.0, -1.0])
    np.testing.assert_allclose(mesh.vertices[3], [-2.0, -2.0, +1.0])

    # UVs in same TL, BL, BR, TR order.
    np.testing.assert_allclose(mesh.uvs[0, 0], [0.0, 1.0])  # TL
    np.testing.assert_allclose(mesh.uvs[0, 1], [0.0, 0.0])  # BL
    np.testing.assert_allclose(mesh.uvs[0, 2], [1.0, 0.0])  # BR
    np.testing.assert_allclose(mesh.uvs[0, 3], [1.0, 1.0])  # TR


def test_non_shared_vertices_per_leaf():
    # Two adjacent leaves should NOT share corner verts — each gets 4
    # fresh vertices (separate per-leaf verts; no manifold needed).
    leaves = [
        QuadtreeLeaf(0, 0, 32, 32, "interior"),
        QuadtreeLeaf(32, 0, 32, 32, "interior"),
    ]
    mesh = build_layer_mesh(
        leaves=leaves,
        image_width=64, image_height=64,
        half_w_plane=1.0, half_h_plane=1.0,
        headplane_offset=1.5, layer_near=3.5,
    )
    assert mesh.vertices.shape == (8, 3)
    assert mesh.faces.shape == (2, 4)
    # Face 0's vertex indices are 0..3, face 1's are 4..7.
    np.testing.assert_array_equal(mesh.faces[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(mesh.faces[1], [4, 5, 6, 7])


def test_face_class_preserved_from_leaf():
    leaves = [
        QuadtreeLeaf(0, 0, 32, 64, "interior"),
        QuadtreeLeaf(32, 0, 32, 64, "border"),
    ]
    mesh = build_layer_mesh(
        leaves=leaves,
        image_width=64, image_height=64,
        half_w_plane=1.0, half_h_plane=1.0,
        headplane_offset=1.5, layer_near=3.5,
    )
    assert list(mesh.face_classes) == ["interior", "border"]


def test_face_winding_normal_points_plus_y():
    # Edges 0→1 and 0→3 of the first face, cross product → normal direction.
    leaves = [QuadtreeLeaf(0, 0, 64, 64, "interior")]
    mesh = build_layer_mesh(
        leaves=leaves,
        image_width=64, image_height=64,
        half_w_plane=2.0, half_h_plane=1.0,
        headplane_offset=1.5, layer_near=3.5,
    )
    v0, v1, v3 = mesh.vertices[0], mesh.vertices[1], mesh.vertices[3]
    e01 = v1 - v0
    e03 = v3 - v0
    normal = np.cross(e01, e03)
    # Should point in +Y direction (toward the head-plane viewer).
    assert normal[1] > 0


# ---- displaced mesh builder -------------------------------------------------

def _flat_depth(W: int, H: int, d: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant-depth canonical (fully covered) at canonical depth d."""
    depth = np.full((H, W), d, dtype=np.float32)
    coverage = np.ones((H, W), dtype=bool)
    return depth, coverage


def test_displaced_empty_leaf_list_returns_empty_arrays():
    depth, coverage = _flat_depth(16, 16, HO)
    mesh = build_displaced_layer_mesh(
        leaves=[],
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    assert mesh.vertices.shape == (0, 3)
    assert mesh.polygons == []
    assert mesh.poly_uvs.shape == (0, 2)


def test_displaced_single_leaf_at_constant_depth_emits_one_quad():
    leaves = [QuadtreeLeaf(0, 0, 16, 16, "interior")]
    depth, coverage = _flat_depth(16, 16, HO)  # d = HO → portal-local y = -HO/2
    mesh = build_displaced_layer_mesh(
        leaves=leaves,
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    # 4 unique corners → 4 verts; one quad polygon.
    assert mesh.vertices.shape == (4, 3)
    assert len(mesh.polygons) == 1
    assert len(mesh.polygons[0]) == 4
    assert all(cls == "interior" for cls in mesh.poly_classes)
    # All verts at the same y_portal = ho/2 - d = 1 - 2 = -1.
    np.testing.assert_allclose(mesh.vertices[:, 1], -1.0, atol=1e-9)


def test_displaced_two_same_size_leaves_share_edge_verts():
    """Two adjacent same-size leaves: the shared corners are deduplicated
    (6 unique verts, not 8)."""
    leaves = [
        QuadtreeLeaf(0, 0, 16, 16, "interior"),
        QuadtreeLeaf(16, 0, 16, 16, "interior"),
    ]
    depth, coverage = _flat_depth(32, 16, HO)
    mesh = build_displaced_layer_mesh(
        leaves=leaves,
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    assert mesh.vertices.shape == (6, 3)
    # 2 leaves → 2 quads (no T-junctions).
    assert len(mesh.polygons) == 2
    assert all(len(p) == 4 for p in mesh.polygons)


def test_displaced_t_junction_inserts_mid_edge_vert_into_big_leaf():
    """One big leaf next to two stacked small leaves on its right edge:
    the mid-edge vert (x=16, y=8) appears in BOTH leaves' polygons. Big
    leaf becomes a 5-vert n-gon; each small leaf is a 4-vert quad."""
    leaves = [
        QuadtreeLeaf(0, 0, 16, 16, "interior"),  # big left
        QuadtreeLeaf(16, 0, 16, 8, "interior"),  # small top-right
        QuadtreeLeaf(16, 8, 16, 8, "interior"),  # small bottom-right
    ]
    depth, coverage = _flat_depth(32, 16, HO)
    mesh = build_displaced_layer_mesh(
        leaves=leaves,
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    # Unique corners: (0,0), (16,0), (16,16), (0,16),
    #                 (32,0), (32,8), (32,16),
    #                 (16,8) ← T-junction vert.
    # = 8 verts.
    assert mesh.vertices.shape == (8, 3)
    # Polygons: big leaf is the 5-vert n-gon, two small leaves are quads.
    assert len(mesh.polygons) == 3
    poly_lens = sorted(len(p) for p in mesh.polygons)
    assert poly_lens == [4, 4, 5]


def test_displaced_t_junction_no_cracks_between_neighbours():
    """For a T-junction at constant depth, the mid-edge vertex sits on the
    same plane as the big leaf's edge — geometry is watertight (no z gap
    at the seam)."""
    leaves = [
        QuadtreeLeaf(0, 0, 16, 16, "interior"),
        QuadtreeLeaf(16, 0, 16, 8, "interior"),
        QuadtreeLeaf(16, 8, 16, 8, "interior"),
    ]
    depth, coverage = _flat_depth(32, 16, HO)
    mesh = build_displaced_layer_mesh(
        leaves=leaves,
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    # All verts at same y_portal (constant depth). No depth jump → no crack.
    np.testing.assert_allclose(mesh.vertices[:, 1], -1.0, atol=1e-9)


def test_displaced_per_vertex_depth_via_inverse_projection():
    """Vertex at canonical NDC (0, 0) at depth d → portal-local (0, ho/2-d, 0)."""
    leaves = [QuadtreeLeaf(0, 0, 16, 16, "interior")]
    # Depth = 2*HO at every pixel; corner-averaged depth is also 2*HO.
    depth = np.full((16, 16), 2.0 * HO, dtype=np.float32)
    coverage = np.ones((16, 16), dtype=bool)
    mesh = build_displaced_layer_mesh(
        leaves=leaves,
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    # Expected y_portal = HO/2 - 2*HO = -1.5*HO.
    np.testing.assert_allclose(mesh.vertices[:, 1], -1.5 * HO, atol=1e-9)
    # TL pixel corner (0, 0) → NDC (-1, +1) → portal +X, +Z corner.
    # x_portal = -(-1) * 2*HO * HALF_W_P * 2/HO = +4*HALF_W_P = 8.
    # z_portal = +1 * 2*HO * HALF_H_P * 2/HO = +4*HALF_H_P = 4.
    np.testing.assert_allclose(mesh.vertices[0, 0], 8.0, atol=1e-9)
    np.testing.assert_allclose(mesh.vertices[0, 2], 4.0, atol=1e-9)


def test_displaced_polygon_winding_normal_points_plus_y():
    """First three verts of the first polygon give +Y normal (pixel-CCW
    polygon walk → portal-local +Y face)."""
    leaves = [QuadtreeLeaf(0, 0, 16, 16, "interior")]
    depth, coverage = _flat_depth(16, 16, HO)
    mesh = build_displaced_layer_mesh(
        leaves=leaves,
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    poly = mesh.polygons[0]
    v0 = mesh.vertices[poly[0]]
    v1 = mesh.vertices[poly[1]]
    v2 = mesh.vertices[poly[2]]
    normal = np.cross(v1 - v0, v2 - v0)
    assert normal[1] > 0


def test_displaced_uvs_screen_space_passthrough():
    """Each polygon-loop UV equals (px/W, 1 - py/H) for that vert. For a
    quad in pixel-CCW order: TL (0, 1), BL (0, 0), BR (1, 0), TR (1, 1)."""
    leaves = [QuadtreeLeaf(0, 0, 16, 16, "interior")]
    depth, coverage = _flat_depth(16, 16, HO)
    mesh = build_displaced_layer_mesh(
        leaves=leaves,
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    np.testing.assert_allclose(mesh.poly_uvs[0], [0.0, 1.0], atol=1e-9)  # TL
    np.testing.assert_allclose(mesh.poly_uvs[1], [0.0, 0.0], atol=1e-9)  # BL
    np.testing.assert_allclose(mesh.poly_uvs[2], [1.0, 0.0], atol=1e-9)  # BR
    np.testing.assert_allclose(mesh.poly_uvs[3], [1.0, 1.0], atol=1e-9)  # TR


def test_displaced_invalid_corners_use_leaf_local_mean():
    """A border leaf with corners in fully-uncovered territory falls back
    to the average of the leaf's own valid corner depths — not the
    layer-wide mean — so silhouette verts match the rest of their quad."""
    H = W = 8
    # Two leaves with very different depths. The left leaf has all-
    # uncovered corners (no covered cells anywhere near it), while the
    # right leaf is fully covered at depth 2·HO. Without leaf-local
    # fallback the left-leaf corners would inherit the layer-wide mean
    # (= 2·HO since it's the only covered region) and then drift far
    # from the layer's actual scene structure on that side.
    leaves = [
        QuadtreeLeaf(0, 0, 4, 4, "border"),       # left, no coverage near
        QuadtreeLeaf(4, 0, 4, 4, "interior"),     # right, depth 2·HO
    ]
    coverage = np.zeros((H, W), dtype=bool)
    coverage[0:4, 4:8] = True  # right leaf fully covered
    depth = np.zeros((H, W), dtype=np.float32)
    depth[0:4, 4:8] = 2.0 * HO

    mesh = build_displaced_layer_mesh(
        leaves=leaves,
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    # Right leaf's 4 corners (px ∈ {4, 8}, py ∈ {0, 4}) all valid → 2·HO.
    # Left leaf's outer corners (px=0) have no covered neighbour even in
    # the dilated 4×4 stencil → invalid → leaf-local fallback to the
    # mean of the left leaf's valid corners. The shared-edge corners at
    # px=4 are valid (right leaf finds them at 2·HO via the 2×2/4×4
    # stencil), so the left leaf's local mean = 2·HO and ALL verts end
    # up at the same y_portal.
    expected_y = 0.5 * HO - 2.0 * HO  # = -3 with HO = 2
    np.testing.assert_allclose(mesh.vertices[:, 1], expected_y, atol=1e-9)


def test_displaced_border_leaf_corners_flatten_to_leaf_min():
    """Border-classified leaves snap their 4 corners to the leaf's own
    MIN-over-covered-cells depth — silhouette cards stay co-planar in
    3D even when per-corner stencils sample different parts of the
    scene's depth profile."""
    H = W = 16
    # Single border leaf with mixed coverage. The covered cells span
    # two depths (HO at TL quadrant, 2·HO at BR quadrant). Without the
    # border-flat snap, the four corners would sample different MIN-
    # stencil values (some HO, some 2·HO) → tilted quad.
    leaves = [QuadtreeLeaf(0, 0, 16, 16, "border")]
    coverage = np.zeros((H, W), dtype=bool)
    coverage[0:8, 0:8] = True
    coverage[8:16, 8:16] = True
    depth = np.zeros((H, W), dtype=np.float32)
    depth[0:8, 0:8] = HO
    depth[8:16, 8:16] = 2.0 * HO

    mesh = build_displaced_layer_mesh(
        leaves=leaves,
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    # MIN across all covered cells in the leaf = HO. All 4 corners snap.
    expected_y = 0.5 * HO - HO  # = -1
    np.testing.assert_allclose(mesh.vertices[:, 1], expected_y, atol=1e-9)


def test_displaced_border_pulls_shared_interior_corner_forward():
    """A corner shared between an interior leaf (deeper) and a border
    leaf (foreground) snaps to the border's flat depth — the user's
    "border quads stay flat" rule wins over the interior's stencil at
    the silhouette edge."""
    H = W = 16
    # Border leaf on the left at depth HO (foreground).
    # Interior leaf on the right at depth 2·HO (background).
    # They share the edge at x=8.
    leaves = [
        QuadtreeLeaf(0, 0, 8, 16, "border"),
        QuadtreeLeaf(8, 0, 8, 16, "interior"),
    ]
    coverage = np.ones((H, W), dtype=bool)
    depth = np.full((H, W), 2.0 * HO, dtype=np.float32)
    depth[:, :8] = HO

    mesh = build_displaced_layer_mesh(
        leaves=leaves,
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    fg_y = 0.5 * HO - HO         # = -1
    bg_y = 0.5 * HO - 2.0 * HO   # = -3
    ys = mesh.vertices[:, 1]
    # Border leaf corners (x in {0, 8}, y in {0, 16}) all at fg_y.
    # Interior corners at x=16 stay at bg_y. Shared corners at x=8
    # snapped to fg_y by the border-flatness pass (overriding the
    # interior's 2·HO MIN-stencil result).
    n_fg = int(np.isclose(ys, fg_y).sum())
    n_bg = int(np.isclose(ys, bg_y).sum())
    # 6 unique verts: x ∈ {0, 8, 16}, y ∈ {0, 16}. 4 at fg, 2 at bg.
    assert mesh.vertices.shape == (6, 3)
    assert n_fg == 4
    assert n_bg == 2


def test_displaced_min_over_depth_jump_picks_foreground():
    """At a within-coverage depth jump (fg|bg both covered with different
    depths), the shared corner takes MIN — foreground depth wins so
    silhouettes don't bleed backward toward the average."""
    H = W = 8
    leaves = [
        QuadtreeLeaf(0, 0, 4, 8, "interior"),  # left half (fg, depth = HO)
        QuadtreeLeaf(4, 0, 4, 8, "interior"),  # right half (bg, depth = 2·HO)
    ]
    coverage = np.ones((H, W), dtype=bool)
    depth = np.full((H, W), 2.0 * HO, dtype=np.float32)
    depth[:, :4] = HO
    mesh = build_displaced_layer_mesh(
        leaves=leaves,
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    # 6 unique corners: x ∈ {0, 4, 8}, y ∈ {0, 8}. Shared at x=4.
    assert mesh.vertices.shape == (6, 3)
    fg_y = 0.5 * HO - HO         # = -1
    bg_y = 0.5 * HO - 2.0 * HO   # = -3
    ys = mesh.vertices[:, 1]
    # Shared corners (x=4) take MIN(fg, bg) = fg → 4 verts at fg_y
    # (left-edge x=0 corners + shared x=4 corners), 2 verts at bg_y
    # (right-edge x=8 corners only).
    assert int(np.isclose(ys, fg_y).sum()) == 4
    assert int(np.isclose(ys, bg_y).sum()) == 2


def test_displaced_dilate_fallback_finds_far_neighbour():
    """A corner whose 2x2 stencil is fully uncovered but whose 4x4
    stencil has covered cells uses the dilated stencil (MIN over those
    cells) instead of falling back to layer-mean."""
    # Coverage occupies a single pixel in the bottom-right area; a
    # corner at (0, 0) finds nothing in 2x2, nothing in 4x4 → falls
    # back to layer-mean. A corner at (4, 4) finds the covered pixel in
    # its 4x4 (offsets 2..3, 2..3) → MIN = its depth.
    H = W = 8
    coverage = np.zeros((H, W), dtype=bool)
    coverage[6, 6] = True
    depth = np.zeros((H, W), dtype=np.float32)
    depth[6, 6] = HO  # single covered cell at depth HO

    leaves = [QuadtreeLeaf(4, 4, 4, 4, "border")]  # 4x4 leaf containing the cell
    mesh = build_displaced_layer_mesh(
        leaves=leaves,
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    # Leaf has 4 corners at (4,4), (4,8), (8,4), (8,8). All within reach
    # of the (6, 6) covered cell via the 4×4 stencil — every corner gets
    # depth = HO → y = ho/2 - HO = -1.
    assert mesh.vertices.shape == (4, 3)
    np.testing.assert_allclose(mesh.vertices[:, 1], -1.0, atol=1e-9)


def test_displaced_class_inherited_per_polygon():
    """Each polygon (one per leaf) carries that leaf's class."""
    leaves = [
        QuadtreeLeaf(0, 0, 8, 16, "interior"),
        QuadtreeLeaf(8, 0, 8, 16, "border"),
    ]
    depth, coverage = _flat_depth(16, 16, HO)
    mesh = build_displaced_layer_mesh(
        leaves=leaves,
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    classes = list(mesh.poly_classes)
    assert classes.count("interior") == 1
    assert classes.count("border") == 1


def test_displaced_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        build_displaced_layer_mesh(
            leaves=[QuadtreeLeaf(0, 0, 4, 4, "interior")],
            canonical_depth=np.zeros((4, 4), dtype=np.float32),
            canonical_coverage=np.ones((8, 8), dtype=bool),
            half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
        )


# ---- filter_displaced_mesh --------------------------------------------------

def _build_two_quad_mesh() -> object:
    """Two adjacent quads sharing the centre edge → 6 unique verts, 2 polygons."""
    leaves = [
        QuadtreeLeaf(0, 0, 16, 16, "interior"),
        QuadtreeLeaf(16, 0, 16, 16, "border"),
    ]
    depth, coverage = _flat_depth(32, 16, HO)
    return build_displaced_layer_mesh(
        leaves=leaves,
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )


def test_filter_keeps_all_returns_input_unchanged():
    mesh = _build_two_quad_mesh()
    out = filter_displaced_mesh(mesh, np.array([True, True]))
    assert out is mesh  # unchanged when nothing to drop


def test_filter_drops_all_returns_empty():
    mesh = _build_two_quad_mesh()
    out = filter_displaced_mesh(mesh, np.array([False, False]))
    assert out.vertices.shape == (0, 3)
    assert out.polygons == []
    assert out.poly_uvs.shape == (0, 2)


def test_filter_drops_one_polygon_compacts_verts():
    """Keeping only the left quad drops the two right-edge verts and
    remaps the kept polygon's indices to the compacted layout."""
    mesh = _build_two_quad_mesh()
    out = filter_displaced_mesh(mesh, np.array([True, False]))
    # Left quad referenced 4 of 6 unique verts; the 2 verts unique to
    # the right quad get dropped.
    assert out.vertices.shape == (4, 3)
    assert len(out.polygons) == 1
    assert len(out.polygons[0]) == 4
    # All kept-poly indices must point into the compacted vertex array.
    assert all(0 <= int(vi) < 4 for vi in out.polygons[0])
    # Class preserved.
    assert out.poly_classes[0] == "interior"
    # Per-loop UV slice preserved (4 loops for the kept quad).
    assert out.poly_uvs.shape == (4, 2)


def test_filter_keeps_loop_uvs_aligned_with_polygons():
    """When a polygon survives, its loops' UVs should match the
    original loop slice for that polygon."""
    mesh = _build_two_quad_mesh()
    # Keep only the second polygon.
    out = filter_displaced_mesh(mesh, np.array([False, True]))
    poly0_len = len(mesh.polygons[0])
    expected_uvs = mesh.poly_uvs[poly0_len : poly0_len + len(mesh.polygons[1])]
    np.testing.assert_allclose(out.poly_uvs, expected_uvs)


def test_filter_rejects_mask_length_mismatch():
    mesh = _build_two_quad_mesh()
    with pytest.raises(ValueError):
        filter_displaced_mesh(mesh, np.array([True, True, True]))


# ---- split helpers ----------------------------------------------------------

def test_split_layer_mesh_partitions_by_class():
    leaves = [
        QuadtreeLeaf(0, 0, 16, 16, "interior"),
        QuadtreeLeaf(16, 0, 16, 16, "border"),
        QuadtreeLeaf(0, 16, 16, 16, "interior"),
    ]
    mesh = build_layer_mesh(
        leaves=leaves,
        image_width=32, image_height=32,
        half_w_plane=2.0, half_h_plane=2.0,
        headplane_offset=1.5, layer_near=3.5,
    )
    interior, border = split_layer_mesh(mesh)
    assert interior.faces.shape == (2, 4)
    assert border.faces.shape == (1, 4)
    assert all(c == "interior" for c in interior.face_classes)
    assert all(c == "border" for c in border.face_classes)
    # Each half has exactly the verts its faces reference (4 fresh per
    # leaf, non-shared) — interior=8, border=4.
    assert interior.vertices.shape == (8, 3)
    assert border.vertices.shape == (4, 3)
    # All face indices land in their compacted vertex array.
    assert int(interior.faces.max()) < interior.vertices.shape[0]
    assert int(border.faces.max()) < border.vertices.shape[0]


def test_split_layer_mesh_empty_returns_empty_pair():
    mesh = build_layer_mesh(
        leaves=[],
        image_width=16, image_height=16,
        half_w_plane=1.0, half_h_plane=1.0,
        headplane_offset=1.0, layer_near=2.0,
    )
    a, b = split_layer_mesh(mesh)
    assert a.faces.shape == (0, 4)
    assert b.faces.shape == (0, 4)


def test_split_layer_mesh_all_one_class_leaves_other_empty():
    leaves = [QuadtreeLeaf(0, 0, 16, 16, "interior")]
    mesh = build_layer_mesh(
        leaves=leaves,
        image_width=16, image_height=16,
        half_w_plane=1.0, half_h_plane=1.0,
        headplane_offset=1.5, layer_near=3.5,
    )
    interior, border = split_layer_mesh(mesh)
    assert interior.faces.shape == (1, 4)
    assert border.faces.shape == (0, 4)
    assert border.vertices.shape == (0, 3)


def test_split_displaced_partitions_by_poly_class():
    """Two adjacent leaves (interior + border) split into two halves
    with one polygon each. The shared edge corners appear in BOTH halves
    (same world position — border-flatness already snapped them)."""
    mesh = _build_two_quad_mesh()  # left=interior, right=border, shared edge at x=16
    interior, border = split_displaced_layer_mesh(mesh)
    assert len(interior.polygons) == 1
    assert len(border.polygons) == 1
    assert interior.poly_classes[0] == "interior"
    assert border.poly_classes[0] == "border"
    # 4 verts per quad after compaction (each half's polygon uses
    # exactly its 4 corners, including the shared-edge ones).
    assert interior.vertices.shape == (4, 3)
    assert border.vertices.shape == (4, 3)
    # Shared verts coincide geometrically — find the two vert positions
    # in each half that match between the two halves.
    int_set = {tuple(np.round(v, 9)) for v in interior.vertices}
    bord_set = {tuple(np.round(v, 9)) for v in border.vertices}
    overlap = int_set & bord_set
    # 2 shared corners on the x=16 edge.
    assert len(overlap) == 2


def test_split_displaced_empty_polys_returns_empty_pair():
    depth, coverage = _flat_depth(8, 8, HO)
    empty = build_displaced_layer_mesh(
        leaves=[],
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    a, b = split_displaced_layer_mesh(empty)
    assert a.polygons == []
    assert b.polygons == []


def test_split_displaced_all_one_class_other_half_empty():
    leaves = [
        QuadtreeLeaf(0, 0, 16, 16, "interior"),
        QuadtreeLeaf(16, 0, 16, 16, "interior"),
    ]
    depth, coverage = _flat_depth(32, 16, HO)
    mesh = build_displaced_layer_mesh(
        leaves=leaves,
        canonical_depth=depth,
        canonical_coverage=coverage,
        half_w_p=HALF_W_P, half_h_p=HALF_H_P, headplane_offset=HO,
    )
    interior, border = split_displaced_layer_mesh(mesh)
    assert len(interior.polygons) == 2
    assert len(border.polygons) == 0
    assert border.vertices.shape == (0, 3)


# ---- layer_mesh_to_displaced ------------------------------------------------

def test_layer_mesh_to_displaced_round_trip_preserves_geometry():
    """Repackaging a LayerMesh as DisplacedLayerMesh keeps verts and
    per-quad polygon indices intact, and flattens UVs to per-loop layout."""
    leaves = [
        QuadtreeLeaf(0, 0, 16, 16, "interior"),
        QuadtreeLeaf(16, 0, 16, 16, "border"),
    ]
    flat = build_layer_mesh(
        leaves=leaves,
        image_width=32, image_height=16,
        half_w_plane=2.0, half_h_plane=1.0,
        headplane_offset=1.5, layer_near=3.5,
    )
    disp = layer_mesh_to_displaced(flat)
    np.testing.assert_array_equal(disp.vertices, flat.vertices)
    assert len(disp.polygons) == flat.faces.shape[0]
    assert disp.polygons[0] == tuple(int(i) for i in flat.faces[0])
    assert disp.polygons[1] == tuple(int(i) for i in flat.faces[1])
    # poly_uvs flattens (M, 4, 2) → (M·4, 2). First 4 rows = first quad's loops.
    np.testing.assert_array_equal(disp.poly_uvs[:4], flat.uvs[0])
    np.testing.assert_array_equal(disp.poly_uvs[4:8], flat.uvs[1])
    assert list(disp.poly_classes) == ["interior", "border"]


def test_layer_mesh_to_displaced_empty():
    empty_flat = build_layer_mesh(
        leaves=[],
        image_width=8, image_height=8,
        half_w_plane=1.0, half_h_plane=1.0,
        headplane_offset=1.0, layer_near=2.0,
    )
    disp = layer_mesh_to_displaced(empty_flat)
    assert disp.vertices.shape == (0, 3)
    assert disp.polygons == []
    assert disp.poly_uvs.shape == (0, 2)
