"""Tests for core.reconstruction mesh builder."""

from __future__ import annotations

import numpy as np
import pytest

from core.quadtree import QuadtreeLeaf
from core.reconstruction import build_layer_mesh


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

    # Vertex positions in TL, TR, BR, BL order (px space).
    # TL pixel (0, 0) → portal (+half_w, y, +half_h)
    # TR pixel (64, 0) → portal (-half_w, y, +half_h)
    # BR pixel (64, 64) → portal (-half_w, y, -half_h)
    # BL pixel (0, 64) → portal (+half_w, y, -half_h)
    np.testing.assert_allclose(mesh.vertices[0], [+2.0, -2.0, +1.0])
    np.testing.assert_allclose(mesh.vertices[1], [-2.0, -2.0, +1.0])
    np.testing.assert_allclose(mesh.vertices[2], [-2.0, -2.0, -1.0])
    np.testing.assert_allclose(mesh.vertices[3], [+2.0, -2.0, -1.0])

    # UVs in same order.
    np.testing.assert_allclose(mesh.uvs[0, 0], [0.0, 1.0])  # TL
    np.testing.assert_allclose(mesh.uvs[0, 1], [1.0, 1.0])  # TR
    np.testing.assert_allclose(mesh.uvs[0, 2], [1.0, 0.0])  # BR
    np.testing.assert_allclose(mesh.uvs[0, 3], [0.0, 0.0])  # BL


def test_non_shared_vertices_per_leaf():
    # Two adjacent leaves should NOT share corner verts — each gets 4
    # fresh vertices (slice-1 design choice; no manifold needed).
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


def test_face_winding_normal_points_minus_y():
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
    # Should point in -Y direction (matches Build Simple Planes winding).
    assert normal[1] < 0
