"""Pure-math tests for HistogramImportance — the importance-only sibling
of HistogramHybrid. Equal-population layer slabs from the CDF, no
uniform safety net.
"""

import numpy as np
import pytest

from core.layer_placement import HistogramImportance


NEAR = 1.5
FAR = 21.5


def _gaussian_depth(rng, n_pixels=128 * 128, mean=11.0, std=4.0):
    return rng.normal(mean, std, size=n_pixels).astype(np.float32)


def _bimodal_depth(rng, n_pixels=64 * 64):
    a = rng.normal(2.5, 0.05, size=n_pixels // 2)
    b = rng.normal(20.0, 0.05, size=n_pixels // 2)
    return np.concatenate([a, b]).astype(np.float32)


def _assert_contiguous_covering(layers):
    assert layers[0][0] == pytest.approx(NEAR)
    assert layers[-1][1] == pytest.approx(FAR)
    for i in range(len(layers) - 1):
        assert layers[i][1] == pytest.approx(layers[i + 1][0])


def test_returns_contiguous_layers_covering_extents():
    rng = np.random.default_rng(42)
    depth = _gaussian_depth(rng)
    layers = HistogramImportance().place(NEAR, FAR, 6, depth=depth)
    assert len(layers) >= 1
    _assert_contiguous_covering(layers)


@pytest.mark.parametrize("n_layers", [2, 4, 6, 8, 10])
def test_dense_distribution_returns_n_layers(n_layers):
    """Without the uniform half, importance-only on a dense distribution
    gives exactly n_layers (no extra dedupe between halves).
    """
    rng = np.random.default_rng(1234)
    depth = _gaussian_depth(rng)
    layers = HistogramImportance().place(NEAR, FAR, n_layers, depth=depth)
    assert len(layers) == n_layers
    _assert_contiguous_covering(layers)


def test_importance_layers_are_equal_population():
    """Each layer should contain ~equal sample counts (the defining
    property of CDF inverse sampling). Tolerance: ±20% of the mean,
    accommodating histogram-bin discretization at 10000 bins.
    """
    rng = np.random.default_rng(0)
    depth = _gaussian_depth(rng, n_pixels=256 * 256)
    n_layers = 8
    layers = HistogramImportance().place(NEAR, FAR, n_layers, depth=depth)

    counts = []
    for near, far in layers:
        counts.append(int(((depth >= near) & (depth < far)).sum()))
    mean = sum(counts) / len(counts)
    for c in counts:
        assert 0.8 * mean <= c <= 1.2 * mean, (
            f"Layer count {c} out of ±20% band around mean {mean:.0f}"
        )


def test_importance_concentrates_layers_in_dense_region():
    """A bimodal distribution with one cluster massively denser than the
    other: importance-only should concentrate layer boundaries in the
    dense cluster (because that's where samples are).
    """
    rng = np.random.default_rng(7)
    dense = rng.normal(5.0, 0.3, size=10000)  # 5x more samples
    sparse = rng.normal(15.0, 0.3, size=2000)
    depth = np.concatenate([dense, sparse]).astype(np.float32)

    layers = HistogramImportance().place(NEAR, FAR, 8, depth=depth)
    # Count boundaries falling inside the dense cluster (4..6 m).
    boundary_depths = [layers[0][0]] + [l[1] for l in layers]
    in_dense = sum(1 for b in boundary_depths if 4.0 < b < 6.0)
    in_sparse = sum(1 for b in boundary_depths if 14.0 < b < 16.0)
    assert in_dense > in_sparse, (
        f"Expected more boundaries in dense cluster; got dense={in_dense}, sparse={in_sparse}"
    )


def test_n_equals_one_returns_single_full_range_layer():
    rng = np.random.default_rng(0)
    depth = _gaussian_depth(rng)
    layers = HistogramImportance().place(NEAR, FAR, 1, depth=depth)
    assert layers == [(NEAR, FAR)]


def test_no_depth_array_raises():
    with pytest.raises(ValueError):
        HistogramImportance().place(NEAR, FAR, 4, depth=None)


def test_all_invalid_depth_raises():
    depth = np.full((16, 16), 1e10, dtype=np.float32)
    with pytest.raises(ValueError):
        HistogramImportance().place(NEAR, FAR, 4, depth=depth)


def test_bimodal_depth_collapses_empty_middle():
    """Two clusters with empty middle: empty buckets dropped, gaps
    closed. Same gap-close semantics as HistogramHybrid.
    """
    rng = np.random.default_rng(99)
    depth = _bimodal_depth(rng)
    layers = HistogramImportance().place(NEAR, FAR, 8, depth=depth)
    assert len(layers) >= 2
    _assert_contiguous_covering(layers)
