"""Pure-math tests for HistogramHybrid layer placement.

The earlier strict-parity tests against `docs/reference/hybrid_depth_layers.py`
were dropped: the reference's empty-layer-collapse loop incorrectly fused
every contiguous non-empty pair into one layer, so any depth distribution
without explicit empty bands collapsed to a single layer (hit during real
Blender HYBRID runs). Our port intentionally diverges by closing only
empty-induced gaps, leaving non-empty layers separate.
"""

import numpy as np
import pytest

from core.layer_placement import HistogramHybrid


NEAR = 1.5
FAR = 21.5


def _gaussian_depth(rng, n_pixels=64 * 64, mean=8.0, std=2.0):
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
    layers = HistogramHybrid().place(NEAR, FAR, 6, depth=depth)
    assert len(layers) >= 1
    _assert_contiguous_covering(layers)


@pytest.mark.parametrize("n_layers", [2, 4, 6, 8, 10])
def test_dense_distribution_returns_close_to_n_layers(n_layers):
    """Densely-populated depth range yields no empty boundary buckets,
    so we should get back near n_layers (modulo dedupe between the
    importance and uniform halves)."""
    rng = np.random.default_rng(1234)
    depth = _gaussian_depth(rng, n_pixels=128 * 128, mean=11.0, std=4.0)
    layers = HistogramHybrid().place(NEAR, FAR, n_layers, depth=depth)
    # Uniform half alone gives ceil(n/2) layers; full hybrid bounded by n.
    assert (n_layers + 1) // 2 <= len(layers) <= n_layers
    _assert_contiguous_covering(layers)


def test_bimodal_depth_collapses_empty_middle_layers():
    """Two clusters with empty middle: empty boundary buckets are dropped
    and the gap is closed by extending the preceding layer's far across
    the void. Both clusters remain represented."""
    rng = np.random.default_rng(7)
    depth = _bimodal_depth(rng)
    requested = 12
    layers = HistogramHybrid().place(NEAR, FAR, requested, depth=depth)
    assert 2 <= len(layers) < requested
    _assert_contiguous_covering(layers)


def test_n_equals_one_returns_single_full_range_layer():
    rng = np.random.default_rng(0)
    depth = _gaussian_depth(rng)
    layers = HistogramHybrid().place(NEAR, FAR, 1, depth=depth)
    assert layers == [(NEAR, FAR)]


def test_no_depth_array_raises():
    with pytest.raises(ValueError):
        HistogramHybrid().place(NEAR, FAR, 4, depth=None)


def test_all_invalid_depth_raises():
    depth = np.full((16, 16), 1e10, dtype=np.float32)
    with pytest.raises(ValueError):
        HistogramHybrid().place(NEAR, FAR, 4, depth=depth)
