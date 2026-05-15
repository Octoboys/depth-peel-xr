"""Pure-math tests for FixedDistribution layer placement (no bpy)."""

import math

import numpy as np
import pytest

from core.layer_placement import Distribution, FixedDistribution


NEAR = 1.5
FAR = 21.5


def _boundaries(layers):
    """Reconstruct N+1 boundaries from N contiguous (near, far) tuples."""
    assert all(layers[i][1] == layers[i + 1][0] for i in range(len(layers) - 1)), (
        "layers must be contiguous"
    )
    return [layers[0][0]] + [pair[1] for pair in layers]


@pytest.mark.parametrize("dist", list(Distribution))
def test_returns_n_contiguous_layers_covering_full_range(dist):
    n = 5
    layers = FixedDistribution(dist).place(NEAR, FAR, n)
    assert len(layers) == n
    bounds = _boundaries(layers)
    assert bounds[0] == pytest.approx(NEAR)
    assert bounds[-1] == pytest.approx(FAR)
    assert all(bounds[i] < bounds[i + 1] for i in range(len(bounds) - 1))


def test_uniform_z_evenly_spaces_in_z():
    layers = FixedDistribution(Distribution.UNIFORM_Z).place(NEAR, FAR, 4)
    bounds = np.array(_boundaries(layers))
    diffs = np.diff(bounds)
    assert np.allclose(diffs, diffs[0])


def test_one_over_z_evenly_spaces_in_inv_z():
    layers = FixedDistribution(Distribution.ONE_OVER_Z).place(NEAR, FAR, 4)
    bounds = np.array(_boundaries(layers))
    inv = 1.0 / bounds
    diffs = np.diff(inv)
    assert np.allclose(diffs, diffs[0])


def test_log_evenly_spaces_in_log_z():
    layers = FixedDistribution(Distribution.LOG).place(NEAR, FAR, 4)
    bounds = np.array(_boundaries(layers))
    log_bounds = np.log(bounds)
    diffs = np.diff(log_bounds)
    assert np.allclose(diffs, diffs[0])


@pytest.mark.parametrize("dist", list(Distribution))
def test_n_equals_one_returns_single_full_range_layer(dist):
    layers = FixedDistribution(dist).place(NEAR, FAR, 1)
    assert layers == [(NEAR, FAR)]


@pytest.mark.parametrize("dist", list(Distribution))
def test_invalid_inputs_raise(dist):
    strat = FixedDistribution(dist)
    with pytest.raises(ValueError):
        strat.place(NEAR, FAR, 0)
    with pytest.raises(ValueError):
        strat.place(0.0, FAR, 4)
    with pytest.raises(ValueError):
        strat.place(FAR, NEAR, 4)
    with pytest.raises(ValueError):
        strat.place(NEAR, NEAR, 4)


def test_distribution_string_alias_accepted():
    strat = FixedDistribution("UNIFORM_Z")
    assert strat.distribution is Distribution.UNIFORM_Z


def test_depth_array_argument_is_accepted_and_ignored():
    fake_depth = np.array([1.0, 2.0, 3.0])
    layers = FixedDistribution(Distribution.UNIFORM_Z).place(NEAR, FAR, 3, depth=fake_depth)
    assert len(layers) == 3
