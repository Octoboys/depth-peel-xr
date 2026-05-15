"""Pure-math tests for core.depth NDC + pooling utilities."""

import numpy as np
import pytest

from core.depth import (
    inverse_ndc_perspective,
    ndc_perspective,
    pool_valid_depths,
)


NEAR = 1.5
FAR = 21.5


def test_ndc_endpoints_are_monotonic():
    near_ndc = ndc_perspective(NEAR, NEAR, FAR)
    far_ndc = ndc_perspective(FAR, NEAR, FAR)
    assert near_ndc < far_ndc


def test_ndc_round_trip_scalar():
    for z in (NEAR, 5.0, 12.5, FAR):
        n = ndc_perspective(z, NEAR, FAR)
        z_back = inverse_ndc_perspective(n, NEAR, FAR)
        assert z_back == pytest.approx(z)


def test_ndc_round_trip_array():
    z = np.linspace(NEAR, FAR, 50)
    n = ndc_perspective(z, NEAR, FAR)
    z_back = inverse_ndc_perspective(n, NEAR, FAR)
    assert np.allclose(z_back, z)


def test_ndc_monotonic_in_z():
    z = np.linspace(NEAR, FAR, 100)
    n = ndc_perspective(z, NEAR, FAR)
    assert np.all(np.diff(n) > 0)


def test_invalid_near_far_raises():
    with pytest.raises(ValueError):
        ndc_perspective(5.0, 0.0, FAR)
    with pytest.raises(ValueError):
        ndc_perspective(5.0, FAR, NEAR)
    with pytest.raises(ValueError):
        inverse_ndc_perspective(0.0, 0.0, FAR)


def test_pool_filters_outside_range_and_flattens():
    depth = np.array(
        [
            [0.5, 1.5, 2.0],
            [10.0, 21.5, 22.0],
            [1e10, 5.0, 8.0],
        ],
        dtype=np.float32,
    )
    pooled = pool_valid_depths(depth, NEAR, FAR)
    assert pooled.ndim == 1
    assert set(pooled.tolist()) == {2.0, 10.0, 5.0, 8.0}


def test_pool_empty_when_all_invalid():
    depth = np.full((4, 4), 1e10, dtype=np.float32)
    pooled = pool_valid_depths(depth, NEAR, FAR)
    assert pooled.size == 0
