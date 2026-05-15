"""Histogram-driven hybrid layer placement.

Port of `docs/reference/hybrid_depth_layers.py` to take a numpy depth
array directly (instead of EXR paths). Algorithm unchanged:

    1. Pool valid depths → NDC.
    2. Importance half: histogram → CDF inverse-sampled at uniform t.
    3. Uniform half: even spacing in NDC.
    4. Merge + dedupe in NDC, convert back to depth.
    5. Empty-layer removal: drop layers with no samples; close the gaps.
    6. Pin to global extents [near, far].

The cube-face pooling logic from the reference is unnecessary here —
slice 2's depth probe is a single centre-camera render.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..depth import (
    inverse_ndc_perspective,
    ndc_perspective,
    pool_valid_depths,
)
from .base import Layer, LayerPlacementStrategy


_HISTOGRAM_BINS = 10000
_CONTIGUITY_TOL = 1e-6


class HistogramHybrid(LayerPlacementStrategy):
    def __init__(self, histogram_bins: int = _HISTOGRAM_BINS):
        self.histogram_bins = histogram_bins

    def place(
        self,
        near: float,
        far: float,
        n_layers: int,
        depth: Optional[np.ndarray] = None,
    ) -> list[Layer]:
        if depth is None:
            raise ValueError("HistogramHybrid requires a depth array")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")

        valid = pool_valid_depths(depth, near, far)
        if valid.size == 0:
            raise ValueError(
                f"No valid depth samples in [{near}, {far}] — empty probe?"
            )

        if n_layers == 1:
            return [(float(near), float(far))]

        ndc_samples = ndc_perspective(valid, near, far)
        ndc_near = ndc_perspective(near, near, far)
        ndc_far = ndc_perspective(far, near, far)
        ndc_min, ndc_max = (
            min(ndc_near, ndc_far),
            max(ndc_near, ndc_far),
        )

        importance_boundaries = self._importance_boundaries(
            ndc_samples, ndc_min, ndc_max, n_layers // 2
        )
        # `+1` compensates for the two endpoints shared between importance
        # and uniform halves: without it, dedupe would knock the merged
        # boundary count down to n_layers, yielding n_layers-1 layers.
        uniform_boundaries = np.linspace(
            ndc_min, ndc_max, (n_layers - n_layers // 2) + 2
        ).tolist()

        merged_ndc = sorted(set(importance_boundaries + uniform_boundaries))
        merged_ndc = _drop_close_boundaries(merged_ndc, ndc_min, ndc_max, n_layers)
        merged_depth = [
            float(inverse_ndc_perspective(n, near, far)) for n in merged_ndc
        ]
        merged_depth.sort()

        return _collapse_empty_layers(merged_depth, valid)

    def _importance_boundaries(
        self,
        ndc_samples: np.ndarray,
        ndc_min: float,
        ndc_max: float,
        num_importance: int,
    ) -> list[float]:
        if num_importance <= 0:
            return [ndc_min, ndc_max]
        hist, edges = np.histogram(
            ndc_samples, bins=self.histogram_bins, range=(ndc_min, ndc_max)
        )
        total = hist.sum()
        if total == 0:
            return [ndc_min, ndc_max]
        cdf = np.cumsum(hist).astype(np.float64) / total
        out = []
        for i in range(num_importance + 1):
            t = i / num_importance
            idx = min(int(np.searchsorted(cdf, t)), len(edges) - 1)
            out.append(float(edges[idx]))
        # Snap endpoints to the exact NDC extents so they coincide with
        # the uniform-half endpoints under dedupe. Without this, t=0 lands
        # at the first occupied histogram bin (≈ ndc_min) and t=1 at the
        # last (≈ ndc_max but not equal), producing a near-zero-thickness
        # tail layer after merge.
        out[0] = ndc_min
        out[-1] = ndc_max
        return out


def _drop_close_boundaries(
    sorted_ndc: list[float], ndc_min: float, ndc_max: float, n_layers: int
) -> list[float]:
    """Drop interior boundaries that sit within a small NDC tolerance of
    their neighbour — handles the near-duplicate case where an importance
    boundary lands within a few ULPs of a uniform boundary, which `set()`
    dedupe misses (different float values, same intent).

    Tolerance = NDC-range / (4 × n_layers): no resulting layer is thinner
    than ¼ of an average uniform layer. Scales naturally so high
    layer-count requests (e.g. n=64) aren't capped by a fixed fraction.
    """
    if len(sorted_ndc) <= 2:
        return list(sorted_ndc)
    tol = (ndc_max - ndc_min) / max(1, n_layers * 4)
    out = [sorted_ndc[0]]
    for v in sorted_ndc[1:-1]:
        if v - out[-1] >= tol:
            out.append(v)
    last = sorted_ndc[-1]
    if last - out[-1] >= tol:
        out.append(last)
    else:
        out[-1] = last
    return out


def _collapse_empty_layers(
    boundaries_depth: list[float], valid: np.ndarray
) -> list[Layer]:
    """Drop empty layers; close gaps left behind by extending the preceding
    layer's far. Pin to global extents.

    Differs from the reference's merge-loop, which incorrectly fused
    every contiguous pair into one layer (yielding a single layer when
    no empties existed). Here, contiguous non-empty layers stay
    *separate*; only empty-induced gaps trigger the gap-close.
    """
    if len(boundaries_depth) < 2:
        raise ValueError("need at least two boundaries to form a layer")

    n = len(boundaries_depth) - 1
    counts = []
    for i in range(n):
        d_near, d_far = boundaries_depth[i], boundaries_depth[i + 1]
        lo, hi = min(d_near, d_far), max(d_near, d_far)
        counts.append(int(((valid >= lo) & (valid < hi)).sum()))

    if not any(c > 0 for c in counts):
        raise ValueError(
            "All hybrid layers came back empty — depth scope contains no geometry"
        )

    # Walk left-to-right keeping non-empty layers verbatim. When the
    # previous kept layer's far doesn't meet the current layer's near,
    # an empty layer was dropped between them — extend the previous
    # layer's far across the gap.
    merged: list[Layer] = []
    for i, c in enumerate(counts):
        if c == 0:
            continue
        cur_near = boundaries_depth[i]
        cur_far = boundaries_depth[i + 1]
        if merged and abs(cur_near - merged[-1][1]) >= _CONTIGUITY_TOL:
            prev_near, _prev_far = merged[-1]
            merged[-1] = (prev_near, cur_near)
        merged.append((cur_near, cur_far))

    # Pin to global extents.
    first_n, first_f = merged[0]
    merged[0] = (boundaries_depth[0], first_f)
    last_n, last_f = merged[-1]
    merged[-1] = (last_n, boundaries_depth[-1])
    return merged
