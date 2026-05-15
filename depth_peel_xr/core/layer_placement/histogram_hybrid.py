"""Histogram-driven layer placement.

Two strategies share most of the pipeline (NDC sampling, histogram +
CDF, near-duplicate dedupe, empty-layer collapse, contiguity pin) and
differ only in how they generate boundaries from the CDF:

  - HistogramHybrid    : N/2 + 1 importance + (N/2 + 1) uniform NDC,
                         merged. Robust baseline coverage.
  - HistogramImportance: N + 1 importance only. All layers track
                         content density; equal-population layers.
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


class _HistogramStrategyBase(LayerPlacementStrategy):
    """Shared scaffolding: validation, NDC sampling, post-process."""

    # HYBRID merges importance + uniform halves and needs to drop
    # near-duplicates that survive `set()` dedupe. IMPORTANCE intentionally
    # clusters boundaries in dense NDC regions (the whole point of CDF
    # inverse) — deduping those defeats the purpose.
    _apply_close_dedupe = True

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
            raise ValueError(f"{type(self).__name__} requires a depth array")
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

        boundaries_ndc = self._compute_ndc_boundaries(
            ndc_samples, ndc_min, ndc_max, n_layers
        )
        boundaries_ndc = sorted(set(boundaries_ndc))
        if self._apply_close_dedupe:
            boundaries_ndc = _drop_close_boundaries(
                boundaries_ndc, ndc_min, ndc_max, n_layers
            )
        boundaries_depth = sorted(
            float(inverse_ndc_perspective(n, near, far)) for n in boundaries_ndc
        )
        return _collapse_empty_layers(boundaries_depth, valid)

    def _compute_ndc_boundaries(
        self,
        ndc_samples: np.ndarray,
        ndc_min: float,
        ndc_max: float,
        n_layers: int,
    ) -> list[float]:
        raise NotImplementedError

    def _importance_boundaries(
        self,
        ndc_samples: np.ndarray,
        ndc_min: float,
        ndc_max: float,
        n_segments: int,
    ) -> list[float]:
        """Sample `n_segments + 1` NDC boundaries via CDF inverse:
        equal-population partitions of the sample distribution.

        Endpoints snapped to the exact ndc_min/ndc_max so that downstream
        dedupe with uniform-grid boundaries matches cleanly (otherwise
        floats from `edges[idx]` differ by ULPs and survive `set()`).
        """
        if n_segments <= 0:
            return [ndc_min, ndc_max]
        hist, edges = np.histogram(
            ndc_samples, bins=self.histogram_bins, range=(ndc_min, ndc_max)
        )
        total = hist.sum()
        if total == 0:
            return [ndc_min, ndc_max]
        cdf = np.cumsum(hist).astype(np.float64) / total
        out = []
        for i in range(n_segments + 1):
            t = i / n_segments
            idx = min(int(np.searchsorted(cdf, t)), len(edges) - 1)
            out.append(float(edges[idx]))
        out[0] = ndc_min
        out[-1] = ndc_max
        return out


class HistogramHybrid(_HistogramStrategyBase):
    """Importance-half + uniform-half boundaries merged in NDC.

    The uniform half guarantees baseline coverage even where the depth
    distribution is sparse — useful for mixed foreground / background
    scenes.
    """

    def _compute_ndc_boundaries(
        self,
        ndc_samples: np.ndarray,
        ndc_min: float,
        ndc_max: float,
        n_layers: int,
    ) -> list[float]:
        importance = self._importance_boundaries(
            ndc_samples, ndc_min, ndc_max, n_layers // 2
        )
        # `+1` compensates for the two endpoints shared between
        # importance and uniform halves: without it, dedupe would knock
        # the merged boundary count down to n_layers, yielding only
        # n_layers-1 layers.
        uniform = np.linspace(
            ndc_min, ndc_max, (n_layers - n_layers // 2) + 2
        ).tolist()
        return importance + uniform


class HistogramImportance(_HistogramStrategyBase):
    """Pure CDF inverse-sampled boundaries. All `n_layers + 1` boundaries
    come from the importance half — equal-population layers, no uniform
    safety net. Best when foreground content density should soak up the
    full layer budget.
    """

    _apply_close_dedupe = False

    def _compute_ndc_boundaries(
        self,
        ndc_samples: np.ndarray,
        ndc_min: float,
        ndc_max: float,
        n_layers: int,
    ) -> list[float]:
        return self._importance_boundaries(
            ndc_samples, ndc_min, ndc_max, n_layers
        )


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
            "All layers came back empty — depth scope contains no geometry"
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
