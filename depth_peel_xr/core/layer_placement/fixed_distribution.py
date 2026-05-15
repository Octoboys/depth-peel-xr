"""Analytic layer placement: even spacing in Z, 1/Z, or log Z.

These ignore scene depth content and just subdivide [near, far] into N
contiguous layers using one of three monotonic distributions.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from .base import Layer, LayerPlacementStrategy, boundaries_to_layers


class Distribution(str, Enum):
    UNIFORM_Z = "UNIFORM_Z"
    ONE_OVER_Z = "ONE_OVER_Z"
    LOG = "LOG"


class FixedDistribution(LayerPlacementStrategy):
    def __init__(self, distribution: Distribution = Distribution.UNIFORM_Z):
        self.distribution = Distribution(distribution)

    def place(
        self,
        near: float,
        far: float,
        n_layers: int,
        depth: Optional[np.ndarray] = None,
    ) -> list[Layer]:
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if not (near > 0.0 and far > near):
            raise ValueError(
                f"require 0 < near < far; got near={near}, far={far}"
            )

        if n_layers == 1:
            return [(float(near), float(far))]

        if self.distribution is Distribution.UNIFORM_Z:
            boundaries = np.linspace(near, far, n_layers + 1)
        elif self.distribution is Distribution.ONE_OVER_Z:
            inv = np.linspace(1.0 / near, 1.0 / far, n_layers + 1)
            boundaries = 1.0 / inv
        elif self.distribution is Distribution.LOG:
            boundaries = np.exp(
                np.linspace(np.log(near), np.log(far), n_layers + 1)
            )
        else:
            raise ValueError(f"unknown distribution {self.distribution!r}")

        boundaries[0] = near
        boundaries[-1] = far
        return boundaries_to_layers([float(b) for b in boundaries])
