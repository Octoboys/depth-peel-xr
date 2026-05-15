"""Layer-placement strategy ABC.

A strategy maps `(near, far, n_layers)` (and optionally a sampled depth
array) to a list of contiguous `(near, far)` tuples that partition the
camera-space depth range into N layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


Layer = tuple[float, float]


class LayerPlacementStrategy(ABC):
    @abstractmethod
    def place(
        self,
        near: float,
        far: float,
        n_layers: int,
        depth: Optional[np.ndarray] = None,
    ) -> list[Layer]:
        """Return `n_layers` contiguous (near, far) layers covering [near, far].

        `depth` is an optional sampled depth array (camera-space Z) used by
        data-driven strategies. Pure analytic strategies ignore it.
        """
        raise NotImplementedError


def boundaries_to_layers(boundaries: list[float]) -> list[Layer]:
    """Convert N+1 sorted boundary depths into N contiguous (near, far) tuples."""
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]
