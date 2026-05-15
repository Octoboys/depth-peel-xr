"""Pure-math layer-placement strategies (no bpy imports).

Each strategy returns a list of contiguous (near, far) tuples covering the
full [near, far] camera-space range, given a user-chosen layer count.
"""

from .base import LayerPlacementStrategy
from .fixed_distribution import Distribution, FixedDistribution
from .histogram_hybrid import HistogramHybrid

__all__ = [
    "LayerPlacementStrategy",
    "FixedDistribution",
    "Distribution",
    "HistogramHybrid",
]
