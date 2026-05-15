"""Depth-buffer utilities for layer-placement strategies.

NDC transform follows the standard OpenGL/Vulkan perspective mapping:
    ndc(z) = (A * z + B) / z, with A = -(far + near) / (far - near)
                                     B = -2 * far * near / (far - near)

Inverse-depth NDC gives perceptually uniform spacing — equal NDC intervals
correspond to equal 1/Z intervals, matching where depth-buffer precision
and parallax error scale linearly. All algorithms in
core/layer_placement/ that bin or sample depth do so in NDC.
"""

from __future__ import annotations

import numpy as np


def _ndc_coefficients(near: float, far: float) -> tuple[float, float]:
    if not (near > 0.0 and far > near):
        raise ValueError(f"require 0 < near < far; got near={near}, far={far}")
    a = -(far + near) / (far - near)
    b = -2.0 * far * near / (far - near)
    return a, b


def ndc_perspective(z, near: float, far: float):
    """Map camera-space depth `z` to perspective NDC depth in [-1, +1].

    Accepts a scalar or numpy array. `z > 0` is assumed (in front of camera).
    """
    a, b = _ndc_coefficients(near, far)
    return (a * z + b) / z


def inverse_ndc_perspective(ndc, near: float, far: float):
    """Inverse of `ndc_perspective` — map NDC depth back to camera-space Z.

    Accepts a scalar or numpy array.
    """
    a, b = _ndc_coefficients(near, far)
    return b / (ndc - a)


def pool_valid_depths(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Return a flat array of depth samples within (near, far).

    Filters out background pixels (Blender writes huge sentinel values for
    pixels with no geometry) and any samples outside the placement scope.
    Preserves the input dtype — callers that want float64 should cast.
    """
    if not (near > 0.0 and far > near):
        raise ValueError(f"require 0 < near < far; got near={near}, far={far}")
    mask = (depth > near) & (depth < far)
    return depth[mask].ravel()
