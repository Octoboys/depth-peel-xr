"""Pure-math geometry helpers shared across overlays and operators.

No bpy imports — safe to pytest standalone.
"""

from __future__ import annotations


def super_frustum_half_extents(
    d: float, half_w_p: float, half_h_p: float, ho: float
) -> tuple[float, float]:
    """Half-width/half-height of the super-frustum cross-section at camera-
    space depth `d`. At d <= ho all cameras frame the same portal rectangle;
    past the portal the union grows linearly so every camera's off-axis
    frustum stays contained.
    """
    if ho <= 0.0 or d <= ho:
        return half_w_p, half_h_p
    scale = 2.0 * d / ho - 1.0
    return half_w_p * scale, half_h_p * scale


def centre_frustum_half_extents(
    d: float, half_w_p: float, half_h_p: float, ho: float
) -> tuple[float, float]:
    """Half-width/half-height of the centre camera's frustum cross-section at
    camera-space depth `d`. Smaller than super_frustum for d > ho — covers only
    what the centre view captured, not the union of all NxN frustums.
    """
    if ho <= 0.0:
        return half_w_p, half_h_p
    scale = d / ho
    return half_w_p * scale, half_h_p * scale
