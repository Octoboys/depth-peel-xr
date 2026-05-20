"""Pure-math geometry helpers shared across overlays and operators.

No bpy imports — safe to pytest standalone.
"""

from __future__ import annotations

import numpy as np


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


def super_frustum_projection(
    half_w_p: float,
    half_h_p: float,
    ho: float,
    far: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Virtual perspective camera that frames the super-frustum exactly.

    The super-frustum widens past the portal at slope 2·half_w_p / ho (see
    super_frustum_half_extents); extrapolating that linear surface back to
    zero width meets the camera axis at d = ho/2 in centre-cam space. So
    the super-frustum's union of off-axis cones can be expressed as one
    symmetric perspective from a virtual apex placed (0, 0, -ho/2) in
    centre-cam space, with portal-corner rays at NDC ±1.

    Returns (canonical_from_centre_cam, canonical_proj). Both 4×4 float64.
    `canonical_from_centre_cam` is a pure +z translation by ho/2 — apply on
    top of centre_cam_from_world to get points into the virtual cam's frame.
    `canonical_proj` uses near = ho/2 with bounds (±half_w_p, ±half_h_p) at
    near; far defaults to 1000·ho since downstream only goes through proj_inv.
    """
    if ho <= 0.0:
        raise ValueError("headplane_offset must be > 0")
    if half_w_p <= 0.0 or half_h_p <= 0.0:
        raise ValueError("portal half-extents must be > 0")
    n = ho / 2.0
    f = float(far) if far is not None else 1000.0 * ho

    canonical_from_centre = np.eye(4, dtype=np.float64)
    canonical_from_centre[2, 3] = n  # virtual apex at -ho/2 in centre-cam

    canonical_proj = np.array(
        [
            [n / half_w_p, 0.0,          0.0,                 0.0],
            [0.0,          n / half_h_p, 0.0,                 0.0],
            [0.0,          0.0,          -(f + n) / (f - n),  -2.0 * f * n / (f - n)],
            [0.0,          0.0,          -1.0,                0.0],
        ],
        dtype=np.float64,
    )
    return canonical_from_centre, canonical_proj


def inverse_canonical_projection(
    u: np.ndarray | float,
    v: np.ndarray | float,
    d: np.ndarray | float,
    half_w_p: float,
    half_h_p: float,
    ho: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of super_frustum_projection: canonical NDC (u, v) at canonical-
    cam-space depth d → portal-local (x, y, z).

    `d` is positive distance from the canonical apex along its -Z axis — the
    value stored in `consolidated/layer_NN_depth.exr` (= centre-cam depth -
    ho/2 for a point at centre-cam depth past the head-plane).

    Frame: canonical apex at portal-local (0, ho/2, 0) — the linear
    super-frustum surface extrapolated back to zero width meets the centre-
    cam axis halfway between portal and head-plane. Cam looks toward portal
    -Y; cam +X = portal -X; cam +Y = portal +Z. So the canonical near plane
    (d = ho/2 from apex) sits exactly at the portal (y_portal = 0).
    """
    if ho <= 0.0:
        raise ValueError("headplane_offset must be > 0")
    inv_n = 2.0 / ho  # 1 / (ho/2)
    x_portal = -u * d * half_w_p * inv_n
    y_portal = 0.5 * ho - d
    z_portal =  v * d * half_h_p * inv_n
    if isinstance(y_portal, float):
        return x_portal, y_portal, z_portal
    # Broadcast scalar y to match u/v shape if needed.
    y_portal = np.broadcast_to(y_portal, np.asarray(x_portal).shape).copy()
    return x_portal, y_portal, z_portal


def super_frustum_canonical_shape(
    layer_near: float,
    centre_width: int,
    centre_height: int,
    half_w_p: float,
    half_h_p: float,
    ho: float,
) -> tuple[int, int]:
    """Canonical buffer dims for a layer, scaled from centre by the per-layer
    ratio of super- vs centre-frustum half-extents at layer.near.

    Preserves centre-view texel density on the reconstructed plane at
    layer.near (the plane is sized to super-frustum extents at that depth).
    Ratio caps at 2 as layer.near → infinity; equals 1 at layer.near = ho.
    """
    super_w, super_h = super_frustum_half_extents(layer_near, half_w_p, half_h_p, ho)
    centre_w, centre_h = centre_frustum_half_extents(layer_near, half_w_p, half_h_p, ho)
    if centre_w <= 0.0 or centre_h <= 0.0:
        return centre_width, centre_height
    canonical_w = int(round(centre_width * super_w / centre_w))
    canonical_h = int(round(centre_height * super_h / centre_h))
    return canonical_w, canonical_h
