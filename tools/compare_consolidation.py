"""Objective quality comparison of forward-splat vs reverse-mapping
consolidation on a synthetic fixture with known ground truth.

Builds a small analytic scene (one or more textured world-space planes),
renders it into a synthetic NxN rig via a numpy raycaster + bilinear
texture lookup, and *also* renders the same scene directly into the
canonical super-frustum camera at canonical resolution — that direct
render is the ground truth. Then runs both `consolidate_layer_forward`
and `consolidate_layer_reverse` on the rig captures and prints:

  - RGB L1 mean + PSNR of each canonical vs ground truth (masked to that
    canonical's coverage) — absolute quality number, not just an A/B.
  - Reverse-path fallback rate = fraction of covered canonical pixels
    where no source view passed both gates and pass-1 colour was kept.
    High fallback rate = pass 2 barely running; usually the depth-
    consistency tolerance is too tight or the scene doesn't stress it.
  - Border-leaf count from `build_quadtree` on each canonical's alpha.
    The annulus-stretching bug produces measurably more border leaves at
    the canvas diagonals; a reverse-mapping win reduces this count.

Pure numpy — no bpy. Runs standalone from repo root:

    python tools/compare_consolidation.py

CLI knobs are for iteration; defaults reproduce the same scene each run.
Higher PSNR / lower L1 / lower border-leaf count on `reverse` vs
`forward` = quality win. If numbers are indistinguishable, the fixture
scene doesn't exercise the failure modes — extend `_build_scene()` with
higher-contrast edges or thinner off-axis detail.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "depth_peel_xr"))

from core.geometry import (  # noqa: E402
    super_frustum_canonical_shape,
    super_frustum_projection,
)
from core.multi_view import (  # noqa: E402
    REVERSE_FALLBACK_SOURCE,
    consolidate_layer_forward,
    consolidate_layer_reverse,
)
from core.quadtree import build_quadtree  # noqa: E402


# ---- synthetic scene -------------------------------------------------------


@dataclass(frozen=True)
class TexturedPlane:
    """Axis-aligned textured plane at world z = z_plane, extending
    [-half_w, half_w] × [-half_h, half_h] in world x/y."""
    z_plane: float
    half_w: float
    half_h: float
    texture: np.ndarray  # (Th, Tw, 3) float32 in [0, 1]

    def sample(self, wx: np.ndarray, wy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bilinear-sample the texture at world (wx, wy). Returns
        (rgb (..., 3), hit_mask (...,)) — hit_mask is False outside the
        plane's extent."""
        Th, Tw, _ = self.texture.shape
        u = (wx + self.half_w) / (2.0 * self.half_w)
        v = (wy + self.half_h) / (2.0 * self.half_h)
        hit = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        u_pix = np.clip(u * Tw - 0.5, 0.0, Tw - 1.0)
        v_pix = np.clip(v * Th - 0.5, 0.0, Th - 1.0)
        u0 = np.floor(u_pix).astype(np.int64)
        v0 = np.floor(v_pix).astype(np.int64)
        u0c = np.clip(u0, 0, Tw - 2)
        v0c = np.clip(v0, 0, Th - 2)
        fu = (u_pix - u0c).astype(np.float32)
        fv = (v_pix - v0c).astype(np.float32)
        t00 = self.texture[v0c, u0c]
        t01 = self.texture[v0c, u0c + 1]
        t10 = self.texture[v0c + 1, u0c]
        t11 = self.texture[v0c + 1, u0c + 1]
        w00 = (1.0 - fu) * (1.0 - fv)
        w01 = fu * (1.0 - fv)
        w10 = (1.0 - fu) * fv
        w11 = fu * fv
        rgb = (
            t00 * w00[..., None]
            + t01 * w01[..., None]
            + t10 * w10[..., None]
            + t11 * w11[..., None]
        )
        return rgb, hit


def _checkerboard(size: int, cells: int, colour_a=(0.05, 0.05, 0.05),
                  colour_b=(0.95, 0.95, 0.95)) -> np.ndarray:
    """size x size RGB checkerboard with `cells` squares per side."""
    idx = (np.arange(size) * cells // size).astype(np.int64)
    grid = (idx[:, None] + idx[None, :]) % 2
    a = np.array(colour_a, dtype=np.float32)
    b = np.array(colour_b, dtype=np.float32)
    return np.where(grid[..., None].astype(bool), a[None, None, :], b[None, None, :]).astype(np.float32)


def _build_scene(half_w_p: float, half_h_p: float, ho: float) -> list[TexturedPlane]:
    """Single wide checkerboard plane sized to fill the super-frustum at
    its depth. Annular region between centre-frustum and super-frustum
    extents is only visible to sibling views — exactly the region where
    forward-splat stretching shows up. Checkerboard edges stress
    sub-pixel aliasing.
    """
    D = 2.0 * ho  # depth past portal (world z = -D)
    d_cam = ho + D  # camera-space depth from centre
    # super-frustum extent at this depth
    scale = 2.0 * d_cam / ho - 1.0
    plane_half_w = half_w_p * scale
    plane_half_h = half_h_p * scale
    texture = _checkerboard(size=1024, cells=24)
    return [TexturedPlane(z_plane=-D, half_w=plane_half_w,
                          half_h=plane_half_h, texture=texture)]


# ---- cameras + rendering ---------------------------------------------------


def _frustum(l: float, r: float, b: float, t: float, n: float, f: float) -> np.ndarray:
    """OpenGL off-axis perspective projection matrix. Matches the recipe
    in operators/consolidate.py::_build_projection_matrix but with
    explicit l/r/b/t."""
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = 2.0 * n / (r - l)
    m[1, 1] = 2.0 * n / (t - b)
    m[0, 2] = (r + l) / (r - l)
    m[1, 2] = (t + b) / (t - b)
    m[2, 2] = -(f + n) / (f - n)
    m[2, 3] = -2.0 * f * n / (f - n)
    m[3, 2] = -1.0
    return m


def _translation(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = x; m[1, 3] = y; m[2, 3] = z
    return m


def _rig_camera(cx: float, cy: float, ho: float, half_w_p: float,
                half_h_p: float, near: float, far: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """One rig camera at world (cx, cy, ho) looking -z through the portal
    at world z=0 sized (±half_w_p, ±half_h_p). Off-axis projection so its
    near-plane frustum passes through the portal rectangle."""
    world_from_cam = _translation(cx, cy, ho)
    # At near-plane (z_cam = -near), frustum bounds in camera space =
    # world portal corners minus camera position.
    proj = _frustum(
        l=-half_w_p - cx, r=half_w_p - cx,
        b=-half_h_p - cy, t=half_h_p - cy,
        n=near, f=far,
    )
    return world_from_cam, proj


def _rig_positions(grid_n: int, spread: float) -> list[tuple[float, float]]:
    """N×N grid positions on the head plane, centred at origin. Centre
    first (matches caller's convention in the real add-on), then siblings
    sorted closest-to-centre."""
    if grid_n < 1:
        raise ValueError("grid_n must be >= 1")
    if grid_n == 1:
        return [(0.0, 0.0)]
    coords = np.linspace(-spread, spread, grid_n)
    all_pos = [(float(x), float(y)) for y in coords for x in coords]
    centre = (0.0, 0.0)
    if grid_n % 2 == 1:
        # Odd N — exact centre exists in the grid.
        centre_pos = (0.0, 0.0)
        siblings = [p for p in all_pos if p != centre_pos]
    else:
        # Even N — no camera at exact centre; caller adds one (matches
        # the rig's even-N append-centre behaviour).
        siblings = list(all_pos)
    siblings.sort(key=lambda p: p[0] ** 2 + p[1] ** 2)
    return [centre] + siblings


def _render_camera(world_from_cam: np.ndarray, proj: np.ndarray,
                   width: int, height: int,
                   planes: list[TexturedPlane]) -> tuple[np.ndarray, np.ndarray]:
    """Raycast one camera view against a list of TexturedPlane scenes,
    keep the closest hit per pixel. Returns (rgba (H, W, 4), depth (H, W))
    where depth is positive cam-space distance along -Z. Miss pixels have
    alpha=0 and depth=0."""
    proj_inv = np.linalg.inv(proj)
    # NDC grid — half-pixel centred, matching core.multi_view._ndc_grid.
    us = np.arange(width, dtype=np.float64) + 0.5
    vs = np.arange(height, dtype=np.float64) + 0.5
    u_ndc = 2.0 * us / width - 1.0
    v_ndc = 1.0 - 2.0 * vs / height
    u_g, v_g = np.meshgrid(u_ndc, v_ndc)
    ones = np.ones_like(u_g)
    # Two NDC points along the same view ray — near and far — unprojected
    # to cam space via proj_inv.
    ndc_near = np.stack([u_g, v_g, -ones, ones], axis=-1).reshape(-1, 4)
    ndc_far = np.stack([u_g, v_g, ones, ones], axis=-1).reshape(-1, 4)
    cam_near = ndc_near @ proj_inv.T
    cam_near /= cam_near[:, 3:4]
    cam_far = ndc_far @ proj_inv.T
    cam_far /= cam_far[:, 3:4]
    # World-space ray origin (cam) + direction.
    cam_pos = world_from_cam[:3, 3]
    world_near = (cam_near @ world_from_cam.T)[:, :3]
    world_far = (cam_far @ world_from_cam.T)[:, :3]
    ray_dir = world_far - world_near
    ray_dir /= np.linalg.norm(ray_dir, axis=-1, keepdims=True) + 1e-12
    ray_o = np.broadcast_to(cam_pos, world_near.shape)

    rgba = np.zeros((height * width, 4), dtype=np.float32)
    depth = np.zeros((height * width), dtype=np.float32)
    best_t = np.full((height * width,), np.inf, dtype=np.float64)

    # Cam looks down -Z of its own frame; cam_from_world @ world_hit gives
    # camera-space z_cam of the hit. Reuse this to output positive depth.
    cam_from_world = np.linalg.inv(world_from_cam)

    for plane in planes:
        # Intersect with z = plane.z_plane: ray_o.z + t * ray_dir.z = z_plane
        denom = ray_dir[:, 2]
        safe = np.abs(denom) > 1e-8
        t = np.where(safe, (plane.z_plane - ray_o[:, 2]) / np.where(safe, denom, 1.0), np.inf)
        hit_pt = ray_o + ray_dir * t[:, None]
        rgb, in_plane = plane.sample(hit_pt[:, 0], hit_pt[:, 1])
        valid = safe & (t > 0.0) & in_plane & (t < best_t)
        if not valid.any():
            continue
        best_t[valid] = t[valid]
        # Cam-space depth = -z_cam of world hit point.
        hit_h = np.concatenate([hit_pt, np.ones((hit_pt.shape[0], 1))], axis=-1)
        z_cam = (hit_h @ cam_from_world.T)[:, 2]
        depth[valid] = -z_cam[valid].astype(np.float32)
        rgba[valid, :3] = rgb[valid]
        rgba[valid, 3] = 1.0

    return rgba.reshape(height, width, 4), depth.reshape(height, width)


# ---- metrics ---------------------------------------------------------------


def _psnr(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(pred.shape[:2], dtype=bool)
    if not mask.any():
        return float("nan")
    diff = (pred[..., :3] - gt[..., :3])[mask]
    mse = float(np.mean(diff * diff))
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _l1(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(pred.shape[:2], dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(pred[..., :3][mask] - gt[..., :3][mask])))


def _border_leaf_count(alpha: np.ndarray, min_quad_size: int, tile_size: int) -> tuple[int, int]:
    """(border, total) leaf counts from build_quadtree on alpha>0.5."""
    coverage = alpha > 0.5
    leaves = build_quadtree(coverage, min_quad_size=min_quad_size, tile_size=tile_size)
    border = sum(1 for L in leaves if L.cls == "border")
    return border, len(leaves)


# ---- driver ----------------------------------------------------------------


def _measure(grid_n: int, view_res: int, half_w_p: float, half_h_p: float,
             ho: float, near: float, far: float, layer_near: float,
             min_quad_size: int, tile_size: int, rel_tol: float | None) -> dict:
    """Render rig + GT, run both consolidations, return a dict of metrics.
    No printing — the caller renders (single-res detailed or sweep table)."""
    planes = _build_scene(half_w_p, half_h_p, ho)
    positions = _rig_positions(grid_n, spread=0.4 * half_w_p)

    views_sorted = []
    for cx, cy in positions:
        world_from_cam, proj = _rig_camera(cx, cy, ho, half_w_p, half_h_p, near, far)
        rgba, depth = _render_camera(world_from_cam, proj, view_res, view_res, planes)
        views_sorted.append({
            "rgba": rgba, "depth": depth,
            "world_from_cam": world_from_cam, "proj": proj,
        })

    canonical_from_centre, canonical_proj = super_frustum_projection(
        half_w_p, half_h_p, ho, far=far,
    )
    centre_world_from_cam = views_sorted[0]["world_from_cam"]
    canonical_cam_from_world = canonical_from_centre @ np.linalg.inv(centre_world_from_cam)
    canonical_world_from_cam = np.linalg.inv(canonical_cam_from_world)
    canonical_W, canonical_H = super_frustum_canonical_shape(
        layer_near, view_res, view_res, half_w_p, half_h_p, ho,
    )

    gt_rgba, _gt_depth = _render_camera(
        canonical_world_from_cam, canonical_proj,
        canonical_W, canonical_H, planes,
    )
    gt_coverage = gt_rgba[..., 3] > 0.5

    fwd_t0 = time.perf_counter()
    fwd_rgba, _fd, fwd_cov, _fs = consolidate_layer_forward(
        views_sorted=views_sorted,
        canonical_shape=(canonical_H, canonical_W),
        canonical_cam_from_world=canonical_cam_from_world,
        canonical_proj=canonical_proj,
    )
    fwd_secs = time.perf_counter() - fwd_t0
    rev_kwargs = {"rel_tol": rel_tol} if rel_tol is not None else {}
    rev_t0 = time.perf_counter()
    rev_rgba, _rd, rev_cov, rev_src = consolidate_layer_reverse(
        views_sorted=views_sorted,
        canonical_shape=(canonical_H, canonical_W),
        canonical_cam_from_world=canonical_cam_from_world,
        canonical_proj=canonical_proj,
        **rev_kwargs,
    )
    rev_secs = time.perf_counter() - rev_t0

    common = fwd_cov & rev_cov & gt_coverage
    if rev_cov.any():
        fallback_rate = float(
            ((rev_src == REVERSE_FALLBACK_SOURCE) & rev_cov).sum() / rev_cov.sum()
        )
    else:
        fallback_rate = 0.0
    fwd_border, fwd_total = _border_leaf_count(fwd_rgba[..., 3], min_quad_size, tile_size)
    rev_border, rev_total = _border_leaf_count(rev_rgba[..., 3], min_quad_size, tile_size)

    return {
        "view_res": view_res,
        "canonical_wh": (canonical_W, canonical_H),
        "n_views": len(views_sorted),
        "common_pixels": int(common.sum()),
        "common_fraction": float(common.mean()),
        "gt_coverage": float(gt_coverage.mean()),
        "fwd_coverage": float(fwd_cov.mean()),
        "rev_coverage": float(rev_cov.mean()),
        "fwd_l1": _l1(fwd_rgba, gt_rgba, common),
        "rev_l1": _l1(rev_rgba, gt_rgba, common),
        "fwd_psnr": _psnr(fwd_rgba, gt_rgba, common),
        "rev_psnr": _psnr(rev_rgba, gt_rgba, common),
        "fallback_rate": fallback_rate,
        "fwd_border": fwd_border, "fwd_total": fwd_total,
        "rev_border": rev_border, "rev_total": rev_total,
        "fwd_secs": fwd_secs, "rev_secs": rev_secs,
    }


def _print_detailed(m: dict, grid_n: int, min_quad_size: int, tile_size: int) -> None:
    cW, cH = m["canonical_wh"]
    print(f"rig: {grid_n}×{grid_n} ({m['n_views']} views) @ {m['view_res']}²  "
          f"→ canonical {cW}×{cH}")
    print()
    print("--- quality vs ground truth (common covered pixels) ---")
    print(f"  common pixel count : {m['common_pixels']:>10d} "
          f"({m['common_fraction']*100:.1f}% of canvas)")
    print(f"  forward  L1        : {m['fwd_l1']:.5f}  PSNR {m['fwd_psnr']:.2f} dB")
    print(f"  reverse  L1        : {m['rev_l1']:.5f}  PSNR {m['rev_psnr']:.2f} dB")
    print()
    print("--- coverage (canvas fraction) ---")
    print(f"  ground truth       : {m['gt_coverage']*100:6.2f}%")
    print(f"  forward canonical  : {m['fwd_coverage']*100:6.2f}%")
    print(f"  reverse canonical  : {m['rev_coverage']*100:6.2f}%")
    print()
    print("--- reverse-mapping health ---")
    print(f"  fallback rate      : {m['fallback_rate']*100:5.2f}%  "
          f"(covered pixels where no source view passed both gates)")
    if m["fallback_rate"] > 0.2:
        print("  WARNING: >20% fallback — pass 2 is barely running. Loosen")
        print("           rel_tol or check the fixture matches the gate model.")
    print()
    print("--- quadtree border-leaf count (alpha > 0.5, min_quad_size="
          f"{min_quad_size}, tile_size={tile_size}) ---")
    print(f"  forward  : {m['fwd_border']:5d} border / {m['fwd_total']:5d} total")
    print(f"  reverse  : {m['rev_border']:5d} border / {m['rev_total']:5d} total")
    delta = m["fwd_border"] - m["rev_border"]
    sign = "-" if delta > 0 else ("+" if delta < 0 else " ")
    print(f"  delta    : {sign}{abs(delta):5d} border leaves  "
          f"({'reverse wins' if delta > 0 else 'forward wins' if delta < 0 else 'tie'})")
    print()
    print("--- wall-clock time (single run, includes numpy alloc + gates) ---")
    print(f"  forward  : {m['fwd_secs']*1000:7.1f} ms")
    print(f"  reverse  : {m['rev_secs']*1000:7.1f} ms  "
          f"({m['rev_secs']/m['fwd_secs']:.2f}× forward)")


def _print_sweep_table(rows: list[dict], grid_n: int) -> None:
    """Compact per-resolution table. Emphasises how the PSNR gap
    collapses (or doesn't) as source density grows."""
    print(f"rig: {grid_n}×{grid_n} — resolution sweep")
    print()
    header = (
        f"  {'view':>5}  {'canonical':>11}  {'fwd PSNR':>9}  {'rev PSNR':>9}  "
        f"{'Δ dB':>7}  {'fwd L1':>8}  {'rev L1':>8}  {'fallback':>9}  "
        f"{'fwd brd':>8}  {'rev brd':>8}  {'fwd ms':>8}  {'rev ms':>8}  {'×':>5}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in rows:
        cW, cH = m["canonical_wh"]
        delta = m["rev_psnr"] - m["fwd_psnr"]
        # PSNR may be inf when pred == GT (rare on the sweep, guard anyway).
        def _fmt(x, w, prec):
            if np.isinf(x): return f"{'inf':>{w}}"
            if np.isnan(x): return f"{'nan':>{w}}"
            return f"{x:>{w}.{prec}f}"
        ratio = m["rev_secs"] / m["fwd_secs"] if m["fwd_secs"] > 0 else float("nan")
        print(
            f"  {m['view_res']:>5d}  "
            f"{f'{cW}×{cH}':>11}  "
            f"{_fmt(m['fwd_psnr'], 9, 2)}  "
            f"{_fmt(m['rev_psnr'], 9, 2)}  "
            f"{_fmt(delta, 7, 2)}  "
            f"{_fmt(m['fwd_l1'], 8, 5)}  "
            f"{_fmt(m['rev_l1'], 8, 5)}  "
            f"{m['fallback_rate']*100:>8.2f}%  "
            f"{m['fwd_border']:>8d}  "
            f"{m['rev_border']:>8d}  "
            f"{m['fwd_secs']*1000:>8.1f}  "
            f"{m['rev_secs']*1000:>8.1f}  "
            f"{_fmt(ratio, 5, 2)}"
        )
    print()
    print("  Read: Δ dB > 0 = reverse wins. Watch where it collapses toward zero")
    print("  — that's the source-density crossover past which reverse-mapping")
    print("  stops earning its keep on this fixture.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grid-n", type=int, default=3,
                    help="N×N rig grid (default 3)")
    ap.add_argument("--view-res", type=int, nargs="+", default=[256],
                    help="per-view render resolution(s), square. One value → "
                         "detailed report; multiple → sweep table (default 256)")
    ap.add_argument("--half-w-p", type=float, default=1.0,
                    help="portal half-width (default 1.0)")
    ap.add_argument("--half-h-p", type=float, default=1.0,
                    help="portal half-height (default 1.0)")
    ap.add_argument("--ho", type=float, default=1.0,
                    help="head-plane offset (default 1.0)")
    ap.add_argument("--near", type=float, default=1.0,
                    help="rig-cam near clip (default = ho)")
    ap.add_argument("--far", type=float, default=100.0,
                    help="rig-cam far clip (default 100)")
    ap.add_argument("--layer-near", type=float, default=3.0,
                    help="synthetic layer.near in centre-cam depth "
                         "(controls canonical resolution scaling; default 3)")
    ap.add_argument("--min-quad-size", type=int, default=8,
                    help="quadtree min quad size (default 8)")
    ap.add_argument("--tile-size", type=int, default=128,
                    help="quadtree tile size (default 128)")
    ap.add_argument("--rel-tol", type=float, default=None,
                    help="override reverse-mapping depth rel-tol "
                         "(default = library default)")
    ap.add_argument("--sweep", action="store_true",
                    help="run the standard sweep [128, 256, 512, 1024, 2048] "
                         "(overrides --view-res)")
    args = ap.parse_args()

    view_res_list = [128, 256, 512, 1024, 2048] if args.sweep else list(args.view_res)

    if len(view_res_list) == 1:
        m = _measure(
            grid_n=args.grid_n, view_res=view_res_list[0],
            half_w_p=args.half_w_p, half_h_p=args.half_h_p, ho=args.ho,
            near=args.near, far=args.far, layer_near=args.layer_near,
            min_quad_size=args.min_quad_size, tile_size=args.tile_size,
            rel_tol=args.rel_tol,
        )
        _print_detailed(m, args.grid_n, args.min_quad_size, args.tile_size)
        return

    rows = []
    for vr in view_res_list:
        print(f"  measuring view_res={vr} …", flush=True)
        rows.append(_measure(
            grid_n=args.grid_n, view_res=vr,
            half_w_p=args.half_w_p, half_h_p=args.half_h_p, ho=args.ho,
            near=args.near, far=args.far, layer_near=args.layer_near,
            min_quad_size=args.min_quad_size, tile_size=args.tile_size,
            rel_tol=args.rel_tol,
        ))
    print()
    _print_sweep_table(rows, args.grid_n)


if __name__ == "__main__":
    main()
