"""
Hybrid Depth Layer Calculator

Computes depth layer boundaries from cubemap face depth renders using a
hybrid approach: half importance-sampled + half uniform, merged and
deduplicated, with post-processing to remove empty layers.

Pools depth from all active cubemap faces to produce one shared set of
boundaries — consistent across face borders (no seams at cube edges).

Usage:
    python hybrid_depth_layers.py

Requires: numpy, OpenImageIO
"""

import numpy as np
import OpenImageIO as oiio
import json
import os

# Cubemap face definitions matching MXI convention
FACES = ["negx", "posx", "negy", "posy", "negz", "posz"]


def load_depth(exr_path):
    """Load depth values from an EXR file.

    Args:
        exr_path: path to a single-channel (or multi-channel) EXR depth render.
                  Only the first channel is read.

    Returns:
        2D numpy array of depth values (height x width).
    """
    buf = oiio.ImageBuf(exr_path)
    spec = buf.spec()
    depth = np.frombuffer(buf.get_pixels(oiio.FLOAT), dtype=np.float32)
    depth = depth.reshape(spec.height, spec.width, spec.nchannels)[:, :, 0]
    return depth


def compute_hybrid_boundaries(depth_exr_paths, num_layers, near, far,
                              skip_faces=None):
    """
    Compute hybrid (importance + uniform) depth layer boundaries from cubemap
    face depth renders. Pools depth from all active faces to produce one shared
    set of boundaries — consistent across face borders.

    The hybrid approach combines:
      - Importance half: CDF inverse sampling, dense where content is dense.
      - Uniform half: even NDC spacing, ensures baseline coverage everywhere.
    These are merged and deduplicated, then empty layers are removed and gaps
    closed to produce a contiguous partition of the depth range.

    Args:
        depth_exr_paths: dict mapping face name -> EXR path
                         e.g. {"negz": "/path/depth_negz.exr", ...}
        num_layers: desired number of depth layers
        near: near clip distance (scene units)
        far: far clip distance (scene units)
        skip_faces: list of face names to skip
                    e.g. ["posz"] for 180-degree hemisphere

    Returns:
        List of (near_depth, far_depth) tuples for each layer (no empty layers).
        Actual count may be fewer than num_layers after empty removal.
        Layers are contiguous: layer[i].far == layer[i+1].near.
    """
    skip_faces = skip_faces or []

    # Pool depth values from all active faces into a single array.
    # Since all face cameras share the same origin, camera-space Z from
    # each face is effectively radial distance — pooling them gives a
    # view-direction-independent depth distribution.
    all_depths = []
    for face, path in depth_exr_paths.items():
        if face in skip_faces:
            continue
        depth = load_depth(path)
        valid = depth[(depth > near) & (depth < far)].flatten()
        all_depths.append(valid)

    if not all_depths:
        raise ValueError("No valid depth data found across any face")

    valid = np.concatenate(all_depths)
    print(f"Pooled {len(valid):,} valid depth pixels from {len(all_depths)} faces")

    # NDC transform (inverse depth for perceptually uniform near-field precision).
    # ndc = (A * depth + B) / depth maps depth so near-field gets more precision.
    A = -(far + near) / (far - near)
    B = -2.0 * far * near / (far - near)
    ndc = (A * valid + B) / valid
    ndc_near = (A * near + B) / near
    ndc_far = (A * far + B) / far
    ndc_min, ndc_max = min(ndc_near, ndc_far), max(ndc_near, ndc_far)

    # --- Importance-sampled half ---
    # CDF inverse sampling: dense where content is dense
    num_importance = num_layers // 2
    hist, edges = np.histogram(ndc, bins=10000, range=(ndc_min, ndc_max))
    cdf = np.cumsum(hist).astype(float) / hist.sum()

    importance_boundaries = []
    for i in range(num_importance + 1):
        t = i / num_importance
        idx = min(np.searchsorted(cdf, t), len(edges) - 1)
        importance_boundaries.append(edges[idx])

    # --- Uniform half ---
    # Even spacing in NDC for baseline coverage across the full range
    num_uniform = num_layers - num_importance
    uniform_boundaries = np.linspace(ndc_min, ndc_max, num_uniform + 1).tolist()

    # --- Merge and deduplicate ---
    all_ndc = sorted(set(importance_boundaries + uniform_boundaries))

    # Convert NDC back to real depth
    all_depth = [B / (n - A) for n in all_ndc]

    # --- Post-process: remove empty layers ---
    # Count actual depth samples per layer to identify empties
    counts = []
    for i in range(len(all_depth) - 1):
        d_near, d_far = all_depth[i], all_depth[i + 1]
        lo, hi = min(d_near, d_far), max(d_near, d_far)
        mask = (valid >= lo) & (valid < hi)
        counts.append(mask.sum())

    # Keep only layers with content
    filtered_near = []
    filtered_far = []
    for i, c in enumerate(counts):
        if c > 0:
            filtered_near.append(all_depth[i])
            filtered_far.append(all_depth[i + 1])

    if not filtered_near:
        raise ValueError("All layers are empty — check near/far range")

    # Merge adjacent kept layers into contiguous ranges, closing any gaps
    # left by removed empties
    merged = [(filtered_near[0], filtered_far[0])]
    for i in range(1, len(filtered_near)):
        prev_near, prev_far = merged[-1]
        cur_near, cur_far = filtered_near[i], filtered_far[i]
        if abs(cur_near - prev_far) < 1e-6:
            # Contiguous — extend previous layer
            merged[-1] = (prev_near, cur_far)
        else:
            # Gap from removed empty layer — close it
            merged[-1] = (prev_near, cur_near)
            merged.append((cur_near, cur_far))

    # Pin to global extents so the full depth range is covered
    merged[0] = (all_depth[0], merged[0][1])
    merged[-1] = (merged[-1][0], all_depth[-1])

    empty_count = len(counts) - sum(1 for c in counts if c > 0)
    if empty_count > 0:
        print(f"Removed {empty_count} empty layers, {len(merged)} layers remain")

    return merged


def write_mxiproc_config(layers, output_dir, texture_format, depth_format=None,
                         fov_radians=np.pi/2, mxitype="mci",
                         config_path=None):
    """
    Write an mxiproc-compatible scene config JSON.

    Args:
        layers: list of (near, far) tuples from compute_hybrid_boundaries
        output_dir: directory where rendered textures live
        texture_format: format string for color textures with two {} placeholders.
                        First {} = layer index, second {} = face name.
                        e.g. "layer_{}/color_{}.exr" -> "layer_0/color_negx.exr"
        depth_format: optional format string for depth textures (same convention)
        fov_radians: vertical FOV per face (pi/2 for standard cubemap)
        mxitype: "mci" for cube, "mpi" for plane
        config_path: output path for the JSON config (default: output_dir/scene_config.json)
    """
    # Extract sorted unique boundary values for mxiproc's layer_depths array
    layer_depths = sorted(set(
        [l[0] for l in layers] + [l[1] for l in layers]
    ))

    config = {
        "mxitype": mxitype,
        "texture": texture_format,
        "layer_depths": layer_depths,
        "fov_in_radians": fov_radians,
        "premultiplied_alpha": True,
        "intput_color_primaries": "color_primaries_p3",
        "output_color_primaries": "color_primaries_p3",
    }
    if depth_format:
        config["depth"] = depth_format

    if config_path is None:
        config_path = os.path.join(output_dir, "scene_config.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Wrote mxiproc config to {config_path}")
    return config_path


# --- Usage ---
if __name__ == "__main__":
    # Depth EXR paths per cubemap face (from your Houdini depth render pass).
    # All faces share the same origin — depth is effectively radial distance.
    depth_dir = "/path/to/depth_renders"
    depth_paths = {
        face: os.path.join(depth_dir, f"depth_{face}.exr")
        for face in FACES
    }

    layers = compute_hybrid_boundaries(
        depth_exr_paths=depth_paths,
        num_layers=8,
        near=0.1,
        far=100.0,
        skip_faces=["posz"],  # Skip back face for 180-degree hemisphere
    )

    print(f"\nFinal layer count: {len(layers)}")
    for i, (n, f) in enumerate(layers):
        print(f"  Layer {i}: near={n:.4f}  far={f:.4f}")

    # Optionally write mxiproc config — shared boundaries for all faces
    # write_mxiproc_config(
    #     layers,
    #     output_dir="/path/to/rendered_layers",
    #     texture_format="layer_{}/color_{}.exr",
    #     depth_format="layer_{}/depth_{}.exr",
    #     config_path="/path/to/scene_config.json",
    # )
