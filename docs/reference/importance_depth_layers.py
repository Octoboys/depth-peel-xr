"""
Importance-Sampled Depth Layer Calculator

Computes depth layer boundaries from cubemap face depth renders using
importance sampling (histogram -> CDF -> inverse CDF). Produces one shared
set of boundaries for all cubemap faces by pooling depth from all active faces.

Depth values from EXR renders are camera-space Z per face. Since all cubemap
face cameras share the same origin, these are equivalent to radial distance
and produce consistent layer boundaries across face borders.

Usage:
    python importance_depth_layers.py

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


def compute_importance_boundaries(depth_exr_paths, num_layers, near, far,
                                  skip_faces=None):
    """
    Compute importance-sampled depth layer boundaries from cubemap face
    depth renders. Pools depth from all active faces to produce one shared
    set of boundaries — consistent across face borders.

    Importance sampling places layer boundaries at equal-probability intervals
    of the depth CDF, so dense regions (lots of content) get more layers and
    sparse regions get fewer. No empty layers by construction.

    Args:
        depth_exr_paths: dict mapping face name -> EXR path
                         e.g. {"negz": "/path/depth_negz.exr", ...}
        num_layers: desired number of depth layers
        near: near clip distance (scene units)
        far: far clip distance (scene units)
        skip_faces: list of face names to skip
                    e.g. ["posz"] for 180-degree hemisphere

    Returns:
        List of (near_depth, far_depth) tuples for each layer.
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

    # Histogram -> CDF (10,000 bins for smooth distribution)
    hist, edges = np.histogram(ndc, bins=10000, range=(ndc_min, ndc_max))
    pdf = hist / hist.sum()
    cdf = np.cumsum(pdf)

    # Inverse CDF sampling at uniform intervals [0, 1/N, 2/N, ..., 1].
    # Each layer gets an equal share of the total depth content.
    boundaries_ndc = []
    for i in range(num_layers + 1):
        target = i / num_layers
        bin_idx = min(np.searchsorted(cdf, target), len(edges) - 1)
        boundaries_ndc.append(edges[bin_idx])

    # Convert NDC back to real depth
    boundaries_depth = [B / (n - A) for n in boundaries_ndc]

    # Pair into contiguous layers
    layers = []
    for i in range(len(boundaries_depth) - 1):
        layers.append((boundaries_depth[i], boundaries_depth[i + 1]))

    return layers


def write_mxiproc_config(layers, output_dir, texture_format, depth_format=None,
                         fov_radians=np.pi/2, mxitype="mci",
                         config_path=None):
    """
    Write an mxiproc-compatible scene config JSON.

    Args:
        layers: list of (near, far) tuples from compute_importance_boundaries
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

    layers = compute_importance_boundaries(
        depth_exr_paths=depth_paths,
        num_layers=8,
        near=0.1,
        far=100.0,
        skip_faces=["posz"],  # Skip back face for 180-degree hemisphere
    )

    print(f"\nLayer count: {len(layers)}")
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
