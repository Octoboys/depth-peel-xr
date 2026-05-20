"""Pre-cull cache: holds the as-built (opaque, alpha) `DisplacedLayerMesh`
halves for each layer's reconstruction, keyed by base_name (= the bpy
object base name, e.g. `dpxr_recon_layer_03` — the spawned objects are
`<base>_opaque` and `<base>_alpha`).

Populated by the reconstruction operators before bpy handoff; consumed
by Cull Occluded Quads (filtered copies used to rebuild the live meshes,
opaque half rasterised for the closer-surface) and Undo Cull (restores
the live meshes from the original cached arrays). Cleared by Clean Up
Geo and on add-on unregister.

Lives in operators/ rather than core/ because both reconstruct.py and
occlusion_cull.py need it; keeping it here avoids a circular import
(occlusion_cull → reconstruct → occlusion_cull) while preserving the
core/-has-no-bpy invariant.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.reconstruction import DisplacedLayerMesh


@dataclass(frozen=True)
class LayerCachePair:
    opaque: DisplacedLayerMesh
    alpha: DisplacedLayerMesh


_CACHE: dict[str, LayerCachePair] = {}


def set_layer_cache(
    name: str, opaque: DisplacedLayerMesh, alpha: DisplacedLayerMesh
) -> None:
    _CACHE[name] = LayerCachePair(opaque=opaque, alpha=alpha)


def get_layer_cache(name: str) -> LayerCachePair | None:
    return _CACHE.get(name)


def all_cached_names() -> list[str]:
    return list(_CACHE.keys())


def clear_cache() -> None:
    _CACHE.clear()
