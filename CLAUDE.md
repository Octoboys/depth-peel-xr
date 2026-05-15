# Depth Peel XR — project guide for Claude Code

This file is the portable source of context for working on this repo. It auto-loads when Claude Code is invoked in this directory on any machine. Keep it in version control.

## Project overview

**Depth Peel XR** — Blender 5.x add-on that optimises a 3D environment for XR viewing from a bounded head-plane in front of a rectangular portal. The add-on re-interprets the scene as a small number of depth-peeled layers — each an adaptive quad mesh displaced to real scene depth with a baked colour texture — so the runtime (an external engine consuming USD) pays only the cost of a handful of textured, displaced quads while preserving parallax for limited head motion.

Pipeline phases, in order: **Phase 0** dev bootstrap → **Phase 1** Portal Rig → **Phase 2** Layer Placement → **Phase 3** Multi-view Capture → **Phase 4** Reconstruction + Preview → **Phase 5** Occlusion Culling → **Phase 6** USD Export. Each phase signs off independently before the next starts.

The authoritative implementation plan — locked-in decisions, architecture, phase scope, acceptance criteria, verification steps — lives at **`docs/implementation_plan.md`** in this repo. Read it at the start of any new session before making changes.

## User profile

Experienced realtime games tech artist. Fluent in real-time rendering concepts (depth peeling, impostors, LODs, overdraw, PVS, displacement, MPI), XR/VR constraints on mobile/standalone headsets, DCC tools (Blender primary, Houdini prior work), USD-based pipelines, Python scripting for DCC.

Implications:
- Skip basic explanations; they already understand depth buffers, NDC, CDF sampling, impostor baking, etc.
- Reason in poly count / texel density / overdraw / memory / runtime shader cost when proposing trade-offs.
- They propose solid technical ideas themselves — engage substantively, don't just accept.
- Treat their existing code/scripts as real prior art worth integrating.

## Collaboration style (important)

**Ask one clarifying question per turn during planning.** Do not batch multiple unrelated design questions into a single turn. Pick the single most impactful unresolved question; wait for the answer before asking the next. Why: the user thinks carefully about each decision in isolation and often reshapes the next question based on the previous answer.

**Phase-by-phase implementation with explicit sign-off.** Ship one phase at a time, produce a concrete testable artefact, summarise what was done + how to test it, and **pause for the user to approve** before starting the next phase. Do not auto-advance. Define phase dependencies explicitly in task tracking (`blockedBy`).

**Do not use "P4" as a phase label** — it collides with Perforce in the user's workflow. Write out `Phase 1`, `Phase 2`, etc. in full. Applies to file/dir names too (no `p4_*` paths).

**Dedicated cleanup operators** for anything that modifies Blender scene state. Every operator that adds entities has a matching cleanup, plus a master `Clean up all`.

**Project-scoped output folders** — all generated files (captures, temp, USD) live under `<blend_dir>/depth_peel_xr/<scene_name>/`, not scattered across the user's filesystem.

## N-panel UI conventions

The `Depth Peel XR` N-panel uses a single, consistent layout pattern. Stick to it when adding new sections — the user has dialed it in deliberately and corrections cost a turn each.

**One aligned column per section.** After each section's `box.label(...)` header, open `col = box.column(align=True)` and stack everything (params, status labels, action buttons) under that one column. Do not mix in other layout containers at the box level.

**Default `col.separator()` for every visual break.** Adjacent items in the column render flush (shared borders, no gap). Drop in a default `col.separator()` to introduce a small consistent gap. Same primitive for all small breaks: param-group → param-group, action → action, etc.

**`col.separator(factor=1.5)` for the param-block → action-button transition.** This is the *only* place a non-default factor is used — it marks the visual jump from "data inputs" to "things to do with them". Don't reach for other custom factors.

**Avoid `box.separator()`** directly on the box — renders bigger and inconsistent with the in-aligned-column separators.

**Conditional enable on rows, not ops.** When an operator should be greyed out conditionally, wrap it in `place_row = col.row(align=True)`, set `place_row.enabled = condition`, then call `place_row.operator(...)`. Keeps the column structure unbroken.

**Proportional rows (icon-only ops with fixed widths).** Use nested `layout.split(factor=...)` calls for explicit width proportions — *not* `row(align=True) + scale_x` on sub-rows. Blender shrink-wraps icon-only ops to natural width and ignores `scale_x`, leaving empty space at the panel edges. For an A/B/C-proportioned row:
```python
outer = layout.split(factor=A_fraction, align=True)
outer.<item-A>(...)               # fills slot 1 (A%)
inner = outer.split(factor=B_fraction_of_remaining, align=True)
inner.<item-B>(...)
inner.<item-C>(...)
```

## Reference material (in-repo)

User-authored Python scripts implementing the layer-placement algorithms being ported into the add-on:

- `docs/reference/hybrid_depth_layers.py` — hybrid importance + uniform NDC sampling with empty-layer removal. Being ported into `depth_peel_xr/core/layer_placement/histogram_hybrid.py`.
- `docs/reference/importance_depth_layers.py` — pure importance-sampling variant; reference for the CDF logic, not ported directly.

Reusable pieces: NDC transform, histogram → CDF inverse sampling, empty-layer detection and gap-closing. Port preserves the proven maths; input changes from EXR paths to numpy arrays from Blender render results.

## Development environment

Primary workflow: **VS Code + Jacques Lucke's "Blender Development" extension**. See `docs/implementation_plan.md` → "Development workflow" for full setup instructions.

- `tools/install_dev_symlink.sh` symlinks `./depth_peel_xr/` into Blender's extensions directory so edits land live.
- `requirements-dev.txt` pins `fake-bpy-module-latest`, `pytest`, `numpy`, `usd-core` for autocomplete and core/ testing.
- `depth_peel_xr/dev_reload.py` is the manual reload fallback when VS Code isn't in play.
- `core/` never imports `bpy` — all pure-math modules are testable via pytest without Blender.

## When starting a new session

1. Read `docs/implementation_plan.md` to understand the current phase and acceptance criteria.
2. Check `git log` and the working tree to see which phases are already implemented vs. still pending.
3. Ask the user which phase to work on (don't assume).
4. Follow the collaboration style above — one question at a time if planning; phase-by-phase sign-off for implementation.
