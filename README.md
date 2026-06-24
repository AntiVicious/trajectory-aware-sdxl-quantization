# Trajectory-Aware Mixed-Precision Quantization of Stable Diffusion XL

This repository contains the framework, runtime architectures, and experimental
artefacts for trajectory-aware mixed-precision quantization of SDXL, as described
in the M.Tech thesis and a BMVC 2026 submission.

## Repository structure

- `src/` — Source code organised by phase:
  - `atlas/` — Sensitivity atlas construction (Phase 1)
  - `predictor/` — Structural-feature predictor (Phase 2)
  - `allocator/` — ILP-based bit allocator (Phase 3)
  - `runtimes/` — V13-V26 runtime architectures (Phase 4)
  - `benchmark/` — 5000-image benchmark machinery
- `data/` — Aggregate metrics, atlas CSV, allocations, VRAM measurements
- `figures/` — Key result figures
- `logs/` — Final-run logs (sanity logs excluded)
- `scripts/` — Entry-point CLIs and figure-generation scripts
- `docs/` — Thesis source + paper supplementary
- `archive/v1_to_v12/` — Legacy V1-V12 variants (superseded; kept for reference)

## Heavy artefacts (not in this repo)

Three things are too large for git and stay on the compute node:

- `models/` (3 GB) — DreamSim/CLIP/DINO pretrained weights. Re-fetch via the
  upstream sources listed in `scripts/download_models.sh`.
- `val2017/` (1.6 GB) — Public COCO val2017 used as the FID reference.
  Download from <https://cocodataset.org/>.
- `benchmark_5k/` (15 GB) — 5000 generated images for each of 49 variants.
  Aggregate scores for every variant are in `data/benchmark_scores/`.

## Reproducibility

See `docs/paper/supplementary/F_reproducibility/hardware_notes.md` for the
hardware, software versions, and determinism notes.
