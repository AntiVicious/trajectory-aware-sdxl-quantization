# Reproduction Guide

This document describes how to reproduce the trajectory-aware mixed-precision SDXL
quantization framework from scratch, in the order each phase depends on the
previous one. All commands assume the working directory is the repo root.

## Prerequisites

1. **Hardware.** A single GPU with at least 24 GB of VRAM (the published atlas was
   measured on an RTX A5000 24 GB; the benchmark runs on an L40S 48 GB).
2. **Python environment.** Python 3.10. Create the environment from
   `requirements.txt`:
   ```bash
   conda create -n sdxl python=3.10
   conda activate sdxl
   pip install -r requirements.txt
   ```
3. **Set the data root environment variable.** All scripts read base paths from `$SDXL_DATA_ROOT`. Set it to the directory where data files and benchmark outputs live:
```bash
   export SDXL_DATA_ROOT=/path/to/your/sdxl/working/dir
```
   Add this to `~/.bashrc` to persist it. If unset, the scripts default to `/home/project/Documents/Maharshi/gem` (the original development path).
4. **Public assets** (not in the repo, fetch separately):
   - **COCO val2017 images** for FID computation. Download from
     <https://cocodataset.org/#download> and unpack to a path you'll reference.
   - **DreamSim ensemble model weights** (~3 GB). Fetched automatically on first
     use of the `dreamsim` package.
   - **CLIP ViT-L/14** weights. Fetched automatically by the `transformers`
     library.
   - **PickScore predictor**. Fetched automatically by the `pickscore` package.
5. **5000 prompt subset**. The exact subset of COCO val2017 captions used in the
   published benchmark is `data/benchmark_prompts_5k.json`. Use this file to
   reproduce the same numbers as the paper.

## Phase 1 — Build the sensitivity atlas (≈16 hours on RTX A5000)

The atlas measures the perceptual cost of quantizing each (layer, commit-step)
cell. Three scripts in sequence:

```bash
# 1a. Enumerate the 794 quantizable layers
python src/atlas/enumerate_layers.py

# 1b. Generate the FP16 baseline trajectories (saved as baseline_full.pt)
python src/atlas/build_baseline_full.py

# 1c. Self-check: verify the custom loop matches the standard pipeline (~5 min)
python src/atlas/diagnose_selfcheck.py

# 1d. Run the atlas: 794 layers × 5 commit steps × 8 prompts = 31,760 measurements
python src/atlas/run_atlas_full.py
```

**Output:** `cv_results.csv`-shaped file with 3970 rows (794 layers × 5 commit
steps). The published copy is in `data/predictor/cv_results.csv` for reference.

## Phase 2 — Train the predictor (≈5 minutes)

The predictor maps layer metadata (role, block, parameter count, layer type,
commit timestep) to a predicted log-CLIP sensitivity.

```bash
python src/predictor/train_predictor_full.py
```

**Output:** Predicted CLIP distances for all 3970 cells, plus 5-fold
layer-grouped cross-validation metrics. Published numbers: MAE 0.151, R² 0.968,
Spearman ρ = 0.980 (best variant: GBR with role + block features).

## Phase 3 — Solve the ILP allocator (≈2 minutes per budget)

Given the predictor's per-cell sensitivities, the allocator selects per-layer
quantization choices (FP16 / INT4 at one of 5 commit timesteps) subject to a
memory budget.

```bash
# Solve for all five published budgets: 95%, 85%, 70%, 55%, 40% of the FP16 baseline
python src/allocator/allocate_full.py

# Optional sanity check at very tight budgets
python src/allocator/probe_low_budgets.py
```

**Output:** Five `allocation_<budget>pct_full.json` files in
`data/allocations/`, each containing the per-layer choices the runtime
architectures consume.

## Phase 4 — Run the runtime architectures and benchmark

### 4a. Generate 5000 images for each variant

V13 is the reference implementation; V19-V26 are the deployable runtime
architectures. Each generation pass produces 5000 PNGs (≈1-4 hours per variant
on L40S, depending on the runtime):

```bash
# V13: trajectory + naive INT4 (reference)
python src/allocator/generate_v13_full_5k.py

# V14: SmoothQuant ablation; V15: GPTQ control; V16: random alloc;
# V17: V13 alloc, no trajectory; V18: V12 quantizer, no trajectory
python src/runtimes/generate_v14_5k.py
python src/runtimes/generate_v15_5k.py
python src/runtimes/generate_v16_v17_v18.py

# V19: V13 with V12-style outlier preservation
python src/runtimes/generate_v19.py

# V20-V26: deployable runtime architectures
python src/runtimes/v20_static_nf4.py
python src/runtimes/v21_two_pass_v2.py
python src/runtimes/v22_with_floor.py        # V22 floor5 (recommended)
python src/runtimes/v22_hot_swap.py           # V22 hotswap (no floor)
python src/runtimes/v24b_pre_quantize.py
python src/runtimes/v25_int8_nf4_swap.py     # abandoned; see thesis Sec. 4.x
python src/runtimes/v26_inplace_swap.py
python src/runtimes/v26_multi_image.py
python src/runtimes/v26_with_v13_alloc.py
```

### 4b. Score each variant

Each scoring pass computes CLIP, FID, DreamSim, aesthetic, CLIP-IQA, and
PickScore against the FP16 baseline:

```bash
python src/runtimes/5k_score_final.py
python src/runtimes/5k_score_final_with_ablations.py
python src/runtimes/dreamsim_compare_full.py
python src/runtimes/extra_metrics.py
python src/runtimes/pickscore_recompute.py
```

**Output:** Per-variant metric JSONs (a copy of the published numbers is in
`data/benchmark_scores/`).

### 4c. Measure peak VRAM

```bash
python src/runtimes/measure_vram.py
python src/runtimes/measure_vram_fix1.py   # one-pass fix for V13/V19 measurements
```

**Output:** 26 `vram_measurement_*.json` files (a copy of the published numbers
is in `data/vram_measurements/`).

## Phase 5 — Build thesis/paper artefacts

After the benchmark is complete, the figure-generation scripts produce the
plots referenced in the thesis and paper:

```bash
python scripts/build_all_thesis_artifacts.py
python scripts/generate_atlas_heatmap.py
python scripts/generate_predictor_scatter.py
python scripts/generate_qualitative_grid.py
```

**Output:** PNG figures in `figures/` and `docs/thesis/Pictures/`.

## Variant cheat-sheet

A short reminder of what each variant tests, for navigating `src/runtimes/`:

| Variant | What it isolates | Recommended runtime script |
|---|---|---|
| V13 | Reference: trajectory + naive INT4 | `src/allocator/generate_v13_full_5k.py` |
| V14 | V13 + SmoothQuant per-layer scales | `src/runtimes/generate_v14_5k.py` |
| V15 | V13 + GPTQ per-layer (null result) | `src/runtimes/generate_v15_5k.py` |
| V16 | Random per-layer allocation (control) | `src/runtimes/generate_v16_v17_v18.py` |
| V17 | V13 allocation, no trajectory dimension | `src/runtimes/generate_v16_v17_v18.py` |
| V18 | V12 quantizer + V13 alloc, no trajectory | `src/runtimes/generate_v16_v17_v18.py` |
| V19 | V13 + V12-style outlier preservation | `src/runtimes/generate_v19.py` |
| V20 | Static NF4 lower bound | `src/runtimes/v20_static_nf4.py` |
| V21 | Two-pass FP16 + NF4 (recommended deployable) | `src/runtimes/v21_two_pass_v2.py` |
| V22-floor5 | Per-layer hot-swap (best deployed quality) | `src/runtimes/v22_with_floor.py` |
| V22-hotswap | V22 without sensitivity floor | `src/runtimes/v22_hot_swap.py` |
| V24a | INT8 + NF4 two-pass (early attempt) | `src/runtimes/v24a_int8_nf4_two_pass.py` |
| V24b | Static INT8 PTQ baseline | `src/runtimes/v24b_pre_quantize.py` |
| V25 | INT8 + NF4 CPU swap (abandoned) | `src/runtimes/v25_int8_nf4_swap.py` |
| V26 | In-place INT8 → NF4 swap (proof of concept) | `src/runtimes/v26_inplace_swap.py` |

## Tips for reproducing exact numbers

1. **Use the same seeds.** Prompt `i` uses seed `42 + i`; this is hard-coded
   throughout the benchmark scripts. Do not change it.
2. **Use the same 5000 prompts.** `data/benchmark_prompts_5k.json` defines them.
3. **Use the same bitsandbytes version (0.43.1).** Newer versions may behave
   differently for V25 and V26 due to the `Params4bit` / `Int8Params` issues
   documented in the thesis negative-results section.
4. **Use the same diffusers version (0.27.x).** The custom forward hooks
   referenced in V13/V19/V22 rely on specific diffusers internals.
5. **Use the same UNet config.** SDXL's `force_zeros_for_empty_prompt=True`
   flag in the text-encoder config is required for the self-check protocol to
   produce a bit-identical FP16 baseline. Atlas measurements assume this
   setting.

## Where to look if something fails

- **Atlas measurements take far longer than 16 hours.** Verify `baseline_full.pt`
  is being reused as the prefix — without it, every cell re-runs FP16 generation
  from scratch.
- **Predictor R² is much worse than 0.96.** Check that the 5-fold split is
  grouped by layer name (no layer should appear in both train and test of any
  fold). The `GroupKFold` instantiation in `train_predictor_full.py` does this.
- **V22 collapses at 40% budget.** Use the `--floor 5` flag (this is V22-floor5).
  Without the floor, 128 layers commit at t=0, which collapses identity. The
  thesis explains why (Section 5.5).
- **bitsandbytes assertion in V26 after ~50 images.** This is the documented
  `Params4bit` corruption. Restart the process or limit batch size; the
  workaround is documented in the thesis V26 implementation notes.

## Compute budget estimate

For a full end-to-end recreation of all published numbers:

- Atlas: ~16 hours (single GPU)
- Predictor: ~5 minutes (CPU is fine)
- Allocator: ~10 minutes (CPU only)
- Benchmark (≈40 variants × 5000 images): ~60-120 GPU-hours total
- Scoring: ~5 hours (DreamSim is the slowest)
- **Total: ~80-150 GPU-hours.** Run in parallel across multiple GPUs if available.
