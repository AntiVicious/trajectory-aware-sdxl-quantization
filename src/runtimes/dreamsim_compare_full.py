"""
DreamSim distance for all variants vs V1 baseline.

Loads previous DreamSim results (if present) and adds new variants.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -u dreamsim_compare_full.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
import time
from pathlib import Path
import torch
from PIL import Image
import numpy as np

BASE = Path.home() / "Documents" / "Maharshi" / "gem"
OUT = BASE / "benchmark_5k"
PROMPTS = json.load(open(BASE / "benchmark_prompts_5k.json"))

REFERENCE = "v1_fp16"

# All variants we want DreamSim for. Existing results are reused; new ones computed.
ALL_VARIANTS = [
    # Prior V1-V12 (selected)
    "v2_int4_naive",
    "v12_outlier_preserved",
    # V13 trajectory (full)
    "v13_70pct_trajectory",
    "v13_55pct_trajectory",
    "v13_40pct_trajectory",
    # V14 trajectory + SmoothQuant
    "v14_70pct_smoothquant",
    "v14_55pct_smoothquant",
    "v14_40pct_smoothquant",
    # V15 trajectory + GPTQ
    "v15_70pct_gptq",
    "v15_55pct_gptq",
    "v15_40pct_gptq",
    # V16 random allocation (ablation)
    "v16_70pct_random_alloc",
    "v16_55pct_random_alloc",
    "v16_40pct_random_alloc",
    # V17 V13 alloc, no trajectory (ablation)
    "v17_70pct_no_trajectory",
    "v17_55pct_no_trajectory",
    "v17_40pct_no_trajectory",
    # V18 V13 alloc, V12 quantizer (ablation)
    "v18_70pct_v12_quantizer",
    "v18_55pct_v12_quantizer",
    "v18_40pct_v12_quantizer",
    "v19_70pct_trajectory_v12quant",
    "v19_55pct_trajectory_v12quant",
    "v19_40pct_trajectory_v12quant",
]


def main():
    # Load existing results to avoid recomputing
    prev_path = OUT / "dreamsim_results.json"
    prev_results = {}
    if prev_path.exists():
        try:
            prev = json.load(open(prev_path))
            prev_results = prev.get("results", {})
            print(f"Loaded {len(prev_results)} previous DreamSim results", flush=True)
        except Exception as e:
            print(f"Warning: couldn't load previous: {e}", flush=True)

    print("Loading DreamSim...", flush=True)
    from dreamsim import dreamsim
    model, preprocess = dreamsim(pretrained=True, device="cuda")
    model.eval()

    ref_dir = OUT / REFERENCE
    if not ref_dir.exists():
        print(f"ERROR: reference {ref_dir} not found")
        return

    # Determine which variants to compute (skip those with cached results)
    todo = []
    for v in ALL_VARIANTS:
        var_dir = OUT / v
        if not var_dir.exists():
            print(f"  [{v}] folder missing — skipping", flush=True)
            continue
        n_existing = len([f for f in var_dir.iterdir() if f.suffix == '.jpg'])
        if n_existing < 100:
            print(f"  [{v}] only {n_existing} images — skipping", flush=True)
            continue
        if v in prev_results and prev_results[v].get("n_compared", 0) >= 4500:
            print(f"  [{v}] cached (n={prev_results[v]['n_compared']}) — skipping", flush=True)
            continue
        todo.append(v)

    print(f"\n{len(todo)} variants to compute DreamSim for", flush=True)

    results = dict(prev_results)
    for variant in todo:
        var_dir = OUT / variant
        print(f"\nProcessing {variant}...", flush=True)
        t0 = time.time()
        distances = []

        for i, item in enumerate(PROMPTS):
            ref_path = ref_dir / item['filename']
            var_path = var_dir / item['filename']
            if not (ref_path.exists() and var_path.exists()):
                continue
            try:
                img_ref = preprocess(Image.open(ref_path).convert("RGB")).to("cuda")
                img_var = preprocess(Image.open(var_path).convert("RGB")).to("cuda")
                with torch.no_grad():
                    dist = model(img_ref, img_var).item()
                distances.append(dist)
            except Exception as e:
                pass
            if (i + 1) % 1000 == 0:
                elapsed = (time.time() - t0) / 60
                print(f"  {i+1}/{len(PROMPTS)}  ({elapsed:.1f}min) "
                      f"running mean={np.mean(distances):.4f}", flush=True)

        if not distances:
            continue
        d = np.array(distances)
        results[variant] = {
            "n_compared": len(distances),
            "mean": float(d.mean()),
            "std": float(d.std()),
            "median": float(np.median(d)),
            "p25": float(np.percentile(d, 25)),
            "p75": float(np.percentile(d, 75)),
        }
        elapsed = (time.time() - t0) / 60
        print(f"  [{variant}] mean DreamSim={d.mean():.4f}  "
              f"std={d.std():.4f}  ({len(distances)} pairs, {elapsed:.1f}min)", flush=True)

    # Print full summary
    print("\n" + "=" * 80)
    print(f"DreamSim distances vs {REFERENCE} (lower = closer to FP16)")
    print("=" * 80)
    print(f"{'Variant':<35} {'N':>6} {'Mean':>8} {'Std':>8} {'Median':>8}")
    print("-" * 80)
    for v in ALL_VARIANTS:
        if v in results:
            r = results[v]
            print(f"{v:<35} {r['n_compared']:>6} {r['mean']:>8.4f} "
                  f"{r['std']:>8.4f} {r['median']:>8.4f}")

    out_path = OUT / "dreamsim_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "reference": REFERENCE,
            "metric": "DreamSim (LDS distance)",
            "n_total_prompts": len(PROMPTS),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()