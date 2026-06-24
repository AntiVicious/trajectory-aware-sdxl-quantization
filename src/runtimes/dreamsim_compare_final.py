"""
DreamSim distance for V14 and V15 vs V1 baseline (V13 already computed).

Loads previous DreamSim results (if present) and adds new variants.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u dreamsim_compare_final.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import time
from pathlib import Path
import torch
from PIL import Image
import numpy as np

BASE = Path("/home/project/Documents/Maharshi/gem")
OUT = BASE / "benchmark_5k"
PROMPTS = json.load(open(BASE / "benchmark_prompts_5k.json"))

REFERENCE = "v1_fp16"
NEW_VARIANTS = [
    "v14_70pct_smoothquant",
    "v14_55pct_smoothquant",
    "v14_40pct_smoothquant",
    "v15_70pct_gptq",
    "v15_55pct_gptq",
    "v15_40pct_gptq",
]


def main():
    # Load existing results
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

    results = dict(prev_results)
    for variant in NEW_VARIANTS:
        var_dir = OUT / variant
        if not var_dir.exists():
            print(f"  [{variant}] folder missing — skipping", flush=True)
            continue
        n_existing = len([f for f in var_dir.iterdir() if f.suffix == '.jpg'])
        if n_existing < 100:
            print(f"  [{variant}] only {n_existing} images — skipping", flush=True)
            continue

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
            if (i + 1) % 500 == 0:
                elapsed = (time.time() - t0) / 60
                rate = (i + 1) / elapsed
                eta = (len(PROMPTS) - i - 1) / rate
                print(f"  {i+1}/{len(PROMPTS)}  ({elapsed:.1f}min, ETA {eta:.1f}min)  "
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

    # Print full summary including previous variants
    ALL_VARIANTS = [
        "v2_int4_naive",
        "v12_outlier_preserved",
        "v13_70pct_trajectory",
        "v13_55pct_trajectory",
        "v13_40pct_trajectory",
        "v14_70pct_smoothquant",
        "v14_55pct_smoothquant",
        "v14_40pct_smoothquant",
        "v15_70pct_gptq",
        "v15_55pct_gptq",
        "v15_40pct_gptq",
    ]
    print("\n" + "=" * 80)
    print(f"DreamSim distances vs {REFERENCE} (lower = closer to FP16)")
    print("=" * 80)
    print(f"{'Variant':<30} {'N':>6} {'Mean':>8} {'Std':>8} {'Median':>8}")
    print("-" * 80)
    for v in ALL_VARIANTS:
        if v in results:
            r = results[v]
            print(f"{v:<30} {r['n_compared']:>6} {r['mean']:>8.4f} "
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
