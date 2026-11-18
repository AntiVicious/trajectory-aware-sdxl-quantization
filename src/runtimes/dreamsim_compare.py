"""
Compute DreamSim distance between each V13 variant and V1 FP16 baseline.

DreamSim is a perceptual similarity metric that correlates with human
judgment better than LPIPS or pixel-level metrics. Lower = more similar
to baseline. Two images of the same scene/composition score near 0.

For each variant (V13-70, V13-55, V13-40), compute mean DreamSim across
all 5000 paired images vs V1 FP16. Also include V12 for reference.

Output:
  benchmark_5k/dreamsim_results.json
  prints summary table

Usage:
    pip install dreamsim  # if not already installed
    CUDA_VISIBLE_DEVICES=0 python dreamsim_compare.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import time
from pathlib import Path
import torch
from PIL import Image
import numpy as np

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
OUT = BASE / "benchmark_5k"
PROMPTS = json.load(open(BASE / "benchmark_prompts_5k.json"))

REFERENCE_VARIANT = "v1_fp16"
COMPARE_VARIANTS = [
    "v2_int4_naive",
    "v12_outlier_preserved",
    "v13_70pct_trajectory",
    "v13_55pct_trajectory",
    "v13_40pct_trajectory",
]


def main():
    print("Loading DreamSim...")
    try:
        from dreamsim import dreamsim
    except ImportError:
        print("DreamSim not installed. Install with: pip install dreamsim")
        return

    # DreamSim returns (model, preprocess) pair
    model, preprocess = dreamsim(pretrained=True, device="cuda")
    model.eval()

    ref_dir = OUT / REFERENCE_VARIANT
    if not ref_dir.exists():
        print(f"ERROR: reference {ref_dir} not found")
        return

    print(f"\nReference: {REFERENCE_VARIANT}")
    print(f"Comparing: {COMPARE_VARIANTS}")
    print()

    results = {}
    for variant in COMPARE_VARIANTS:
        var_dir = OUT / variant
        if not var_dir.exists():
            print(f"  [{variant}] folder missing — skipping")
            continue

        print(f"Processing {variant}...")
        t0 = time.time()
        distances = []
        n_compared = 0
        n_skipped = 0

        for i, item in enumerate(PROMPTS):
            ref_path = ref_dir / item['filename']
            var_path = var_dir / item['filename']

            if not (ref_path.exists() and var_path.exists()):
                n_skipped += 1
                continue

            try:
                img_ref = preprocess(Image.open(ref_path).convert("RGB")).to("cuda")
                img_var = preprocess(Image.open(var_path).convert("RGB")).to("cuda")
                with torch.no_grad():
                    dist = model(img_ref, img_var).item()
                distances.append(dist)
                n_compared += 1
            except Exception as e:
                print(f"    error on {item['filename']}: {e}")
                n_skipped += 1

            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(PROMPTS) - i - 1) / rate
                print(f"  {i+1}/{len(PROMPTS)}  "
                      f"({elapsed/60:.1f}min, ETA {eta/60:.1f}min)  "
                      f"running mean={np.mean(distances):.4f}")

        if not distances:
            print(f"  [{variant}] no images compared!")
            continue

        d = np.array(distances)
        results[variant] = {
            "n_compared": n_compared,
            "n_skipped": n_skipped,
            "mean": float(d.mean()),
            "std": float(d.std()),
            "median": float(np.median(d)),
            "p25": float(np.percentile(d, 25)),
            "p75": float(np.percentile(d, 75)),
            "max": float(d.max()),
        }
        elapsed = (time.time() - t0) / 60
        print(f"  [{variant}] mean DreamSim={d.mean():.4f}  "
              f"std={d.std():.4f}  median={np.median(d):.4f}  "
              f"({n_compared} pairs in {elapsed:.1f} min)")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"DreamSim distances vs {REFERENCE_VARIANT} (lower = closer to FP16)")
    print("=" * 80)
    print(f"{'Variant':<30} {'N':>6} {'Mean':>8} {'Std':>8} {'Median':>8} {'P75':>8}")
    print("-" * 80)
    for v in COMPARE_VARIANTS:
        if v in results:
            r = results[v]
            print(f"{v:<30} {r['n_compared']:>6} {r['mean']:>8.4f} "
                  f"{r['std']:>8.4f} {r['median']:>8.4f} {r['p75']:>8.4f}")

    out_path = OUT / "dreamsim_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "reference": REFERENCE_VARIANT,
            "metric": "DreamSim (LDS distance)",
            "n_total_prompts": len(PROMPTS),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Interpretation
    print("\nInterpretation:")
    print("  DreamSim ~0.00-0.05: indistinguishable from FP16")
    print("  DreamSim ~0.05-0.15: visible differences but same scene")
    print("  DreamSim ~0.15-0.30: noticeably different scene/composition")
    print("  DreamSim >0.30:      different image content")


if __name__ == "__main__":
    main()
