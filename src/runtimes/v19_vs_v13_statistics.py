"""
Statistical comparison of V13-70 vs V19-70 on per-image DreamSim.

Tests whether V19's lower mean DreamSim is a real effect or just noise.
Uses paired t-test and Wilcoxon signed-rank test (paired = same prompt+seed).

Also bins images by V13's DreamSim value to check whether V19 helps uniformly
or only on specific subsets.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -u v19_vs_v13_statistics.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
import time
from pathlib import Path
import torch
from PIL import Image
import numpy as np
from scipy import stats

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
OUT = BASE / "benchmark_5k"
PROMPTS = json.load(open(BASE / "benchmark_prompts_5k.json"))

V1_DIR = OUT / "v1_fp16"
V13_DIR = OUT / "v13_70pct_trajectory"
V19_DIR = OUT / "v19_70pct_trajectory_v12quant"


def main():
    print("Loading DreamSim model...")
    from dreamsim import dreamsim
    model, preprocess = dreamsim(pretrained=True, device="cuda")
    model.eval()

    print("Computing per-image DreamSim for V13 and V19 vs V1...")
    print(f"  Total prompts: {len(PROMPTS)}")
    t0 = time.time()

    d13_list = []
    d19_list = []
    filenames = []
    captions = []

    for i, item in enumerate(PROMPTS):
        fn = item['filename']
        p_v1 = V1_DIR / fn
        p_v13 = V13_DIR / fn
        p_v19 = V19_DIR / fn
        if not (p_v1.exists() and p_v13.exists() and p_v19.exists()):
            continue

        try:
            img_v1 = preprocess(Image.open(p_v1).convert("RGB")).to("cuda")
            img_v13 = preprocess(Image.open(p_v13).convert("RGB")).to("cuda")
            img_v19 = preprocess(Image.open(p_v19).convert("RGB")).to("cuda")
            with torch.no_grad():
                d13 = model(img_v1, img_v13).item()
                d19 = model(img_v1, img_v19).item()
        except Exception as e:
            continue

        d13_list.append(d13)
        d19_list.append(d19)
        filenames.append(fn)
        captions.append(item['caption'])

        if (i + 1) % 500 == 0:
            elapsed = (time.time() - t0) / 60
            print(f"  {i+1}/{len(PROMPTS)}  ({elapsed:.1f}min)  "
                  f"V13={np.mean(d13_list):.4f} V19={np.mean(d19_list):.4f}",
                  flush=True)

    d13 = np.array(d13_list)
    d19 = np.array(d19_list)
    diff = d13 - d19  # positive = V19 closer to V1 (better)

    print(f"\n{'='*70}")
    print(f"PAIRED COMPARISON: V13-70 vs V19-70 ({len(d13)} pairs)")
    print(f"{'='*70}")
    print(f"\nMeans:")
    print(f"  V13-70 DreamSim: {d13.mean():.4f}  (std {d13.std():.4f})")
    print(f"  V19-70 DreamSim: {d19.mean():.4f}  (std {d19.std():.4f})")
    print(f"  Diff (V13 - V19): {diff.mean():+.4f}  (std {diff.std():.4f})")

    print(f"\nMedians:")
    print(f"  V13-70: {np.median(d13):.4f}")
    print(f"  V19-70: {np.median(d19):.4f}")

    # Win/loss/tie counts
    THRESHOLD = 0.005  # treat differences smaller than this as "tied"
    v19_wins = np.sum(diff > THRESHOLD)
    v13_wins = np.sum(diff < -THRESHOLD)
    ties = np.sum(np.abs(diff) <= THRESHOLD)
    print(f"\nPer-image winners (threshold {THRESHOLD}):")
    print(f"  V19 closer to V1: {v19_wins}  ({100*v19_wins/len(diff):.1f}%)")
    print(f"  V13 closer to V1: {v13_wins}  ({100*v13_wins/len(diff):.1f}%)")
    print(f"  Tied:             {ties}  ({100*ties/len(diff):.1f}%)")

    # Statistical tests
    t_stat, t_pval = stats.ttest_rel(d13, d19)
    w_stat, w_pval = stats.wilcoxon(d13, d19)
    print(f"\nStatistical tests (null: V13 = V19):")
    print(f"  Paired t-test:           t={t_stat:.3f}, p={t_pval:.2e}")
    print(f"  Wilcoxon signed-rank:    W={w_stat:.0f}, p={w_pval:.2e}")
    if t_pval < 0.001:
        print(f"  -> Difference is highly statistically significant.")
    elif t_pval < 0.05:
        print(f"  -> Difference is statistically significant.")
    else:
        print(f"  -> Difference is NOT statistically significant (within noise).")

    # Cohen's d (effect size)
    pooled_std = np.sqrt((d13.var() + d19.var()) / 2)
    cohens_d = diff.mean() / pooled_std
    print(f"  Cohen's d (effect size): {cohens_d:.3f}")
    if abs(cohens_d) < 0.2:
        print(f"  -> Negligible effect size.")
    elif abs(cohens_d) < 0.5:
        print(f"  -> Small effect size.")
    elif abs(cohens_d) < 0.8:
        print(f"  -> Medium effect size.")
    else:
        print(f"  -> Large effect size.")

    # Binned analysis: where does V19 help?
    print(f"\n{'='*70}")
    print(f"BINNED ANALYSIS — does V19 help uniformly or only at certain quality?")
    print(f"{'='*70}")
    bins = [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0)]
    print(f"{'V13 DSim range':<20} {'N':>6} {'V13 mean':>10} {'V19 mean':>10} "
          f"{'Diff':>10} {'V19 better %':>14}")
    print("-" * 75)
    for lo, hi in bins:
        mask = (d13 >= lo) & (d13 < hi)
        n = mask.sum()
        if n == 0:
            continue
        d13_sub = d13[mask]
        d19_sub = d19[mask]
        diff_sub = d13_sub - d19_sub
        v19_better = np.sum(diff_sub > THRESHOLD) / n * 100
        print(f"  [{lo:.2f}, {hi:.2f})       {n:>6} "
              f"{d13_sub.mean():>10.4f} {d19_sub.mean():>10.4f} "
              f"{diff_sub.mean():>+10.4f} {v19_better:>13.1f}%")

    # Save the per-image results for later figure generation
    out_path = OUT / "v13_v19_per_image_dreamsim.json"
    with open(out_path, "w") as f:
        json.dump({
            "n": len(d13_list),
            "v13": [float(x) for x in d13_list],
            "v19": [float(x) for x in d19_list],
            "filenames": filenames,
        }, f)
    print(f"\nPer-image data saved: {out_path}")

    # Print top 30 V19-wins and V13-wins for inspection
    sorted_idx = np.argsort(diff)
    print(f"\n=== Top 30 V13 wins (V13 much closer to V1 than V19) ===")
    for idx in sorted_idx[:30]:
        print(f"  fn={filenames[idx]}  d13={d13[idx]:.4f}  d19={d19[idx]:.4f}  Δ={diff[idx]:+.4f}")
        print(f"        '{captions[idx][:80]}'")

    print(f"\n=== Top 30 V19 wins (V19 much closer to V1 than V13) ===")
    for idx in sorted_idx[-30:][::-1]:
        print(f"  fn={filenames[idx]}  d13={d13[idx]:.4f}  d19={d19[idx]:.4f}  Δ={diff[idx]:+.4f}")
        print(f"        '{captions[idx][:80]}'")


if __name__ == "__main__":
    main()
