"""
Run only CLIP-IQA and PickScore. Aesthetic already completed in extra_metrics.log.

Usage:
    pip install setuptools
    CUDA_VISIBLE_DEVICES=0 python extra_metrics_part2.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extra_metrics import (
    compute_clip_iqa, compute_pickapic, filter_existing_variants, VARIANTS,
)

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "/home/project/Documents/Maharshi/gem"))
OUT = BASE / "benchmark_5k"

import torch

variants = filter_existing_variants(VARIANTS)
print(f"Variants present: {variants}")

clip_iqa = compute_clip_iqa(variants)
torch.cuda.empty_cache()

pickapic = compute_pickapic(variants, baseline="v1_fp16")

# Load aesthetic from previous run if available
prev_path = OUT / "extra_metrics_results.json"
aesthetic = {}
if prev_path.exists():
    try:
        prev = json.load(open(prev_path))
        aesthetic = prev.get("aesthetic", {})
        print(f"\nLoaded {len(aesthetic)} aesthetic results from previous run")
    except Exception as e:
        print(f"Warning: couldn't load previous: {e}")

# Print summary
print("\n" + "=" * 100)
print(f"{'Variant':<30} {'Aesthetic':>10} {'CLIP-IQA':>10} {'Pick-vs-V1 win':>15}")
print("=" * 100)
for v in variants:
    aes = f"{aesthetic.get(v, {}).get('mean', '—'):.3f}" if v in aesthetic else "—"
    ciq = f"{clip_iqa.get(v, {}).get('mean', '—'):.4f}" if v in clip_iqa else "—"
    if v == "v1_fp16":
        pap = "(ref)"
    else:
        pap = f"{pickapic.get(v, {}).get('win_rate', '—'):.3f}" if v in pickapic else "—"
    print(f"{v:<30} {aes:>10} {ciq:>10} {pap:>15}")

# Save merged
merged = {
    "aesthetic": aesthetic,
    "clip_iqa": clip_iqa,
    "pickapic_vs_v1": pickapic,
}
out_path = OUT / "extra_metrics_results.json"
with open(out_path, "w") as f:
    json.dump(merged, f, indent=2)
print(f"\nSaved: {out_path}")