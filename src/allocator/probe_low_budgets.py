"""
Probe the lower compression frontier:
  - What's the lowest feasible budget?
  - What's the predicted quality (LPIPS sum) at each step toward the floor?

This adds budget points 35%, 32%, 31.5%, 31% to the allocator.
Doesn't generate any images — just tests feasibility.
Run this first to see if V13 at V12-equivalent memory is even possible.

Usage:
    python probe_low_budgets.py
"""
import json
import pickle
from pathlib import Path
import numpy as np
import pulp

# Reuse Phase 3 helpers
import sys
sys.path.insert(0, str(Path(__file__).parent))
from allocate_full import (
    classify_role, classify_block_depth, feature_vec,
    build_candidates, ilp_allocate,
    LAYERS_JSON, PREDICTOR_PATH,
)

with open(LAYERS_JSON) as f:
    layers_meta = json.load(f)["layers"]
with open(PREDICTOR_PATH, "rb") as f:
    pack = pickle.load(f)
predictor = pack["model"]
pred_meta = pack["meta"]

print(f"[probe] {len(layers_meta)} layers, predictor: {pack['variant']}")
cand_rows = build_candidates(layers_meta, predictor, pred_meta)

fp16_total_b = sum(c["bytes"] for c in cand_rows if c["choice"] == "FP16")
int4_min_b = sum(next(c for c in cand_rows
                      if c["layer"] == l["name"] and c["choice"] == "INT4@t0")["bytes"]
                 for l in layers_meta)
print(f"[probe] FP16 total:    {fp16_total_b/1e6:.1f} MB")
print(f"[probe] INT4@t0 floor: {int4_min_b/1e6:.1f} MB ({int4_min_b/fp16_total_b*100:.1f}% of FP16)")
print(f"[probe] V12 reference: 1248 MB")

print()
print(f"{'Budget %':>10}  {'Budget MB':>10}  {'Feasible':>9}  {'Total LPIPS':>12}  {'FP16 / INT4 split':>20}")
print("-" * 80)

# Try: 50%, 45%, 40%, 35%, 33%, 32%, 31.5%, 31.4%, 31.3% etc
budgets_pct = [50, 45, 40, 35, 33, 32, 31.5, 31.4, 31.3]
results = []
for pct in budgets_pct:
    budget = fp16_total_b * (pct / 100.0)
    if budget < int4_min_b:
        # Definitely infeasible
        print(f"{pct:>9.1f}%  {budget/1e6:>10.0f}  {'NO (floor)':>9}  {'—':>12}  {'—':>20}")
        continue

    combo, tot_lpips, status = ilp_allocate(cand_rows, budget, time_limit=60)
    if combo is None:
        print(f"{pct:>9.1f}%  {budget/1e6:>10.0f}  {status[:9]:>9}  {'—':>12}  {'—':>20}")
        continue

    n_fp16 = sum(1 for c in combo if c["choice"] == "FP16")
    n_int4 = len(combo) - n_fp16
    by_step = {}
    for c in combo:
        by_step[c["choice"]] = by_step.get(c["choice"], 0) + 1
    actual = sum(c["bytes"] for c in combo) / 1e6
    split_str = f"FP16={n_fp16}, INT4={n_int4}"
    print(f"{pct:>9.1f}%  {actual:>10.0f}  {'OK':>9}  {tot_lpips:>12.3e}  {split_str:>20}")
    results.append({
        "pct": pct, "actual_MB": actual,
        "by_choice": {k: v for k, v in sorted(by_step.items())},
        "n_fp16": n_fp16,
        "lpips_sum": tot_lpips,
    })

print()
print("[probe] Detailed by-step distribution at each feasible budget:")
for r in results:
    print(f"  {r['pct']:>4.1f}%: {r['by_choice']}")