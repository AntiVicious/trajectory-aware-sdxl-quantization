"""
Phase 3 allocator on the full atlas (794 layers).

Uses gbr_role_block predictor (Spearman ρ=0.98 on held-out layers).

For each layer, choices are:
  - FP16 (cost = n_params × 2 bytes, predicted_lpips = 0)
  - INT4 @ step_from for step_from in {0, 5, 10, 15, 19}
    Cost model: step-weighted — layer spends step_from steps FP16, rest INT4.
    eff_bytes = (step_from / 20) × fp16_bytes + (1 - step_from / 20) × int4_bytes
    (intuition: committing earlier saves more, but increases sensitivity cost)

ILP via PuLP CBC:
  minimize:  sum over layers of predicted_CLIP_dist
  subject to:  sum of bytes <= budget
             for each layer, exactly one choice

Budgets swept: 95%, 85%, 70%, 55%, 40% of FP16 total.

Outputs:
  phase3_allocator_full/candidates_full.csv
  phase3_allocator_full/allocation_results_full.csv
  phase3_allocator_full/allocation_70pct_full.json    (for V13 validation)

Usage:
    python allocate_full.py
"""
import csv
import json
import pickle
import os
from pathlib import Path
import numpy as np
import pulp

BASE_DIR = Path(os.environ.get("SDXL_DATA_ROOT", "/home/project/Documents/Maharshi/gem"))
LAYERS_JSON = BASE_DIR / "trajectory" / "phase1_atlas_full" / "layers_full.json"
P2_DIR = BASE_DIR / "trajectory" / "phase2_predictor_full"
PREDICTOR_PATH = P2_DIR / "models" / "gbr_role_block_final.pkl"
OUT_DIR = BASE_DIR / "trajectory" / "phase3_allocator_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FP16_BPP = 2.0      # bytes per parameter in FP16
INT4_BPP = 0.5 + 0.125   # 0.5 byte/param (4-bit) + scale overhead (1/8 byte/param)
TOTAL_STEPS = 20
STEP_OPTIONS = [0, 5, 10, 15, 19]


# ============ Feature builder must match Phase 2 ============

def classify_role(name):
    n = name.lower()
    if "attn1" in n and ("to_q" in n or "to_k" in n or "to_v" in n):
        return "self_attn_qkv"
    if "attn1" in n and "to_out" in n:
        return "self_attn_out"
    if "attn2" in n and ("to_q" in n or "to_k" in n or "to_v" in n):
        return "cross_attn_qkv"
    if "attn2" in n and "to_out" in n:
        return "cross_attn_out"
    if ".ff.net.0.proj" in n:
        return "ffn_in"
    if ".ff.net.2" in n:
        return "ffn_out"
    if ".proj_in" in n:
        return "transformer_proj_in"
    if ".proj_out" in n:
        return "transformer_proj_out"
    if "resnets." in n and "conv1" in n:
        return "resnet_conv1"
    if "resnets." in n and "conv2" in n:
        return "resnet_conv2"
    if "resnets." in n and "time_emb_proj" in n:
        return "resnet_time_proj"
    if "resnets." in n and "conv_shortcut" in n:
        return "resnet_shortcut"
    if "downsamplers." in n or "upsamplers." in n:
        return "sampler_conv"
    if name.startswith("conv_in"):
        return "stem_conv_in"
    if name.startswith("conv_out"):
        return "stem_conv_out"
    if "time_embedding" in n or "add_embedding" in n:
        return "time_embedding"
    return "other"


def classify_block_depth(name):
    import re
    m = re.match(r"down_blocks\.(\d+)", name)
    if m:
        return int(m.group(1))
    if name.startswith("mid_block"):
        return 3
    m = re.match(r"up_blocks\.(\d+)", name)
    if m:
        return 4 + int(m.group(1))
    return -1


def feature_vec(layer_meta, step_from, meta):
    """Build the 29-dim feature vector for (layer, step_from)."""
    layer_type = 1.0 if layer_meta["type"] == "Conv2d" else 0.0
    log_n_in = np.log10(layer_meta["n_in"] + 1)
    log_n_out = np.log10(layer_meta["n_out"] + 1)
    log_n_params = np.log10(layer_meta["n_params"])
    step_norm = step_from / TOTAL_STEPS

    dense = np.array([layer_type, log_n_in, log_n_out, log_n_params, step_norm],
                     dtype=np.float32)
    dense_z = (dense - meta["dense_mean"].flatten()) / meta["dense_std"].flatten()

    role = classify_role(layer_meta["name"])
    role_onehot = np.zeros(len(meta["role_set"]), dtype=np.float32)
    if role in meta["role_set"]:
        role_onehot[meta["role_set"].index(role)] = 1.0

    block = classify_block_depth(layer_meta["name"])
    block_onehot = np.zeros(len(meta["block_set"]), dtype=np.float32)
    if block in meta["block_set"]:
        block_onehot[meta["block_set"].index(block)] = 1.0

    return np.concatenate([dense_z, role_onehot, block_onehot]).reshape(1, -1).astype(np.float32)


# ============ Build candidate table ============

def build_candidates(layers_meta, predictor, pred_meta):
    """For each layer, generate FP16 + INT4@t choices with cost and predicted LPIPS."""
    rows = []
    for lm in layers_meta:
        n_params = lm["n_params"]
        fp16_b = n_params * FP16_BPP
        int4_b = n_params * INT4_BPP

        # FP16 choice: zero sensitivity, full memory
        rows.append({
            "layer": lm["name"], "choice": "FP16", "step_from": -1,
            "lpips_pred": 0.0, "bytes": fp16_b, "n_params": n_params,
        })
        # INT4 @ various step_from
        for s in STEP_OPTIONS:
            x = feature_vec(lm, s, pred_meta)
            log_dist = float(predictor.predict(x)[0])
            dist_pred = 10 ** log_dist  # back from log10

            frac_fp16 = s / TOTAL_STEPS
            eff_bytes = frac_fp16 * fp16_b + (1 - frac_fp16) * int4_b

            rows.append({
                "layer": lm["name"], "choice": f"INT4@t{s}", "step_from": s,
                "lpips_pred": dist_pred, "bytes": eff_bytes, "n_params": n_params,
            })
    return rows


# ============ ILP ============

def ilp_allocate(cand_rows, budget_bytes, time_limit=120):
    """Multiple-choice knapsack via CBC. Returns (combo, total_lpips)."""
    LPIPS_SCALE = 1e6  # scale up so CBC sees integer-friendly coefficients
    prob = pulp.LpProblem("traj_alloc_full", pulp.LpMinimize)

    # Group candidates by layer
    by_layer = {}
    for i, c in enumerate(cand_rows):
        by_layer.setdefault(c["layer"], []).append((i, c))

    x = {}
    for layer, choices in by_layer.items():
        for j, (idx, c) in enumerate(choices):
            x[(layer, j)] = pulp.LpVariable(f"x_{idx}", cat="Binary")

    # Objective
    prob += pulp.lpSum(
        x[(layer, j)] * (c["lpips_pred"] * LPIPS_SCALE)
        for layer, choices in by_layer.items()
        for j, (_, c) in enumerate(choices)
    )

    # Exactly one choice per layer
    for layer, choices in by_layer.items():
        prob += pulp.lpSum(x[(layer, j)] for j in range(len(choices))) == 1

    # Budget
    prob += pulp.lpSum(
        x[(layer, j)] * c["bytes"]
        for layer, choices in by_layer.items()
        for j, (_, c) in enumerate(choices)
    ) <= budget_bytes

    solver = pulp.PULP_CBC_CMD(msg=0, gapRel=1e-4, timeLimit=time_limit)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    if status not in ("Optimal", "Integer Feasible"):
        return None, float("inf"), status

    combo = []
    for layer, choices in by_layer.items():
        for j, (_, c) in enumerate(choices):
            if pulp.value(x[(layer, j)]) > 0.5:
                combo.append(c); break
    return combo, sum(c["lpips_pred"] for c in combo), status


# ============ Main ============

def main():
    with open(LAYERS_JSON) as f:
        layers_meta = json.load(f)["layers"]
    print(f"[phase3] {len(layers_meta)} layers")

    with open(PREDICTOR_PATH, "rb") as f:
        pack = pickle.load(f)
    predictor = pack["model"]
    pred_meta = pack["meta"]
    print(f"[phase3] Loaded predictor: {pack['variant']}  "
          f"({len(pack['feat_names'])} features)")

    # Build candidates
    print("[phase3] Building candidate table (FP16 + INT4@t for each layer)...")
    cand_rows = build_candidates(layers_meta, predictor, pred_meta)
    print(f"[phase3] {len(cand_rows)} candidates = "
          f"{len(layers_meta)} layers × {1 + len(STEP_OPTIONS)} choices")

    with open(OUT_DIR / "candidates_full.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "choice", "step_from",
                                           "lpips_pred", "bytes", "n_params"])
        w.writeheader()
        w.writerows(cand_rows)

    # Totals
    fp16_total_bytes = sum(c["bytes"] for c in cand_rows if c["choice"] == "FP16")
    int4_agg_bytes = sum(next(c for c in cand_rows
                              if c["layer"] == l["name"] and c["choice"] == "INT4@t0")["bytes"]
                         for l in layers_meta)
    print(f"\n[phase3] FP16 total:    {fp16_total_bytes/1e6:.1f} MB")
    print(f"[phase3] INT4@t0 (min): {int4_agg_bytes/1e6:.1f} MB")

    # Budget sweep
    budgets = [(ratio, fp16_total_bytes * ratio)
               for ratio in [0.95, 0.85, 0.70, 0.55, 0.40]]
    results = []
    allocations = {}
    for ratio, budget in budgets:
        pct = int(ratio * 100)
        print(f"\n=== Budget: {budget/1e6:.1f} MB ({pct}% of FP16) ===")
        combo, tot_lpips, status = ilp_allocate(cand_rows, budget)
        if combo is None:
            print(f"  ILP {status}")
            continue
        tot_b = sum(c["bytes"] for c in combo)
        n_fp16 = sum(1 for c in combo if c["choice"] == "FP16")
        n_int4 = len(combo) - n_fp16
        by_step = {}
        for c in combo:
            by_step[c["choice"]] = by_step.get(c["choice"], 0) + 1

        print(f"  Status: {status}")
        print(f"  Total: {tot_b/1e6:.1f} MB (budget {budget/1e6:.1f} MB)")
        print(f"  FP16 layers: {n_fp16}  INT4 layers: {n_int4}")
        print(f"  By choice: {dict(sorted(by_step.items()))}")
        print(f"  Predicted LPIPS sum: {tot_lpips:.3e}")

        results.append({
            "budget_pct": pct,
            "budget_MB": budget / 1e6,
            "actual_MB": tot_b / 1e6,
            "n_fp16": n_fp16,
            "n_int4": n_int4,
            "lpips_sum": tot_lpips,
            "status": status,
        })
        allocations[pct] = {
            "budget_pct": pct,
            "budget_MB": budget / 1e6,
            "actual_MB": tot_b / 1e6,
            "total_lpips_pred": tot_lpips,
            "predictor": pack["variant"],
            "allocations": [
                {"layer": c["layer"], "choice": c["choice"],
                 "step_from": c["step_from"],
                 "lpips_pred": c["lpips_pred"], "bytes": c["bytes"],
                 "n_params": c["n_params"]}
                for c in combo
            ],
        }

    # Save budget-sweep results CSV
    with open(OUT_DIR / "allocation_results_full.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n[phase3] Results -> {OUT_DIR / 'allocation_results_full.csv'}")

    # Save full allocation for 70% budget (primary target)
    for pct, alloc in allocations.items():
        path = OUT_DIR / f"allocation_{pct}pct_full.json"
        with open(path, "w") as f:
            json.dump(alloc, f, indent=2)
        print(f"[phase3] Saved {pct}% allocation -> {path.name}")

    print(f"\n[phase3] Primary target for validation: allocation_70pct_full.json")


if __name__ == "__main__":
    main()