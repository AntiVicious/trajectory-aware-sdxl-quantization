"""
Build all thesis artifacts from existing data files.

Inputs (expected in --data-dir, default $SDXL_DATA_ROOT):
  layers_full.json
  allocation_70pct_full.json, allocation_55pct_full.json, allocation_40pct_full.json
  dreamsim_results.json
  extra_metrics_results.json
  scores_5k_with_ablations.json
  pickscore_results_clean.json
  vram_measurement_*.json  (optional - VRAM measurements from runtime experiments)

Outputs (in --out-dir):
  tables/
    layers_by_role.csv
    layers_by_block.csv
    allocation_summary.csv
    memory_breakdown_by_block_70.csv (and 55, 40)
    memory_breakdown_by_role_70.csv (and 55, 40)
    principal_results.csv          (master table for Ch5)
    pareto_data.csv                (variant, vram, dreamsim, fid, clip)
    runtime_summary.csv            (V19-V26 runtime architecture summary)
    negative_results.csv
  figures/
    atlas_heatmap_top100.png        (existing - copy if present)
    allocation_heatmap.png
    role_quantization_breakdown.png
    pareto_dreamsim_vs_vram.png
    pareto_all_metrics.png
    runtime_comparison.png
  summary.json                     (everything in one place for thesis-writer)

Usage:
  python build_all_thesis_artifacts.py \
    --data-dir $SDXL_DATA_ROOT \
    --out-dir ./thesis_outputs
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys


def load_json(p):
    with open(p) as f:
        return json.load(f)


def safe_get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# Hardcoded peak VRAM measurements (we measured these earlier; they're not in any JSON)
# If you have vram_measurement_*.json files, those will override these.
DEFAULT_VRAM = {
    "v1_fp16":                       {"peak_gb": 9.50, "time_per_image_s": 2.11, "notes": "FP16 baseline"},
    "v13_70pct_trajectory":          {"peak_gb": 14.62, "time_per_image_s": 2.00, "notes": "Reference impl, holds FP16+INT4-grid"},
    "v13_55pct_trajectory":          {"peak_gb": 14.62, "time_per_image_s": 2.00},
    "v13_40pct_trajectory":          {"peak_gb": 14.62, "time_per_image_s": 2.00},
    "v19_70pct_trajectory_v12quant": {"peak_gb": 14.62, "time_per_image_s": 2.00, "notes": "V13 + outlier preservation"},
    "v19_55pct_trajectory_v12quant": {"peak_gb": 14.62, "time_per_image_s": 2.00},
    "v19_40pct_trajectory_v12quant": {"peak_gb": 14.62, "time_per_image_s": 2.00},
    "v20_70pct_nf4_static":          {"peak_gb": 6.16, "time_per_image_s": 1.83, "notes": "bnb NF4 static, no trajectory"},
    "v20_55pct_nf4_static":          {"peak_gb": 6.16, "time_per_image_s": 1.83},
    "v20_40pct_nf4_static":          {"peak_gb": 6.16, "time_per_image_s": 1.83},
    "v21_70pct_two_pass_t10_v2":     {"peak_gb": 7.30, "time_per_image_s": 3.47, "notes": "Two-pass FP16+NF4, t_split=10"},
    "v21_55pct_two_pass_t10_v2":     {"peak_gb": 7.30, "time_per_image_s": 3.47},
    "v21_40pct_two_pass_t10_v2":     {"peak_gb": 7.30, "time_per_image_s": 3.47},
    "v22_70pct_floor5":              {"peak_gb": 10.69, "time_per_image_s": 1.50, "notes": "Per-layer hot-swap, floor=5"},
    "v22_55pct_floor5":              {"peak_gb": 10.69, "time_per_image_s": 1.50},
    "v22_40pct_floor5":              {"peak_gb": 10.69, "time_per_image_s": 1.50},
    "v22_70pct_hotswap":             {"peak_gb": 10.69, "time_per_image_s": 1.50, "notes": "V13 schedule, V22 mechanism"},
    "v22_55pct_hotswap":             {"peak_gb": 10.69, "time_per_image_s": 1.50},
    "v22_40pct_hotswap":             {"peak_gb": 10.69, "time_per_image_s": 1.50, "notes": "Aggressive t=0 commits damage trajectory"},
    "v24b_int8_full":                {"peak_gb": 7.28, "time_per_image_s": 2.50, "notes": "bnb INT8 PTQ, full UNet"},
}


def gather_vram_files(data_dir):
    """Look for vram_measurement_*.json files and merge with defaults."""
    out = dict(DEFAULT_VRAM)
    for f in Path(data_dir).glob("vram_measurement_*.json"):
        try:
            d = load_json(f)
            v = d.get("variant", f.stem.replace("vram_measurement_", ""))
            out[v] = {
                "peak_gb": d.get("max_peak_vram_gb", d.get("inference_peak_gb")),
                "time_per_image_s": d.get("avg_time_per_image_s", d.get("time_per_image_s")),
                "notes": "measured",
            }
        except Exception as e:
            print(f"  warn: couldn't parse {f}: {e}", file=sys.stderr)
    return out


# ============================================================================
# Layer / role / block tables
# ============================================================================

def write_layers_tables(layers, out_dir):
    """Write layers_by_role.csv and layers_by_block.csv."""
    role_counts = layers.get("role_counts", {})
    block_counts = layers.get("block_counts", {})

    # Role table
    rows = sorted(role_counts.items(), key=lambda kv: -kv[1])
    with open(out_dir / "layers_by_role.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["role", "count"])
        for r, c in rows:
            w.writerow([r, c])
        w.writerow(["TOTAL", sum(role_counts.values())])

    # Block table
    rows = sorted(block_counts.items(), key=lambda kv: -kv[1])
    with open(out_dir / "layers_by_block.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "count"])
        for b, c in rows:
            w.writerow([b, c])
        w.writerow(["TOTAL", sum(block_counts.values())])

    print(f"  Wrote layers_by_role.csv ({len(rows)} rows)")
    print(f"  Wrote layers_by_block.csv")


# ============================================================================
# Allocation tables
# ============================================================================

def write_allocation_summary(allocs, out_dir):
    """Write allocation_summary.csv: budget, actual_MB, FP16/t0/t5/t10/t15/t19 counts."""
    with open(out_dir / "allocation_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["budget_pct", "actual_MB", "FP16", "INT4@t0", "INT4@t5",
                    "INT4@t10", "INT4@t15", "INT4@t19", "total_lpips_pred"])
        for budget, alloc in allocs.items():
            counts = {"FP16": 0, "INT4@t0": 0, "INT4@t5": 0, "INT4@t10": 0,
                      "INT4@t15": 0, "INT4@t19": 0}
            for a in alloc["allocations"]:
                counts[a["choice"]] = counts.get(a["choice"], 0) + 1
            w.writerow([budget, round(alloc["actual_MB"], 1),
                        counts["FP16"], counts["INT4@t0"], counts["INT4@t5"],
                        counts["INT4@t10"], counts["INT4@t15"], counts["INT4@t19"],
                        round(alloc["total_lpips_pred"], 4)])
    print(f"  Wrote allocation_summary.csv ({len(allocs)} budgets)")


def write_memory_breakdowns(allocs, layers_data, out_dir):
    """Write memory_breakdown_by_block_BUDGET.csv and _by_role_BUDGET.csv for each budget."""
    layer_meta = {L["name"]: L for L in layers_data["layers"]}

    for budget, alloc in allocs.items():
        # Aggregate by block and role
        by_block = {}  # block -> {n, fp16_count, int4_count, fp16_mb, int4_mb}
        by_role = {}

        for a in alloc["allocations"]:
            name = a["layer"]
            meta = layer_meta.get(name, {})
            block = meta.get("block", "unknown")
            role = meta.get("role", "unknown")
            n_params = a.get("n_params", meta.get("n_params", 0))
            choice = a["choice"]
            is_fp16 = (choice == "FP16")
            # FP16 = 2 bytes/param; INT4 effective from `bytes` field (step-weighted)
            bytes_used = a.get("bytes", n_params * 2)

            for d, k in [(by_block, block), (by_role, role)]:
                if k not in d:
                    d[k] = {"n": 0, "fp16_count": 0, "int4_count": 0,
                            "fp16_mb": 0.0, "int4_mb": 0.0}
                d[k]["n"] += 1
                if is_fp16:
                    d[k]["fp16_count"] += 1
                    d[k]["fp16_mb"] += bytes_used / 1e6
                else:
                    d[k]["int4_count"] += 1
                    d[k]["int4_mb"] += bytes_used / 1e6

        # Write per-block CSV
        with open(out_dir / f"memory_breakdown_by_block_{budget}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["block", "layers", "FP16_count", "INT4_count",
                        "FP16_MB", "INT4_MB", "Total_MB"])
            for k in sorted(by_block.keys()):
                v = by_block[k]
                total = v["fp16_mb"] + v["int4_mb"]
                w.writerow([k, v["n"], v["fp16_count"], v["int4_count"],
                            round(v["fp16_mb"], 1), round(v["int4_mb"], 1),
                            round(total, 1)])

        # Write per-role CSV
        with open(out_dir / f"memory_breakdown_by_role_{budget}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["role", "layers", "FP16_count", "INT4_count",
                        "FP16_MB", "INT4_MB", "Total_MB"])
            for k in sorted(by_role.keys(), key=lambda r: -by_role[r]["n"]):
                v = by_role[k]
                total = v["fp16_mb"] + v["int4_mb"]
                w.writerow([k, v["n"], v["fp16_count"], v["int4_count"],
                            round(v["fp16_mb"], 1), round(v["int4_mb"], 1),
                            round(total, 1)])

        print(f"  Wrote memory_breakdown_by_block_{budget}.csv, _by_role_{budget}.csv")


# ============================================================================
# Principal results table
# ============================================================================

# Ordering and labels for the master table
VARIANT_ORDER = [
    ("v1_fp16",                          "V1: FP16 baseline"),
    ("v2_int4_naive",                    "V2: Naive INT4"),
    ("v4_hybrid_best",                   "V4: Hybrid ternary + INT4"),
    ("v6_fully_distilled",               "V6: Fully distilled hybrid"),
    ("v7_timestep_lora",                 "V7: Timestep-aware LoRA"),
    ("v8_smoothquant_hybrid",            "V8: SmoothQuant + ternary"),
    ("v12_outlier_preserved",            "V12: Outlier-preserved + ternary"),
    ("v13_70pct_trajectory",             "V13-70: Trajectory + naive INT4"),
    ("v13_55pct_trajectory",             "V13-55"),
    ("v13_40pct_trajectory",             "V13-40"),
    ("v14_70pct_smoothquant",            "V14-70: V13 + SmoothQuant filter"),
    ("v14_55pct_smoothquant",            "V14-55"),
    ("v14_40pct_smoothquant",            "V14-40"),
    ("v15_70pct_gptq",                   "V15-70: V13 + GPTQ filter"),
    ("v15_55pct_gptq",                   "V15-55"),
    ("v15_40pct_gptq",                   "V15-40"),
    ("v16_70pct_random_alloc",           "V16-70: Random allocation control"),
    ("v16_55pct_random_alloc",           "V16-55"),
    ("v16_40pct_random_alloc",           "V16-40"),
    ("v17_70pct_no_trajectory",          "V17-70: No trajectory (all t=0)"),
    ("v17_55pct_no_trajectory",          "V17-55"),
    ("v17_40pct_no_trajectory",          "V17-40"),
    ("v18_70pct_v12_quantizer",          "V18-70: V12 quantizer no trajectory"),
    ("v18_55pct_v12_quantizer",          "V18-55"),
    ("v18_40pct_v12_quantizer",          "V18-40"),
    ("v19_70pct_trajectory_v12quant",    "V19-70: Trajectory + outlier INT4"),
    ("v19_55pct_trajectory_v12quant",    "V19-55"),
    ("v19_40pct_trajectory_v12quant",    "V19-40"),
    ("v20_70pct_nf4_static",             "V20-70: NF4 static"),
    ("v20_55pct_nf4_static",             "V20-55"),
    ("v20_40pct_nf4_static",             "V20-40"),
    ("v21_70pct_two_pass_t10_v2",        "V21-70: Two-pass FP16+NF4"),
    ("v21_55pct_two_pass_t10_v2",        "V21-55"),
    ("v21_40pct_two_pass_t10_v2",        "V21-40"),
    ("v22_70pct_floor5",                 "V22-70 floor5: Hot-swap NF4"),
    ("v22_55pct_floor5",                 "V22-55 floor5"),
    ("v22_40pct_floor5",                 "V22-40 floor5"),
    ("v22_40pct_hotswap",                "V22-40 raw: Without floor"),
    ("v24b_int8_full",                   "V24b: bnb INT8 baseline"),
]


def write_principal_results(scores, dreamsim, extra, vram, out_dir):
    """Write the master CSV combining all metrics."""
    with open(out_dir / "principal_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant_id", "label", "CLIP", "FID", "DreamSim",
                    "Aesthetic", "CLIP-IQA", "Memory_MB", "Memory_reduction_pct",
                    "Peak_VRAM_GB", "Time_per_image_s", "Notes"])
        for vid, label in VARIANT_ORDER:
            clip = safe_get(scores, "clip", vid)
            fid_v = safe_get(scores, "fid", vid)
            dsim = safe_get(dreamsim, "results", vid, "mean")
            aesth = safe_get(extra, "aesthetic", vid, "mean")
            iqa = safe_get(extra, "clip_iqa", vid, "mean")
            mem = safe_get(scores, "memory", vid, "size_mb")
            mem_red = safe_get(scores, "memory", vid, "reduction")
            v = vram.get(vid, {})
            peak = v.get("peak_gb")
            tpimg = v.get("time_per_image_s")
            notes = v.get("notes", "")
            w.writerow([
                vid, label,
                f"{clip:.4f}" if clip is not None else "",
                f"{fid_v:.2f}" if fid_v is not None else "",
                f"{dsim:.4f}" if dsim is not None else "",
                f"{aesth:.3f}" if aesth is not None else "",
                f"{iqa:.4f}" if iqa is not None else "",
                mem if mem else "",
                f"{mem_red:.1f}" if mem_red and mem_red > 0 else "",
                f"{peak:.2f}" if peak else "",
                f"{tpimg:.2f}" if tpimg else "",
                notes,
            ])
    print(f"  Wrote principal_results.csv ({len(VARIANT_ORDER)} variants)")


def write_pareto_data(scores, dreamsim, vram, out_dir):
    """Subset of variants for Pareto plots: highlight V20, V21, V22-floor5, V24b, V13."""
    pareto_set = [
        ("v1_fp16",                       "V1 (FP16)",          "baseline"),
        ("v20_70pct_nf4_static",          "V20-70 (NF4)",       "static_quant"),
        ("v20_55pct_nf4_static",          "V20-55",             "static_quant"),
        ("v20_40pct_nf4_static",          "V20-40",             "static_quant"),
        ("v24b_int8_full",                "V24b (INT8 baseline)", "static_quant"),
        ("v21_70pct_two_pass_t10_v2",     "V21-70 (FP16+NF4)",  "trajectory_runtime"),
        ("v21_55pct_two_pass_t10_v2",     "V21-55",             "trajectory_runtime"),
        ("v21_40pct_two_pass_t10_v2",     "V21-40",             "trajectory_runtime"),
        ("v22_70pct_floor5",              "V22-70 floor5",      "trajectory_runtime"),
        ("v22_55pct_floor5",              "V22-55 floor5",      "trajectory_runtime"),
        ("v22_40pct_floor5",              "V22-40 floor5",      "trajectory_runtime"),
        ("v13_70pct_trajectory",          "V13-70 (reference)", "trajectory_reference"),
        ("v13_55pct_trajectory",          "V13-55",             "trajectory_reference"),
        ("v13_40pct_trajectory",          "V13-40",             "trajectory_reference"),
        ("v17_70pct_no_trajectory",       "V17-70 (no traj.)",  "ablation"),
        ("v17_55pct_no_trajectory",       "V17-55",             "ablation"),
        ("v17_40pct_no_trajectory",       "V17-40",             "ablation"),
    ]
    with open(out_dir / "pareto_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant_id", "label", "category", "Peak_VRAM_GB",
                    "Memory_MB", "DreamSim", "FID", "CLIP"])
        for vid, label, cat in pareto_set:
            v = vram.get(vid, {})
            peak = v.get("peak_gb")
            mem = safe_get(scores, "memory", vid, "size_mb")
            dsim = safe_get(dreamsim, "results", vid, "mean")
            fid_v = safe_get(scores, "fid", vid)
            clip = safe_get(scores, "clip", vid)
            w.writerow([vid, label, cat,
                        f"{peak:.2f}" if peak else "",
                        mem if mem else "",
                        f"{dsim:.4f}" if dsim is not None else "",
                        f"{fid_v:.2f}" if fid_v is not None else "",
                        f"{clip:.4f}" if clip is not None else ""])
    print(f"  Wrote pareto_data.csv")


# ============================================================================
# Runtime architecture summary
# ============================================================================

def write_runtime_summary(out_dir):
    """Hand-curated summary of V19-V26 runtime architectures."""
    rows = [
        # (variant, mechanism, weight_storage, peak_vram, dreamsim_70, status)
        ("V13", "Reference (FP16+INT4-grid in same tensor)", "FP16", "14.62 GB", "0.026", "documented"),
        ("V19", "Reference + outlier preservation (top 0.1% kept FP16)", "FP16", "14.62 GB", "0.017", "documented"),
        ("V20", "Static NF4 (no trajectory)", "NF4 (4-bit packed)", "6.16 GB", "0.125", "deployed"),
        ("V21", "Two-pass: FP16 base, NF4 refiner (denoising_end/start)", "FP16 + NF4 swap", "7.30 GB", "0.020", "deployed"),
        ("V22", "Per-layer hot-swap, V13 schedule + floor=5", "FP16 + NF4 simultaneously held", "10.69 GB", "0.0165", "deployed"),
        ("V24b", "Static INT8 PTQ (bnb Linear8bitLt)", "INT8 (8-bit packed)", "7.28 GB", "0.065", "deployed (baseline)"),
        ("V25", "INT8 base + NF4 refiner two-pass (CPU swap attempted)", "INT8 + NF4 (transient peak issue)", "8.57 GB", "not measured", "abandoned"),
        ("V26", "In-place INT8\u2192NF4 module swap on single UNet", "INT8 then NF4 (in-place)", "7.23 GB (multi)", "0.064", "demonstrated, multi-image limited"),
    ]
    with open(out_dir / "runtime_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "mechanism", "weight_storage",
                    "peak_VRAM", "DreamSim_70pct", "status"])
        for r in rows:
            w.writerow(r)
    print(f"  Wrote runtime_summary.csv")


# ============================================================================
# Negative results
# ============================================================================

def write_negative_results(out_dir):
    rows = [
        ("V15", "GPTQ-filtered weight substitution",
         "Per-layer recovery is bottleneck",
         "Null effect across all 3 budgets (DreamSim deltas < 0.003)",
         "Bottleneck is allocation, not per-layer quantization quality"),
        ("LoRA recovery (rank 8/16/32/64)", "SVD-based low-rank approximation of (FP16-NF4) error",
         "Quantization error is high-rank",
         "4-15% recoverability at rank 64; far below threshold for useful correction",
         "LoRA-based recovery is infeasible for this magnitude of quantization"),
        ("V25", "INT8+NF4 two-pass with CPU swap",
         "Reduce peak VRAM below V21",
         "Peak rose to 8.57 GB due to transient co-residency during GPU\u2194CPU transfer",
         "bitsandbytes Params4bit/Int8Params do not survive multi-step GPU\u2194CPU moves cleanly"),
        ("V26 multi-image (initial)", "In-place INT8\u2192NF4 swap with module replacement",
         "Single UNet, lower peak VRAM",
         "weight.shape[1]==1 assertion in bnb fix_4bit_weight_quant_state_from_module after multi-cycle swaps",
         "Single-image works (peak 6.11 GB); multi-image works for 50+ images then fails. Production use limited."),
    ]
    with open(out_dir / "negative_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "approach", "hypothesis_tested",
                    "result", "implication"])
        for r in rows:
            w.writerow(r)
    print(f"  Wrote negative_results.csv")


# ============================================================================
# Figures (matplotlib)
# ============================================================================

def make_figures(allocs, layers, scores, dreamsim, vram, fig_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available; skipping figures")
        return

    plt.rcParams.update({"font.size": 10, "figure.dpi": 120})

    # ---------- Figure 1: allocation heatmap ----------
    layer_meta = {L["name"]: L for L in layers["layers"]}
    layer_order = sorted(layers["layers"], key=lambda L: (
        list(layers.get("block_counts", {}).keys()).index(L["block"])
        if L["block"] in layers.get("block_counts", {}) else 99,
        L["name"]))
    layer_names = [L["name"] for L in layer_order]

    choice_to_int = {"FP16": 0, "INT4@t19": 1, "INT4@t15": 2,
                     "INT4@t10": 3, "INT4@t5": 4, "INT4@t0": 5}

    matrix = np.zeros((len(layer_names), len(allocs)))
    budget_labels = sorted(allocs.keys(), reverse=True)
    for j, b in enumerate(budget_labels):
        choice_map = {a["layer"]: a["choice"] for a in allocs[b]["allocations"]}
        for i, name in enumerate(layer_names):
            matrix[i, j] = choice_to_int.get(choice_map.get(name, "FP16"), 0)

    fig, ax = plt.subplots(figsize=(6, 12))
    cmap = plt.get_cmap("RdYlGn_r")
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=5,
                   interpolation="nearest")
    ax.set_xticks(range(len(budget_labels)))
    ax.set_xticklabels([f"{b}%" for b in budget_labels])
    ax.set_yticks([])
    ax.set_xlabel("Memory budget")
    ax.set_ylabel(f"Layers (n={len(layer_names)}, ordered by block)")
    ax.set_title("Per-layer allocation across budgets")
    cbar = fig.colorbar(im, ax=ax, ticks=range(6))
    cbar.ax.set_yticklabels(["FP16", "t=19", "t=15", "t=10", "t=5", "t=0"])
    plt.tight_layout()
    plt.savefig(fig_dir / "allocation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Wrote allocation_heatmap.png")

    # ---------- Figure 2: role x budget stacked bars ----------
    role_list = list(layers.get("role_counts", {}).keys())
    fig, axes = plt.subplots(1, len(allocs), figsize=(4 * len(allocs), 6),
                              sharey=True)
    if len(allocs) == 1:
        axes = [axes]
    for ax, b in zip(axes, budget_labels):
        choice_map = {a["layer"]: a["choice"] for a in allocs[b]["allocations"]}
        # for each role: count fp16, int4_late (t>=10), int4_early (t<10)
        fp16_count = []
        int4_late_count = []
        int4_early_count = []
        for r in role_list:
            f, latent, e = 0, 0, 0
            for L in layers["layers"]:
                if L["role"] != r:
                    continue
                ch = choice_map.get(L["name"], "FP16")
                if ch == "FP16":
                    f += 1
                else:
                    t = int(ch.split("t")[1])
                    if t >= 10:
                        latent += 1
                    else:
                        e += 1
            fp16_count.append(f)
            int4_late_count.append(latent)
            int4_early_count.append(e)
        x = np.arange(len(role_list))
        ax.bar(x, fp16_count, label="FP16", color="forestgreen")
        ax.bar(x, int4_late_count, bottom=fp16_count, label="INT4@t≥10",
               color="goldenrod")
        ax.bar(x, int4_early_count,
               bottom=[a + b for a, b in zip(fp16_count, int4_late_count)],
               label="INT4@t<10", color="firebrick")
        ax.set_xticks(x)
        ax.set_xticklabels(role_list, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"{b}% budget")
        ax.set_ylabel("Layer count" if ax is axes[0] else "")
    axes[-1].legend(loc="upper right", fontsize=9)
    plt.suptitle("Layer-role allocation across budgets")
    plt.tight_layout()
    plt.savefig(fig_dir / "role_quantization_breakdown.png",
                dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Wrote role_quantization_breakdown.png")

    # ---------- Figure 3: Pareto DreamSim vs Peak VRAM ----------
    fig, ax = plt.subplots(figsize=(8, 6))
    pareto_set = [
        ("v1_fp16", "V1 FP16", "baseline", "k", "*", 220),
        ("v20_70pct_nf4_static", "V20-70", "static", "tab:orange", "s", 80),
        ("v20_55pct_nf4_static", "V20-55", "static", "tab:orange", "s", 80),
        ("v20_40pct_nf4_static", "V20-40", "static", "tab:orange", "s", 80),
        ("v24b_int8_full", "V24b INT8", "static", "tab:red", "D", 100),
        ("v21_70pct_two_pass_t10_v2", "V21-70", "trajectory", "tab:blue", "o", 110),
        ("v21_55pct_two_pass_t10_v2", "V21-55", "trajectory", "tab:blue", "o", 110),
        ("v21_40pct_two_pass_t10_v2", "V21-40", "trajectory", "tab:blue", "o", 110),
        ("v22_70pct_floor5", "V22-70 f5", "trajectory", "tab:green", "^", 110),
        ("v22_55pct_floor5", "V22-55 f5", "trajectory", "tab:green", "^", 110),
        ("v22_40pct_floor5", "V22-40 f5", "trajectory", "tab:green", "^", 110),
        ("v17_70pct_no_trajectory", "V17-70 (no traj)", "ablation", "tab:gray", "x", 50),
        ("v17_55pct_no_trajectory", "V17-55 (no traj)", "ablation", "tab:gray", "x", 50),
        ("v17_40pct_no_trajectory", "V17-40 (no traj)", "ablation", "tab:gray", "x", 50),
    ]
    seen_categories = set()
    for vid, label, cat, color, marker, size in pareto_set:
        peak = vram.get(vid, {}).get("peak_gb")
        dsim = safe_get(dreamsim, "results", vid, "mean")
        if peak is None or dsim is None:
            continue
        legend_label = cat if cat not in seen_categories else None
        seen_categories.add(cat)
        ax.scatter(peak, dsim, c=color, marker=marker, s=size,
                   label=legend_label, edgecolors="black", linewidth=0.5)
        ax.annotate(label, (peak, dsim),
                    xytext=(4, 3), textcoords="offset points",
                    fontsize=7, alpha=0.9)

    ax.axhline(0.05, color="green", linestyle=":", linewidth=1, alpha=0.5,
               label="DreamSim 0.05 (perceptual threshold)")
    ax.set_xlabel("Peak VRAM during inference (GB)")
    ax.set_ylabel("DreamSim distance to FP16 (lower is better)")
    ax.set_title("Quality vs. peak VRAM Pareto front")
    ax.set_yscale("log")
    ax.set_ylim(0.005, 0.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "pareto_dreamsim_vs_vram.png",
                dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Wrote pareto_dreamsim_vs_vram.png")

    # ---------- Figure 4: 2x2 multi-metric Pareto ----------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metric_panels = [
        ("DreamSim",  "results",                          dreamsim, "mean", True),
        ("FID",       None,                               scores,   "fid",  False),
        ("CLIP",      None,                               scores,   "clip", False),
        ("CLIP-IQA",  "clip_iqa",                         None,     "mean", False),
    ]
    extra = load_json(Path(fig_dir).parent.parent / "_extra.json") if (Path(fig_dir).parent.parent / "_extra.json").exists() else None

    # Use what we have in scope
    extra_metrics = scores  # placeholder; we'll handle each panel by metric

    for ax, (title, _, _, _, log) in zip(axes.flatten(), metric_panels):
        ax.set_title(title)
        ax.set_xlabel("Peak VRAM (GB)")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

    # Plot DreamSim panel
    ax = axes[0, 0]
    for vid, label, cat, color, marker, size in pareto_set:
        peak = vram.get(vid, {}).get("peak_gb")
        dsim = safe_get(dreamsim, "results", vid, "mean")
        if peak is None or dsim is None:
            continue
        ax.scatter(peak, dsim, c=color, marker=marker, s=size,
                   edgecolors="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_title("DreamSim distance to FP16")
    ax.set_xlabel("Peak VRAM (GB)")
    ax.set_ylabel("DreamSim")
    ax.grid(True, alpha=0.3)

    # Plot FID
    ax = axes[0, 1]
    for vid, label, cat, color, marker, size in pareto_set:
        peak = vram.get(vid, {}).get("peak_gb")
        fid_v = safe_get(scores, "fid", vid)
        if peak is None or fid_v is None:
            continue
        ax.scatter(peak, fid_v, c=color, marker=marker, s=size,
                   edgecolors="black", linewidth=0.5)
    ax.set_title("FID vs COCO val2017")
    ax.set_xlabel("Peak VRAM (GB)")
    ax.set_ylabel("FID")
    ax.grid(True, alpha=0.3)

    # Plot CLIP
    ax = axes[1, 0]
    for vid, label, cat, color, marker, size in pareto_set:
        peak = vram.get(vid, {}).get("peak_gb")
        clip = safe_get(scores, "clip", vid)
        if peak is None or clip is None:
            continue
        ax.scatter(peak, clip, c=color, marker=marker, s=size,
                   edgecolors="black", linewidth=0.5)
    ax.set_title("CLIP score (prompt-image alignment)")
    ax.set_xlabel("Peak VRAM (GB)")
    ax.set_ylabel("CLIP")
    ax.grid(True, alpha=0.3)

    # Plot CLIP-IQA — needs extra_metrics_results.json which we'll load if available
    ax = axes[1, 1]
    extra_path = Path(fig_dir).parent.parent / "thesis_data" / "extra_metrics_results.json"
    if not extra_path.exists():
        # try the data_dir input
        extra_path_default = Path(args.data_dir) / "extra_metrics_results.json" if 'args' in dir() else None
    extra_metrics_data = None
    for p in [extra_path]:
        if p and p.exists():
            extra_metrics_data = load_json(p)
            break
    if extra_metrics_data:
        for vid, label, cat, color, marker, size in pareto_set:
            peak = vram.get(vid, {}).get("peak_gb")
            iqa = safe_get(extra_metrics_data, "clip_iqa", vid, "mean")
            if peak is None or iqa is None:
                continue
            ax.scatter(peak, iqa, c=color, marker=marker, s=size,
                       edgecolors="black", linewidth=0.5)
    ax.set_title("CLIP-IQA (no-reference quality)")
    ax.set_xlabel("Peak VRAM (GB)")
    ax.set_ylabel("CLIP-IQA")
    ax.grid(True, alpha=0.3)

    plt.suptitle("Multi-metric Pareto front", fontsize=14)
    plt.tight_layout()
    plt.savefig(fig_dir / "pareto_all_metrics.png",
                dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Wrote pareto_all_metrics.png")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with the input JSON files")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for tables, figures, summary")
    global args
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {data_dir}")
    print(f"Output: {out_dir}")

    # Load inputs
    print("\nLoading inputs...")
    layers = load_json(data_dir / "layers_full.json")
    print(f"  layers_full: {layers['n_total']} layers")

    allocs = {}
    for budget in [70, 55, 40]:
        path = data_dir / f"allocation_{budget}pct_full.json"
        if path.exists():
            allocs[budget] = load_json(path)
            print(f"  allocation_{budget}pct: {len(allocs[budget]['allocations'])} entries")

    scores = load_json(data_dir / "scores_5k_with_ablations.json")
    dreamsim = load_json(data_dir / "dreamsim_results.json")
    extra = load_json(data_dir / "extra_metrics_results.json")
    print(f"  scores_5k_with_ablations: {len(scores.get('clip', {}))} variants")

    vram = gather_vram_files(data_dir)
    print(f"  VRAM measurements: {len(vram)} variants")

    # Generate tables
    print("\nGenerating tables...")
    write_layers_tables(layers, tables_dir)
    write_allocation_summary(allocs, tables_dir)
    write_memory_breakdowns(allocs, layers, tables_dir)
    write_principal_results(scores, dreamsim, extra, vram, tables_dir)
    write_pareto_data(scores, dreamsim, vram, tables_dir)
    write_runtime_summary(tables_dir)
    write_negative_results(tables_dir)

    # Generate figures
    print("\nGenerating figures...")
    make_figures(allocs, layers, scores, dreamsim, vram, figures_dir)

    # Summary JSON
    summary = {
        "n_layers": layers["n_total"],
        "n_linear": layers["n_linear"],
        "n_conv": layers["n_conv"],
        "total_params_millions": layers["total_params_millions"],
        "budgets": list(allocs.keys()),
        "n_metrics_in_scores": len(scores.get("clip", {})),
        "n_dreamsim_variants": len(dreamsim.get("results", {})),
        "n_vram_measurements": len(vram),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote summary.json")
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
