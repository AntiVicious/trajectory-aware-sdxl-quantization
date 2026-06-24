#!/usr/bin/env python3
"""
Generate atlas_heatmap_top100.png from atlas_full.csv

Auto-detects column names. Works with either the wide-format atlas_full.csv
or the cv_results.csv (both have the right columns).

Output: 100 rows (top-100 most-sensitive layers) x 5 columns (commit steps),
colored by log10(clip_dist).
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def find_default_atlas_csv():
    """Try several plausible locations so the script Just Works on most machines."""
    candidates = [
        os.path.expanduser("~/Documents/Maharshi/gem/trajectory/phase1_atlas_full/atlas_full.csv"),
        # Try the cv_results.csv too -- it has the same atlas data plus predictor preds
        os.path.expanduser("~/Documents/Maharshi/gem/trajectory/phase2_predictor_full/cv_results.csv"),
        "trajectory/phase1_atlas_full/atlas_full.csv",
        "trajectory/phase2_predictor_full/cv_results.csv",
        "../trajectory/phase1_atlas_full/atlas_full.csv",
        "phase1_atlas_full/atlas_full.csv",
        "atlas_full.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas-csv", default=find_default_atlas_csv())
    p.add_argument("--out", default="atlas_heatmap_top100.png")
    p.add_argument("--top-k", type=int, default=100)
    args = p.parse_args()

    if not os.path.exists(args.atlas_csv):
        print(f"ERROR: atlas CSV not found at {args.atlas_csv}", file=sys.stderr)
        print("Pass it with --atlas-csv /path/to/atlas_full.csv", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.atlas_csv)
    print(f"Loaded {len(df)} rows from {args.atlas_csv}")
    print(f"Columns: {list(df.columns)}")

    # --- Robust column auto-detect ---
    layer_col = next((c for c in ["layer", "layer_name", "name"] if c in df.columns), None)
    t_col = next(
        (c for c in ["t", "commit_step", "step", "step_from", "commit_t", "timestep"]
         if c in df.columns),
        None,
    )
    d_col = next(
        (c for c in ["clip_dist_mean", "clip_dist", "distance", "clip_distance", "dist"]
         if c in df.columns),
        None,
    )

    if layer_col is None or t_col is None or d_col is None:
        print(
            f"ERROR: Could not auto-detect columns. Got:\n"
            f"  layer:   {layer_col}\n"
            f"  step:    {t_col}\n"
            f"  dist:    {d_col}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Edit the auto-detect lists in main().",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"Using: layer={layer_col}, step={t_col}, dist={d_col}")

    # Average across prompts if a prompt column is present
    if "prompt" in df.columns or "prompt_idx" in df.columns:
        df = df.groupby([layer_col, t_col], as_index=False)[d_col].mean()
        print(f"Averaged across prompts -> {len(df)} (layer, step) rows.")

    # Pivot: rows=layer, cols=commit step
    pivot = df.pivot_table(index=layer_col, columns=t_col, values=d_col, aggfunc="mean")
    pivot = pivot.sort_index(axis=1)
    pivot["__mean__"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("__mean__", ascending=False).drop(columns="__mean__")
    pivot = pivot.head(args.top_k)
    print(f"Top-{args.top_k} layers, columns = {list(pivot.columns)}")

    eps = 1e-7
    matrix = np.log10(np.maximum(pivot.values, eps))

    fig, ax = plt.subplots(figsize=(5.5, 12))
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=np.nanpercentile(matrix, 2), vmax=np.nanpercentile(matrix, 98))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns], fontsize=10)
    ax.set_xlabel("Commit timestep $t$", fontsize=12)
    ax.set_ylabel(f"Layer (top-{args.top_k} by mean sensitivity)", fontsize=12)
    ax.set_yticks([])
    ax.set_title(
        f"Atlas heatmap: top-{args.top_k} most-sensitive layers\n"
        f"(rows sorted by mean CLIP distance, descending)",
        fontsize=11,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"$\log_{10}$ CLIP-feature distance", fontsize=10)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}  ({os.path.getsize(args.out)/1024:.1f} KB)")


if __name__ == "__main__":
    main()