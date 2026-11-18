#!/usr/bin/env python3
"""
Generate predictor_scatter.png from cv_results.csv

Reads the wide-format cv_results.csv with columns:
  layer, step_from, layer_type, n_params, clip_dist_mean, clip_dist_std,
  img_mse_mean, n_prompts, seconds, y_true_log10,
  gbr_pred, gbr_role_pred, mlp_pred, mlp_role_pred

Output: 4-panel scatter showing predicted vs. true log10(CLIP distance) for
each predictor variant, with MAE / R^2 / Spearman in the title.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


# Map (column name in cv_results.csv) -> (display title for panel)
PREDICTOR_COLS = [
    ("gbr_pred",      "GBR (dense, 5 features)"),
    ("gbr_role_pred", "GBR (dense + role + block, 29 ft.)"),
    ("mlp_pred",      "MLP (dense)"),
    ("mlp_role_pred", "MLP (dense + role + block)"),
]


def find_default_cv_csv():
    candidates = [
        os.path.join(os.environ.get("SDXL_DATA_ROOT", "."), "trajectory/phase2_predictor_full/cv_results.csv"),
        "trajectory/phase2_predictor_full/cv_results.csv",
        "../trajectory/phase2_predictor_full/cv_results.csv",
        "phase2_predictor_full/cv_results.csv",
        "cv_results.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def plot_panel(ax, true, pred, title):
    ax.scatter(true, pred, alpha=0.3, s=10, edgecolors="none")
    lo = float(np.nanmin([true.min(), pred.min()]))
    hi = float(np.nanmax([true.max(), pred.max()]))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    rho, _ = spearmanr(true, pred)
    ax.set_title(f"{title}\nMAE={mae:.3f}, $R^2$={r2:.3f}, $\\rho$={rho:.3f}", fontsize=10)
    ax.set_xlabel(r"true $\log_{10}$ CLIP distance")
    ax.set_ylabel(r"predicted $\log_{10}$ CLIP distance")
    ax.grid(True, alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cv-csv", default=find_default_cv_csv())
    p.add_argument("--out", default="predictor_scatter.png")
    args = p.parse_args()

    if not os.path.exists(args.cv_csv):
        print(f"ERROR: cv_results.csv not found at {args.cv_csv}", file=sys.stderr)
        print("Pass it with --cv-csv /path/to/cv_results.csv", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.cv_csv)
    print(f"Loaded {len(df)} rows from {args.cv_csv}")
    print(f"Columns: {list(df.columns)}")

    # Detect ground truth column
    true_col = next(
        (c for c in ["y_true_log10", "true_log10", "y_true", "true", "log10_true"]
         if c in df.columns),
        None,
    )
    if true_col is None:
        print(
            f"ERROR: could not find ground-truth column. Looked for "
            f"[y_true_log10, true_log10, y_true, true, log10_true]\n"
            f"Available: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Verify all four predictor columns exist
    missing = [c for c, _ in PREDICTOR_COLS if c not in df.columns]
    if missing:
        print(
            f"ERROR: missing predictor columns: {missing}\n"
            f"Available: {list(df.columns)}\n"
            "Edit PREDICTOR_COLS at top of script to match your file.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(f"Using true column: {true_col}")

    # Drop any rows with NaN in true or predictor columns
    needed = [true_col] + [c for c, _ in PREDICTOR_COLS]
    df = df[needed].dropna()
    print(f"After NaN drop: {len(df)} rows.")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (col, title) in zip(axes.flat, PREDICTOR_COLS):
        plot_panel(ax, df[true_col].values, df[col].values, title)

    fig.suptitle(
        "Predictor: 5-fold layer-grouped CV - predicted vs. true",
        fontsize=12,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}  ({os.path.getsize(args.out)/1024:.1f} KB)")


if __name__ == "__main__":
    main()