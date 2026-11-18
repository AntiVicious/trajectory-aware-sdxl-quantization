"""
Phase 2 predictor on the full atlas (3970 cells, 794 layers).

Uses structural features ONLY (no layer one-hot):
  - layer_type (Linear=0, Conv2d=1)
  - log_n_in, log_n_out, log_n_params
  - step_norm (step_from / 20)
  - role categorical (one-hot over ~20 role types — structural, transfers)
  - block depth (indicator 0/1 for up/down/mid)

CV scheme: 5-fold GroupKFold by layer — strictly held-out layers.

Trains 3 models for comparison:
  1. GradientBoostedRegressor (sklearn)
  2. MLP on dense features
  3. MLP on dense + role one-hot

Output:
  phase2_predictor_full/models/gbr_model.pkl
  phase2_predictor_full/models/mlp_dense.pt
  phase2_predictor_full/models/mlp_role.pt
  phase2_predictor_full/figures/atlas_heatmap_top60.png
  phase2_predictor_full/figures/predictor_comparison.png
  phase2_predictor_full/cv_results.csv
  phase2_predictor_full/summary.json

Usage:
    python train_predictors_full.py
"""
import json
import pickle
import re
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


BASE_DIR = Path(os.environ.get("SDXL_DATA_ROOT", "."))
ATLAS_CSV = BASE_DIR / "trajectory" / "phase1_atlas_full" / "results" / "atlas_full.csv"
LAYERS_JSON = BASE_DIR / "trajectory" / "phase1_atlas_full" / "layers_full.json"
OUT_DIR = BASE_DIR / "trajectory" / "phase2_predictor_full"
(OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

DEVICE = "cpu"
TOTAL_STEPS = 20


# ============ Role classification (matches enumerate_layers.py) ============

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
    """Return numeric block depth: down=[0-2], mid=3, up=[4-6], other=-1."""
    m = re.match(r"down_blocks\.(\d+)", name)
    if m:
        return int(m.group(1))
    if name.startswith("mid_block"):
        return 3
    m = re.match(r"up_blocks\.(\d+)", name)
    if m:
        return 4 + int(m.group(1))
    return -1


# ============ Feature matrix ============

def build_feature_matrix(df):
    """Dense structural features + role one-hot."""
    layer_type = (df["layer_type"] == "Conv2d").astype(np.float32).to_numpy()
    log_n_params = np.log10(df["n_params"].to_numpy(dtype=np.float32))
    step_norm = df["step_from"].to_numpy(dtype=np.float32) / TOTAL_STEPS

    # We need n_in and n_out, but the atlas CSV only has n_params and type.
    # Load from layers_full.json which has complete metadata.
    with open(LAYERS_JSON) as f:
        layers_meta = {L["name"]: L for L in json.load(f)["layers"]}
    n_in = np.array([layers_meta[n]["n_in"] for n in df["layer"]], dtype=np.float32)
    n_out = np.array([layers_meta[n]["n_out"] for n in df["layer"]], dtype=np.float32)
    log_n_in = np.log10(n_in + 1)
    log_n_out = np.log10(n_out + 1)

    # Role one-hot
    roles = [classify_role(n) for n in df["layer"]]
    role_set = sorted(set(roles))
    role_to_idx = {r: i for i, r in enumerate(role_set)}
    role_onehot = np.zeros((len(df), len(role_set)), dtype=np.float32)
    for i, r in enumerate(roles):
        role_onehot[i, role_to_idx[r]] = 1.0

    # Block depth one-hot
    block_depths = np.array([classify_block_depth(n) for n in df["layer"]])
    block_set = sorted(set(block_depths))
    block_to_idx = {b: i for i, b in enumerate(block_set)}
    block_onehot = np.zeros((len(df), len(block_set)), dtype=np.float32)
    for i, b in enumerate(block_depths):
        block_onehot[i, block_to_idx[b]] = 1.0

    # Normalize dense features
    dense = np.stack([layer_type, log_n_in, log_n_out, log_n_params, step_norm], axis=1)
    dense_mean = dense.mean(axis=0, keepdims=True)
    dense_std = dense.std(axis=0, keepdims=True) + 1e-8
    dense_z = (dense - dense_mean) / dense_std

    # Feature variants
    X_dense = dense_z.astype(np.float32)
    X_with_role = np.concatenate([dense_z, role_onehot, block_onehot], axis=1).astype(np.float32)

    y = np.log10(df["clip_dist_mean"].to_numpy(dtype=np.float32) + 1e-12)

    feat_names_dense = ["layer_type", "log_n_in", "log_n_out", "log_n_params", "step_norm"]
    feat_names_role = feat_names_dense + \
                      [f"role_{r}" for r in role_set] + \
                      [f"block_{b}" for b in block_set]

    meta = {
        "dense_mean": dense_mean, "dense_std": dense_std,
        "role_set": role_set, "block_set": [int(b) for b in block_set],
    }
    return X_dense, X_with_role, y, feat_names_dense, feat_names_role, meta


# ============ Models ============

class MLP(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_val, y_val, epochs=3000, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    model = MLP(X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)
    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        F.mse_loss(model(Xt), yt).backward()
        opt.step()
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                v = F.mse_loss(model(Xv), yv).item()
            if v < best_val:
                best_val = v
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def mlp_predict(m, X):
    with torch.no_grad():
        return m(torch.tensor(X, dtype=torch.float32)).cpu().numpy()


def make_gbr(X_tr, y_tr, *_):
    m = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0,
    )
    m.fit(X_tr, y_tr)
    return m


def gbr_predict(m, X):
    return m.predict(X)


# ============ CV ============

def eval_cv(X, y, groups, train_fn, pred_fn, n_splits=5):
    preds = np.zeros_like(y)
    unique_groups = np.unique(groups)
    actual_splits = min(n_splits, len(unique_groups))
    for tr, te in GroupKFold(n_splits=actual_splits).split(X, y, groups=groups):
        m = train_fn(X[tr], y[tr], X[te], y[te])
        preds[te] = pred_fn(m, X[te])
    return preds


def metrics(y_true, y_pred):
    return {
        "mae_log10": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "rank_corr": float(spearmanr(y_true, y_pred).statistic),
    }


# ============ Viz ============

def plot_heatmap_top_n(df, out_path, top_n=60):
    """Show top-N most-sensitive layers (by mean CLIP dist) as heatmap."""
    pivot = df.pivot_table(index="layer", columns="step_from", values="clip_dist_mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    pivot = pivot.head(top_n)
    log_mat = np.log10(pivot.values + 1e-12)

    # Short names for display
    def short(name):
        s = name
        s = s.replace("down_blocks.", "dn").replace("up_blocks.", "up")
        s = s.replace("mid_block", "mid").replace("attentions", "attn")
        s = s.replace(".transformer_blocks.0", "")
        s = s.replace("resnets", "res").replace(".to_out.0", ".to_out")
        s = s.replace(".net.0.proj", ".ff0").replace(".net.2", ".ff2")
        return s[-60:]

    fig, ax = plt.subplots(figsize=(7, 0.25 * top_n + 1.5))
    im = ax.imshow(log_mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([f"t={s}" for s in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([short(l) for l in pivot.index], fontsize=7, family="monospace")
    ax.set_xlabel("step_from (INT4 from this step onward)")
    ax.set_title(f"Top-{top_n} most sensitive layers (log10 CLIP distance)")
    plt.colorbar(im, ax=ax, label="log10(dist)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[phase2] Heatmap -> {out_path}")


def plot_comparison(y_true, preds_dict, out_path):
    fig, axes = plt.subplots(1, len(preds_dict), figsize=(5 * len(preds_dict), 4.5))
    if len(preds_dict) == 1:
        axes = [axes]
    vals = [y_true] + list(preds_dict.values())
    lo = float(min(v.min() for v in vals)) - 0.3
    hi = float(max(v.max() for v in vals)) + 0.3
    for ax, (name, pred) in zip(axes, preds_dict.items()):
        ax.scatter(y_true, pred, alpha=0.3, s=8)
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("true log10 CLIP dist"); ax.set_ylabel("pred log10 CLIP dist")
        m = metrics(y_true, pred)
        ax.set_title(f"{name}\nMAE={m['mae_log10']:.3f}, R²={m['r2']:.3f}, ρ={m['rank_corr']:.3f}")
        ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[phase2] Comparison -> {out_path}")


# ============ Main ============

def main():
    df = pd.read_csv(ATLAS_CSV)
    print(f"[phase2] Loaded {ATLAS_CSV.name}: {len(df)} cells, "
          f"{df['layer'].nunique()} unique layers")

    # Filter out rows with NaN/inf (shouldn't be any, but safety)
    before = len(df)
    df = df[np.isfinite(df["clip_dist_mean"])]
    if len(df) < before:
        print(f"[phase2] Dropped {before - len(df)} rows with non-finite clip_dist")

    # Stats
    c = df["clip_dist_mean"]
    print(f"[phase2] CLIP distance: min={c.min():.3e}  median={c.median():.3e}  "
          f"max={c.max():.3e}  spread={c.max()/c.min():.1f}×")

    plot_heatmap_top_n(df, OUT_DIR / "figures" / "atlas_heatmap_top60.png", 60)

    X_dense, X_role, y, feat_dense, feat_role, meta = build_feature_matrix(df)
    groups = df["layer"].to_numpy()
    print(f"[phase2] X_dense={X_dense.shape}  X_role={X_role.shape}  "
          f"y_range=[{y.min():.2f}, {y.max():.2f}] (log10)")

    # 5-fold GroupKFold CV
    print("\n[phase2] Training GBR (dense features) with 5-fold GroupKFold...")
    gbr_preds = eval_cv(X_dense, y, groups, make_gbr, gbr_predict)
    gbr_m = metrics(y, gbr_preds)
    print(f"[phase2] GBR:       {gbr_m}")

    print("[phase2] Training GBR (dense + role + block) with 5-fold GroupKFold...")
    gbr_role_preds = eval_cv(X_role, y, groups, make_gbr, gbr_predict)
    gbr_role_m = metrics(y, gbr_role_preds)
    print(f"[phase2] GBR+role:  {gbr_role_m}")

    print("[phase2] Training MLP (dense features)...")
    mlp_dense_preds = eval_cv(X_dense, y, groups, train_mlp, mlp_predict)
    mlp_dense_m = metrics(y, mlp_dense_preds)
    print(f"[phase2] MLP:       {mlp_dense_m}")

    print("[phase2] Training MLP (dense + role + block)...")
    mlp_role_preds = eval_cv(X_role, y, groups, train_mlp, mlp_predict)
    mlp_role_m = metrics(y, mlp_role_preds)
    print(f"[phase2] MLP+role:  {mlp_role_m}")

    # Save CV results
    cv_df = df.copy()
    cv_df["y_true_log10"] = y
    cv_df["gbr_pred"] = gbr_preds
    cv_df["gbr_role_pred"] = gbr_role_preds
    cv_df["mlp_pred"] = mlp_dense_preds
    cv_df["mlp_role_pred"] = mlp_role_preds
    cv_df.to_csv(OUT_DIR / "cv_results.csv", index=False)

    plot_comparison(y, {
        "GBR (dense)":     gbr_preds,
        "GBR (+role+blk)": gbr_role_preds,
        "MLP (dense)":     mlp_dense_preds,
        "MLP (+role+blk)": mlp_role_preds,
    }, OUT_DIR / "figures" / "predictor_comparison.png")

    # Pick winner based on Spearman rho
    results = [
        ("gbr_dense",      gbr_m,      X_dense,    make_gbr,    gbr_predict, feat_dense),
        ("gbr_role_block", gbr_role_m, X_role,     make_gbr,    gbr_predict, feat_role),
        ("mlp_dense",      mlp_dense_m, X_dense,   train_mlp,   mlp_predict, feat_dense),
        ("mlp_role_block", mlp_role_m, X_role,     train_mlp,   mlp_predict, feat_role),
    ]
    results.sort(key=lambda r: r[1]["rank_corr"], reverse=True)
    print(f"\n[phase2] Ranked by Spearman ρ:")
    for name, m, *_ in results:
        print(f"  {name:20s}  MAE={m['mae_log10']:.3f}  R²={m['r2']:.3f}  ρ={m['rank_corr']:.3f}")
    best = results[0]
    print(f"[phase2] Winner: {best[0]}")

    # Train winner on ALL data, save
    print(f"\n[phase2] Training final {best[0]} on all {len(df)} cells...")
    if "gbr" in best[0]:
        final = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0,
        )
        final.fit(best[2], y)
        with open(OUT_DIR / "models" / f"{best[0]}_final.pkl", "wb") as f:
            pickle.dump({"model": final, "meta": meta, "feat_names": best[5],
                         "variant": best[0]}, f)
    else:
        final = train_mlp(best[2], y, best[2], y)
        torch.save({
            "state_dict": final.state_dict(),
            "meta": meta, "feat_names": best[5],
            "n_features": best[2].shape[1],
            "variant": best[0],
        }, OUT_DIR / "models" / f"{best[0]}_final.pt")

    # Also always save GBR dense (robust baseline)
    gbr_final = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0,
    )
    gbr_final.fit(X_dense, y)
    with open(OUT_DIR / "models" / "gbr_dense_final.pkl", "wb") as f:
        pickle.dump({"model": gbr_final, "meta": meta, "feat_names": feat_dense,
                     "variant": "gbr_dense"}, f)

    summary = {
        "atlas_csv": str(ATLAS_CSV.name),
        "n_cells": len(df),
        "n_layers": int(df["layer"].nunique()),
        "metric": "CLIP-feature distance (1 - cos)",
        "clip_dist_range": [float(c.min()), float(c.max())],
        "cv_scheme": "GroupKFold(5)-by-layer",
        "cv_results": {
            "gbr_dense":      gbr_m,
            "gbr_role_block": gbr_role_m,
            "mlp_dense":      mlp_dense_m,
            "mlp_role_block": mlp_role_m,
        },
        "winner": best[0],
        "feat_dense": feat_dense,
        "feat_role_block": feat_role,
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[phase2] Saved -> {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()