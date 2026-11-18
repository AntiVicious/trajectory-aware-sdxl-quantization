"""
Compute additional metrics on the 5k benchmark across all variants.

Metrics added:
  - Aesthetic score (LAION's CLIP-based predictor v2.5)
  - CLIP-IQA (image quality assessment via CLIP)
  - Pick-a-Pic preference: V13-XX vs V1, V12 baseline (per-pair preference)

Output:
  benchmark_5k/extra_metrics_results.json
  prints summary tables

Usage:
    pip install pyiqa  # for CLIP-IQA
    pip install transformers  # for aesthetic and Pick-a-Pic
    CUDA_VISIBLE_DEVICES=0 python extra_metrics.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
OUT = BASE / "benchmark_5k"
PROMPTS = json.load(open(BASE / "benchmark_prompts_5k.json"))

VARIANTS = [
    "v1_fp16",
    "v2_int4_naive",
    "v4_hybrid_best",
    "v6_fully_distilled",
    "v7_timestep_lora",
    "v8_smoothquant_hybrid",
    "v12_outlier_preserved",
    "v13_70pct_trajectory",
    "v13_55pct_trajectory",
    "v13_40pct_trajectory",
]

# Subsample for speed — 5k * 10 variants * 3 metrics is a lot
# Aesthetic and CLIP-IQA are fast, can do all 5k.
# Pick-a-Pic compares pairs, do a 1k subset.
N_SUBSAMPLE_PICKAPIC = 1000


def filter_existing_variants(variants):
    """Keep only variants whose folder has images."""
    keep = []
    for v in variants:
        folder = OUT / v
        if folder.exists() and any(folder.iterdir()):
            keep.append(v)
        else:
            print(f"  [{v}] folder missing or empty — skipping")
    return keep


# =============================================================================
# Aesthetic scorer (LAION CLIP+MLP)
# =============================================================================

class AestheticMLP(nn.Module):
    """MLP head on top of CLIP image embeddings, trained on LAION aesthetic ratings."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.layers(x)


def load_aesthetic_scorer():
    """Try to download and load LAION aesthetic predictor v2."""
    from urllib.request import urlretrieve
    from transformers import CLIPProcessor, CLIPModel

    print("  Loading CLIP-L for aesthetic scoring...")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14",
                                       use_safetensors=True).to("cuda").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    weights_path = Path("/tmp/aesthetic_v2.pth")
    if not weights_path.exists():
        print("  Downloading aesthetic v2 weights...")
        url = ("https://github.com/christophschuhmann/improved-aesthetic-predictor/"
               "raw/main/sac+logos+ava1-l14-linearMSE.pth")
        urlretrieve(url, weights_path)

    head = AestheticMLP().to("cuda")
    state = torch.load(weights_path, map_location="cuda", weights_only=False)
    head.load_state_dict(state)
    head.eval()

    @torch.no_grad()
    def score(img_pil):
        inp = proc(images=img_pil, return_tensors="pt").to("cuda")
        feats = model.get_image_features(**inp)
        # Handle different return types across transformers versions:
        if isinstance(feats, torch.Tensor):
            pass  # already a tensor
        elif hasattr(feats, "image_embeds") and feats.image_embeds is not None:
            feats = feats.image_embeds
        elif hasattr(feats, "pooler_output") and feats.pooler_output is not None:
            # Newer transformers: pooler_output of get_image_features() is already
            # the projected feature (768-dim for CLIP-L). Don't re-project.
            feats = feats.pooler_output
        elif hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state[:, 0]
        else:
            raise RuntimeError(f"Unexpected get_image_features return: {type(feats)}")
        feats = F.normalize(feats, dim=-1)
        return head(feats).squeeze().item()

    return score


def compute_aesthetic(variants):
    print("\n=== Aesthetic Score (higher = better) ===")
    scorer = load_aesthetic_scorer()
    results = {}
    for v in variants:
        folder = OUT / v
        scores = []
        t0 = time.time()
        for i, item in enumerate(PROMPTS):
            p = folder / item['filename']
            if not p.exists():
                continue
            try:
                img = Image.open(p).convert("RGB")
                scores.append(scorer(img))
            except Exception as e:
                print(f"    err on {item['filename']}: {e}")
            if (i + 1) % 1000 == 0:
                print(f"  [{v}] {i+1}/{len(PROMPTS)}  running mean={np.mean(scores):.3f}")
        if scores:
            arr = np.array(scores)
            results[v] = {"mean": float(arr.mean()),
                          "std": float(arr.std()),
                          "n": len(arr)}
            print(f"  [{v}] aesthetic = {arr.mean():.3f} ± {arr.std():.3f}  "
                  f"(n={len(arr)}, {(time.time()-t0)/60:.1f}min)")
    return results


# =============================================================================
# CLIP-IQA via pyiqa
# =============================================================================

def compute_clip_iqa(variants):
    print("\n=== CLIP-IQA Score (0-1, higher = better quality) ===")
    try:
        import pyiqa
    except ImportError:
        print("  pyiqa not installed. Skipping CLIP-IQA.")
        return {}

    iqa = pyiqa.create_metric("clipiqa", device="cuda")
    results = {}
    for v in variants:
        folder = OUT / v
        scores = []
        t0 = time.time()
        for i, item in enumerate(PROMPTS):
            p = folder / item['filename']
            if not p.exists():
                continue
            try:
                with torch.no_grad():
                    score = iqa(str(p)).item()
                scores.append(score)
            except Exception as e:
                print(f"    err on {item['filename']}: {e}")
            if (i + 1) % 1000 == 0:
                print(f"  [{v}] {i+1}/{len(PROMPTS)}  running mean={np.mean(scores):.4f}")
        if scores:
            arr = np.array(scores)
            results[v] = {"mean": float(arr.mean()),
                          "std": float(arr.std()),
                          "n": len(arr)}
            print(f"  [{v}] CLIP-IQA = {arr.mean():.4f} ± {arr.std():.4f}  "
                  f"(n={len(arr)}, {(time.time()-t0)/60:.1f}min)")
    return results


# =============================================================================
# Pick-a-Pic preference (vs V1 baseline)
# =============================================================================

def compute_pickapic(variants, baseline="v1_fp16", n_pairs=N_SUBSAMPLE_PICKAPIC):
    print(f"\n=== Pick-a-Pic preference vs {baseline} (P > 0.5 means preferred over baseline) ===")
    try:
        from transformers import AutoProcessor, AutoModel
    except ImportError:
        print("  transformers needed for Pick-a-Pic. Skipping.")
        return {}

    print("  Loading Pick-a-Pic v2 model...")
    try:
        proc = AutoProcessor.from_pretrained(
            "yuvalkirstain/PickScore_v1", use_safetensors=True)
        model = AutoModel.from_pretrained(
            "yuvalkirstain/PickScore_v1", use_safetensors=True).to("cuda").eval()
    except Exception as e:
        print(f"  Could not load PickScore_v1: {e}")
        print("  Skipping Pick-a-Pic")
        return {}

    def _extract_tensor(out):
        """Coerce CLIP feature output to a tensor across transformers versions.
        Note: pooler_output from get_*_features() is already projected — don't reproject.
        """
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "image_embeds") and out.image_embeds is not None:
            return out.image_embeds
        if hasattr(out, "text_embeds") and out.text_embeds is not None:
            return out.text_embeds
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output  # already projected
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0]
        raise RuntimeError(f"Cannot extract tensor from {type(out)}")

    @torch.no_grad()
    def pick_score(prompt, image):
        text_inp = proc(text=prompt, padding=True, truncation=True,
                        max_length=77, return_tensors="pt").to("cuda")
        img_inp = proc(images=image, return_tensors="pt").to("cuda")
        text_emb = _extract_tensor(model.get_text_features(**text_inp))
        img_emb = _extract_tensor(model.get_image_features(**img_inp))
        text_emb = F.normalize(text_emb, dim=-1)
        img_emb = F.normalize(img_emb, dim=-1)
        return (text_emb * img_emb).sum().item()

    rng = np.random.default_rng(42)
    indices = rng.choice(len(PROMPTS), size=min(n_pairs, len(PROMPTS)), replace=False)
    indices = sorted(indices.tolist())

    baseline_dir = OUT / baseline
    results = {}
    for v in variants:
        if v == baseline:
            continue
        folder = OUT / v
        wins, ties, total = 0, 0, 0
        score_diff_sum = 0.0
        t0 = time.time()
        for ii, i in enumerate(indices):
            item = PROMPTS[i]
            p_base = baseline_dir / item['filename']
            p_var = folder / item['filename']
            if not (p_base.exists() and p_var.exists()):
                continue
            try:
                img_base = Image.open(p_base).convert("RGB")
                img_var = Image.open(p_var).convert("RGB")
                s_base = pick_score(item['caption'], img_base)
                s_var = pick_score(item['caption'], img_var)
                score_diff_sum += (s_var - s_base)
                if s_var > s_base + 1e-6:
                    wins += 1
                elif abs(s_var - s_base) <= 1e-6:
                    ties += 1
                total += 1
            except Exception as e:
                print(f"    err: {e}")
            if (ii + 1) % 200 == 0:
                wr = wins / total if total else 0
                print(f"  [{v}] {ii+1}/{len(indices)}  win_rate={wr:.3f}")
        if total:
            results[v] = {
                "n_compared": total,
                "win_rate": wins / total,
                "tie_rate": ties / total,
                "loss_rate": (total - wins - ties) / total,
                "mean_score_diff": score_diff_sum / total,
            }
            print(f"  [{v}] vs {baseline}: win={wins/total:.3f}  tie={ties/total:.3f}  "
                  f"loss={(total-wins-ties)/total:.3f}  mean_diff={score_diff_sum/total:+.4f}")
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    variants = filter_existing_variants(VARIANTS)
    print(f"Variants present: {variants}")

    aesthetic = compute_aesthetic(variants)
    torch.cuda.empty_cache()
    clip_iqa = compute_clip_iqa(variants)
    torch.cuda.empty_cache()
    pickapic = compute_pickapic(variants, baseline="v1_fp16")

    # Print summary
    print("\n" + "=" * 100)
    print(f"{'Variant':<30} {'Aesthetic':>10} {'CLIP-IQA':>10} {'Pick-vs-V1 win':>15}")
    print("=" * 100)
    for v in variants:
        aes = f"{aesthetic.get(v, {}).get('mean', '—'):.3f}" if v in aesthetic else "—"
        ciq = f"{clip_iqa.get(v, {}).get('mean', '—'):.4f}" if v in clip_iqa else "—"
        pap = f"{pickapic.get(v, {}).get('win_rate', '—'):.3f}" if v in pickapic else "—" if v != "v1_fp16" else "(ref)"
        print(f"{v:<30} {aes:>10} {ciq:>10} {pap:>15}")

    out_path = OUT / "extra_metrics_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "aesthetic": aesthetic,
            "clip_iqa": clip_iqa,
            "pickapic_vs_v1": pickapic,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()