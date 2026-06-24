"""
Score V19 (and optionally V13 if missing) on aesthetic, CLIP-IQA, and PickScore.
PickScore is paired against V1.

This version handles transformers 5.x where get_image_features / get_text_features
return BaseModelOutputWithPooling rather than raw tensors.

Reuses cached results from extra_metrics_results.json if present.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -u score_extra_metrics_v19.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoProcessor, AutoModel,
)

BASE = Path.home() / "Documents" / "Maharshi" / "gem"
OUT = BASE / "benchmark_5k"
PROMPTS = json.load(open(BASE / "benchmark_prompts_5k.json"))

TARGET_VARIANTS = [
    "v19_70pct_trajectory_v12quant",
    "v13_70pct_trajectory",
    "v19_55pct_trajectory_v12quant",
    "v19_40pct_trajectory_v12quant",
]

N_PICK_SAMPLE = 1000


# -------------- Helpers to unwrap CLIP outputs --------------

def unwrap_image_features(out):
    """Return a tensor regardless of whether `out` is already a tensor or a
    HuggingFace BaseModelOutputWithPooling-like object."""
    if torch.is_tensor(out):
        return out
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        # Last resort: take CLS token
        return out.last_hidden_state[:, 0]
    raise RuntimeError(f"Could not unwrap image features from {type(out)}")


def unwrap_text_features(out):
    if torch.is_tensor(out):
        return out
    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    raise RuntimeError(f"Could not unwrap text features from {type(out)}")


# -------------- Aesthetic predictor --------------

class AestheticPredictor(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.layers(x)


def load_aesthetic_predictor():
    import urllib.request
    cache_dir = Path.home() / ".cache" / "aesthetic_predictor"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / "sa_0_4_vit_l_14_linear.pth"
    if not weights_path.exists():
        url = "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth"
        print(f"Downloading aesthetic predictor weights from {url}...")
        urllib.request.urlretrieve(url, weights_path)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if any(k.startswith("layers") for k in state.keys()):
        m = AestheticPredictor()
        m.load_state_dict(state)
    else:
        m = nn.Linear(768, 1)
        m.load_state_dict(state)
    return m.eval().to("cuda")


def score_aesthetic(model_clip, processor_clip, predictor, folder, prompts):
    scores = []
    for item in prompts:
        p = folder / item['filename']
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        inputs = processor_clip(images=img, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            emb = unwrap_image_features(model_clip.get_image_features(**inputs))
            emb = F.normalize(emb, dim=-1)
            score = predictor(emb).item()
        scores.append(score)
    return np.array(scores)


# -------------- CLIP-IQA --------------

CLIP_IQA_PROMPTS = [
    ("Good photo.", "Bad photo."),
    ("Sharp photo.", "Blurry photo."),
    ("High-quality photo.", "Low-quality photo."),
]


def score_clip_iqa(model_clip, processor_clip, folder, prompts):
    pos_embs = []
    neg_embs = []
    with torch.inference_mode():
        for pos, neg in CLIP_IQA_PROMPTS:
            txt_inputs = processor_clip(
                text=[pos, neg], return_tensors="pt",
                padding=True, truncation=True
            ).to("cuda")
            tf = unwrap_text_features(model_clip.get_text_features(**txt_inputs))
            tf = F.normalize(tf, dim=-1)
            pos_embs.append(tf[0])
            neg_embs.append(tf[1])
    pos_emb = torch.stack(pos_embs).mean(dim=0, keepdim=True)
    neg_emb = torch.stack(neg_embs).mean(dim=0, keepdim=True)
    pos_emb = F.normalize(pos_emb, dim=-1)
    neg_emb = F.normalize(neg_emb, dim=-1)

    scores = []
    for item in prompts:
        p = folder / item['filename']
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        inputs = processor_clip(images=img, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            img_emb = unwrap_image_features(model_clip.get_image_features(**inputs))
            img_emb = F.normalize(img_emb, dim=-1)
            sim_pos = (img_emb @ pos_emb.T).item()
            sim_neg = (img_emb @ neg_emb.T).item()
            x = np.array([sim_pos, sim_neg]) * 100
            p_good = np.exp(x[0]) / np.exp(x).sum()
        scores.append(p_good)
    return np.array(scores)


# -------------- PickScore --------------

def load_pickscore():
    processor = AutoProcessor.from_pretrained("yuvalkirstain/PickScore_v1")
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to("cuda").eval()
    return model, processor


def pickscore_compare(model, processor, prompt, img_a, img_b):
    img_inputs = processor(
        images=[img_a, img_b], padding=True, truncation=True,
        return_tensors="pt"
    ).to("cuda")
    txt_inputs = processor(
        text=prompt, padding=True, truncation=True, max_length=77,
        return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        img_embs = unwrap_image_features(model.get_image_features(**img_inputs))
        img_embs = F.normalize(img_embs, dim=-1)
        txt_emb = unwrap_text_features(model.get_text_features(**txt_inputs))
        txt_emb = F.normalize(txt_emb, dim=-1)
        scores = (txt_emb @ img_embs.T) * model.logit_scale.exp()
        probs = F.softmax(scores, dim=-1).squeeze()
    return probs[0].item()


def main():
    results_path = OUT / "extra_metrics_results.json"
    results = {}
    if results_path.exists():
        try:
            results = json.load(open(results_path))
            print(f"Loaded existing results from {results_path}")
        except Exception:
            pass
    results.setdefault("aesthetic", {})
    results.setdefault("clip_iqa", {})
    results.setdefault("pickapic_vs_v1", {})

    print("Loading CLIP ViT-L/14 (for aesthetic + CLIP-IQA)...")
    clip_l_name = "openai/clip-vit-large-patch14"
    model_clip_l = CLIPModel.from_pretrained(clip_l_name, use_safetensors=True).to("cuda").eval()
    processor_clip_l = CLIPProcessor.from_pretrained(clip_l_name)

    print("Loading aesthetic predictor...")
    try:
        predictor = load_aesthetic_predictor()
    except Exception as e:
        print(f"  Aesthetic predictor failed to load: {e}")
        predictor = None

    for variant in TARGET_VARIANTS:
        folder = OUT / variant
        if not folder.exists():
            print(f"  [{variant}] folder missing — skipping")
            continue

        if variant in results["aesthetic"]:
            print(f"  [{variant}] aesthetic cached — skipping")
        elif predictor is not None:
            print(f"\nScoring aesthetic for {variant}...")
            t0 = time.time()
            scores = score_aesthetic(model_clip_l, processor_clip_l,
                                      predictor, folder, PROMPTS)
            results["aesthetic"][variant] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "n": len(scores),
            }
            print(f"  aesthetic mean={scores.mean():.3f}  ({len(scores)} images, {(time.time()-t0)/60:.1f}min)")

        if variant in results["clip_iqa"]:
            print(f"  [{variant}] CLIP-IQA cached — skipping")
        else:
            print(f"\nScoring CLIP-IQA for {variant}...")
            t0 = time.time()
            scores = score_clip_iqa(model_clip_l, processor_clip_l, folder, PROMPTS)
            results["clip_iqa"][variant] = {
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "n": len(scores),
            }
            print(f"  CLIP-IQA mean={scores.mean():.4f}  ({len(scores)} images, {(time.time()-t0)/60:.1f}min)")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    del model_clip_l, processor_clip_l, predictor
    torch.cuda.empty_cache()

    v1_dir = OUT / "v1_fp16"
    if not v1_dir.exists():
        print("V1 folder missing, skipping PickScore")
    else:
        print("\nLoading PickScore...")
        try:
            ps_model, ps_processor = load_pickscore()
        except Exception as e:
            print(f"PickScore load failed: {e}")
            ps_model = None

        if ps_model is not None:
            for variant in TARGET_VARIANTS:
                if variant in results["pickapic_vs_v1"]:
                    print(f"  [{variant}] PickScore cached — skipping")
                    continue
                folder = OUT / variant
                if not folder.exists():
                    continue
                print(f"\nPickScore for {variant} (vs V1, {N_PICK_SAMPLE} prompts)...")
                t0 = time.time()
                wins = 0
                ties = 0
                n = 0
                for item in PROMPTS[:N_PICK_SAMPLE]:
                    p_v1 = v1_dir / item['filename']
                    p_var = folder / item['filename']
                    if not (p_v1.exists() and p_var.exists()):
                        continue
                    img_v1 = Image.open(p_v1).convert("RGB")
                    img_var = Image.open(p_var).convert("RGB")
                    p_var_pref = pickscore_compare(
                        ps_model, ps_processor, item['caption'], img_var, img_v1)
                    n += 1
                    if p_var_pref > 0.55:
                        wins += 1
                    elif p_var_pref < 0.45:
                        pass
                    else:
                        ties += 1
                if n > 0:
                    win_rate = wins / n
                    results["pickapic_vs_v1"][variant] = {
                        "n": n, "wins": wins, "ties": ties,
                        "win_rate": float(win_rate),
                    }
                    print(f"  win-rate vs V1: {win_rate:.3f} ({wins}/{n}, ties={ties}, "
                          f"{(time.time()-t0)/60:.1f}min)")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    print("\n" + "=" * 80)
    print(f"{'Variant':<40} {'Aesthetic':>10} {'CLIP-IQA':>10} {'PickScore-vs-V1':>16}")
    print("=" * 80)
    for v in TARGET_VARIANTS:
        a = results["aesthetic"].get(v, {}).get("mean", "—")
        c = results["clip_iqa"].get(v, {}).get("mean", "—")
        p = results["pickapic_vs_v1"].get(v, {}).get("win_rate", "—")
        a_str = f"{a:.3f}" if isinstance(a, float) else "—"
        c_str = f"{c:.4f}" if isinstance(c, float) else "—"
        p_str = f"{p:.3f}" if isinstance(p, float) else "—"
        print(f"{v:<40} {a_str:>10} {c_str:>10} {p_str:>16}")


if __name__ == "__main__":
    main()