"""
Score V14, V20, V21 (all 3 budgets each) across all metrics.

Metrics:
  - CLIP score (image-text alignment)
  - FID (image distribution distance to COCO val2017)
  - DreamSim (paired identity preservation vs V1)
  - Aesthetic (LAION's improved-aesthetic-predictor)
  - CLIP-IQA (zero-shot image quality)

Reuses cached results when present. Skips PickScore (we dropped it).

Usage:
    CUDA_VISIBLE_DEVICES=1 python -u score_v14_v20_v21.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
import time
import urllib.request
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from cleanfid import fid

BASE = Path.home() / "Documents" / "Maharshi" / "gem"
OUT = BASE / "benchmark_5k"
REAL_DIR = BASE / "val2017"
PROMPTS = json.load(open(BASE / "benchmark_prompts_5k.json"))

TARGET_VARIANTS = [
    "v24b_int8_full",
    "v24b_int8_full",
    "v22_70pct_floor5",
    "v22_55pct_floor5",
    "v22_40pct_floor5",
       "v22_70pct_hotswap",
       "v22_55pct_hotswap",
       "v22_40pct_hotswap",
   ]

# Memory mapping for new variants
ALLOC_DIR = BASE / "trajectory" / "phase3_allocator_full"


# ===================== Helpers =====================

def unwrap_image_features(out):
    if torch.is_tensor(out): return out
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    raise RuntimeError("Could not unwrap image features")


def unwrap_text_features(out):
    if torch.is_tensor(out): return out
    if hasattr(out, "text_embeds") and out.text_embeds is not None:
        return out.text_embeds
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    raise RuntimeError("Could not unwrap text features")


# ===================== CLIP score (using base CLIP) =====================

def score_clip(model, processor, folder, prompts):
    scores = []
    for item in prompts:
        p = folder / item['filename']
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        inputs = processor(
            text=[item['caption']], images=img,
            return_tensors="pt", padding=True, truncation=True, max_length=77,
        ).to("cuda")
        with torch.inference_mode():
            out = model(**inputs)
            img_emb = F.normalize(unwrap_image_features(out.image_embeds if hasattr(out, 'image_embeds') else out), dim=-1)
            txt_emb = F.normalize(unwrap_text_features(out.text_embeds if hasattr(out, 'text_embeds') else out), dim=-1)
            score = (img_emb * txt_emb).sum().item()
        scores.append(score)
    return scores


# ===================== Aesthetic =====================

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
    cache_dir = Path.home() / ".cache" / "aesthetic_predictor"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / "sa_0_4_vit_l_14_linear.pth"
    if not weights_path.exists():
        url = "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth"
        urllib.request.urlretrieve(url, weights_path)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if any(k.startswith("layers") for k in state.keys()):
        m = AestheticPredictor()
        m.load_state_dict(state)
    else:
        m = nn.Linear(768, 1)
        m.load_state_dict(state)
    return m.eval().to("cuda")


def score_aesthetic(model_clip_l, processor_clip_l, predictor, folder, prompts):
    scores = []
    for item in prompts:
        p = folder / item['filename']
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        inputs = processor_clip_l(images=img, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            emb = unwrap_image_features(model_clip_l.get_image_features(**inputs))
            emb = F.normalize(emb, dim=-1)
            score = predictor(emb).item()
        scores.append(score)
    return scores


# ===================== CLIP-IQA =====================

CLIP_IQA_PROMPTS = [
    ("Good photo.", "Bad photo."),
    ("Sharp photo.", "Blurry photo."),
    ("High-quality photo.", "Low-quality photo."),
]


def score_clip_iqa(model_clip_l, processor_clip_l, folder, prompts):
    pos_embs = []
    neg_embs = []
    with torch.inference_mode():
        for pos, neg in CLIP_IQA_PROMPTS:
            txt_inputs = processor_clip_l(
                text=[pos, neg], return_tensors="pt",
                padding=True, truncation=True
            ).to("cuda")
            tf = unwrap_text_features(model_clip_l.get_text_features(**txt_inputs))
            tf = F.normalize(tf, dim=-1)
            pos_embs.append(tf[0])
            neg_embs.append(tf[1])
    pos_emb = F.normalize(torch.stack(pos_embs).mean(dim=0, keepdim=True), dim=-1)
    neg_emb = F.normalize(torch.stack(neg_embs).mean(dim=0, keepdim=True), dim=-1)

    scores = []
    for item in prompts:
        p = folder / item['filename']
        if not p.exists():
            continue
        img = Image.open(p).convert("RGB")
        inputs = processor_clip_l(images=img, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            img_emb = unwrap_image_features(model_clip_l.get_image_features(**inputs))
            img_emb = F.normalize(img_emb, dim=-1)
            sim_pos = (img_emb @ pos_emb.T).item()
            sim_neg = (img_emb @ neg_emb.T).item()
            x = np.array([sim_pos, sim_neg]) * 100
            p_good = np.exp(x[0]) / np.exp(x).sum()
        scores.append(p_good)
    return scores


# ===================== Memory =====================

def get_memory_for_variant(v):
    """Return {size_mb, reduction} for a variant."""
    parts = v.split("_")
    if "70pct" in v: pct = 70
    elif "55pct" in v: pct = 55
    elif "40pct" in v: pct = 40
    else: return {"size_mb": -1, "reduction": -1}

    alloc_path = ALLOC_DIR / f"allocation_{pct}pct_full.json"
    if not alloc_path.exists():
        return {"size_mb": -1, "reduction": -1}
    a = json.load(open(alloc_path))
    return {"size_mb": int(a["actual_MB"]),
            "reduction": round((5131.8 - int(a["actual_MB"])) / 5131.8 * 100, 1)}


# ===================== Main =====================

def main():
    # Load existing results
    clip_path = OUT / "scores_5k_with_ablations.json"
    extra_path = OUT / "extra_metrics_results.json"
    dreamsim_path = OUT / "dreamsim_results.json"

    clip_data = json.load(open(clip_path)) if clip_path.exists() else {"clip": {}, "fid": {}, "memory": {}}
    extra_data = json.load(open(extra_path)) if extra_path.exists() else {"aesthetic": {}, "clip_iqa": {}}
    dreamsim_data = json.load(open(dreamsim_path)) if dreamsim_path.exists() else {"results": {}}

    # ==================== CLIP + FID + Aesthetic + CLIP-IQA ====================
    print("Loading CLIP base (for CLIP score) and CLIP-L (for aesthetic+IQA)...")
    clip_b = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True).to("cuda").eval()
    proc_b = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    clip_l = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", use_safetensors=True).to("cuda").eval()
    proc_l = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    print("Loading aesthetic predictor...")
    try:
        predictor = load_aesthetic_predictor()
    except Exception as e:
        print(f"  Aesthetic predictor failed: {e}")
        predictor = None

    for variant in TARGET_VARIANTS:
        folder = OUT / variant
        if not folder.exists():
            print(f"  [{variant}] folder missing — skipping")
            continue
        n_existing = len([f for f in folder.iterdir() if f.suffix == '.jpg'])
        if n_existing < 100:
            print(f"  [{variant}] only {n_existing} images — skipping")
            continue

        # Memory mapping
        if variant not in clip_data.get("memory", {}):
            clip_data.setdefault("memory", {})[variant] = get_memory_for_variant(variant)

        # CLIP
        if variant in clip_data.get("clip", {}):
            print(f"  [{variant}] CLIP cached")
        else:
            print(f"\nCLIP for {variant}...")
            t0 = time.time()
            scores = score_clip(clip_b, proc_b, folder, PROMPTS)
            clip_data.setdefault("clip", {})[variant] = round(np.mean(scores), 4)
            print(f"  CLIP={np.mean(scores):.4f}  ({len(scores)} images, {(time.time()-t0)/60:.1f}min)")

        # FID
        if variant in clip_data.get("fid", {}):
            print(f"  [{variant}] FID cached")
        else:
            print(f"  FID for {variant}...")
            t0 = time.time()
            try:
                f = fid.compute_fid(str(folder), str(REAL_DIR))
                clip_data.setdefault("fid", {})[variant] = round(f, 2)
                print(f"  FID={f:.2f}  ({(time.time()-t0)/60:.1f}min)")
            except Exception as e:
                print(f"  FID failed: {e}")
                clip_data.setdefault("fid", {})[variant] = -1

        # Aesthetic
        if variant in extra_data.get("aesthetic", {}):
            print(f"  [{variant}] Aesthetic cached")
        elif predictor is not None:
            print(f"  Aesthetic for {variant}...")
            t0 = time.time()
            scores = score_aesthetic(clip_l, proc_l, predictor, folder, PROMPTS)
            extra_data.setdefault("aesthetic", {})[variant] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "n": len(scores),
            }
            print(f"  Aesthetic={np.mean(scores):.3f}  ({(time.time()-t0)/60:.1f}min)")

        # CLIP-IQA
        if variant in extra_data.get("clip_iqa", {}):
            print(f"  [{variant}] CLIP-IQA cached")
        else:
            print(f"  CLIP-IQA for {variant}...")
            t0 = time.time()
            scores = score_clip_iqa(clip_l, proc_l, folder, PROMPTS)
            extra_data.setdefault("clip_iqa", {})[variant] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "n": len(scores),
            }
            print(f"  CLIP-IQA={np.mean(scores):.4f}  ({(time.time()-t0)/60:.1f}min)")

        # Save partial progress
        with open(clip_path, "w") as f:
            json.dump(clip_data, f, indent=2)
        with open(extra_path, "w") as f:
            json.dump(extra_data, f, indent=2)

    # Free CLIP models before DreamSim
    del clip_b, proc_b, clip_l, proc_l, predictor
    torch.cuda.empty_cache()

    # ==================== DreamSim (paired vs V1) ====================
    print("\n\nLoading DreamSim...")
    from dreamsim import dreamsim
    ds_model, ds_preprocess = dreamsim(pretrained=True, device="cuda")
    ds_model.eval()

    v1_dir = OUT / "v1_fp16"
    for variant in TARGET_VARIANTS:
        folder = OUT / variant
        if not folder.exists() or not v1_dir.exists():
            continue
        if variant in dreamsim_data.get("results", {}) and \
           dreamsim_data["results"][variant].get("n_compared", 0) >= 4500:
            print(f"  [{variant}] DreamSim cached")
            continue

        print(f"\nDreamSim for {variant} vs V1...")
        t0 = time.time()
        distances = []
        for item in PROMPTS:
            p_v1 = v1_dir / item['filename']
            p_var = folder / item['filename']
            if not (p_v1.exists() and p_var.exists()):
                continue
            try:
                img_v1 = ds_preprocess(Image.open(p_v1).convert("RGB")).to("cuda")
                img_var = ds_preprocess(Image.open(p_var).convert("RGB")).to("cuda")
                with torch.no_grad():
                    d = ds_model(img_v1, img_var).item()
                distances.append(d)
            except Exception:
                continue
        d = np.array(distances)
        dreamsim_data.setdefault("results", {})[variant] = {
            "n_compared": len(distances),
            "mean": float(d.mean()),
            "std": float(d.std()),
            "median": float(np.median(d)),
            "p25": float(np.percentile(d, 25)),
            "p75": float(np.percentile(d, 75)),
        }
        print(f"  DreamSim={d.mean():.4f}  ({len(distances)} pairs, {(time.time()-t0)/60:.1f}min)")
        with open(dreamsim_path, "w") as f:
            json.dump({"reference": "v1_fp16",
                        "metric": "DreamSim (LDS distance)",
                        "n_total_prompts": len(PROMPTS),
                        "results": dreamsim_data["results"]}, f, indent=2)

    # ==================== Final summary ====================
    print("\n" + "=" * 110)
    print(f"{'Variant':<40} {'CLIP':>8} {'FID':>8} {'DSim':>8} {'Aesthetic':>10} {'CLIP-IQA':>10}")
    print("=" * 110)
    for v in TARGET_VARIANTS:
        clip = clip_data.get("clip", {}).get(v, "—")
        fid_v = clip_data.get("fid", {}).get(v, "—")
        dsim = dreamsim_data.get("results", {}).get(v, {}).get("mean", "—")
        aesth = extra_data.get("aesthetic", {}).get(v, {}).get("mean", "—")
        iqa = extra_data.get("clip_iqa", {}).get(v, {}).get("mean", "—")
        clip_s = f"{clip:.4f}" if isinstance(clip, float) else "—"
        fid_s = f"{fid_v:.2f}" if isinstance(fid_v, float) else "—"
        dsim_s = f"{dsim:.4f}" if isinstance(dsim, float) else "—"
        aesth_s = f"{aesth:.3f}" if isinstance(aesth, float) else "—"
        iqa_s = f"{iqa:.4f}" if isinstance(iqa, float) else "—"
        print(f"{v:<40} {clip_s:>8} {fid_s:>8} {dsim_s:>8} {aesth_s:>10} {iqa_s:>10}")

    print("\nAll scoring complete.")
    print(f"  CLIP/FID/Memory: {clip_path}")
    print(f"  Aesthetic/IQA:   {extra_path}")
    print(f"  DreamSim:        {dreamsim_path}")


if __name__ == "__main__":
    main()
