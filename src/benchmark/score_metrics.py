import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from cleanfid import fid

BASE     = os.environ.get("SDXL_DATA_ROOT", ".")
OUT      = BASE + "benchmark_5k/"
REAL_DIR = BASE + "val2017/"
PROMPTS  = json.load(open(BASE + "benchmark_prompts_5k.json"))

VARIANTS = {
    "v1_fp16":               "V1 — FP16 baseline",
    "v2_int4_naive":         "V2 — Naive INT4",
    "v4_hybrid_best":        "V4 — Hybrid: ternary mid + INT4 (naive)",
    "v6_fully_distilled":    "V6 — Fully distilled hybrid",
    "v7_timestep_lora":      "V7 — Timestep-aware LoRA hybrid",
    "v8_smoothquant_hybrid": "V8 — SmoothQuant INT4 + ternary mid",
    "v12_outlier_preserved": "V12 — Outlier-preserved INT4 + ternary mid",
}

MEMORY = {
    "v1_fp16":               {"size_mb": 4487, "reduction": 0.0},
    "v2_int4_naive":         {"size_mb": 1258, "reduction": 71.9},
    "v4_hybrid_best":        {"size_mb": 1228, "reduction": 72.6},
    "v6_fully_distilled":    {"size_mb": 1228, "reduction": 72.6},
    "v7_timestep_lora":      {"size_mb": 1228, "reduction": 72.6},
    "v8_smoothquant_hybrid": {"size_mb": 1228, "reduction": 72.6},
    "v12_outlier_preserved": {"size_mb": 1248, "reduction": 72.2},
}

# ── CLIP ──────────────────────────────────────────────────────────────
print("Loading CLIP...")
model     = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch16").to("cuda")
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch16")
model.eval()

clip_scores = {}
print("Computing CLIP scores on 5000 images per variant...")
for variant, label in VARIANTS.items():
    folder = os.path.join(OUT, variant)
    if not os.path.exists(folder):
        print(f"  [{label}] Folder not found — skipping")
        continue
    scores = []
    for item in PROMPTS:
        img_path = os.path.join(folder, item['filename'])
        if not os.path.exists(img_path):
            continue
        img    = Image.open(img_path).convert("RGB")
        inputs = processor(
            text=[item['caption']], images=img,
            return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to("cuda")
        with torch.inference_mode():
            out     = model(**inputs)
            img_emb = F.normalize(out.image_embeds, dim=-1)
            txt_emb = F.normalize(out.text_embeds,  dim=-1)
            score   = (img_emb * txt_emb).sum().item()
        scores.append(score)
    avg = sum(scores) / len(scores) if scores else 0
    clip_scores[variant] = round(avg, 4)
    print(f"  [{label}] CLIP = {avg:.4f}  ({len(scores)} images)")

del model, processor
torch.cuda.empty_cache()

# ── FID ───────────────────────────────────────────────────────────────
print("\nComputing FID scores vs 5000 real COCO images...")
fid_scores = {}
for variant, label in VARIANTS.items():
    folder = os.path.join(OUT, variant)
    if not os.path.exists(folder):
        continue
    try:
        score = fid.compute_fid(folder, REAL_DIR)
        fid_scores[variant] = round(score, 2)
        print(f"  [{label}] FID = {score:.2f}")
    except Exception as e:
        print(f"  [{label}] FID error: {e}")
        fid_scores[variant] = -1

# ── Final table ───────────────────────────────────────────────────────
print("\n" + "=" * 90)
print(f"{'Variant':<44} {'CLIP':>7} {'FID':>8} "
      f"{'Size MB':>9} {'Reduction':>10}")
print("=" * 90)
for v, label in VARIANTS.items():
    if v in clip_scores:
        print(f"{label:<44} "
              f"{clip_scores[v]:>7.4f} "
              f"{fid_scores.get(v, -1):>8.2f} "
              f"{MEMORY[v]['size_mb']:>9} "
              f"{MEMORY[v]['reduction']:>9.1f}%")

# ── Save ──────────────────────────────────────────────────────────────
with open(OUT + "scores_5k.json", "w") as f:
    json.dump({
        "clip": clip_scores,
        "fid":  fid_scores,
        "memory": MEMORY,
        "n_images": 5000,
        "n_real_reference": 5000
    }, f, indent=2)
print(f"\nSaved: {OUT}scores_5k.json")