"""
Final 5k scoring including V14 (SmoothQuant) and V15 (GPTQ) variants.

Computes CLIP, FID for all V1-V15 variants.

Usage: CUDA_VISIBLE_DEVICES=0 python 5k_score_final.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import json
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from cleanfid import fid

BASE = Path("/home/project/Documents/Maharshi/gem/")
OUT = BASE / "benchmark_5k"
REAL_DIR = BASE / "val2017"
PROMPTS = json.load(open(BASE / "benchmark_prompts_5k.json"))

VARIANTS = {
    "v1_fp16":                "V1 — FP16 baseline",
    "v2_int4_naive":          "V2 — Naive INT4",
    "v4_hybrid_best":         "V4 — Hybrid: ternary mid + INT4 naive",
    "v6_fully_distilled":     "V6 — Fully distilled hybrid",
    "v7_timestep_lora":       "V7 — Timestep-aware LoRA hybrid",
    "v8_smoothquant_hybrid":  "V8 — SmoothQuant INT4 + ternary mid",
    "v12_outlier_preserved":  "V12 — Outlier-preserved INT4 + ternary mid",
    "v13_70pct_trajectory":   "V13-70 — Trajectory-aware (70% budget)",
    "v13_55pct_trajectory":   "V13-55 — Trajectory-aware (55% budget)",
    "v13_40pct_trajectory":   "V13-40 — Trajectory-aware (40% budget)",
    "v14_70pct_smoothquant":  "V14-70 — Trajectory + SmoothQuant (70%)",
    "v14_55pct_smoothquant":  "V14-55 — Trajectory + SmoothQuant (55%)",
    "v14_40pct_smoothquant":  "V14-40 — Trajectory + SmoothQuant (40%)",
    "v15_70pct_gptq":         "V15-70 — Trajectory + GPTQ (70%)",
    "v15_55pct_gptq":         "V15-55 — Trajectory + GPTQ (55%)",
    "v15_40pct_gptq":         "V15-40 — Trajectory + GPTQ (40%)",
}

# Memory: V1-V12 known fixed; V13/14/15 share allocation memory (same allocator)
MEMORY = {
    "v1_fp16":                {"size_mb": 4487, "reduction": 0.0},
    "v2_int4_naive":          {"size_mb": 1258, "reduction": 71.9},
    "v4_hybrid_best":         {"size_mb": 1228, "reduction": 72.6},
    "v6_fully_distilled":     {"size_mb": 1228, "reduction": 72.6},
    "v7_timestep_lora":       {"size_mb": 1228, "reduction": 72.6},
    "v8_smoothquant_hybrid":  {"size_mb": 1228, "reduction": 72.6},
    "v12_outlier_preserved":  {"size_mb": 1248, "reduction": 72.2},
}

ALLOC_DIR = BASE / "trajectory" / "phase3_allocator_full"
for pct in [70, 55, 40]:
    alloc_path = ALLOC_DIR / f"allocation_{pct}pct_full.json"
    if alloc_path.exists():
        with open(alloc_path) as f:
            a = json.load(f)
        size_mb = int(a["actual_MB"])
        reduction = (5131.8 - size_mb) / 5131.8 * 100
        # V13, V14, V15 all share the same allocation memory — just different INT4 quality
        for prefix in ["v13", "v14", "v15"]:
            suffix = {"v13": "trajectory", "v14": "smoothquant", "v15": "gptq"}[prefix]
            key = f"{prefix}_{pct}pct_{suffix}"
            MEMORY[key] = {"size_mb": size_mb, "reduction": round(reduction, 1)}

# ── CLIP ──────────────────────────────────────────────────────────────
print("Loading CLIP...")
model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch16", use_safetensors=True
).to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model.eval()

clip_scores = {}
print("Computing CLIP scores...")
for variant, label in VARIANTS.items():
    folder = OUT / variant
    if not folder.exists():
        print(f"  [{label}] folder missing — skipping")
        continue
    n_existing = len([f for f in folder.iterdir() if f.suffix == '.jpg'])
    if n_existing == 0:
        print(f"  [{label}] empty — skipping")
        continue

    scores = []
    for item in PROMPTS:
        img_path = folder / item['filename']
        if not img_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        inputs = processor(
            text=[item['caption']], images=img,
            return_tensors="pt", padding=True, truncation=True, max_length=77,
        ).to("cuda")
        with torch.inference_mode():
            out = model(**inputs)
            img_emb = F.normalize(out.image_embeds, dim=-1)
            txt_emb = F.normalize(out.text_embeds, dim=-1)
            score = (img_emb * txt_emb).sum().item()
        scores.append(score)
    avg = sum(scores) / len(scores) if scores else 0
    clip_scores[variant] = round(avg, 4)
    print(f"  [{label:<55}] CLIP = {avg:.4f}  ({len(scores)} images)")

del model, processor
torch.cuda.empty_cache()

# ── FID ───────────────────────────────────────────────────────────────
print("\nComputing FID...")
fid_scores = {}
for variant, label in VARIANTS.items():
    folder = OUT / variant
    if not folder.exists():
        continue
    n_existing = len([f for f in folder.iterdir() if f.suffix == '.jpg'])
    if n_existing < 100:  # FID needs reasonable sample size
        continue
    try:
        score = fid.compute_fid(str(folder), str(REAL_DIR))
        fid_scores[variant] = round(score, 2)
        print(f"  [{label:<55}] FID = {score:.2f}")
    except Exception as e:
        print(f"  [{label}] FID error: {e}")
        fid_scores[variant] = -1

# ── Final table ───────────────────────────────────────────────────────
print("\n" + "=" * 110)
print(f"{'Variant':<55} {'CLIP':>7} {'FID':>8} {'Size MB':>9} {'Reduction':>10}")
print("=" * 110)
for v, label in VARIANTS.items():
    if v in clip_scores:
        m = MEMORY.get(v, {"size_mb": -1, "reduction": -1})
        print(f"{label:<55} "
              f"{clip_scores[v]:>7.4f} "
              f"{fid_scores.get(v, -1):>8.2f} "
              f"{m['size_mb']:>9} "
              f"{m['reduction']:>9.1f}%")

# ── Save ──────────────────────────────────────────────────────────────
out_json = OUT / "scores_5k_final.json"
with open(out_json, "w") as f:
    json.dump({
        "clip": clip_scores,
        "fid":  fid_scores,
        "memory": MEMORY,
    }, f, indent=2)
print(f"\nSaved: {out_json}")
