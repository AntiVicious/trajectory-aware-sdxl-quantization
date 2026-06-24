"""
5k scoring including V14 (SmoothQuant), V15 (GPTQ), and V16/V17/V18 ablations.

Computes CLIP and FID for all variants present in benchmark_5k/.
DreamSim and PickScore are paired metrics — must be computed on the machine
that has both V1 and the variant images.

Usage: CUDA_VISIBLE_DEVICES=1 python -u 5k_score_final.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn.functional as F
import json
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from cleanfid import fid

BASE = Path.home() / "Documents" / "Maharshi" / "gem"
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
    "v16_70pct_random_alloc":  "V16-70 — Random alloc, naive INT4 from t=0 (70%)",
    "v16_55pct_random_alloc":  "V16-55 — Random alloc, naive INT4 from t=0 (55%)",
    "v16_40pct_random_alloc":  "V16-40 — Random alloc, naive INT4 from t=0 (40%)",
    "v17_70pct_no_trajectory": "V17-70 — V13 alloc, no trajectory (70%)",
    "v17_55pct_no_trajectory": "V17-55 — V13 alloc, no trajectory (55%)",
    "v17_40pct_no_trajectory": "V17-40 — V13 alloc, no trajectory (40%)",
    "v18_70pct_v12_quantizer": "V18-70 — V13 alloc, V12 quantizer (70%)",
    "v18_55pct_v12_quantizer": "V18-55 — V13 alloc, V12 quantizer (55%)",
    "v18_40pct_v12_quantizer": "V18-40 — V13 alloc, V12 quantizer (40%)",
    "v19_70pct_trajectory_v12quant": "V19-70 — V13 trajectory + V12 quantizer (70%)",
    "v19_55pct_trajectory_v12quant": "V19-55 — V13 trajectory + V12 quantizer (55%)",
    "v19_40pct_trajectory_v12quant": "V19-40 — V13 trajectory + V12 quantizer (40%)",
}

# Memory mapping (V13/V14/V15/V17/V18 share allocations because same allocator;
# V16 has different memory because random allocation)
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
        # V13/V14/V15 use full trajectory model. V17/V18 use same allocation but
        # commit at t=0, so memory is slightly less (but we report same as V13
        # for cleanest comparison since allocation budget was the constraint).
        for prefix, suffix in [("v13", "trajectory"), ("v14", "smoothquant"),
                                ("v15", "gptq"), ("v17", "no_trajectory"),
                                ("v18", "v12_quantizer"), ("v16", "random_alloc"), ("v19", "trajectory_v12quant")]:
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
        continue
    n_existing = len([f for f in folder.iterdir() if f.suffix == '.jpg'])
    if n_existing == 0:
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
    print(f"  [{label:<60}] CLIP = {avg:.4f}  ({len(scores)} images)")

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
    if n_existing < 100:
        continue
    try:
        score = fid.compute_fid(str(folder), str(REAL_DIR))
        fid_scores[variant] = round(score, 2)
        print(f"  [{label:<60}] FID = {score:.2f}")
    except Exception as e:
        print(f"  [{label}] FID error: {e}")
        fid_scores[variant] = -1

# ── Final table ───────────────────────────────────────────────────────
print("\n" + "=" * 120)
print(f"{'Variant':<60} {'CLIP':>7} {'FID':>8} {'Size MB':>9} {'Reduction':>10}")
print("=" * 120)
for v, label in VARIANTS.items():
    if v in clip_scores:
        m = MEMORY.get(v, {"size_mb": -1, "reduction": -1})
        print(f"{label:<60} "
              f"{clip_scores[v]:>7.4f} "
              f"{fid_scores.get(v, -1):>8.2f} "
              f"{m['size_mb']:>9} "
              f"{m['reduction']:>9.1f}%")

# ── Save ──────────────────────────────────────────────────────────────
out_json = OUT / "scores_5k_with_ablations.json"
with open(out_json, "w") as f:
    json.dump({
        "clip": clip_scores,
        "fid":  fid_scores,
        "memory": MEMORY,
    }, f, indent=2)
print(f"\nSaved: {out_json}")