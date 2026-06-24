import torch
import torch.nn.functional as F
import os
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from cleanfid import fid

BASE     = "/home/project/Documents/Maharshi/gem/"
OUT      = BASE + "benchmark_hybrid/"
REAL_DIR = BASE + "coco_standard_real/"
PROMPTS  = json.load(open(BASE + "benchmark_prompts.json"))

prev = json.load(open(OUT + "final_scores_v11.json"))

VARIANTS = {
    "v1_fp16":               "V1 — FP16 baseline",
    "v2_int4_naive":         "V2 — Naive INT4",
    "v3_ternary_mid_only":   "V3 — Ternary mid + distilled LoRA",
    "v4_hybrid_best":        "V4 — Hybrid: ternary mid + INT4",
    "v5_naive_ternary_int4": "V5 — Naive ternary mid + INT4",
    "v6_fully_distilled":    "V6 — Fully distilled hybrid",
    "v7_timestep_lora":      "V7 — Timestep-aware LoRA hybrid",
    "v8_smoothquant_hybrid": "V8 — SmoothQuant INT4 + ternary mid",
    "v9_smooth_int4_lora":   "V9 — SmoothQuant + distilled LoRA",
    "v10_icec_hybrid":       "V10 — V6 + ICEC correction (α=1.0)",
    "v11_icec_damped":       "V11 — V6 + ICEC damped (α=0.3)",
    "v12_outlier_preserved": "V12 — Outlier-preserved INT4 + ternary mid",
}

MEMORY = {k: {"size_mb": 1228, "reduction": 72.6} for k in VARIANTS}
MEMORY["v1_fp16"]              = {"size_mb": 4487, "reduction": 0.0}
MEMORY["v2_int4_naive"]        = {"size_mb": 1258, "reduction": 71.9}
MEMORY["v3_ternary_mid_only"]  = {"size_mb": 4296, "reduction": 4.3}
MEMORY["v5_naive_ternary_int4"]= {"size_mb": 1243, "reduction": 72.3}
# V12 has slight overhead from storing outliers in FP16
MEMORY["v12_outlier_preserved"]= {"size_mb": 1248, "reduction": 72.2}

print("Loading CLIP...")
model     = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch16").to("cuda")
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch16")
model.eval()

v12_folder = os.path.join(OUT, "v12_outlier_preserved")
scores = []
for item in PROMPTS:
    img_path = os.path.join(v12_folder, item['filename'])
    if not os.path.exists(img_path):
        continue
    img    = Image.open(img_path).convert("RGB")
    inputs = processor(
        text=[item['caption']], images=img,
        return_tensors="pt", padding=True
    ).to("cuda")
    with torch.inference_mode():
        out     = model(**inputs)
        img_emb = F.normalize(out.image_embeds, dim=-1)
        txt_emb = F.normalize(out.text_embeds,  dim=-1)
        score   = (img_emb * txt_emb).sum().item()
    scores.append(score)
v12_clip = sum(scores) / len(scores)
print(f"V12 CLIP: {v12_clip:.4f}")

del model, processor
torch.cuda.empty_cache()

print("\nComputing V12 FID...")
v12_fid = fid.compute_fid(v12_folder, REAL_DIR)
print(f"V12 FID: {v12_fid:.2f}")

clip_scores = prev['clip']
fid_scores  = prev['fid']
clip_scores['v12_outlier_preserved'] = round(v12_clip, 4)
fid_scores['v12_outlier_preserved']  = round(v12_fid,  2)

print("\n" + "=" * 92)
print(f"{'Variant':<46} {'CLIP':>7} {'FID':>8} "
      f"{'Size MB':>9} {'Reduction':>10}")
print("=" * 92)
for v, label in VARIANTS.items():
    print(f"{label:<46} "
          f"{clip_scores.get(v, 0):>7.4f} "
          f"{fid_scores.get(v, -1):>8.2f} "
          f"{MEMORY[v]['size_mb']:>9} "
          f"{MEMORY[v]['reduction']:>9.1f}%")

with open(OUT + "final_scores_v12.json", "w") as f:
    json.dump({"clip": clip_scores, "fid": fid_scores,
               "memory": MEMORY}, f, indent=2)
print("\nSaved.")