# Save as score_v19.py and run with python score_v19.py
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import torch, json
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from cleanfid import fid

BASE = Path.home() / "Documents" / "Maharshi" / "gem"
OUT = BASE / "benchmark_5k"
REAL_DIR = BASE / "val2017"
PROMPTS = json.load(open(BASE / "benchmark_prompts_5k.json"))

V19_VARIANTS = [
    "v19_70pct_trajectory_v12quant",
    "v19_55pct_trajectory_v12quant",
    "v19_40pct_trajectory_v12quant",
]

# Load existing cache
cache_path = OUT / "scores_5k_with_ablations.json"
cached = json.load(open(cache_path)) if cache_path.exists() else {"clip": {}, "fid": {}, "memory": {}}

# CLIP
print("Loading CLIP...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", use_safetensors=True).to("cuda").eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

for v in V19_VARIANTS:
    folder = OUT / v
    if not folder.exists() or len(list(folder.iterdir())) < 100:
        print(f"  {v}: skip"); continue
    scores = []
    for item in PROMPTS:
        p = folder / item['filename']
        if not p.exists(): continue
        img = Image.open(p).convert("RGB")
        inputs = processor(text=[item['caption']], images=img, return_tensors="pt",
                           padding=True, truncation=True, max_length=77).to("cuda")
        with torch.inference_mode():
            out = model(**inputs)
            i_emb = F.normalize(out.image_embeds, dim=-1)
            t_emb = F.normalize(out.text_embeds, dim=-1)
            scores.append((i_emb * t_emb).sum().item())
    avg = sum(scores) / len(scores)
    cached["clip"][v] = round(avg, 4)
    print(f"  {v}: CLIP = {avg:.4f}")

del model, processor
torch.cuda.empty_cache()

# FID
print("Computing FID...")
for v in V19_VARIANTS:
    folder = OUT / v
    if not folder.exists() or len(list(folder.iterdir())) < 100: continue
    score = fid.compute_fid(str(folder), str(REAL_DIR))
    cached["fid"][v] = round(score, 2)
    print(f"  {v}: FID = {score:.2f}")

# Add memory mapping
ALLOC_DIR = BASE / "trajectory" / "phase3_allocator_full"
for pct in [70, 55, 40]:
    a = json.load(open(ALLOC_DIR / f"allocation_{pct}pct_full.json"))
    size_mb = int(a["actual_MB"])
    cached["memory"][f"v19_{pct}pct_trajectory_v12quant"] = {
        "size_mb": size_mb, "reduction": round((5131.8 - size_mb) / 5131.8 * 100, 1)
    }

# Save
json.dump(cached, open(cache_path, "w"), indent=2)
print(f"Updated: {cache_path}")
print()
for v in V19_VARIANTS:
    print(f"  {v}: CLIP={cached['clip'].get(v, '?')}, FID={cached['fid'].get(v, '?')}")