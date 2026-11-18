# Save as inspect_v19_vs_v13.py
import os
import json
from pathlib import Path
from PIL import Image
import torch
import numpy as np

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
OUT = BASE / "benchmark_5k"
PROMPTS = json.load(open(BASE / "benchmark_prompts_5k.json"))

V1 = OUT / "v1_fp16"
V13 = OUT / "v13_70pct_trajectory"
V19 = OUT / "v19_70pct_trajectory_v12quant"
V12 = OUT / "v12_outlier_preserved"

# Find prompts where V19 is closer to V1 than V13 — these are V19's "wins"
print("Loading DreamSim model...")
from dreamsim import dreamsim
model, preprocess = dreamsim(pretrained=True, device="cuda")
model.eval()

records = []
print("Computing per-image DreamSim for V13 and V19...")
for i, item in enumerate(PROMPTS[:500]):  # subsample for speed
    fn = item['filename']
    paths = {p: P/fn for p, P in [("v1", V1), ("v13", V13), ("v19", V19)]}
    if not all(p.exists() for p in paths.values()):
        continue
    imgs = {k: preprocess(Image.open(p).convert("RGB")).to("cuda") for k, p in paths.items()}
    with torch.no_grad():
        d13 = model(imgs["v1"], imgs["v13"]).item()
        d19 = model(imgs["v1"], imgs["v19"]).item()
    records.append({"i": i, "fn": fn, "caption": item['caption'][:80],
                    "d13": d13, "d19": d19, "diff": d13 - d19})

# Top wins for V19 (V19 much closer to V1 than V13)
records.sort(key=lambda r: r["diff"], reverse=True)
print("\n=== Top 5 V19 wins (V19 much closer to V1 than V13) ===")
for r in records[:5]:
    print(f"  i={r['i']:4d}  fn={r['fn']}  d13={r['d13']:.3f}  d19={r['d19']:.3f}  Δ={r['diff']:+.3f}")
    print(f"        '{r['caption']}'")

# Top losses (V13 much closer than V19)
print("\n=== Top 5 V13 wins (V13 much closer to V1 than V19) ===")
for r in records[-5:]:
    print(f"  i={r['i']:4d}  fn={r['fn']}  d13={r['d13']:.3f}  d19={r['d19']:.3f}  Δ={r['diff']:+.3f}")
    print(f"        '{r['caption']}'")

# Median pairs (mostly tied — used to verify both are close to V1)
records.sort(key=lambda r: abs(r["diff"]))
print("\n=== Top 5 ties (both V13 and V19 similarly close to V1) ===")
for r in records[:5]:
    print(f"  i={r['i']:4d}  fn={r['fn']}  d13={r['d13']:.3f}  d19={r['d19']:.3f}  Δ={r['diff']:+.3f}")
    print(f"        '{r['caption']}'")

# Save the top-win filenames so you can grab them
top_wins = [r['fn'] for r in records[-5:][::-1]]  # this got sorted by abs(diff), let me redo
records.sort(key=lambda r: r["diff"], reverse=True)
top_v19_wins = [r['fn'] for r in records[:5]]
records.sort(key=lambda r: r["diff"])
top_v13_wins = [r['fn'] for r in records[:5]]
records.sort(key=lambda r: abs(r["diff"]))
top_ties = [r['fn'] for r in records[:5]]

print("\nFiles to inspect:")
print(f"V19 wins: {top_v19_wins}")
print(f"V13 wins: {top_v13_wins}")
print(f"Ties:     {top_ties}")