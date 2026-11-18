"""
Sanity check PickScore: V1 vs V1 should give win-rate exactly 0.5.

If it doesn't, the script has a bug.

Also run V13 vs V19 head-to-head and V12 vs V13 for the comparisons we want.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -u pickscore_sanity.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModel

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
OUT = BASE / "benchmark_5k"
PROMPTS = json.load(open(BASE / "benchmark_prompts_5k.json"))[:200]   # 200 for speed


def unwrap_image_features(out):
    if torch.is_tensor(out): return out
    if hasattr(out, "image_embeds") and out.image_embeds is not None: return out.image_embeds
    if hasattr(out, "pooler_output") and out.pooler_output is not None: return out.pooler_output
    raise RuntimeError("Could not unwrap image features")


def unwrap_text_features(out):
    if torch.is_tensor(out): return out
    if hasattr(out, "text_embeds") and out.text_embeds is not None: return out.text_embeds
    if hasattr(out, "pooler_output") and out.pooler_output is not None: return out.pooler_output
    raise RuntimeError("Could not unwrap text features")


def pickscore_compare(model, processor, prompt, img_a, img_b):
    """Returns probability that img_a is preferred over img_b."""
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


def run_comparison(name, dir_a, dir_b, model, processor, prompts):
    """Compare images in dir_a vs dir_b for matching filenames.
    Returns: win_rate of A over B, and average raw probability."""
    print(f"\n=== {name} ===")
    print(f"  A: {dir_a.name}")
    print(f"  B: {dir_b.name}")
    wins = 0
    losses = 0
    ties = 0
    n = 0
    raw_probs = []
    for item in prompts:
        p_a = dir_a / item['filename']
        p_b = dir_b / item['filename']
        if not (p_a.exists() and p_b.exists()):
            continue
        img_a = Image.open(p_a).convert("RGB")
        img_b = Image.open(p_b).convert("RGB")
        prob_a = pickscore_compare(model, processor, item['caption'], img_a, img_b)
        raw_probs.append(prob_a)
        n += 1
        if prob_a > 0.55:
            wins += 1
        elif prob_a < 0.45:
            losses += 1
        else:
            ties += 1
    avg_prob = sum(raw_probs) / len(raw_probs) if raw_probs else 0
    win_rate = wins / n if n > 0 else 0
    print(f"  N: {n}")
    print(f"  Avg P(A preferred): {avg_prob:.3f}")
    print(f"  Wins (P>0.55): {wins}  ({100*wins/n:.1f}%)")
    print(f"  Losses (P<0.45): {losses}  ({100*losses/n:.1f}%)")
    print(f"  Ties (0.45-0.55): {ties}  ({100*ties/n:.1f}%)")
    return win_rate, avg_prob


def main():
    print("Loading PickScore...")
    processor = AutoProcessor.from_pretrained("yuvalkirstain/PickScore_v1")
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to("cuda").eval()

    # Sanity: V1 vs V1 — should be exactly 0.5
    run_comparison(
        "SANITY: V1 vs V1 (should be exactly 0.5)",
        OUT / "v1_fp16", OUT / "v1_fp16",
        model, processor, PROMPTS
    )

    # V13 vs V1 — original measurement said 0.565
    run_comparison(
        "V13-70 vs V1",
        OUT / "v13_70pct_trajectory", OUT / "v1_fp16",
        model, processor, PROMPTS
    )

    # V19 vs V1
    run_comparison(
        "V19-70 vs V1",
        OUT / "v19_70pct_trajectory_v12quant", OUT / "v1_fp16",
        model, processor, PROMPTS
    )

    # The actually interesting head-to-head
    run_comparison(
        "V19-70 vs V13-70 (head-to-head)",
        OUT / "v19_70pct_trajectory_v12quant", OUT / "v13_70pct_trajectory",
        model, processor, PROMPTS
    )

    # Compare to V12 (prior best)
    run_comparison(
        "V13-70 vs V12 (framework vs prior best)",
        OUT / "v13_70pct_trajectory", OUT / "v12_outlier_preserved",
        model, processor, PROMPTS
    )

    run_comparison(
        "V19-70 vs V12",
        OUT / "v19_70pct_trajectory_v12quant", OUT / "v12_outlier_preserved",
        model, processor, PROMPTS
    )


if __name__ == "__main__":
    main()
