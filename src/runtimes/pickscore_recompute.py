"""
Recompute PickScore for ALL variants vs V1 using the correct (unbiased) metric.

Reports:
  - avg P(variant preferred over V1)  — unbiased, primary metric
  - wins/losses/ties counts (with 0.55/0.45 thresholds) — for context

Also computes head-to-head comparisons:
  - V13 vs V19
  - V13 vs V12
  - V19 vs V12

Uses 1000 paired prompts (first 1000) for consistency with prior reports.

Output: pickscore_results_clean.json
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
OUT = BASE / "benchmark_5k"
PROMPTS = json.load(open(BASE / "benchmark_prompts_5k.json"))[:1000]

# All variants we want PickScore vs V1 for
VS_V1_VARIANTS = [
    "v2_int4_naive",
    "v4_hybrid_best",
    "v6_fully_distilled",
    "v7_timestep_lora",
    "v8_smoothquant_hybrid",
    "v12_outlier_preserved",
    "v13_70pct_trajectory",
    "v13_55pct_trajectory",
    "v13_40pct_trajectory",
    "v14_70pct_smoothquant",
    "v15_70pct_gptq",
    "v15_55pct_gptq",
    "v15_40pct_gptq",
    "v16_70pct_random_alloc",
    "v16_55pct_random_alloc",
    "v16_40pct_random_alloc",
    "v17_70pct_no_trajectory",
    "v17_55pct_no_trajectory",
    "v17_40pct_no_trajectory",
    "v18_70pct_v12_quantizer",
    "v18_55pct_v12_quantizer",
    "v18_40pct_v12_quantizer",
    "v19_70pct_trajectory_v12quant",
    "v19_55pct_trajectory_v12quant",
    "v19_40pct_trajectory_v12quant",
]

# Head-to-head comparisons we want
HEAD_TO_HEAD = [
    ("v13_70pct_trajectory", "v19_70pct_trajectory_v12quant"),
    ("v13_55pct_trajectory", "v19_55pct_trajectory_v12quant"),
    ("v13_40pct_trajectory", "v19_40pct_trajectory_v12quant"),
    ("v13_70pct_trajectory", "v12_outlier_preserved"),
    ("v19_70pct_trajectory_v12quant", "v12_outlier_preserved"),
    ("v13_70pct_trajectory", "v17_70pct_no_trajectory"),  # ablation: trajectory effect
    ("v13_70pct_trajectory", "v16_70pct_random_alloc"),    # ablation: allocation effect
]


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


def run_comparison(name_a, name_b, dir_a, dir_b, model, processor, prompts):
    """Compute paired PickScore A vs B."""
    if not (dir_a.exists() and dir_b.exists()):
        return None

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

    if not raw_probs:
        return None

    arr = np.array(raw_probs)
    wins = int(np.sum(arr > 0.55))
    losses = int(np.sum(arr < 0.45))
    ties = int(np.sum((arr >= 0.45) & (arr <= 0.55)))
    n = len(arr)

    return {
        "n": n,
        "avg_prob_a_preferred": float(arr.mean()),
        "std_prob": float(arr.std()),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "tie_rate": ties / n,
    }


def main():
    print("Loading PickScore...")
    processor = AutoProcessor.from_pretrained("yuvalkirstain/PickScore_v1")
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to("cuda").eval()

    results = {
        "vs_v1": {},
        "head_to_head": {},
        "metric_used": "avg_prob_a_preferred",
        "n_prompts": len(PROMPTS),
    }

    # Sanity check first
    print("\n=== SANITY: V1 vs V1 (should be 0.500) ===")
    sanity = run_comparison(
        "v1_fp16", "v1_fp16",
        OUT / "v1_fp16", OUT / "v1_fp16",
        model, processor, PROMPTS[:100]   # only 100 for sanity
    )
    if sanity is None or abs(sanity["avg_prob_a_preferred"] - 0.5) > 0.01:
        print(f"  SANITY FAILED: {sanity}")
        print("  Aborting.")
        return
    print(f"  ✓ V1 vs V1 = {sanity['avg_prob_a_preferred']:.4f}")
    results["sanity_v1_vs_v1"] = sanity

    # === All variants vs V1 ===
    v1_dir = OUT / "v1_fp16"
    print(f"\n=== Variants vs V1 (N={len(PROMPTS)}) ===")
    for variant in VS_V1_VARIANTS:
        var_dir = OUT / variant
        if not var_dir.exists():
            print(f"  [{variant}] folder missing — skipping")
            continue
        n_existing = len([f for f in var_dir.iterdir() if f.suffix == '.jpg'])
        if n_existing < 100:
            print(f"  [{variant}] only {n_existing} images — skipping")
            continue

        t0 = time.time()
        r = run_comparison(variant, "v1_fp16", var_dir, v1_dir,
                            model, processor, PROMPTS)
        if r is None:
            continue
        results["vs_v1"][variant] = r
        elapsed = (time.time() - t0) / 60
        print(f"  [{variant:<35}] avg_P={r['avg_prob_a_preferred']:.3f}  "
              f"wins={r['win_rate']:.2f}  losses={r['loss_rate']:.2f}  "
              f"ties={r['tie_rate']:.2f}  ({r['n']} pairs, {elapsed:.1f}min)", flush=True)

    # === Head-to-head ===
    print(f"\n=== Head-to-head comparisons ===")
    for var_a, var_b in HEAD_TO_HEAD:
        dir_a = OUT / var_a
        dir_b = OUT / var_b
        if not (dir_a.exists() and dir_b.exists()):
            continue
        t0 = time.time()
        r = run_comparison(var_a, var_b, dir_a, dir_b, model, processor, PROMPTS)
        if r is None:
            continue
        key = f"{var_a}_vs_{var_b}"
        results["head_to_head"][key] = r
        elapsed = (time.time() - t0) / 60
        print(f"  [{var_a} vs {var_b}]")
        print(f"    avg_P(A)={r['avg_prob_a_preferred']:.3f}  "
              f"A wins={r['win_rate']:.2f}  B wins={r['loss_rate']:.2f}  "
              f"ties={r['tie_rate']:.2f}  ({elapsed:.1f}min)", flush=True)

    # Save
    out_path = OUT / "pickscore_results_clean.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Variant':<40} {'avg_P(>V1)':>10} {'Wins':>7} {'Losses':>8} {'Ties':>7}")
    print("=" * 90)
    for v, r in results["vs_v1"].items():
        print(f"{v:<40} {r['avg_prob_a_preferred']:>10.3f} "
              f"{100*r['win_rate']:>6.1f}% {100*r['loss_rate']:>7.1f}% "
              f"{100*r['tie_rate']:>6.1f}%")

    print("\n" + "=" * 90)
    print("HEAD-TO-HEAD")
    print("=" * 90)
    for key, r in results["head_to_head"].items():
        var_a, var_b = key.rsplit("_vs_", 1)
        print(f"{var_a:<35} vs {var_b:<25} avg_P(A)={r['avg_prob_a_preferred']:.3f}  "
              f"A:{100*r['win_rate']:.1f}% B:{100*r['loss_rate']:.1f}% T:{100*r['tie_rate']:.1f}%")


if __name__ == "__main__":
    main()
