"""
Enumerate every nn.Linear and nn.Conv2d layer in SDXL UNet with metadata.

Outputs:
    layers_full.json — list of dicts with layer info, sorted by dotted name.

Usage:
    CUDA_VISIBLE_DEVICES=0 python enumerate_layers.py
"""
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

BASE_DIR = Path(os.environ.get("SDXL_DATA_ROOT", "/home/project/Documents/Maharshi/gem"))
OUT_DIR = BASE_DIR / "trajectory" / "phase1_atlas_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "layers_full.json"


def classify_block(name):
    """Return the top-level block group for a dotted layer name."""
    if name.startswith("down_blocks."):
        return f"down_block_{name.split('.')[1]}"
    if name.startswith("up_blocks."):
        return f"up_block_{name.split('.')[1]}"
    if name.startswith("mid_block"):
        return "mid_block"
    if name.startswith("time_emb") or name.startswith("add_embedding"):
        return "time_emb"
    if name.startswith("conv_in") or name.startswith("conv_out"):
        return "stem"
    return "other"


def classify_role(name):
    """Identify what the layer does functionally."""
    n = name.lower()
    if "attn1" in n and ("to_q" in n or "to_k" in n or "to_v" in n):
        return "self_attn_qkv"
    if "attn1" in n and "to_out" in n:
        return "self_attn_out"
    if "attn2" in n and ("to_q" in n or "to_k" in n or "to_v" in n):
        return "cross_attn_qkv"
    if "attn2" in n and "to_out" in n:
        return "cross_attn_out"
    if ".ff.net.0.proj" in n:
        return "ffn_in"
    if ".ff.net.2" in n:
        return "ffn_out"
    if ".proj_in" in n:
        return "transformer_proj_in"
    if ".proj_out" in n:
        return "transformer_proj_out"
    if "resnets." in n and "conv1" in n:
        return "resnet_conv1"
    if "resnets." in n and "conv2" in n:
        return "resnet_conv2"
    if "resnets." in n and "time_emb_proj" in n:
        return "resnet_time_proj"
    if "resnets." in n and "conv_shortcut" in n:
        return "resnet_shortcut"
    if "downsamplers." in n or "upsamplers." in n:
        return "sampler_conv"
    if name.startswith("conv_in"):
        return "stem_conv_in"
    if name.startswith("conv_out"):
        return "stem_conv_out"
    if "time_embedding" in n or "add_embedding" in n:
        return "time_embedding"
    return "other"


def main():
    print("[enumerate] Loading SDXL UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet", torch_dtype=torch.float16,
        use_safetensors=True, variant="fp16",
    )

    records = []
    total_params_linear = 0
    total_params_conv = 0
    n_linear = n_conv = 0

    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            w = module.weight
            records.append({
                "name": name,
                "type": "Linear",
                "n_in": int(w.shape[1]),
                "n_out": int(w.shape[0]),
                "n_params": int(w.numel()),
                "has_bias": module.bias is not None,
                "block": classify_block(name),
                "role": classify_role(name),
            })
            total_params_linear += w.numel()
            n_linear += 1
        elif isinstance(module, nn.Conv2d):
            w = module.weight
            records.append({
                "name": name,
                "type": "Conv2d",
                "n_out": int(w.shape[0]),
                "n_in": int(w.shape[1]),
                "kernel_h": int(w.shape[2]),
                "kernel_w": int(w.shape[3]),
                "n_params": int(w.numel()),
                "has_bias": module.bias is not None,
                "block": classify_block(name),
                "role": classify_role(name),
            })
            total_params_conv += w.numel()
            n_conv += 1

    records.sort(key=lambda r: r["name"])

    # Report
    print(f"\n[enumerate] Found {n_linear} Linear + {n_conv} Conv2d = "
          f"{len(records)} total layers")
    print(f"[enumerate] Linear params: {total_params_linear/1e6:.1f}M")
    print(f"[enumerate] Conv2d params: {total_params_conv/1e6:.1f}M")
    print(f"[enumerate] Total:         {(total_params_linear+total_params_conv)/1e6:.1f}M")

    # Block distribution
    block_counts = {}
    for r in records:
        block_counts[r["block"]] = block_counts.get(r["block"], 0) + 1
    print(f"\n[enumerate] By block:")
    for b in sorted(block_counts.keys()):
        print(f"  {b:<22s}: {block_counts[b]:>4d}")

    # Role distribution
    role_counts = {}
    for r in records:
        role_counts[r["role"]] = role_counts.get(r["role"], 0) + 1
    print(f"\n[enumerate] By role:")
    for r_name in sorted(role_counts.keys()):
        print(f"  {r_name:<22s}: {role_counts[r_name]:>4d}")

    # Save
    out = {
        "n_total": len(records),
        "n_linear": n_linear,
        "n_conv": n_conv,
        "total_params_millions": (total_params_linear + total_params_conv) / 1e6,
        "block_counts": block_counts,
        "role_counts": role_counts,
        "layers": records,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[enumerate] Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
