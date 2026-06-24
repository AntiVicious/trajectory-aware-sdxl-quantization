"""
V20-style: static NF4 quantization via bitsandbytes at V19's allocation.

Replaces each Linear layer scheduled for INT4 (in V19's allocation) with
Linear4bit(NF4). Conv2d layers cannot be NF4-quantized via bitsandbytes,
so they stay FP16. The 75 layers V13's allocator marked as FP16 also stay FP16.

This is "V19 without the trajectory dimension" — quality will be in V17/V18
range (DreamSim ~0.20), but the VRAM savings will be real.

Usage:
    # Sanity test (5 images, measure VRAM)
    CUDA_VISIBLE_DEVICES=1 python -u v20_static_nf4.py --budget 70 --n-images 5 --measure-vram

    # Full benchmark (5000 images for quality measurement)
    CUDA_VISIBLE_DEVICES=1 python -u v20_static_nf4.py --budget 70 --n-images 5000
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

BASE = Path.home() / "Documents" / "Maharshi" / "gem"
PROMPTS_PATH = BASE / "benchmark_prompts_5k.json"
ALLOC_DIR = BASE / "trajectory" / "phase3_allocator_full"
OUT = BASE / "benchmark_5k"

NUM_STEPS = 20
GUIDANCE_SCALE = 7.5
HEIGHT = 768
WIDTH = 768
SEED = 42


def load_v13_allocation(pct):
    """V19 uses V13's allocation."""
    p = ALLOC_DIR / f"allocation_{pct}pct_full.json"
    with open(p) as f:
        alloc = json.load(f)
    fp16_layers = set()
    int4_layers = set()
    for a in alloc["allocations"]:
        if a["choice"] == "FP16":
            fp16_layers.add(a["layer"])
        else:
            int4_layers.add(a["layer"])
    return fp16_layers, int4_layers, alloc


def replace_linear_with_nf4(unet, int4_layer_names):
    """For each Linear layer in int4_layer_names, replace it with Linear4bit(NF4).
    Returns: dict of (n_replaced_linear, n_skipped_conv2d, n_other)."""
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Params4bit

    n_replaced = 0
    n_skipped_conv = 0
    n_skipped_other = 0
    n_not_found = 0

    # We need to walk the module tree and replace by reference
    # Strategy: find each (parent_module, attr_name) for each target layer
    layer_paths = {name: None for name in int4_layer_names}
    for name, module in unet.named_modules():
        if name in layer_paths:
            layer_paths[name] = module

    for name in int4_layer_names:
        module = layer_paths.get(name)
        if module is None:
            n_not_found += 1
            continue

        if isinstance(module, nn.Conv2d):
            n_skipped_conv += 1
            continue

        if not isinstance(module, nn.Linear):
            n_skipped_other += 1
            continue

        # Replace this Linear with a Linear4bit
        # Find parent module
        parts = name.split(".")
        parent_name = ".".join(parts[:-1])
        attr_name = parts[-1]
        parent = unet.get_submodule(parent_name) if parent_name else unet

        in_features = module.in_features
        out_features = module.out_features
        bias = module.bias is not None
        device = module.weight.device
        weight_data = module.weight.data.clone().contiguous()
        bias_data = module.bias.data.clone() if bias else None

        new_module = Linear4bit(
            in_features, out_features, bias=bias,
            quant_type="nf4",
            compute_dtype=torch.float16,
        )
        # Manually place quantized weight
        new_module.weight = Params4bit(
            weight_data, requires_grad=False, quant_type="nf4"
        )
        if bias:
            new_module.bias = nn.Parameter(bias_data, requires_grad=False)

        new_module = new_module.to(device)

        setattr(parent, attr_name, new_module)
        n_replaced += 1

    print(f"  [NF4 replacement] Linear→Linear4bit: {n_replaced}  "
          f"Conv2d skipped: {n_skipped_conv}  "
          f"Other skipped: {n_skipped_other}  "
          f"Not found: {n_not_found}", flush=True)
    return n_replaced, n_skipped_conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=70, choices=[95, 85, 70, 55, 40])
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--measure-vram", action="store_true",
                    help="Skip image saving, just generate and measure VRAM")
    args = ap.parse_args()

    print(f"=== V20-NF4 at {args.budget}% budget ===")

    # Reset peak
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Load pipeline (FP16)
    print("Loading SDXL pipeline (FP16)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    pipeline_load_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Pipeline loaded. VRAM: {pipeline_load_gb:.2f} GB")

    # Load V19's allocation (which is V13's allocation)
    fp16_layers, int4_layers, alloc = load_v13_allocation(args.budget)
    print(f"  Allocation: {len(fp16_layers)} FP16, {len(int4_layers)} INT4")

    # Replace INT4 Linear layers with Linear4bit(NF4)
    print("  Replacing Linear layers with Linear4bit(NF4)...")
    n_nf4, n_conv = replace_linear_with_nf4(pipe.unet, int4_layers)

    # GC and clear cache to actually free FP16 weights
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    after_quant_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  After NF4 quantization. VRAM: {after_quant_gb:.2f} GB "
          f"(delta: {after_quant_gb - pipeline_load_gb:+.2f} GB)")

    # Load prompts
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)[:args.n_images]
    print(f"  Generating {len(prompts)} images...")

    # Output folder (only if not just measuring VRAM)
    variant_name = f"v20_{args.budget}pct_nf4_static"
    if not args.measure_vram:
        folder = OUT / variant_name
        folder.mkdir(parents=True, exist_ok=True)

    # Generate
    t0 = time.time()
    for i, item in enumerate(prompts):
        gen = torch.Generator("cuda").manual_seed(SEED + i)
        with torch.inference_mode():
            image = pipe(
                item['caption'],
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=gen, height=HEIGHT, width=WIDTH,
            ).images[0]
        if not args.measure_vram:
            image.save(folder / item['filename'])
        if (i + 1) % 100 == 0 or i < 5:
            elapsed = (time.time() - t0) / 60
            eta = (len(prompts) - i - 1) / max((i + 1) / elapsed, 1e-9)
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"  [{i+1}/{len(prompts)}] {elapsed:.2f}min  "
                  f"peak VRAM: {peak:.2f} GB  ETA {eta:.1f}min", flush=True)

    elapsed = time.time() - t0
    final_peak = torch.cuda.max_memory_allocated() / 1e9
    final_alloc = torch.cuda.memory_allocated() / 1e9

    print(f"\n=== Result ===")
    print(f"  Variant:               {variant_name}")
    print(f"  Pipeline load VRAM:    {pipeline_load_gb:.2f} GB")
    print(f"  After NF4 VRAM:        {after_quant_gb:.2f} GB")
    print(f"  Inference peak VRAM:   {final_peak:.2f} GB")
    print(f"  Final allocated:       {final_alloc:.2f} GB")
    print(f"  Time per image:        {elapsed/len(prompts):.2f} s")
    print(f"  N Linear quantized:    {n_nf4}")
    print(f"  N Conv2d skipped:      {n_conv}")

    # Save measurement
    out_path = BASE / f"vram_measurement_{variant_name}.json"
    with open(out_path, "w") as f:
        json.dump({
            "variant": variant_name,
            "budget_pct": args.budget,
            "n_images": len(prompts),
            "pipeline_load_gb": pipeline_load_gb,
            "after_quant_gb": after_quant_gb,
            "inference_peak_gb": final_peak,
            "final_alloc_gb": final_alloc,
            "time_per_image_s": elapsed / len(prompts),
            "n_linear_nf4": n_nf4,
            "n_conv2d_skipped_fp16": n_conv,
            "gpu_name": torch.cuda.get_device_name(0),
        }, f, indent=2)
    print(f"  Saved measurement: {out_path}")


if __name__ == "__main__":
    main()
