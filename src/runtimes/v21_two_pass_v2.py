"""
V21 Two-pass v2: uses diffusers' native denoising_end/denoising_start.

This is the "Mixture of Denoisers" pattern that SDXL uses natively for
base+refiner workflows. We use it with the same model at two precisions:
  - Phase 1: FP16 UNet via StableDiffusionXLPipeline with denoising_end=t_split/N
  - Phase 2: NF4 UNet via StableDiffusionXLImg2ImgPipeline with denoising_start=t_split/N

Why this version: the previous custom denoising loop produced broken images
(DreamSim 0.94 — pure noise). The pipelines handle VAE upcasting, scheduler
state, and image post-processing correctly.

Implementation: hold both UNets, swap by reassigning pipe.unet between phases.
For VRAM realism, we also support CPU-GPU swap of UNets between phases.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u v21_two_pass_v2.py \
        --budget 70 --t-split 10 --n-images 5 --no-cpu-swap   # quick test
    CUDA_VISIBLE_DEVICES=2 python -u v21_two_pass_v2.py \
        --budget 70 --t-split 10 --n-images 200               # with CPU swap
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import gc
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerDiscreteScheduler,
)

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
    return fp16_layers, int4_layers


def replace_linear_with_nf4(unet, int4_layer_names):
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Params4bit

    n_replaced = 0
    n_skipped_conv = 0
    layer_paths = {name: None for name in int4_layer_names}
    for name, module in unet.named_modules():
        if name in layer_paths:
            layer_paths[name] = module

    for name in int4_layer_names:
        module = layer_paths.get(name)
        if module is None:
            continue
        if isinstance(module, nn.Conv2d):
            n_skipped_conv += 1
            continue
        if not isinstance(module, nn.Linear):
            continue

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
        new_module.weight = Params4bit(
            weight_data, requires_grad=False, quant_type="nf4"
        )
        if bias:
            new_module.bias = nn.Parameter(bias_data, requires_grad=False)
        new_module = new_module.to(device)

        setattr(parent, attr_name, new_module)
        n_replaced += 1

    return n_replaced, n_skipped_conv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=70)
    ap.add_argument("--t-split", type=int, default=10)
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--no-cpu-swap", action="store_true",
                    help="Hold both UNets on GPU (faster, more VRAM)")
    args = ap.parse_args()

    cpu_swap = not args.no_cpu_swap
    high_noise_frac = args.t_split / NUM_STEPS

    print(f"=== V21 Two-Pass v2 at {args.budget}% budget, t_split={args.t_split} ===")
    print(f"  high_noise_frac (denoising split): {high_noise_frac}")
    print(f"  CPU swap between phases: {cpu_swap}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # === Load pipelines ===
    print("\nLoading FP16 base pipeline...")
    base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    base.scheduler = EulerDiscreteScheduler.from_config(base.scheduler.config)
    base.set_progress_bar_config(disable=True)
    base_unet_fp16 = base.unet
    print(f"  Base loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("\nLoading second pipeline for NF4 conversion...")
    pipe_for_nf4 = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    unet_nf4 = pipe_for_nf4.unet

    fp16_layers, int4_layers = load_v13_allocation(args.budget)
    n_nf4, n_conv = replace_linear_with_nf4(unet_nf4, int4_layers)
    print(f"  NF4 UNet built: {n_nf4} Linear→Linear4bit, {n_conv} Conv2d skipped")

    # Now build the Img2Img refiner pipeline that REUSES the base's text encoders + VAE
    # but uses the NF4 UNet
    print("\nBuilding refiner pipeline (Img2Img with NF4 UNet, sharing components)...")
    refiner = StableDiffusionXLImg2ImgPipeline(
        vae=base.vae,
        text_encoder=base.text_encoder,
        text_encoder_2=base.text_encoder_2,
        tokenizer=base.tokenizer,
        tokenizer_2=base.tokenizer_2,
        unet=unet_nf4,
        scheduler=base.scheduler,
    ).to("cuda")
    refiner.set_progress_bar_config(disable=True)

    # release the second pipeline's other components (we only kept its unet)
    del pipe_for_nf4
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  Refiner ready. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # If CPU swap, move NF4 UNet to CPU initially
    if cpu_swap:
        unet_nf4.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  NF4 UNet moved to CPU. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # === Generate ===
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)[:args.n_images]
    print(f"\nGenerating {len(prompts)} images...")

    variant_name = f"v21_{args.budget}pct_two_pass_t{args.t_split}_v2"
    folder = OUT / variant_name
    folder.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    peaks = []
    timings = []
    t_overall = time.time()

    for i, item in enumerate(prompts):
        out_path = folder / item['filename']
        if out_path.exists():
            continue
        prompt = item['caption']
        gen = torch.Generator("cuda").manual_seed(SEED + i)

        t_img = time.time()

        # === Phase 1: FP16 base, denoising_end=high_noise_frac ===
        if cpu_swap:
            base_unet_fp16.to("cuda")
            base.unet = base_unet_fp16  # re-attach
            torch.cuda.empty_cache()

        with torch.inference_mode():
            latent = base(
                prompt=prompt,
                num_inference_steps=NUM_STEPS,
                denoising_end=high_noise_frac,
                guidance_scale=GUIDANCE_SCALE,
                generator=gen,
                height=HEIGHT, width=WIDTH,
                output_type="latent",
            ).images

        if cpu_swap:
            base_unet_fp16.to("cpu")
            unet_nf4.to("cuda")
            refiner.unet = unet_nf4
            gc.collect()
            torch.cuda.empty_cache()

        # === Phase 2: NF4 refiner, denoising_start=high_noise_frac ===
        with torch.inference_mode():
            image = refiner(
                prompt=prompt,
                num_inference_steps=NUM_STEPS,
                denoising_start=high_noise_frac,
                guidance_scale=GUIDANCE_SCALE,
                image=latent,
                generator=gen,
            ).images[0]

        if cpu_swap:
            unet_nf4.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        image.save(out_path)
        timings.append(time.time() - t_img)
        peaks.append(torch.cuda.max_memory_allocated() / 1e9)
        torch.cuda.reset_peak_memory_stats()

        if (i + 1) % 10 == 0 or i < 3:
            elapsed = (time.time() - t_overall) / 60
            avg_t = sum(timings) / len(timings) if timings else 0
            eta = (len(prompts) - i - 1) * avg_t / 60
            avg_peak = sum(peaks) / len(peaks) if peaks else 0
            max_peak = max(peaks) if peaks else 0
            print(f"  [{i+1}/{len(prompts)}] {elapsed:.1f}min  "
                  f"avg t/img: {avg_t:.1f}s  "
                  f"avg peak: {avg_peak:.2f} GB  max peak: {max_peak:.2f} GB  "
                  f"ETA: {eta:.1f}min", flush=True)

    elapsed = time.time() - t_overall
    avg_peak = sum(peaks) / len(peaks) if peaks else 0
    max_peak = max(peaks) if peaks else 0
    avg_time = sum(timings) / len(timings) if timings else 0

    print(f"\n=== Result ===")
    print(f"  Variant:                {variant_name}")
    print(f"  N images:               {len(timings)}")
    print(f"  Average peak VRAM:      {avg_peak:.2f} GB")
    print(f"  Max peak VRAM:          {max_peak:.2f} GB")
    print(f"  Average time per image: {avg_time:.2f} s")
    print(f"  Total time:             {elapsed/60:.1f} min")

    out_path = BASE / f"vram_measurement_{variant_name}.json"
    with open(out_path, "w") as f:
        json.dump({
            "variant": variant_name,
            "budget_pct": args.budget,
            "t_split": args.t_split,
            "high_noise_frac": high_noise_frac,
            "cpu_swap": cpu_swap,
            "n_images": len(timings),
            "avg_peak_vram_gb": avg_peak,
            "max_peak_vram_gb": max_peak,
            "avg_time_per_image_s": avg_time,
            "total_time_min": elapsed / 60,
            "n_linear_nf4": n_nf4,
            "n_conv2d_skipped_fp16": n_conv,
            "gpu_name": torch.cuda.get_device_name(0),
        }, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
