"""
V24a: INT8 base + NF4 refiner two-pass.

Phase 1 (steps 0-4):  All Linear layers as Linear8bitLt (INT8). Conv2d stays FP16.
Phase 2 (steps 5-19): All Linear layers as Linear4bit (NF4). Conv2d stays FP16.

NO FP16 Linear anywhere. Real peak VRAM saving:
  - INT8 UNet weights: ~2.5 GB
  - NF4 UNet weights: ~1.7 GB
  - Conv2d FP16: ~1 GB (in each)
  - Both UNets held on GPU simultaneously: ~9 GB total weights
  vs V21's Phase 1 FP16 (~5 GB) + V21's Phase 2 NF4 weights (~1.7 GB) = 6.7 GB but ALSO the
     FP16 weight. V21 holds 5 GB FP16 + 1.7 GB NF4 = 6.7 GB.

Wait, V24 and V21 weight memory is similar. The key difference: V24's Phase 1 runs INT8 forward
(which uses ~2.5 GB activated) vs V21's FP16 forward (which uses ~5 GB activated).

Either way, let's measure and see.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u v24a_int8_nf4_two_pass.py \
        --t-split 5 --n-images 5 --measure-vram
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
OUT = BASE / "benchmark_5k"

NUM_STEPS = 20
GUIDANCE_SCALE = 7.5
HEIGHT = 768
WIDTH = 768
SEED = 42


def replace_all_linear_with_int8(unet):
    """Replace ALL Linear layers in unet with Linear8bitLt."""
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear8bitLt

    layers = []
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            layers.append((name, module))

    n_replaced = 0
    for name, module in layers:
        parts = name.split(".")
        parent_name = ".".join(parts[:-1])
        attr_name = parts[-1]
        parent = unet.get_submodule(parent_name) if parent_name else unet

        new_module = Linear8bitLt(
            module.in_features, module.out_features,
            bias=(module.bias is not None),
            has_fp16_weights=False,
            threshold=6.0,
        )
        new_module.weight = bnb.nn.Int8Params(
            module.weight.data.clone(),
            has_fp16_weights=False,
            requires_grad=False,
        )
        if module.bias is not None:
            new_module.bias = nn.Parameter(module.bias.data.clone(), requires_grad=False)
        new_module = new_module.to(module.weight.device)
        setattr(parent, attr_name, new_module)
        n_replaced += 1
    return n_replaced


def replace_all_linear_with_nf4(unet):
    """Replace ALL Linear layers in unet with Linear4bit(NF4)."""
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Params4bit

    layers = []
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            layers.append((name, module))

    n_replaced = 0
    for name, module in layers:
        parts = name.split(".")
        parent_name = ".".join(parts[:-1])
        attr_name = parts[-1]
        parent = unet.get_submodule(parent_name) if parent_name else unet

        new_module = Linear4bit(
            module.in_features, module.out_features,
            bias=(module.bias is not None),
            quant_type="nf4",
            compute_dtype=torch.float16,
        )
        new_module.weight = Params4bit(
            module.weight.data.clone().contiguous(),
            requires_grad=False, quant_type="nf4"
        )
        if module.bias is not None:
            new_module.bias = nn.Parameter(module.bias.data.clone(), requires_grad=False)
        new_module = new_module.to(module.weight.device)
        setattr(parent, attr_name, new_module)
        n_replaced += 1
    return n_replaced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-split", type=int, default=5)
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--measure-vram", action="store_true")
    args = ap.parse_args()

    high_noise_frac = args.t_split / NUM_STEPS

    print(f"=== V24a: INT8 base + NF4 refiner, t_split={args.t_split} ===")
    print(f"  high_noise_frac: {high_noise_frac}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # === Load FP16 base, prepare INT8 UNet ===
    print("\nLoading FP16 base pipeline (will become INT8)...")
    base = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    base.scheduler = EulerDiscreteScheduler.from_config(base.scheduler.config)
    base.set_progress_bar_config(disable=True)

    print("  Replacing UNet Linear layers with Linear8bitLt...")
    n_int8 = replace_all_linear_with_int8(base.unet)
    gc.collect()
    torch.cuda.empty_cache()
    after_int8_vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Replaced {n_int8} Linear → Linear8bitLt")
    print(f"  After INT8 setup. VRAM: {after_int8_vram:.2f} GB")

    # === Load second pipeline, convert to NF4 UNet ===
    print("\nLoading second pipeline (will become NF4)...")
    pipe_for_nf4 = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    unet_nf4 = pipe_for_nf4.unet
    n_nf4 = replace_all_linear_with_nf4(unet_nf4)
    gc.collect()
    torch.cuda.empty_cache()
    after_nf4_vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Replaced {n_nf4} Linear → Linear4bit(NF4)")
    print(f"  After NF4 setup. VRAM: {after_nf4_vram:.2f} GB "
          f"(delta from INT8 stage: {after_nf4_vram - after_int8_vram:+.2f} GB)")

    # === Build the refiner pipeline (Img2Img) sharing components with base ===
    print("\nBuilding refiner pipeline (NF4 UNet + base's text encoders + VAE)...")
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

    # release the second pipeline's other components
    del pipe_for_nf4
    gc.collect()
    torch.cuda.empty_cache()

    final_setup_vram = torch.cuda.memory_allocated() / 1e9
    print(f"  After full setup. VRAM: {final_setup_vram:.2f} GB")

    # === Generate ===
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)[:args.n_images]
    print(f"\nGenerating {len(prompts)} images...")

    variant_name = f"v24a_int8_nf4_t{args.t_split}"
    if not args.measure_vram:
        folder = OUT / variant_name
        folder.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    peaks = []
    timings = []
    t_overall = time.time()

    for i, item in enumerate(prompts):
        if not args.measure_vram:
            out_path = folder / item['filename']
            if out_path.exists():
                continue
        prompt = item['caption']
        gen = torch.Generator("cuda").manual_seed(SEED + i)

        t_img = time.time()

        # Phase 1: INT8 base, denoising_end=high_noise_frac
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

        # Phase 2: NF4 refiner, denoising_start=high_noise_frac
        with torch.inference_mode():
            image = refiner(
                prompt=prompt,
                num_inference_steps=NUM_STEPS,
                denoising_start=high_noise_frac,
                guidance_scale=GUIDANCE_SCALE,
                image=latent,
                generator=gen,
            ).images[0]

        if not args.measure_vram:
            image.save(out_path)
        timings.append(time.time() - t_img)
        peaks.append(torch.cuda.max_memory_allocated() / 1e9)
        torch.cuda.reset_peak_memory_stats()

        if (i + 1) % 25 == 0 or i < 5:
            elapsed = (time.time() - t_overall) / 60
            avg_t = sum(timings) / len(timings)
            avg_peak = sum(peaks) / len(peaks)
            max_peak = max(peaks)
            print(f"  [{i+1}/{len(prompts)}] {elapsed:.1f}min  "
                  f"avg t/img: {avg_t:.2f}s  "
                  f"avg peak: {avg_peak:.2f} GB  max peak: {max_peak:.2f} GB",
                  flush=True)

    elapsed = time.time() - t_overall
    avg_peak = sum(peaks) / len(peaks) if peaks else 0
    max_peak = max(peaks) if peaks else 0
    avg_time = sum(timings) / len(timings) if timings else 0

    print(f"\n=== Result ===")
    print(f"  Variant:                {variant_name}")
    print(f"  N images:               {len(timings)}")
    print(f"  Setup VRAM:             {final_setup_vram:.2f} GB")
    print(f"  Average peak VRAM:      {avg_peak:.2f} GB")
    print(f"  Max peak VRAM:          {max_peak:.2f} GB")
    print(f"  Average time per image: {avg_time:.2f} s")
    print(f"  Total time:             {elapsed/60:.1f} min")

    out_path = BASE / f"vram_measurement_{variant_name}.json"
    with open(out_path, "w") as f:
        json.dump({
            "variant": variant_name,
            "t_split": args.t_split,
            "n_images": len(timings),
            "setup_vram_gb": final_setup_vram,
            "avg_peak_vram_gb": avg_peak,
            "max_peak_vram_gb": max_peak,
            "avg_time_per_image_s": avg_time,
            "total_time_min": elapsed / 60,
            "n_int8_layers": n_int8,
            "n_nf4_layers": n_nf4,
            "gpu_name": torch.cuda.get_device_name(0),
        }, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
