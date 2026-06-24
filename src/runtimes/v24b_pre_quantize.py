"""
V24b: pre-quantize UNet at load time using diffusers' BitsAndBytesConfig.

Unlike v24a which post-hoc replaced Linear modules (and held FP16 briefly during
the conversion), this loads the UNet directly in quantized form via
BitsAndBytesConfig — bnb's loading machinery quantizes weights as they're
streamed from disk. FP16 is never materialized on GPU.

Two configurations tested:
  --mode int8: full UNet INT8 (single pass, no trajectory)
  --mode int8_nf4: two-pass — INT8 base + NF4 refiner (trajectory)
  --mode nf4: full UNet NF4 (single pass) — equivalent to V20 but via proper loading

Usage:
    # Just INT8, single pass (simplest test of "is INT8 alone good enough")
    CUDA_VISIBLE_DEVICES=2 python -u v24b_pre_quantize.py \
        --mode int8 --n-images 5 --measure-vram

    # INT8 + NF4 two-pass (the actual idea)
    CUDA_VISIBLE_DEVICES=2 python -u v24b_pre_quantize.py \
        --mode int8_nf4 --t-split 5 --n-images 5 --measure-vram
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import gc
import json
import time
from pathlib import Path
import torch

# diffusers' own BitsAndBytesConfig (NOT transformers')
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
)
from diffusers import BitsAndBytesConfig as DiffusersBnBConfig

BASE = Path.home() / "Documents" / "Maharshi" / "gem"
PROMPTS_PATH = BASE / "benchmark_prompts_5k.json"
OUT = BASE / "benchmark_5k"

NUM_STEPS = 20
GUIDANCE_SCALE = 7.5
HEIGHT = 768
WIDTH = 768
SEED = 42
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def load_unet_quantized(quant_type):
    """Load SDXL UNet with bnb quantization at load time. Never materializes FP16.

    quant_type: 'int8', 'nf4', or 'fp4'
    """
    print(f"  Loading UNet with {quant_type} quantization...")
    if quant_type == "int8":
        cfg = DiffusersBnBConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    elif quant_type == "nf4":
        cfg = DiffusersBnBConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quant_type == "fp4":
        cfg = DiffusersBnBConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")

    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID,
        subfolder="unet",
        quantization_config=cfg,
        torch_dtype=torch.float16,
    )
    return unet


def measure_unet_vram_independent(quant_type):
    """Helper: load just the UNet quantized, measure VRAM, return."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated() / 1e9

    unet = load_unet_quantized(quant_type)
    after = torch.cuda.memory_allocated() / 1e9

    print(f"  UNet {quant_type}: {after - before:.2f} GB on GPU")
    del unet
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["int8", "nf4", "int8_nf4", "vram_only"],
                    default="int8")
    ap.add_argument("--t-split", type=int, default=5)
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--measure-vram", action="store_true")
    args = ap.parse_args()

    if args.mode == "vram_only":
        # Just measure each quantization type's UNet size
        for qt in ["int8", "nf4"]:
            measure_unet_vram_independent(qt)
        return

    print(f"=== V24b mode={args.mode} t_split={args.t_split} ===")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # =========== Load text encoders, VAE, scheduler from FP16 base ===========
    print("\nLoading FP16 pipeline (we'll replace UNet)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    # We'll replace the UNet, so kill the FP16 one
    del pipe.unet
    pipe.unet = None
    gc.collect()
    torch.cuda.empty_cache()

    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    pipe.vae.to("cuda")
    base_components_vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Base components (encoders + VAE) on GPU: {base_components_vram:.2f} GB")

    # =========== Load primary UNet ===========
    if args.mode == "int8":
        primary_quant = "int8"
    elif args.mode == "nf4":
        primary_quant = "nf4"
    elif args.mode == "int8_nf4":
        primary_quant = "int8"   # Phase 1 = INT8

    print(f"\nLoading primary UNet ({primary_quant})...")
    unet_primary = load_unet_quantized(primary_quant)
    pipe.unet = unet_primary
    primary_loaded_vram = torch.cuda.memory_allocated() / 1e9
    print(f"  After primary UNet load. VRAM: {primary_loaded_vram:.2f} GB "
          f"(UNet adds: {primary_loaded_vram - base_components_vram:.2f} GB)")

    # =========== If two-pass, also load secondary NF4 UNet for refiner ===========
    refiner = None
    if args.mode == "int8_nf4":
        print(f"\nLoading secondary UNet (nf4) for Phase 2 refiner...")
        unet_nf4 = load_unet_quantized("nf4")
        secondary_loaded_vram = torch.cuda.memory_allocated() / 1e9
        print(f"  After NF4 UNet load. VRAM: {secondary_loaded_vram:.2f} GB "
              f"(NF4 adds: {secondary_loaded_vram - primary_loaded_vram:.2f} GB)")

        # Build refiner pipeline sharing components
        refiner = StableDiffusionXLImg2ImgPipeline(
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            text_encoder_2=pipe.text_encoder_2,
            tokenizer=pipe.tokenizer,
            tokenizer_2=pipe.tokenizer_2,
            unet=unet_nf4,
            scheduler=pipe.scheduler,
        )
        refiner.set_progress_bar_config(disable=True)

    final_setup_vram = torch.cuda.memory_allocated() / 1e9
    print(f"\nSetup complete. Total VRAM: {final_setup_vram:.2f} GB")

    # =========== Generate ===========
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)[:args.n_images]
    print(f"\nGenerating {len(prompts)} images...")

    if args.mode == "int8_nf4":
        variant_name = f"v24b_int8_nf4_t{args.t_split}"
    else:
        variant_name = f"v24b_{args.mode}_full"

    if not args.measure_vram:
        folder = OUT / variant_name
        folder.mkdir(parents=True, exist_ok=True)

    high_noise_frac = args.t_split / NUM_STEPS

    torch.cuda.reset_peak_memory_stats()
    peaks = []
    timings = []
    t0 = time.time()

    for i, item in enumerate(prompts):
        if not args.measure_vram:
            out_path = folder / item['filename']
            if out_path.exists():
                continue

        prompt = item['caption']
        gen = torch.Generator("cuda").manual_seed(SEED + i)
        t_img = time.time()

        if args.mode == "int8_nf4":
            # Phase 1: INT8, denoising_end
            with torch.inference_mode():
                latent = pipe(
                    prompt=prompt,
                    num_inference_steps=NUM_STEPS,
                    denoising_end=high_noise_frac,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=gen,
                    height=HEIGHT, width=WIDTH,
                    output_type="latent",
                ).images
            # Phase 2: NF4 refiner, denoising_start
            with torch.inference_mode():
                image = refiner(
                    prompt=prompt,
                    num_inference_steps=NUM_STEPS,
                    denoising_start=high_noise_frac,
                    guidance_scale=GUIDANCE_SCALE,
                    image=latent,
                    generator=gen,
                ).images[0]
        else:
            # Single-pass mode
            with torch.inference_mode():
                image = pipe(
                    prompt,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=gen, height=HEIGHT, width=WIDTH,
                ).images[0]

        if not args.measure_vram:
            image.save(out_path)
        timings.append(time.time() - t_img)
        peaks.append(torch.cuda.max_memory_allocated() / 1e9)
        torch.cuda.reset_peak_memory_stats()

        if (i + 1) % 25 == 0 or i < 5:
            elapsed = (time.time() - t0) / 60
            avg_t = sum(timings) / len(timings)
            avg_peak = sum(peaks) / len(peaks)
            max_peak = max(peaks)
            print(f"  [{i+1}/{len(prompts)}] {elapsed:.1f}min  "
                  f"avg t/img: {avg_t:.2f}s  "
                  f"avg peak: {avg_peak:.2f} GB  max peak: {max_peak:.2f} GB",
                  flush=True)

    elapsed = time.time() - t0
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
            "mode": args.mode,
            "t_split": args.t_split,
            "n_images": len(timings),
            "setup_vram_gb": final_setup_vram,
            "avg_peak_vram_gb": avg_peak,
            "max_peak_vram_gb": max_peak,
            "avg_time_per_image_s": avg_time,
            "total_time_min": elapsed / 60,
            "gpu_name": torch.cuda.get_device_name(0),
        }, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
