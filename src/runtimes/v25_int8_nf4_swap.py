"""
V25: INT8 + NF4 two-pass with CPU swap (no FP16 anywhere).

Like V21's FP16+NF4 two-pass, but Phase 1 uses an INT8-quantized UNet instead of FP16.
The CPU swap mechanism keeps only ONE UNet on GPU at a time.

Expected:
  - Peak VRAM: ~6.3 GB (vs V21's 7.30 GB) — under V21!
  - Quality: DreamSim ~0.02 (similar to V21)
  - Speed: ~3-4 sec/image (slightly more than V21's 3.47s due to smaller transfer)

Why this should beat V21:
  - V21 Phase 1 holds 5 GB FP16 UNet. Peak = 7.30 GB.
  - V25 Phase 1 holds 3 GB INT8 UNet. Peak = ~6.3 GB.
  - Both phases use NF4 in Phase 2.

Quality intuition:
  - INT8 SDXL quantization is well-established as near-lossless.
  - INT8 in Phase 1 (early steps) introduces minor error, NF4 in Phase 2 is well-protected
    by the stable trajectory established in Phase 1.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u v25_int8_nf4_swap.py \
        --t-split 10 --n-images 5 --measure-vram   # quick sanity
    CUDA_VISIBLE_DEVICES=2 python -u v25_int8_nf4_swap.py \
        --t-split 10 --n-images 200                # quality measurement
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import gc
import json
import time
from pathlib import Path
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
    BitsAndBytesConfig as DiffusersBnBConfig,
)

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
PROMPTS_PATH = BASE / "benchmark_prompts_5k.json"
OUT = BASE / "benchmark_5k"

NUM_STEPS = 20
GUIDANCE_SCALE = 7.5
HEIGHT = 768
WIDTH = 768
SEED = 42
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def load_unet_quantized(quant_type):
    """Load SDXL UNet pre-quantized via diffusers BitsAndBytesConfig.

    Returns the UNet on whatever device bnb places it (typically CUDA — bnb
    requires CUDA for quantization to work correctly).
    """
    if quant_type == "int8":
        cfg = DiffusersBnBConfig(load_in_8bit=True, llm_int8_threshold=6.0)
    elif quant_type == "nf4":
        cfg = DiffusersBnBConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-split", type=int, default=10)
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--measure-vram", action="store_true")
    ap.add_argument("--no-swap", action="store_true",
                    help="Hold both UNets on GPU (faster, more VRAM)")
    args = ap.parse_args()

    cpu_swap = not args.no_swap
    high_noise_frac = args.t_split / NUM_STEPS

    print(f"=== V25 INT8+NF4 two-pass, t_split={args.t_split} ===")
    print(f"  high_noise_frac (denoising split): {high_noise_frac}")
    print(f"  CPU swap between phases: {cpu_swap}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # === Load FP16 pipeline for base components, then discard its UNet ===
    print("\nLoading FP16 base components (text encoders + VAE)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    # Discard the FP16 UNet
    del pipe.unet
    pipe.unet = None
    gc.collect()
    torch.cuda.empty_cache()

    # Move base components to GPU (always on GPU during inference)
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")
    pipe.vae.to("cuda")
    base_vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Base components on GPU: {base_vram:.2f} GB")

    # === Load INT8 UNet ===
    print("\nLoading INT8 UNet...")
    unet_int8 = load_unet_quantized("int8")
    after_int8_load = torch.cuda.memory_allocated() / 1e9
    int8_unet_size = after_int8_load - base_vram
    print(f"  INT8 UNet loaded. UNet adds: {int8_unet_size:.2f} GB. "
          f"Total VRAM: {after_int8_load:.2f} GB")

    # === Load NF4 UNet ===
    print("\nLoading NF4 UNet...")
    unet_nf4 = load_unet_quantized("nf4")
    after_nf4_load = torch.cuda.memory_allocated() / 1e9
    nf4_unet_size = after_nf4_load - after_int8_load
    print(f"  NF4 UNet loaded. UNet adds: {nf4_unet_size:.2f} GB. "
          f"Total VRAM: {after_nf4_load:.2f} GB")

    # === Build pipelines ===
    pipe.unet = unet_int8

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

    # === Initial swap setup: both on CPU initially if swap mode ===
    if cpu_swap:
        # bnb-quantized models DO support .to("cpu"), but they may need a forward
        # pass on GPU first to "lock in" the quantization. Let's see.
        print("\nMoving NF4 UNet to CPU (Phase 1 won't need it)...")
        try:
            unet_nf4.to("cpu")
        except Exception as e:
            print(f"  Failed to move NF4 to CPU: {e}")
            print(f"  Falling back to no-swap mode")
            cpu_swap = False
        gc.collect()
        torch.cuda.empty_cache()
        post_move_vram = torch.cuda.memory_allocated() / 1e9
        print(f"  After NF4→CPU. VRAM: {post_move_vram:.2f} GB")

    final_setup_vram = torch.cuda.memory_allocated() / 1e9
    print(f"\nSetup complete. Total VRAM: {final_setup_vram:.2f} GB")

    # === Generate ===
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)[:args.n_images]
    print(f"\nGenerating {len(prompts)} images...")

    swap_label = "swap" if cpu_swap else "noswap"
    variant_name = f"v25_int8_nf4_{swap_label}_t{args.t_split}"
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

        # === Phase 1 setup ===
        if cpu_swap and i > 0:
            # First image: INT8 already on GPU. Subsequent: bring it back from CPU.
            unet_int8.to("cuda")
            torch.cuda.empty_cache()

        # === Phase 1: INT8 base ===
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

        # === Phase 1→2 swap ===
        if cpu_swap:
            unet_int8.to("cpu")
            unet_nf4.to("cuda")
            gc.collect()
            torch.cuda.empty_cache()

        # === Phase 2: NF4 refiner ===
        with torch.inference_mode():
            image = refiner(
                prompt=prompt,
                num_inference_steps=NUM_STEPS,
                denoising_start=high_noise_frac,
                guidance_scale=GUIDANCE_SCALE,
                image=latent,
                generator=gen,
            ).images[0]

        # === Cleanup for next image ===
        if cpu_swap:
            unet_nf4.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

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
            "cpu_swap": cpu_swap,
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
