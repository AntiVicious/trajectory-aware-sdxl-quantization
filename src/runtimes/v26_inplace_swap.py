"""
V26: in-place INT8→NF4 weight swap on a single UNet.

ONE UNet on GPU at all times. Each Linear layer starts as Linear8bitLt (INT8)
for steps 0..t_split-1, then is replaced in-place with Linear4bit (NF4)
for steps t_split..NUM_STEPS-1.

Pre-computation:
  1. Load FP16 UNet on CUDA
  2. For each Linear layer:
     a. Create Linear4bit (NF4) version, move Params4bit to CPU
     b. Replace original Linear with Linear8bitLt (INT8) on GPU
  3. FP16 weights are released as Linear8bitLt takes ownership

At t_split (between Phase 1 and Phase 2):
  For each Linear layer:
    a. Move NF4 Params4bit from CPU to CUDA
    b. Build a new Linear4bit module with these params
    c. Replace the Linear8bitLt with the new Linear4bit
    d. Free the Linear8bitLt

Memory profile:
  - Phase 1: INT8 UNet + base + activations = ~6.3 GB peak
  - Swap transient: per-layer ~50 MB extra (one Linear's worth of FP16+NF4+INT8)
  - Phase 2: NF4 UNet + base + activations = ~5.8 GB peak
  - Overall peak: ~6.3 GB (Phase 1 dominates)

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u v26_inplace_swap.py \
        --t-split 10 --n-images 5 --measure-vram
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

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
PROMPTS_PATH = BASE / "benchmark_prompts_5k.json"
OUT = BASE / "benchmark_5k"

NUM_STEPS = 20
GUIDANCE_SCALE = 7.5
HEIGHT = 768
WIDTH = 768
SEED = 42
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def precompute_nf4_and_install_int8(unet):
    """For each Linear layer in unet:
      1. Build a Linear4bit (NF4) version, move its Params4bit to CPU
      2. Replace the Linear with Linear8bitLt (INT8) on GPU
      3. Save metadata for later swap

    Returns: dict mapping (parent_name, attr_name) -> {nf4_params, bias, in/out features}
    """
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Linear8bitLt, Params4bit, Int8Params

    # Find all Linear layers
    targets = []
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            targets.append((name, module))

    nf4_storage = {}   # key: (parent_name, attr_name) -> precomputed NF4 stuff
    n_replaced = 0

    for name, module in targets:
        parts = name.split(".")
        parent_name = ".".join(parts[:-1])
        attr_name = parts[-1]
        parent = unet.get_submodule(parent_name) if parent_name else unet

        # Step 1: build NF4 weight (uses original FP16 weight on GPU)
        weight_fp16 = module.weight.data.clone().contiguous()
        nf4_params = Params4bit(
            weight_fp16, requires_grad=False, quant_type="nf4"
        )
        # NF4 quantization happens during .to(device); do this on CPU to keep VRAM low
        nf4_params = nf4_params.to("cuda")  # Quantize on GPU first
        # Then move quantized data to CPU
        nf4_params_cpu = nf4_params.to("cpu")
        del nf4_params

        bias_fp16 = module.bias.data.clone() if module.bias is not None else None

        nf4_storage[(parent_name, attr_name)] = {
            "nf4_params": nf4_params_cpu,
            "bias": bias_fp16,
            "in_features": module.in_features,
            "out_features": module.out_features,
            "has_bias": module.bias is not None,
        }

        # Step 2: replace with Linear8bitLt
        new_int8 = Linear8bitLt(
            module.in_features, module.out_features,
            bias=(module.bias is not None),
            has_fp16_weights=False,
            threshold=6.0,
        )
        new_int8.weight = Int8Params(
            module.weight.data.clone(),
            has_fp16_weights=False,
            requires_grad=False,
        )
        if module.bias is not None:
            new_int8.bias = nn.Parameter(module.bias.data.clone(), requires_grad=False)
        new_int8 = new_int8.to(module.weight.device)
        setattr(parent, attr_name, new_int8)
        n_replaced += 1

        # Free the temporary FP16 weight clone
        del weight_fp16
        if bias_fp16 is not None:
            del bias_fp16

    gc.collect()
    torch.cuda.empty_cache()

    return nf4_storage, n_replaced


def swap_int8_to_nf4(unet, nf4_storage):
    """Replace all Linear8bitLt with Linear4bit using pre-computed NF4 params.
    Move each layer's NF4 params CPU→GPU one at a time to keep transient peak small.
    """
    from bitsandbytes.nn import Linear4bit, Params4bit

    n_swapped = 0
    for (parent_name, attr_name), meta in nf4_storage.items():
        parent = unet.get_submodule(parent_name) if parent_name else unet
        old_int8 = getattr(parent, attr_name)

        # Build new Linear4bit
        new_nf4 = Linear4bit(
            meta["in_features"], meta["out_features"],
            bias=meta["has_bias"],
            quant_type="nf4",
            compute_dtype=torch.float16,
        )

        # Move pre-quantized NF4 params from CPU to GPU
        nf4_on_gpu = meta["nf4_params"].to("cuda")
        new_nf4.weight = nf4_on_gpu
        if meta["has_bias"]:
            new_nf4.bias = nn.Parameter(meta["bias"].to("cuda"), requires_grad=False)
        new_nf4 = new_nf4.to("cuda")

        # Replace in parent
        setattr(parent, attr_name, new_nf4)
        n_swapped += 1

        # Delete the old INT8 module - it's now unreferenced
        del old_int8

        # Clear the CPU NF4 reference (now on GPU as new_nf4.weight)
        meta["nf4_params"] = None

    gc.collect()
    torch.cuda.empty_cache()
    return n_swapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-split", type=int, default=10)
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--measure-vram", action="store_true")
    args = ap.parse_args()

    high_noise_frac = args.t_split / NUM_STEPS

    print(f"=== V26 in-place INT8→NF4 swap, t_split={args.t_split} ===")
    print(f"  high_noise_frac (denoising split): {high_noise_frac}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # === Load FP16 pipeline ===
    print("\nLoading FP16 SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    print(f"  After FP16 load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # === Pre-compute NF4 params for each Linear, install Linear8bitLt ===
    print("\nPre-computing NF4 params and installing INT8 layers...")
    nf4_storage, n_replaced = precompute_nf4_and_install_int8(pipe.unet)
    after_setup_vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Replaced {n_replaced} Linear → Linear8bitLt")
    print(f"  After INT8 setup. VRAM: {after_setup_vram:.2f} GB")
    print(f"  NF4 params held on CPU (not counted in GPU VRAM)")

    # Build refiner pipeline (Img2Img) sharing components — points to SAME UNet
    refiner = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,    # SAME UNet — Phase 2 will use it after the in-place swap
        scheduler=pipe.scheduler,
    )
    refiner.set_progress_bar_config(disable=True)

    # === Generate ===
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)[:args.n_images]
    # === Warm-up forward to trigger INT8 first-forward quantization ===
    print("\nWarm-up forward pass (triggers INT8 quantization, not measured)...")
    with torch.inference_mode():
        _ = pipe(
            prompt="a cat",
            num_inference_steps=2,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(0),
            height=HEIGHT, width=WIDTH,
            output_type="latent",
        ).images
    gc.collect()
    torch.cuda.empty_cache()
    after_warmup_vram = torch.cuda.memory_allocated() / 1e9
    print(f"  After warm-up. VRAM: {after_warmup_vram:.2f} GB")

    print(f"\nGenerating {len(prompts)} images...")

    variant_name = f"v26_inplace_t{args.t_split}"
    if not args.measure_vram:
        folder = OUT / variant_name
        folder.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()  # reset AFTER warm-up
    peaks = []
    timings = []
    swap_times = []
    t_overall = time.time()

    is_int8_active = True   # which precision is currently installed

    for i, item in enumerate(prompts):
        if not args.measure_vram:
            out_path = folder / item['filename']
            if out_path.exists():
                continue
        prompt = item['caption']
        gen = torch.Generator("cuda").manual_seed(SEED + i)

        t_img = time.time()

        # === Ensure INT8 is active for Phase 1 ===
        # First image: already INT8. Subsequent: need to swap NF4 → INT8.
        if not is_int8_active:
            print(f"  [Image {i+1}] Re-installing INT8 (was NF4)...", flush=True)
            # We need to re-do the precompute_nf4_and_install_int8 from current
            # NF4 modules. But we don't have FP16 weights anymore!
            # FIX: this version only generates ONE image correctly.
            # For multi-image, we'd need to keep INT8 weights stashed too.
            print(f"  ERROR: V26 only supports one image without re-init")
            break

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
        post_phase1_vram = torch.cuda.memory_allocated() / 1e9

        # === In-place swap: INT8 → NF4 ===
        t_swap = time.time()
        n_swapped = swap_int8_to_nf4(pipe.unet, nf4_storage)
        is_int8_active = False
        swap_time = time.time() - t_swap
        swap_times.append(swap_time)
        post_swap_vram = torch.cuda.memory_allocated() / 1e9
        if i < 3:
            print(f"  [Image {i+1}] Swap completed in {swap_time:.2f}s. "
                  f"After Phase1: {post_phase1_vram:.2f} GB → "
                  f"After swap: {post_swap_vram:.2f} GB", flush=True)

        # === Phase 2: NF4 refiner (uses SAME pipe.unet, now with NF4 modules) ===
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

        if (i + 1) % 5 == 0 or i < 3:
            elapsed = (time.time() - t_overall) / 60
            avg_t = sum(timings) / len(timings)
            avg_peak = sum(peaks) / len(peaks)
            max_peak = max(peaks)
            print(f"  [{i+1}/{len(prompts)}] {elapsed:.1f}min  "
                  f"avg t/img: {avg_t:.2f}s  "
                  f"avg peak: {avg_peak:.2f} GB  max peak: {max_peak:.2f} GB",
                  flush=True)

        # ⚠️ V26 LIMITATION: After NF4 swap, FP16 weights are gone.
        # Cannot re-build INT8 for next image. Break after first image.
        if i == 0:
            print(f"\n  V26 limitation: Cannot generate Image 2 without FP16 weights.")
            print(f"  This run measures peak VRAM and quality for image 1 only.")
            break

    elapsed = time.time() - t_overall
    avg_peak = sum(peaks) / len(peaks) if peaks else 0
    max_peak = max(peaks) if peaks else 0
    avg_time = sum(timings) / len(timings) if timings else 0
    avg_swap = sum(swap_times) / len(swap_times) if swap_times else 0

    print(f"\n=== Result ===")
    print(f"  Variant:                {variant_name}")
    print(f"  N images generated:     {len(timings)}")
    print(f"  Setup VRAM:             {after_setup_vram:.2f} GB")
    print(f"  Average peak VRAM:      {avg_peak:.2f} GB")
    print(f"  Max peak VRAM:          {max_peak:.2f} GB")
    print(f"  Average time per image: {avg_time:.2f} s")
    print(f"  Average swap time:      {avg_swap:.2f} s")
    print(f"  Total time:             {elapsed/60:.1f} min")

    out_path = BASE / f"vram_measurement_{variant_name}.json"
    with open(out_path, "w") as f:
        json.dump({
            "variant": variant_name,
            "t_split": args.t_split,
            "n_images": len(timings),
            "setup_vram_gb": after_setup_vram,
            "avg_peak_vram_gb": avg_peak,
            "max_peak_vram_gb": max_peak,
            "avg_time_per_image_s": avg_time,
            "avg_swap_time_s": avg_swap,
            "total_time_min": elapsed / 60,
            "n_layers_swapped": n_swapped if 'n_swapped' in dir() else 0,
            "gpu_name": torch.cuda.get_device_name(0),
        }, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
