"""
V26 multi-image: in-place INT8↔NF4 swap with CPU stash for both precisions.

Setup:
  1. Load FP16 UNet on GPU
  2. For each Linear:
     - Build NF4 Params4bit, stash on CPU
     - Build INT8 Int8Params (after first forward), stash on CPU
     - Install INT8 module on GPU (active)
  3. After warm-up, FP16 weights released

Per image:
  Phase 1 (INT8 on GPU):
    - INT8 modules already installed
    - Run scheduler steps 0..t_split-1
  Swap NF4-on:
    - For each Linear: copy NF4 params CPU→GPU, replace module
    - Stash INT8 params back to CPU
    - GC + cache clear
  Phase 2 (NF4 on GPU):
    - Run scheduler steps t_split..NUM_STEPS-1
  Swap INT8-on (preparing for next image):
    - For each Linear: copy INT8 params CPU→GPU, replace module
    - Stash NF4 params back to CPU

Memory profile per phase:
  Phase 1: INT8 on GPU (~3 GB) + base (~1.8 GB) + acts (~1.5 GB) ≈ 6.3 GB peak
  Phase 2: NF4 on GPU (~2 GB) + base + acts (~2 GB) ≈ 5.8 GB peak
  Per-image peak: ~6.3 GB (Phase 1 dominates)

CPU memory: INT8 (~3 GB) + NF4 (~2 GB) = ~5 GB stashed.

Speed: each swap requires moving ~2-3 GB across PCIe = ~0.5-1 sec per swap.
Per image: 2 swaps = ~1-2 sec overhead. Plus actual generation (~2 sec).

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u v26_multi_image.py \
        --t-split 10 --n-images 5
    CUDA_VISIBLE_DEVICES=2 python -u v26_multi_image.py \
        --t-split 10 --n-images 200
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
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def precompute_both_quantizations(unet):
    """
    For each Linear layer:
      1. Compute NF4 Params4bit on GPU (using FP16 weight), then move to CPU
      2. Replace original Linear with Linear8bitLt on GPU (Int8Params not yet quantized)
      3. After this function returns, caller does a warm-up forward to trigger INT8
         quantization (Linear8bitLt with has_fp16_weights=False quantizes on first forward)
      4. After warm-up, INT8 params can be stashed to CPU separately

    Returns: list of (parent_name, attr_name, in_features, out_features, has_bias, bias_data)
             metadata for each layer. NF4 params are stored in nf4_stash dict keyed by
             (parent_name, attr_name).
    """
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Linear8bitLt, Params4bit, Int8Params

    layer_meta = []
    nf4_stash = {}

    targets = [(name, module) for name, module in unet.named_modules()
               if isinstance(module, nn.Linear)]

    for name, module in targets:
        parts = name.split(".")
        parent_name = ".".join(parts[:-1])
        attr_name = parts[-1]
        parent = unet.get_submodule(parent_name) if parent_name else unet

        in_features = module.in_features
        out_features = module.out_features
        has_bias = module.bias is not None
        bias_data = module.bias.data.clone().to("cpu") if has_bias else None

        # === Compute NF4 params on GPU, then move to CPU ===
        weight_fp16 = module.weight.data.clone().contiguous()
        nf4_params_gpu = Params4bit(weight_fp16, requires_grad=False, quant_type="nf4").to("cuda")
        # Move quantized data to CPU for stash
        nf4_params_cpu = nf4_params_gpu.to("cpu")
        del nf4_params_gpu, weight_fp16

        nf4_stash[(parent_name, attr_name)] = nf4_params_cpu

        # === Install Linear8bitLt on GPU ===
        new_int8 = Linear8bitLt(
            in_features, out_features,
            bias=has_bias,
            has_fp16_weights=False,
            threshold=6.0,
        )
        new_int8.weight = Int8Params(
            module.weight.data.clone(),
            has_fp16_weights=False,
            requires_grad=False,
        )
        if has_bias:
            new_int8.bias = nn.Parameter(module.bias.data.clone(), requires_grad=False)
        new_int8 = new_int8.to(module.weight.device)
        setattr(parent, attr_name, new_int8)

        layer_meta.append({
            "parent_name": parent_name,
            "attr_name": attr_name,
            "in_features": in_features,
            "out_features": out_features,
            "has_bias": has_bias,
            "bias_data": bias_data,
        })

    gc.collect()
    torch.cuda.empty_cache()

    return layer_meta, nf4_stash


def stash_int8_params(unet, layer_meta):
    """After warm-up, the Int8Params has been quantized. Stash a CPU copy of each layer's
    Int8Params (CB, SCB, etc.) so we can restore later.
    Returns: int8_stash dict keyed by (parent_name, attr_name)."""
    int8_stash = {}
    for meta in layer_meta:
        parent_name = meta["parent_name"]
        attr_name = meta["attr_name"]
        parent = unet.get_submodule(parent_name) if parent_name else unet
        module = getattr(parent, attr_name)
        # The Int8Params after quantization has CB (column-wise compressed buffer)
        # We can stash the whole module's state_dict
        # But moving Linear8bitLt as a whole to CPU is problematic
        # Instead: take its weight (Int8Params) and bias references
        # Note: bnb's Int8Params might not survive .to("cpu") cleanly

        # Try the simple approach first: just keep the module reference
        # We'll move them between devices module-by-module
        int8_stash[(parent_name, attr_name)] = module
    return int8_stash


def swap_to_nf4(unet, layer_meta, int8_stash, nf4_stash):
    """Replace each Linear8bitLt with Linear4bit using stashed NF4 params.
    Stash the INT8 modules to CPU."""
    from bitsandbytes.nn import Linear4bit

    for meta in layer_meta:
        parent_name = meta["parent_name"]
        attr_name = meta["attr_name"]
        parent = unet.get_submodule(parent_name) if parent_name else unet

        # Build new Linear4bit
        new_nf4 = Linear4bit(
            meta["in_features"], meta["out_features"],
            bias=meta["has_bias"],
            quant_type="nf4",
            compute_dtype=torch.float16,
        )
        # Move stashed NF4 params from CPU to GPU
        nf4_on_gpu = nf4_stash[(parent_name, attr_name)].to("cuda")
        new_nf4.weight = nf4_on_gpu
        if meta["has_bias"]:
            new_nf4.bias = nn.Parameter(
                meta["bias_data"].to("cuda"), requires_grad=False)
        new_nf4 = new_nf4.to("cuda")

        # Get current INT8 module, move it to CPU stash
        old_int8 = getattr(parent, attr_name)
        try:
            old_int8_cpu = old_int8.to("cpu")
            int8_stash[(parent_name, attr_name)] = old_int8_cpu
        except Exception as e:
            # If INT8 can't move to CPU, just discard (we'll rebuild from NF4 next time)
            int8_stash[(parent_name, attr_name)] = None

        # Update NF4 stash to reflect current GPU state
        nf4_stash[(parent_name, attr_name)] = nf4_on_gpu

        # Install new module
        setattr(parent, attr_name, new_nf4)

    gc.collect()
    torch.cuda.empty_cache()


def swap_to_int8(unet, layer_meta, int8_stash, nf4_stash):
    """Replace each Linear4bit with stashed Linear8bitLt.
    Stash the NF4 modules to CPU."""
    for meta in layer_meta:
        parent_name = meta["parent_name"]
        attr_name = meta["attr_name"]
        parent = unet.get_submodule(parent_name) if parent_name else unet

        # Move stashed INT8 module from CPU back to GPU
        stashed_int8 = int8_stash[(parent_name, attr_name)]
        if stashed_int8 is None:
            raise RuntimeError(
                f"INT8 stash for {parent_name}.{attr_name} is None. Cannot recover.")
        int8_on_gpu = stashed_int8.to("cuda")
        int8_stash[(parent_name, attr_name)] = int8_on_gpu

        # Get current NF4 module, move to CPU stash
        old_nf4 = getattr(parent, attr_name)
        # Take the NF4 weight, move to CPU
        nf4_weight_cpu = old_nf4.weight.to("cpu")
        nf4_stash[(parent_name, attr_name)] = nf4_weight_cpu

        # Install INT8 module
        setattr(parent, attr_name, int8_on_gpu)

    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-split", type=int, default=10)
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--measure-vram", action="store_true")
    args = ap.parse_args()

    high_noise_frac = args.t_split / NUM_STEPS

    print(f"=== V26 multi-image, t_split={args.t_split} ===")
    print(f"  high_noise_frac: {high_noise_frac}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\nLoading FP16 SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    print(f"  After FP16 load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("\nPre-computing NF4 stash + installing INT8 layers...")
    layer_meta, nf4_stash = precompute_both_quantizations(pipe.unet)
    print(f"  {len(layer_meta)} Linear layers prepared")
    print(f"  After setup. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Build refiner pipeline using SAME UNet
    refiner = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
    )
    refiner.set_progress_bar_config(disable=True)

    # === Warm-up forward to trigger INT8 quantization ===
    print("\nWarm-up forward (triggers INT8 quantization)...")
    with torch.inference_mode():
        _ = pipe(
            prompt="warmup",
            num_inference_steps=2,
            guidance_scale=GUIDANCE_SCALE,
            generator=torch.Generator("cuda").manual_seed(0),
            height=HEIGHT, width=WIDTH,
            output_type="latent",
        ).images
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  After warm-up. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # === Stash INT8 module references (they're now on GPU, quantized) ===
    int8_stash = stash_int8_params(pipe.unet, layer_meta)

    # === Generate ===
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)[:args.n_images]
    print(f"\nGenerating {len(prompts)} images...")

    variant_name = f"v26m_inplace_t{args.t_split}"
    if not args.measure_vram:
        folder = OUT / variant_name
        folder.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    peaks = []
    timings = []
    swap_to_nf4_times = []
    swap_to_int8_times = []
    t_overall = time.time()

    for i, item in enumerate(prompts):
        if not args.measure_vram:
            out_path = folder / item['filename']
            if out_path.exists():
                continue
        prompt = item['caption']
        gen = torch.Generator("cuda").manual_seed(SEED + i)

        t_img = time.time()

        # === If not first image, restore INT8 from stash ===
        if i > 0:
            t_swap = time.time()
            try:
                swap_to_int8(pipe.unet, layer_meta, int8_stash, nf4_stash)
            except Exception as e:
                print(f"  [Image {i+1}] swap_to_int8 failed: {e}", flush=True)
                break
            swap_to_int8_times.append(time.time() - t_swap)

        # === Phase 1: INT8 ===
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

        # === Swap to NF4 ===
        t_swap = time.time()
        swap_to_nf4(pipe.unet, layer_meta, int8_stash, nf4_stash)
        swap_to_nf4_times.append(time.time() - t_swap)

        # === Phase 2: NF4 ===
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

        if (i + 1) % 10 == 0 or i < 5:
            elapsed = (time.time() - t_overall) / 60
            avg_t = sum(timings) / len(timings)
            avg_peak = sum(peaks) / len(peaks)
            max_peak = max(peaks)
            avg_swap_nf4 = sum(swap_to_nf4_times) / len(swap_to_nf4_times) if swap_to_nf4_times else 0
            avg_swap_int8 = sum(swap_to_int8_times) / len(swap_to_int8_times) if swap_to_int8_times else 0
            print(f"  [{i+1}/{len(prompts)}] {elapsed:.1f}min  "
                  f"t/img: {avg_t:.1f}s (swap NF4: {avg_swap_nf4:.1f}s, INT8: {avg_swap_int8:.1f}s)  "
                  f"avg peak: {avg_peak:.2f} GB  max peak: {max_peak:.2f} GB",
                  flush=True)

    elapsed = time.time() - t_overall
    avg_peak = sum(peaks) / len(peaks) if peaks else 0
    max_peak = max(peaks) if peaks else 0
    avg_time = sum(timings) / len(timings) if timings else 0
    avg_swap_nf4 = sum(swap_to_nf4_times) / len(swap_to_nf4_times) if swap_to_nf4_times else 0
    avg_swap_int8 = sum(swap_to_int8_times) / len(swap_to_int8_times) if swap_to_int8_times else 0

    print(f"\n=== Result ===")
    print(f"  Variant:                {variant_name}")
    print(f"  N images generated:     {len(timings)}")
    print(f"  Average peak VRAM:      {avg_peak:.2f} GB")
    print(f"  Max peak VRAM:          {max_peak:.2f} GB")
    print(f"  Average time per image: {avg_time:.2f} s")
    print(f"  Avg swap to NF4:        {avg_swap_nf4:.2f} s")
    print(f"  Avg swap to INT8:       {avg_swap_int8:.2f} s")
    print(f"  Total time:             {elapsed/60:.1f} min")

    out_path = BASE / f"vram_measurement_{variant_name}.json"
    with open(out_path, "w") as f:
        json.dump({
            "variant": variant_name,
            "t_split": args.t_split,
            "n_images": len(timings),
            "avg_peak_vram_gb": avg_peak,
            "max_peak_vram_gb": max_peak,
            "avg_time_per_image_s": avg_time,
            "avg_swap_to_nf4_s": avg_swap_nf4,
            "avg_swap_to_int8_s": avg_swap_int8,
            "total_time_min": elapsed / 60,
            "n_layers": len(layer_meta),
            "gpu_name": torch.cuda.get_device_name(0),
        }, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
