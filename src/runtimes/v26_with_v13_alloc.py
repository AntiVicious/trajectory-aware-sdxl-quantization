"""
V26 with V13 allocation: respects V13's per-layer FP16/INT4 split.

Same INT8→NF4 in-place swap mechanism as V26-multi, BUT only applies to
layers V13's allocator marked as INT4. Layers V13 marked as FP16 stay
FP16 throughout generation.

This is the V26 mechanism integrated with the trajectory framework's
allocation. Three budgets (70%, 55%, 40%) produce different effective
quantization sets.

Memory profile (estimated for 70% budget):
  - 75 V13-FP16 layers stay FP16 throughout: ~1.2 GB
  - 690 V13-INT4 Linear layers as INT8 then NF4 (in-place swap):
    Phase 1 (INT8): ~2.5 GB on GPU
    Phase 2 (NF4): ~1.6 GB on GPU
  - 29 Conv2d FP16: ~0.4 GB
  - Base components: ~1.8 GB
  Phase 1 peak: ~5.9 GB UNet weights + activations ≈ 8 GB
  Phase 2 peak: ~5 GB UNet weights + activations ≈ 7 GB

Note: peak VRAM may be HIGHER than V26 without allocation because some layers
stay FP16. Trade-off is potentially better quality.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u v26_with_v13_alloc.py \
        --budget 70 --t-split 10 --n-images 200
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
ALLOC_DIR = BASE / "trajectory" / "phase3_allocator_full"
OUT = BASE / "benchmark_5k"

NUM_STEPS = 20
GUIDANCE_SCALE = 7.5
HEIGHT = 768
WIDTH = 768
SEED = 42
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def load_v13_allocation(pct):
    """Returns set of layer names V13 marked as INT4 (to be quantized)."""
    p = ALLOC_DIR / f"allocation_{pct}pct_full.json"
    with open(p) as f:
        alloc = json.load(f)
    int4_layers = set()
    for a in alloc["allocations"]:
        if a["choice"] != "FP16":
            int4_layers.add(a["layer"])
    return int4_layers


def precompute_for_int4_set(unet, int4_layer_names):
    """Pre-compute NF4 params and install Linear8bitLt for ONLY the layers in int4_layer_names.
    Other Linear layers stay FP16 throughout."""
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear8bitLt, Params4bit, Int8Params

    layer_meta = []
    nf4_stash = {}

    targets = []
    for name, module in unet.named_modules():
        if name in int4_layer_names and isinstance(module, nn.Linear):
            targets.append((name, module))

    for name, module in targets:
        parts = name.split(".")
        parent_name = ".".join(parts[:-1])
        attr_name = parts[-1]
        parent = unet.get_submodule(parent_name) if parent_name else unet

        in_features = module.in_features
        out_features = module.out_features
        has_bias = module.bias is not None
        bias_data = module.bias.data.clone().to("cpu") if has_bias else None

        # Compute NF4 params on GPU, then stash on CPU
        weight_fp16 = module.weight.data.clone().contiguous()
        nf4_params_gpu = Params4bit(weight_fp16, requires_grad=False, quant_type="nf4").to("cuda")
        nf4_params_cpu = nf4_params_gpu.to("cpu")
        del nf4_params_gpu, weight_fp16

        nf4_stash[(parent_name, attr_name)] = nf4_params_cpu

        # Install Linear8bitLt
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
    """After warm-up, stash references to the (now-quantized) Linear8bitLt modules."""
    int8_stash = {}
    for meta in layer_meta:
        parent = unet.get_submodule(meta["parent_name"]) if meta["parent_name"] else unet
        module = getattr(parent, meta["attr_name"])
        int8_stash[(meta["parent_name"], meta["attr_name"])] = module
    return int8_stash


def swap_to_nf4(unet, layer_meta, int8_stash, nf4_stash):
    """Replace Linear8bitLt with Linear4bit using stashed NF4 params; stash old INT8 to CPU."""
    from bitsandbytes.nn import Linear4bit

    for meta in layer_meta:
        parent = unet.get_submodule(meta["parent_name"]) if meta["parent_name"] else unet

        new_nf4 = Linear4bit(
            meta["in_features"], meta["out_features"],
            bias=meta["has_bias"],
            quant_type="nf4",
            compute_dtype=torch.float16,
        )
        nf4_on_gpu = nf4_stash[(meta["parent_name"], meta["attr_name"])].to("cuda")
        new_nf4.weight = nf4_on_gpu
        if meta["has_bias"]:
            new_nf4.bias = nn.Parameter(meta["bias_data"].to("cuda"), requires_grad=False)
        new_nf4 = new_nf4.to("cuda")

        old_int8 = getattr(parent, meta["attr_name"])
        try:
            old_int8_cpu = old_int8.to("cpu")
            int8_stash[(meta["parent_name"], meta["attr_name"])] = old_int8_cpu
        except Exception:
            int8_stash[(meta["parent_name"], meta["attr_name"])] = None

        nf4_stash[(meta["parent_name"], meta["attr_name"])] = nf4_on_gpu
        setattr(parent, meta["attr_name"], new_nf4)

    gc.collect()
    torch.cuda.empty_cache()


def swap_to_int8(unet, layer_meta, int8_stash, nf4_stash):
    """Replace Linear4bit with stashed Linear8bitLt; stash NF4 weight back to CPU."""
    for meta in layer_meta:
        parent = unet.get_submodule(meta["parent_name"]) if meta["parent_name"] else unet

        stashed_int8 = int8_stash[(meta["parent_name"], meta["attr_name"])]
        if stashed_int8 is None:
            raise RuntimeError(
                f"INT8 stash for {meta['parent_name']}.{meta['attr_name']} is None.")
        int8_on_gpu = stashed_int8.to("cuda")
        int8_stash[(meta["parent_name"], meta["attr_name"])] = int8_on_gpu

        old_nf4 = getattr(parent, meta["attr_name"])
        nf4_weight_cpu = old_nf4.weight.to("cpu")
        nf4_stash[(meta["parent_name"], meta["attr_name"])] = nf4_weight_cpu

        setattr(parent, meta["attr_name"], int8_on_gpu)

    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=70)
    ap.add_argument("--t-split", type=int, default=10)
    ap.add_argument("--n-images", type=int, default=200)
    args = ap.parse_args()

    high_noise_frac = args.t_split / NUM_STEPS

    print(f"=== V26-V13alloc, budget={args.budget}%, t_split={args.t_split} ===")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Load V13 allocation
    int4_layers = load_v13_allocation(args.budget)
    print(f"  V13 allocation: {len(int4_layers)} layers in INT4 set "
          f"(794 - {len(int4_layers)} = {794 - len(int4_layers)} stay FP16)")

    print("\nLoading FP16 SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    print(f"  After FP16 load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print(f"\nPre-computing NF4 stash + installing INT8 for {args.budget}% allocation...")
    layer_meta, nf4_stash = precompute_for_int4_set(pipe.unet, int4_layers)
    print(f"  {len(layer_meta)} Linear layers prepared (subset of {len(int4_layers)} V13 INT4 set)")
    print(f"  After setup. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

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

    # Warm-up
    print("\nWarm-up forward...")
    with torch.inference_mode():
        _ = pipe(prompt="warmup", num_inference_steps=2,
                 guidance_scale=GUIDANCE_SCALE,
                 generator=torch.Generator("cuda").manual_seed(0),
                 height=HEIGHT, width=WIDTH, output_type="latent").images
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  After warm-up. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    int8_stash = stash_int8_params(pipe.unet, layer_meta)

    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)[:args.n_images]
    print(f"\nGenerating {len(prompts)} images...")

    variant_name = f"v26v13_{args.budget}pct_t{args.t_split}"
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

        if i > 0:
            try:
                swap_to_int8(pipe.unet, layer_meta, int8_stash, nf4_stash)
            except Exception as e:
                print(f"  [Image {i+1}] swap_to_int8 failed: {e}", flush=True)
                break

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

        try:
            swap_to_nf4(pipe.unet, layer_meta, int8_stash, nf4_stash)
        except Exception as e:
            print(f"  [Image {i+1}] swap_to_nf4 failed: {e}", flush=True)
            break

        with torch.inference_mode():
            image = refiner(
                prompt=prompt,
                num_inference_steps=NUM_STEPS,
                denoising_start=high_noise_frac,
                guidance_scale=GUIDANCE_SCALE,
                image=latent,
                generator=gen,
            ).images[0]

        image.save(out_path)
        timings.append(time.time() - t_img)
        peaks.append(torch.cuda.max_memory_allocated() / 1e9)
        torch.cuda.reset_peak_memory_stats()

        if (i + 1) % 25 == 0 or i < 5:
            elapsed = (time.time() - t_overall) / 60
            avg_t = sum(timings) / len(timings)
            avg_peak = sum(peaks) / len(peaks)
            max_peak = max(peaks)
            eta = (len(prompts) - i - 1) * avg_t / 60
            print(f"  [{i+1}/{len(prompts)}] {elapsed:.1f}min  "
                  f"t/img: {avg_t:.1f}s  "
                  f"avg peak: {avg_peak:.2f} GB  max peak: {max_peak:.2f} GB  "
                  f"ETA: {eta:.1f}min", flush=True)

    elapsed = time.time() - t_overall
    avg_peak = sum(peaks) / len(peaks) if peaks else 0
    max_peak = max(peaks) if peaks else 0
    avg_time = sum(timings) / len(timings) if timings else 0

    print(f"\n=== Result ===")
    print(f"  Variant:                {variant_name}")
    print(f"  Budget:                 {args.budget}%")
    print(f"  N images:               {len(timings)}")
    print(f"  N quantized layers:     {len(layer_meta)}")
    print(f"  Average peak VRAM:      {avg_peak:.2f} GB")
    print(f"  Max peak VRAM:          {max_peak:.2f} GB")
    print(f"  Average time per image: {avg_time:.2f} s")

    out_json = BASE / f"vram_measurement_{variant_name}.json"
    with open(out_json, "w") as f:
        json.dump({
            "variant": variant_name,
            "budget_pct": args.budget,
            "t_split": args.t_split,
            "n_images": len(timings),
            "n_quantized_layers": len(layer_meta),
            "avg_peak_vram_gb": avg_peak,
            "max_peak_vram_gb": max_peak,
            "avg_time_per_image_s": avg_time,
            "gpu_name": torch.cuda.get_device_name(0),
        }, f, indent=2)
    print(f"  Saved: {out_json}")


if __name__ == "__main__":
    main()
