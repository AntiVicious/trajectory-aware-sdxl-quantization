"""
V21 Two-pass: FP16 UNet for early denoising steps, NF4 UNet for late steps.

Approximates V13's trajectory dimension with a single global split point t_split.
Layers committed before t_split should be FP16 (we use FP16 UNet); layers
committed at/after t_split should be quantized (we use NF4 UNet, which has
all V13's "INT4 layers" replaced with Linear4bit).

Implementation: hold one UNet on GPU at a time. Swap via .to('cpu')/.to('cuda')
between phases. This trades latency (~10 sec/image overhead for the swap)
for VRAM (peak ~ max(FP16 UNet phase, NF4 UNet phase) instead of both).

For each image:
  Phase 1 (steps 0..t_split-1):
    - FP16 UNet on GPU, NF4 UNet on CPU
    - Run partial denoising, save latent
  Phase 2 (steps t_split..NUM_STEPS-1):
    - FP16 UNet to CPU, NF4 UNet to GPU
    - Resume denoising from saved latent

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u v21_two_pass.py \
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
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
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
    return fp16_layers, int4_layers, alloc


def replace_linear_with_nf4(unet, int4_layer_names):
    """Replace Linear layers in int4_layer_names with Linear4bit(NF4) in place."""
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Params4bit

    n_replaced = 0
    n_skipped_conv = 0
    n_other = 0
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
            n_other += 1
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


def encode_prompt(pipe, prompt):
    """Encode the prompt using SDXL's two text encoders. Returns (prompt_embeds,
    pooled_prompt_embeds) and the negative versions."""
    with torch.inference_mode():
        return pipe.encode_prompt(
            prompt=prompt, prompt_2=None,
            device="cuda", num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None, negative_prompt_2=None,
        )


def prepare_latents(pipe, generator):
    """Prepare initial random latents."""
    shape = (1, pipe.unet.config.in_channels,
             HEIGHT // pipe.vae_scale_factor,
             WIDTH // pipe.vae_scale_factor)
    latents = torch.randn(shape, generator=generator,
                           dtype=torch.float16, device="cuda")
    latents = latents * pipe.scheduler.init_noise_sigma
    return latents


def get_added_cond_kwargs(pipe, pooled_prompt_embeds, neg_pooled_embeds, batch_size=1):
    """Build the SDXL added_cond_kwargs (text_embeds + time_ids)."""
    add_text_embeds = torch.cat([neg_pooled_embeds, pooled_prompt_embeds])
    original_size = (HEIGHT, WIDTH)
    target_size = (HEIGHT, WIDTH)
    crops_coords_top_left = (0, 0)
    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=torch.float16, device="cuda")
    add_time_ids = torch.cat([add_time_ids, add_time_ids])  # negative + positive
    return {"text_embeds": add_text_embeds, "time_ids": add_time_ids}


def denoise_step(unet, latents, t, prompt_embeds_combined, added_cond_kwargs,
                  guidance_scale):
    """Single denoising step: run UNet, apply CFG, return noise prediction."""
    latent_model_input = torch.cat([latents, latents])  # negative + positive
    latent_model_input = pipe_global.scheduler.scale_model_input(latent_model_input, t)
    with torch.inference_mode():
        noise_pred = unet(
            latent_model_input, t,
            encoder_hidden_states=prompt_embeds_combined,
            added_cond_kwargs=added_cond_kwargs,
        ).sample
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred


def two_pass_generate(pipe_fp16, unet_nf4, prompt, t_split, generator,
                       verbose=False):
    """Generate one image with two-pass UNet swap.
    pipe_fp16: full pipeline with FP16 UNet currently on CPU or GPU
    unet_nf4: NF4 UNet (currently on CPU)

    Returns peak VRAM observed during this call."""
    global pipe_global
    pipe_global = pipe_fp16

    # Encode prompt (text encoders should be on GPU already)
    prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = encode_prompt(
        pipe_fp16, prompt)
    prompt_embeds_combined = torch.cat([neg_prompt_embeds, prompt_embeds])
    added_cond_kwargs = get_added_cond_kwargs(
        pipe_fp16, pooled_prompt_embeds, neg_pooled_prompt_embeds)

    # Set scheduler to NUM_STEPS
    pipe_fp16.scheduler.set_timesteps(NUM_STEPS, device="cuda")
    timesteps = pipe_fp16.scheduler.timesteps  # tensor of NUM_STEPS values

    # Prepare initial latents
    latents = prepare_latents(pipe_fp16, generator)

    # === Phase 1: FP16 UNet, steps 0..t_split-1 ===
    # Move FP16 UNet to GPU
    t0 = time.time()
    pipe_fp16.unet.to("cuda")
    torch.cuda.synchronize()
    move1_time = time.time() - t0
    if verbose:
        print(f"    Phase 1 setup ({move1_time:.2f}s): FP16 UNet on GPU. "
              f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    for i, t in enumerate(timesteps[:t_split]):
        noise_pred = denoise_step(
            pipe_fp16.unet, latents, t,
            prompt_embeds_combined, added_cond_kwargs,
            GUIDANCE_SCALE
        )
        latents = pipe_fp16.scheduler.step(noise_pred, t, latents).prev_sample

    # === Swap: FP16 UNet → CPU, NF4 UNet → GPU ===
    t0 = time.time()
    pipe_fp16.unet.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    unet_nf4.to("cuda")
    torch.cuda.synchronize()
    swap_time = time.time() - t0
    if verbose:
        print(f"    Phase 2 setup ({swap_time:.2f}s): NF4 UNet on GPU. "
              f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # === Phase 2: NF4 UNet, steps t_split..NUM_STEPS-1 ===
    for i, t in enumerate(timesteps[t_split:]):
        noise_pred = denoise_step(
            unet_nf4, latents, t,
            prompt_embeds_combined, added_cond_kwargs,
            GUIDANCE_SCALE
        )
        latents = pipe_fp16.scheduler.step(noise_pred, t, latents).prev_sample

    # === Decode latents to image ===
    # Move NF4 UNet back to CPU to free VRAM during VAE decode
    t0 = time.time()
    unet_nf4.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    move_back_time = time.time() - t0
    if verbose:
        print(f"    NF4 UNet to CPU ({move_back_time:.2f}s). "
              f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    # Decode using VAE (on GPU)
    latents = latents / pipe_fp16.vae.config.scaling_factor
    with torch.inference_mode():
        image = pipe_fp16.vae.decode(latents.to(torch.float32)
                                      if pipe_fp16.vae.dtype == torch.float32
                                      else latents).sample
    image = pipe_fp16.image_processor.postprocess(image, output_type="pil")[0]
    return image, torch.cuda.max_memory_allocated() / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=70)
    ap.add_argument("--t-split", type=int, default=10,
                    help="Step at which to switch from FP16 to NF4 (0..NUM_STEPS)")
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"=== V21 Two-Pass at {args.budget}% budget, t_split={args.t_split} ===")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # === Load FP16 pipeline (the "main" pipeline) ===
    print("Loading SDXL FP16 pipeline...")
    pipe_fp16 = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    )
    pipe_fp16.scheduler = EulerDiscreteScheduler.from_config(pipe_fp16.scheduler.config)
    pipe_fp16.set_progress_bar_config(disable=True)
    # text_encoders and VAE on GPU
    pipe_fp16.text_encoder.to("cuda")
    pipe_fp16.text_encoder_2.to("cuda")
    pipe_fp16.vae.to("cuda")
    # UNet starts on CPU (we'll move to GPU in Phase 1)
    pipe_fp16.unet.to("cpu")
    print(f"  Pipeline loaded. text_encoders+VAE on GPU.")

    fp16_only_vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM with FP16 UNet on CPU: {fp16_only_vram:.2f} GB")

    # === Build NF4 UNet by quantizing a copy ===
    print("Building NF4 UNet...")
    # Load a SECOND UNet, this time we'll quantize it
    pipe_for_nf4 = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    )
    unet_nf4 = pipe_for_nf4.unet
    # Move to GPU temporarily so bitsandbytes can quantize
    unet_nf4.to("cuda")

    fp16_layers, int4_layers, alloc = load_v13_allocation(args.budget)
    n_nf4, n_conv = replace_linear_with_nf4(unet_nf4, int4_layers)
    print(f"  NF4 UNet: {n_nf4} Linear→Linear4bit, {n_conv} Conv2d skipped")
    print(f"  VRAM after NF4 UNet on GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Now move NF4 UNet to CPU (we'll move it back when needed)
    unet_nf4.to("cpu")
    del pipe_for_nf4   # release the second pipeline's other components
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  NF4 UNet moved to CPU. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # === Generate images ===
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)[:args.n_images]
    print(f"Generating {len(prompts)} images, t_split={args.t_split}...")

    variant_name = f"v21_{args.budget}pct_two_pass_t{args.t_split}"
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
        gen = torch.Generator("cuda").manual_seed(SEED + i)
        t_img = time.time()
        try:
            image, peak_vram = two_pass_generate(
                pipe_fp16, unet_nf4,
                item['caption'], args.t_split, gen,
                verbose=args.verbose and i < 2,
            )
            image.save(out_path)
            timings.append(time.time() - t_img)
            peaks.append(peak_vram)
        except Exception as e:
            print(f"  Image {i+1} failed: {e}", flush=True)
            continue

        if (i + 1) % 10 == 0 or i < 5:
            elapsed = (time.time() - t_overall) / 60
            avg_t = sum(timings) / len(timings) if timings else 0
            eta = (len(prompts) - i - 1) * avg_t / 60
            avg_peak = sum(peaks) / len(peaks) if peaks else 0
            print(f"  [{i+1}/{len(prompts)}] {elapsed:.1f}min  "
                  f"avg t/img: {avg_t:.1f}s  avg peak: {avg_peak:.2f} GB  "
                  f"ETA: {eta:.1f}min", flush=True)

    elapsed = time.time() - t_overall
    final_peak = max(peaks) if peaks else torch.cuda.max_memory_allocated() / 1e9
    avg_peak = sum(peaks) / len(peaks) if peaks else 0
    avg_time = sum(timings) / len(timings) if timings else 0

    print(f"\n=== Result ===")
    print(f"  Variant:                {variant_name}")
    print(f"  N images:               {len(timings)}")
    print(f"  Average peak VRAM:      {avg_peak:.2f} GB")
    print(f"  Max peak VRAM:          {final_peak:.2f} GB")
    print(f"  Average time per image: {avg_time:.2f} s")
    print(f"  Total time:             {elapsed/60:.1f} min")

    out_path = BASE / f"vram_measurement_{variant_name}.json"
    with open(out_path, "w") as f:
        json.dump({
            "variant": variant_name,
            "budget_pct": args.budget,
            "t_split": args.t_split,
            "n_images": len(timings),
            "avg_peak_vram_gb": avg_peak,
            "max_peak_vram_gb": final_peak,
            "avg_time_per_image_s": avg_time,
            "total_time_min": elapsed / 60,
            "n_linear_nf4": n_nf4,
            "n_conv2d_skipped_fp16": n_conv,
            "gpu_name": torch.cuda.get_device_name(0),
        }, f, indent=2)
    print(f"  Saved measurement: {out_path}")


if __name__ == "__main__":
    main()
