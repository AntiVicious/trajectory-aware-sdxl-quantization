"""
Level 1 VRAM measurement: peak GPU memory during inference for V1, V13-70, V19-70.

The point: the thesis claims theoretical memory savings of 30-45%. This script
measures the *actual* GPU memory used during generation. The "Level 1" approach
is straightforward: load the pipeline, run N images, record torch.cuda.max_memory_allocated().

Run this as separate processes so each measurement is clean (no leftover state).

Usage:
    CUDA_VISIBLE_DEVICES=1 python -u measure_vram.py --variant v1
    CUDA_VISIBLE_DEVICES=1 python -u measure_vram.py --variant v13_70
    CUDA_VISIBLE_DEVICES=1 python -u measure_vram.py --variant v19_70

    # Or do all three with a small driver script
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

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
PROMPTS_PATH = BASE / "benchmark_prompts_5k.json"
ALLOC_DIR = BASE / "trajectory" / "phase3_allocator_full"

NUM_STEPS = 20
GUIDANCE_SCALE = 7.5
HEIGHT = 768
WIDTH = 768
SEED = 42

# Number of images to generate for the measurement.
# More images = more reliable peak (some allocations only happen on certain prompts).
N_TEST_IMAGES = 5


# -------------- Quantizers (matching V13 and V19) --------------

def quantize_int4_per_channel_naive(w):
    qmax = 7
    if w.ndim == 4:
        scales = w.abs().amax(dim=(1, 2, 3), keepdim=True) / qmax
    else:
        scales = w.abs().amax(dim=1, keepdim=True) / qmax
    scales = scales.clamp(min=1e-8)
    q = (w / scales).round().clamp(-qmax, qmax)
    return q * scales


def quantize_outlier_preserved_int4(w, outlier_pct=0.1):
    flat = w.abs().flatten()
    n = flat.numel()
    k = max(1, int(n * outlier_pct / 100.0))
    if k >= n:
        return w
    threshold = torch.topk(flat, k, largest=True).values.min()
    outlier_mask = w.abs() >= threshold
    w_for_quant = w.clone()
    w_for_quant[outlier_mask] = 0.0
    w_quant = quantize_int4_per_channel_naive(w_for_quant)
    w_quant[outlier_mask] = w[outlier_mask]
    return w_quant


def quantize_ternary(w):
    if w.ndim == 4:
        scales = w.abs().mean(dim=(1, 2, 3), keepdim=True)
    else:
        scales = w.abs().mean(dim=1, keepdim=True)
    scales = scales.clamp(min=1e-8)
    threshold = scales * 0.7
    q = torch.zeros_like(w)
    q[w > threshold] = 1.0
    q[w < -threshold] = -1.0
    return q * scales


# -------------- Trajectory swap (used by V13 and V19) --------------

class StepCounter:
    def __init__(self): self.current = -1
    def advance(self): self.current += 1


def load_v13_allocation(pct):
    p = ALLOC_DIR / f"allocation_{pct}pct_full.json"
    with open(p) as f:
        alloc = json.load(f)
    assignments = []
    for a in alloc["allocations"]:
        if a["choice"] == "FP16":
            t = NUM_STEPS
        else:
            t = int(a["choice"].split("t")[1])
        assignments.append((a["layer"], t))
    return assignments, alloc


class TrajectorySwap:
    """Trajectory-aware weight swap. quantizer_fn determines V13 vs V19.
    For mid-block layers, mid_block_quantizer is used if provided."""
    def __init__(self, unet, assignments, counter, quantizer_fn, mid_block_quantizer=None):
        self.unet = unet
        self.assignments = assignments
        self.counter = counter
        self.quantizer_fn = quantizer_fn
        self.mid_block_quantizer = mid_block_quantizer
        self.originals = {}

    def __enter__(self):
        for layer_name, threshold in self.assignments:
            try:
                module = self.unet.get_submodule(layer_name)
            except AttributeError:
                continue
            if not hasattr(module, 'weight'):
                continue
            self.originals[layer_name] = module.forward
            if threshold >= NUM_STEPS:
                continue

            w_orig = module.weight.data
            with torch.no_grad():
                if (self.mid_block_quantizer is not None and
                        layer_name.startswith("mid_block")):
                    w_quant = self.mid_block_quantizer(w_orig.cpu().float())
                else:
                    w_quant = self.quantizer_fn(w_orig.cpu().float())
                w_quant = w_quant.to(w_orig.dtype).to(w_orig.device)

            counter_ref = self.counter
            t_thr = threshold
            w_orig_ref = w_orig

            if isinstance(module, nn.Linear):
                bias = module.bias
                def make_lin(wq=w_quant, wo=w_orig_ref, b=bias, thr=t_thr, ctr=counter_ref):
                    def fwd(x):
                        step = ctr.current // 2
                        return torch.nn.functional.linear(
                            x, wq if step >= thr else wo, b)
                    return fwd
                module.forward = make_lin()
            elif isinstance(module, nn.Conv2d):
                bias = module.bias; stride = module.stride; padding = module.padding
                dilation = module.dilation; groups = module.groups
                def make_conv(wq=w_quant, wo=w_orig_ref, b=bias,
                               s=stride, p=padding, d=dilation, g=groups,
                               thr=t_thr, ctr=counter_ref):
                    def fwd(x):
                        step = ctr.current // 2
                        return torch.nn.functional.conv2d(
                            x, wq if step >= thr else wo, b, s, p, d, g)
                    return fwd
                module.forward = make_conv()
        return self

    def __exit__(self, *args):
        for name, orig in self.originals.items():
            try:
                m = self.unet.get_submodule(name)
                m.forward = orig
            except AttributeError:
                pass
        self.originals = {}
        torch.cuda.empty_cache()


# -------------- Measurement --------------

def measure_pipeline_loaded_vram():
    """VRAM after loading SDXL pipeline, before any inference."""
    return torch.cuda.max_memory_allocated() / 1e9   # in GB


def run_measurement(variant_name, swap_context, prompts):
    """Run N test images with the given swap context. Return peak VRAM."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    baseline_alloc = torch.cuda.memory_allocated() / 1e9

    counter = StepCounter()
    if swap_context is not None:
        # Need to register step counter hook on the UNet
        pipe_unet = swap_context.unet

        def pre_hook(module, args, kwargs):
            counter.advance()
        handle = pipe_unet.register_forward_pre_hook(pre_hook, with_kwargs=True)

    t_start = time.time()
    try:
        ctx = swap_context if swap_context is not None else None
        if ctx is not None:
            ctx_entered = ctx.__enter__()
        for i in range(N_TEST_IMAGES):
            counter.current = -1
            gen = torch.Generator("cuda").manual_seed(SEED + i)
            with torch.inference_mode():
                _ = pipe(
                    prompts[i]['caption'],
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=gen,
                    height=HEIGHT, width=WIDTH,
                ).images[0]
            print(f"  [{variant_name}] image {i+1}/{N_TEST_IMAGES} done "
                  f"(running peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB)",
                  flush=True)
        if ctx is not None:
            ctx.__exit__(None, None, None)
    finally:
        if swap_context is not None:
            handle.remove()

    elapsed = time.time() - t_start
    peak = torch.cuda.max_memory_allocated() / 1e9
    return {
        "variant": variant_name,
        "peak_vram_gb": peak,
        "baseline_alloc_gb": baseline_alloc,
        "n_images": N_TEST_IMAGES,
        "time_per_image_s": elapsed / N_TEST_IMAGES,
    }


def main():
    
    # 1. Declare the global variable at the very top of the function
    global N_TEST_IMAGES

    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["v1", "v13_70", "v13_55", "v13_40",
                             "v19_70", "v19_55", "v19_40"])
    # 2. Now it is safe to use it here
    ap.add_argument("--n-images", type=int, default=N_TEST_IMAGES)
    args = ap.parse_args()

    # 3. Update the global variable with the parsed arguments
    N_TEST_IMAGES = args.n_images

    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)[:args.n_images]

    # ... rest of your code remains exactly the same ...

    print(f"=== {args.variant} ===")
    print(f"Loading SDXL pipeline...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    init_alloc = torch.cuda.memory_allocated() / 1e9

    global pipe
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    after_load = torch.cuda.memory_allocated() / 1e9
    after_load_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Pipeline loaded.")
    print(f"  VRAM allocated: {after_load:.2f} GB  (peak so far: {after_load_peak:.2f} GB)")

    # Build swap context based on variant
    if args.variant == "v1":
        swap = None
    else:
        # Parse v13_70 / v19_55 etc.
        parts = args.variant.split("_")
        kind = parts[0]    # v13 or v19
        budget = int(parts[1])
        assignments, alloc = load_v13_allocation(budget)
        counter_for_swap = StepCounter()
        if kind == "v13":
            swap = TrajectorySwap(pipe.unet, assignments, counter_for_swap,
                                   quantize_int4_per_channel_naive, mid_block_quantizer=None)
        elif kind == "v19":
            swap = TrajectorySwap(pipe.unet, assignments, counter_for_swap,
                                   quantize_outlier_preserved_int4,
                                   mid_block_quantizer=quantize_ternary)
        # Replace counter ref so the StepCounter inside swap is the one we use for the run hook
        global counter_ref_for_run
        counter_ref_for_run = counter_for_swap

    # Reset peak before measuring
    torch.cuda.reset_peak_memory_stats()
    after_reset = torch.cuda.memory_allocated() / 1e9
    print(f"  Reset peak. Baseline: {after_reset:.2f} GB")

    # Run measurement (handles its own swap context entry)
    if swap is None:
        # V1 baseline: run without swap context
        counter = StepCounter()
        t0 = time.time()
        for i in range(args.n_images):
            counter.current = -1
            gen = torch.Generator("cuda").manual_seed(SEED + i)
            with torch.inference_mode():
                _ = pipe(
                    prompts[i]['caption'],
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=gen,
                    height=HEIGHT, width=WIDTH,
                ).images[0]
            print(f"  [v1] image {i+1}/{args.n_images} done "
                  f"(running peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB)",
                  flush=True)
        elapsed = time.time() - t0
    else:
        # V13/V19: enter swap context first
        counter = swap.counter
        def pre_hook(module, args_, kwargs_):
            counter.advance()
        handle = pipe.unet.register_forward_pre_hook(pre_hook, with_kwargs=True)

        t0 = time.time()
        try:
            with swap:
                for i in range(args.n_images):
                    counter.current = -1
                    gen = torch.Generator("cuda").manual_seed(SEED + i)
                    with torch.inference_mode():
                        _ = pipe(
                            prompts[i]['caption'],
                            num_inference_steps=NUM_STEPS,
                            guidance_scale=GUIDANCE_SCALE,
                            generator=gen,
                            height=HEIGHT, width=WIDTH,
                        ).images[0]
                    print(f"  [{args.variant}] image {i+1}/{args.n_images} done "
                          f"(running peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB)",
                          flush=True)
            elapsed = time.time() - t0
        finally:
            handle.remove()

    final_peak = torch.cuda.max_memory_allocated() / 1e9
    final_alloc = torch.cuda.memory_allocated() / 1e9

    print(f"\n=== Result for {args.variant} ===")
    print(f"  Pipeline-load alloc:  {after_load:.2f} GB")
    print(f"  Peak during inference: {final_peak:.2f} GB")
    print(f"  Final allocated:       {final_alloc:.2f} GB")
    print(f"  Time per image:        {elapsed/args.n_images:.2f} s")

    # Save to a per-variant JSON
    out_path = BASE / f"vram_measurement_{args.variant}.json"
    with open(out_path, "w") as f:
        json.dump({
            "variant": args.variant,
            "n_images": args.n_images,
            "pipeline_load_gb": after_load,
            "pipeline_load_peak_gb": after_load_peak,
            "inference_peak_gb": final_peak,
            "final_alloc_gb": final_alloc,
            "time_per_image_s": elapsed / args.n_images,
            "gpu_name": torch.cuda.get_device_name(0),
        }, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
