"""
VRAM measurement with Fix 1 applied.

Fix 1: For layers that commit at t=0 (no FP16 needed at any step), replace
       module.weight.data with the quantized tensor. This frees the FP16 copy.

Expected effect: significant VRAM reduction for the layers that commit at t=0.
At V13-70, this is ~30-50% of quantized layers.

Note: V13's allocation has many layers committing at t=10/15/19, not t=0.
For those, both FP16 and INT4 are still held. So this Fix 1 only partially
realizes the theoretical savings.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -u measure_vram_fix1.py --variant v13_70
    CUDA_VISIBLE_DEVICES=1 python -u measure_vram_fix1.py --variant v19_70
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

NUM_STEPS = 20
GUIDANCE_SCALE = 7.5
HEIGHT = 768
WIDTH = 768
SEED = 42


# Quantizers
def quantize_int4_per_channel_naive(w):
    qmax = 7
    if w.ndim == 4:
        scales = w.abs().amax(dim=(1, 2, 3), keepdim=True) / qmax
    else:
        scales = w.abs().amax(dim=1, keepdim=True) / qmax
    scales = scales.clamp(min=1e-8)
    return (w / scales).round().clamp(-qmax, qmax) * scales


def quantize_outlier_preserved_int4(w, outlier_pct=0.1):
    flat = w.abs().flatten()
    n = flat.numel()
    k = max(1, int(n * outlier_pct / 100.0))
    if k >= n: return w
    threshold = torch.topk(flat, k, largest=True).values.min()
    mask = w.abs() >= threshold
    w_for_quant = w.clone()
    w_for_quant[mask] = 0.0
    w_quant = quantize_int4_per_channel_naive(w_for_quant)
    w_quant[mask] = w[mask]
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


class StepCounter:
    def __init__(self): self.current = -1
    def advance(self): self.current += 1


def load_v13_allocation(pct):
    p = ALLOC_DIR / f"allocation_{pct}pct_full.json"
    with open(p) as f: alloc = json.load(f)
    assignments = []
    for a in alloc["allocations"]:
        if a["choice"] == "FP16": t = NUM_STEPS
        else: t = int(a["choice"].split("t")[1])
        assignments.append((a["layer"], t))
    return assignments


def apply_swap_with_fix1(unet, assignments, quantizer_fn, counter, mid_quant=None):
    """Apply trajectory swap. For layers committing at t=0, REPLACE module.weight.data
    with the quantized tensor (frees FP16 copy). For t>0 layers, keep both as before."""
    originals = {}
    n_replaced = 0
    n_kept_both = 0
    n_fp16 = 0

    for layer_name, threshold in assignments:
        try:
            module = unet.get_submodule(layer_name)
        except AttributeError:
            continue
        if not hasattr(module, 'weight'):
            continue

        originals[layer_name] = (module.forward, None)

        if threshold >= NUM_STEPS:
            n_fp16 += 1
            continue   # FP16 throughout, no swap needed

        w_orig = module.weight.data
        with torch.no_grad():
            if mid_quant is not None and layer_name.startswith("mid_block"):
                w_quant = mid_quant(w_orig.cpu().float())
            else:
                w_quant = quantizer_fn(w_orig.cpu().float())
            w_quant = w_quant.contiguous().to(w_orig.dtype).to(w_orig.device)

        if threshold == 0:
            # FIX 1: free the FP16 copy by replacing module.weight.data
            # No need for closure-based swap, no need for FP16 fallback
            module.weight.data = w_quant   # FP16 weight is now eligible for GC
            n_replaced += 1
        else:
            # t > 0: keep both, swap with closure as before
            counter_ref = counter
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
                originals[layer_name] = (module.forward, None)  # update
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
            n_kept_both += 1

    print(f"  [Swap-Fix1] {n_replaced} layers t=0 (FP16 freed), "
          f"{n_kept_both} layers t>0 (both held), "
          f"{n_fp16} layers FP16 throughout", flush=True)

    # Force a GC and clear cache to actually free the freed FP16 weights
    torch.cuda.empty_cache()
    return originals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["v13_70", "v13_55", "v13_40",
                             "v19_70", "v19_55", "v19_40"])
    ap.add_argument("--n-images", type=int, default=5)
    args = ap.parse_args()

    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)[:args.n_images]

    parts = args.variant.split("_")
    kind = parts[0]
    budget = int(parts[1])

    print(f"=== {args.variant} (with Fix 1) ===")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    after_load = torch.cuda.memory_allocated() / 1e9
    print(f"  Pipeline loaded. VRAM: {after_load:.2f} GB")

    # Apply swap with Fix 1
    assignments = load_v13_allocation(budget)
    counter = StepCounter()
    if kind == "v13":
        apply_swap_with_fix1(pipe.unet, assignments,
                             quantize_int4_per_channel_naive, counter)
    elif kind == "v19":
        apply_swap_with_fix1(pipe.unet, assignments,
                             quantize_outlier_preserved_int4, counter,
                             mid_quant=quantize_ternary)

    after_swap = torch.cuda.memory_allocated() / 1e9
    print(f"  After swap+Fix1: {after_swap:.2f} GB (delta: {after_swap-after_load:+.2f} GB)")

    # Reset peak
    torch.cuda.reset_peak_memory_stats()

    # Step counter hook
    def pre_hook(module, args_, kwargs_):
        counter.advance()
    handle = pipe.unet.register_forward_pre_hook(pre_hook, with_kwargs=True)

    t0 = time.time()
    try:
        for i in range(args.n_images):
            counter.current = -1
            gen = torch.Generator("cuda").manual_seed(SEED + i)
            with torch.inference_mode():
                _ = pipe(
                    prompts[i]['caption'],
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=gen, height=HEIGHT, width=WIDTH,
                ).images[0]
            print(f"  [{args.variant}] image {i+1}/{args.n_images} done "
                  f"(running peak: {torch.cuda.max_memory_allocated()/1e9:.2f} GB)",
                  flush=True)
    finally:
        handle.remove()

    elapsed = time.time() - t0
    final_peak = torch.cuda.max_memory_allocated() / 1e9
    final_alloc = torch.cuda.memory_allocated() / 1e9

    print(f"\n=== Result for {args.variant} (Fix 1) ===")
    print(f"  Pipeline-load alloc:    {after_load:.2f} GB")
    print(f"  After swap+Fix1:        {after_swap:.2f} GB")
    print(f"  Peak during inference:  {final_peak:.2f} GB")
    print(f"  Time per image:         {elapsed/args.n_images:.2f} s")

    out_path = BASE / f"vram_measurement_{args.variant}_fix1.json"
    with open(out_path, "w") as f:
        json.dump({
            "variant": args.variant + "_fix1",
            "n_images": args.n_images,
            "pipeline_load_gb": after_load,
            "after_swap_gb": after_swap,
            "inference_peak_gb": final_peak,
            "final_alloc_gb": final_alloc,
            "time_per_image_s": elapsed / args.n_images,
            "gpu_name": torch.cuda.get_device_name(0),
        }, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
