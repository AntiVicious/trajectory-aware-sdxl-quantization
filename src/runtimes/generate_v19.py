"""
V19: V13 allocation WITH trajectory + V12's quantizer (outlier-preserved + ternary mid).

This is the missing cell in the ablation matrix:
  V13: trajectory + naive INT4
  V14: trajectory + SmoothQuant
  V15: trajectory + GPTQ
  V17: no trajectory + naive INT4
  V18: no trajectory + V12 quantizer
  V19: trajectory + V12 quantizer  <-- this script

Tests whether V12's outlier preservation + ternary mid-block helps when given
the trajectory framework. Likely null based on V14/V15/V18 patterns.

Usage:
    CUDA_VISIBLE_DEVICES=N python -u generate_v19.py --budgets 70 55 40 --n-prompts 5000
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
OUT = BASE / "benchmark_5k"

NUM_STEPS = 20
GUIDANCE_SCALE = 7.5
HEIGHT = 768
WIDTH = 768
SEED = 42


# ===================== V12 quantizers =====================

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
    """V12's outlier-preserved INT4: top-N magnitudes kept fp16."""
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
    """V12's ternary quantizer for mid-block."""
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


# ===================== V13-style trajectory swap =====================

class StepCounter:
    def __init__(self):
        self.current = -1
    def advance(self):
        self.current += 1


class TrajectorySwap:
    """V13-style: layer commits to INT4 at its own threshold timestep.
    V19 uses V12's quantizer instead of naive INT4."""
    def __init__(self, unet, assignments, counter):
        self.unet = unet
        self.assignments = assignments
        self.counter = counter
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
                continue   # FP16 throughout

            # Pre-quantize using V12's quantizer (outlier-preserved or ternary)
            w_orig = module.weight.data
            with torch.no_grad():
                if layer_name.startswith("mid_block"):
                    w_quant = quantize_ternary(w_orig.cpu().float())
                else:
                    w_quant = quantize_outlier_preserved_int4(w_orig.cpu().float())
                w_quant = w_quant.to(w_orig.dtype).to(w_orig.device)

            # Wrap forward: use FP16 for steps < threshold, quantized for steps >= threshold
            counter_ref = self.counter
            t_threshold = threshold
            w_orig_ref = w_orig

            if isinstance(module, nn.Linear):
                bias = module.bias
                def make_linear_fwd(wq=w_quant, wo=w_orig_ref, b=bias,
                                     thr=t_threshold, ctr=counter_ref):
                    def fwd(x):
                        # CFG runs uncond + cond per step, so use floor division
                        step = ctr.current // 2
                        w_use = wq if step >= thr else wo
                        return torch.nn.functional.linear(x, w_use, b)
                    return fwd
                module.forward = make_linear_fwd()
            elif isinstance(module, nn.Conv2d):
                bias = module.bias
                stride = module.stride; padding = module.padding
                dilation = module.dilation; groups = module.groups
                def make_conv_fwd(wq=w_quant, wo=w_orig_ref, b=bias,
                                   s=stride, p=padding, d=dilation, g=groups,
                                   thr=t_threshold, ctr=counter_ref):
                    def fwd(x):
                        step = ctr.current // 2
                        w_use = wq if step >= thr else wo
                        return torch.nn.functional.conv2d(x, w_use, b, s, p, d, g)
                    return fwd
                module.forward = make_conv_fwd()

        n_fp16 = sum(1 for _, t in self.assignments if t >= NUM_STEPS)
        n_int4 = len(self.assignments) - n_fp16
        print(f"  [TrajectorySwap] {n_int4} layers using V12 quantizer with trajectory, "
              f"{n_fp16} layers FP16", flush=True)
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


def load_pipe():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    # No memory-saving features for L40S (48GB)
    return pipe


def generate_variant(pipe, variant_name, assignments, prompts, seed):
    folder = OUT / variant_name
    folder.mkdir(parents=True, exist_ok=True)
    done = len([f for f in folder.iterdir() if f.suffix == '.jpg'])
    if done == len(prompts):
        print(f"  [{variant_name}] Already complete.", flush=True)
        return
    print(f"  [{variant_name}] Starting from {done}/{len(prompts)}", flush=True)

    counter = StepCounter()
    def pre_hook(module, args, kwargs):
        counter.advance()
    handle = pipe.unet.register_forward_pre_hook(pre_hook, with_kwargs=True)

    t0 = time.time()
    try:
        with TrajectorySwap(pipe.unet, assignments, counter):
            for i, item in enumerate(prompts):
                out_path = folder / item['filename']
                if out_path.exists():
                    continue
                counter.current = -1
                gen = torch.Generator("cuda").manual_seed(seed + i)
                with torch.inference_mode():
                    image = pipe(
                        item['caption'],
                        num_inference_steps=NUM_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        generator=gen,
                        height=HEIGHT, width=WIDTH,
                    ).images[0]
                image.save(out_path)
                if (i + 1) % 100 == 0:
                    elapsed = (time.time() - t0) / 60
                    eta = (len(prompts) - i - 1) / max((i + 1) / elapsed, 1e-9)
                    print(f"    [{i+1}/{len(prompts)}] {elapsed:.1f}min ETA {eta:.1f}min",
                          flush=True)
    finally:
        handle.remove()
    print(f"  [{variant_name}] Done in {(time.time()-t0)/60:.1f}min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", nargs="+", type=int, default=[70, 55, 40])
    ap.add_argument("--n-prompts", type=int, default=5000)
    args = ap.parse_args()

    with open(PROMPTS_PATH) as f:
        all_prompts = json.load(f)
    prompts = all_prompts[:args.n_prompts]
    print(f"Loaded {len(prompts)} prompts", flush=True)

    print("Loading SDXL pipeline...", flush=True)
    pipe = load_pipe()
    print(f"Pipeline on {torch.cuda.get_device_name(0)}", flush=True)

    t_total = time.time()
    for pct in args.budgets:
        alloc_path = ALLOC_DIR / f"allocation_{pct}pct_full.json"
        if not alloc_path.exists():
            print(f"  MISSING: {alloc_path}, skipping", flush=True)
            continue

        assignments, alloc = load_v13_allocation(pct)
        variant_name = f"v19_{pct}pct_trajectory_v12quant"

        print(f"\n{'='*60}", flush=True)
        print(f"Generating {variant_name}", flush=True)
        print(f"{'='*60}", flush=True)

        n_fp16 = sum(1 for _, t in assignments if t >= NUM_STEPS)
        n_int4 = len(assignments) - n_fp16
        print(f"  Layers: {len(assignments)}  FP16: {n_fp16}  INT4: {n_int4}",
              flush=True)
        print(f"  Memory: {alloc['actual_MB']:.0f} MB", flush=True)

        generate_variant(pipe, variant_name, assignments, prompts, SEED)

    print(f"\nALL DONE in {(time.time()-t_total)/3600:.1f} hours", flush=True)


if __name__ == "__main__":
    main()