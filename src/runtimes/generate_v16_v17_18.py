"""
V16 / V17 / V18 ablation generation.

Runs three ablations against V13:
  V16: random allocation, naive INT4 from t=0 always
       — tests if sensitivity-aware allocation is needed at all
  V17: V13's allocation, naive INT4 from t=0 always (no trajectory dim)
       — tests if trajectory dimension contributes beyond static allocation
  V18: V13's allocation, V12's quantizer (outlier-preserved + ternary mid-block)
       — tests if better per-layer quantization helps once allocation is fixed

Generation uses the same prompts (5000 from benchmark_prompts_5k.json) and
seeds (42 + i) as V1..V15 so results are directly comparable.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u generate_v16_v17_v18.py \
        --variants v16 v17 v18 \
        --budgets 70 55 40 \
        --n-prompts 5000
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

BASE = Path("/home/project/Documents/Maharshi/gem")
PROMPTS_PATH = BASE / "benchmark_prompts_5k.json"
ALLOC_DIR = BASE / "trajectory" / "phase3_allocator_full"
LAYERS_JSON = BASE / "trajectory" / "phase1_atlas_full" / "layers_full.json"
OUT = BASE / "benchmark_5k"

NUM_STEPS = 20
GUIDANCE_SCALE = 7.5
HEIGHT = 768
WIDTH = 768
SEED = 42

# ===================== Quantizer implementations =====================

def quantize_int4_per_channel_naive(w):
    """Per-output-channel symmetric INT4 (V2/V13's quantizer)."""
    qmax = 7
    if w.ndim == 4:  # Conv2d [C_out, C_in, kH, kW]
        scales = w.abs().amax(dim=(1, 2, 3), keepdim=True) / qmax
    else:  # Linear [C_out, C_in]
        scales = w.abs().amax(dim=1, keepdim=True) / qmax
    scales = scales.clamp(min=1e-8)
    q = (w / scales).round().clamp(-qmax, qmax)
    return q * scales


def quantize_outlier_preserved_int4(w, outlier_pct=0.1):
    """V12's outlier-preserved INT4: keep top-N magnitude weights as fp16,
    quantize the rest. outlier_pct is the percentage to preserve (default 0.1%)."""
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
    """V12's ternary quantizer for mid-block: weights -> {-1, 0, 1} per-channel scale."""
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


# ===================== Allocation helpers =====================

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


def make_random_allocation(layers_meta, budget_mb, seed=12345):
    """V16's random allocation: randomly choose which layers to keep FP16
    until cumulative FP16 memory matches the target FP16 fraction.
    All other layers become INT4@t=0."""
    rng = np.random.default_rng(seed)
    int4_total = sum(L["n_params"] * 0.625 / 1e6 for L in layers_meta)
    promotion_surplus = [(L["name"], L["n_params"] * (2.0 - 0.625) / 1e6,
                          L["n_params"]) for L in layers_meta]
    needed = budget_mb - int4_total
    if needed <= 0:
        return [(L["name"], 0) for L in layers_meta]

    shuffled = promotion_surplus[:]
    rng.shuffle(shuffled)
    fp16_layers = set()
    accumulated = 0.0
    for name, surplus, _ in shuffled:
        if accumulated + surplus <= needed * 1.01:
            fp16_layers.add(name)
            accumulated += surplus
        if accumulated >= needed * 0.99:
            break

    assignments = []
    for L in layers_meta:
        if L["name"] in fp16_layers:
            assignments.append((L["name"], NUM_STEPS))
        else:
            assignments.append((L["name"], 0))
    return assignments


# ===================== Step counter and swap context =====================

class StepCounter:
    def __init__(self):
        self.current = -1
    def advance(self):
        self.current += 1


class StaticSwap:
    """Swaps a layer's forward to use a quantized weight from step 0 onward.
    Used by V16, V17, V18 (no trajectory — full quantization always)."""
    def __init__(self, unet, assignments, counter, quantizer_fn,
                 mid_block_quantizer=None):
        self.unet = unet
        self.assignments = assignments
        self.counter = counter
        self.quantizer_fn = quantizer_fn
        self.mid_block_quantizer = mid_block_quantizer
        self.originals = {}

    def __enter__(self):
        n_quant = 0
        n_fp16 = 0
        n_mid_special = 0
        for layer_name, threshold in self.assignments:
            try:
                module = self.unet.get_submodule(layer_name)
            except AttributeError:
                continue
            if not hasattr(module, 'weight'):
                continue
            self.originals[layer_name] = module.forward
            if threshold >= NUM_STEPS:
                n_fp16 += 1
                continue

            w_orig = module.weight.data
            with torch.no_grad():
                if (self.mid_block_quantizer is not None and
                        layer_name.startswith("mid_block")):
                    w_quant = self.mid_block_quantizer(w_orig.cpu().float())
                    n_mid_special += 1
                else:
                    w_quant = self.quantizer_fn(w_orig.cpu().float())
                    n_quant += 1
                w_quant = w_quant.to(w_orig.dtype).to(w_orig.device)

            if isinstance(module, nn.Linear):
                bias = module.bias
                def make_linear_fwd(wq=w_quant, b=bias):
                    def fwd(x):
                        return torch.nn.functional.linear(x, wq, b)
                    return fwd
                module.forward = make_linear_fwd()
            elif isinstance(module, nn.Conv2d):
                bias = module.bias
                stride = module.stride
                padding = module.padding
                dilation = module.dilation
                groups = module.groups
                def make_conv_fwd(wq=w_quant, b=bias, s=stride, p=padding,
                                  d=dilation, g=groups):
                    def fwd(x):
                        return torch.nn.functional.conv2d(x, wq, b, s, p, d, g)
                    return fwd
                module.forward = make_conv_fwd()

        print(f"  [Swap] {n_quant} layers using main quantizer, "
              f"{n_mid_special} mid-block layers using special quantizer, "
              f"{n_fp16} layers FP16", flush=True)
        return self

    def __exit__(self, *args):
        for name, orig_forward in self.originals.items():
            try:
                m = self.unet.get_submodule(name)
                m.forward = orig_forward
            except AttributeError:
                pass
        self.originals = {}
        torch.cuda.empty_cache()


# ===================== Pipeline loading =====================

def load_pipe():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    return pipe


# ===================== Generation loop =====================

def generate_variant(pipe, variant_name, assignments, prompts, seed,
                     quantizer_fn, mid_block_quantizer=None):
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
        with StaticSwap(pipe.unet, assignments, counter,
                        quantizer_fn, mid_block_quantizer):
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


# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["v16", "v17", "v18"],
                    choices=["v16", "v17", "v18"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[70, 55, 40])
    ap.add_argument("--n-prompts", type=int, default=5000,
                    help="Number of prompts to generate per variant (default 5000). "
                         "For ablations, 1000 is usually sufficient.")
    args = ap.parse_args()

    with open(PROMPTS_PATH) as f:
        all_prompts = json.load(f)
    prompts = all_prompts[:args.n_prompts]
    print(f"Loaded {len(prompts)} prompts (capped from {len(all_prompts)})", flush=True)

    with open(LAYERS_JSON) as f:
        layers_meta = json.load(f)["layers"]

    print("Loading SDXL pipeline...", flush=True)
    pipe = load_pipe()
    print(f"Pipeline on {torch.cuda.get_device_name(0)}", flush=True)

    t_total = time.time()
    for variant in args.variants:
        for pct in args.budgets:
            alloc_path = ALLOC_DIR / f"allocation_{pct}pct_full.json"
            if not alloc_path.exists():
                print(f"  MISSING: {alloc_path}, skipping", flush=True)
                continue

            if variant == "v16":
                with open(alloc_path) as f:
                    target_mb = json.load(f)["actual_MB"]
                assignments = make_random_allocation(
                    layers_meta, target_mb, seed=12345 + pct)
                quantizer_fn = quantize_int4_per_channel_naive
                mid_block_quantizer = None
                variant_name = f"v16_{pct}pct_random_alloc"

            elif variant == "v17":
                v13_assignments, _ = load_v13_allocation(pct)
                assignments = [
                    (name, NUM_STEPS if t >= NUM_STEPS else 0)
                    for name, t in v13_assignments
                ]
                quantizer_fn = quantize_int4_per_channel_naive
                mid_block_quantizer = None
                variant_name = f"v17_{pct}pct_no_trajectory"

            elif variant == "v18":
                v13_assignments, _ = load_v13_allocation(pct)
                assignments = [
                    (name, NUM_STEPS if t >= NUM_STEPS else 0)
                    for name, t in v13_assignments
                ]
                quantizer_fn = quantize_outlier_preserved_int4
                mid_block_quantizer = quantize_ternary
                variant_name = f"v18_{pct}pct_v12_quantizer"

            print(f"\n{'='*60}", flush=True)
            print(f"Generating {variant_name}", flush=True)
            print(f"{'='*60}", flush=True)

            n_fp16 = sum(1 for _, t in assignments if t >= NUM_STEPS)
            n_int4 = len(assignments) - n_fp16
            print(f"  Layers: {len(assignments)}  FP16: {n_fp16}  INT4: {n_int4}",
                  flush=True)

            generate_variant(pipe, variant_name, assignments, prompts, SEED,
                             quantizer_fn, mid_block_quantizer)

    print(f"\nALL DONE in {(time.time()-t_total)/3600:.1f} hours", flush=True)


if __name__ == "__main__":
    main()