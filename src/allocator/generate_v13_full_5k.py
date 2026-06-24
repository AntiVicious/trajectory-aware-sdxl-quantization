"""
Generate V13 5k benchmark images for 3 Pareto points: 70%, 55%, 40% budgets.

Uses allocations from phase3_allocator_full:
  - allocation_70pct_full.json
  - allocation_55pct_full.json
  - allocation_40pct_full.json

For each allocation, applies MultiLayerSwap to vanilla SDXL and generates
all 5000 images with exact same prompts/seeds as your V1-V12 benchmark.

Matches 5k_benchmark.py conventions exactly:
  - PROMPTS from benchmark_prompts_5k.json
  - seed = 42 + prompt_idx
  - 20 steps, CFG 7.5, 768x768
  - JPG output, filename = item['filename']
  - Saved under benchmark_5k/<variant_name>/

Resumable: existing JPGs are skipped on rerun.

Runtime estimate (A5000): ~4 hours per variant, ~12 hours total.

Usage:
    CUDA_VISIBLE_DEVICES=0 python generate_v13_full_5k.py
    # or specify only some budgets:
    CUDA_VISIBLE_DEVICES=0 python generate_v13_full_5k.py --budgets 70 55
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
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "/home/project/Documents/Maharshi/gem"))
OUT = BASE / "benchmark_5k"
ALLOC_DIR = BASE / "trajectory" / "phase3_allocator_full"
PROMPTS_PATH = BASE / "benchmark_prompts_5k.json"

SEED = 42
NUM_STEPS = 20
HEIGHT = WIDTH = 768
GUIDANCE_SCALE = 7.5

OUT.mkdir(parents=True, exist_ok=True)


# ============ Step-conditional quantization ============

def quantize_int4_per_channel(w: torch.Tensor) -> torch.Tensor:
    orig_shape = w.shape
    orig_dtype = w.dtype
    w2 = w.reshape(w.shape[0], -1).float() if w.ndim == 4 else w.float()
    scale = (w2.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-8)
    q = (w2 / scale).round().clamp(-7, 7)
    return (q * scale).to(orig_dtype).reshape(orig_shape)


class StepCounter:
    def __init__(self):
        self.current = -1
    def advance(self):
        self.current += 1


class MultiLayerSwap:
    """Install step-conditional INT4 on multiple layers. Each (layer, threshold):
       - threshold >= NUM_STEPS: stays FP16
       - threshold < NUM_STEPS: FP16 before step threshold, INT4 from threshold onward.
    """
    def __init__(self, unet, assignments, counter):
        self.unet = unet
        self.assignments = assignments
        self.counter = counter
        self.originals = {}

    def __enter__(self):
        skipped = 0
        for layer_name, threshold in self.assignments:
            try:
                module = self.unet.get_submodule(layer_name)
            except AttributeError:
                skipped += 1
                continue
            self.originals[layer_name] = module.forward
            if threshold >= NUM_STEPS:
                continue
            # Memory optimization: don't duplicate FP16 weights. module.weight
            # IS the FP16 copy, so just quantize once and keep w_quant. At
            # runtime, look up module.weight for the FP16 branch.
            w_quant = quantize_int4_per_channel(module.weight.data).to("cuda")

            if isinstance(module, nn.Linear):
                bias = module.bias
                counter_ref = self.counter
                thr = threshold
                module_ref = module  # capture reference, not tensor
                def make_fn(w_quant_=w_quant, bias_=bias,
                            counter_=counter_ref, thr_=thr, module_=module_ref):
                    def forward(x, *args, **kwargs):
                        w = w_quant_ if counter_.current >= thr_ else module_.weight
                        return F.linear(x, w, bias_)
                    return forward
                module.forward = make_fn()
            elif isinstance(module, nn.Conv2d):
                bias = module.bias
                stride, padding = module.stride, module.padding
                dilation, groups = module.dilation, module.groups
                counter_ref = self.counter
                thr = threshold
                module_ref = module
                def make_fn(w_quant_=w_quant, bias_=bias,
                            stride_=stride, padding_=padding,
                            dilation_=dilation, groups_=groups,
                            counter_=counter_ref, thr_=thr, module_=module_ref):
                    def forward(x, *args, **kwargs):
                        w = w_quant_ if counter_.current >= thr_ else module_.weight
                        return F.conv2d(x, w, bias_, stride_, padding_, dilation_, groups_)
                    return forward
                module.forward = make_fn()
        if skipped:
            print(f"  [MultiLayerSwap] skipped {skipped} unresolvable layer names")
        return self

    def __exit__(self, *args):
        for layer_name in list(self.originals.keys()):
            try:
                module = self.unet.get_submodule(layer_name)
                module.forward = self.originals[layer_name]
            except AttributeError:
                pass
        self.originals = {}
        torch.cuda.empty_cache()


# ============ Allocation loader ============

def load_assignments(alloc_path):
    with open(alloc_path) as f:
        alloc = json.load(f)
    assignments = []
    for a in alloc["allocations"]:
        choice = a["choice"]
        if choice == "FP16":
            thr = NUM_STEPS
        elif choice.startswith("INT4@t"):
            thr = int(choice.split("t")[1])
        else:
            raise ValueError(f"Unknown choice: {choice}")
        assignments.append((a["layer"], thr))
    return assignments, alloc


# ============ Pipeline ============

def load_pipe():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.unet = pipe.unet.to("cuda")
    pipe.vae = pipe.vae.to("cuda")
    pipe.text_encoder = pipe.text_encoder.to("cuda")
    pipe.text_encoder_2 = pipe.text_encoder_2.to("cuda")
    # Match your 5k_benchmark.py settings
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ============ Generation loop ============

def generate_variant(pipe, variant_name, assignments, prompts, seed):
    folder = OUT / variant_name
    folder.mkdir(parents=True, exist_ok=True)

    done = len([f for f in folder.iterdir() if f.suffix == '.jpg'])
    if done == len(prompts):
        print(f"  [{variant_name}] Complete ({done}/{len(prompts)}). Skipping.")
        return

    print(f"  [{variant_name}] Starting from {done}/{len(prompts)}")
    counter = StepCounter()

    # UNet pre-forward hook: with CFG, pipe concats uncond+cond in batch dim,
    # so each denoising step fires UNet once. Counter increments once per step.
    def pre_hook(module, args, kwargs):
        counter.advance()
    handle = pipe.unet.register_forward_pre_hook(pre_hook, with_kwargs=True)

    t0 = time.time()
    try:
        with MultiLayerSwap(pipe.unet, assignments, counter):
            for i, item in enumerate(prompts):
                out_path = folder / item['filename']
                if out_path.exists():
                    continue

                counter.current = -1  # reset per prompt
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
                    rate = (i + 1 - done) / elapsed if elapsed > 0 else 0
                    remaining = (len(prompts) - i - 1) / rate if rate > 0 else 0
                    print(f"  [{variant_name}] {i+1}/{len(prompts)}  "
                          f"{elapsed:.0f}min elapsed, ETA {remaining:.0f}min")
    finally:
        handle.remove()

    elapsed = (time.time() - t0) / 60
    final = len([f for f in folder.iterdir() if f.suffix == '.jpg'])
    print(f"  [{variant_name}] Done: {final}/{len(prompts)} in {elapsed:.1f} min")


# ============ Main ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", nargs="+", type=int, default=[70, 55, 40],
                    help="Which budget levels to generate (e.g. --budgets 70 55 40)")
    args = ap.parse_args()

    # Load prompts
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts")

    # Confirm V1 exists (sanity)
    if not (OUT / "v1_fp16").exists():
        print(f"WARNING: {OUT/'v1_fp16'} not found. Still proceeding but no "
              f"V1 baseline for comparison.")

    # Load pipeline once — we'll re-inject assignments per variant
    print("Loading SDXL pipeline once...")
    pipe = load_pipe()
    print(f"Pipeline on {torch.cuda.get_device_name(0)}")

    # Build variant plan
    variants = []
    for pct in args.budgets:
        alloc_path = ALLOC_DIR / f"allocation_{pct}pct_full.json"
        if not alloc_path.exists():
            print(f"  MISSING: {alloc_path}")
            continue
        assignments, alloc = load_assignments(alloc_path)
        variant_name = f"v13_{pct}pct_trajectory"
        n_fp16 = sum(1 for _, t in assignments if t >= NUM_STEPS)
        n_int4 = len(assignments) - n_fp16
        print(f"\n=== {variant_name} ===")
        print(f"  Total layers: {len(assignments)}")
        print(f"  FP16: {n_fp16}  INT4: {n_int4}")
        print(f"  Memory: {alloc['actual_MB']:.0f} MB "
              f"({alloc['budget_pct']}% of FP16)")
        print(f"  Predicted LPIPS sum: {alloc['total_lpips_pred']:.3e}")
        variants.append((variant_name, assignments, alloc))

    if not variants:
        print("No allocations to generate.")
        return

    print(f"\nWill generate {len(variants)} variants × {len(prompts)} images")
    t_total = time.time()
    for variant_name, assignments, alloc in variants:
        print(f"\n{'='*60}")
        print(f"Generating {variant_name}")
        print(f"{'='*60}")
        generate_variant(pipe, variant_name, assignments, prompts, SEED)

    total_hr = (time.time() - t_total) / 3600
    print(f"\n{'='*60}")
    print(f"ALL VARIANTS DONE in {total_hr:.1f} hours")
    print(f"{'='*60}")
    for variant_name, _, _ in variants:
        folder = OUT / variant_name
        n = len([f for f in folder.iterdir() if f.suffix == '.jpg'])
        print(f"  {variant_name}: {n} images in {folder}")
    print(f"\nNext: run 5k_score_v13_full.py to score against V1-V12")


if __name__ == "__main__":
    main()