"""
V22 with --commit-floor: bumps any layer's commit threshold up to at least the floor.

If commit_floor=10, all layers commit at t=10 or later. This effectively
makes V22 equivalent to V21's coarse rule but uses the per-layer mechanism.

Tests whether V22's quality drop at 40% is due to the aggressive t=0/t=5
commits in V13-40's allocation. If V22-40 with floor=10 matches V21-40 (~0.02),
we've isolated the issue.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -u v22_with_floor.py --budget 40 --commit-floor 10 --n-images 200
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


class HotSwapLinear(nn.Module):
    def __init__(self, original_linear, commit_step, step_counter):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.commit_step = commit_step
        self.step_counter = step_counter

        self._fp16_weight = original_linear.weight.data.clone().contiguous()

        from bitsandbytes.nn import Linear4bit, Params4bit
        self._nf4_layer = Linear4bit(
            self.in_features, self.out_features,
            bias=(original_linear.bias is not None),
            quant_type="nf4",
            compute_dtype=torch.float16,
        )
        self._nf4_layer.weight = Params4bit(
            original_linear.weight.data.clone().contiguous(),
            requires_grad=False, quant_type="nf4"
        )
        if original_linear.bias is not None:
            self._nf4_layer.bias = nn.Parameter(
                original_linear.bias.data.clone(), requires_grad=False)
        self._nf4_layer = self._nf4_layer.to("cuda")

        if original_linear.bias is not None:
            self._fp16_bias = original_linear.bias.data.clone()
        else:
            self._fp16_bias = None

    def forward(self, x):
        step = self.step_counter.current // 2
        if step < self.commit_step:
            return F.linear(x, self._fp16_weight, self._fp16_bias)
        else:
            return self._nf4_layer(x)


class StepCounter:
    def __init__(self): self.current = -1
    def advance(self): self.current += 1
    def reset(self): self.current = -1


def load_v13_allocation_with_floor(pct, floor):
    """Load V13's allocation but bump any commit < floor up to floor."""
    p = ALLOC_DIR / f"allocation_{pct}pct_full.json"
    with open(p) as f:
        alloc = json.load(f)
    layer_to_threshold = {}
    n_bumped = 0
    n_orig = {0: 0, 5: 0, 10: 0, 15: 0, 19: 0}
    n_after = {0: 0, 5: 0, 10: 0, 15: 0, 19: 0, NUM_STEPS: 0}
    for a in alloc["allocations"]:
        if a["choice"] == "FP16":
            t = NUM_STEPS
        else:
            t = int(a["choice"].split("t")[1])
            n_orig[t] += 1
            if t < floor:
                t = floor
                n_bumped += 1
        layer_to_threshold[a["layer"]] = t
        n_after[t] = n_after.get(t, 0) + 1
    print(f"  Original commits: {n_orig}")
    print(f"  After floor={floor}: {n_after}")
    print(f"  Bumped {n_bumped} layers to t>={floor}")
    return layer_to_threshold


def install_hot_swap_layers(unet, layer_to_threshold, step_counter):
    n_replaced = 0
    n_skipped_conv = 0
    n_fp16 = 0
    layer_paths = {name: None for name in layer_to_threshold}
    for name, module in unet.named_modules():
        if name in layer_paths:
            layer_paths[name] = module

    for name, threshold in layer_to_threshold.items():
        module = layer_paths.get(name)
        if module is None:
            continue
        if threshold >= NUM_STEPS:
            n_fp16 += 1
            continue
        if isinstance(module, nn.Conv2d):
            n_skipped_conv += 1
            continue
        if not isinstance(module, nn.Linear):
            continue

        parts = name.split(".")
        parent_name = ".".join(parts[:-1])
        attr_name = parts[-1]
        parent = unet.get_submodule(parent_name) if parent_name else unet

        new_module = HotSwapLinear(module, threshold, step_counter)
        setattr(parent, attr_name, new_module)
        n_replaced += 1

    print(f"  [HotSwap] {n_replaced} HotSwapLinear, "
          f"{n_skipped_conv} Conv2d FP16, "
          f"{n_fp16} explicit FP16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--commit-floor", type=int, default=10,
                    help="Minimum commit step (any layer below this gets bumped up)")
    ap.add_argument("--n-images", type=int, default=200)
    args = ap.parse_args()

    print(f"=== V22 + floor at {args.budget}% budget, floor={args.commit_floor} ===")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    layer_to_threshold = load_v13_allocation_with_floor(args.budget, args.commit_floor)
    step_counter = StepCounter()
    install_hot_swap_layers(pipe.unet, layer_to_threshold, step_counter)

    after_setup = torch.cuda.memory_allocated() / 1e9
    print(f"  After hot-swap setup. VRAM: {after_setup:.2f} GB")

    def pre_hook(module, args_, kwargs_):
        step_counter.advance()
    handle = pipe.unet.register_forward_pre_hook(pre_hook, with_kwargs=True)

    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)[:args.n_images]
    print(f"\nGenerating {len(prompts)} images...")

    variant_name = f"v22_{args.budget}pct_floor{args.commit_floor}"
    folder = OUT / variant_name
    folder.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    peaks = []
    timings = []
    t_overall = time.time()

    try:
        for i, item in enumerate(prompts):
            out_path = folder / item['filename']
            if out_path.exists():
                continue
            step_counter.reset()
            gen = torch.Generator("cuda").manual_seed(SEED + i)
            t_img = time.time()
            with torch.inference_mode():
                image = pipe(
                    item['caption'],
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    generator=gen,
                    height=HEIGHT, width=WIDTH,
                ).images[0]
            image.save(out_path)
            timings.append(time.time() - t_img)
            peaks.append(torch.cuda.max_memory_allocated() / 1e9)
            torch.cuda.reset_peak_memory_stats()
            if (i + 1) % 25 == 0 or i < 3:
                avg_t = sum(timings) / len(timings)
                avg_peak = sum(peaks) / len(peaks)
                print(f"  [{i+1}/{len(prompts)}] avg t/img: {avg_t:.2f}s  "
                      f"avg peak: {avg_peak:.2f} GB", flush=True)
    finally:
        handle.remove()

    elapsed = time.time() - t_overall
    avg_peak = sum(peaks) / len(peaks) if peaks else 0
    avg_time = sum(timings) / len(timings) if timings else 0

    print(f"\n=== Result ===")
    print(f"  Variant:                {variant_name}")
    print(f"  N images:               {len(timings)}")
    print(f"  Average peak VRAM:      {avg_peak:.2f} GB")
    print(f"  Average time per image: {avg_time:.2f} s")
    print(f"  Total time:             {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
