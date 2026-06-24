"""
V22 Hot-swap: per-layer dynamic FP16 -> NF4 transition during trajectory.

Each Linear layer in V13's INT4 set has TWO weights pre-loaded:
  - FP16 original weight (~3 MB per layer)
  - NF4 quantized weight (~0.75 MB per layer)
The forward pass uses FP16 for steps < commit_step, NF4 for steps >= commit_step.

After each generation, layers whose current_step has just crossed commit_step
have their FP16 weight FREED (set to None). The GPU memory is then released
via torch.cuda.empty_cache() at strategic points.

NOTE: This is the "best of both worlds" attempt.
  - Quality should match V13/V19 (per-layer trajectory, exact thresholds)
  - VRAM should approach V20's footprint by the END of generation,
    though peak VRAM is still bounded by Phase 1's FP16 phase

Usage:
    CUDA_VISIBLE_DEVICES=2 python -u v22_hot_swap.py \
        --budget 70 --n-images 5

Test extensively at small N first. This is experimental.
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

BASE = Path.home() / "Documents" / "Maharshi" / "gem"
PROMPTS_PATH = BASE / "benchmark_prompts_5k.json"
ALLOC_DIR = BASE / "trajectory" / "phase3_allocator_full"
OUT = BASE / "benchmark_5k"

NUM_STEPS = 20
GUIDANCE_SCALE = 7.5
HEIGHT = 768
WIDTH = 768
SEED = 42


# ===================== Hot-swap layer =====================

class HotSwapLinear(nn.Module):
    """A Linear that internally has both FP16 and NF4 weights, switching
    based on a step counter and the layer's commit threshold."""
    def __init__(self, original_linear, commit_step, step_counter):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.commit_step = commit_step
        self.step_counter = step_counter

        # FP16 weight: keep as a parameter we can free
        self._fp16_weight = original_linear.weight.data.clone().contiguous()

        # NF4 weight: build a Linear4bit, take its quantized weight
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

        # Original bias for FP16 path
        if original_linear.bias is not None:
            self._fp16_bias = original_linear.bias.data.clone()
        else:
            self._fp16_bias = None

        self._fp16_freed = False  # set True after we free FP16

    def forward(self, x):
        # CFG-style scheduler: counter advances 2x per scheduler step (uncond + cond)
        # but our commit thresholds are in scheduler steps. So divide.
        step = self.step_counter.current // 2
        if step < self.commit_step and not self._fp16_freed:
            # FP16 path
            return F.linear(x, self._fp16_weight, self._fp16_bias)
        else:
            # NF4 path
            return self._nf4_layer(x)

    def try_free_fp16(self):
        """Free the FP16 weight if its commit step has passed."""
        if self._fp16_freed:
            return False
        step = self.step_counter.current // 2
        if step >= self.commit_step:
            del self._fp16_weight
            del self._fp16_bias
            self._fp16_weight = None
            self._fp16_bias = None
            self._fp16_freed = True
            return True
        return False

    def reset_for_new_image(self):
        """Re-allocate FP16 weight from a saved copy. Required for next image."""
        # We don't actually want to re-allocate every image — that defeats the purpose.
        # Instead, we just reset the freed flag. But once FP16 is freed, it's gone.
        # For multi-image generation, we'd need to either:
        #   (a) Keep FP16 weight permanently (no VRAM saving)
        #   (b) Re-load from CPU each image (slow but works)
        # Default behavior: once freed, stays freed. Use one image at a time
        # OR reset before each image by calling reload_fp16().
        pass


class StepCounter:
    def __init__(self): self.current = -1
    def advance(self): self.current += 1
    def reset(self): self.current = -1


# ===================== Allocation loading =====================

def load_v13_allocation(pct):
    p = ALLOC_DIR / f"allocation_{pct}pct_full.json"
    with open(p) as f:
        alloc = json.load(f)
    layer_to_threshold = {}
    for a in alloc["allocations"]:
        if a["choice"] == "FP16":
            t = NUM_STEPS  # Never quantize
        else:
            t = int(a["choice"].split("t")[1])
        layer_to_threshold[a["layer"]] = t
    return layer_to_threshold


# ===================== Hot-swap setup =====================

def install_hot_swap_layers(unet, layer_to_threshold, step_counter):
    """Replace each Linear in V13's INT4 set with a HotSwapLinear.
    Conv2d layers stay FP16 throughout (bnb doesn't support Conv2d)."""
    n_replaced = 0
    n_skipped_conv = 0
    n_fp16_throughout = 0

    layer_paths = {name: None for name in layer_to_threshold}
    for name, module in unet.named_modules():
        if name in layer_paths:
            layer_paths[name] = module

    hot_swap_layers = []   # keep references for try_free_fp16 calls

    for name, threshold in layer_to_threshold.items():
        module = layer_paths.get(name)
        if module is None:
            continue
        if threshold >= NUM_STEPS:
            n_fp16_throughout += 1
            continue   # FP16 layer — leave alone
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
        hot_swap_layers.append((name, threshold, new_module))
        n_replaced += 1

    print(f"  [HotSwap] {n_replaced} HotSwapLinear, "
          f"{n_skipped_conv} Conv2d FP16, "
          f"{n_fp16_throughout} explicit FP16", flush=True)
    return hot_swap_layers


def free_committed_fp16(hot_swap_layers, verbose=False):
    """Sweep over hot-swap layers and free FP16 weights for ones that committed."""
    n_freed = 0
    for name, threshold, layer in hot_swap_layers:
        if layer.try_free_fp16():
            n_freed += 1
    if n_freed > 0:
        gc.collect()
        torch.cuda.empty_cache()
    return n_freed


# ===================== Main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=70)
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--free-fp16", action="store_true",
                    help="Aggressively free FP16 weights mid-generation. "
                         "Saves VRAM but only works for ONE image (FP16 is gone after).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"=== V22 Hot-Swap at {args.budget}% budget ===")
    if args.free_fp16:
        print(f"  --free-fp16 enabled: VRAM will drop after each commit step.")
        print(f"  WARNING: This works for one image only. Multi-image generation")
        print(f"           with --free-fp16 requires resetting FP16 between images.")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("\nLoading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    print(f"  Pipeline loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Build hot-swap layers
    layer_to_threshold = load_v13_allocation(args.budget)
    step_counter = StepCounter()
    hot_swap_layers = install_hot_swap_layers(
        pipe.unet, layer_to_threshold, step_counter)

    after_setup = torch.cuda.memory_allocated() / 1e9
    print(f"  After hot-swap setup. VRAM: {after_setup:.2f} GB "
          f"(both FP16 and NF4 weights held)")

    # Step counter hook
    def pre_hook(module, args_, kwargs_):
        step_counter.advance()
        # If we want to free FP16 mid-generation, do it here on each step
        if args.free_fp16:
            free_committed_fp16(hot_swap_layers)
    handle = pipe.unet.register_forward_pre_hook(pre_hook, with_kwargs=True)

    # Generate
    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)[:args.n_images]
    print(f"\nGenerating {len(prompts)} images...")

    free_suffix = "_freed" if args.free_fp16 else ""
    variant_name = f"v22_{args.budget}pct_hotswap{free_suffix}"
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
            try:
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
            except Exception as e:
                print(f"  Image {i+1} failed: {e}", flush=True)
                continue

            # If we freed FP16 weights, the next image won't have them.
            # Stop here unless we re-load. For now: only first image is reliable.
            if args.free_fp16 and i == 0:
                print(f"  --free-fp16: stopping after image 1 (FP16 weights freed).")
                print(f"  This image is correct; subsequent images would error or degrade.")
                break

            torch.cuda.reset_peak_memory_stats()

            if (i + 1) % 5 == 0 or i < 3:
                elapsed = (time.time() - t_overall) / 60
                avg_t = sum(timings) / len(timings) if timings else 0
                avg_peak = sum(peaks) / len(peaks) if peaks else 0
                print(f"  [{i+1}/{len(prompts)}] {elapsed:.1f}min  "
                      f"avg t/img: {avg_t:.2f}s  avg peak: {avg_peak:.2f} GB",
                      flush=True)
    finally:
        handle.remove()

    elapsed = time.time() - t_overall
    avg_peak = sum(peaks) / len(peaks) if peaks else 0
    max_peak = max(peaks) if peaks else 0
    avg_time = sum(timings) / len(timings) if timings else 0

    # Final VRAM after generation completes
    final_vram = torch.cuda.memory_allocated() / 1e9
    print(f"\n=== Result ===")
    print(f"  Variant:                {variant_name}")
    print(f"  N images:               {len(timings)}")
    print(f"  Average peak VRAM:      {avg_peak:.2f} GB")
    print(f"  Max peak VRAM:          {max_peak:.2f} GB")
    print(f"  Final VRAM after gen:   {final_vram:.2f} GB")
    print(f"  Average time per image: {avg_time:.2f} s")
    print(f"  Total time:             {elapsed/60:.1f} min")

    out_path = BASE / f"vram_measurement_{variant_name}.json"
    with open(out_path, "w") as f:
        json.dump({
            "variant": variant_name,
            "budget_pct": args.budget,
            "free_fp16": args.free_fp16,
            "n_images": len(timings),
            "avg_peak_vram_gb": avg_peak,
            "max_peak_vram_gb": max_peak,
            "final_vram_after_gen_gb": final_vram,
            "avg_time_per_image_s": avg_time,
            "total_time_min": elapsed / 60,
            "gpu_name": torch.cuda.get_device_name(0),
        }, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
