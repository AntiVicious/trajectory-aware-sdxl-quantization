"""
Generate V14 5k benchmark: V13 allocation + SmoothQuant per-channel scales.

For each layer the allocator marked as INT4@t:
  W_smooth = W * s        (multiply by per-input-channel migration scale)
  Quantize W_smooth to INT4  (now-gentler distribution quantizes better)
  At forward:  output = (X / s) @ (s * W)_quant  ≡  X @ W mathematically
  In practice: quality of output >>  X @ W_naive_int4

For FP16 layers: unchanged, no scale applied.

Output folders:
  benchmark_5k/v14_70pct_smoothquant/
  benchmark_5k/v14_55pct_smoothquant/
  benchmark_5k/v14_40pct_smoothquant/

Usage:
    CUDA_VISIBLE_DEVICES=0 python generate_v14_5k.py [--alpha 0.5] [--budgets 70 55 40]
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
OUT = BASE / "benchmark_5k"
ALLOC_DIR = BASE / "trajectory" / "phase3_allocator_full"
SQ_DIR = BASE / "trajectory" / "phase4_full_bench"
PROMPTS_PATH = BASE / "benchmark_prompts_5k.json"

SEED = 42
NUM_STEPS = 20
HEIGHT = WIDTH = 768
GUIDANCE_SCALE = 7.5

OUT.mkdir(parents=True, exist_ok=True)


# ============ INT4 quantization (same as V13, but applied to scaled W) ============

def quantize_int4_per_channel_cpu(w: torch.Tensor) -> torch.Tensor:
    """Quantize on CPU to avoid VRAM spikes during setup.
    Returns CPU tensor; caller moves to GPU."""
    w_cpu = w.detach().cpu()
    orig_shape = w_cpu.shape
    orig_dtype = w_cpu.dtype
    w2 = w_cpu.reshape(w_cpu.shape[0], -1).float() if w_cpu.ndim == 4 else w_cpu.float()
    scale = (w2.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-8)
    q = (w2 / scale).round().clamp(-7, 7)
    return (q * scale).to(orig_dtype).reshape(orig_shape)


def quantize_int4_per_channel(w: torch.Tensor) -> torch.Tensor:
    """Backward-compat wrapper. Use *_cpu version directly when possible."""
    return quantize_int4_per_channel_cpu(w)


def apply_smoothquant_to_weight_cpu(w_cpu: torch.Tensor, s_cpu: torch.Tensor):
    """Multiply weight by per-input-channel scale s. CPU-side. Returns CPU tensor."""
    if w_cpu.ndim == 2:                # Linear: [out, in]
        return w_cpu * s_cpu.view(1, -1).to(w_cpu.dtype)
    elif w_cpu.ndim == 4:              # Conv2d: [out, in, kH, kW]
        return w_cpu * s_cpu.view(1, -1, 1, 1).to(w_cpu.dtype)
    else:
        raise ValueError(f"Unexpected weight ndim: {w_cpu.ndim}")


def apply_smoothquant_to_weight(w: torch.Tensor, s: torch.Tensor):
    """Backward-compat — delegates to CPU."""
    return apply_smoothquant_to_weight_cpu(w.cpu(), s.cpu())


# ============ Step counter ============

class StepCounter:
    def __init__(self):
        self.current = -1
    def advance(self):
        self.current += 1


# ============ MultiLayerSwap with SmoothQuant ============

class MultiLayerSwapSQ:
    """Step-conditional INT4 swap, but with SmoothQuant scale rebalancing.

    For each layer with threshold T < NUM_STEPS:
      - Compute W_smooth = W * s  (s from calibration)
      - Quantize W_smooth to INT4
      - At step >= T, forward uses: F.linear(X / s, W_smooth_q)
      - At step <  T, forward uses: F.linear(X, W) ← unchanged FP16 path
    """
    def __init__(self, unet, assignments, sq_scales, counter):
        self.unet = unet
        self.assignments = assignments
        self.sq_scales = sq_scales       # {layer_name: {"s": tensor}}
        self.counter = counter
        self.originals = {}
        self.skipped_no_scale = 0
        self.skipped_unresolvable = 0

    def __enter__(self):
        n_processed = 0
        for layer_name, threshold in self.assignments:
            try:
                module = self.unet.get_submodule(layer_name)
            except AttributeError:
                self.skipped_unresolvable += 1
                continue
            self.originals[layer_name] = module.forward
            if threshold >= NUM_STEPS:
                # FP16 forever, no swap needed
                continue

            # All quantization math on CPU to avoid VRAM spikes during setup.
            # Final w_quant is moved to GPU; original FP16 weight stays on GPU
            # as module.weight (referenced by closure, no duplicate copy needed).
            target_dtype = module.weight.dtype

            if layer_name not in self.sq_scales:
                self.skipped_no_scale += 1
                w_quant_cpu = quantize_int4_per_channel_cpu(module.weight.data)
                w_quant = w_quant_cpu.to("cuda", non_blocking=True)
                del w_quant_cpu
                s_inv = None
            else:
                s_cpu = self.sq_scales[layer_name]["s"].to(target_dtype).cpu()
                w_smooth_cpu = apply_smoothquant_to_weight_cpu(
                    module.weight.data.cpu(), s_cpu)
                w_quant_cpu = quantize_int4_per_channel_cpu(w_smooth_cpu)
                del w_smooth_cpu
                w_quant = w_quant_cpu.to("cuda", non_blocking=True)
                del w_quant_cpu
                s_inv = (1.0 / s_cpu).to(target_dtype).to("cuda", non_blocking=True)

            counter_ref = self.counter
            thr = threshold
            module_ref = module

            if isinstance(module, nn.Linear):
                bias = module.bias
                if s_inv is None:
                    # No scale: naive INT4 path (V13 fallback)
                    def make_fn(w_quant_=w_quant, bias_=bias,
                                counter_=counter_ref, thr_=thr, module_=module_ref):
                        def forward(x, *args, **kwargs):
                            w = w_quant_ if counter_.current >= thr_ else module_.weight
                            return F.linear(x, w, bias_)
                        return forward
                else:
                    s_inv_view = s_inv.view(1, -1)  # broadcast over batch dim
                    def make_fn(w_quant_=w_quant, s_inv_=s_inv_view, bias_=bias,
                                counter_=counter_ref, thr_=thr, module_=module_ref):
                        def forward(x, *args, **kwargs):
                            if counter_.current >= thr_:
                                # SmoothQuant path: X/s @ (s*W)_q
                                # Need to handle 3D inputs (transformer-style: B, L, D)
                                x_scaled = x * s_inv_ if x.ndim == 2 else \
                                          x * s_inv_.view(1, 1, -1)
                                return F.linear(x_scaled, w_quant_, bias_)
                            else:
                                return F.linear(x, module_.weight, bias_)
                        return forward
                module.forward = make_fn()

            elif isinstance(module, nn.Conv2d):
                bias = module.bias
                stride, padding = module.stride, module.padding
                dilation, groups = module.dilation, module.groups
                if s_inv is None:
                    def make_fn(w_quant_=w_quant, bias_=bias,
                                stride_=stride, padding_=padding,
                                dilation_=dilation, groups_=groups,
                                counter_=counter_ref, thr_=thr, module_=module_ref):
                        def forward(x, *args, **kwargs):
                            w = w_quant_ if counter_.current >= thr_ else module_.weight
                            return F.conv2d(x, w, bias_, stride_, padding_, dilation_, groups_)
                        return forward
                else:
                    # Conv2d input shape [B, C_in, H, W]; scale per C_in
                    s_inv_view = s_inv.view(1, -1, 1, 1)
                    def make_fn(w_quant_=w_quant, s_inv_=s_inv_view, bias_=bias,
                                stride_=stride, padding_=padding,
                                dilation_=dilation, groups_=groups,
                                counter_=counter_ref, thr_=thr, module_=module_ref):
                        def forward(x, *args, **kwargs):
                            if counter_.current >= thr_:
                                x_scaled = x * s_inv_
                                return F.conv2d(x_scaled, w_quant_, bias_,
                                                stride_, padding_, dilation_, groups_)
                            else:
                                return F.conv2d(x, module_.weight, bias_,
                                                stride_, padding_, dilation_, groups_)
                        return forward
                module.forward = make_fn()

            n_processed += 1
            # Periodic cache clear during the long setup loop (~700 iterations)
            if n_processed % 100 == 0:
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        if self.skipped_no_scale:
            print(f"  [SwapSQ] {self.skipped_no_scale} layers had no SmoothQuant scale "
                  f"(fell back to naive INT4)")
        if self.skipped_unresolvable:
            print(f"  [SwapSQ] {self.skipped_unresolvable} unresolvable layer names")
        return self

    def __exit__(self, *args):
        for name in list(self.originals.keys()):
            try:
                m = self.unet.get_submodule(name)
                m.forward = self.originals[name]
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
        thr = NUM_STEPS if choice == "FP16" else int(choice.split("t")[1])
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
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ============ Generation ============

def generate_variant(pipe, variant_name, assignments, sq_scales, prompts, seed):
    folder = OUT / variant_name
    folder.mkdir(parents=True, exist_ok=True)
    done = len([f for f in folder.iterdir() if f.suffix == '.jpg'])
    if done == len(prompts):
        print(f"  [{variant_name}] Already complete, skipping.")
        return
    print(f"  [{variant_name}] Starting from {done}/{len(prompts)}")

    counter = StepCounter()
    def pre_hook(module, args, kwargs):
        counter.advance()
    handle = pipe.unet.register_forward_pre_hook(pre_hook, with_kwargs=True)

    t0 = time.time()
    try:
        with MultiLayerSwapSQ(pipe.unet, assignments, sq_scales, counter):
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
                    rate = (i + 1 - done) / elapsed if elapsed > 0 else 0
                    remaining = (len(prompts) - i - 1) / rate if rate > 0 else 0
                    print(f"  [{variant_name}] {i+1}/{len(prompts)}  "
                          f"{elapsed:.0f}min  ETA {remaining:.0f}min")
    finally:
        handle.remove()
    final = len([f for f in folder.iterdir() if f.suffix == '.jpg'])
    print(f"  [{variant_name}] Done {final}/{len(prompts)} in {(time.time()-t0)/60:.1f} min")


# ============ Main ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", nargs="+", type=int, default=[70, 55, 40])
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Which alpha file to load (must match what calibration produced)")
    args = ap.parse_args()

    # Load SmoothQuant scales
    sq_path = SQ_DIR / f"smoothquant_scales_alpha{args.alpha:.2f}.pt"
    if not sq_path.exists():
        print(f"ERROR: SmoothQuant scales not found at {sq_path}")
        print(f"Run capture_smoothquant_scales.py first")
        return
    print(f"Loading SmoothQuant scales from {sq_path.name}...")
    sq_data = torch.load(sq_path, map_location="cpu", weights_only=False)
    sq_scales = sq_data["scales"]
    print(f"  Loaded {len(sq_scales)} layer scales (alpha={sq_data['alpha']})")

    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts")

    print("Loading SDXL pipeline...")
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
        variant_name = f"v14_{pct}pct_smoothquant"
        n_fp16 = sum(1 for _, t in assignments if t >= NUM_STEPS)
        n_int4 = len(assignments) - n_fp16
        print(f"\n=== {variant_name} ===")
        print(f"  Layers: {len(assignments)}  FP16: {n_fp16}  INT4: {n_int4}")
        print(f"  Memory: {alloc['actual_MB']:.0f} MB ({alloc['budget_pct']}% of FP16)")
        variants.append((variant_name, assignments, alloc))

    if not variants:
        print("No allocations to generate.")
        return

    print(f"\nWill generate {len(variants)} variants × {len(prompts)} images")
    t_total = time.time()
    for variant_name, assignments, alloc in variants:
        print(f"\n{'='*60}\nGenerating {variant_name}\n{'='*60}")
        generate_variant(pipe, variant_name, assignments, sq_scales, prompts, SEED)

    print(f"\nALL DONE in {(time.time()-t_total)/3600:.1f} hours")


if __name__ == "__main__":
    main()
