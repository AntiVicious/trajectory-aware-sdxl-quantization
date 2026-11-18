"""
Generate V15 5k benchmark: V13 allocation + GPTQ-quantized Linear weights.

For each layer the allocator marked as INT4@t:
  - If Linear AND we have a GPTQ weight on disk → use GPTQ-quantized weight
  - If Conv2d (no GPTQ) → fall back to naive per-channel INT4 (V13 behavior)

Output folders:
  benchmark_5k/v15_70pct_gptq/
  benchmark_5k/v15_55pct_gptq/
  benchmark_5k/v15_40pct_gptq/

Usage:
    CUDA_VISIBLE_DEVICES=0 python generate_v15_5k.py [--budgets 70 55 40]
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
GPTQ_DIR = BASE / "trajectory" / "phase4_full_bench" / "gptq_quantized_weights"
PROMPTS_PATH = BASE / "benchmark_prompts_5k.json"

SEED = 42
NUM_STEPS = 20
HEIGHT = WIDTH = 768
GUIDANCE_SCALE = 7.5

OUT.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    return name.replace(".", "__").replace("/", "__")


def quantize_int4_per_channel_naive(w: torch.Tensor) -> torch.Tensor:
    """Fallback for Conv2d (no GPTQ available)."""
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


class MultiLayerSwapGPTQ:
    """Step-conditional swap with GPTQ weights (filtered by holdout) for Linear,
    naive INT4 for Conv2d and for GPTQ-overfit Linear layers."""
    def __init__(self, unet, assignments, counter, gptq_eligible=None):
        """
        Args:
            gptq_eligible: optional set of layer_names that passed holdout test.
                If None, all GPTQ weights are used (no filter).
                If a set, only layers in this set use GPTQ; rest use naive INT4.
        """
        self.unet = unet
        self.assignments = assignments
        self.counter = counter
        self.gptq_eligible = gptq_eligible
        self.originals = {}
        self.gptq_used = 0
        self.naive_fallback = 0
        self.gptq_filtered_out = 0

    def __enter__(self):
        for layer_name, threshold in self.assignments:
            try:
                module = self.unet.get_submodule(layer_name)
            except AttributeError:
                continue
            self.originals[layer_name] = module.forward
            if threshold >= NUM_STEPS:
                continue   # FP16 for this layer

            # Source the quantized weight
            if isinstance(module, nn.Linear):
                gptq_path = GPTQ_DIR / f"{safe_filename(layer_name)}.pt"
                use_gptq = (
                    gptq_path.exists()
                    and (self.gptq_eligible is None or layer_name in self.gptq_eligible)
                )
                if use_gptq:
                    pack = torch.load(gptq_path, map_location="cuda", weights_only=False)
                    w_quant = pack["w_dequant"].to(module.weight.dtype).to("cuda")
                    self.gptq_used += 1
                elif gptq_path.exists():
                    # GPTQ available but filtered out by holdout test
                    w_quant = quantize_int4_per_channel_naive(module.weight.data).to("cuda")
                    self.gptq_filtered_out += 1
                else:
                    w_quant = quantize_int4_per_channel_naive(module.weight.data).to("cuda")
                    self.naive_fallback += 1

                bias = module.bias
                counter_ref = self.counter
                thr = threshold
                module_ref = module
                def make_fn(w_quant_=w_quant, bias_=bias,
                            counter_=counter_ref, thr_=thr, module_=module_ref):
                    def forward(x, *args, **kwargs):
                        w = w_quant_ if counter_.current >= thr_ else module_.weight
                        return F.linear(x, w, bias_)
                    return forward
                module.forward = make_fn()

            elif isinstance(module, nn.Conv2d):
                # Conv2d: always naive INT4 (no GPTQ for Conv in this setup)
                w_quant = quantize_int4_per_channel_naive(module.weight.data).to("cuda")
                self.naive_fallback += 1
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

        print(f"  [SwapGPTQ] {self.gptq_used} layers using GPTQ (passed holdout filter), "
              f"{self.gptq_filtered_out} layers GPTQ-available-but-filtered-out (using naive), "
              f"{self.naive_fallback} layers using naive INT4 (Conv2d or no GPTQ data)", flush=True)
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


def load_assignments(alloc_path):
    with open(alloc_path) as f:
        alloc = json.load(f)
    assignments = []
    for a in alloc["allocations"]:
        choice = a["choice"]
        thr = NUM_STEPS if choice == "FP16" else int(choice.split("t")[1])
        assignments.append((a["layer"], thr))
    return assignments, alloc


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


def generate_variant(pipe, variant_name, assignments, prompts, seed, gptq_eligible=None):
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
        with MultiLayerSwapGPTQ(pipe.unet, assignments, counter, gptq_eligible=gptq_eligible):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", nargs="+", type=int, default=[70, 55, 40])
    ap.add_argument("--holdout-threshold", type=float, default=20.0,
                    help="Min holdout improvement %% to use GPTQ for a layer (default 20). "
                         "Set to -999 to use all GPTQ weights without filtering.")
    args = ap.parse_args()

    # Sanity check GPTQ weights present
    n_gptq = len(list(GPTQ_DIR.glob("*.pt")))
    print(f"GPTQ weights available: {n_gptq} layers", flush=True)
    if n_gptq == 0:
        print(f"ERROR: no GPTQ weights at {GPTQ_DIR}. Run run_gptq.py first.")
        return

    # Load holdout-test results to filter out overfit/negative layers
    holdout_log = BASE / "trajectory" / "phase4_full_bench" / "gptq_holdout_log.csv"
    gptq_eligible = None
    if holdout_log.exists() and args.holdout_threshold > -999:
        gptq_eligible = set()
        with open(holdout_log) as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 7:
                    layer, improve_pct = parts[0], float(parts[6])
                    if improve_pct >= args.holdout_threshold:
                        gptq_eligible.add(layer)
        print(f"Holdout filter: {len(gptq_eligible)} of {n_gptq} GPTQ layers passed "
              f"(holdout improve >= {args.holdout_threshold}%)", flush=True)
    else:
        print(f"No holdout filter applied — using all GPTQ weights.", flush=True)

    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts", flush=True)

    print("Loading SDXL pipeline...", flush=True)
    pipe = load_pipe()
    print(f"Pipeline on {torch.cuda.get_device_name(0)}", flush=True)

    variants = []
    for pct in args.budgets:
        alloc_path = ALLOC_DIR / f"allocation_{pct}pct_full.json"
        if not alloc_path.exists():
            print(f"  MISSING: {alloc_path}")
            continue
        assignments, alloc = load_assignments(alloc_path)
        variant_name = f"v15_{pct}pct_gptq"
        n_fp16 = sum(1 for _, t in assignments if t >= NUM_STEPS)
        n_int4 = len(assignments) - n_fp16
        print(f"\n=== {variant_name} ===")
        print(f"  Layers: {len(assignments)}  FP16: {n_fp16}  INT4: {n_int4}")
        print(f"  Memory: {alloc['actual_MB']:.0f} MB")
        variants.append((variant_name, assignments, alloc))

    if not variants:
        print("No allocations to generate.")
        return

    print(f"\nWill generate {len(variants)} variants × {len(prompts)} images")
    t_total = time.time()
    for variant_name, assignments, alloc in variants:
        print(f"\n{'='*60}\nGenerating {variant_name}\n{'='*60}", flush=True)
        generate_variant(pipe, variant_name, assignments, prompts, SEED,
                         gptq_eligible=gptq_eligible)

    print(f"\nALL DONE in {(time.time()-t_total)/3600:.1f} hours")


if __name__ == "__main__":
    main()