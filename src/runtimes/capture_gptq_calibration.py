"""
Capture per-layer calibration inputs for GPTQ.

For each Linear layer in the UNet, run N calibration prompts through the
full SDXL pipeline and capture a sample of inputs that arrive at that layer.

GPTQ needs: a matrix of activations [N_samples, in_features] per layer.
We sample inputs to keep memory bounded; full sequences would be huge.

Storage strategy: stream to disk per-layer to avoid 100+ GB RAM.

Output: gptq_calibration_data/<layer_name>.pt
  Each file: tensor [n_samples, in_features]

Usage:
    CUDA_VISIBLE_DEVICES=0 python capture_gptq_calibration.py \
        [--n-calibration 64] [--samples-per-layer 512]
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline

BASE = Path("/home/project/Documents/Maharshi/gem")
PROMPTS_PATH = BASE / "benchmark_prompts_5k.json"
OUT_DIR = BASE / "trajectory" / "phase4_full_bench" / "gptq_calibration_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LAYERS_JSON = BASE / "trajectory" / "phase1_atlas_full" / "layers_full.json"


class CalibCapture:
    """Captures activation rows per layer.
    Appends fast (no per-row Python loops). Subsamples at end of run."""
    def __init__(self, samples_per_layer):
        self.samples_per_layer = samples_per_layer
        # layer_name -> list of CPU tensors collected during the run
        self.chunks = {}
        # layer_name -> total rows seen so far (for early stopping)
        self.seen = {}

    def update(self, layer_name, x):
        # Reshape to 2D: [N, in_features]
        if x.ndim == 2:
            x2d = x
        elif x.ndim == 3:
            x2d = x.reshape(-1, x.shape[-1])
        else:
            return  # Conv inputs not supported (Linear-only GPTQ)

        # Cap collected rows per layer at 4x target so we have variety to subsample.
        # Once we've collected enough, skip further captures for this layer.
        cap = self.samples_per_layer * 4
        seen = self.seen.get(layer_name, 0)
        if seen >= cap:
            return

        # Take a stride to limit per-call rows, and only keep up to needed amount.
        n = x2d.shape[0]
        remaining = cap - seen
        if n > remaining:
            # Keep first `remaining` rows
            x2d = x2d[:remaining]
            n = remaining

        # Subsample at most samples_per_layer / 4 per call to bound memory
        per_call_max = max(1, self.samples_per_layer // 4)
        if n > per_call_max:
            stride = max(1, n // per_call_max)
            x2d = x2d[::stride][:per_call_max]

        x_cpu = x2d.detach().to(torch.float16).cpu()
        self.chunks.setdefault(layer_name, []).append(x_cpu)
        self.seen[layer_name] = seen + x.shape[0] if x.ndim == 2 else seen + x.shape[0] * x.shape[1]

    def finalize(self):
        """Concatenate chunks per layer, subsample to samples_per_layer rows."""
        result = {}
        for layer_name, chunks in self.chunks.items():
            full = torch.cat(chunks, dim=0)
            if full.shape[0] > self.samples_per_layer:
                # Random subsample to exactly samples_per_layer
                idx = torch.randperm(full.shape[0])[:self.samples_per_layer]
                full = full[idx]
            result[layer_name] = full
        return result


def install_hooks(unet, layer_names, capture):
    handles = []
    skipped_non_linear = 0
    skipped_unresolvable = 0
    for name in layer_names:
        try:
            module = unet.get_submodule(name)
        except AttributeError:
            skipped_unresolvable += 1
            continue
        # GPTQ scope: Linear only
        if not isinstance(module, nn.Linear):
            skipped_non_linear += 1
            continue
        def make_hook(layer_name=name):
            def hook(mod, args, kwargs):
                if len(args) > 0:
                    capture.update(layer_name, args[0])
                elif "input" in kwargs:
                    capture.update(layer_name, kwargs["input"])
            return hook
        handles.append(module.register_forward_pre_hook(make_hook(), with_kwargs=True))
    print(f"[gptq-cal] Hooked {len(handles)} Linear layers", flush=True)
    print(f"[gptq-cal] Skipped {skipped_non_linear} non-Linear layers (Conv2d kept FP16/naive INT4)", flush=True)
    print(f"[gptq-cal] Skipped {skipped_unresolvable} unresolvable names", flush=True)
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calibration", type=int, default=64)
    ap.add_argument("--samples-per-layer", type=int, default=512,
                    help="Max calibration rows to keep per layer. 512 is standard.")
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed-base", type=int, default=42)
    args = ap.parse_args()

    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)[:args.n_calibration]
    print(f"[gptq-cal] Calibrating on {len(prompts)} prompts", flush=True)

    with open(LAYERS_JSON) as f:
        layers_meta = json.load(f)["layers"]
    layer_names = [L["name"] for L in layers_meta]
    print(f"[gptq-cal] {len(layer_names)} candidate layer names", flush=True)

    print("[gptq-cal] Loading SDXL pipeline...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    print("[gptq-cal] Installing hooks...", flush=True)
    capture = CalibCapture(samples_per_layer=args.samples_per_layer)
    handles = install_hooks(pipe.unet, layer_names, capture)

    print(f"[gptq-cal] Running {len(prompts)} forwards (each = 20 denoising steps)...", flush=True)
    t0 = time.time()
    try:
        with torch.inference_mode():
            for i, item in enumerate(prompts):
                gen = torch.Generator("cuda").manual_seed(args.seed_base + i)
                _ = pipe(
                    item["caption"],
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance,
                    generator=gen,
                    height=args.height, width=args.width,
                )
                if (i + 1) % 8 == 0:
                    elapsed = (time.time() - t0) / 60
                    eta = (len(prompts) - i - 1) / max((i + 1) / elapsed, 1e-9)
                    n_layers = len(capture.chunks)
                    n_full = sum(1 for cs in capture.chunks.values()
                                 if sum(c.shape[0] for c in cs) >= args.samples_per_layer)
                    print(f"  [{i+1}/{len(prompts)}]  {elapsed:.1f}min  ETA {eta:.1f}min  "
                          f"layers_seen={n_layers}  layers_full={n_full}", flush=True)
    finally:
        for h in handles:
            h.remove()

    print(f"\n[gptq-cal] Done in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"[gptq-cal] Finalizing {len(capture.chunks)} layers...", flush=True)
    finalized = capture.finalize()
    print(f"[gptq-cal] Got {len(finalized)} layer matrices", flush=True)

    # Save per-layer activation matrices
    print(f"[gptq-cal] Saving to {OUT_DIR}...", flush=True)
    saved_total_bytes = 0
    for layer_name, mat in finalized.items():
        # Filename-safe layer name
        safe = layer_name.replace(".", "__").replace("/", "__")
        path = OUT_DIR / f"{safe}.pt"
        mat_fp16 = mat.to(torch.float16)
        torch.save(mat_fp16, path)
        saved_total_bytes += mat_fp16.numel() * 2
    print(f"[gptq-cal] Saved {len(finalized)} files, "
          f"~{saved_total_bytes/1e9:.2f} GB total", flush=True)

    # Save metadata for the GPTQ runner
    meta_path = OUT_DIR / "_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "n_calibration": args.n_calibration,
            "samples_per_layer": args.samples_per_layer,
            "num_steps": args.num_steps,
            "height": args.height, "width": args.width,
            "guidance": args.guidance,
            "n_layers_captured": len(finalized),
            "layer_names": list(finalized.keys()),
        }, f, indent=2)
    print(f"[gptq-cal] Metadata -> {meta_path}", flush=True)


if __name__ == "__main__":
    main()