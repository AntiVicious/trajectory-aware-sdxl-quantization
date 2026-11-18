"""
Capture SmoothQuant activation scales for every quantizable layer in the UNet.

For each layer (Linear or Conv2d), we hook the input and track per-channel
max-abs across N calibration prompts. This gives us the activation scale `s`
that SmoothQuant uses to migrate outliers from activations to weights.

Output: smoothquant_scales.pt
  Dict mapping layer_name -> {"act_scale": tensor [in_channels]}

After this, V14 quantization uses:
   W_smooth = W * s     (multiply per-input-channel)
   X_smooth = X / s     (divide per-input-channel — applied at runtime)
   Then quantize W_smooth as usual, multiply X_smooth by quantized W_smooth.
   Mathematically: X_smooth @ W_smooth = (X/s) @ (s*W) = X @ W   (no change)
   But quantization quality of W_smooth >> W when activations have outliers.

Usage:
    CUDA_VISIBLE_DEVICES=0 python capture_smoothquant_scales.py \
        [--n-calibration 128] [--alpha 0.5]

Args:
  --n-calibration: number of prompts (default 128, use 64 if VRAM tight)
  --alpha: SmoothQuant migration strength, 0.0=no migration, 1.0=full migration.
           Default 0.5. Recommended range 0.3-0.7.
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

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
PROMPTS_PATH = BASE / "benchmark_prompts_5k.json"
OUT_DIR = BASE / "trajectory" / "phase4_full_bench"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LAYERS_JSON = BASE / "trajectory" / "phase1_atlas_full" / "layers_full.json"


# ============ Hook bookkeeping ============

class ActScaleTracker:
    """Track per-input-channel max-abs activation across forward passes."""
    def __init__(self):
        # layer_name -> tensor of shape [in_channels]
        self.scales = {}

    def update(self, layer_name, x):
        # x can be (B, in) for Linear or (B, C, H, W) for Conv2d
        if x.ndim == 2:                # Linear
            mx = x.abs().amax(dim=0)   # per-input-feature
        elif x.ndim == 3:              # Linear after reshape (e.g. transformer)
            mx = x.abs().amax(dim=(0, 1))
        elif x.ndim == 4:              # Conv2d
            mx = x.abs().amax(dim=(0, 2, 3))
        else:
            return
        mx = mx.detach().to(torch.float32).cpu()
        if layer_name not in self.scales:
            self.scales[layer_name] = mx
        else:
            self.scales[layer_name] = torch.maximum(self.scales[layer_name], mx)


def install_hooks(unet, layer_names, tracker):
    """Hook every targeted layer's pre-forward to capture inputs."""
    handles = []
    skipped = 0
    for name in layer_names:
        try:
            module = unet.get_submodule(name)
        except AttributeError:
            skipped += 1
            continue
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        def make_hook(layer_name=name):
            def hook(mod, args, kwargs):
                if len(args) > 0:
                    tracker.update(layer_name, args[0])
                elif "input" in kwargs:
                    tracker.update(layer_name, kwargs["input"])
            return hook
        handles.append(module.register_forward_pre_hook(make_hook(), with_kwargs=True))
    if skipped:
        print(f"  [hooks] skipped {skipped} unresolvable layer names")
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-calibration", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="SmoothQuant migration strength (0..1)")
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed-base", type=int, default=42)
    args = ap.parse_args()

    # Load prompts
    with open(PROMPTS_PATH) as f:
        all_prompts = json.load(f)
    prompts = all_prompts[:args.n_calibration]
    print(f"[smoothquant] Calibrating on {len(prompts)} prompts", flush=True)

    # Load layer enumeration
    with open(LAYERS_JSON) as f:
        layers_meta = json.load(f)["layers"]
    layer_names = [L["name"] for L in layers_meta]
    print(f"[smoothquant] Targeting {len(layer_names)} layers", flush=True)

    # Load pipeline
    print("[smoothquant] Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    # Tight memory mode for 2080 Ti
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    # Install hooks
    print("[smoothquant] Installing hooks...")
    tracker = ActScaleTracker()
    handles = install_hooks(pipe.unet, layer_names, tracker)
    print(f"[smoothquant] {len(handles)} hooks installed", flush=True)

    # Run calibration
    print(f"[smoothquant] Running calibration forwards...", flush=True)
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
                if (i + 1) % 16 == 0:
                    elapsed = (time.time() - t0) / 60
                    rate = (i + 1) / elapsed
                    eta = (len(prompts) - i - 1) / rate
                    n_seen = len(tracker.scales)
                    print(f"  [{i+1}/{len(prompts)}]  {elapsed:.1f}min  "
                          f"ETA {eta:.1f}min  layers_seen={n_seen}")
    finally:
        for h in handles:
            h.remove()

    print(f"\n[smoothquant] Calibration done in {(time.time()-t0)/60:.1f} min")
    print(f"[smoothquant] Captured scales for {len(tracker.scales)} layers", flush=True)

    # Compute SmoothQuant migration scale per layer:
    #   s_i = max(|X_i|)^alpha / max(|W_i|)^(1-alpha)
    #         where i is the input channel
    # Quantize W_smooth = W * s, divide activations by s at runtime.
    print(f"[smoothquant] Computing migration scales (alpha={args.alpha})...", flush=True)
    migration_scales = {}
    for layer_name, act_scale in tracker.scales.items():
        try:
            module = pipe.unet.get_submodule(layer_name)
        except AttributeError:
            continue
        w = module.weight.data.detach().to(torch.float32).cpu()
        if isinstance(module, nn.Linear):
            # Linear: W shape [out_features, in_features]
            w_scale = w.abs().amax(dim=0)         # per input feature
        elif isinstance(module, nn.Conv2d):
            # Conv2d: W shape [out_C, in_C, kH, kW]
            w_scale = w.abs().amax(dim=(0, 2, 3))   # per input channel
        else:
            continue
        # Avoid div-zero
        act_safe = act_scale.clamp(min=1e-5)
        w_safe = w_scale.clamp(min=1e-5)
        s = (act_safe ** args.alpha) / (w_safe ** (1.0 - args.alpha))
        s = s.clamp(min=1e-5)  # absolutely no zeros
        migration_scales[layer_name] = {
            "s": s.to(torch.float16),
            "act_scale": act_scale.to(torch.float16),
            "w_scale": w_scale.to(torch.float16),
        }

    out_path = OUT_DIR / f"smoothquant_scales_alpha{args.alpha:.2f}.pt"
    torch.save({
        "alpha": args.alpha,
        "n_calibration": args.n_calibration,
        "num_steps": args.num_steps,
        "height": args.height, "width": args.width,
        "guidance": args.guidance,
        "scales": migration_scales,
    }, out_path)
    print(f"[smoothquant] Saved -> {out_path}", flush=True)
    print(f"[smoothquant] {len(migration_scales)} layer scales saved", flush=True)

    # Sanity print
    sample_names = list(migration_scales.keys())[:5]
    print(f"\n[smoothquant] Sample scales:")
    for name in sample_names:
        s = migration_scales[name]["s"]
        print(f"  {name[:60]:<62}  s shape={tuple(s.shape)}  "
              f"min={s.min().item():.3e} max={s.max().item():.3e}")


if __name__ == "__main__":
    main()
