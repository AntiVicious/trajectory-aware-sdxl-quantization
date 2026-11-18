"""
Held-out test for GPTQ-quantized weights.

For each Linear layer with GPTQ weights, capture activations on 64 NEW prompts
(different seeds + indices than calibration), then compute:
  - naive INT4 error vs FP16
  - GPTQ INT4 error vs FP16
  - improvement %

If GPTQ generalizes: held-out improvements should still be meaningful (>20%).
If GPTQ overfit: held-out improvements will collapse, possibly even be negative.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u test_gptq_holdout.py [--n-test 64]
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
from diffusers import StableDiffusionXLPipeline

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
PROMPTS_PATH = BASE / "benchmark_prompts_5k.json"
GPTQ_DIR = BASE / "trajectory" / "phase4_full_bench" / "gptq_quantized_weights"
LAYERS_JSON = BASE / "trajectory" / "phase1_atlas_full" / "layers_full.json"
OUT_DIR = BASE / "trajectory" / "phase4_full_bench"


def safe_filename(name: str) -> str:
    return name.replace(".", "__").replace("/", "__")


# Capture activations on held-out prompts. Uses same fast capture as calibration.
class HoldoutCapture:
    def __init__(self, samples_per_layer=512):
        self.samples_per_layer = samples_per_layer
        self.chunks = {}
        self.seen = {}

    def update(self, layer_name, x):
        if x.ndim == 2:
            x2d = x
        elif x.ndim == 3:
            x2d = x.reshape(-1, x.shape[-1])
        else:
            return
        cap = self.samples_per_layer * 4
        seen = self.seen.get(layer_name, 0)
        if seen >= cap:
            return
        n = x2d.shape[0]
        remaining = cap - seen
        if n > remaining:
            x2d = x2d[:remaining]
            n = remaining
        per_call_max = max(1, self.samples_per_layer // 4)
        if n > per_call_max:
            stride = max(1, n // per_call_max)
            x2d = x2d[::stride][:per_call_max]
        x_cpu = x2d.detach().to(torch.float16).cpu()
        self.chunks.setdefault(layer_name, []).append(x_cpu)
        self.seen[layer_name] = seen + (x.shape[0] if x.ndim == 2 else x.shape[0] * x.shape[1])

    def finalize(self):
        result = {}
        for layer_name, chunks in self.chunks.items():
            full = torch.cat(chunks, dim=0)
            if full.shape[0] > self.samples_per_layer:
                idx = torch.randperm(full.shape[0])[:self.samples_per_layer]
                full = full[idx]
            result[layer_name] = full
        return result


def install_hooks(unet, layer_names, capture):
    handles = []
    for name in layer_names:
        try:
            module = unet.get_submodule(name)
        except AttributeError:
            continue
        if not isinstance(module, nn.Linear):
            continue
        def make_hook(layer_name=name):
            def hook(mod, args, kwargs):
                if len(args) > 0:
                    capture.update(layer_name, args[0])
                elif "input" in kwargs:
                    capture.update(layer_name, kwargs["input"])
            return hook
        handles.append(module.register_forward_pre_hook(make_hook(), with_kwargs=True))
    return handles


def quantize_int4_naive(w):
    qmax = 7
    scales = (w.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-8)
    q = (w / scales).round().clamp(-qmax, qmax)
    return q * scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=64,
                    help="Number of held-out prompts (different from calibration)")
    ap.add_argument("--samples-per-layer", type=int, default=512)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--guidance", type=float, default=7.5)
    args = ap.parse_args()

    # Load DIFFERENT prompts than calibration (we used 0..64; use 64..128)
    with open(PROMPTS_PATH) as f:
        all_prompts = json.load(f)
    test_prompts = all_prompts[64:64 + args.n_test]
    print(f"[holdout] Using prompts {64}..{64 + len(test_prompts)} (held-out from calibration)", flush=True)

    with open(LAYERS_JSON) as f:
        layers_meta = json.load(f)["layers"]
    layer_names = [L["name"] for L in layers_meta]

    print("[holdout] Loading SDXL pipeline...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    # Use DIFFERENT seeds than calibration: calibration used 42+i, we use 9000+i
    HOLDOUT_SEED_BASE = 9000

    capture = HoldoutCapture(samples_per_layer=args.samples_per_layer)
    handles = install_hooks(pipe.unet, layer_names, capture)
    print(f"[holdout] Hooked {len(handles)} Linear layers", flush=True)

    print(f"[holdout] Running {len(test_prompts)} forwards on held-out prompts...", flush=True)
    t0 = time.time()
    try:
        with torch.inference_mode():
            for i, item in enumerate(test_prompts):
                gen = torch.Generator("cuda").manual_seed(HOLDOUT_SEED_BASE + i)
                _ = pipe(
                    item["caption"],
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance,
                    generator=gen,
                    height=args.height, width=args.width,
                )
                if (i + 1) % 16 == 0:
                    elapsed = (time.time() - t0) / 60
                    print(f"  [{i+1}/{len(test_prompts)}]  {elapsed:.1f}min", flush=True)
    finally:
        for h in handles:
            h.remove()

    print(f"\n[holdout] Capture done in {(time.time()-t0)/60:.1f} min. Finalizing...", flush=True)
    holdout_acts = capture.finalize()
    print(f"[holdout] Got {len(holdout_acts)} layer matrices", flush=True)

    # Now compare GPTQ vs naive on held-out data
    print("\n[holdout] Computing held-out errors per layer...", flush=True)

    # Reload UNet for original weights (capture didn't modify them but be safe)
    rows = []
    n_done = 0
    n_overfit = 0  # GPTQ improve > 80% on calib, < 30% on holdout
    n_genuine = 0  # GPTQ improve > 30% on holdout
    n_negative = 0 # GPTQ worse than naive on holdout
    n_no_gptq = 0

    # Read calibration log to get the calib improvement per layer
    calib_log_path = OUT_DIR / "gptq_log.csv"
    calib_data = {}
    if calib_log_path.exists():
        with open(calib_log_path) as f:
            next(f)  # header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 7:
                    calib_data[parts[0]] = float(parts[6])

    out_csv_path = OUT_DIR / "gptq_holdout_log.csv"
    f_out = open(out_csv_path, "w", buffering=1)
    f_out.write("layer,n_in,n_out,calib_improve_pct,holdout_naive_err,holdout_gptq_err,"
                "holdout_improve_pct,verdict\n")

    for layer_name in holdout_acts.keys():
        try:
            module = pipe.unet.get_submodule(layer_name)
        except AttributeError:
            continue
        if not isinstance(module, nn.Linear):
            continue

        gptq_path = GPTQ_DIR / f"{safe_filename(layer_name)}.pt"
        if not gptq_path.exists():
            n_no_gptq += 1
            continue

        # Load fresh activations and weights
        X = holdout_acts[layer_name].to("cuda").to(torch.float32)
        W = module.weight.data.detach().to("cuda").to(torch.float32)
        gptq_pack = torch.load(gptq_path, map_location="cuda", weights_only=False)
        Q_gptq = gptq_pack["w_dequant"].to(torch.float32)

        # Naive INT4
        Q_naive = quantize_int4_naive(W)

        # Errors on held-out X
        with torch.no_grad():
            Y_orig = X @ W.T
            Y_naive = X @ Q_naive.T
            Y_gptq = X @ Q_gptq.T
            naive_err = ((Y_orig - Y_naive) ** 2).mean().item()
            gptq_err = ((Y_orig - Y_gptq) ** 2).mean().item()

        if naive_err <= 0:
            improve = 0.0
        else:
            improve = 100.0 * (naive_err - gptq_err) / naive_err

        calib_improve = calib_data.get(layer_name, -1.0)

        # Verdict
        if improve < 0:
            verdict = "NEGATIVE"
            n_negative += 1
        elif calib_improve > 80 and improve < 30:
            verdict = "OVERFIT"
            n_overfit += 1
        elif improve > 30:
            verdict = "GENUINE"
            n_genuine += 1
        else:
            verdict = "MARGINAL"

        f_out.write(f"{layer_name},{W.shape[1]},{W.shape[0]},"
                    f"{calib_improve:.2f},{naive_err:.6e},{gptq_err:.6e},"
                    f"{improve:.2f},{verdict}\n")

        rows.append({
            "layer": layer_name,
            "calib_improve": calib_improve,
            "holdout_improve": improve,
            "verdict": verdict,
        })
        n_done += 1
        if n_done % 100 == 0:
            print(f"  [{n_done}]  layer={layer_name[:50]}  "
                  f"calib={calib_improve:.1f}%  holdout={improve:.1f}%  {verdict}", flush=True)

        del X, W, Q_gptq, Q_naive, Y_orig, Y_naive, Y_gptq
        torch.cuda.empty_cache()

    f_out.close()

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"HELD-OUT TEST SUMMARY ({n_done} layers tested)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  GENUINE  (holdout improve > 30%):  {n_genuine}  "
          f"({100*n_genuine/n_done:.1f}%)", flush=True)
    print(f"  OVERFIT  (calib > 80%, holdout < 30%):  {n_overfit}  "
          f"({100*n_overfit/n_done:.1f}%)", flush=True)
    print(f"  NEGATIVE (holdout worse than naive):  {n_negative}  "
          f"({100*n_negative/n_done:.1f}%)", flush=True)
    print(f"  No GPTQ available: {n_no_gptq}", flush=True)

    if rows:
        improves = [r["holdout_improve"] for r in rows]
        print(f"\nHoldout improvement statistics:", flush=True)
        print(f"  mean:    {np.mean(improves):>+7.2f}%", flush=True)
        print(f"  median:  {np.median(improves):>+7.2f}%", flush=True)
        print(f"  min:     {np.min(improves):>+7.2f}%", flush=True)
        print(f"  max:     {np.max(improves):>+7.2f}%", flush=True)
        print(f"  p25:     {np.percentile(improves, 25):>+7.2f}%", flush=True)
        print(f"  p75:     {np.percentile(improves, 75):>+7.2f}%", flush=True)

        # Compare to calibration
        calib_improves = [r["calib_improve"] for r in rows if r["calib_improve"] >= 0]
        if calib_improves:
            print(f"\nCalibration improvement (for comparison):", flush=True)
            print(f"  mean:    {np.mean(calib_improves):>+7.2f}%", flush=True)
            print(f"  median:  {np.median(calib_improves):>+7.2f}%", flush=True)

    print(f"\nLog -> {out_csv_path}", flush=True)
    print(f"\n{'='*70}", flush=True)

    if rows:
        mean_holdout = np.mean(improves)
        print("\nINTERPRETATION:", flush=True)
        if mean_holdout > 50:
            print(f"  GPTQ generalized strongly. V15 should be a real improvement over V13.", flush=True)
            print(f"  PROCEED to V15 generation.", flush=True)
        elif mean_holdout > 20:
            print(f"  GPTQ partially generalized. V15 likely improves over V13 modestly.", flush=True)
            print(f"  Worth running V15.", flush=True)
        elif mean_holdout > 0:
            print(f"  GPTQ marginally generalized. V15 might be ~equivalent to V13.", flush=True)
            print(f"  Consider re-running GPTQ with more calibration data, or skip V15.", flush=True)
        else:
            print(f"  GPTQ failed to generalize (or made things worse).", flush=True)
            print(f"  DO NOT run V15. Focus on V14 SmoothQuant or skip both.", flush=True)


if __name__ == "__main__":
    main()
