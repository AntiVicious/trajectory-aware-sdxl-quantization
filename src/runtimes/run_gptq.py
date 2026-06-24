"""
Run GPTQ on each captured Linear layer.

GPTQ algorithm (per layer):
  Given X (calibration inputs [N, in_features]) and W [out, in],
  find quantized W_q minimizing ||XW^T - XW_q^T||^2.
  Uses inverse Hessian H^-1 = (X^T X)^-1 for closed-form error compensation.

  Algorithm:
    1. H = 2 X^T X / N + lambda*I
    2. Cholesky: H_inv_chol = chol(H_inv)
    3. For each column c in W^T (= row in W):
       a. Quantize W[:, c] to INT4 with per-row scale
       b. Compute error: err = (W[:, c] - W_q[:, c]) / H_inv_chol[c, c]
       c. Update W[:, c+1:] -= err * H_inv_chol[c, c+1:]  (compensate)
    4. Output: W_q [out, in], scales [out]

Output: gptq_quantized_weights/<layer_name>.pt
  Each: {"w_dequant": tensor [out, in] in fp16,    # already-dequantized for fast load
         "scales": tensor [out],                    # for reference
         "method": "gptq", "bits": 4}

Note: We store dequantized weights (fp16 values that lie on INT4 grid) rather than
packed INT4. This is for fast loading at generation time. Same memory at runtime
as V13/V14 since our generation script also uses dequantized form.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_gptq.py [--lambda 0.01] [--bits 4]
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

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "/home/project/Documents/Maharshi/gem"))
CAL_DIR = BASE / "trajectory" / "phase4_full_bench" / "gptq_calibration_data"
OUT_DIR = BASE / "trajectory" / "phase4_full_bench" / "gptq_quantized_weights"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    return name.replace(".", "__").replace("/", "__")


def gptq_quantize_layer(W: torch.Tensor, X: torch.Tensor,
                         bits: int = 4, damp: float = 0.01,
                         block_size: int = 128) -> tuple:
    """
    Args:
      W: [out_features, in_features] fp32
      X: [n_samples, in_features] fp32 calibration data
      bits: 4 (INT4 in [-7, 7] for symmetric per-row quant)
      damp: Hessian damping (lambda * trace(H) / d)
      block_size: process 128 columns at a time for memory efficiency

    Returns:
      W_q: [out_features, in_features] fp32 (dequantized values on INT4 grid)
      scales: [out_features, 1]
    """
    device = W.device
    out_features, in_features = W.shape
    n_samples = X.shape[0]

    qmax = 2 ** (bits - 1) - 1   # for bits=4 → qmax=7

    # ---- Hessian: H = 2 X^T X / n_samples (factor 2 from MSE derivative) ----
    H = 2.0 * (X.T @ X) / n_samples
    # Damping for invertibility
    diag_mean = torch.mean(torch.diag(H))
    damping = damp * diag_mean
    H[torch.arange(in_features), torch.arange(in_features)] += damping

    # ---- Mark zero columns dead ----
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W = W.clone()
    W[:, dead] = 0

    # ---- Inverse Hessian via Cholesky decomposition ----
    # H = L L^T → H^-1 = L^-T L^-1; we need upper-triangular factor of H^-1
    try:
        L = torch.linalg.cholesky(H)
    except Exception as e:
        # Add more damping and retry once
        H[torch.arange(in_features), torch.arange(in_features)] += diag_mean
        L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)  # upper-triangular factor of H^-1

    # ---- GPTQ main loop, in blocks for cache locality ----
    Q = torch.zeros_like(W)
    Losses = torch.zeros_like(W)

    # Compute per-row scale once (symmetric per-channel quant)
    # We'll refine if needed but standard GPTQ uses fixed scale derived from W
    scales = (W.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-8)

    for i1 in range(0, in_features, block_size):
        i2 = min(i1 + block_size, in_features)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for j in range(count):
            w = W1[:, j]
            d = Hinv1[j, j]
            # Quantize this column
            q = torch.round(w / scales.squeeze(1)).clamp(-qmax, qmax)
            wq = q * scales.squeeze(1)
            Q1[:, j] = wq
            Losses1[:, j] = (w - wq) ** 2 / d ** 2
            err = (w - wq) / d
            # Compensate the remaining columns in this block
            W1[:, j+1:] -= err.unsqueeze(1) * Hinv1[j, j+1:].unsqueeze(0)
            Err1[:, j] = err

        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2

        # Compensate columns AFTER this block using the accumulated errors
        if i2 < in_features:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    total_loss = float(Losses.sum().item())
    return Q, scales, total_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--start", type=int, default=0,
                    help="Skip the first N layers (for resume)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N layers (for testing)")
    args = ap.parse_args()

    # Load calibration metadata
    with open(CAL_DIR / "_meta.json") as f:
        meta = json.load(f)
    layer_names = meta["layer_names"]
    print(f"[gptq] {len(layer_names)} layers in calibration set")
    if args.limit:
        layer_names = layer_names[args.start:args.start + args.limit]
    elif args.start:
        layer_names = layer_names[args.start:]
    print(f"[gptq] Processing {len(layer_names)} layers")

    # Need actual UNet weights — load just the UNet
    print("[gptq] Loading SDXL UNet (for original weights)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    )
    unet = pipe.unet.to("cuda")
    del pipe.text_encoder, pipe.text_encoder_2, pipe.vae
    torch.cuda.empty_cache()

    log_path = BASE / "trajectory" / "phase4_full_bench" / "gptq_log.csv"
    write_header = not log_path.exists()
    f_log = open(log_path, "a", buffering=1)
    if write_header:
        f_log.write("layer,n_in,n_out,n_samples,gptq_loss,naive_loss,improvement_pct,seconds\n")

    t0 = time.time()
    for li, layer_name in enumerate(layer_names):
        try:
            module = unet.get_submodule(layer_name)
        except AttributeError:
            print(f"  [{li+1}] {layer_name}: unresolvable, skipping")
            continue
        if not isinstance(module, nn.Linear):
            continue

        cal_path = CAL_DIR / f"{safe_filename(layer_name)}.pt"
        if not cal_path.exists():
            print(f"  [{li+1}] {layer_name}: no calibration data, skipping")
            continue

        out_path = OUT_DIR / f"{safe_filename(layer_name)}.pt"
        if out_path.exists():
            continue   # already done

        t1 = time.time()
        # Load calibration X
        X = torch.load(cal_path, map_location="cuda", weights_only=False).to(torch.float32)
        W = module.weight.data.detach().to(torch.float32).to("cuda")

        n_in = W.shape[1]
        n_out = W.shape[0]
        n_samples = X.shape[0]

        # GPTQ-quantize
        try:
            Q, scales, gptq_loss = gptq_quantize_layer(
                W, X, bits=args.bits, damp=args.damp, block_size=args.block_size,
            )
        except Exception as e:
            print(f"  [{li+1}] {layer_name}: GPTQ failed ({e}), falling back to naive")
            qmax = 2 ** (args.bits - 1) - 1
            scales = (W.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-8)
            Q = torch.round(W / scales).clamp(-qmax, qmax) * scales
            gptq_loss = -1.0

        # Compute naive baseline error for comparison
        qmax = 2 ** (args.bits - 1) - 1
        scales_naive = (W.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-8)
        Q_naive = torch.round(W / scales_naive).clamp(-qmax, qmax) * scales_naive
        # Error on calibration data
        with torch.no_grad():
            Y_orig = X @ W.T
            Y_gptq = X @ Q.T
            Y_naive = X @ Q_naive.T
            naive_err = ((Y_orig - Y_naive) ** 2).mean().item()
            gptq_err = ((Y_orig - Y_gptq) ** 2).mean().item()
        improve_pct = 100.0 * (naive_err - gptq_err) / naive_err if naive_err > 0 else 0

        # Save
        torch.save({
            "w_dequant": Q.to(torch.float16).cpu(),
            "scales": scales.to(torch.float16).cpu(),
            "method": "gptq",
            "bits": args.bits,
            "calib_n_samples": n_samples,
            "naive_err": naive_err,
            "gptq_err": gptq_err,
            "improve_pct": improve_pct,
        }, out_path)

        elapsed_layer = time.time() - t1
        f_log.write(f"{layer_name},{n_in},{n_out},{n_samples},"
                    f"{gptq_err:.6e},{naive_err:.6e},{improve_pct:.2f},{elapsed_layer:.2f}\n")

        if (li + 1) % 25 == 0 or li == 0:
            elapsed = (time.time() - t0) / 60
            rate = (li + 1) / elapsed if elapsed > 0 else 0
            eta = (len(layer_names) - li - 1) / rate / 60 if rate > 0 else 0
            print(f"  [{li+1}/{len(layer_names)}]  "
                  f"layer={layer_name[:50]}  "
                  f"GPTQ improve={improve_pct:+.1f}%  "
                  f"({elapsed_layer:.1f}s/layer, {elapsed:.1f}min total, ETA {eta:.1f}h)")

        # Free memory
        del X, W, Q, scales, Q_naive, Y_orig, Y_gptq, Y_naive
        torch.cuda.empty_cache()

    f_log.close()
    print(f"\n[gptq] Done in {(time.time()-t0)/60:.1f} min")
    print(f"[gptq] Quantized weights -> {OUT_DIR}")
    print(f"[gptq] Per-layer log -> {log_path}")


if __name__ == "__main__":
    main()
