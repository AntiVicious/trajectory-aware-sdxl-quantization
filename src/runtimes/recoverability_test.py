"""
Recoverability test: is V13-40's quantization error low-rank enough for LoRA to fix?

This adapts Gemini's recoverability test to OUR specific case: V13-40 (which has
DreamSim 0.166, broken quality) — measures whether the per-layer quantization
error is recoverable by a low-rank adapter.

If recovery > 60-70%, LoRA could realistically fix V13-40 without retraining.
If recovery < 40%, the error is too high-rank — same problem as 1-bit.

Three error magnitudes are tested:
  1. Naive INT4 per-channel (V13's quantizer at t=0)
  2. NF4 (V20/V22 quantizer)
  3. NF4 limited to V13-40's allocation (V22-40's exact effective error)

Usage:
    CUDA_VISIBLE_DEVICES=1 python -u recoverability_test.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
from pathlib import Path
import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline

BASE = Path.home() / "Documents" / "Maharshi" / "gem"
ALLOC_DIR = BASE / "trajectory" / "phase3_allocator_full"

RANKS_TO_TEST = [8, 16, 32, 64]


# ===================== Quantizers =====================

def quantize_int4_per_channel_naive(w):
    qmax = 7
    if w.ndim == 4:
        scales = w.abs().amax(dim=(1, 2, 3), keepdim=True) / qmax
    else:
        scales = w.abs().amax(dim=1, keepdim=True) / qmax
    scales = scales.clamp(min=1e-8)
    return (w / scales).round().clamp(-qmax, qmax) * scales


def quantize_nf4_simulated(w):
    """Simulate NF4 by snapping each weight to nearest NF4 grid point.
    The actual bnb implementation does this with optimized kernels."""
    # NF4 grid (Normal-distribution-based, from QLoRA paper)
    nf4_grid = torch.tensor([
        -1.0, -0.6961928, -0.5250730, -0.39491748,
        -0.28444138, -0.18477343, -0.09105003, 0.0,
        0.07958029, 0.16093020, 0.24611229, 0.33791524,
        0.44070983, 0.56261723, 0.72295684, 1.0
    ], device=w.device)

    # Per-block scaling (block size 64, like bnb)
    block_size = 64
    w_flat = w.flatten()
    n_blocks = (w_flat.numel() + block_size - 1) // block_size
    n_padded = n_blocks * block_size
    if w_flat.numel() < n_padded:
        w_flat = torch.cat([w_flat, torch.zeros(n_padded - w_flat.numel(), device=w.device)])
    w_blocks = w_flat.view(n_blocks, block_size)

    # Per-block max-abs scale
    scales = w_blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    w_normalized = w_blocks / scales

    # Snap each value to nearest NF4 grid point
    distances = (w_normalized.unsqueeze(-1) - nf4_grid.view(1, 1, -1)).abs()
    indices = distances.argmin(dim=-1)
    w_quant_normalized = nf4_grid[indices]
    w_quant_blocks = w_quant_normalized * scales

    w_quant = w_quant_blocks.flatten()[:w.numel()].view(w.shape)
    return w_quant


# ===================== Recoverability test =====================

def measure_recoverability(error_matrix, rank):
    """How well can a rank-K matrix approximate this error?"""
    if error_matrix.numel() == 0:
        return 0.0
    if error_matrix.ndim != 2:
        # Conv2d: reshape to 2D for SVD
        error_matrix = error_matrix.reshape(error_matrix.shape[0], -1)
    if min(error_matrix.shape) < rank:
        return 100.0  # Trivially recoverable since rank exceeds matrix dim

    try:
        # Low-rank approximation via SVD
        U, S, V = torch.svd_lowrank(error_matrix.float(), q=rank)
        reconstructed = U @ torch.diag(S) @ V.T

        residual = error_matrix.float() - reconstructed
        orig_norm = torch.norm(error_matrix.float())
        if orig_norm < 1e-8:
            return 100.0
        residual_norm = torch.norm(residual)
        recovery = (1 - (residual_norm / orig_norm)) * 100
        return recovery.item()
    except Exception as e:
        return -1.0


def load_v13_allocation(pct):
    p = ALLOC_DIR / f"allocation_{pct}pct_full.json"
    with open(p) as f:
        alloc = json.load(f)
    layer_to_threshold = {}
    for a in alloc["allocations"]:
        if a["choice"] == "FP16":
            t = 20  # NUM_STEPS
        else:
            t = int(a["choice"].split("t")[1])
        layer_to_threshold[a["layer"]] = t
    return layer_to_threshold


def main():
    print("Loading SDXL...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
    ).to("cuda")
    unet = pipe.unet

    # Categorize layers
    layer_to_threshold_70 = load_v13_allocation(70)
    layer_to_threshold_40 = load_v13_allocation(40)

    # Get all Linear layers in UNet
    all_linear = []
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear) and module.weight.numel() > 1000:
            all_linear.append((name, module))

    print(f"Total Linear layers: {len(all_linear)}")

    # === Test 1: Recoverability of INT4 vs NF4 errors across all linear layers ===
    print("\n=== Test 1: Average recoverability by quantizer type ===")
    print(f"{'Quantizer':<25} {'Rank':>6} {'Mean recovery %':>18} {'Min %':>10} {'Max %':>10}")
    print("-" * 70)

    sample_layers = all_linear[::20]  # subsample every 20th for speed
    print(f"  (sampling {len(sample_layers)} of {len(all_linear)} layers)\n")

    for quant_name, quant_fn in [("Naive INT4 per-channel", quantize_int4_per_channel_naive),
                                   ("NF4 (simulated)", quantize_nf4_simulated)]:
        for rank in RANKS_TO_TEST:
            recoveries = []
            for name, module in sample_layers:
                w = module.weight.data
                w_q = quant_fn(w.float())
                err = w.float() - w_q
                rec = measure_recoverability(err, rank)
                if rec >= 0:
                    recoveries.append(rec)
            if recoveries:
                print(f"{quant_name:<25} rank={rank:>2}   "
                      f"{sum(recoveries)/len(recoveries):>16.2f}%   "
                      f"{min(recoveries):>8.2f}%   {max(recoveries):>8.2f}%")
        print()

    # === Test 2: Recoverability per V13-40 commit-step group ===
    # Layers that commit at t=0 are quantized for ALL steps - these are the
    # "lossy" ones. Their error matters most.
    print("\n=== Test 2: V13-40 specific - which layers are most damaging? ===")

    groups = {0: [], 5: [], 10: [], 15: [], 19: [], 20: []}
    for name, module in all_linear:
        if name in layer_to_threshold_40:
            t = layer_to_threshold_40[name]
            groups[t].append((name, module))

    print(f"{'V13-40 t=0 group:':<25} {len(groups[0])} layers (quantized from step 0)")
    print(f"{'V13-40 t=5 group:':<25} {len(groups[5])} layers")
    print(f"{'V13-40 t=10 group:':<25} {len(groups[10])} layers")
    print(f"{'V13-40 t=15 group:':<25} {len(groups[15])} layers")
    print(f"{'V13-40 t=19 group:':<25} {len(groups[19])} layers")
    print(f"{'FP16 throughout:':<25} {len(groups[20])} layers")
    print()

    for t, layers in [(0, groups[0]), (5, groups[5]), (10, groups[10])]:
        if not layers:
            continue
        print(f"V13-40 commit step t={t}: {len(layers)} layers, sampling 30")
        sample = layers[:30]
        for rank in [16, 32, 64]:
            recoveries_int4 = []
            recoveries_nf4 = []
            for name, module in sample:
                w = module.weight.data.float()
                err_int4 = w - quantize_int4_per_channel_naive(w)
                err_nf4 = w - quantize_nf4_simulated(w)
                r_int4 = measure_recoverability(err_int4, rank)
                r_nf4 = measure_recoverability(err_nf4, rank)
                if r_int4 >= 0: recoveries_int4.append(r_int4)
                if r_nf4 >= 0: recoveries_nf4.append(r_nf4)
            avg_int4 = sum(recoveries_int4)/len(recoveries_int4) if recoveries_int4 else 0
            avg_nf4 = sum(recoveries_nf4)/len(recoveries_nf4) if recoveries_nf4 else 0
            print(f"  rank={rank:>2}: INT4 recovery {avg_int4:>5.2f}%, NF4 recovery {avg_nf4:>5.2f}%")
        print()

    # === Final verdict ===
    print("=== INTERPRETATION ===")
    print("Recovery > 60%: LoRA can likely fix this quantization error well")
    print("Recovery 40-60%: LoRA helps but won't fully recover")
    print("Recovery < 40%: Error is too high-rank for LoRA")
    print()
    print("Compare to Gemini's V13-style 1-bit test: scored 21.5%")
    print("If our INT4/NF4 errors recover > 60%, LoRA correction is feasible")


if __name__ == "__main__":
    main()
