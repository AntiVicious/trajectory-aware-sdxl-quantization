"""
Diagnostic: why is V19 slower than V13?

Tests several hypotheses:
  1. Per-image latency excluding setup (just the inference loop)
  2. Whether the quantized tensors have different properties (NaN, inf, dtype, contiguity)
  3. Whether matmul speed depends on tensor content

Usage:
    CUDA_VISIBLE_DEVICES=1 python -u diagnose_v19_speed.py
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import time
from pathlib import Path
import json
import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
ALLOC_DIR = BASE / "trajectory" / "phase3_allocator_full"
NUM_STEPS = 20


def quantize_int4_per_channel_naive(w):
    qmax = 7
    if w.ndim == 4:
        scales = w.abs().amax(dim=(1, 2, 3), keepdim=True) / qmax
    else:
        scales = w.abs().amax(dim=1, keepdim=True) / qmax
    scales = scales.clamp(min=1e-8)
    return (w / scales).round().clamp(-qmax, qmax) * scales


def quantize_outlier_preserved_int4(w, outlier_pct=0.1):
    flat = w.abs().flatten()
    n = flat.numel()
    k = max(1, int(n * outlier_pct / 100.0))
    if k >= n:
        return w
    threshold = torch.topk(flat, k, largest=True).values.min()
    mask = w.abs() >= threshold
    w_for_quant = w.clone()
    w_for_quant[mask] = 0.0
    w_quant = quantize_int4_per_channel_naive(w_for_quant)
    w_quant[mask] = w[mask]
    return w_quant


def quantize_ternary(w):
    if w.ndim == 4:
        scales = w.abs().mean(dim=(1, 2, 3), keepdim=True)
    else:
        scales = w.abs().mean(dim=1, keepdim=True)
    scales = scales.clamp(min=1e-8)
    threshold = scales * 0.7
    q = torch.zeros_like(w)
    q[w > threshold] = 1.0
    q[w < -threshold] = -1.0
    return q * scales


def benchmark_single_matmul(w, n_iters=50, x_shape=None):
    """Time a representative matmul with this weight on GPU."""
    if x_shape is None:
        x_shape = (1, 4096, w.shape[-1] if w.ndim == 2 else w.shape[1])
    x = torch.randn(x_shape, dtype=w.dtype, device=w.device)
    if w.ndim == 2:
        # Linear
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iters):
            _ = torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / n_iters
    else:
        # Conv2d - skip detailed test
        elapsed = 0.0
    return elapsed


def inspect_tensor(w, name):
    """Report tensor properties."""
    return {
        "name": name,
        "shape": list(w.shape),
        "dtype": str(w.dtype),
        "device": str(w.device),
        "contiguous": w.is_contiguous(),
        "n_unique_vals": min(w.unique().numel(), 1000000),
        "min": float(w.min()),
        "max": float(w.max()),
        "mean": float(w.mean()),
        "std": float(w.std()),
        "abs_mean": float(w.abs().mean()),
    }


def main():
    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
    ).to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    # Pick a representative layer to test
    sample_layer_name = None
    for name, m in pipe.unet.named_modules():
        if isinstance(m, nn.Linear) and m.weight.numel() > 100000:
            sample_layer_name = name
            break

    print(f"\nSampling layer: {sample_layer_name}")
    sample_layer = pipe.unet.get_submodule(sample_layer_name)
    w = sample_layer.weight.data.clone()
    print(f"  Original weight shape: {list(w.shape)}, dtype: {w.dtype}")

    # Quantize three ways
    w_naive = quantize_int4_per_channel_naive(w.cpu().float()).to(w.dtype).to(w.device)
    w_outlier = quantize_outlier_preserved_int4(w.cpu().float()).to(w.dtype).to(w.device)

    print("\n=== Tensor inspection ===")
    for q, name in [(w, "FP16 original"), (w_naive, "V13 naive INT4"), (w_outlier, "V19 outlier INT4")]:
        info = inspect_tensor(q, name)
        print(f"  [{name:<25}] shape={info['shape']}, dtype={info['dtype']}, "
              f"contig={info['contiguous']}, n_unique={info['n_unique_vals']}, "
              f"abs_mean={info['abs_mean']:.4f}")

    # Benchmark matmul speed for each
    print("\n=== Matmul benchmark (50 iters each, single layer) ===")
    for q, name in [(w, "FP16 original"), (w_naive, "V13 naive INT4"), (w_outlier, "V19 outlier INT4")]:
        t = benchmark_single_matmul(q, n_iters=50)
        print(f"  [{name:<25}] {t*1000:.3f} ms per matmul")

    # Now check with a deeper test: full UNet forward with each quantizer
    print("\n=== Full UNet forward (10 iters each, no scheduler) ===")
    # Set up a dummy input
    bs = 2
    x = torch.randn(bs, 4, 96, 96, dtype=torch.float16, device="cuda")
    t = torch.tensor([500] * bs, device="cuda")
    enc = torch.randn(bs, 77, 2048, dtype=torch.float16, device="cuda")

    # Get encoder hidden states from pipe's text encoder for valid input
    pipe.text_encoder.to("cuda")
    pipe.text_encoder_2.to("cuda")

    # Need correct added_cond_kwargs for SDXL
    text_embeds = torch.randn(bs, 1280, dtype=torch.float16, device="cuda")
    time_ids = torch.tensor([[768., 768., 0., 0., 768., 768.]] * bs,
                             dtype=torch.float16, device="cuda")
    added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}

    # Warmup
    with torch.no_grad():
        _ = pipe.unet(x, t, encoder_hidden_states=enc,
                       added_cond_kwargs=added_cond_kwargs).sample

    # Baseline (FP16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = pipe.unet(x, t, encoder_hidden_states=enc,
                           added_cond_kwargs=added_cond_kwargs).sample
    torch.cuda.synchronize()
    fp16_time = (time.time() - t0) / 10
    print(f"  FP16 baseline:        {fp16_time*1000:.1f} ms per forward")

    # Now apply V13 swap and measure
    def apply_swap_to_unet(unet, allocation, quantizer_fn, mid_quant=None):
        """Apply trajectory swap with given quantizer. Returns originals dict."""
        originals = {}
        for layer_name, threshold in allocation:
            try:
                module = unet.get_submodule(layer_name)
            except AttributeError:
                continue
            if not hasattr(module, 'weight'):
                continue
            originals[layer_name] = module.forward
            if threshold >= NUM_STEPS:
                continue

            w_orig = module.weight.data
            with torch.no_grad():
                if mid_quant is not None and layer_name.startswith("mid_block"):
                    w_quant = mid_quant(w_orig.cpu().float())
                else:
                    w_quant = quantizer_fn(w_orig.cpu().float())
                w_quant = w_quant.to(w_orig.dtype).to(w_orig.device)

            if isinstance(module, nn.Linear):
                bias = module.bias
                # Force commit at step 0 always (for benchmark)
                def make_fwd(wq=w_quant, b=bias):
                    def fwd(x):
                        return torch.nn.functional.linear(x, wq, b)
                    return fwd
                module.forward = make_fwd()
            elif isinstance(module, nn.Conv2d):
                bias = module.bias
                stride = module.stride; padding = module.padding
                dilation = module.dilation; groups = module.groups
                def make_conv(wq=w_quant, b=bias, s=stride, p=padding,
                               d=dilation, g=groups):
                    def fwd(x):
                        return torch.nn.functional.conv2d(x, wq, b, s, p, d, g)
                    return fwd
                module.forward = make_conv()
        return originals

    def restore(unet, originals):
        for name, orig in originals.items():
            try:
                m = unet.get_submodule(name)
                m.forward = orig
            except AttributeError:
                pass

    # Load V13-70 allocation
    with open(ALLOC_DIR / "allocation_70pct_full.json") as f:
        alloc_data = json.load(f)
    assignments = []
    for a in alloc_data["allocations"]:
        if a["choice"] == "FP16":
            t = NUM_STEPS
        else:
            t = int(a["choice"].split("t")[1])
        assignments.append((a["layer"], t))

    # V13 timing
    print("  Applying V13 quantization (this takes ~30s)...")
    originals = apply_swap_to_unet(pipe.unet, assignments, quantize_int4_per_channel_naive)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = pipe.unet(x, t, encoder_hidden_states=enc,
                           added_cond_kwargs=added_cond_kwargs).sample
    torch.cuda.synchronize()
    v13_time = (time.time() - t0) / 10
    print(f"  V13 (naive INT4):     {v13_time*1000:.1f} ms per forward")
    restore(pipe.unet, originals)

    # V19 timing
    print("  Applying V19 quantization (this takes ~60s, slower due to outlier preservation)...")
    originals = apply_swap_to_unet(pipe.unet, assignments,
                                     quantize_outlier_preserved_int4,
                                     mid_quant=quantize_ternary)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = pipe.unet(x, t, encoder_hidden_states=enc,
                           added_cond_kwargs=added_cond_kwargs).sample
    torch.cuda.synchronize()
    v19_time = (time.time() - t0) / 10
    print(f"  V19 (outlier INT4):   {v19_time*1000:.1f} ms per forward")
    restore(pipe.unet, originals)

    print(f"\n=== Summary ===")
    print(f"  FP16:  {fp16_time*1000:.1f} ms")
    print(f"  V13:   {v13_time*1000:.1f} ms ({v13_time/fp16_time:.2f}× FP16)")
    print(f"  V19:   {v19_time*1000:.1f} ms ({v19_time/fp16_time:.2f}× FP16, "
          f"{v19_time/v13_time:.2f}× V13)")


if __name__ == "__main__":
    main()
