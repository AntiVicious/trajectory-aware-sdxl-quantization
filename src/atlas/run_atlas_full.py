"""
Full SDXL UNet sensitivity atlas with CFG and prefix-skip.

Iterates over all (layer, step_from) cells. For each cell:
  1. Load baseline x_{step_from} from cache
  2. Run denoising from step_from to 19 with INT4 swap on `layer`
  3. Decode, compare to baseline final image via CLIP-feature distance
  4. Average across 8 prompts, save to CSV

Resumable: already-completed cells are skipped on restart.

Usage:
    CUDA_VISIBLE_DEVICES=0 python run_atlas_full.py                    # full run
    CUDA_VISIBLE_DEVICES=0 python run_atlas_full.py --smoke            # 5 cells only
    CUDA_VISIBLE_DEVICES=0 python run_atlas_full.py --limit 100        # first 100 cells
"""
import argparse
import csv
import json
import os
import time
import traceback
import gc
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPModel
from torchvision import transforms

BASE_DIR = Path(os.environ.get("SDXL_DATA_ROOT", "."))
ATLAS_DIR = BASE_DIR / "trajectory" / "phase1_atlas_full"
BASELINE_PATH = ATLAS_DIR / "baseline" / "baseline_full.pt"
LAYERS_PATH = ATLAS_DIR / "layers_full.json"
RESULTS_DIR = ATLAS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_DIR / "atlas_full.csv"

DEVICE = "cuda"
DTYPE = torch.float16
NUM_STEPS = 20
HEIGHT = WIDTH = 768
GUIDANCE_SCALE = 7.5
STEPS = [0, 5, 10, 15, 19]

FIELDNAMES = [
    "layer", "step_from", "layer_type", "n_params",
    "clip_dist_mean", "clip_dist_std", "img_mse_mean",
    "n_prompts", "seconds"
]


# ============ Quantization ============

def quantize_int4_per_channel(w: torch.Tensor) -> torch.Tensor:
    orig_shape = w.shape
    orig_dtype = w.dtype
    w2 = w.reshape(w.shape[0], -1).float() if w.ndim == 4 else w.float()
    scale = (w2.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-8)
    q = (w2 / scale).round().clamp(-7, 7)
    return (q * scale).to(orig_dtype).reshape(orig_shape)


# ============ Step counter + surgical swap ============

class StepCounter:
    def __init__(self, start=0):
        self.current = start - 1  # pre-incremented in hook
    def advance(self):
        self.current += 1


class SurgicalSwap:
    """Per-layer swap where quantization kicks in at counter.current >= threshold."""
    def __init__(self, unet, layer_name, threshold, counter):
        self.unet = unet
        self.layer_name = layer_name
        self.threshold = threshold
        self.counter = counter
        self.module = None
        self.original_forward = None

    def __enter__(self):
        self.module = self.unet.get_submodule(self.layer_name)
        self.original_forward = self.module.forward
        w_fp16 = self.module.weight.data.detach().clone().to(DEVICE)
        w_quant = quantize_int4_per_channel(w_fp16).to(DEVICE)

        if isinstance(self.module, nn.Linear):
            bias = self.module.bias
            counter = self.counter
            thr = self.threshold
            def forward(x, *args, **kwargs):
                w = w_quant if counter.current >= thr else w_fp16
                return F.linear(x, w, bias)
            self.module.forward = forward
        elif isinstance(self.module, nn.Conv2d):
            bias = self.module.bias
            stride, padding = self.module.stride, self.module.padding
            dilation, groups = self.module.dilation, self.module.groups
            counter = self.counter
            thr = self.threshold
            def forward(x, *args, **kwargs):
                w = w_quant if counter.current >= thr else w_fp16
                return F.conv2d(x, w, bias, stride, padding, dilation, groups)
            self.module.forward = forward
        else:
            raise TypeError(f"Unsupported layer type: {type(self.module)}")
        return self

    def __exit__(self, *args):
        self.module.forward = self.original_forward
        torch.cuda.empty_cache()


# ============ CLIP distance ============

_clip_preprocess = transforms.Compose([
    transforms.Resize(224, antialias=True),
    transforms.CenterCrop(224),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ),
])


@torch.no_grad()
def clip_distance(clip_model, img_a, img_b):
    """img_a, img_b: [3, H, W] in [0, 1]. Returns 1 - cos(CLIP features)."""
    pa = _clip_preprocess(img_a).unsqueeze(0).to(DEVICE)
    pb = _clip_preprocess(img_b).unsqueeze(0).to(DEVICE)
    fa = clip_model.get_image_features(pixel_values=pa)
    fb = clip_model.get_image_features(pixel_values=pb)
    # Some transformers versions wrap this in BaseModelOutputWithPooling
    if hasattr(fa, "image_embeds"):
        fa = fa.image_embeds
    elif hasattr(fa, "pooler_output"):
        fa = fa.pooler_output
    elif hasattr(fa, "last_hidden_state"):
        fa = fa.last_hidden_state[:, 0]  # CLS token fallback
    if hasattr(fb, "image_embeds"):
        fb = fb.image_embeds
    elif hasattr(fb, "pooler_output"):
        fb = fb.pooler_output
    elif hasattr(fb, "last_hidden_state"):
        fb = fb.last_hidden_state[:, 0]
    fa = fa / fa.norm(dim=-1, keepdim=True)
    fb = fb / fb.norm(dim=-1, keepdim=True)
    return float((1.0 - (fa * fb).sum(dim=-1)).item())


# ============ Scheduler helpers ============

def load_euler_clean():
    import inspect
    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        filename="scheduler/scheduler_config.json",
    )
    with open(cfg_path) as f:
        cfg = json.load(f)
    valid = set(inspect.signature(EulerDiscreteScheduler.__init__).parameters.keys())
    valid.discard("self")
    return EulerDiscreteScheduler(**{k: v for k, v in cfg.items() if k in valid})


def set_scheduler_to_step(scheduler, step_from):
    """Reset scheduler and advance internal step_index to step_from."""
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    if step_from > 0:
        scheduler._init_step_index(scheduler.timesteps[step_from])
    return scheduler.timesteps


# ============ Denoising with CFG from a starting latent ============

@torch.no_grad()
def denoise_from(unet, scheduler, start_latent, step_from,
                 prompt_embeds, pooled, neg_prompt_embeds, neg_pooled,
                 add_time_ids):
    """Denoise from step_from to step NUM_STEPS-1 using CFG."""
    timesteps = set_scheduler_to_step(scheduler, step_from)
    latents = start_latent.to(DEVICE, dtype=DTYPE).clone()

    prompt_embeds_cfg = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
    pooled_cfg = torch.cat([neg_pooled, pooled], dim=0)
    add_time_ids_cfg = torch.cat([add_time_ids, add_time_ids], dim=0)
    added_cond = {"text_embeds": pooled_cfg, "time_ids": add_time_ids_cfg}

    for i in range(step_from, NUM_STEPS):
        t = timesteps[i]
        latent_input = torch.cat([latents, latents], dim=0)
        latent_input = scheduler.scale_model_input(latent_input, t)
        noise_pred = unet(
            latent_input, t,
            encoder_hidden_states=prompt_embeds_cfg,
            added_cond_kwargs=added_cond,
            return_dict=False,
        )[0]
        nu, nc = noise_pred.chunk(2)
        noise_pred_cfg = nu + GUIDANCE_SCALE * (nc - nu)
        latents = scheduler.step(noise_pred_cfg, t, latents, return_dict=False)[0]
    return latents


# ============ VAE decode ============

@torch.no_grad()
def decode_vae(vae, latents_batch):
    vae_fp32 = vae.to(torch.float32)
    latents = latents_batch.to(DEVICE, torch.float32) / vae_fp32.config.scaling_factor
    imgs = vae_fp32.decode(latents, return_dict=False)[0]
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    vae.to(DTYPE)
    return imgs.cpu().float()


# ============ Resumable CSV ============

def load_completed(csv_path):
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            done.add((row["layer"], int(row["step_from"])))
    return done


def append_row(csv_path, row):
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new:
            w.writeheader()
        w.writerow(row)


# ============ One cell ============

@torch.no_grad()
def run_one_cell(unet, vae, scheduler, clip_model, baseline,
                 layer_name, step_from, layer_meta):
    per_prompt = baseline["per_prompt"]
    neg_pe = baseline["neg_prompt_embeds"].to(DEVICE, dtype=DTYPE)
    neg_po = baseline["neg_pooled_embeds"].to(DEVICE, dtype=DTYPE)
    add_time_ids = baseline["add_time_ids"].to(DEVICE, dtype=DTYPE)

    counter = StepCounter(start=step_from)

    def pre_hook(module, args, kwargs):
        counter.advance()
    handle = unet.register_forward_pre_hook(pre_hook, with_kwargs=True)

    cell_latents = []

    try:
        with SurgicalSwap(unet, layer_name, step_from, counter):
            for ent in per_prompt:
                counter.current = step_from - 1   # reset per prompt
                pe = ent["prompt_embeds"].to(DEVICE, dtype=DTYPE)
                po = ent["pooled_embeds"].to(DEVICE, dtype=DTYPE)
                start_latent = ent["x_trajectory"][step_from]
                final_latent = denoise_from(
                    unet, scheduler, start_latent, step_from,
                    pe, po, neg_pe, neg_po, add_time_ids,
                )
                cell_latents.append(final_latent.cpu())
                torch.cuda.empty_cache()
    finally:
        handle.remove()

    clip_dists = []
    img_mses = []
    for i, lat in enumerate(cell_latents):
        img = decode_vae(vae, lat)
        base_img = per_prompt[i]["final_image"]
        d = clip_distance(clip_model, img.squeeze(0), base_img)
        clip_dists.append(d)
        img_mses.append(F.mse_loss(img.squeeze(0), base_img).item())
        torch.cuda.empty_cache()

    clip_dists_t = torch.tensor(clip_dists)
    return {
        "clip_dist_mean": float(clip_dists_t.mean().item()),
        "clip_dist_std": float(clip_dists_t.std().item()),
        "img_mse_mean": float(sum(img_mses) / len(img_mses)),
        "n_prompts": len(per_prompt),
    }


# ============ Main ============

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    with open(LAYERS_PATH) as f:
        layers_data = json.load(f)
    all_layers = layers_data["layers"]
    print(f"[atlas] {len(all_layers)} layers in enumeration")

    all_cells = [(L, s) for L in all_layers for s in STEPS]
    print(f"[atlas] Total cells: {len(all_cells)} ({len(all_layers)} x {len(STEPS)})")

    done = load_completed(OUT_CSV)
    remaining = [(L, s) for (L, s) in all_cells if (L["name"], s) not in done]
    print(f"[atlas] Already completed: {len(done)}")
    print(f"[atlas] Remaining: {len(remaining)}")

    if args.start:
        remaining = remaining[args.start:]
        print(f"[atlas] After --start={args.start}: {len(remaining)} cells")
    if args.smoke:
        remaining = remaining[:5]
        print(f"[atlas] Smoke mode: {len(remaining)} cells")
    elif args.limit:
        remaining = remaining[:args.limit]
        print(f"[atlas] Limited to {len(remaining)} cells")

    if not remaining:
        print("[atlas] Nothing to do.")
        return

    print(f"[atlas] Loading baseline from {BASELINE_PATH}...")
    if not BASELINE_PATH.exists():
        print("ERROR: baseline not found. Run build_baseline_full.py first.")
        return
    baseline = torch.load(BASELINE_PATH, map_location="cpu")
    print(f"[atlas] Baseline: {baseline['n_prompts']} prompts, "
          f"self_check_max={baseline.get('self_check_max_diff', '?')}")

    print("[atlas] Loading UNet + VAE...")
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet", torch_dtype=DTYPE,
        use_safetensors=True, variant="fp16",
    ).to(DEVICE)
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae", torch_dtype=DTYPE, use_safetensors=True,
    ).to(DEVICE)
    scheduler = load_euler_clean()

    print("[atlas] Loading CLIP...")
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14").to(DEVICE).eval()

    t_total = time.time()
    fail_count = 0

    for i, (layer_meta, step) in enumerate(remaining):
        layer_name = layer_meta["name"]
        print(f"\n[atlas] ({i+1}/{len(remaining)}) {layer_name} @ step={step}")
        t0 = time.time()
        try:
            r = run_one_cell(unet, vae, scheduler, clip_model, baseline,
                             layer_name, step, layer_meta)
            elapsed = round(time.time() - t0, 2)
            row = {
                "layer": layer_name,
                "step_from": step,
                "layer_type": layer_meta["type"],
                "n_params": layer_meta["n_params"],
                "clip_dist_mean": r["clip_dist_mean"],
                "clip_dist_std": r["clip_dist_std"],
                "img_mse_mean": r["img_mse_mean"],
                "n_prompts": r["n_prompts"],
                "seconds": elapsed,
            }
            append_row(OUT_CSV, row)
            print(f"    CLIP_dist={r['clip_dist_mean']:.4e}  "
                  f"img_MSE={r['img_mse_mean']:.4e}  ({elapsed}s)")
        except Exception as e:
            fail_count += 1
            print(f"    FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            if fail_count > 10:
                print("[atlas] Too many failures, aborting.")
                break
            continue

        if (i + 1) % 20 == 0:
            elapsed_min = (time.time() - t_total) / 60
            rate = (i + 1) / elapsed_min
            eta_hr = (len(remaining) - i - 1) / rate / 60
            print(f"\n[atlas] {i+1}/{len(remaining)}  "
                  f"{elapsed_min:.1f}min elapsed, ETA {eta_hr:.1f}hr")

    total_hr = (time.time() - t_total) / 3600
    print(f"\n[atlas] Finished. {len(remaining)} cells in {total_hr:.1f}hr")
    print(f"[atlas] Results -> {OUT_CSV}")


if __name__ == "__main__":
    main()