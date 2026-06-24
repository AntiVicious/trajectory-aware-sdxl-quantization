"""
Diagnose the self-check failure by isolating possible causes.

Runs pipe(...) and our custom loop side-by-side, saving intermediates
at each stage so we can see exactly where they diverge.

Usage:
    CUDA_VISIBLE_DEVICES=0 python diagnose_selfcheck.py
"""
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler,
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from torchvision.utils import save_image

BASE = Path("/home/project/Documents/Maharshi/gem")
OUT = BASE / "trajectory" / "phase1_atlas_full" / "baseline" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda"
DTYPE = torch.float16
NUM_STEPS = 20
HEIGHT = WIDTH = 768
GUIDANCE_SCALE = 7.5
SEED = 42
PROMPT = "a photo of a dog sitting in a park, golden hour"


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


# =============================================================================
# STEP 1: Use pipe() to encode prompts ourselves, capture everything
# =============================================================================

@torch.no_grad()
def run_pipe_capture(prompt, seed):
    """Run pipe() and capture its initial noise, embeddings, and final image.
    Uses hooks to intercept the pipe's internal state."""
    print("\n[diag] STEP 1: Running pipe() with hooks...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=DTYPE, use_safetensors=True, variant="fp16",
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)

    # Use pipe's own encode_prompt method to get reference embeddings
    (prompt_embeds, negative_prompt_embeds,
     pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=prompt,
        device=DEVICE,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt="",
    )

    print(f"  pipe prompt_embeds shape: {prompt_embeds.shape}")
    print(f"  pipe pooled shape: {pooled_prompt_embeds.shape}")
    print(f"  pipe neg_prompt_embeds shape: {negative_prompt_embeds.shape}")
    print(f"  pipe neg_pooled shape: {negative_pooled_prompt_embeds.shape}")

    # Hook: capture initial noise tensor
    captured = {"init_noise": None, "post_step_latents": []}

    orig_step = pipe.scheduler.step
    def hooked_step(*args, **kwargs):
        result = orig_step(*args, **kwargs)
        captured["post_step_latents"].append(result[0].detach().cpu().clone())
        return result
    pipe.scheduler.step = hooked_step

    # Run pipe
    gen = torch.Generator(DEVICE).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=gen,
        height=HEIGHT, width=WIDTH,
        output_type="pt",
        return_dict=True,
    )
    img_pipe = result.images[0].cpu().float()
    save_image(img_pipe, OUT / "diag_pipe.png")

    pipe.scheduler.step = orig_step

    # Also capture time_ids — pipe computes its own
    # For SDXL, _get_add_time_ids is called; we can reconstruct
    orig_w, orig_h = WIDTH, HEIGHT
    target_w, target_h = WIDTH, HEIGHT
    add_time_ids = pipe._get_add_time_ids(
        (orig_h, orig_w), (0, 0), (target_h, target_w),
        dtype=DTYPE, text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
    )
    print(f"  pipe add_time_ids: {add_time_ids}  shape={add_time_ids.shape}")

    # Snapshot the embeddings
    ref_data = {
        "prompt_embeds": prompt_embeds.cpu(),
        "negative_prompt_embeds": negative_prompt_embeds.cpu(),
        "pooled_prompt_embeds": pooled_prompt_embeds.cpu(),
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds.cpu(),
        "add_time_ids": add_time_ids.cpu(),
        "post_step_latents": captured["post_step_latents"],
        "final_image": img_pipe,
    }
    del pipe
    torch.cuda.empty_cache()
    return ref_data


# =============================================================================
# STEP 2: Run our custom loop with SAME embeddings from pipe, compare
# =============================================================================

@torch.no_grad()
def run_custom_with_ref_embeds(ref_data, seed):
    """Run our custom loop using pipe's captured embeddings + time_ids.
    Isolates whether the divergence is in embeddings vs the denoising loop."""
    print("\n[diag] STEP 2: Running custom loop with pipe's embeddings...")

    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="unet", torch_dtype=DTYPE, use_safetensors=True, variant="fp16",
    ).to(DEVICE)
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae", torch_dtype=DTYPE, use_safetensors=True,
    ).to(DEVICE)
    scheduler = load_euler_clean()

    pe = ref_data["prompt_embeds"].to(DEVICE)
    neg_pe = ref_data["negative_prompt_embeds"].to(DEVICE)
    po = ref_data["pooled_prompt_embeds"].to(DEVICE)
    neg_po = ref_data["negative_pooled_prompt_embeds"].to(DEVICE)
    add_time_ids = ref_data["add_time_ids"].to(DEVICE)

    # Identical noise generation as our custom loop
    gen = torch.Generator(DEVICE).manual_seed(seed)
    latents = torch.randn(
        (1, 4, HEIGHT // 8, WIDTH // 8),
        generator=gen, device=DEVICE, dtype=DTYPE,
    )

    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    timesteps = scheduler.timesteps
    latents = latents * scheduler.init_noise_sigma

    prompt_embeds_cfg = torch.cat([neg_pe, pe], dim=0)
    pooled_cfg = torch.cat([neg_po, po], dim=0)
    add_time_ids_cfg = torch.cat([add_time_ids, add_time_ids], dim=0)
    added_cond = {"text_embeds": pooled_cfg, "time_ids": add_time_ids_cfg}

    captured_latents = []
    for i, t in enumerate(timesteps):
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
        captured_latents.append(latents.detach().cpu().clone())

    # Decode
    vae_fp32 = vae.to(torch.float32)
    lat_fp32 = latents.to(torch.float32) / vae_fp32.config.scaling_factor
    img = vae_fp32.decode(lat_fp32, return_dict=False)[0]
    img = (img / 2 + 0.5).clamp(0, 1)[0].cpu().float()
    save_image(img, OUT / "diag_custom_same_embeds.png")

    # Compare final images
    diff_img = (img - ref_data["final_image"]).abs()
    print(f"  Final image diff (pipe vs custom w/ pipe's embeds):")
    print(f"    max={diff_img.max().item():.4f}  mean={diff_img.mean().item():.4f}")

    # Compare latents step by step
    print(f"\n  Per-step latent divergence:")
    ref_lats = ref_data["post_step_latents"]
    for i in [0, 1, 5, 10, 15, 19]:
        if i < len(ref_lats) and i < len(captured_latents):
            d = (ref_lats[i].float() - captured_latents[i].float()).abs()
            print(f"    step {i:2d}: max={d.max().item():.4e}  mean={d.mean().item():.4e}")

    del unet, vae
    torch.cuda.empty_cache()
    return img, captured_latents


# =============================================================================
# STEP 3: Compare our encode_prompt against pipe's encode_prompt
# =============================================================================

@torch.no_grad()
def compare_embeddings(prompt, ref_data):
    print("\n[diag] STEP 3: Compare our text encoder output to pipe's...")
    te1 = CLIPTextModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder",
        torch_dtype=DTYPE, use_safetensors=True, variant="fp16",
    ).to(DEVICE)
    te2 = CLIPTextModelWithProjection.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="text_encoder_2",
        torch_dtype=DTYPE, use_safetensors=True, variant="fp16",
    ).to(DEVICE)
    tok1 = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2")

    def my_encode(text):
        t1 = tok1(text, padding="max_length", max_length=77,
                  truncation=True, return_tensors="pt").input_ids.to(DEVICE)
        t2 = tok2(text, padding="max_length", max_length=77,
                  truncation=True, return_tensors="pt").input_ids.to(DEVICE)
        out1 = te1(t1, output_hidden_states=True)
        embeds1 = out1.hidden_states[-2]
        out2 = te2(t2, output_hidden_states=True)
        embeds2 = out2.hidden_states[-2]
        pooled = out2.text_embeds
        return torch.cat([embeds1, embeds2], dim=-1), pooled

    my_pe, my_po = my_encode(prompt)
    my_neg_pe, my_neg_po = my_encode("")

    ref_pe = ref_data["prompt_embeds"].to(DEVICE)
    ref_po = ref_data["pooled_prompt_embeds"].to(DEVICE)
    ref_neg_pe = ref_data["negative_prompt_embeds"].to(DEVICE)
    ref_neg_po = ref_data["negative_pooled_prompt_embeds"].to(DEVICE)

    def cmp(name, mine, ref):
        d = (mine - ref).abs()
        cos = F.cosine_similarity(mine.flatten().unsqueeze(0),
                                   ref.flatten().unsqueeze(0)).item()
        print(f"  {name}:  shape={tuple(mine.shape)} vs {tuple(ref.shape)}")
        print(f"    max_diff={d.max().item():.4e}  mean_diff={d.mean().item():.4e}  "
              f"cos_sim={cos:.6f}")

    cmp("prompt_embeds       ", my_pe, ref_pe)
    cmp("pooled_prompt_embeds", my_po, ref_po)
    cmp("neg_prompt_embeds   ", my_neg_pe, ref_neg_pe)
    cmp("neg_pooled          ", my_neg_po, ref_neg_po)


def main():
    ref_data = run_pipe_capture(PROMPT, SEED)
    img_custom, custom_lats = run_custom_with_ref_embeds(ref_data, SEED)
    compare_embeddings(PROMPT, ref_data)

    print(f"\n[diag] Outputs saved to: {OUT}")
    print(f"[diag] Compare visually:")
    print(f"  {OUT}/diag_pipe.png  (pipe output)")
    print(f"  {OUT}/diag_custom_same_embeds.png  (custom loop w/ pipe's embeds)")
    print(f"\n[diag] Interpretation:")
    print(f"  - If step-2 diff is near 0, divergence is in our encode_prompt.")
    print(f"  - If step-2 diff is still large, divergence is in our denoise loop.")
    print(f"  - Step-3 shows exactly how our embeddings differ from pipe's.")


if __name__ == "__main__":
    main()
