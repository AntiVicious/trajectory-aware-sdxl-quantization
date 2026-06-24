"""
Deeper diagnosis: dump shapes/dtypes of what pipe sends vs what we send.
Previous diag showed encoder_hidden_states max_diff=854 but it might have
been a shape mismatch, not a value mismatch.

Uses pipe.encode_prompt exclusively — same as the current build_baseline.
"""
import json
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

DEVICE = "cuda"
DTYPE = torch.float16
HEIGHT = WIDTH = 768
SEED = 42
NUM_STEPS = 20
GUIDANCE_SCALE = 7.5
PROMPT = "a photo of a dog sitting in a park, golden hour"


def dump(name, t):
    print(f"  {name}: shape={tuple(t.shape)}  dtype={t.dtype}  "
          f"min={t.min().item():.4f}  max={t.max().item():.4f}  "
          f"nonzero={(t.abs() > 1e-8).float().mean().item():.4f}")


pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=DTYPE, use_safetensors=True, variant="fp16",
).to(DEVICE)
pipe.set_progress_bar_config(disable=True)

# ============ Method A: what pipe ACTUALLY feeds to UNet ============
print("\n=== What pipe actually feeds to UNet at step 0 ===")
pipe_captured = {}

orig_fwd = pipe.unet.forward
def cap(sample, timestep, encoder_hidden_states, **kwargs):
    if "ehs" not in pipe_captured:
        pipe_captured["sample"] = sample.detach().cpu().clone()
        pipe_captured["ehs"] = encoder_hidden_states.detach().cpu().clone()
        ac = kwargs.get("added_cond_kwargs", {})
        if ac:
            pipe_captured["text_embeds"] = ac["text_embeds"].detach().cpu().clone()
            pipe_captured["time_ids"] = ac["time_ids"].detach().cpu().clone()
        pipe_captured["t"] = timestep.detach().cpu().clone() if torch.is_tensor(timestep) else float(timestep)
    return orig_fwd(sample, timestep, encoder_hidden_states, **kwargs)
pipe.unet.forward = cap

gen = torch.Generator(DEVICE).manual_seed(SEED)
_ = pipe(
    prompt=PROMPT, num_inference_steps=NUM_STEPS, guidance_scale=GUIDANCE_SCALE,
    generator=gen, height=HEIGHT, width=WIDTH, output_type="pt",
).images[0]
pipe.unet.forward = orig_fwd

print(f"  timestep: {pipe_captured['t']}")
dump("sample     ", pipe_captured["sample"])
dump("ehs        ", pipe_captured["ehs"])
dump("text_embeds", pipe_captured["text_embeds"])
dump("time_ids   ", pipe_captured["time_ids"])

# ============ Method B: what WE feed to UNet ============
print("\n=== What we would feed using pipe.encode_prompt ===")
pe, neg_pe, po, neg_po = pipe.encode_prompt(
    prompt=PROMPT, device=DEVICE, num_images_per_prompt=1,
    do_classifier_free_guidance=True, negative_prompt="",
)
add_time_ids = pipe._get_add_time_ids(
    (HEIGHT, WIDTH), (0, 0), (HEIGHT, WIDTH),
    dtype=DTYPE,
    text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
).to(DEVICE)

print(f"  Individual embeddings from pipe.encode_prompt:")
dump("pe         ", pe)
dump("neg_pe     ", neg_pe)
dump("po         ", po)
dump("neg_po     ", neg_po)
dump("add_time_ids", add_time_ids)

ehs_ours = torch.cat([neg_pe, pe], dim=0)
text_embeds_ours = torch.cat([neg_po, po], dim=0)
time_ids_ours = torch.cat([add_time_ids, add_time_ids], dim=0)

print(f"\n  After CFG concat:")
dump("ehs (ours)       ", ehs_ours)
dump("text_embeds (ours)", text_embeds_ours)
dump("time_ids (ours)  ", time_ids_ours)

# ============ Compare point-by-point ============
print("\n=== Comparison ===")
d_ehs = (pipe_captured["ehs"] - ehs_ours.cpu()).abs()
d_te = (pipe_captured["text_embeds"] - text_embeds_ours.cpu()).abs()
d_ti = (pipe_captured["time_ids"] - time_ids_ours.cpu()).abs()

print(f"  ehs diff:         max={d_ehs.max().item():.4e}  mean={d_ehs.mean().item():.4e}")
print(f"  text_embeds diff: max={d_te.max().item():.4e}  mean={d_te.mean().item():.4e}")
print(f"  time_ids diff:    max={d_ti.max().item():.4e}  mean={d_ti.mean().item():.4e}")

# Where is the ehs max diff?
if d_ehs.max().item() > 0.001:
    worst_idx = d_ehs.flatten().argmax().item()
    B, L, D = d_ehs.shape
    b = worst_idx // (L * D)
    l = (worst_idx % (L * D)) // D
    dd = worst_idx % D
    print(f"\n  Worst ehs diff at [batch={b}, pos={l}, dim={dd}]:")
    print(f"    pipe value:  {pipe_captured['ehs'][b, l, dd].item():.4f}")
    print(f"    ours value:  {ehs_ours.cpu()[b, l, dd].item():.4f}")
    # What's at batch 0 vs batch 1 of pipe?
    print(f"\n  Pipe ehs[0] (neg) row sums: "
          f"{pipe_captured['ehs'][0].abs().sum(dim=-1)[:5].tolist()}")
    print(f"  Pipe ehs[1] (pos) row sums: "
          f"{pipe_captured['ehs'][1].abs().sum(dim=-1)[:5].tolist()}")
    print(f"  Ours ehs[0] (neg) row sums: "
          f"{ehs_ours.cpu()[0].abs().sum(dim=-1)[:5].tolist()}")
    print(f"  Ours ehs[1] (pos) row sums: "
          f"{ehs_ours.cpu()[1].abs().sum(dim=-1)[:5].tolist()}")