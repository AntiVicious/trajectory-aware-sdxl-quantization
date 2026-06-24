import torch
import torch.nn as nn
import gc
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

CACHE_PATH  = "/home/project/Documents/Maharshi/gem/coco_latents_cache.pt"
STATS_PATH  = "/home/project/Documents/Maharshi/gem/smooth_activation_stats.pt"
N_SAMPLES   = 50  # enough for reliable statistics

print("Loading SDXL...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
scheduler = DDPMScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="scheduler"
)
pipe.unet = pipe.unet.to("cuda")
pipe.unet.eval()

# Collect per-channel activation max across all linear layers
# EXCEPT mid_block (which is ternary)
activation_stats = {}  # layer_name -> running max tensor

def make_hook(name):
    def hook(module, input, output):
        # input[0] is the activation going into this linear layer
        x = input[0]
        # Flatten batch and spatial dims, keep channel dim
        if x.dim() == 4:  # (B, C, H, W) - unlikely for linear but just in case
            x = x.flatten(2).transpose(1, 2).flatten(0, 1)  # (B*H*W, C)
        elif x.dim() == 3:  # (B, seq, C)
            x = x.flatten(0, 1)  # (B*seq, C)
        # Now x is (N, C), take max absolute value per channel
        ch_max = x.abs().amax(dim=0).float().cpu()
        if name in activation_stats:
            activation_stats[name] = torch.maximum(
                activation_stats[name], ch_max
            )
        else:
            activation_stats[name] = ch_max
    return hook

# Register hooks on all Linear layers OUTSIDE mid_block
hooks = []
for name, module in pipe.unet.named_modules():
    if isinstance(module, nn.Linear) and 'mid_block' not in name:
        hooks.append(module.register_forward_hook(make_hook(name)))

print(f"Registered hooks on {len(hooks)} INT4-target layers")

# Run calibration data
coco = torch.load(CACHE_PATH, map_location="cpu")

print(f"Running {N_SAMPLES} calibration forward passes...")
for i, sample in enumerate(coco[:N_SAMPLES]):
    latent = sample['latents'].to("cuda")
    pe     = sample['prompt_embeds'].to("cuda")
    ppe    = sample['pooled_embeds'].to("cuda")

    # Sample random timestep
    t_val  = torch.randint(0, 1000, (1,)).item()
    noise  = torch.randn_like(latent)
    t      = torch.tensor([t_val], device="cuda").long()
    noisy  = scheduler.add_noise(latent, noise, t)

    add_kwargs = {
        "text_embeds": ppe,
        "time_ids": torch.tensor(
            [[1024,1024,0,0,1024,1024]],
            device="cuda", dtype=torch.float16
        )
    }

    with torch.inference_mode():
        pipe.unet(noisy, t, encoder_hidden_states=pe,
                  added_cond_kwargs=add_kwargs)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{N_SAMPLES}")

for h in hooks:
    h.remove()

print(f"\nCollected stats for {len(activation_stats)} layers")

# Show some stats
print("\nSample activation magnitudes:")
sample_layers = list(activation_stats.keys())[:5]
for name in sample_layers:
    s = activation_stats[name]
    print(f"  {name}:")
    print(f"    shape={s.shape}, "
          f"min={s.min():.4f}, max={s.max():.4f}, "
          f"mean={s.mean():.4f}")
    # Check outlier ratio — max/mean tells us how "spiky" activations are
    outlier_ratio = (s.max() / s.mean()).item()
    print(f"    outlier ratio (max/mean): {outlier_ratio:.2f}x")

torch.save(activation_stats, STATS_PATH)
print(f"\nSaved: {STATS_PATH}")
print("Done.")