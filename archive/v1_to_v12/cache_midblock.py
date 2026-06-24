import torch
import gc
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

CACHE_PATH     = "/home/project/Documents/Maharshi/gem/coco_latents_cache.pt"
MIDBLOCK_CACHE = "/home/project/Documents/Maharshi/gem/midblock_cache.pt"
N_SAMPLES      = 200

print("Loading SDXL pipeline...")
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

# Move only what we need to GPU
pipe.unet = pipe.unet.to("cuda")
pipe.unet.eval()
torch.cuda.empty_cache()

# Load COCO latents cache
coco_cache = torch.load(CACHE_PATH, map_location="cpu")
print(f"COCO cache: {len(coco_cache)} samples")

# Hook to capture mid_block input and output
midblock_data = []
captured      = {}

def hook_input(module, input, output):
    # input[0] is hidden_states going INTO mid_block
    captured['input']  = input[0].detach().cpu()
    captured['temb']   = input[1].detach().cpu() if len(input) > 1 and input[1] is not None else None
    captured['output'] = output.detach().cpu() if torch.is_tensor(output) else output[0].detach().cpu()

hook = pipe.unet.mid_block.register_forward_hook(hook_input)

print(f"Caching {N_SAMPLES} mid-block input/output pairs...")

for i, sample in enumerate(coco_cache[:N_SAMPLES]):
    latent = sample['latents'].to("cuda")
    pe     = sample['prompt_embeds'].to("cuda")
    ppe    = sample['pooled_embeds'].to("cuda")

    # Add noise at random timestep
    noise     = torch.randn_like(latent)
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (1,), device="cuda"
    ).long()
    noisy = scheduler.add_noise(latent, noise, timesteps)

    add_kwargs = {
        "text_embeds": ppe,
        "time_ids": torch.tensor(
            [[1024, 1024, 0, 0, 1024, 1024]],
            device="cuda", dtype=torch.float16
        )
    }

    with torch.inference_mode():
        pipe.unet(
            noisy, timesteps,
            encoder_hidden_states=pe,
            added_cond_kwargs=add_kwargs
        )

    if 'input' in captured:
        midblock_data.append({
            'input':  captured['input'].clone(),
            'temb':   captured['temb'].clone() if captured['temb'] is not None else None,
            'output': captured['output'].clone(),
            'encoder_hidden_states': pe.detach().cpu(),
        })
        captured.clear()

    if (i + 1) % 20 == 0:
        print(f"  Cached {i+1}/{N_SAMPLES}")

hook.remove()

# Verify shapes
sample = midblock_data[0]
print(f"\nCached {len(midblock_data)} samples")
print(f"  input shape:   {sample['input'].shape}")
print(f"  output shape:  {sample['output'].shape}")
print(f"  temb shape:    {sample['temb'].shape if sample['temb'] is not None else None}")
print(f"  encoder shape: {sample['encoder_hidden_states'].shape}")

torch.save(midblock_data, MIDBLOCK_CACHE)
print(f"\nSaved to: {MIDBLOCK_CACHE}")
print("Done.")