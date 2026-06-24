import torch
import gc
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

CACHE_PATH    = "/home/project/Documents/Maharshi/gem/coco_latents_cache.pt"
UP1_CACHE     = "/home/project/Documents/Maharshi/gem/upblock1_cache.pt"
N_SAMPLES     = 200

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
pipe.unet = pipe.unet.to("cuda")
pipe.unet.eval()
torch.cuda.empty_cache()

coco_cache = torch.load(CACHE_PATH, map_location="cpu")
print(f"COCO cache: {len(coco_cache)} samples")

up1_data  = []
captured = {}

def pre_hook_fn(module, args, kwargs):
    # Capture from kwargs since up_blocks passes everything as kwargs
    if 'hidden_states' in kwargs:
        captured['input'] = kwargs['hidden_states'].detach().cpu()
    elif len(args) > 0 and torch.is_tensor(args[0]):
        captured['input'] = args[0].detach().cpu()
    if 'temb' in kwargs and kwargs['temb'] is not None:
        captured['temb'] = kwargs['temb'].detach().cpu()
    else:
        captured['temb'] = None
    if 'encoder_hidden_states' in kwargs:
        captured['encoder_hidden_states'] = kwargs['encoder_hidden_states'].detach().cpu()
    return args, kwargs

def post_hook_fn(module, args, output):
    if 'input' in captured:
        out = output[0] if isinstance(output, tuple) else output
        captured['output'] = out.detach().cpu()

pre_hook  = pipe.unet.up_blocks[1].register_forward_pre_hook(pre_hook_fn, with_kwargs=True)
post_hook = pipe.unet.up_blocks[1].register_forward_hook(post_hook_fn)
hook = pipe.unet.up_blocks[1].register_forward_hook(hook_fn)

print(f"Caching {N_SAMPLES} up_blocks.1 activation pairs...")
for i, sample in enumerate(coco_cache[:N_SAMPLES]):
    latent = sample['latents'].to("cuda")
    pe     = sample['prompt_embeds'].to("cuda")
    ppe    = sample['pooled_embeds'].to("cuda")

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

    if 'input' in captured and 'output' in captured:
        up1_data.append({
            'input':  captured['input'].clone(),
            'temb':   captured['temb'].clone() if captured['temb'] is not None else None,
            'output': captured['output'].clone(),
            'encoder_hidden_states': captured.get('encoder_hidden_states',
                                    sample['prompt_embeds']).cpu(),
        })
        captured.clear()

    if (i + 1) % 20 == 0:
        print(f"  Cached {i+1}/{N_SAMPLES}")

pre_hook.remove()
post_hook.remove()
sample = up1_data[0]
print(f"\nCached {len(up1_data)} samples")
print(f"  input:  {sample['input'].shape}")
print(f"  output: {sample['output'].shape}")
print(f"  temb:   {sample['temb'].shape if sample['temb'] is not None else None}")

torch.save(up1_data, UP1_CACHE)
print(f"Saved: {UP1_CACHE}")