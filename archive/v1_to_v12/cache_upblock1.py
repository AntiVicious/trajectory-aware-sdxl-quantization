import torch
import gc
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

CACHE_PATH = "/home/project/Documents/Maharshi/gem/coco_latents_cache.pt"
UP1_CACHE  = "/home/project/Documents/Maharshi/gem/upblock1_cache_v2.pt"
N_SAMPLES  = 200

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

up1_data = []
captured = {}

def pre_hook_fn(module, args, kwargs):
    # Capture all inputs including res_hidden_states_tuple
    if 'hidden_states' in kwargs:
        captured['input'] = kwargs['hidden_states'].detach().cpu()
    elif len(args) > 0 and torch.is_tensor(args[0]):
        captured['input'] = args[0].detach().cpu()

    captured['temb'] = kwargs['temb'].detach().cpu() \
        if 'temb' in kwargs and kwargs['temb'] is not None else None

    if 'encoder_hidden_states' in kwargs and kwargs['encoder_hidden_states'] is not None:
        captured['encoder_hidden_states'] = kwargs['encoder_hidden_states'].detach().cpu()

    # Capture residual hidden states (skip connections from encoder)
    if 'res_hidden_states_tuple' in kwargs and kwargs['res_hidden_states_tuple'] is not None:
        captured['res_hidden_states_tuple'] = tuple(
            r.detach().cpu() for r in kwargs['res_hidden_states_tuple']
        )
    return args, kwargs

def post_hook_fn(module, args, output):
    if 'input' in captured:
        out = output[0] if isinstance(output, tuple) else output
        captured['output'] = out.detach().cpu()

pre_hook  = pipe.unet.up_blocks[1].register_forward_pre_hook(
    pre_hook_fn, with_kwargs=True
)
post_hook = pipe.unet.up_blocks[1].register_forward_hook(post_hook_fn)

print(f"Caching {N_SAMPLES} samples...")

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
        enc = captured.get('encoder_hidden_states', sample['prompt_embeds'])
        up1_data.append({
            'input':                 captured['input'].clone(),
            'temb':                  captured['temb'].clone() if captured['temb'] is not None else None,
            'output':                captured['output'].clone(),
            'encoder_hidden_states': enc.cpu() if torch.is_tensor(enc) else enc,
            'res_hidden_states_tuple': captured.get('res_hidden_states_tuple', None),
        })
    captured.clear()

    if (i + 1) % 20 == 0:
        print(f"  Cached {i+1}/{N_SAMPLES} | collected: {len(up1_data)}")

pre_hook.remove()
post_hook.remove()

print(f"\nTotal collected: {len(up1_data)}")
if len(up1_data) > 0:
    s = up1_data[0]
    print(f"  input:   {s['input'].shape}")
    print(f"  output:  {s['output'].shape}")
    print(f"  temb:    {s['temb'].shape if s['temb'] is not None else None}")
    print(f"  encoder: {s['encoder_hidden_states'].shape}")
    if s['res_hidden_states_tuple'] is not None:
        print(f"  res_hidden_states: {len(s['res_hidden_states_tuple'])} tensors")
        for j, r in enumerate(s['res_hidden_states_tuple']):
            print(f"    [{j}]: {r.shape}")
    torch.save(up1_data, UP1_CACHE)
    print(f"\nSaved: {UP1_CACHE}")