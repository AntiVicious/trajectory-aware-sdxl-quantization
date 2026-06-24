import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from diffusers import StableDiffusionXLPipeline

CKPT_PATH       = "/home/project/Documents/Maharshi/gem/sdxl_timestep_lora.pt"
SAVE_PATH       = "v12_timestep_lora.png"
TIMESTEP_SPLIT  = 500

class TernaryLoRASwitchable(nn.Module):
    """TernaryLoRA with two LoRAs that switch based on current timestep.
    The timestep is set externally before each forward pass."""
    current_timestep = 1000  # class variable — shared across all instances

    def __init__(self, w_skel, scale, bias, in_f, out_f, rank=32):
        super().__init__()
        self.register_buffer('w_skel', w_skel)
        self.register_buffer('scale', scale)
        if bias is not None:
            self.register_buffer('bias_buf', bias)
            self.has_bias = True
        else:
            self.has_bias = False
        # Two LoRAs
        self.lora_A_early = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B_early = nn.Parameter(torch.zeros(out_f, rank))
        self.lora_A_late  = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B_late  = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A_early, a=5**0.5)
        nn.init.kaiming_uniform_(self.lora_A_late,  a=5**0.5)

    def forward(self, x, *args, **kwargs):
        dtype = x.dtype
        w = (self.w_skel * self.scale).to(dtype)
        if TernaryLoRASwitchable.current_timestep >= TIMESTEP_SPLIT:
            w = w + (self.lora_B_early @ self.lora_A_early).to(dtype)
        else:
            w = w + (self.lora_B_late  @ self.lora_A_late ).to(dtype)
        bias = self.bias_buf.to(dtype) if self.has_bias else None
        return F.linear(x, w, bias)

print("Loading SDXL on CPU...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

print("Loading timestep-LoRA checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location="cpu")

# Find ternary skeletons
skel_keys = [k for k in ckpt if k.endswith('.w_skel')
             and 'mid_block' in k]
print(f"  Ternary layers: {len(skel_keys)}")

for sk in skel_keys:
    layer_path  = sk.replace('.w_skel', '')
    w_skel      = ckpt[sk].float()
    scale       = ckpt[layer_path + '.scale'].float()
    bias_k      = layer_path + '.bias'
    bias        = ckpt[bias_k].float() if bias_k in ckpt else None
    out_f, in_f = w_skel.shape

    lora_A_early = ckpt[f"{layer_path}.lora_A__early"].float()
    lora_B_early = ckpt[f"{layer_path}.lora_B__early"].float()
    lora_A_late  = ckpt[f"{layer_path}.lora_A__late"].float()
    lora_B_late  = ckpt[f"{layer_path}.lora_B__late"].float()
    rank         = lora_A_early.shape[0]

    new_layer = TernaryLoRASwitchable(w_skel, scale, bias,
                                      in_f, out_f, rank=rank)
    new_layer.lora_A_early = nn.Parameter(lora_A_early)
    new_layer.lora_B_early = nn.Parameter(lora_B_early)
    new_layer.lora_A_late  = nn.Parameter(lora_A_late)
    new_layer.lora_B_late  = nn.Parameter(lora_B_late)

    parts  = layer_path.split('.')
    module = pipe.unet
    for p in parts[:-1]:
        module = getattr(module, p) if not p.isdigit() else module[int(p)]
    setattr(module, parts[-1], new_layer)

del ckpt
gc.collect()
print("Injection done.")

# ── Hook into scheduler to update timestep ────────────────────────────
# Patch the unet forward to capture current timestep
original_forward = pipe.unet.forward

def forward_with_timestep(sample, timestep, *args, **kwargs):
    # Extract scalar timestep value
    if torch.is_tensor(timestep):
        t_val = timestep.flatten()[0].item()
    else:
        t_val = timestep
    TernaryLoRASwitchable.current_timestep = t_val
    return original_forward(sample, timestep, *args, **kwargs)

pipe.unet.forward = forward_with_timestep

print("Moving to CUDA...")
pipe.unet           = pipe.unet.to("cuda")
pipe.vae            = pipe.vae.to("cuda")
pipe.text_encoder   = pipe.text_encoder.to("cuda")
pipe.text_encoder_2 = pipe.text_encoder_2.to("cuda")
pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

torch.cuda.reset_peak_memory_stats()
generator = torch.Generator("cuda").manual_seed(42)

print("Generating (with timestep-aware LoRA switching)...")
with torch.inference_mode():
    image = pipe(
        "a photo of a dog sitting in a park, golden hour, photorealistic",
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
        height=768, width=768,
    ).images[0]

image.save(SAVE_PATH)
peak_vram = torch.cuda.max_memory_allocated() / 1024**2
print(f"Saved: {SAVE_PATH}")
print(f"Peak VRAM: {peak_vram:.0f} MB")