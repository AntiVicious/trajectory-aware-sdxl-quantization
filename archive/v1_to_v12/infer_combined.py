import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from diffusers import StableDiffusionXLPipeline

CKPT_PATH = "/home/project/Documents/Maharshi/gem/sdxl_v111_v72_combined.pt"
SAVE_PATH = "v5_combined_inference.png"

class TernaryLoRALinear(nn.Module):
    def __init__(self, w_skel, scale, bias, in_f, out_f, rank=32):
        super().__init__()
        self.register_buffer('w_skel', w_skel)
        self.register_buffer('scale', scale)
        if bias is not None:
            self.register_buffer('bias_buf', bias)
            self.has_bias = True
        else:
            self.has_bias = False
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x, *args, **kwargs):
        dtype = x.dtype
        w    = (self.w_skel * self.scale).to(dtype)
        w    = w + (self.lora_B @ self.lora_A).to(dtype)
        bias = self.bias_buf.to(dtype) if self.has_bias else None
        return F.linear(x, w, bias)

print("Loading SDXL on CPU...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

print("Loading combined checkpoint...")
ckpt      = torch.load(CKPT_PATH, map_location="cpu")
skel_keys = [k for k in ckpt if k.endswith('.w_skel')]
print(f"Found {len(skel_keys)} ternary layers.")

for skel_key in skel_keys:
    layer_path  = skel_key.replace('.w_skel', '')
    w_skel      = ckpt[skel_key].float()
    scale       = ckpt[layer_path + '.scale'].float()
    lora_A      = ckpt[layer_path + '.lora_A'].float()
    lora_B      = ckpt[layer_path + '.lora_B'].float()
    bias_k      = layer_path + '.bias'
    bias        = ckpt[bias_k].float() if bias_k in ckpt else None
    out_f, in_f = w_skel.shape
    rank        = lora_A.shape[0]

    new_layer         = TernaryLoRALinear(w_skel, scale, bias, in_f, out_f, rank=rank)
    new_layer.lora_A  = nn.Parameter(lora_A)
    new_layer.lora_B  = nn.Parameter(lora_B)

    parts  = layer_path.split('.')
    module = pipe.unet
    for p in parts[:-1]:
        module = getattr(module, p) if not p.isdigit() else module[int(p)]
    setattr(module, parts[-1], new_layer)

del ckpt
gc.collect()
print("Injection done.")

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

print("Generating...")
with torch.inference_mode():
    image = pipe(
        "a photo of a dog sitting in a park, golden hour, photorealistic",
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
        height=768,
        width=768,
    ).images[0]

peak_vram = torch.cuda.max_memory_allocated() / 1024**2
image.save(SAVE_PATH)
print(f"Saved: {SAVE_PATH}")
print(f"Peak VRAM: {peak_vram:.0f} MB")