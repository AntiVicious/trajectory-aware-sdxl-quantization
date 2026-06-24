import torch
import torch.nn as nn
import gc
from diffusers import StableDiffusionXLPipeline

CHECKPOINT_PATH = "/home/project/Documents/Maharshi/gem/sdxl_bitlora_hybrid_v72_best.pt"
SAVE_PATH = "v72_inference.png"

class LoRALinear(nn.Module):
    def __init__(self, original_layer, lora_A, lora_B):
        super().__init__()
        self.weight = nn.Parameter(original_layer.weight.data.clone())
        self.bias = nn.Parameter(original_layer.bias.data.clone()) if original_layer.bias is not None else None
        self.lora_A = nn.Parameter(lora_A)
        self.lora_B = nn.Parameter(lora_B)

    def forward(self, x, *args, **kwargs):
        dtype = x.dtype
        base = nn.functional.linear(x, self.weight.to(dtype),
                                    self.bias.to(dtype) if self.bias is not None else None)
        delta = nn.functional.linear(
            nn.functional.linear(x, self.lora_A.to(dtype)),
            self.lora_B.to(dtype)
        )
        return base + delta

# Step 1: load everything on CPU only
print("Loading SDXL on CPU...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)  # no .to("cuda") yet

# Step 2: inject LoRA on CPU
print("Loading checkpoint on CPU...")
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

lora_a_keys = [k for k in ckpt if k.endswith('.lora_A')]
print(f"Injecting {len(lora_a_keys)} LoRA layers on CPU...")

for key in lora_a_keys:
    layer_path = key.replace('.lora_A', '')
    lora_A = ckpt[key].half()
    lora_B = ckpt[layer_path + '.lora_B'].half()

    parts = layer_path.split('.')
    module = pipe.unet
    for p in parts[:-1]:
        module = getattr(module, p) if not p.isdigit() else module[int(p)]
    layer_name = parts[-1]
    original_layer = getattr(module, layer_name)

    new_layer = LoRALinear(original_layer, lora_A, lora_B)
    setattr(module, layer_name, new_layer)

del ckpt
gc.collect()
print("Injection done.")

# Step 3: now move to GPU
print("Moving to CUDA...")
pipe.unet = pipe.unet.to("cuda")
pipe.vae = pipe.vae.to("cuda")
pipe.text_encoder = pipe.text_encoder.to("cuda")
pipe.text_encoder_2 = pipe.text_encoder_2.to("cuda")

pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()      # <-- add this
pipe.vae.enable_tiling()      # <-- and this

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