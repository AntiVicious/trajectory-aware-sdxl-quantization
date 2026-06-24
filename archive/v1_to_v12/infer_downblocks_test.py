import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from diffusers import StableDiffusionXLPipeline

SAVE_PATH = "v7_downblock_test.png"

print("Loading SDXL on CPU...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

# Naive ternarization of down_blocks.2 only
print("Ternarizing down_blocks.2...")
count = 0
for name, module in pipe.unet.down_blocks[2].named_modules():
    if isinstance(module, nn.Linear):
        w     = module.weight.data.float()
        scale = w.abs().mean()
        module.weight.data = (w.sign() * scale).half()
        count += 1
print(f"Ternarized {count} layers in down_blocks.2.")

print("Moving to CUDA...")
pipe.unet           = pipe.unet.to("cuda")
pipe.vae            = pipe.vae.to("cuda")
pipe.text_encoder   = pipe.text_encoder.to("cuda")
pipe.text_encoder_2 = pipe.text_encoder_2.to("cuda")
pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

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

image.save(SAVE_PATH)
print(f"Saved: {SAVE_PATH}")