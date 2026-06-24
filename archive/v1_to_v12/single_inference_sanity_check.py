import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

CHECKPOINT_PATH = "sdxl_bitlora_hybrid_v111_best.pt"
SAVE_PATH = "sanity_check.png"

# --- Rebuild your BitLoRA layer (copy your exact class from training script) ---
import torch.nn as nn

class BitLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=32, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_skel = nn.Parameter(torch.zeros(out_features, in_features))
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale  = nn.Parameter(torch.ones(1))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x, *args, **kwargs):
        # Reconstruct ternary + LoRA weight
        w_ternary = self.w_skel.sign()
        w_lora    = (self.lora_B @ self.lora_A) * self.scale
        w         = (w_ternary + w_lora).to(x.dtype)
        out       = nn.functional.linear(x, w)
        if self.bias is not None:
            out = out + self.bias.to(x.dtype)
        return out

# --- Load base SDXL ---
print("Loading base SDXL pipeline...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

# --- Inject BitLoRA layers into mid_block ---
print("Injecting ternary layers...")
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

def replace_with_bitlora(model, ckpt, prefix=""):
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        key_skel = f"{full_name}.w_skel"
        if key_skel in ckpt:
            # Get dimensions from checkpoint
            w_skel = ckpt[key_skel]
            out_f, in_f = w_skel.shape
            lora_A = ckpt[f"{full_name}.lora_A"]
            rank = lora_A.shape[0]
            has_bias = f"{full_name}.bias" in ckpt

            new_layer = BitLoRALinear(in_f, out_f, rank=rank, bias=has_bias)
            new_layer.w_skel.data = w_skel.half()
            new_layer.lora_A.data = ckpt[f"{full_name}.lora_A"].half()
            new_layer.lora_B.data = ckpt[f"{full_name}.lora_B"].half()
            new_layer.scale.data  = ckpt[f"{full_name}.scale"].half()
            if has_bias:
                new_layer.bias.data = ckpt[f"{full_name}.bias"].half()

            # Set it on the parent
            parent = model
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], new_layer)
        else:
            replace_with_bitlora(module, ckpt, full_name)

replace_with_bitlora(pipe.unet.mid_block, ckpt, prefix="mid_block")
print(f"Injection done.")

# --- Run inference ---
pipe.enable_vae_slicing()
pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()  # streams model blocks to GPU on demand

prompt = "a photo of a dog sitting in a park, golden hour, photorealistic"
print(f"Generating image...")

torch.cuda.reset_peak_memory_stats()
generator = torch.Generator("cuda").manual_seed(42)

with torch.inference_mode():
    image = pipe(
        prompt,
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]

peak_vram = torch.cuda.max_memory_allocated() / 1024**2
image.save(SAVE_PATH)

print(f"\nImage saved to: {SAVE_PATH}")
print(f"Peak VRAM used: {peak_vram:.0f} MB")
print(f"Done.")