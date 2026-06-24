import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from diffusers import StableDiffusionXLPipeline

CKPT_BEST = "/home/project/Documents/Maharshi/gem/sdxl_v111_mid_r64_1000.pt"
SAVE_PATH = "hybrid_full_test.png"

class TernaryLoRALinear(nn.Module):
    def __init__(self, w_skel, scale, bias, in_f, out_f, rank=64):
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

def quantize_int4(module):
    w     = module.weight.data.float()
    scale = w.abs().max() / 7.0
    scale = scale.clamp(min=1e-8)
    w_q   = (w / scale).round().clamp(-8, 7)
    module.weight.data = (w_q * scale).half()

print("Loading SDXL on CPU...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

# ── Step 1: inject ternary+LoRA into mid_block ────────────────────────
print("Injecting distilled ternary mid_block...")
ckpt      = torch.load(CKPT_BEST, map_location="cpu")
skel_keys = [k for k in ckpt if k.endswith('.w_skel')]
for sk in skel_keys:
    layer_path  = sk.replace('.w_skel', '')
    w_skel      = ckpt[sk].float()
    scale       = ckpt[layer_path + '.scale'].float()
    lora_A      = ckpt[layer_path + '.lora_A'].float()
    lora_B      = ckpt[layer_path + '.lora_B'].float()
    bias_k      = layer_path + '.bias'
    bias        = ckpt[bias_k].float() if bias_k in ckpt else None
    out_f, in_f = w_skel.shape
    rank        = lora_A.shape[0]
    new_layer        = TernaryLoRALinear(w_skel, scale, bias,
                                         in_f, out_f, rank=rank)
    new_layer.lora_A = nn.Parameter(lora_A)
    new_layer.lora_B = nn.Parameter(lora_B)
    parts  = layer_path.split('.')
    module = pipe.unet
    for p in parts[:-1]:
        module = getattr(module, p) if not p.isdigit() else module[int(p)]
    setattr(module, parts[-1], new_layer)
del ckpt; gc.collect()
print(f"  Injected {len(skel_keys)} ternary layers.")

# ── Step 2: INT4 all linear layers outside mid_block ─────────────────
print("Applying INT4 to all linear layers outside mid_block...")
int4_count = 0
for name, module in pipe.unet.named_modules():
    if isinstance(module, nn.Linear) and 'mid_block' not in name:
        quantize_int4(module)
        int4_count += 1
print(f"  Quantized {int4_count} linear layers to INT4.")

# ── Step 3: measure theoretical sizes ────────────────────────────────
ternary_params = sum(
    m.w_skel.numel()
    for _, m in pipe.unet.named_modules()
    if isinstance(m, TernaryLoRALinear)
)
total_linear = sum(
    m.weight.numel()
    for _, m in pipe.unet.named_modules()
    if isinstance(m, nn.Linear) and not isinstance(m, TernaryLoRALinear)
)
conv_params = sum(
    m.weight.numel()
    for _, m in pipe.unet.named_modules()
    if isinstance(m, nn.Conv2d)
)
other_params = sum(p.numel() for p in pipe.unet.parameters()) \
             - ternary_params - total_linear - conv_params

original_mb   = sum(p.numel() for p in pipe.unet.parameters()) * 2 / 1024**2
compressed_mb = (ternary_params * 0.2 +
                 total_linear   * 0.5 +
                 conv_params    * 2   +
                 other_params   * 2) / 1024**2
reduction_pct = (1 - compressed_mb / original_mb) * 100

print(f"\n=== Compression Stats ===")
print(f"Original FP16:     {original_mb:.0f} MB")
print(f"Compressed:        {compressed_mb:.0f} MB")
print(f"Reduction:         {reduction_pct:.1f}%")
print(f"Ternary (mid):     {ternary_params/1e6:.1f}M params")
print(f"INT4 (linear):     {total_linear/1e6:.1f}M params")
print(f"FP16 (conv+norm):  {(conv_params+other_params)/1e6:.1f}M params")

# ── Step 4: generate ─────────────────────────────────────────────────
print("\nMoving to CUDA...")
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
        height=768, width=768,
    ).images[0]

peak_vram = torch.cuda.max_memory_allocated() / 1024**2
image.save(SAVE_PATH)
print(f"Saved: {SAVE_PATH}")
print(f"Peak VRAM: {peak_vram:.0f} MB")