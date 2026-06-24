import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from diffusers import StableDiffusionXLPipeline

BASE       = "/home/project/Documents/Maharshi/gem/"
CKPT_BEST  = BASE + "sdxl_v111_mid_r64_1000.pt"
STATS_PATH = BASE + "smooth_activation_stats.pt"
SAVE_PATH  = "smooth_hybrid_test.png"
ALPHA      = 0.5  # SmoothQuant migration strength (0=no smooth, 1=full)

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

class SmoothInt4Linear(nn.Module):
    """INT4 quantized linear with SmoothQuant-style pre-scaling.
    The smoothing factor s divides activations and multiplies weights."""
    def __init__(self, original_linear, act_max, alpha=0.5):
        super().__init__()
        w = original_linear.weight.data.float()

        act_max = act_max.float().clamp(min=1e-5)
        w_max   = w.abs().amax(dim=0).clamp(min=1e-5)

        s = (act_max.pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)

        w_smooth = w * s.unsqueeze(0)

        scale = w_smooth.abs().max(dim=1, keepdim=True).values / 7.0
        scale = scale.clamp(min=1e-8)
        w_q   = (w_smooth / scale).round().clamp(-8, 7)
        w_deq = (w_q * scale).half()

        self.register_buffer('weight', w_deq)
        self.register_buffer('smooth_scale', s.half())

        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.data.clone())
            self.has_bias = True
        else:
            self.has_bias = False

        # Add attributes that diffusers may inspect
        self.in_features  = original_linear.in_features
        self.out_features = original_linear.out_features
    def forward(self, x, *args, **kwargs):
        # Divide activation by smooth scale (channel-wise)
        x_smooth = x / self.smooth_scale.to(x.dtype)
        bias     = self.bias.to(x.dtype) if self.has_bias else None
        return F.linear(x_smooth, self.weight.to(x.dtype), bias)

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

# ── Step 2: load activation statistics ────────────────────────────────
print("Loading activation statistics...")
stats = torch.load(STATS_PATH, map_location="cpu")
print(f"  Stats for {len(stats)} layers.")

# ── Step 3: apply SmoothQuant INT4 to all non-mid linear layers ───────
print(f"Applying SmoothQuant INT4 (alpha={ALPHA}) to linear layers...")
count   = 0
missing = 0
for name, module in pipe.unet.named_modules():
    if (isinstance(module, nn.Linear) and
            not isinstance(module, TernaryLoRALinear) and
            'mid_block' not in name):
        if name in stats:
            new_layer = SmoothInt4Linear(module, stats[name], alpha=ALPHA)
            parts  = name.split('.')
            parent = pipe.unet
            for p in parts[:-1]:
                parent = getattr(parent, p) if not p.isdigit() \
                         else parent[int(p)]
            setattr(parent, parts[-1], new_layer)
            count += 1
        else:
            missing += 1

print(f"  SmoothQuant applied: {count} layers | Missing stats: {missing}")

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