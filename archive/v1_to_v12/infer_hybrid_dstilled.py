import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from diffusers import StableDiffusionXLPipeline

CKPT_PATH = "/home/project/Documents/Maharshi/gem/sdxl_hybrid_distilled_full.pt"
SAVE_PATH = "hybrid_distilled_full.png"

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

class INT4LoRALinear(nn.Module):
    def __init__(self, w_q, scale, bias, in_f, out_f, rank=8):
        super().__init__()
        self.register_buffer('w_q',   w_q)
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
        w    = (self.w_q * self.scale).to(dtype)
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

print("Loading checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location="cpu")

# ── Inject TernaryLoRA for mid_block ──────────────────────────────────
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
print(f"  Injected {len(skel_keys)} ternary layers (mid_block).")

# ── Inject INT4LoRA for other blocks ──────────────────────────────────
wq_keys = [k for k in ckpt if k.endswith('.w_q')]
injected = 0
for wq_key in wq_keys:
    layer_path = wq_key.replace('.w_q', '')
    w_q        = ckpt[wq_key]
    scale      = ckpt[layer_path + '.scale']
    lora_A     = ckpt[layer_path + '.lora_A'].float()
    lora_B     = ckpt[layer_path + '.lora_B'].float()
    bias_k     = layer_path + '.bias'
    bias       = ckpt[bias_k] if bias_k in ckpt else None
    out_f, in_f = w_q.shape
    rank        = lora_A.shape[0]

    new_layer        = INT4LoRALinear(w_q, scale, bias,
                                      in_f, out_f, rank=rank)
    new_layer.lora_A = nn.Parameter(lora_A)
    new_layer.lora_B = nn.Parameter(lora_B)

    parts  = layer_path.split('.')
    module = pipe.unet
    for p in parts[:-1]:
        module = getattr(module, p) if not p.isdigit() else module[int(p)]
    setattr(module, parts[-1], new_layer)
    injected += 1

print(f"  Injected {injected} INT4LoRA layers.")

# ── Apply naive INT4 to remaining linear layers ───────────────────────
# (down_blocks.0, up_blocks.2 — too small to distill)
naive_count = 0
for name, module in pipe.unet.named_modules():
    if (isinstance(module, nn.Linear) and
            not isinstance(module, (TernaryLoRALinear, INT4LoRALinear))):
        w     = module.weight.data.float()
        scale = w.abs().max(dim=1, keepdim=True).values / 7.0
        scale = scale.clamp(min=1e-8)
        w_q   = (w / scale).round().clamp(-8, 7)
        module.weight.data = (w_q * scale).half()
        naive_count += 1
print(f"  Naive INT4: {naive_count} remaining layers.")

del ckpt; gc.collect()

# ── Compression stats ─────────────────────────────────────────────────
ternary_params = sum(m.w_skel.numel()
    for _, m in pipe.unet.named_modules()
    if isinstance(m, TernaryLoRALinear))
int4_lora_params = sum(m.w_q.numel()
    for _, m in pipe.unet.named_modules()
    if isinstance(m, INT4LoRALinear))
int4_naive_params = sum(m.weight.numel()
    for _, m in pipe.unet.named_modules()
    if isinstance(m, nn.Linear) and
    not isinstance(m, (TernaryLoRALinear, INT4LoRALinear)))
other_params = (sum(p.numel() for p in pipe.unet.parameters())
                - ternary_params - int4_lora_params - int4_naive_params)

original_mb   = sum(p.numel()
                    for p in pipe.unet.parameters()) * 2 / 1024**2
compressed_mb = (ternary_params   * 0.2 +
                 int4_lora_params * 0.5 +
                 int4_naive_params* 0.5 +
                 other_params     * 2) / 1024**2
reduction_pct = (1 - compressed_mb / original_mb) * 100

print(f"\n=== Compression Stats ===")
print(f"Original:    {original_mb:.0f} MB")
print(f"Compressed:  {compressed_mb:.0f} MB")
print(f"Reduction:   {reduction_pct:.1f}%")
print(f"  Ternary (mid):      {ternary_params/1e6:.1f}M")
print(f"  INT4+LoRA (distil): {int4_lora_params/1e6:.1f}M")
print(f"  INT4 naive (small): {int4_naive_params/1e6:.1f}M")
print(f"  FP16 (conv+norm):   {other_params/1e6:.1f}M")

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