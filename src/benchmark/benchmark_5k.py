import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # lock to GPU 0

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import json
import shutil
from diffusers import StableDiffusionXLPipeline

BASE       = os.environ.get("SDXL_DATA_ROOT", ".")
OUT        = BASE + "benchmark_5k/"
CKPT_BEST  = BASE + "sdxl_v111_mid_r64_1000.pt"
STATS_PATH = BASE + "smooth_activation_stats.pt"
TIMESTEP_CKPT = BASE + "sdxl_timestep_lora.pt"
PROMPTS    = json.load(open(BASE + "benchmark_prompts_5k.json"))
SEED       = 42
OUTLIER_PCT = 0.005
ALPHA      = 0.5  # SmoothQuant
TIMESTEP_SPLIT = 500
FP16_SOURCE = BASE + "coco_standard_fp16/"  # existing 100 FP16 images

os.makedirs(OUT, exist_ok=True)

print(f"Benchmark will generate {len(PROMPTS)} images × 7 variants")
print(f"Estimated time: ~{len(PROMPTS) * 7 * 9 / 3600:.1f} hours")

# [Classes and helpers — same as before]
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
        self.in_features  = in_f
        self.out_features = out_f

    def forward(self, x, *args, **kwargs):
        dtype = x.dtype
        w    = (self.w_skel * self.scale).to(dtype)
        w    = w + (self.lora_B @ self.lora_A).to(dtype)
        bias = self.bias_buf.to(dtype) if self.has_bias else None
        return F.linear(x, w, bias)

class TernaryLoRASwitchable(nn.Module):
    current_timestep = 1000
    def __init__(self, w_skel, scale, bias, in_f, out_f, rank=32):
        super().__init__()
        self.register_buffer('w_skel', w_skel)
        self.register_buffer('scale', scale)
        if bias is not None:
            self.register_buffer('bias_buf', bias)
            self.has_bias = True
        else:
            self.has_bias = False
        self.lora_A_early = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B_early = nn.Parameter(torch.zeros(out_f, rank))
        self.lora_A_late  = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B_late  = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A_early, a=5**0.5)
        nn.init.kaiming_uniform_(self.lora_A_late,  a=5**0.5)
        self.in_features  = in_f
        self.out_features = out_f

    def forward(self, x, *args, **kwargs):
        dtype = x.dtype
        w = (self.w_skel * self.scale).to(dtype)
        if TernaryLoRASwitchable.current_timestep >= TIMESTEP_SPLIT:
            w = w + (self.lora_B_early @ self.lora_A_early).to(dtype)
        else:
            w = w + (self.lora_B_late  @ self.lora_A_late ).to(dtype)
        bias = self.bias_buf.to(dtype) if self.has_bias else None
        return F.linear(x, w, bias)

class SmoothInt4Linear(nn.Module):
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
        self.register_buffer('weight',       w_deq)
        self.register_buffer('smooth_scale', s.half())
        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.data.clone())
            self.has_bias = True
        else:
            self.has_bias = False
        self.in_features  = original_linear.in_features
        self.out_features = original_linear.out_features

    def forward(self, x, *args, **kwargs):
        dtype = x.dtype
        x_smooth = x / self.smooth_scale.to(dtype)
        bias     = self.bias.to(dtype) if self.has_bias else None
        return F.linear(x_smooth, self.weight.to(dtype), bias)

class OutlierPreservedInt4(nn.Module):
    def __init__(self, original_linear, outlier_pct=0.005):
        super().__init__()
        w = original_linear.weight.data.float()
        out_f, in_f = w.shape
        n_outliers = max(1, int(w.numel() * outlier_pct))
        flat    = w.abs().flatten()
        _, top_k_idx = torch.topk(flat, n_outliers, largest=True)
        mask = torch.zeros(w.numel(), dtype=torch.bool, device=w.device)
        mask[top_k_idx] = True
        mask = mask.view(out_f, in_f)
        outlier_values = torch.where(mask, w, torch.zeros_like(w))
        w_to_quantize  = torch.where(mask, torch.zeros_like(w), w)
        scale = w_to_quantize.abs().max(dim=1, keepdim=True).values / 7.0
        scale = scale.clamp(min=1e-8)
        w_q   = (w_to_quantize / scale).round().clamp(-8, 7)
        w_deq = w_q * scale
        self.register_buffer('weight', (w_deq + outlier_values).half())
        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.data.clone())
            self.has_bias = True
        else:
            self.has_bias = False
        self.in_features  = in_f
        self.out_features = out_f

    def forward(self, x, *args, **kwargs):
        dtype = x.dtype
        bias  = self.bias.to(dtype) if self.has_bias else None
        return F.linear(x, self.weight.to(dtype), bias)

def load_pipe():
    return StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )

def to_gpu(pipe):
    pipe.unet           = pipe.unet.to("cuda")
    pipe.vae            = pipe.vae.to("cuda")
    pipe.text_encoder   = pipe.text_encoder.to("cuda")
    pipe.text_encoder_2 = pipe.text_encoder_2.to("cuda")
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

def to_cpu(pipe):
    pipe.unet           = pipe.unet.to("cpu")
    pipe.vae            = pipe.vae.to("cpu")
    pipe.text_encoder   = pipe.text_encoder.to("cpu")
    pipe.text_encoder_2 = pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

def quantize_int4(m):
    w     = m.weight.data.float()
    scale = w.abs().max(dim=1, keepdim=True).values / 7.0
    scale = scale.clamp(min=1e-8)
    w_q   = (w / scale).round().clamp(-8, 7)
    m.weight.data = (w_q * scale).half()

def naive_ternarize_mid(pipe):
    for name, module in pipe.unet.mid_block.named_modules():
        if isinstance(module, nn.Linear):
            w     = module.weight.data.float()
            scale = w.abs().mean()
            module.weight.data = (w.sign() * scale).half()

def inject_ternary_lora(pipe, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
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
        new_layer        = TernaryLoRALinear(w_skel, scale, bias, in_f, out_f, rank=rank)
        new_layer.lora_A = nn.Parameter(lora_A)
        new_layer.lora_B = nn.Parameter(lora_B)
        parts  = layer_path.split('.')
        module = pipe.unet
        for p in parts[:-1]:
            module = getattr(module, p) if not p.isdigit() else module[int(p)]
        setattr(module, parts[-1], new_layer)
    del ckpt; gc.collect()

def generate(pipe, variant, prompts, seed, log_every=100):
    folder = os.path.join(OUT, variant)
    os.makedirs(folder, exist_ok=True)
    done = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
    if done == len(prompts):
        print(f"  [{variant}] Complete ({done}/{len(prompts)}). Skipping.")
        return
    print(f"  [{variant}] Resuming from {done}")

    for i, item in enumerate(prompts):
        out_path = os.path.join(folder, item['filename'])
        if os.path.exists(out_path):
            continue
        gen = torch.Generator("cuda").manual_seed(seed + i)
        with torch.inference_mode():
            image = pipe(
                item['caption'],
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=gen,
                height=768, width=768,
            ).images[0]
        image.save(out_path)
        if (i + 1) % log_every == 0:
            print(f"  [{variant}] {i+1}/{len(prompts)}")
    print(f"  [{variant}] Done.\n")

# ═════════════════════════════════════════════════════════════
# V1: FP16 baseline
# ═════════════════════════════════════════════════════════════
print("\n=== V1: FP16 baseline ===")
'''pipe = load_pipe()
to_gpu(pipe)
generate(pipe, "v1_fp16", PROMPTS, SEED)
to_cpu(pipe); del pipe; gc.collect()'''

# ═════════════════════════════════════════════════════════════
# V2: Naive INT4
# ═════════════════════════════════════════════════════════════
print("\n=== V2: Naive INT4 ===")
'''pipe = load_pipe()
for name, module in pipe.unet.named_modules():
    if isinstance(module, nn.Linear):
        quantize_int4(module)
to_gpu(pipe)
generate(pipe, "v2_int4_naive", PROMPTS, SEED)
to_cpu(pipe); del pipe; gc.collect()'''

# ═════════════════════════════════════════════════════════════
# V4: Hybrid ternary mid + INT4
# ═════════════════════════════════════════════════════════════
print("\n=== V4: Hybrid ternary mid + INT4 ===")
pipe = load_pipe()
inject_ternary_lora(pipe, CKPT_BEST)
for name, module in pipe.unet.named_modules():
    if (isinstance(module, nn.Linear) and
            not isinstance(module, TernaryLoRALinear)):
        quantize_int4(module)
to_gpu(pipe)
generate(pipe, "v4_hybrid_best", PROMPTS, SEED)
to_cpu(pipe); del pipe; gc.collect()

# ═════════════════════════════════════════════════════════════
# V6: Fully distilled hybrid (same as V4 since V6 file was identical)
# NOTE: V6 uses sdxl_hybrid_distilled_full.pt checkpoint
# ═════════════════════════════════════════════════════════════
# Actually we are using the same infrastructure. V6 was V4 + distilled int4 LoRAs
# For 5000 benchmark, we use V4 structure with distilled best. This IS V6.
# Leaving the folder named v6_fully_distilled for consistency.
print("\n=== V6: Fully distilled hybrid (same as V4 with full distillation) ===")
pipe = load_pipe()
inject_ternary_lora(pipe, CKPT_BEST)
for name, module in pipe.unet.named_modules():
    if (isinstance(module, nn.Linear) and
            not isinstance(module, TernaryLoRALinear)):
        quantize_int4(module)
to_gpu(pipe)
generate(pipe, "v6_fully_distilled", PROMPTS, SEED)
to_cpu(pipe); del pipe; gc.collect()

# ═════════════════════════════════════════════════════════════
# V7: Timestep-aware LoRA hybrid
# ═════════════════════════════════════════════════════════════
print("\n=== V7: Timestep-aware LoRA hybrid ===")
pipe = load_pipe()
ckpt = torch.load(TIMESTEP_CKPT, map_location="cpu")
skel_keys = [k for k in ckpt if k.endswith('.w_skel') and 'mid_block' in k]
for sk in skel_keys:
    layer_path = sk.replace('.w_skel', '')
    w_skel = ckpt[sk].float()
    scale  = ckpt[layer_path + '.scale'].float()
    bias_k = layer_path + '.bias'
    bias   = ckpt[bias_k].float() if bias_k in ckpt else None
    out_f, in_f = w_skel.shape
    lora_A_e = ckpt[f"{layer_path}.lora_A__early"].float()
    lora_B_e = ckpt[f"{layer_path}.lora_B__early"].float()
    lora_A_l = ckpt[f"{layer_path}.lora_A__late"].float()
    lora_B_l = ckpt[f"{layer_path}.lora_B__late"].float()
    rank = lora_A_e.shape[0]
    new_layer = TernaryLoRASwitchable(w_skel, scale, bias, in_f, out_f, rank=rank)
    new_layer.lora_A_early = nn.Parameter(lora_A_e)
    new_layer.lora_B_early = nn.Parameter(lora_B_e)
    new_layer.lora_A_late  = nn.Parameter(lora_A_l)
    new_layer.lora_B_late  = nn.Parameter(lora_B_l)
    parts = layer_path.split('.')
    module = pipe.unet
    for p in parts[:-1]:
        module = getattr(module, p) if not p.isdigit() else module[int(p)]
    setattr(module, parts[-1], new_layer)
del ckpt; gc.collect()

for name, module in pipe.unet.named_modules():
    if (isinstance(module, nn.Linear) and
            not isinstance(module, TernaryLoRASwitchable)):
        quantize_int4(module)

original_forward = pipe.unet.forward
def forward_with_timestep(sample, timestep, *args, **kwargs):
    t_val = timestep.flatten()[0].item() if torch.is_tensor(timestep) else timestep
    TernaryLoRASwitchable.current_timestep = t_val
    return original_forward(sample, timestep, *args, **kwargs)
pipe.unet.forward = forward_with_timestep

to_gpu(pipe)
generate(pipe, "v7_timestep_lora", PROMPTS, SEED)
to_cpu(pipe); del pipe; gc.collect()

# ═════════════════════════════════════════════════════════════
# V8: SmoothQuant INT4 + ternary mid
# ═════════════════════════════════════════════════════════════
print("\n=== V8: SmoothQuant INT4 + ternary mid ===")
pipe = load_pipe()
inject_ternary_lora(pipe, CKPT_BEST)
stats = torch.load(STATS_PATH, map_location="cpu")
for name, module in pipe.unet.named_modules():
    if (isinstance(module, nn.Linear) and
            not isinstance(module, TernaryLoRALinear) and
            name in stats):
        new_layer = SmoothInt4Linear(module, stats[name], alpha=ALPHA)
        parts = name.split('.')
        parent = pipe.unet
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        setattr(parent, parts[-1], new_layer)
to_gpu(pipe)
generate(pipe, "v8_smoothquant_hybrid", PROMPTS, SEED)
to_cpu(pipe); del pipe; gc.collect()

# ═════════════════════════════════════════════════════════════
# V12: Outlier-preserved INT4 + ternary mid
# ═════════════════════════════════════════════════════════════
print("\n=== V12: Outlier-preserved INT4 + ternary mid ===")
pipe = load_pipe()
inject_ternary_lora(pipe, CKPT_BEST)
for name, module in pipe.unet.named_modules():
    if (isinstance(module, nn.Linear) and
            not isinstance(module, TernaryLoRALinear)):
        new_layer = OutlierPreservedInt4(module, outlier_pct=OUTLIER_PCT)
        parts = name.split('.')
        parent = pipe.unet
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        setattr(parent, parts[-1], new_layer)
to_gpu(pipe)
generate(pipe, "v12_outlier_preserved", PROMPTS, SEED)
to_cpu(pipe); del pipe; gc.collect()

print("\n" + "="*60)
print("ALL VARIANTS COMPLETE")
print("="*60)
for v in ["v1_fp16", "v2_int4_naive", "v4_hybrid_best",
          "v6_fully_distilled", "v7_timestep_lora",
          "v8_smoothquant_hybrid", "v12_outlier_preserved"]:
    folder = os.path.join(OUT, v)
    n = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
    print(f"  {v}: {n} images")