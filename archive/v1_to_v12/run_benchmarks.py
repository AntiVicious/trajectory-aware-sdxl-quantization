import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
import random
from diffusers import StableDiffusionXLPipeline

BASE = "/home/project/Documents/Maharshi/gem/"
OUT  = "/home/project/Documents/Maharshi/gem/benchmark_results/"
os.makedirs(OUT, exist_ok=True)

PROMPTS = [
    "a photo of a dog sitting in a park, golden hour, photorealistic",
    "a busy street market in an Indian city, afternoon light, DSLR photo",
    "portrait of an elderly woman smiling, natural light, sharp focus",
    "a bowl of fresh fruit on a wooden table, studio lighting",
    "snow covered mountains at sunrise, landscape photography",
]
SEED = 42

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

def load_base_pipe():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    return pipe

def move_to_gpu(pipe):
    pipe.unet           = pipe.unet.to("cuda")
    pipe.vae            = pipe.vae.to("cuda")
    pipe.text_encoder   = pipe.text_encoder.to("cuda")
    pipe.text_encoder_2 = pipe.text_encoder_2.to("cuda")
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

def move_to_cpu_and_clear(pipe):
    pipe.unet           = pipe.unet.to("cpu")
    pipe.vae            = pipe.vae.to("cpu")
    pipe.text_encoder   = pipe.text_encoder.to("cpu")
    pipe.text_encoder_2 = pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

def generate_images(pipe, variant_name, prompts, seed):
    folder = os.path.join(OUT, variant_name)
    os.makedirs(folder, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    for i, prompt in enumerate(prompts):
        gen = torch.Generator("cuda").manual_seed(seed)
        with torch.inference_mode():
            image = pipe(
                prompt,
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=gen,
                height=768,
                width=768,
            ).images[0]
        path = os.path.join(folder, f"{i:02d}.png")
        image.save(path)
        print(f"  [{variant_name}] {i+1}/5 saved.")

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  Peak VRAM: {peak_mb:.0f} MB\n")
    return peak_mb

def inject_ternary_lora(pipe, ckpt_path):
    ckpt      = torch.load(ckpt_path, map_location="cpu")
    skel_keys = [k for k in ckpt if k.endswith('.w_skel')]
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

        new_layer        = TernaryLoRALinear(w_skel, scale, bias, in_f, out_f, rank=rank)
        new_layer.lora_A = nn.Parameter(lora_A)
        new_layer.lora_B = nn.Parameter(lora_B)

        parts  = layer_path.split('.')
        module = pipe.unet
        for p in parts[:-1]:
            module = getattr(module, p) if not p.isdigit() else module[int(p)]
        setattr(module, parts[-1], new_layer)

    del ckpt
    gc.collect()
    print(f"  Injected {len(skel_keys)} ternary+LoRA layers.")

results = {}

# ── V1: FP16 Baseline ─────────────────────────────────────────────────
print("=== V1: FP16 Baseline ===")
pipe = load_base_pipe()
move_to_gpu(pipe)
results["v1_fp16_baseline"] = generate_images(pipe, "v1_fp16_baseline", PROMPTS, SEED)
move_to_cpu_and_clear(pipe)
del pipe; gc.collect()

# ── V2: Naive Ternary (no LoRA) ───────────────────────────────────────
print("=== V2: Naive Ternary ===")
pipe = load_base_pipe()
count = 0
for name, module in pipe.unet.mid_block.named_modules():
    if isinstance(module, nn.Linear):
        w     = module.weight.data.float()
        scale = w.abs().mean()
        module.weight.data = (w.sign() * scale).half()
        count += 1
print(f"  Ternarized {count} mid_block layers (no LoRA).")
move_to_gpu(pipe)
results["v2_naive_ternary"] = generate_images(pipe, "v2_naive_ternary", PROMPTS, SEED)
move_to_cpu_and_clear(pipe)
del pipe; gc.collect()

# ── V3: LoRA Only (v72) ───────────────────────────────────────────────
print("=== V3: LoRA Only (v72) ===")

class LoRALinear(nn.Module):
    def __init__(self, original_layer, lora_A, lora_B):
        super().__init__()
        self.weight = nn.Parameter(original_layer.weight.data.clone())
        self.bias   = nn.Parameter(original_layer.bias.data.clone()) if original_layer.bias is not None else None
        self.lora_A = nn.Parameter(lora_A)
        self.lora_B = nn.Parameter(lora_B)

    def forward(self, x, *args, **kwargs):
        dtype = x.dtype
        base  = F.linear(x, self.weight.to(dtype),
                         self.bias.to(dtype) if self.bias is not None else None)
        delta = F.linear(F.linear(x, self.lora_A.to(dtype)), self.lora_B.to(dtype))
        return base + delta

pipe = load_base_pipe()
ckpt = torch.load(BASE + "sdxl_bitlora_hybrid_v72_best.pt", map_location="cpu")
for key in [k for k in ckpt if k.endswith('.lora_A')]:
    layer_path = key.replace('.lora_A', '')
    lora_A     = ckpt[key].half()
    lora_B     = ckpt[layer_path + '.lora_B'].half()
    parts      = layer_path.split('.')
    module     = pipe.unet
    for p in parts[:-1]:
        module = getattr(module, p) if not p.isdigit() else module[int(p)]
    original = getattr(module, parts[-1])
    setattr(module, parts[-1], LoRALinear(original, lora_A, lora_B))
del ckpt; gc.collect()
print(f"  Injected v72 LoRA layers.")
move_to_gpu(pipe)
results["v3_lora_only_v72"] = generate_images(pipe, "v3_lora_only_v72", PROMPTS, SEED)
move_to_cpu_and_clear(pipe)
del pipe; gc.collect()

# ── V4: Ternary + LoRA untrained (v111 skeleton + v72 LoRA) ───────────
print("=== V4: Ternary + Untrained LoRA (v111 skeleton) ===")
pipe = load_base_pipe()

# Ternarize mid_block using v111 w_skel
ckpt_111  = torch.load(BASE + "sdxl_bitlora_hybrid_v111_best.pt", map_location="cpu")
skel_keys = [k for k in ckpt_111 if k.endswith('.w_skel')]
for skel_key in skel_keys:
    layer_path  = skel_key.replace('.w_skel', '')
    w_skel      = ckpt_111[skel_key].float()
    scale       = ckpt_111[layer_path + '.scale'].float()
    bias_k      = layer_path + '.bias'
    bias        = ckpt_111[bias_k].float() if bias_k in ckpt_111 else None
    out_f, in_f = w_skel.shape

    # Use zero LoRA (no correction) to show ternary-only quality
    new_layer = TernaryLoRALinear(w_skel, scale, bias, in_f, out_f, rank=32)
    # lora_B is already zeros — so this is pure ternary

    parts  = layer_path.split('.')
    module = pipe.unet
    for p in parts[:-1]:
        module = getattr(module, p) if not p.isdigit() else module[int(p)]
    setattr(module, parts[-1], new_layer)

del ckpt_111; gc.collect()
print(f"  Injected {len(skel_keys)} pure ternary layers (zero LoRA).")
move_to_gpu(pipe)
results["v4_ternary_no_lora"] = generate_images(pipe, "v4_ternary_no_lora", PROMPTS, SEED)
move_to_cpu_and_clear(pipe)
del pipe; gc.collect()

# ── V5: Trained Ternary + LoRA (best model) ───────────────────────────
print("=== V5: Trained Ternary + LoRA (best model) ===")
pipe = load_base_pipe()
inject_ternary_lora(pipe, BASE + "sdxl_v111_v72_combined.pt")
move_to_gpu(pipe)
results["v5_trained_ternary_lora"] = generate_images(pipe, "v5_trained_ternary_lora", PROMPTS, SEED)
move_to_cpu_and_clear(pipe)
del pipe; gc.collect()

# ── Summary ───────────────────────────────────────────────────────────
print("=" * 50)
print("BENCHMARK COMPLETE")
print("=" * 50)
for name, vram in results.items():
    print(f"  {name}: peak VRAM = {vram:.0f} MB")
print(f"\nImages saved to: {OUT}")
print("Total images: 25 (5 variants x 5 prompts)")