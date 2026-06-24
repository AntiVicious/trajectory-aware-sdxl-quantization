import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
import json
import shutil
from diffusers import StableDiffusionXLPipeline

BASE      = "/home/project/Documents/Maharshi/gem/"
OUT       = BASE + "benchmark_hybrid/"
CKPT_BEST = BASE + "sdxl_v111_mid_r64_1000.pt"
PROMPTS   = json.load(open(BASE + "benchmark_prompts.json"))
SEED      = 42

os.makedirs(OUT, exist_ok=True)

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
    scale = w.abs().max(dim=1, keepdim=True).values / 7.0
    scale = scale.clamp(min=1e-8)
    w_q   = (w / scale).round().clamp(-8, 7)
    module.weight.data = (w_q * scale).half()

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

def generate(pipe, variant, prompts, seed):
    folder = os.path.join(OUT, variant)
    os.makedirs(folder, exist_ok=True)
    done = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
    if done == len(prompts):
        print(f"  [{variant}] Already complete. Skipping.")
        return
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
        if (i + 1) % 20 == 0:
            print(f"  [{variant}] {i+1}/{len(prompts)}")
    print(f"  [{variant}] Done.\n")

def inject_ternary_lora(pipe):
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
    return len(skel_keys)

def apply_int4_non_mid(pipe):
    count = 0
    for name, module in pipe.unet.named_modules():
        if (isinstance(module, nn.Linear) and
                not isinstance(module, TernaryLoRALinear) and
                'mid_block' not in name):
            quantize_int4(module)
            count += 1
    return count

def apply_int4_all(pipe):
    count = 0
    for name, module in pipe.unet.named_modules():
        if isinstance(module, nn.Linear):
            quantize_int4(module)
            count += 1
    return count

def naive_ternarize_mid(pipe):
    count = 0
    for name, module in pipe.unet.mid_block.named_modules():
        if isinstance(module, nn.Linear):
            w     = module.weight.data.float()
            scale = w.abs().mean()
            module.weight.data = (w.sign() * scale).half()
            count += 1
    return count

# ── V1: FP16 baseline ─────────────────────────────────────────────────
print("=== V1: FP16 Baseline ===")
v1_folder = os.path.join(OUT, "v1_fp16")
os.makedirs(v1_folder, exist_ok=True)
for fname in os.listdir(BASE + "coco_standard_fp16/"):
    if fname.endswith('.jpg'):
        shutil.copy2(BASE + "coco_standard_fp16/" + fname,
                     v1_folder + "/" + fname)
print(f"  Copied {len(os.listdir(v1_folder))} images.\n")

# ── V2: Naive INT4 only (per-channel, no ternary) ─────────────────────
print("=== V2: Naive INT4 (per-channel) ===")
pipe = load_pipe()
n = apply_int4_all(pipe)
print(f"  Quantized {n} linear layers to INT4.")
to_gpu(pipe)
generate(pipe, "v2_int4_naive", PROMPTS, SEED)
to_cpu(pipe); del pipe; gc.collect()

# ── V3: Ternary mid + distilled LoRA only (no INT4) ───────────────────
print("=== V3: Ternary mid + Distilled LoRA ===")
pipe = load_pipe()
n = inject_ternary_lora(pipe)
print(f"  Injected {n} ternary layers.")
to_gpu(pipe)
generate(pipe, "v3_ternary_mid_only", PROMPTS, SEED)
to_cpu(pipe); del pipe; gc.collect()

# ── V4: HYBRID — ternary mid + INT4 rest (your best) ─────────────────
print("=== V4: Hybrid (Ternary mid + INT4 rest) ===")
pipe = load_pipe()
n1 = inject_ternary_lora(pipe)
n2 = apply_int4_non_mid(pipe)
print(f"  Ternary: {n1} layers | INT4: {n2} layers.")
to_gpu(pipe)
generate(pipe, "v4_hybrid_best", PROMPTS, SEED)
to_cpu(pipe); del pipe; gc.collect()

# ── V5: Naive ternary mid + INT4 rest (ablation — no distillation) ────
print("=== V5: Naive Ternary mid + INT4 rest (ablation) ===")
pipe = load_pipe()
n1 = naive_ternarize_mid(pipe)
n2 = apply_int4_non_mid(pipe)
print(f"  Naive ternary: {n1} mid layers | INT4: {n2} layers.")
to_gpu(pipe)
generate(pipe, "v5_naive_ternary_int4", PROMPTS, SEED)
to_cpu(pipe); del pipe; gc.collect()

# ── Summary ───────────────────────────────────────────────────────────
print("=" * 50)
print("ALL VARIANTS DONE")
print("=" * 50)
for v in ["v1_fp16", "v2_int4_naive", "v3_ternary_mid_only",
          "v4_hybrid_best", "v5_naive_ternary_int4"]:
    folder = os.path.join(OUT, v)
    n = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
    print(f"  {v}: {n} images")
print(f"\nOutput: {OUT}")