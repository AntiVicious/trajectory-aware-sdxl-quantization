import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
import json
from diffusers import StableDiffusionXLPipeline
from PIL import Image

BASE      = "/home/project/Documents/Maharshi/gem/"
OUT       = BASE + "benchmark_final/"
CKPT_BEST = BASE + "sdxl_v111_mid_r64_1000.pt"  # your 2000-step model
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

def load_base_pipe():
    return StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )

def move_to_gpu(pipe):
    pipe.unet           = pipe.unet.to("cuda")
    pipe.vae            = pipe.vae.to("cuda")
    pipe.text_encoder   = pipe.text_encoder.to("cuda")
    pipe.text_encoder_2 = pipe.text_encoder_2.to("cuda")
    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

def move_to_cpu(pipe):
    pipe.unet           = pipe.unet.to("cpu")
    pipe.vae            = pipe.vae.to("cpu")
    pipe.text_encoder   = pipe.text_encoder.to("cpu")
    pipe.text_encoder_2 = pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

def generate_images(pipe, variant_name, prompts, seed):
    folder = os.path.join(OUT, variant_name)
    os.makedirs(folder, exist_ok=True)

    # Skip if already fully generated
    existing = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
    if existing == len(prompts):
        print(f"  [{variant_name}] Already complete ({existing} images). Skipping.")
        return

    torch.cuda.reset_peak_memory_stats()

    for i, item in enumerate(prompts):
        out_path = os.path.join(folder, item['filename'])
        if os.path.exists(out_path):
            continue  # resume support

        gen = torch.Generator("cuda").manual_seed(seed + i)
        with torch.inference_mode():
            image = pipe(
                item['caption'],
                num_inference_steps=20,
                guidance_scale=7.5,
                generator=gen,
                height=768,
                width=768,
            ).images[0]
        image.save(out_path)

        if (i + 1) % 10 == 0:
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  [{variant_name}] {i+1}/{len(prompts)} | "
                  f"VRAM: {peak:.0f} MB")

    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  [{variant_name}] Done. Peak VRAM: {peak:.0f} MB\n")

def inject_ternary_lora(pipe, ckpt_path):
    ckpt      = torch.load(ckpt_path, map_location="cpu")
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
    print(f"  Injected {len(skel_keys)} ternary+LoRA layers.")

results = {}

# ── V1: FP16 already done — just copy to benchmark folder ─────────────
print("=== V1: FP16 Baseline (already generated) ===")
import shutil
v1_folder = os.path.join(OUT, "v1_fp16_baseline")
os.makedirs(v1_folder, exist_ok=True)
src = BASE + "coco_standard_fp16/"
for fname in os.listdir(src):
    if fname.endswith('.jpg'):
        shutil.copy2(os.path.join(src, fname),
                     os.path.join(v1_folder, fname))
print(f"  Copied {len(os.listdir(v1_folder))} FP16 images.\n")

# ── V2: Naive ternary (mid_block, no LoRA) ────────────────────────────
print("=== V2: Naive Ternary (mid_block, no LoRA) ===")
pipe = load_base_pipe()
count = 0
for name, module in pipe.unet.mid_block.named_modules():
    if isinstance(module, nn.Linear):
        w     = module.weight.data.float()
        scale = w.abs().mean()
        module.weight.data = (w.sign() * scale).half()
        count += 1
print(f"  Ternarized {count} mid_block layers.")
move_to_gpu(pipe)
generate_images(pipe, "v2_naive_ternary", PROMPTS, SEED)
move_to_cpu(pipe)
del pipe; gc.collect()

# ── V3: Distilled ternary — your best model ───────────────────────────
print("=== V3: Distilled Ternary + LoRA (best model) ===")
pipe = load_base_pipe()
inject_ternary_lora(pipe, CKPT_BEST)
move_to_gpu(pipe)
generate_images(pipe, "v3_distilled_best", PROMPTS, SEED)
move_to_cpu(pipe)
del pipe; gc.collect()

# ── V4: Ternary skeleton only (zero LoRA — ablation) ──────────────────
print("=== V4: Ternary Skeleton Only (zero LoRA) ===")
pipe  = load_base_pipe()
ckpt  = torch.load(CKPT_BEST, map_location="cpu")
skel_keys = [k for k in ckpt if k.endswith('.w_skel')]
for sk in skel_keys:
    layer_path  = sk.replace('.w_skel', '')
    w_skel      = ckpt[sk].float()
    scale       = ckpt[layer_path + '.scale'].float()
    bias_k      = layer_path + '.bias'
    bias        = ckpt[bias_k].float() if bias_k in ckpt else None
    out_f, in_f = w_skel.shape
    # Zero LoRA — pure ternary skeleton
    new_layer        = TernaryLoRALinear(w_skel, scale, bias,
                                         in_f, out_f, rank=64)
    # lora_B is already zeros — no correction
    parts  = layer_path.split('.')
    module = pipe.unet
    for p in parts[:-1]:
        module = getattr(module, p) if not p.isdigit() else module[int(p)]
    setattr(module, parts[-1], new_layer)
del ckpt; gc.collect()
print(f"  Injected {len(skel_keys)} pure ternary layers (no LoRA).")
move_to_gpu(pipe)
generate_images(pipe, "v4_ternary_skeleton", PROMPTS, SEED)
move_to_cpu(pipe)
del pipe; gc.collect()

# ── V5: Naive ternary (mid + up_blocks.1) ────────────────────────────
print("=== V5: Naive Ternary (mid + up_blocks.1) ===")
pipe = load_base_pipe()
count = 0
for block_name in ['mid_block', 'up_blocks']:
    block = pipe.unet.mid_block if block_name == 'mid_block' \
            else pipe.unet.up_blocks[1]
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            w     = module.weight.data.float()
            scale = w.abs().mean()
            module.weight.data = (w.sign() * scale).half()
            count += 1
print(f"  Ternarized {count} layers (mid + up1).")
move_to_gpu(pipe)
generate_images(pipe, "v5_naive_mid_up1", PROMPTS, SEED)
move_to_cpu(pipe)
del pipe; gc.collect()

print("=" * 55)
print("ALL VARIANTS GENERATED")
print("=" * 55)
for v in ["v1_fp16_baseline", "v2_naive_ternary",
          "v3_distilled_best", "v4_ternary_skeleton",
          "v5_naive_mid_up1"]:
    folder = os.path.join(OUT, v)
    n = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
    print(f"  {v}: {n} images")

print(f"\nOutput: {OUT}")
print("Run scoring script next.")