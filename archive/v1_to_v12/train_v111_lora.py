import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
import random
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

CKPT_PATH  = "/home/project/Documents/Maharshi/gem/sdxl_bitlora_hybrid_v111_best.pt"
CACHE_PATH = "/home/project/Documents/Maharshi/gem/coco_latents_cache.pt"
SAVE_PATH  = "/home/project/Documents/Maharshi/gem/sdxl_v111_v72_combined.pt"
LORA_RANK  = 32
STEPS      = 300
LR         = 1e-4
LOG_EVERY  = 10

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

# ── Load pipeline on CPU ──────────────────────────────────────────────
print("Loading SDXL on CPU...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
scheduler = DDPMScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="scheduler"
)

# ── Inject ternary layers from v111 ──────────────────────────────────
print("Injecting v111 ternary layers...")
ckpt      = torch.load(CKPT_PATH, map_location="cpu")
skel_keys = [k for k in ckpt if k.endswith('.w_skel')]

for skel_key in skel_keys:
    layer_path = skel_key.replace('.w_skel', '')
    w_skel     = ckpt[skel_key].float()
    scale      = ckpt[layer_path + '.scale'].float()
    bias_k     = layer_path + '.bias'
    bias       = ckpt[bias_k].float() if bias_k in ckpt else None
    out_f, in_f = w_skel.shape

    new_layer = TernaryLoRALinear(w_skel, scale, bias, in_f, out_f, rank=LORA_RANK)

    parts  = layer_path.split('.')
    module = pipe.unet
    for p in parts[:-1]:
        module = getattr(module, p) if not p.isdigit() else module[int(p)]
    setattr(module, parts[-1], new_layer)

del ckpt
gc.collect()
print(f"Injected {len(skel_keys)} ternary layers.")

# ── Freeze everything except LoRA ─────────────────────────────────────
for name, param in pipe.unet.named_parameters():
    param.requires_grad = 'lora_A' in name or 'lora_B' in name

trainable = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
print(f"Trainable: {trainable/1e6:.2f}M params (LoRA only)")

# ── Move only UNet to GPU — keep others on CPU ────────────────────────
print("Moving UNet to CUDA (VAE + text encoders stay on CPU)...")
pipe.unet = pipe.unet.to("cuda")
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.text_encoder_2.requires_grad_(False)
torch.cuda.empty_cache()
gc.collect()
print(f"VRAM after UNet load: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

# ── Enable gradient checkpointing ────────────────────────────────────
pipe.unet.enable_gradient_checkpointing()
print("Gradient checkpointing enabled.")

# ── Load cache ────────────────────────────────────────────────────────
print("Loading COCO cache...")
cache = torch.load(CACHE_PATH, map_location="cpu")
print(f"Cache: {len(cache)} samples ready.")

# ── Optimizer ─────────────────────────────────────────────────────────
lora_params  = [p for p in pipe.unet.parameters() if p.requires_grad]
optimizer    = AdamW(lora_params, lr=LR, weight_decay=1e-4)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=LR/10)
scaler       = torch.cuda.amp.GradScaler()

# ── Training loop ─────────────────────────────────────────────────────
print(f"\nTraining {STEPS} steps | rank={LORA_RANK} | lr={LR}\n")
pipe.unet.train()

best_loss  = float('inf')
best_state = None

for step in range(1, STEPS + 1):
    sample = random.choice(cache)

    # Crop latent to 64x64 to reduce activation memory during training
    latent = sample['latents'][:, :, :64, :64].to("cuda")
    pe     = sample['prompt_embeds'].to("cuda")
    ppe    = sample['pooled_embeds'].to("cuda")

    noise     = torch.randn_like(latent)
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (1,), device="cuda"
    ).long()
    noisy = scheduler.add_noise(latent, noise, timesteps)

    add_kwargs = {
        "text_embeds": ppe,
        "time_ids": torch.tensor(
            [[512, 512, 0, 0, 512, 512]],
            device="cuda", dtype=torch.float16
        )
    }

    optimizer.zero_grad()

    with torch.cuda.amp.autocast(dtype=torch.float16):
        noise_pred = pipe.unet(
            noisy, timesteps,
            encoder_hidden_states=pe,
            added_cond_kwargs=add_kwargs
        ).sample
        loss = F.mse_loss(noise_pred.float(), noise.float())

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    lr_scheduler.step()

    loss_val = loss.item()

    if not torch.isnan(loss) and loss_val < best_loss:
        best_loss  = loss_val
        best_state = {
            name: param.data.cpu().clone()
            for name, param in pipe.unet.named_parameters()
            if 'lora_A' in name or 'lora_B' in name
        }

    if step % LOG_EVERY == 0:
        print(f"  step {step:4d}/{STEPS} | loss={loss_val:.6f} "
              f"| best={best_loss:.6f} "
              f"| lr={optimizer.param_groups[0]['lr']:.2e}")

# ── Save combined checkpoint ──────────────────────────────────────────
print(f"\nBest loss: {best_loss:.6f}")
print("Merging trained LoRA into v111 checkpoint...")

ckpt_final = torch.load(CKPT_PATH, map_location="cpu")
for k, v in best_state.items():
    clean_k = k[len("unet."):] if k.startswith("unet.") else k
    ckpt_final[clean_k] = v
    print(f"  merged: {clean_k}")

torch.save(ckpt_final, SAVE_PATH)
print(f"\nSaved to: {SAVE_PATH}")
print("Done.")