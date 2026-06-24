import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import random
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

COMBINED_CKPT = "/home/project/Documents/Maharshi/gem/sdxl_v111_v72_combined.pt"
CACHE_PATH    = "/home/project/Documents/Maharshi/gem/coco_latents_cache.pt"
SAVE_PATH     = "/home/project/Documents/Maharshi/gem/sdxl_v111_upblock_combined.pt"
LORA_RANK     = 32
STEPS         = 300
LR            = 5e-5  # lower LR — more conservative since adding new block
LOG_EVERY     = 10

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

def ternarize_linear(module):
    """Convert a single nn.Linear to TernaryLoRALinear in-place."""
    w      = module.weight.data.float()
    scale  = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    w_skel = (w / scale).sign()
    bias   = module.bias.data.float() if module.bias is not None else None
    out_f, in_f = w.shape
    new = TernaryLoRALinear(w_skel, scale, bias, in_f, out_f, rank=LORA_RANK)
    return new

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

# ── Step 1: inject trained mid_block from combined checkpoint ─────────
print("Loading trained mid_block from combined checkpoint...")
ckpt      = torch.load(COMBINED_CKPT, map_location="cpu")
skel_keys = [k for k in ckpt if k.endswith('.w_skel')]
print(f"  Found {len(skel_keys)} mid_block ternary layers.")

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

# ── Step 2: ternarize up_blocks.1 fresh ──────────────────────────────
print("Ternarizing up_blocks.1...")
up1         = pipe.unet.up_blocks[1]
up1_layers  = []

for name, module in up1.named_modules():
    if isinstance(module, nn.Linear):
        new_layer = ternarize_linear(module)
        # Navigate to parent and replace
        parts  = name.split('.')
        parent = up1
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        setattr(parent, parts[-1], new_layer)
        up1_layers.append(name)

print(f"  Ternarized {len(up1_layers)} layers in up_blocks.1.")

# ── Freeze: keep mid_block LoRA frozen, only train up_blocks.1 LoRA ──
print("Setting trainable params...")
for name, param in pipe.unet.named_parameters():
    if 'up_blocks.1' in name and ('lora_A' in name or 'lora_B' in name):
        param.requires_grad = True   # train new up_block LoRA
    elif 'mid_block' in name and ('lora_A' in name or 'lora_B' in name):
        param.requires_grad = False  # keep mid_block LoRA frozen
    else:
        param.requires_grad = False

trainable = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
print(f"Trainable: {trainable/1e6:.2f}M params (up_blocks.1 LoRA only)")

# ── Move UNet to GPU only ─────────────────────────────────────────────
print("Moving UNet to CUDA...")
pipe.unet = pipe.unet.to("cuda")
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.text_encoder_2.requires_grad_(False)
torch.cuda.empty_cache()
gc.collect()
print(f"VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

pipe.unet.enable_gradient_checkpointing()

# ── Load cache ────────────────────────────────────────────────────────
print("Loading COCO cache...")
cache = torch.load(CACHE_PATH, map_location="cpu")
print(f"Cache: {len(cache)} samples.")

# ── Optimizer — only up_blocks.1 LoRA ─────────────────────────────────
lora_params  = [p for p in pipe.unet.parameters() if p.requires_grad]
optimizer    = AdamW(lora_params, lr=LR, weight_decay=1e-4)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=LR/10)
scaler       = torch.cuda.amp.GradScaler()

# ── Training ──────────────────────────────────────────────────────────
print(f"\nTraining {STEPS} steps | rank={LORA_RANK} | lr={LR}\n")
pipe.unet.train()

best_loss  = float('inf')
best_state = None

for step in range(1, STEPS + 1):
    sample = random.choice(cache)
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

# ── Save: merge everything into one checkpoint ────────────────────────
print(f"\nBest loss: {best_loss:.6f}")
print("Saving expanded checkpoint...")

# Start from the combined checkpoint (has mid_block ternary + trained LoRA)
final_ckpt = torch.load(COMBINED_CKPT, map_location="cpu")

# Add up_blocks.1 ternary weights
for name, module in pipe.unet.up_blocks[1].named_modules():
    if isinstance(module, TernaryLoRALinear):
        prefix = f"up_blocks.1.{name}"
        final_ckpt[f"{prefix}.w_skel"] = module.w_skel.cpu()
        final_ckpt[f"{prefix}.scale"]  = module.scale.cpu()
        if module.has_bias:
            final_ckpt[f"{prefix}.bias"] = module.bias_buf.cpu()

# Add trained up_blocks.1 LoRA from best_state
for k, v in best_state.items():
    if 'up_blocks.1' in k:
        clean_k = k[len("unet."):] if k.startswith("unet.") else k
        final_ckpt[clean_k] = v
        print(f"  merged: {clean_k}")

torch.save(final_ckpt, SAVE_PATH)
print(f"\nSaved: {SAVE_PATH}")
print("Done.")