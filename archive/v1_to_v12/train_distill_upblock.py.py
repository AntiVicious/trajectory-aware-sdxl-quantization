import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import random
from diffusers import StableDiffusionXLPipeline

DISTILLED_CKPT = "/home/project/Documents/Maharshi/gem/sdxl_v111_distilled.pt"
UP1_CACHE      = "/home/project/Documents/Maharshi/gem/upblock1_cache_v2.pt"
SAVE_PATH      = "/home/project/Documents/Maharshi/gem/sdxl_v111_distilled_up1.pt"
LORA_RANK      = 32
STEPS          = 500
LR             = 2e-4
LOG_EVERY      = 10

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

def ternarize_linear(module, rank=32):
    w      = module.weight.data.float()
    scale  = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    w_skel = w.sign()
    bias   = module.bias.data.float() if module.bias is not None else None
    out_f, in_f = w.shape
    return TernaryLoRALinear(w_skel, scale, bias, in_f, out_f, rank=rank)

# ── Load only up_blocks[1] ────────────────────────────────────────────
print("Loading SDXL up_blocks[1] only...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
up1 = pipe.unet.up_blocks[1]
del pipe
gc.collect()
print("Pipeline deleted — keeping only up_blocks[1].")

# ── Ternarize all linear layers ───────────────────────────────────────
print("Ternarizing up_blocks[1]...")
count = 0
for name, module in up1.named_modules():
    if isinstance(module, nn.Linear):
        new_layer = ternarize_linear(module, rank=LORA_RANK)
        parts  = name.split('.')
        parent = up1
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        setattr(parent, parts[-1], new_layer)
        count += 1
print(f"Ternarized {count} layers.")

# ── Freeze everything except LoRA ─────────────────────────────────────
for name, param in up1.named_parameters():
    param.requires_grad = 'lora_A' in name or 'lora_B' in name

trainable = sum(p.numel() for p in up1.parameters() if p.requires_grad)
print(f"Trainable: {trainable/1e6:.3f}M params")

# ── Move to GPU ───────────────────────────────────────────────────────
up1 = up1.to("cuda")
torch.cuda.empty_cache()
print(f"VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

# ── Load cache ────────────────────────────────────────────────────────
print("Loading up_blocks[1] cache...")
cache = torch.load(UP1_CACHE, map_location="cpu")
print(f"Cache: {len(cache)} samples")

# ── Optimizer ─────────────────────────────────────────────────────────
lora_params  = [p for p in up1.parameters() if p.requires_grad]
optimizer    = torch.optim.AdamW(lora_params, lr=LR, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=STEPS, eta_min=LR/20
)
scaler = torch.cuda.amp.GradScaler()

# ── Training ──────────────────────────────────────────────────────────
print(f"\nDistilling up_blocks[1]: {STEPS} steps | rank={LORA_RANK} | lr={LR}\n")
up1.train()

best_loss  = float('inf')
best_state = None

for step in range(1, STEPS + 1):
    sample  = random.choice(cache)
    inp     = sample['input'].to("cuda")
    target  = sample['output'].to("cuda")
    temb    = sample['temb'].to("cuda") if sample['temb'] is not None else None
    encoder = sample['encoder_hidden_states'].to("cuda")
    res_tuple = tuple(
        r.to("cuda") for r in sample['res_hidden_states_tuple']
    ) if sample['res_hidden_states_tuple'] is not None else None

    optimizer.zero_grad()

    with torch.cuda.amp.autocast(dtype=torch.float16):
        pred = up1(
            hidden_states=inp,
            temb=temb,
            res_hidden_states_tuple=res_tuple,
            encoder_hidden_states=encoder,
            upsample_size=None,
            attention_mask=None,
            cross_attention_kwargs=None,
        )
        if isinstance(pred, tuple):
            pred = pred[0]
        loss = F.mse_loss(pred.float(), target.float())

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
            for name, param in up1.named_parameters()
            if 'lora_A' in name or 'lora_B' in name
        }

    if step % LOG_EVERY == 0:
        print(f"  step {step:4d}/{STEPS} | loss={loss_val:.6f} "
              f"| best={best_loss:.6f} "
              f"| lr={optimizer.param_groups[0]['lr']:.2e}")

# ── Save: merge into distilled checkpoint ─────────────────────────────
print(f"\nBest loss: {best_loss:.6f}")
print("Merging up_blocks[1] into distilled checkpoint...")

final_ckpt = torch.load(DISTILLED_CKPT, map_location="cpu")

# Save ternary weights for up_blocks[1]
for name, module in up1.named_modules():
    if isinstance(module, TernaryLoRALinear):
        prefix = f"up_blocks.1.{name}"
        final_ckpt[f"{prefix}.w_skel"] = module.w_skel.cpu()
        final_ckpt[f"{prefix}.scale"]  = module.scale.cpu()
        if module.has_bias:
            final_ckpt[f"{prefix}.bias"] = module.bias_buf.cpu()

# Save trained LoRA weights
for k, v in best_state.items():
    full_key = f"up_blocks.1.{k}"
    final_ckpt[full_key] = v

torch.save(final_ckpt, SAVE_PATH)
print(f"Saved: {SAVE_PATH}")
print("Done.")