import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import random
from diffusers import StableDiffusionXLPipeline

CKPT_V111      = "/home/project/Documents/Maharshi/gem/sdxl_bitlora_hybrid_v111_best.pt"
MIDBLOCK_CACHE = "/home/project/Documents/Maharshi/gem/midblock_cache.pt"
SAVE_PATH      = "/home/project/Documents/Maharshi/gem/sdxl_v111_mid_r64_1000.pt"
LORA_RANK      = 64
STEPS          = 2000
LR             = 2e-4
LOG_EVERY      = 50

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

# ── Load only mid_block ───────────────────────────────────────────────
print("Loading SDXL mid_block only...")
pipe      = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
mid_block = pipe.unet.mid_block
del pipe
gc.collect()
print("Pipeline deleted — keeping only mid_block.")

# ── Inject v111 ternary weights ───────────────────────────────────────
print("Injecting v111 ternary layers...")
ckpt      = torch.load(CKPT_V111, map_location="cpu")
skel_keys = [k for k in ckpt if k.endswith('.w_skel')
             and k.startswith('mid_block')]

for skel_key in skel_keys:
    layer_path  = skel_key.replace('.w_skel', '')
    w_skel      = ckpt[skel_key].float()
    scale       = ckpt[layer_path + '.scale'].float()
    bias_k      = layer_path + '.bias'
    bias        = ckpt[bias_k].float() if bias_k in ckpt else None
    out_f, in_f = w_skel.shape

    new_layer  = TernaryLoRALinear(w_skel, scale, bias,
                                   in_f, out_f, rank=LORA_RANK)
    local_path = layer_path[len('mid_block.'):]
    parts      = local_path.split('.')
    module     = mid_block
    for p in parts[:-1]:
        module = getattr(module, p) if not p.isdigit() else module[int(p)]
    setattr(module, parts[-1], new_layer)

del ckpt
gc.collect()
print(f"Injected {len(skel_keys)} ternary layers.")

# ── Freeze everything except LoRA ─────────────────────────────────────
for name, param in mid_block.named_parameters():
    param.requires_grad = 'lora_A' in name or 'lora_B' in name

trainable = sum(p.numel() for p in mid_block.parameters()
                if p.requires_grad)
total     = sum(p.numel() for p in mid_block.parameters())
print(f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

# ── Move to GPU ───────────────────────────────────────────────────────
mid_block = mid_block.to("cuda")
torch.cuda.empty_cache()
print(f"VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

# ── Load cache ────────────────────────────────────────────────────────
print("Loading mid-block cache...")
cache = torch.load(MIDBLOCK_CACHE, map_location="cpu")
print(f"Cache: {len(cache)} samples | "
      f"input: {cache[0]['input'].shape} | "
      f"output: {cache[0]['output'].shape}")

# ── Optimizer ─────────────────────────────────────────────────────────
lora_params  = [p for p in mid_block.parameters() if p.requires_grad]
optimizer    = torch.optim.AdamW(lora_params, lr=LR, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=STEPS, eta_min=LR/20
)
scaler = torch.cuda.amp.GradScaler()

# ── Training ──────────────────────────────────────────────────────────
print(f"\nDistilling mid_block: {STEPS} steps | "
      f"rank={LORA_RANK} | lr={LR}\n")
mid_block.train()

best_loss  = float('inf')
best_state = None

for step in range(1, STEPS + 1):
    sample  = random.choice(cache)
    inp     = sample['input'].to("cuda")
    target  = sample['output'].to("cuda")
    temb    = sample['temb'].to("cuda") \
              if sample['temb'] is not None else None
    encoder = sample['encoder_hidden_states'].to("cuda")

    optimizer.zero_grad()

    with torch.cuda.amp.autocast(dtype=torch.float16):
        pred = mid_block(inp, temb, encoder_hidden_states=encoder)
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
            for name, param in mid_block.named_parameters()
            if 'lora_A' in name or 'lora_B' in name
        }

    if step % LOG_EVERY == 0:
        print(f"  step {step:5d}/{STEPS} | loss={loss_val:.6f} "
              f"| best={best_loss:.6f} "
              f"| lr={optimizer.param_groups[0]['lr']:.2e}")

# ── Save ──────────────────────────────────────────────────────────────
print(f"\nBest loss: {best_loss:.6f}")
print("Saving checkpoint...")

final_ckpt = torch.load(CKPT_V111, map_location="cpu")
for k, v in best_state.items():
    full_key = f"mid_block.{k}"
    final_ckpt[full_key] = v

torch.save(final_ckpt, SAVE_PATH)
print(f"Saved: {SAVE_PATH}")
print("Done.")