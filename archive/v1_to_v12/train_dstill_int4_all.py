import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import random
from diffusers import StableDiffusionXLPipeline

BASE           = "/home/project/Documents/Maharshi/gem/"
CKPT_BEST      = BASE + "sdxl_v111_mid_r64_1000.pt"
SAVE_PATH      = BASE + "sdxl_hybrid_distilled_full.pt"
LORA_RANK      = 8
STEPS          = 500
LR             = 2e-4
LOG_EVERY      = 50

BLOCK_CONFIGS = [
    {
        'name':       'down_blocks.1',
        'cache':      BASE + 'cache_down1.pt',
        'is_up':      False,
        'has_res':    False,
    },
    {
        'name':       'down_blocks.2',
        'cache':      BASE + 'cache_down2.pt',
        'is_up':      False,
        'has_res':    False,
    },
    {
        'name':       'up_blocks.0',
        'cache':      BASE + 'cache_up0.pt',
        'is_up':      True,
        'has_res':    True,
    },
    {
        'name':       'up_blocks.1',
        'cache':      BASE + 'cache_up1.pt',
        'is_up':      True,
        'has_res':    True,
    },
]

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
    """INT4 quantized linear layer with trainable LoRA correction."""
    def __init__(self, original_linear, rank=8):
        super().__init__()
        w     = original_linear.weight.data.float()
        scale = w.abs().max(dim=1, keepdim=True).values / 7.0
        scale = scale.clamp(min=1e-8)
        w_q   = (w / scale).round().clamp(-8, 7)
        # Store quantized weight and scale
        self.register_buffer('w_q',   w_q.half())
        self.register_buffer('scale', scale.half())
        if original_linear.bias is not None:
            self.register_buffer('bias_buf',
                                 original_linear.bias.data.clone())
            self.has_bias = True
        else:
            self.has_bias = False
        out_f, in_f = w.shape
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x, *args, **kwargs):
        dtype = x.dtype
        # Dequantize
        w    = (self.w_q * self.scale).to(dtype)
        # Add LoRA correction
        w    = w + (self.lora_B @ self.lora_A).to(dtype)
        bias = self.bias_buf.to(dtype) if self.has_bias else None
        return F.linear(x, w, bias)

def get_block(pipe, block_name):
    parts  = block_name.split('.')
    module = pipe.unet
    for p in parts:
        module = getattr(module, p) if not p.isdigit() else module[int(p)]
    return module

def inject_int4_lora(block, rank=8):
    """Replace all Linear layers with INT4LoRALinear."""
    count = 0
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear) and \
                not isinstance(module, (INT4LoRALinear, TernaryLoRALinear)):
            new_layer = INT4LoRALinear(module, rank=rank)
            parts  = name.split('.')
            parent = block
            for p in parts[:-1]:
                parent = getattr(parent, p) if not p.isdigit() \
                         else parent[int(p)]
            setattr(parent, parts[-1], new_layer)
            count += 1
    return count

def train_block(block, cache, config, rank, steps, lr, log_every):
    # Freeze everything except LoRA
    for name, param in block.named_parameters():
        param.requires_grad = 'lora_A' in name or 'lora_B' in name

    trainable = sum(p.numel() for p in block.parameters()
                    if p.requires_grad)
    print(f"  Trainable: {trainable/1e6:.3f}M params (LoRA only)")

    block = block.to("cuda")
    torch.cuda.empty_cache()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

    lora_params  = [p for p in block.parameters() if p.requires_grad]
    optimizer    = torch.optim.AdamW(lora_params, lr=lr,
                                     weight_decay=1e-4)
    lr_sched     = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr/20
    )
    scaler = torch.cuda.amp.GradScaler()
    block.train()

    best_loss  = float('inf')
    best_state = None

    for step in range(1, steps + 1):
        sample  = random.choice(cache)
        inp     = sample['input'].to("cuda")
        target  = sample['output'].to("cuda")
        temb    = sample['temb'].to("cuda") \
                  if sample['temb'] is not None else None
        enc     = sample['enc'].to("cuda") \
                  if sample['enc'] is not None else None

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(dtype=torch.float16):
            if config['is_up']:
                res = tuple(r.to("cuda") for r in sample['res']) \
                      if sample['res'] is not None else None
                pred = block(
                    hidden_states=inp,
                    temb=temb,
                    res_hidden_states_tuple=res,
                    encoder_hidden_states=enc,
                    upsample_size=None,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                )
            else:
                pred = block(inp, temb,
                             encoder_hidden_states=enc)

            if isinstance(pred, tuple):
                pred = pred[0]
            loss = F.mse_loss(pred.float(), target.float())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        lr_sched.step()

        loss_val = loss.item()
        if not torch.isnan(loss) and loss_val < best_loss:
            best_loss  = loss_val
            best_state = {
                name: param.data.cpu().clone()
                for name, param in block.named_parameters()
                if 'lora_A' in name or 'lora_B' in name
            }

        if step % log_every == 0:
            print(f"  [{config['name']}] step {step:4d}/{steps} "
                  f"| loss={loss_val:.6f} | best={best_loss:.6f} "
                  f"| lr={optimizer.param_groups[0]['lr']:.2e}")

    print(f"  [{config['name']}] Best loss: {best_loss:.6f}")
    block = block.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    return best_state

# ── Start from the existing best checkpoint ───────────────────────────
print("Loading final checkpoint as base...")
final_ckpt = torch.load(CKPT_BEST, map_location="cpu")
print(f"  Base checkpoint keys: {len(final_ckpt)}")

# ── Train each block ──────────────────────────────────────────────────
for config in BLOCK_CONFIGS:
    print(f"\n{'='*55}")
    print(f"Distilling {config['name']} "
          f"(rank={LORA_RANK}, steps={STEPS})")
    print(f"{'='*55}")

    # Load fresh pipeline, get block
    pipe  = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    block = get_block(pipe, config['name'])
    del pipe; gc.collect()

    # Inject INT4+LoRA
    count = inject_int4_lora(block, rank=LORA_RANK)
    print(f"  Replaced {count} layers with INT4LoRALinear.")

    # Load cache
    cache = torch.load(config['cache'], map_location="cpu")
    print(f"  Cache: {len(cache)} samples")

    # Train
    best_state = train_block(block, cache, config,
                              LORA_RANK, STEPS, LR, LOG_EVERY)

    # Merge into final checkpoint
    print(f"  Merging {len(best_state)} LoRA tensors into checkpoint...")
    for k, v in best_state.items():
        full_key = f"{config['name']}.{k}"
        final_ckpt[full_key] = v

    # Also save INT4 quantized weights + scales for this block
    for name, module in block.named_modules():
        if isinstance(module, INT4LoRALinear):
            prefix = f"{config['name']}.{name}"
            final_ckpt[f"{prefix}.w_q"]   = module.w_q.cpu()
            final_ckpt[f"{prefix}.scale"] = module.scale.cpu()
            if module.has_bias:
                final_ckpt[f"{prefix}.bias"] = module.bias_buf.cpu()

    del block, cache, best_state
    gc.collect()
    print(f"  Done. Checkpoint now has {len(final_ckpt)} keys.")

# ── Save final checkpoint ─────────────────────────────────────────────
torch.save(final_ckpt, SAVE_PATH)
print(f"\nFull distilled hybrid saved: {SAVE_PATH}")
print("Done.")