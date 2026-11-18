import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import random
from diffusers import StableDiffusionXLPipeline

BASE       = "/home/project/Documents/Maharshi/gem/"
STATS_PATH = BASE + "smooth_activation_stats.pt"
CKPT_BEST  = BASE + "sdxl_v111_mid_r64_1000.pt"
SAVE_PATH  = BASE + "sdxl_smooth_int4_lora.pt"
LORA_RANK  = 8
STEPS      = 500
LR         = 2e-4
LOG_EVERY  = 50
ALPHA      = 0.5

BLOCK_CONFIGS = [
    {'name': 'down_blocks.1', 'cache': BASE + 'cache_down1.pt', 'is_up': False},
    {'name': 'down_blocks.2', 'cache': BASE + 'cache_down2.pt', 'is_up': False},
    {'name': 'up_blocks.0',   'cache': BASE + 'cache_up0.pt',   'is_up': True},
    {'name': 'up_blocks.1',   'cache': BASE + 'cache_up1.pt',   'is_up': True},
]

class SmoothInt4LoRALinear(nn.Module):
    """SmoothQuant-INT4 linear with trainable LoRA correction."""
    def __init__(self, original_linear, act_max, alpha=0.5, rank=8):
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

        out_f, in_f = w.shape
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.in_features  = in_f
        self.out_features = out_f

    def forward(self, x, *args, **kwargs):
        dtype = x.dtype
        x_smooth = x / self.smooth_scale.to(dtype)
        w_base   = self.weight.to(dtype)
        lora     = (self.lora_B @ self.lora_A).to(dtype)
        w        = w_base + lora
        bias     = self.bias.to(dtype) if self.has_bias else None
        return F.linear(x_smooth, w, bias)

def get_block(pipe, block_name):
    parts  = block_name.split('.')
    module = pipe.unet
    for p in parts:
        module = getattr(module, p) if not p.isdigit() else module[int(p)]
    return module

def inject_smooth_int4_lora(block, block_name, stats, alpha, rank):
    count = 0
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            full_name = f"{block_name}.{name}"
            if full_name in stats:
                new_layer = SmoothInt4LoRALinear(
                    module, stats[full_name], alpha=alpha, rank=rank
                )
                parts  = name.split('.')
                parent = block
                for p in parts[:-1]:
                    parent = getattr(parent, p) if not p.isdigit() \
                             else parent[int(p)]
                setattr(parent, parts[-1], new_layer)
                count += 1
    return count

def train_block(block, cache, config, steps, lr, log_every):
    for name, param in block.named_parameters():
        param.requires_grad = 'lora_A' in name or 'lora_B' in name

    trainable = sum(p.numel() for p in block.parameters()
                    if p.requires_grad)
    print(f"  Trainable: {trainable/1e6:.3f}M params")

    block = block.to("cuda")
    torch.cuda.empty_cache()

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
                    hidden_states=inp, temb=temb,
                    res_hidden_states_tuple=res,
                    encoder_hidden_states=enc,
                    upsample_size=None,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                )
            else:
                pred = block(inp, temb, encoder_hidden_states=enc)
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
                  f"| loss={loss_val:.6f} | best={best_loss:.6f}")

    print(f"  [{config['name']}] Best: {best_loss:.6f}")
    block = block.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    return best_state, block

# ── Start from V6 checkpoint ──────────────────────────────────────────
print("Loading V6 checkpoint as base...")
final_ckpt = torch.load(CKPT_BEST, map_location="cpu")
stats      = torch.load(STATS_PATH, map_location="cpu")

for config in BLOCK_CONFIGS:
    print(f"\n{'='*55}")
    print(f"Distilling {config['name']} "
          f"(rank={LORA_RANK}, steps={STEPS})")
    print(f"{'='*55}")

    pipe  = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    block = get_block(pipe, config['name'])
    del pipe; gc.collect()

    count = inject_smooth_int4_lora(
        block, config['name'], stats, ALPHA, LORA_RANK
    )
    print(f"  Replaced {count} layers with SmoothInt4LoRALinear.")

    cache = torch.load(config['cache'], map_location="cpu")
    print(f"  Cache: {len(cache)} samples")

    best_state, trained_block = train_block(
        block, cache, config, STEPS, LR, LOG_EVERY
    )

    # Merge into final_ckpt
    for k, v in best_state.items():
        full_key = f"{config['name']}.{k}"
        final_ckpt[full_key] = v

    # Save SmoothInt4 weights + smooth_scale for this block
    for name, module in trained_block.named_modules():
        if isinstance(module, SmoothInt4LoRALinear):
            prefix = f"{config['name']}.{name}"
            final_ckpt[f"{prefix}.weight"]       = module.weight.cpu()
            final_ckpt[f"{prefix}.smooth_scale"] = module.smooth_scale.cpu()
            if module.has_bias:
                final_ckpt[f"{prefix}.bias"] = module.bias.cpu()

    del block, trained_block, cache, best_state
    gc.collect()

torch.save(final_ckpt, SAVE_PATH)
print(f"\nSaved: {SAVE_PATH}")
print("Done.")