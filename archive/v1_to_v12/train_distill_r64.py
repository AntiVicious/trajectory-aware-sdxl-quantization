import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import random
from diffusers import StableDiffusionXLPipeline

CKPT_V111      = "/home/project/Documents/Maharshi/gem/sdxl_bitlora_hybrid_v111_best.pt"
MIDBLOCK_CACHE = "/home/project/Documents/Maharshi/gem/midblock_cache.pt"
UP1_CACHE      = "/home/project/Documents/Maharshi/gem/upblock1_cache_v2.pt"
SAVE_PATH      = "/home/project/Documents/Maharshi/gem/sdxl_v111_distilled_r64.pt"
LORA_RANK      = 64
STEPS          = 1000
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

def ternarize_linear(module, rank=64):
    w       = module.weight.data.float()
    scale   = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    w_skel  = w.sign()
    bias    = module.bias.data.float() if module.bias is not None else None
    out_f, in_f = w.shape
    return TernaryLoRALinear(w_skel, scale, bias, in_f, out_f, rank=rank)

def train_block(block, cache, steps, lr, log_every, block_name,
                is_mid=False):
    for name, param in block.named_parameters():
        param.requires_grad = 'lora_A' in name or 'lora_B' in name

    trainable = sum(p.numel() for p in block.parameters()
                    if p.requires_grad)
    print(f"  Trainable: {trainable/1e6:.2f}M params")

    block = block.to("cuda")
    torch.cuda.empty_cache()
    print(f"  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

    lora_params  = [p for p in block.parameters() if p.requires_grad]
    optimizer    = torch.optim.AdamW(lora_params, lr=lr, weight_decay=1e-4)
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
        encoder = sample['encoder_hidden_states'].to("cuda")

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(dtype=torch.float16):
            if is_mid:
                pred = block(inp, temb,
                             encoder_hidden_states=encoder)
            else:
                res_tuple = tuple(
                    r.to("cuda")
                    for r in sample['res_hidden_states_tuple']
                ) if sample['res_hidden_states_tuple'] is not None else None
                pred = block(
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
            print(f"  [{block_name}] step {step:5d}/{steps} "
                  f"| loss={loss_val:.6f} | best={best_loss:.6f} "
                  f"| lr={optimizer.param_groups[0]['lr']:.2e}")

    print(f"  [{block_name}] Best loss: {best_loss:.6f}")
    block = block.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    return best_state

# ═══════════════════════════════════════════════════════
# PART 1: Mid-block distillation at rank 64
# ═══════════════════════════════════════════════════════
print("\n" + "="*55)
print("PART 1: Mid-block distillation (rank=64, steps=1000)")
print("="*55)

pipe     = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
mid_block = pipe.unet.mid_block
del pipe
gc.collect()

# Inject v111 ternary weights into mid_block
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

    new_layer = TernaryLoRALinear(w_skel, scale, bias,
                                  in_f, out_f, rank=LORA_RANK)

    local_path = layer_path[len('mid_block.'):]
    parts      = local_path.split('.')
    module     = mid_block
    for p in parts[:-1]:
        module = getattr(module, p) if not p.isdigit() else module[int(p)]
    setattr(module, parts[-1], new_layer)

del ckpt
gc.collect()
print(f"Injected {len(skel_keys)} ternary layers into mid_block.")

mid_cache      = torch.load(MIDBLOCK_CACHE, map_location="cpu")
mid_best_state = train_block(
    mid_block, mid_cache, STEPS, LR, LOG_EVERY,
    "mid_block", is_mid=True
)
del mid_block, mid_cache
gc.collect()

# ═══════════════════════════════════════════════════════
# PART 2: up_blocks[1] distillation at rank 64
# ═══════════════════════════════════════════════════════
print("\n" + "="*55)
print("PART 2: up_blocks[1] distillation (rank=64, steps=1000)")
print("="*55)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
up1 = pipe.unet.up_blocks[1]
del pipe
gc.collect()

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
print(f"Ternarized {count} layers in up_blocks[1].")

up1_cache      = torch.load(UP1_CACHE, map_location="cpu")
up1_best_state = train_block(
    up1, up1_cache, STEPS, LR, LOG_EVERY,
    "up_blocks.1", is_mid=False
)
del up1, up1_cache
gc.collect()

# ═══════════════════════════════════════════════════════
# SAVE: merge everything into one checkpoint
# ═══════════════════════════════════════════════════════
print("\n" + "="*55)
print("Saving final checkpoint...")
print("="*55)

final_ckpt = torch.load(CKPT_V111, map_location="cpu")

# Mid-block: update LoRA weights (w_skel already in v111)
for k, v in mid_best_state.items():
    full_key = f"mid_block.{k}"
    final_ckpt[full_key] = v
    print(f"  mid: {full_key}")

# up_blocks.1: add ternary weights + trained LoRA
pipe_tmp = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
up1_tmp = pipe_tmp.unet.up_blocks[1]
del pipe_tmp
gc.collect()

for name, module in up1_tmp.named_modules():
    if isinstance(module, nn.Linear):
        new_layer = ternarize_linear(module, rank=LORA_RANK)
        parts  = name.split('.')
        parent = up1_tmp
        for p in parts[:-1]:
            parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
        setattr(parent, parts[-1], new_layer)

for name, module in up1_tmp.named_modules():
    if isinstance(module, TernaryLoRALinear):
        prefix = f"up_blocks.1.{name}"
        final_ckpt[f"{prefix}.w_skel"] = module.w_skel.cpu()
        final_ckpt[f"{prefix}.scale"]  = module.scale.cpu()
        if module.has_bias:
            final_ckpt[f"{prefix}.bias"] = module.bias_buf.cpu()

for k, v in up1_best_state.items():
    full_key = f"up_blocks.1.{k}"
    final_ckpt[full_key] = v

del up1_tmp
gc.collect()

torch.save(final_ckpt, SAVE_PATH)
print(f"\nSaved: {SAVE_PATH}")
print("Done.")