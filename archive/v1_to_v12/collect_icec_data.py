import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import random
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

BASE       = "/home/project/Documents/Maharshi/gem/"
CKPT_BEST  = BASE + "sdxl_v111_mid_r64_1000.pt"
CACHE_PATH = BASE + "coco_latents_cache.pt"
SAVE_PATH  = BASE + "icec_training_data.pt"
N_SAMPLES  = 500  # (sample_idx, timestep) pairs

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
        self.in_features  = in_f
        self.out_features = out_f

    def forward(self, x, *args, **kwargs):
        dtype = x.dtype
        w    = (self.w_skel * self.scale).to(dtype)
        w    = w + (self.lora_B @ self.lora_A).to(dtype)
        bias = self.bias_buf.to(dtype) if self.has_bias else None
        return F.linear(x, w, bias)

def quantize_int4(m):
    w     = m.weight.data.float()
    scale = w.abs().max(dim=1, keepdim=True).values / 7.0
    scale = scale.clamp(min=1e-8)
    w_q   = (w / scale).round().clamp(-8, 7)
    m.weight.data = (w_q * scale).half()

def build_v6(pipe):
    ckpt = torch.load(CKPT_BEST, map_location="cpu")
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
    for name, module in pipe.unet.named_modules():
        if (isinstance(module, nn.Linear) and
                not isinstance(module, TernaryLoRALinear)):
            quantize_int4(module)
    del ckpt; gc.collect()

scheduler = DDPMScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="scheduler"
)

coco = torch.load(CACHE_PATH, map_location="cpu")

# ── Phase 1: collect FP16 predictions + inputs ────────────────────────
print("Loading FP16 pipeline...")
pipe_fp = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe_fp.unet = pipe_fp.unet.to("cuda")
pipe_fp.unet.eval()

print(f"Collecting {N_SAMPLES} FP16 predictions...")
fp_data = []
for i in range(N_SAMPLES):
    sample = coco[i % len(coco)]
    latent = sample['latents'].to("cuda")
    pe     = sample['prompt_embeds'].to("cuda")
    ppe    = sample['pooled_embeds'].to("cuda")

    t_val = random.randint(0, 999)
    noise = torch.randn_like(latent)
    t     = torch.tensor([t_val], device="cuda").long()
    noisy = scheduler.add_noise(latent, noise, t)

    add_kwargs = {
        "text_embeds": ppe,
        "time_ids": torch.tensor(
            [[1024,1024,0,0,1024,1024]],
            device="cuda", dtype=torch.float16
        )
    }

    with torch.inference_mode():
        pred = pipe_fp.unet(noisy, t,
                            encoder_hidden_states=pe,
                            added_cond_kwargs=add_kwargs).sample

    fp_data.append({
        'noisy':   noisy.detach().cpu(),
        'pe':      pe.detach().cpu(),
        'ppe':     ppe.detach().cpu(),
        't':       t_val,
        'fp_pred': pred.detach().cpu(),
    })

    if (i + 1) % 50 == 0:
        print(f"  FP16 {i+1}/{N_SAMPLES}")

del pipe_fp
gc.collect()
torch.cuda.empty_cache()

# ── Phase 2: collect quantized predictions on same inputs ─────────────
print("\nLoading V6 quantized pipeline...")
pipe_q = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
build_v6(pipe_q)
pipe_q.unet = pipe_q.unet.to("cuda")
pipe_q.unet.eval()

print(f"Collecting {N_SAMPLES} quantized predictions...")
for i, d in enumerate(fp_data):
    noisy = d['noisy'].to("cuda")
    pe    = d['pe'].to("cuda")
    ppe   = d['ppe'].to("cuda")
    t     = torch.tensor([d['t']], device="cuda").long()

    add_kwargs = {
        "text_embeds": ppe,
        "time_ids": torch.tensor(
            [[1024,1024,0,0,1024,1024]],
            device="cuda", dtype=torch.float16
        )
    }

    with torch.inference_mode():
        pred = pipe_q.unet(noisy, t,
                           encoder_hidden_states=pe,
                           added_cond_kwargs=add_kwargs).sample

    d['q_pred'] = pred.detach().cpu()

    if (i + 1) % 50 == 0:
        print(f"  Quantized {i+1}/{N_SAMPLES}")

del pipe_q
gc.collect()
torch.cuda.empty_cache()

# ── Compute summary features and residuals ────────────────────────────
print("\nComputing features and residuals...")
training_data = []
for d in fp_data:
    noisy   = d['noisy'].squeeze(0).float()    # (4, 128, 128)
    fp_pred = d['fp_pred'].squeeze(0).float()  # (4, 128, 128)
    q_pred  = d['q_pred'].squeeze(0).float()   # (4, 128, 128)

    # Summary: per-channel mean + std of noisy input (4+4=8 features)
    ch_mean = noisy.mean(dim=[1,2])
    ch_std  = noisy.std(dim=[1,2])
    summary = torch.cat([ch_mean, ch_std])  # (8,)

    residual = fp_pred - q_pred  # what ICEC should predict

    training_data.append({
        't':        d['t'],
        'summary':  summary,
        'residual': residual,
    })

# Print stats
residuals = torch.stack([d['residual'] for d in training_data])
print(f"\nCollected {len(training_data)} training samples")
print(f"Residual stats:")
print(f"  shape: {residuals.shape}")
print(f"  mean magnitude: {residuals.abs().mean():.4f}")
print(f"  max magnitude:  {residuals.abs().max():.4f}")

torch.save(training_data, SAVE_PATH)
print(f"\nSaved: {SAVE_PATH}")