import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

BASE       = "/home/project/Documents/Maharshi/gem/"
CKPT_BEST  = BASE + "sdxl_v111_mid_r64_1000.pt"
CACHE_PATH = BASE + "coco_latents_cache.pt"

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
    del ckpt
    gc.collect()

# ── Load FP16 pipeline ────────────────────────────────────────────────
print("Loading FP16 pipeline...")
pipe_fp = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
scheduler = DDPMScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="scheduler"
)
pipe_fp.unet = pipe_fp.unet.to("cuda")
pipe_fp.unet.eval()

coco = torch.load(CACHE_PATH, map_location="cpu")
timesteps_to_test = [100, 300, 500, 700, 900]
n_samples = 20

# Collect FP16 predictions
print("Collecting FP16 predictions...")
fp_preds = {t: [] for t in timesteps_to_test}
noise_inputs = {t: [] for t in timesteps_to_test}
embeddings   = {t: [] for t in timesteps_to_test}

for t_val in timesteps_to_test:
    for i in range(n_samples):
        sample = coco[i]
        latent = sample['latents'].to("cuda")
        pe     = sample['prompt_embeds'].to("cuda")
        ppe    = sample['pooled_embeds'].to("cuda")

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

        fp_preds[t_val].append(pred.detach().cpu())
        noise_inputs[t_val].append((noisy.cpu(), pe.cpu(), ppe.cpu()))

pipe_fp.unet = pipe_fp.unet.to("cpu")
del pipe_fp
gc.collect()
torch.cuda.empty_cache()

# ── Load quantized pipeline (V6) ──────────────────────────────────────
print("Loading V6 quantized pipeline...")
pipe_q = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
build_v6(pipe_q)
pipe_q.unet = pipe_q.unet.to("cuda")
pipe_q.unet.eval()

# Collect quantized predictions on SAME inputs
print("Collecting quantized predictions on same inputs...")
q_preds = {t: [] for t in timesteps_to_test}

for t_val in timesteps_to_test:
    for i in range(n_samples):
        noisy_cpu, pe_cpu, ppe_cpu = noise_inputs[t_val][i]
        noisy = noisy_cpu.to("cuda")
        pe    = pe_cpu.to("cuda")
        ppe   = ppe_cpu.to("cuda")
        t     = torch.tensor([t_val], device="cuda").long()

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

        q_preds[t_val].append(pred.detach().cpu())

del pipe_q
gc.collect()
torch.cuda.empty_cache()

# ── Analyze error structure ───────────────────────────────────────────
print("\n=== Error Structure Analysis ===\n")

for t_val in timesteps_to_test:
    errors = []
    for i in range(n_samples):
        err = fp_preds[t_val][i].float() - q_preds[t_val][i].float()
        errors.append(err)

    # Stack all errors for this timestep: (n_samples, C, H, W)
    err_stack = torch.stack(errors, dim=0).squeeze(1)  # remove batch dim

    # Key statistics:
    # 1. Mean error across samples — is there a systematic component?
    mean_err = err_stack.mean(dim=0)
    # 2. Std across samples — how variable is the error?
    std_err  = err_stack.std(dim=0)

    mean_magnitude = mean_err.abs().mean().item()
    std_magnitude  = std_err.abs().mean().item()
    total_magnitude = err_stack.abs().mean().item()

    # Systematic ratio: if mean >> std, error is mostly systematic
    # If std >> mean, error is mostly input-dependent (noise)
    sys_ratio = mean_magnitude / (std_magnitude + 1e-8)

    print(f"t={t_val:4d}:")
    print(f"  total error magnitude: {total_magnitude:.4f}")
    print(f"  systematic (mean):     {mean_magnitude:.4f}")
    print(f"  variable (std):        {std_magnitude:.4f}")
    print(f"  systematic/variable:   {sys_ratio:.3f}")
    print()

print("Interpretation:")
print("  If systematic/variable > 0.3 — REPN idea viable (systematic errors exist)")
print("  If systematic/variable < 0.1 — REPN won't help (error is input-dependent)")