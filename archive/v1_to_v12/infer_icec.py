import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from diffusers import StableDiffusionXLPipeline

BASE       = "/home/project/Documents/Maharshi/gem/"
CKPT_BEST  = BASE + "sdxl_v111_mid_r64_1000.pt"
ICEC_PATH  = BASE + "icec_model.pt"
DAMP       = 0.3  # dampen correction — try 0.3 first

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

class ICEC(nn.Module):
    def __init__(self, n_channels=4, hidden_dim=128):
        super().__init__()
        self.t_embed = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 32),
        )
        self.net = nn.Sequential(
            nn.Linear(32 + 8, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, n_channels * 2),
        )
        self.alpha = nn.Parameter(torch.zeros(1))

def quantize_int4(m):
    w     = m.weight.data.float()
    scale = w.abs().max(dim=1, keepdim=True).values / 7.0
    scale = scale.clamp(min=1e-8)
    w_q   = (w / scale).round().clamp(-8, 7)
    m.weight.data = (w_q * scale).half()

print(f"Testing ICEC with damping factor {DAMP}...")
print("Loading SDXL...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

# Build V6
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

icec = ICEC(n_channels=4, hidden_dim=128).to("cuda")
icec.load_state_dict(torch.load(ICEC_PATH, map_location="cuda"))
icec.eval()

original_forward = pipe.unet.forward

def forward_with_damped_icec(sample, timestep, *args, **kwargs):
    out = original_forward(sample, timestep, *args, **kwargs)
    if isinstance(out, tuple):
        q_pred = out[0]; returns_tuple = True
    else:
        q_pred = out.sample; returns_tuple = False

    with torch.no_grad():
        B = sample.shape[0]
        summary = torch.cat([
            sample.mean(dim=[2,3]),
            sample.std(dim=[2,3])
        ], dim=-1).float()
        t_val = timestep.flatten()[0].item() \
                if torch.is_tensor(timestep) else timestep
        t     = torch.tensor([t_val] * B,
                             device=sample.device, dtype=torch.float32)
        t_norm = (t / 1000.0).unsqueeze(-1)
        t_emb  = icec.t_embed(t_norm)
        x      = torch.cat([t_emb, summary], dim=-1)
        out_icec = icec.net(x)
        C      = q_pred.shape[1]
        pred_scale = out_icec[:, :C].unsqueeze(-1).unsqueeze(-1)
        pred_bias  = out_icec[:, C:].unsqueeze(-1).unsqueeze(-1)
        correction = q_pred * pred_scale.to(q_pred.dtype) + \
                     pred_bias.to(q_pred.dtype)

    # DAMPED correction
    corrected = q_pred + DAMP * correction
    if returns_tuple:
        return (corrected,) + out[1:]
    else:
        out.sample = corrected
        return out

pipe.unet.forward = forward_with_damped_icec
pipe.unet           = pipe.unet.to("cuda")
pipe.vae            = pipe.vae.to("cuda")
pipe.text_encoder   = pipe.text_encoder.to("cuda")
pipe.text_encoder_2 = pipe.text_encoder_2.to("cuda")
pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# Test one image first
generator = torch.Generator("cuda").manual_seed(42)
with torch.inference_mode():
    image = pipe(
        "a photo of a dog sitting in a park, golden hour, photorealistic",
        num_inference_steps=20,
        guidance_scale=7.5,
        generator=generator,
        height=768, width=768,
    ).images[0]
image.save(f"icec_damped_{DAMP}.png")
print(f"Saved: icec_damped_{DAMP}.png")