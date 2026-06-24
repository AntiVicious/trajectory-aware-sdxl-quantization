import torch

CHECKPOINT_PATH = "sdxl_bitlora_hybrid_v111_best.pt"
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

print("=== Checking for NaNs and zeros in ternary layers ===\n")

skel_keys = [k for k in ckpt.keys() if k.endswith('.w_skel')]

total_nan = 0
total_zero = 0
total_elements = 0

for key in skel_keys[:5]:  # check first 5 layers
    t = ckpt[key]
    n_nan  = torch.isnan(t).sum().item()
    n_zero = (t == 0).sum().item()
    n_total = t.numel()
    print(f"{key}")
    print(f"  shape: {t.shape}, dtype: {t.dtype}")
    print(f"  NaNs: {n_nan}, Zeros: {n_zero}/{n_total} ({100*n_zero/n_total:.1f}%)")
    print(f"  min: {t.min():.4f}, max: {t.max():.4f}, mean: {t.abs().mean():.4f}")
    total_nan += n_nan
    total_zero += n_zero
    total_elements += n_total

print(f"\nAlso checking lora_A and lora_B of first layer:")
k = skel_keys[0].replace('.w_skel', '')
for suffix in ['.lora_A', '.lora_B', '.scale']:
    t = ckpt[k + suffix]
    print(f"  {suffix}: shape={t.shape}, NaNs={torch.isnan(t).sum().item()}, "
          f"min={t.min():.4f}, max={t.max():.4f}")

print(f"\nTotal NaNs across first 5 w_skel tensors: {total_nan}")
print(f"Total zeros across first 5 w_skel tensors: {total_zero}/{total_elements}")