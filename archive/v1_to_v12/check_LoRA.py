import torch
import os

# Check the directory where your checkpoint lives
CHECKPOINT_PATH = "sdxl_bitlora_hybrid_v111_best.pt"
checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)

print(f"Scanning directory: {checkpoint_dir}")
print()

# List all files in that folder and nearby folders
for root, dirs, files in os.walk(os.path.dirname(checkpoint_dir)):
    for f in files:
        fpath = os.path.join(root, f)
        size_mb = os.path.getsize(fpath) / 1024**2
        if size_mb > 1:  # ignore tiny files
            print(f"  {size_mb:7.1f} MB  {fpath}")

print()
print("=== Also checking if lora_A is nonzero in any other checkpoint ===")

# If you find other .pt or .safetensors files, check them too
candidates = []
for root, dirs, files in os.walk(os.path.dirname(checkpoint_dir)):
    for f in files:
        if f.endswith('.pt') or f.endswith('.pth') or f.endswith('.safetensors') or f.endswith('.bin'):
            candidates.append(os.path.join(root, f))

for cpath in candidates:
    try:
        c = torch.load(cpath, map_location="cpu")
        if isinstance(c, dict):
            lora_keys = [k for k in c.keys() if 'lora_A' in k]
            if lora_keys:
                sample = c[lora_keys[0]]
                nonzero = (sample != 0).sum().item()
                print(f"\n  {cpath}")
                print(f"    lora_A keys: {len(lora_keys)}, nonzero in first: {nonzero}/{sample.numel()}")
    except:
        pass