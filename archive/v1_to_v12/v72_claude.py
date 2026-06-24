import torch

path = "/home/project/Documents/Maharshi/gem/sdxl_bitlora_hybrid_v72_best.pt"
ckpt = torch.load(path, map_location="cpu")

keys = list(ckpt.keys())

# Categorize all keys
lora_a  = [k for k in keys if k.endswith('.lora_A')]
lora_b  = [k for k in keys if k.endswith('.lora_B')]
w_skel  = [k for k in keys if k.endswith('.w_skel')]
scale   = [k for k in keys if k.endswith('.scale')]
weights = [k for k in keys if k.endswith('.weight')]
biases  = [k for k in keys if k.endswith('.bias')]

print(f"Total keys: {len(keys)}")
print(f"  lora_A:  {len(lora_a)}")
print(f"  lora_B:  {len(lora_b)}")
print(f"  w_skel:  {len(w_skel)}")
print(f"  scale:   {len(scale)}")
print(f"  .weight: {len(weights)}")
print(f"  .bias:   {len(biases)}")

print(f"\nFirst 5 lora_A keys:")
for k in lora_a[:5]:
    t = ckpt[k]
    nonzero = (t != 0).sum().item()
    print(f"  {k}")
    print(f"    shape={t.shape}, nonzero={nonzero}/{t.numel()}, "
          f"min={t.min():.4f}, max={t.max():.4f}")

print(f"\nWhich blocks contain LoRA?")
blocks = set()
for k in lora_a:
    # Extract block name: mid_block, down_blocks.X, up_blocks.X
    parts = k.split('.')
    if parts[0] == 'mid_block':
        blocks.add('mid_block')
    elif len(parts) > 1:
        blocks.add(f"{parts[0]}.{parts[1]}")
for b in sorted(blocks):
    count = sum(1 for k in lora_a if k.startswith(b))
    print(f"  {b}: {count} lora_A layers")

print(f"\nLoRA rank (from first lora_A shape):")
if lora_a:
    shape = ckpt[lora_a[0]].shape
    print(f"  rank = {shape[0]}  (lora_A shape: {shape})")