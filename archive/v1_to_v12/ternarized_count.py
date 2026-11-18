import torch
import os

ckpt = torch.load("sdxl_bitlora_hybrid_v111_best.pt", map_location="cpu")

total_params = 0
ternary_params = 0
ternary_layers = []
fp16_layers = []

# Group keys by layer
seen_layers = {}
for key in ckpt.keys():
    # Get the layer name (everything before the last dot)
    parts = key.rsplit('.', 1)
    if len(parts) == 2:
        layer_name, param_name = parts
    else:
        continue
    
    if layer_name not in seen_layers:
        seen_layers[layer_name] = []
    seen_layers[layer_name].append(param_name)

# Classify each layer
for layer_name, param_names in seen_layers.items():
    is_ternary = 'w_skel' in param_names  # your ternary marker
    
    # Count params in this layer
    for param_name in param_names:
        key = f"{layer_name}.{param_name}"
        t = ckpt[key]
        n = t.numel()
        total_params += n
        if is_ternary:
            ternary_params += n

    if is_ternary:
        ternary_layers.append(layer_name)
    else:
        fp16_layers.append(layer_name)

# Memory calculation
# FP16 = 2 bytes per param
# Ternary = theoretically 1.58 bits = ~0.2 bytes per param
# But stored as INT8 in practice = 1 byte per param
fp16_size_mb = total_params * 2 / 1024**2
ternary_stored_mb = ternary_params * 1 / 1024**2   # stored as int8
ternary_ideal_mb  = ternary_params * 0.2 / 1024**2  # true 1.58-bit
remaining_fp16_mb = (total_params - ternary_params) * 2 / 1024**2

print(f"=== Parameter Count ===")
print(f"Total params:    {total_params/1e6:.2f}M")
print(f"Ternary params:  {ternary_params/1e6:.2f}M  ({100*ternary_params/total_params:.1f}%)")
print(f"FP16 params:     {(total_params-ternary_params)/1e6:.2f}M  ({100*(total_params-ternary_params)/total_params:.1f}%)")

print(f"\n=== Memory ===")
print(f"Original FP16 size:       {fp16_size_mb:.0f} MB")
print(f"Ternary layers (stored):  {ternary_stored_mb:.0f} MB  (as INT8)")
print(f"Ternary layers (ideal):   {ternary_ideal_mb:.0f} MB  (true 1.58-bit)")
print(f"Remaining FP16:           {remaining_fp16_mb:.0f} MB")
print(f"Actual saving (INT8):     {fp16_size_mb - (ternary_stored_mb + remaining_fp16_mb):.0f} MB")
print(f"Theoretical saving:       {fp16_size_mb - (ternary_ideal_mb + remaining_fp16_mb):.0f} MB")

print(f"\n=== Ternary layers ({len(ternary_layers)}) ===")
for l in ternary_layers[:5]:
    print(f"  {l}")
if len(ternary_layers) > 5:
    print(f"  ... and {len(ternary_layers)-5} more")

print(f"\nTotal FP16 layers: {len(fp16_layers)}")