import torch
import os

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")

# Check your checkpoint exists
checkpoint_path = "sdxl_bitlora_hybrid_v111_best.pt"  # <-- update this
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print(f"\nCheckpoint loaded OK")
    print(f"Keys in checkpoint: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
    print(f"Checkpoint size: {os.path.getsize(checkpoint_path) / 1024**2:.1f} MB")
else:
    print(f"\nCheckpoint NOT FOUND at: {checkpoint_path}")
    print("Please check your path or filename")