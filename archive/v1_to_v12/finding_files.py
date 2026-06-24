import os
from PIL import Image

BASE = "/home/project/Documents/Maharshi/gem/benchmark_5k/"

for variant in os.listdir(BASE):
    folder = os.path.join(BASE, variant)
    if not os.path.isdir(folder):
        continue
    corrupted = []
    for fname in os.listdir(folder):
        if not fname.endswith('.jpg'):
            continue
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path)
            img.verify()
        except Exception:
            corrupted.append(fname)
    if corrupted:
        print(f"{variant}: {len(corrupted)} corrupted files")
        for f in corrupted:
            path = os.path.join(folder, f)
            os.remove(path)
            print(f"  deleted: {f}")
    else:
        print(f"{variant}: OK")