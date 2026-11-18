"""
Generate a qualitative grid figure showing V19 vs V1, V13, V12, V2.
Outputs a single PNG suitable for the thesis Chapter 5 qualitative section.

Picks N prompts at random with diverse content. Renders each prompt as a row
across columns: V1 (FP16), V19-70 (best), V13-70 (framework), V12 (prior best), V2 (naive).

Usage:
    python make_qualitative_grid_v19.py [--n-prompts 10]
"""
import os
import argparse
import json
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
OUT = BASE / "benchmark_5k"

VARIANTS = [
    ("v1_fp16",                       "V1 — FP16 baseline"),
    ("v19_70pct_trajectory_v12quant", "V19-70 (DSim 0.017)"),
    ("v13_70pct_trajectory",          "V13-70 (DSim 0.026)"),
    ("v19_55pct_trajectory_v12quant", "V19-55 (DSim 0.032)"),
    ("v13_40pct_trajectory",          "V13-40 (DSim 0.166)"),
    ("v12_outlier_preserved",         "V12 (DSim 0.188)"),
]

PROMPTS = json.load(open(BASE / "benchmark_prompts_5k.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=10,
                    help="Number of prompt rows in the grid")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cell-size", type=int, default=320)
    args = ap.parse_args()

    # Pick prompts where ALL variants exist
    def has_all(item):
        return all((OUT / v / item['filename']).exists() for v, _ in VARIANTS)

    candidates = [p for p in PROMPTS if has_all(p)]
    print(f"Found {len(candidates)} prompts with images in all {len(VARIANTS)} variants")

    np.random.seed(args.seed)
    chosen = list(np.random.choice(candidates, size=min(args.n_prompts, len(candidates)),
                                    replace=False))

    # Layout
    cell_w = args.cell_size
    cell_h = args.cell_size
    cap_w = 220
    pad = 8
    header_h = 40
    n_cols = len(VARIANTS)
    n_rows = len(chosen)
    total_w = cap_w + n_cols * (cell_w + pad) + pad
    total_h = header_h + n_rows * (cell_h + pad) + pad

    fig = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(fig)

    try:
        font_h = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_c = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        font_h = ImageFont.load_default()
        font_c = ImageFont.load_default()

    # Headers
    for col_idx, (_, label) in enumerate(VARIANTS):
        x = cap_w + pad + col_idx * (cell_w + pad)
        bbox = draw.textbbox((0, 0), label, font=font_h)
        text_w = bbox[2] - bbox[0]
        draw.text((x + cell_w // 2 - text_w // 2, pad + 5),
                  label, fill="black", font=font_h)

    # Rows
    for row_idx, item in enumerate(chosen):
        y = header_h + pad + row_idx * (cell_h + pad)
        wrapped = textwrap.fill(item['caption'], width=26)
        # Caption left-aligned in cap_w column
        draw.text((pad, y + cell_h // 2 - 30), wrapped,
                  fill="black", font=font_c)
        # Filename below caption
        draw.text((pad, y + cell_h - 16), item['filename'],
                  fill="gray", font=font_c)
        for col_idx, (variant, _) in enumerate(VARIANTS):
            img_path = OUT / variant / item['filename']
            x = cap_w + pad + col_idx * (cell_w + pad)
            if img_path.exists():
                img = Image.open(img_path).convert("RGB").resize(
                    (cell_w, cell_h), Image.LANCZOS)
                fig.paste(img, (x, y))
            else:
                draw.rectangle([x, y, x + cell_w, y + cell_h],
                               outline="red", width=2)

    out_path = BASE / "qualitative_grid_v19.png"
    fig.save(out_path, dpi=(150, 150))
    print(f"Saved: {out_path}  ({fig.size[0]}x{fig.size[1]})")
    print(f"\nCopy to laptop with:")
    print(f"  scp arjun.badola@singularity:{out_path} ./")


if __name__ == "__main__":
    main()
