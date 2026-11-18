"""
Make a side-by-side comparison grid: V1 / V13-70 / V19-70 / V12

Generates three grids:
  - V19 wins (V19 closer to V1 than V13) — these test whether V19 is better
  - V13 wins (V13 closer to V1 than V19) — these test if V19 has failure cases
  - Ties (both very close to V1) — these test that the framework basically works

Usage:
    python make_comparison_grid.py
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
import textwrap

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
OUT = BASE / "benchmark_5k"

# Filenames from inspect_v19_vs_v13.py output
V19_WINS = [
    "000000039480.jpg",
    "000000054164.jpg",
    "000000052591.jpg",
    "000000002261.jpg",
    "000000011699.jpg",
]
V13_WINS = [
    "000000010764.jpg",
    "000000033221.jpg",
    "000000013177.jpg",
    "000000046048.jpg",
    "000000031217.jpg",
]
TIES = [
    "000000026564.jpg",
    "000000006614.jpg",
    "000000034417.jpg",
    "000000042888.jpg",
    "000000000776.jpg",
]

VARIANTS = [
    ("v1_fp16", "V1 (FP16)"),
    ("v13_70pct_trajectory", "V13-70 (DSim mean=0.026)"),
    ("v19_70pct_trajectory_v12quant", "V19-70 (DSim mean=0.017)"),
    ("v12_outlier_preserved", "V12 (DSim mean=0.188)"),
]

# Load prompt captions for labels
PROMPTS = json.load(open(BASE / "benchmark_prompts_5k.json"))
caption_by_filename = {item['filename']: item['caption'] for item in PROMPTS}


def make_grid(filenames, title, output_path):
    if not filenames:
        print(f"  [{title}] No filenames provided, skipping")
        return

    cell_w = 320
    cell_h = 320
    header_h = 36
    caption_w = 200
    pad = 8

    n_rows = len(filenames)
    n_cols = len(VARIANTS)
    W = caption_w + n_cols * (cell_w + pad) + pad
    H = header_h + n_rows * (cell_h + pad) + pad + 30  # +30 for title

    grid = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_caption = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        font_title = ImageFont.load_default()
        font_header = ImageFont.load_default()
        font_caption = ImageFont.load_default()

    # Title at top
    draw.text((pad, 5), title, fill="black", font=font_title)

    # Column headers
    y_header = 30
    for c, (_, label) in enumerate(VARIANTS):
        x = caption_w + pad + c * (cell_w + pad)
        bbox = draw.textbbox((0, 0), label, font=font_header)
        text_w = bbox[2] - bbox[0]
        draw.text((x + cell_w // 2 - text_w // 2, y_header + 8),
                  label, fill="black", font=font_header)

    # Rows
    for r, fn in enumerate(filenames):
        y = header_h + 30 + r * (cell_h + pad)

        # Caption (truncated, wrapped)
        caption = caption_by_filename.get(fn, fn)
        if len(caption) > 80:
            caption = caption[:77] + "..."
        wrapped = textwrap.fill(caption, width=24)
        draw.text((pad, y + cell_h // 2 - 30), wrapped,
                  fill="black", font=font_caption)
        # Add filename below caption
        draw.text((pad, y + cell_h - 20), fn, fill="gray", font=font_caption)

        # Images
        for c, (folder, _) in enumerate(VARIANTS):
            x = caption_w + pad + c * (cell_w + pad)
            p = OUT / folder / fn
            if p.exists():
                img = Image.open(p).convert("RGB").resize(
                    (cell_w, cell_h), Image.LANCZOS)
                grid.paste(img, (x, y))
            else:
                draw.rectangle([x, y, x + cell_w, y + cell_h],
                               outline="red", width=2)
                draw.text((x + 10, y + cell_h // 2),
                          f"MISSING\n{folder}", fill="red", font=font_caption)

    grid.save(output_path)
    print(f"  Saved: {output_path}  ({grid.size[0]}x{grid.size[1]})")


def main():
    out_dir = BASE
    print(f"Output directory: {out_dir}")

    make_grid(V19_WINS, "V19 wins — V19 closer to V1 than V13",
              out_dir / "comparison_v19_wins.png")
    make_grid(V13_WINS, "V13 wins — V13 closer to V1 than V19",
              out_dir / "comparison_v13_wins.png")
    make_grid(TIES, "Ties — V13 and V19 similarly close to V1",
              out_dir / "comparison_ties.png")

    print("\nDone. Copy to laptop with:")
    print(f"  scp arjun.badola@singularity:{out_dir}/comparison_*.png ./")


if __name__ == "__main__":
    main()