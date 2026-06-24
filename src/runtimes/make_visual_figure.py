"""
Build a qualitative comparison figure for the thesis.

Picks N diverse prompts, lays out a grid showing V1 / V12 / V13-40 for each.
Output is one PNG that goes directly into the thesis.

Selection strategy:
  - Sample prompts that are visually diverse (different content categories
    inferred from caption keywords)
  - Prioritize prompts where all four variants generated images
  - Annotate each row with a short caption excerpt

Usage:
    python make_visual_figure.py [--n 6]
"""
import argparse
import json
import re
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

BASE = Path("/home/project/Documents/Maharshi/gem")
OUT = BASE / "benchmark_5k"
PROMPTS = json.load(open(BASE / "benchmark_prompts_5k.json"))
FIGURE_OUT = BASE / "trajectory" / "phase4_full_bench" / "figures"
FIGURE_OUT.mkdir(parents=True, exist_ok=True)

# Variants to include in figure (column order)
COLUMNS = [
    ("v1_fp16",                "V1 (FP16, 4487 MB)"),
    ("v12_outlier_preserved",  "V12 (1248 MB, prior best)"),
    ("v13_55pct_trajectory",   "V13-55 (2822 MB)"),
    ("v13_40pct_trajectory",   "V13-40 (2052 MB)"),
]


def has_keywords(caption, kws):
    c = caption.lower()
    return any(kw in c for kw in kws)


# Diverse content categories — pick one prompt from each
CATEGORIES = [
    ("animal",   ["dog", "cat", "bear", "horse", "elephant", "zebra", "bird"]),
    ("person",   ["man", "woman", "person", "people", "child"]),
    ("food",     ["pizza", "cake", "food", "fruit", "bowl"]),
    ("vehicle",  ["car", "truck", "train", "bus", "motorcycle"]),
    ("indoor",   ["room", "kitchen", "bedroom", "bathroom"]),
    ("outdoor",  ["beach", "mountain", "park", "field", "street"]),
    ("object",   ["clock", "vase", "lamp", "computer", "phone"]),
    ("sports",   ["skateboard", "tennis", "ski", "surf", "skating"]),
]


def all_have_image(prompt_item):
    fn = prompt_item['filename']
    return all((OUT / v / fn).exists() for v, _ in COLUMNS)


def select_prompts(n, seed=42):
    np.random.seed(seed)
    chosen = []
    used_categories = set()

    # First pass: one per category
    for cat_name, kws in CATEGORIES:
        if len(chosen) >= n:
            break
        # Filter prompts in this category that have all images
        candidates = [p for p in PROMPTS
                     if has_keywords(p['caption'], kws) and all_have_image(p)]
        if candidates:
            picked = candidates[np.random.randint(len(candidates))]
            chosen.append(picked)
            used_categories.add(cat_name)

    # Fill remaining slots with any prompts that have all images
    if len(chosen) < n:
        remaining = [p for p in PROMPTS
                    if all_have_image(p) and p not in chosen]
        np.random.shuffle(remaining)
        chosen.extend(remaining[:n - len(chosen)])

    return chosen[:n]


def get_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_figure(prompts, image_size=384, padding=20, header_h=40, caption_h=60):
    cols = len(COLUMNS)
    rows = len(prompts)

    # Layout: caption_h on left for prompt text, header_h on top for variant labels,
    # then rows × cols grid of images at image_size each, padded.
    caption_w = 220
    grid_w = cols * (image_size + padding) - padding
    grid_h = rows * (image_size + padding) - padding

    total_w = caption_w + grid_w + 2 * padding
    total_h = header_h + grid_h + 2 * padding

    fig = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(fig)
    font_header = get_font(15)
    font_caption = get_font(11)

    # Headers (variant labels)
    for col_idx, (variant, label) in enumerate(COLUMNS):
        x = caption_w + padding + col_idx * (image_size + padding)
        y = padding
        # Center the label over the column
        bbox = draw.textbbox((0, 0), label, font=font_header)
        text_w = bbox[2] - bbox[0]
        draw.text((x + image_size // 2 - text_w // 2, y), label,
                  fill="black", font=font_header)

    # Rows
    for row_idx, item in enumerate(prompts):
        y = header_h + padding + row_idx * (image_size + padding)
        # Caption on left
        wrapped = textwrap.fill(item['caption'], width=28)
        draw.text((padding, y + image_size // 2 - 30), wrapped,
                  fill="black", font=font_caption)

        # Images
        for col_idx, (variant, _) in enumerate(COLUMNS):
            img_path = OUT / variant / item['filename']
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                img = img.resize((image_size, image_size), Image.LANCZOS)
                x = caption_w + padding + col_idx * (image_size + padding)
                fig.paste(img, (x, y))
            else:
                # missing image
                x = caption_w + padding + col_idx * (image_size + padding)
                draw.rectangle([x, y, x + image_size, y + image_size],
                              outline="red", width=2)
                draw.text((x + 20, y + image_size // 2),
                         "missing", fill="red", font=font_caption)

    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="Number of prompts to show")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Selecting {args.n} diverse prompts...")
    prompts = select_prompts(args.n, seed=args.seed)
    print(f"Selected {len(prompts)} prompts:")
    for p in prompts:
        print(f"  [{p['filename']}] {p['caption']}")

    print(f"\nBuilding figure...")
    fig = make_figure(prompts)

    out_path = FIGURE_OUT / f"qualitative_comparison_n{len(prompts)}.png"
    fig.save(out_path, dpi=(150, 150))
    print(f"Saved: {out_path}")
    print(f"Size: {fig.size[0]}x{fig.size[1]} px")


if __name__ == "__main__":
    main()
