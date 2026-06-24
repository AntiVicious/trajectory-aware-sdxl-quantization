#!/usr/bin/env python3
"""
Generate qualitative_grid_extended.png

Builds an 8-column qualitative comparison grid:
  V1 (FP16) | V2 (naive INT4) | V12 (outlier+ternary) |
  V13-70 | V20-70 | V21-70 | V22-70 floor5 | V24b

Rows: a fixed list of prompts (default 6 indices).

Uses EXPLICIT directory names to avoid substring mismatches like
v1 -> v12_outlier_preserved.

Override directory names with --variant-dirs if your benchmark_5k/
folders are named differently.
"""
import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# Default mapping: (column header for the figure) -> (exact subdirectory name under bench-root)
# Edit this list, or pass --variant-dirs on the command line, if your dirs are different.
DEFAULT_VARIANT_DIRS = [
    ("V1 (FP16)",            "v1_fp16"),
    ("V2 (naive INT4)",      "v2_int4_naive"),
    ("V12 (outlier+ternary)", "v12_outlier_preserved"),
    ("V13-70",               "v13_70pct_trajectory"),
    ("V20-70 (NF4 static)",  "v20_70pct_nf4_static"),
    ("V21-70 (FP16+NF4)",    "v21_70pct_two_pass_t10_v2"),
    ("V22-70 floor5",        "v22_70pct_floor5"),
    ("V24b (INT8 PTQ)",      "v24b_int8_full"),
]


def find_default_bench_root():
    candidates = [
        os.path.expanduser("~/Documents/Maharshi/gem/benchmark_5k"),
        "benchmark_5k",
        "../benchmark_5k",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


def find_default_prompts_json():
    candidates = [
        os.path.expanduser("~/Documents/Maharshi/gem/benchmark_prompts_5k.json"),
        "benchmark_prompts_5k.json",
        "../benchmark_prompts_5k.json",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def find_image(variant_dir: Path, prompt_idx: int):
    """Find image file for prompt_idx under variant_dir.

    Tries common naming conventions then falls back to glob.
    """
    candidates = [
        variant_dir / f"{prompt_idx:04d}.png",
        variant_dir / f"{prompt_idx:04d}.jpg",
        variant_dir / f"{prompt_idx}.png",
        variant_dir / f"{prompt_idx}.jpg",
        variant_dir / f"img_{prompt_idx:04d}.png",
        variant_dir / f"sample_{prompt_idx:04d}.png",
        variant_dir / f"image_{prompt_idx:04d}.png",
        variant_dir / f"prompt_{prompt_idx:04d}.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback: glob anything containing the index
    for f in variant_dir.glob(f"*{prompt_idx:04d}*"):
        if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
            return f
    return None


def parse_variant_dirs(spec):
    """Parse --variant-dirs spec like 'V1=v1_fp16,V2=v2_int4_naive,...'"""
    result = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                f"Bad --variant-dirs entry {chunk!r}; expected HEADER=DIRNAME"
            )
        header, dirname = chunk.split("=", 1)
        result.append((header.strip(), dirname.strip()))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench-root", default=find_default_bench_root())
    p.add_argument("--prompts-json", default=find_default_prompts_json())
    p.add_argument("--out", required=True)
    p.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=[12, 47, 88, 153, 240, 410],
        help="Prompt indices to display as rows (default: 6 picks)",
    )
    p.add_argument("--thumb-size", type=int, default=400)
    p.add_argument(
        "--variant-dirs",
        default=None,
        help=(
            "Override variant -> dir mapping. Format: "
            "'V1=v1_fp16,V2=v2_int4_naive,V12=v12_outlier_preserved,...' "
            "If omitted, uses the DEFAULT_VARIANT_DIRS list at the top of the script."
        ),
    )
    args = p.parse_args()

    bench_root = Path(args.bench_root)
    if not bench_root.exists():
        print(f"ERROR: --bench-root {bench_root} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.variant_dirs:
        variants = parse_variant_dirs(args.variant_dirs)
    else:
        variants = DEFAULT_VARIANT_DIRS

    print(f"Looking under {bench_root}/")
    available = {p.name for p in bench_root.iterdir() if p.is_dir()}

    # Resolve each variant by EXACT name
    resolved = []
    for header, dirname in variants:
        if dirname in available:
            resolved.append((header, dirname))
            print(f"  OK   {header:30s} -> {dirname}")
        else:
            resolved.append((header, None))
            print(f"  MISS {header:30s} -> {dirname} not found", file=sys.stderr)

    n_missing = sum(1 for _, d in resolved if d is None)
    if n_missing > 0:
        print(
            f"\n{n_missing} variant(s) not found. Available dirs:\n  "
            + "\n  ".join(sorted(available))
            + "\n\nFix by either renaming dirs OR passing "
            "--variant-dirs 'V1=actual_name,V2=actual_name,...' on the command line.",
            file=sys.stderr,
        )

    # Load prompts (for row labels)
    prompts = []
    if os.path.exists(args.prompts_json):
        with open(args.prompts_json) as f:
            prompts = json.load(f)
        print(f"Loaded {len(prompts)} prompts from {args.prompts_json}")
    else:
        print(
            f"WARNING: prompts JSON not found at {args.prompts_json}; "
            "row labels will be blank.",
            file=sys.stderr,
        )

    n_rows = len(args.indices)
    n_cols = len(resolved)

    fig = plt.figure(figsize=(2.0 * n_cols + 2.0, 2.0 * n_rows + 0.6))
    gs = gridspec.GridSpec(
        n_rows + 1,
        n_cols + 1,
        width_ratios=[1.6] + [3] * n_cols,
        height_ratios=[0.4] + [3] * n_rows,
        wspace=0.03,
        hspace=0.03,
    )

    # Header row
    for j, (header, _) in enumerate(resolved):
        ax = fig.add_subplot(gs[0, j + 1])
        ax.text(0.5, 0.5, header, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.set_axis_off()

    # Body rows
    for i, idx in enumerate(args.indices):
        # Prompt text cell
        prompt_text = ""
        if prompts and idx < len(prompts):
            entry = prompts[idx]
            if isinstance(entry, dict):
                prompt_text = entry.get("caption") or entry.get("prompt") or ""
            else:
                prompt_text = str(entry)
        prompt_text = textwrap.fill(prompt_text, width=22)
        ax_text = fig.add_subplot(gs[i + 1, 0])
        ax_text.text(0.95, 0.5, prompt_text, ha="right", va="center", fontsize=8)
        ax_text.set_axis_off()

        for j, (header, dirname) in enumerate(resolved):
            ax = fig.add_subplot(gs[i + 1, j + 1])
            ax.set_axis_off()
            if dirname is None:
                ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=14, color="gray")
                continue
            img_path = find_image(bench_root / dirname, idx)
            if img_path is None:
                ax.text(
                    0.5, 0.5, f"missing\n{idx}",
                    ha="center", va="center", fontsize=8, color="gray",
                )
                continue
            img = Image.open(img_path)
            img.thumbnail((args.thumb_size, args.thumb_size))
            ax.imshow(img)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"\nSaved {args.out}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()