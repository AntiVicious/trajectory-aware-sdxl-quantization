"""
One-shot patch for score_extra_metrics_v19.py.

Handles both the original and partially-patched versions. Scans the file
line-by-line and replaces `X = MODEL.get_*_features(...)` followed by
`X = F.normalize(X, dim=-1)` with a safe 3-line version that unwraps
the model output object if needed.

Usage:
    python patch_score_extra_metrics.py
"""
import os
import re
import sys
from pathlib import Path

candidates = [
    Path(os.environ.get("SDXL_DATA_ROOT", ".")) / "trajectory/phase4_full_bench/score_extra_metrics_v19.py",
    Path("./score_extra_metrics_v19.py"),
]
target = None
for c in candidates:
    if c.exists():
        target = c
        break

if target is None:
    print("ERROR: score_extra_metrics_v19.py not found.")
    sys.exit(1)

print(f"Patching: {target}")
src = target.read_text()

lines = src.split("\n")
new_lines = []
i = 0
patches = 0

while i < len(lines):
    line = lines[i]

    # Match: <indent>NAME = MODEL.get_image_features(ARGS)
    #     or <indent>NAME = MODEL.get_text_features(ARGS)
    m = re.match(
        r'^(\s+)(\w+)\s*=\s*(\w+)\.get_(image|text)_features\((.*)\)\s*$',
        line
    )
    if m:
        indent = m.group(1)
        var = m.group(2)
        model = m.group(3)
        kind = m.group(4)        # "image" or "text"
        args = m.group(5)

        # Look at next 1-3 lines for F.normalize(var, dim=-1) on the same var
        normalize_idx = None
        for j in range(i + 1, min(i + 4, len(lines))):
            if re.match(rf'^\s+{var}\s*=\s*F\.normalize\({var}\s*,\s*dim\s*=\s*-1\)\s*$', lines[j]):
                normalize_idx = j
                break

        if normalize_idx is not None:
            embeds_attr = "image_embeds" if kind == "image" else "text_embeds"
            tmp_var = var + "_out"
            new_lines.append(f"{indent}{tmp_var} = {model}.get_{kind}_features({args})")
            new_lines.append(
                f"{indent}{var} = {tmp_var} if hasattr({tmp_var}, 'norm') else "
                f"({tmp_var}.{embeds_attr} if hasattr({tmp_var}, '{embeds_attr}') "
                f"else {tmp_var}.pooler_output)"
            )
            for k in range(i + 1, normalize_idx):
                new_lines.append(lines[k])
            new_lines.append(lines[normalize_idx])
            i = normalize_idx + 1
            patches += 1
            continue

    new_lines.append(line)
    i += 1

if patches == 0:
    if ("_out = " in src and "pooler_output" in src and "image_embeds" in src):
        print("File appears already fully patched. No changes needed.")
        sys.exit(0)
    else:
        print("WARNING: No patches applied AND no signs of prior patching.")
        print(f"  Inspect manually: grep -n 'get_image_features\\|get_text_features' {target}")
        sys.exit(1)

new_src = "\n".join(new_lines)
target.write_text(new_src)
print(f"Applied {patches} patches.")

content = target.read_text()
print("\nVerification:")
for marker in ["pooler_output", "image_embeds", "text_embeds", "hasattr"]:
    n = content.count(marker)
    print(f"  '{marker}': {n} occurrences")

print("\nRe-run with:")
print("  CUDA_VISIBLE_DEVICES=1 python -u trajectory/phase4_full_bench/score_extra_metrics_v19.py")
