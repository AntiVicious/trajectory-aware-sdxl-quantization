"""
Diagnose whether V21-40 / V21-55 / V21-70 actually use different quantization sets.

Loads each budget's allocation, counts FP16/INT4 splits, computes total memory.

If all three budgets give ~the same set of NF4 layers, V21's invariance to
budget is just because the allocations are largely overlapping at the Linear
layer level (Conv2d gets dropped because bnb skips Conv2d).

Usage:
    python diagnose_v21_budgets.py
"""
import os
from pathlib import Path
import json

BASE = Path(os.environ.get("SDXL_DATA_ROOT", "."))
ALLOC_DIR = BASE / "trajectory" / "phase3_allocator_full"
LAYERS_JSON = BASE / "trajectory" / "phase1_atlas_full" / "layers_full.json"

with open(LAYERS_JSON) as f:
    layers_data = json.load(f)["layers"]
layer_to_meta = {L["name"]: L for L in layers_data}


def load_alloc(pct):
    p = ALLOC_DIR / f"allocation_{pct}pct_full.json"
    with open(p) as f:
        return json.load(f)


def classify_layer_type(name):
    # Quick check using layers_full.json metadata
    meta = layer_to_meta.get(name, {})
    return meta.get("type", "unknown")  # "Linear" or "Conv2d"


print(f"{'Budget':<10} {'FP16':>6} {'INT4':>6} {'INT4@t0':>8} {'INT4@t5':>8} {'INT4@t10':>9} {'INT4@t15':>9} {'INT4@t19':>9}")
print("=" * 80)

for pct in [70, 55, 40]:
    a = load_alloc(pct)
    counts = {"FP16": 0, "INT4@t0": 0, "INT4@t5": 0, "INT4@t10": 0,
              "INT4@t15": 0, "INT4@t19": 0}
    for x in a["allocations"]:
        counts[x["choice"]] = counts.get(x["choice"], 0) + 1
    int4_total = sum(v for k, v in counts.items() if k.startswith("INT4"))
    print(f"{pct}%        {counts['FP16']:>6} {int4_total:>6} "
          f"{counts['INT4@t0']:>8} {counts['INT4@t5']:>8} {counts['INT4@t10']:>9} "
          f"{counts['INT4@t15']:>9} {counts['INT4@t19']:>9}")

print()

# Now: which layers are actually QUANTIZED by V21? Only Linear layers in INT4 set.
# V21's effective NF4 set = {layer for layer in INT4_set if layer is Linear}
print("Effective NF4 set (only Linear layers — Conv2d are kept FP16 in V21):")
print(f"{'Budget':<10} {'NF4 Linear layers':>18} {'Skipped (Conv2d)':>17}")
print("=" * 50)

nf4_sets = {}
for pct in [70, 55, 40]:
    a = load_alloc(pct)
    nf4_layers = []
    skipped_conv = []
    for x in a["allocations"]:
        if x["choice"] == "FP16":
            continue
        layer_name = x["layer"]
        if classify_layer_type(layer_name) == "Linear":
            nf4_layers.append(layer_name)
        else:
            skipped_conv.append(layer_name)
    nf4_sets[pct] = set(nf4_layers)
    print(f"{pct}%        {len(nf4_layers):>18} {len(skipped_conv):>17}")

print()

# How much do the sets differ?
print("Set overlap (Jaccard intersection between budget pairs):")
for a, b in [(70, 55), (70, 40), (55, 40)]:
    common = len(nf4_sets[a] & nf4_sets[b])
    total = len(nf4_sets[a] | nf4_sets[b])
    overlap_pct = 100 * common / total if total > 0 else 0
    only_a = nf4_sets[a] - nf4_sets[b]
    only_b = nf4_sets[b] - nf4_sets[a]
    print(f"  {a}% vs {b}%: {common} shared, {len(only_a)} only-{a}, {len(only_b)} only-{b}, "
          f"overlap={overlap_pct:.1f}%")

# Check threshold differences for shared layers
print()
print("For shared layers between 70% and 40%, do thresholds differ?")
shared = nf4_sets[70] & nf4_sets[40]
allocs_70 = {x["layer"]: x["choice"] for x in load_alloc(70)["allocations"]}
allocs_40 = {x["layer"]: x["choice"] for x in load_alloc(40)["allocations"]}
threshold_diffs = {}
for layer in shared:
    c70 = allocs_70[layer]
    c40 = allocs_40[layer]
    if c70 != c40:
        key = (c70, c40)
        threshold_diffs[key] = threshold_diffs.get(key, 0) + 1
total_shared = len(shared)
total_changed = sum(threshold_diffs.values())
print(f"  Shared layers: {total_shared}")
print(f"  Same threshold in both: {total_shared - total_changed}")
print(f"  Different threshold:    {total_changed}")
print(f"  Most common transitions:")
for (c70, c40), n in sorted(threshold_diffs.items(), key=lambda x: -x[1])[:5]:
    print(f"    {c70:<10} -> {c40:<10}: {n} layers")

# CRUCIAL: For V21, ALL non-FP16 layers run quantized from step 10 onward
# regardless of their threshold. So the actual quantization schedule is:
# "always-NF4-from-step-10" for all layers in NF4 set.
# The only thing that varies between budgets is WHICH layers are in the NF4 set.

print()
print("CRUCIAL OBSERVATION: V21 runs all NF4 layers identically from step 10 on,")
print("regardless of their V13 commit threshold. So the budget difference comes")
print("from how many layers are in the NF4 set, not from per-layer scheduling.")
print()

# Simulate V21's behavior: number of "FP16" steps each layer gets in V21
print("V21's effective per-layer commit (all NF4 set layers commit at t=10 in V21):")
for pct in [70, 55, 40]:
    a = load_alloc(pct)
    fp16_layers = sum(1 for x in a["allocations"] if x["choice"] == "FP16")
    nf4_linear = len([x for x in a["allocations"]
                       if x["choice"] != "FP16" and
                       classify_layer_type(x["layer"]) == "Linear"])
    fp16_conv2d = len([x for x in a["allocations"]
                        if x["choice"] != "FP16" and
                        classify_layer_type(x["layer"]) == "Conv2d"])
    print(f"  {pct}%: {fp16_layers} explicit FP16 + {fp16_conv2d} Conv2d-as-FP16 "
          f"= {fp16_layers + fp16_conv2d} FP16 throughout, "
          f"{nf4_linear} layers FP16-then-NF4@t10")
