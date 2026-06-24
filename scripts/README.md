# Figure-generation scripts for thesis_v3

These three scripts produce the placeholder figures in `thesis_v3/Pictures/`:
- `atlas_heatmap_top100.png`
- `predictor_scatter.png`
- `qualitative_grid_extended.png`

## Where to put these scripts

Copy this entire folder into your project directory, e.g.:

    ~/Documents/Maharshi/gem/figure_scripts/

The scripts default to reading from these paths on your machine:
- `~/Documents/Maharshi/gem/trajectory/phase1_atlas_full/atlas_full.csv`
- `~/Documents/Maharshi/gem/trajectory/phase1_atlas_full/layers_full.json`
- `~/Documents/Maharshi/gem/trajectory/phase2_predictor_full/cv_results.csv`
- `~/Documents/Maharshi/gem/benchmark_5k/{v1,v2,v12,v13_70,...}/`
- `~/Documents/Maharshi/gem/benchmark_prompts_5k.json`

If your paths differ, pass them via the CLI flags shown below.

## Where to run from

Run from the directory that contains your `thesis_v3/` folder. For example
if your tree looks like this:

    ~/Documents/Maharshi/gem/
    |-- thesis_v3/
    |   `-- Pictures/   (output images go here)
    |-- trajectory/
    |   |-- phase1_atlas_full/
    |   `-- phase2_predictor_full/
    |-- benchmark_5k/
    `-- figure_scripts/   (extracted here)

Then `cd ~/Documents/Maharshi/gem` and run:

```bash
# Activate your sdxl conda env first if needed
conda activate sdxl

python figure_scripts/generate_atlas_heatmap.py \
    --out thesis_v3/Pictures/atlas_heatmap_top100.png

python figure_scripts/generate_predictor_scatter.py \
    --out thesis_v3/Pictures/predictor_scatter.png

python figure_scripts/generate_qualitative_grid.py \
    --out thesis_v3/Pictures/qualitative_grid_extended.png \
    --indices 12 47 88 153 240 410
```

## Dependencies

```
pip install pandas numpy matplotlib scipy scikit-learn pillow
```
(All already in your sdxl env.)

## Per-script notes

### generate_atlas_heatmap.py
Reads atlas_full.csv. Auto-detects column names (`layer`/`layer_name`/`name`,
`t`/`commit_step`/`step`/`timestep`, `clip_dist`/`distance`/`clip_distance`).
If your columns are named differently it prints what it tried and exits.

Override defaults:
```bash
python figure_scripts/generate_atlas_heatmap.py \
    --atlas-csv /path/to/your/atlas_full.csv \
    --out thesis_v3/Pictures/atlas_heatmap_top100.png \
    --top-k 100
```

### generate_predictor_scatter.py
Two modes:

(A) Fast: needs cv_results.csv with columns `model`, `true_log10`, `pred_log10`:
```bash
python figure_scripts/generate_predictor_scatter.py \
    --cv-csv /path/to/cv_results.csv \
    --out thesis_v3/Pictures/predictor_scatter.png
```

(B) Slow (~5 min): re-fits all four predictors from the atlas + layer metadata:
```bash
python figure_scripts/generate_predictor_scatter.py --retrain \
    --atlas-csv /path/to/atlas_full.csv \
    --layers-json /path/to/layers_full.json \
    --out thesis_v3/Pictures/predictor_scatter.png
```

If your `cv_results.csv` has different column names, look at the
auto-detection lists at the top of `from_cv_csv()` in the script and edit them.

### generate_qualitative_grid.py
Needs a directory layout like:
    benchmark_5k/v1/0012.png
    benchmark_5k/v1/0047.png
    benchmark_5k/v12/0012.png
    ...
The script tries `{idx:04d}.png`, `{idx}.png`, `img_{idx:04d}.png`, then
falls back to glob. If your filenames are different, edit the `find_image()`
function in the script (small change).

The expected variant directories are listed in the COLUMNS list at the top
of the script:

    v1, v2, v12, v13_70, v20_70, v21_70, v22_70_floor5, v24b

If your dirs are named differently (e.g., `V13-70` instead of `v13_70`),
the script does soft prefix matching, but if that fails edit the COLUMNS
list to match your actual directory names.

Pick 6 representative prompt indices via `--indices`. Run once, look at
the output, swap indices for boring rows, re-run.

```bash
python figure_scripts/generate_qualitative_grid.py \
    --bench-root ~/Documents/Maharshi/gem/benchmark_5k \
    --prompts-json ~/Documents/Maharshi/gem/benchmark_prompts_5k.json \
    --out thesis_v3/Pictures/qualitative_grid_extended.png \
    --indices 12 47 88 153 240 410
```
