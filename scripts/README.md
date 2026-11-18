# Figure Scripts

These scripts generate three thesis figures in `docs/thesis/Pictures/`:

- `atlas_heatmap_top100.png`
- `predictor_scatter.png`
- `qualitative_grid_extended.png`

`build_all_thesis_artifacts.py` builds the remaining figures in `figures/`.

## Setup

```bash
pip install pandas numpy matplotlib scipy scikit-learn pillow
export SDXL_DATA_ROOT=/path/to/your/sdxl/working/dir
```

By default the scripts read from:

- `$SDXL_DATA_ROOT/trajectory/phase1_atlas_full/atlas_full.csv`
- `$SDXL_DATA_ROOT/trajectory/phase1_atlas_full/layers_full.json`
- `$SDXL_DATA_ROOT/trajectory/phase2_predictor_full/cv_results.csv`
- `$SDXL_DATA_ROOT/benchmark_5k/{v1,v2,v12,v13_70,...}/`
- `$SDXL_DATA_ROOT/benchmark_prompts_5k.json`

Each path can be overridden with the flags shown below. Run from the
repository root.

## generate_atlas_heatmap.py

Reads the atlas CSV. Column names are detected automatically
(`layer`/`layer_name`/`name`, `t`/`commit_step`/`step`/`timestep`,
`clip_dist`/`distance`/`clip_distance`); if none match, the script lists what
it tried and exits.

```bash
python scripts/generate_atlas_heatmap.py \
    --atlas-csv /path/to/atlas_full.csv \
    --out docs/thesis/Pictures/atlas_heatmap_top100.png \
    --top-k 100
```

## generate_predictor_scatter.py

Two modes:

- **From cross-validation results** (fast): needs `cv_results.csv` with
  `model`, `true_log10` and `pred_log10` columns. The published copy is
  `data/predictor/cv_results.csv`.

  ```bash
  python scripts/generate_predictor_scatter.py \
      --cv-csv data/predictor/cv_results.csv \
      --out docs/thesis/Pictures/predictor_scatter.png
  ```

- **Retrain** (~5 minutes): refits all four predictors from the atlas and
  layer metadata.

  ```bash
  python scripts/generate_predictor_scatter.py --retrain \
      --atlas-csv /path/to/atlas_full.csv \
      --layers-json data/atlas/layers_full.json \
      --out docs/thesis/Pictures/predictor_scatter.png
  ```

## generate_qualitative_grid.py

Expects one directory of generated images per variant:

    benchmark_5k/v1/0012.png
    benchmark_5k/v12/0012.png
    ...

File names `{idx:04d}.png`, `{idx}.png` and `img_{idx:04d}.png` are tried
before falling back to a glob (see `find_image()`). The variant directories
are listed in `COLUMNS` at the top of the script:

    v1, v2, v12, v13_70, v20_70, v21_70, v22_70_floor5, v24b

Directory names are matched by prefix, so `V13-70` also matches `v13_70`.
`--indices` selects the prompt rows to show.

```bash
python scripts/generate_qualitative_grid.py \
    --bench-root $SDXL_DATA_ROOT/benchmark_5k \
    --prompts-json data/benchmark_prompts_5k.json \
    --out docs/thesis/Pictures/qualitative_grid_extended.png \
    --indices 12 47 88 153 240 410
```
