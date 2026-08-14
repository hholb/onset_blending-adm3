# 2025 Weekly-Bin Blending Pipeline

Cross-validated weekly-bin multinomial onset blending. Combines climatology, NeuralGCM, and AIFS forecasts into multi-model ensembles, evaluates skill, and produces publication figures.

## Stages

### Stage 0: `0_connect_prepare_data_to_2025_pipeline.py`

Converts day-level wide pickle from the prepare_data pipeline into weekly-bin format. Aggregates daily onset probabilities into 4 weekly bins plus a "later" bin, computes logit-scale climatology features, and derives rain-based predictors.

Driven by YAML specs in `specs/2025_blend/connect_*.yml`:
- `--spec_id connect_fixed_cutoff`: Uses fixed climatological MOK date (June 1)

Each connect spec defines `mode`, `input_rds`, `output_rds`, `forecast_models` (with `rain_predictors` in dict format), and `climatology` prefix settings.

**Rain predictors** are specified per model as a list of `{ agg, window }` dicts. Three aggregation types are supported: `diff` (max rolling sum minus onset threshold), `max` (max rolling sum), and `min` (min rolling sum). The current default uses only `{ agg: diff, window: 3 }` for both ngcm and aifs. The window size is fully configurable — no code changes needed to add new predictors.

### Stage 1: `1_blend_evaluation.py`

Main cross-validation engine. Fits weekly-bin multinomial logistic regression models using formulas defined in the YAML spec. Formula terms ending in `_qx` are expanded automatically to `_week1` through `_week4` at runtime. Computes metrics (Brier, RPS, AUC), reliability plots, and (when enabled) optimized multi-model ensemble (MME) weights. The `summary_models` outputs in the results folder store the primary metrics of interest.

Currently active model formulas (in `cv_models_fixed_cutoff.yml`):
- `ngcm_blend`: climatology × ngcm diff
- `int_all`: climatology × ngcm diff × aifs diff (interaction)
- `add_blend`: climatology + ngcm diff + aifs diff (additive)
- `blended_model`: climatology × ngcm diff × aifs diff

MME optimisation and training-window variants are currently disabled (`mme.enabled: false`, `window_variants.enabled: false`).

### Stage 2: `2_2025_evaluation.py`

Out-of-sample evaluation of 2025 monsoon onset forecasts. Scores the blended model, climatologies, and Platt-calibrated forecasts against ground-truth variants. Stage 2 is only designed to run on the paper's dataset.

### Stage 3: `3_produce_figures.py`

Reads pre-computed model summaries and produces publication-ready figures: overall metrics by period, model skill comparisons, weekly performance, yearly time series, reliability diagrams, and spatial skill maps. Stage 3 is only designed to run on the paper's dataset.

---

## Running

All commands run from the repository root:

```bash
# Stage 0: Convert daily -> weekly bins
python python/pipelines/blending_process/0_connect_prepare_data_to_2025_pipeline.py --spec_id connect_fixed_cutoff

# Stage 1: CV evaluation
python python/pipelines/blending_process/1_blend_evaluation.py --spec_id cv_models_fixed_cutoff

# Parallel global-CV folds (models remain sequential)
python python/pipelines/blending_process/1_blend_evaluation.py --spec_id cv_models_fixed_cutoff --cores 4

# Stage 2: 2025 out-of-sample evaluation
python python/pipelines/blending_process/2_2025_evaluation.py

# Stage 3: Publication figures
python python/pipelines/blending_process/3_produce_figures.py
```

`--cores N` processes independent global holdout-year folds with up to `N`
worker processes, capped by the number of folds. Omitting it retains serial
execution. Model formulas are evaluated sequentially to avoid nested process
parallelism.

---

## Spec Files

Located in `specs/2025_blend/*.yml`. Key fields:

### `connect_*.yml`

| Field | Description |
|-------|-------------|
| `mode` | MOK filter mode: `fixed_cutoff` |
| `input_rds` | Path to combined wide pickle from prepare_data pipeline |
| `output_rds` | Output path for weekly-bin pickle |
| `day_max`, `days_per_bin`, `n_bins` | Weekly binning parameters |
| `climatology.base_prefix` | Column prefix for conditional climatology |
| `climatology.unconditional_prefix` | Column prefix for unconditional climatology |
| `forecast_models[].name` | Model name — drives all output column names downstream |
| `forecast_models[].variants` | Onset start-date variants to compute (earliest date an onset can count, e.g. `fixed_cutoff`) |
| `forecast_models[].rain_predictors` | List of `{ agg, window }` dicts (or legacy strings) |
| `forecast_models[].rain_day_max` | Exact model rainfall horizon. An explicit value activates strict `1..N` column/value validation; omission uses the legacy `day_max + 10` minimum and permits later columns. |
| `forecast_models[].rain_horizon_policy` | Optional compatibility override. `exact` is the default with explicit `rain_day_max`; `truncate` validates `1..N` but permits and ignores later rain columns. |

### `cv_models*.yml`

| Field | Description |
|-------|-------------|
| `run.cutoff_mode` | MOK filter mode (`fixed_cutoff`) |
| `run.training_years` | Years used for CV training (currently 2019–2022) |
| `run.cv_holdout_years` | Years used as CV holdout (currently 2019–2022) |
| `cv.methods` | CV strategy: `global` |
| `models.formulas` | Named multinomial logistic regression formulas using `_qx` shorthand |
| `models.window_variants` | Training-window climatology variants (currently disabled) |
| `mme` | Multi-model ensemble config (currently disabled) |
| `extras.clim_logits` | Climatology baseline model definitions |
| `extras.forecasts` | Per-system Platt calibration and scoring options |

---

## Inputs

- `Monsoon_Data/Processed_Data/Combined/*.pkl` (from prepare_data pipeline stage 3)
- `Monsoon_Data/dissemination_cells.csv`
- `Monsoon_Data/evaluation_2025/*.csv` (for Stage 2)

## Outputs

- `Monsoon_Data/Processed_Data/pipeline_input/*.pkl` — weekly-bin data (Stage 0)
- `Monsoon_Data/results/2025_model_evaluation/` — CV metrics, reliability plots, blend weights
- `Monsoon_Data/results/2025_model_evaluation/evaluation/` — 2025 out-of-sample metrics
- `figures/` — Publication-ready figures

---

## Adding a New Model Formula

1. Open the relevant spec in `specs/2025_blend/` (e.g. `cv_models_fixed_cutoff.yml`).
2. Add a new entry under `models.formulas` with a name and `text` string.
3. Use `_qx` as a shorthand for week columns — it expands to `_week1` through `_week4` automatically.
4. Ensure any new predictor columns exist in the weekly-bin pickle (add them via `rain_predictors` in the connect spec if needed).
5. Re-run Stage 1.

Example:
```yaml
models:
  formulas:
    my_new_model:
      enabled: true
      text: "outcome ~ prob_clim_qx + diff_ngcm_qx"
```

## Adding a New Rain Predictor

1. Add a `{ agg, window }` entry to `rain_predictors` under the relevant model in the connect spec:
   ```yaml
   rain_day_max: 15
   rain_predictors:
     - { agg: diff, window: 5 }
     - { agg: min,  window: 10 }
   ```
2. Add the corresponding `_qx` term to the formula in `cv_models*.yml`:
   ```yaml
   text: "outcome ~ prob_clim_qx * diff_ngcm_qx + min_ngcm_7day_qx"
   ```
3. Re-run Stage 0, then Stage 1.

The output column name is built automatically as `{agg}_{model}_{window}day_week{w}` (e.g. `min_ngcm_7day_week1`), except for `diff` which omits the window: `diff_{model}_week{w}`.

Only complete rolling windows are used. With `rain_day_max: 15` and a
five-day window, week 1 uses start days 1–7, week 2 uses 8–11, and later weeks
are unavailable (`NaN`). Configure the same explicit `rain_day_max` under
`options` in the model's raw-data spec to validate every ensemble member before
aggregation.

## Changing a Model Name

The `name` field under `forecast_models` in the connect spec is the single source of truth. It propagates automatically into all column names in the weekly-bin pickle. If you rename a model:

1. Change `name` in `specs/2025_blend/connect_*.yml`.
2. Update all formula terms referencing the old name in `cv_models*.yml` (e.g. `diff_ngcm_qx` → `diff_newname_qx`).
3. Regenerate the weekly-bin pickle by re-running Stage 0.
