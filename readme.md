#  Blending (AI + Climatology) model for probabilistic rainy season onset forecasts

Package for probabilistic rainy season onset forecasts that blend rainfall observations with AI weather prediction model forecasts (NeuralGCM, AIFS, GenCast) through multinomial blending models. This package reproduces all results from data preparation through cross-validated evaluation to realtime operational forecast. 

> **Note:** This package is an administrative district blending version of the original lat/lon gridded [blending code](git@github.com:bosup/onset_blending.git)

---

## Adapting This Code to Your Own Data

All pipeline behaviour is controlled by YAML spec files. You can point the pipeline at your own data without modifying any Python code.

- **Different ground truth rainfall**: Create a new spec in `specs/raw_data/` modelled after `ref_rain_fixed_cutoff_2026.yml`. Point `input.nc_folder` at your NetCDF files and configure the variable name, dimension mappings, and onset thresholds for your grid.
- **Different AI forecast models**: Create a new spec in `specs/raw_data/` modelled after `ngcm.yml` (for ensembles) or `aifs.yml`. The pipeline handles ensemble forecasts in NetCDF format with configurable dimension names and variable mappings. The `name` field under `forecast_models` in the connect spec controls all downstream column names and formula terms — changing it there propagates everywhere automatically.
- **Different grid resolutions**: Not being used in the current version.
- **Different onset definitions**: Edit `options.onset_definition` in your raw data spec (see [Onset Definition](#onset-definition) below). All numerical parameters — trigger window, wet-day threshold, accumulation threshold, follow-up period, and dry-spell check — are fully configurable from the yml. No code changes needed.
- **Different blending model formulas**: Edit `models.formulas` in `specs/2025_blend/cv_models*.yml`. Formula terms use `_qx` as a shorthand that expands across the configured week bins for which each rain feature is constructible.
- **Different forecast systems in the MME**: Not being used in the current version. Edit `mme.blend_models` in `specs/2025_blend/cv_models*.yml`.
- **Different rain predictors**: Edit `rain_predictors` under each model in `specs/2025_blend/connect_*.yml`. Supports both legacy string format (`diff_5day`) and new dict format (`{ agg: diff, window: 5 }`). The dict format is preferred as it generalises to any window size without code changes.

If you add new input sources, you will need to create matching `combine` and `2025_blend` spec files so they are included in the combined dataset and blending pipeline.

---

## Repository Layout

```
onset_blending_for_laude/
├── python/
│   ├── _shared/                      Core utilities
│   │   ├── misc.py                     Null-coalescing and general helpers
│   │   └── read_spec.py                YAML spec loading and validation
│   ├── prepare_data/                 Data preparation helpers
│   │   ├── nc_utils.py                 NetCDF reading, regridding, wide-table construction,
│   │   │                               onset processing pipeline driver
│   │   ├── onset_utils.py              Onset detection (two dry-spell modes), reference onset dates,
│   │   │                               threshold loading, onset parameter parsing
│   │   ├── climatology_utils.py        KDE fitting, climatological forecasts
│   │   └── combine_forecasts_utils.py  Merging climatology + forecast families
│   └── blending_process/             Blending pipeline helpers
│       ├── connect_utils.py            Day-to-week aggregation, logit transforms,
│       │                               rain predictor computation (generic window/agg)
│       ├── blend_evaluation_utils.py   CV evaluation, multinomial logistic regression,
│       │                               Platt calibration, formula expansion
│       ├── evaluation_2025_utils.py    Out-of-sample scoring utilities
│       └── blend_figure_utils.py       Figure generation utilities
├── pipelines/
│   ├── prepare_data/
│   │   ├── 1_process_raw_nc_files.py
│   │   ├── 2_build_climatology.py
│   │   └── 3_combine_datasets.py
│   └── blending_process/
│       ├── 0_connect_prepare_data_to_2025_pipeline.py
│       ├── 1_blend_evaluation.py
│       ├── 2_2025_evaluation.py
│       └── 3_produce_figures.py
├── specs/
│   ├── raw_data/                     NetCDF input specs (aifs, ngcm, ref_rain variants)
│   ├── combine/                      Data combination templates
│   └── 2025_blend/                   Blended model specs (formulas, MME config, connect specs)
├── Monsoon_Data/                     Data directory (not tracked in git)
│   ├── raw_nc/                         Raw NetCDF inputs (IMD, NGCM, AIFS)
│   ├── reference/                      Onset thresholds, reference onset dates
│   ├── Processed_Data/
│   │   ├── Models/                       Per-system onset tables (.pkl)
│   │   ├── Climatology/                  KDE climatology forecasts (.pkl)
│   │   ├── Combined/                     Merged modeling-ready wide tables (.pkl)
│   │   └── 2025_pipeline_input/          Weekly-bin data for blending (.pkl)
│   ├── results/
│   │   └── 2025_model_evaluation/        Model metrics, blend weights, figures
│   └── evaluation_2025/                Out-of-sample forecast + ground truth files
└── test_onset.py                     Standalone onset detection test suite with plots
```

---

## Prerequisites

### Required Input Data

| Path | Description |
|------|-------------|
| `Monsoon_Data/raw_nc/IMD_2by2/` | IMD gridded rainfall NetCDF files (`data_YYYY.nc`), one per year |
| `Monsoon_Data/raw_nc/ngcm/` | NeuralGCM ensemble forecast NetCDF files, one per year |
| `Monsoon_Data/raw_nc/aifs/` | AIFS ensemble forecast NetCDF files, one per year |
| `Monsoon_Data/reference/thresholds_df.csv` | Per-unit onset accumulation thresholds (`adm3_name`, `onset_threshold`); legacy `lat`/`lon` keying also supported. Or omit the file and set `thresholds.constant` |
| `Monsoon_Data/reference/MOK Onset May.csv` | Per-year reference onset dates (`Unnamed: 0`, `Year`, `MOK`) |
| `Monsoon_Data/evaluation_2025/` | Out-of-sample forecast and ground truth CSVs (for stage 2) |

#### Reference file formats

**`thresholds_df.csv`** — one row per unit (adm3 format, current):
```
adm3_name,onset_threshold
Addi Arekay,20
...
```
(Legacy `lat,lon,onset_threshold` keying is also accepted. For a single value everywhere, skip the file and set `thresholds: { constant: 20.0 }`.)

**`MOK Onset May.csv`** — one row per year; `MOK` is days since `base_date` (May 1):
```
Unnamed: 0,Year,MOK
1,2000,14
...
```

### Python Dependencies

```bash
pip install numpy pandas scipy netCDF4 pyyaml matplotlib
```

The pipeline has been tested on Python 3.10+. No R installation is required.

---

## Running the Pipeline

All scripts must be run from the repository root. Scripts use relative paths that break from other working directories.

### Stage 1: Prepare Data

```bash
# Process raw NetCDF files into per-system onset tables
python python/pipelines/prepare_data/1_process_raw_nc_files.py --spec_id ref_rain_fixed_cutoff_2026
python python/pipelines/prepare_data/1_process_raw_nc_files.py --spec_id ngcm_2026
python python/pipelines/prepare_data/1_process_raw_nc_files.py --spec_id aifs_2026

# Build KDE climatology forecasts from the reference-rainfall onset dates
python python/pipelines/prepare_data/2_build_climatology.py --spec_id ref_rain_fixed_cutoff_2026

# Combine ground truth, model forecasts, and climatology into wide tables
python python/pipelines/prepare_data/3_combine_datasets.py --spec_id combine_template_fixed_cutoff_2026
```

**Outputs**: `Monsoon_Data/Processed_Data/Combined/combine_template_fixed_cutoff_2026_combined_wide.pkl`

### Stage 2: Blending Pipeline

```bash
# Convert daily onset probabilities to weekly bins
python python/pipelines/blending_process/0_connect_prepare_data_to_2025_pipeline.py --spec_id connect_fixed_cutoff_2026

# Cross-validated model evaluation + MME weight optimisation
python python/pipelines/blending_process/1_blend_evaluation.py

# Out-of-sample 2025 evaluation
python python/pipelines/blending_process/2_2025_evaluation.py

# Publication figures
python python/pipelines/blending_process/3_produce_figures.py
```

**Outputs**: `Monsoon_Data/results/2025_model_evaluation/`

### Fit the production model

After selecting a formula through cross-validation, fit it once on all
spec-configured training years for operational forecasting:

```bash
python predict/3_fit_final_model.py \
    --spec_id cv_models_fixed_cutoff_2026 \
    --model blended_model \
    --input_path Monsoon_Data/Processed_Data/training/cv_data_fixed_cutoff_new_pipeline.pkl \
    --out_dir Monsoon_Data/results/training \
    --tag final
```

This remains a separate step from cross-validation. It uses the same connector
input, dissemination IDs, common formula sample, training-year exclusions, and
outcome classes as the evaluator, then saves the existing coefficient-bundle
format used by `apply_blend_model.py`.

---

## Operational Forecasting (Single New Forecast Year)

For generating a forecast for a new year (e.g. 2026), use
`predict/run_operational_pipeline.py`. Forecast sources are supplied with one
repeatable `--forecast MODEL SPEC_ID` argument per model. With two forecasts,
the wrapper preserves the historical eight-step numbering and verifies each
declared output before continuing.

### What it does

| Step | Script | Description |
|------|--------|-------------|
| 1 | `2_build_climatology.py` | Build KDE climatology for the forecast year |
| 2..N+1 | `1_process_raw_nc_files.py` | Process each configured forecast NetCDF source |
| N+2 | `3_combine_datasets.py` | Merge forecasts + climatology + ground truth into a wide table |
| N+3 | `0_connect_prepare_data_to_2025_pipeline.py` | Convert daily inputs to weekly bins and rain predictors |
| N+4 | `predict/apply_blend_model.py` | Apply the saved coefficient bundle |
| N+5 | `predict/export_blend_output.py` | Extract blended + climatology probabilities to CSV |
| N+6 | `predict/run_maps.py` | Generate forecast maps from the summary CSV |

Here `N` is the number of `--forecast` arguments. Export and map generation
still follow the existing Ethiopia-oriented output contract. For a new country,
use `--stop_at N+4` to validate the generic workflow through model application;
country-generic export and mapping are a separate extension.

### Prerequisites

Before running, ensure you have:
- A raw-data spec and forecast NetCDF files for every configured model
- A final coefficient bundle produced by `predict/3_fit_final_model.py`
- Historical ground truth wide pkl (e.g. `ref_rain_fixed_cutoff_wide.pkl`) covering 2000–2022
- Matching raw, combine, connect, and blend specs in their existing spec directories

New coefficient bundles store the resolved training formula. Operational
application uses that saved formula and requires its Patsy design columns to
match the saved feature schema exactly. Older bundles without a saved formula
fall back to the current spec, but the same schema check still applies.

The training and operational wrappers also resolve forecast-probability needs
at runtime. A complete daily probability series is still aggregated for
backward-compatible outputs. If a series is shorter than the configured
probability horizon, the connector skips it only when no downstream formula,
raw/calibrated evaluation, enabled MME, or saved operational feature requires
it; otherwise it fails with the missing daily columns. This permits a
short-horizon forecast to supply constructible rainfall predictors without
claiming support for incomplete raw or Platt-calibrated probability products.
No additional YAML option is required.

### Minimal example

```bash
python predict/run_operational_pipeline.py \
    --year 2026 \
    --issue_date 2026-06-09 \
    --forecast aifs aifs_2026 \
    --forecast ngcm ngcm_2026 \
    --clim_spec ref_rain_fixed_cutoff_2026 \
    --combine_spec combine_template_fixed_cutoff_2026_ngcm \
    --connect_spec connect_fixed_cutoff_2026_ngcm \
    --blend_spec cv_models_fixed_cutoff_2026_ngcm \
    --coef_dir Monsoon_Data/results/training \
    --coef_tag final \
    --work_dir Monsoon_Data/Processed_Data/2026
```

Unless `--blend_input` is supplied, the connector output basename from the
selected connect spec is placed in `work_dir` and passed unchanged to
`apply_blend_model.py`. This preserves model-specific names such as the NGCM or
GenCast connector artifacts.

### With yml field overrides

The complete historical two-slot interface remains available for existing AIFS
operations. Its NetCDF paths and historical ground truth can be overridden
without editing YAML. `--gt_path` patches both the climatology spec
(`input.gt_path`) and combine spec (`ground_truth_wide_rds`).

```bash
python predict/run_operational_pipeline.py \
    --year 2026 \
    --issue_date 2026-06-09 \
    --model_single aifs \
    --model_ens aifs_ens \
    --aifs_spec aifs_2026 \
    --aifs_ens_spec aifs_ens_2026 \
    --clim_spec ref_rain_fixed_cutoff_2026 \
    --combine_spec combine_template_fixed_cutoff_2026 \
    --connect_spec connect_fixed_cutoff_2026 \
    --blend_spec cv_models_fixed_cutoff_2026 \
    --coef_dir Monsoon_Data/results/wet_spell_aifs_aifs_ens \
    --coef_tag fixed_cutoff_2022_year2022 \
    --work_dir Monsoon_Data/Processed_Data/2026 \
    --aifs_nc_folder /data/forecasts/aifs/2026 \
    --aifs_ens_nc_folder /data/forecasts/aifs_ens/2026 \
    --gt_path Monsoon_Data/Processed_Data/Models/wet_spell/ref_rain_fixed_cutoff_wide.pkl
```

The wrapper always writes runtime `_op.yml` copies so output paths, forecast
artifacts, and model names are coherent without mutating the selected base
specs. These temporary specs are deleted on exit.

### Resuming after a failure

Use `--skip_to N` to restart from a specific step without rerunning earlier (potentially expensive) steps:

```bash
# Re-run from step 6 onward (apply blend model through maps)
python predict/run_operational_pipeline.py \
    ... \
    --skip_to 6
```

### Dry run

Use `--dry_run` to print all commands that would be executed without running them — useful for verifying paths before committing to a full run:

```bash
python predict/run_operational_pipeline.py \
    ... \
    --dry_run
```

### All arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--year` | Yes | Forecast year, e.g. `2026` |
| `--issue_date` | Yes | Forecast issue date, e.g. `2026-06-09` |
| `--forecast MODEL SPEC_ID` | Conditional | Repeat once per forecast; use this or the complete legacy argument set |
| `--model_single`, `--model_ens` | Legacy | Model names for the historical two-slot interface |
| `--aifs_spec`, `--aifs_ens_spec` | Legacy | Raw-data spec IDs for the historical two-slot interface |
| `--clim_spec` | Yes | Spec ID for `2_build_climatology`, e.g. `ref_rain_fixed_cutoff_2026` |
| `--combine_spec` | Yes | Spec ID for `3_combine_datasets`, e.g. `combine_template_fixed_cutoff_2026` |
| `--connect_spec` | Yes | Spec ID for `0_connect_prepare_data_to_2025_pipeline` |
| `--blend_spec` | Yes | Spec ID for `apply_blend_model` |
| `--coef_dir` | Yes | Directory containing the trained blending model coef pkl |
| `--coef_tag` | Yes | Coef tag passed to `apply_blend_model --coef_tag`, e.g. `fixed_cutoff_2022_year2022` |
| `--blend_input` | No | Explicit connector-output/application-input path; otherwise derived from the connect spec basename |
| `--work_dir` | Yes | Output directory for all intermediate and final files |
| `--aifs_nc_folder` | No | Override `input.nc_folder` in the aifs spec yml |
| `--aifs_ens_nc_folder` | No | Override `input.nc_folder` in the aifs_ens spec yml |
| `--gt_path` | No | Override ground truth pkl path in both the clim and combine specs |
| `--map_output_path` | No | Reserved for compatibility; maps currently use `work_dir` |
| `--blend_model` | No | Blended model name (default: `blended_model`) |
| `--region` | No | Region for map generation (default: `Ethiopia`) |
| `--skip_to` | No | Skip to step N, 1-indexed (default: 1, run all) |
| `--stop_at` | No | Stop after step N, 1-indexed |
| `--dry_run` | No | Print commands without executing |

### Testing the Onset Detection Logic

```bash
python test_onset.py
```

Runs 13 test cases covering both dry-spell modes, prints a PASS/FAIL summary, and saves `test_onset_results.png` with annotated time-series plots (wet/dry periods highlighted, onset date marked).

---

## Data Flow

```
Raw NetCDF (Monsoon_Data/raw_nc/)
    │
    ▼
┌──────────────────────────────────┐
│  1_process_raw_nc_files.py       │  specs/raw_data/*.yml
│  → Processed_Data/Models/*.pkl   │
└──────────┬───────────────────────┘
           │
           ├──▶ 2_build_climatology.py
           │    → Processed_Data/Climatology/*.pkl
           │                 │
           ▼                 ▼
┌──────────────────────────────────┐
│  3_combine_datasets.py           │  specs/combine/*.yml
│  → Processed_Data/Combined/*.pkl │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  0_connect (day → week bins)             │  specs/2025_blend/connect_*.yml
│  → Processed_Data/2025_pipeline_input/   │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  1_blend_evaluation.py                   │  specs/2025_blend/cv_models*.yml
│  Cross-validated multinomial logistic    │
│  Platt calibration, MME optimisation     │
│  → results/2025_model_evaluation/        │
└──────────┬───────────────────────────────┘
           │
           ├──▶ 2_2025_evaluation.py
           │    Out-of-sample scoring (Brier, RPS, AUC)
           ▼
┌──────────────────────────────────────────┐
│  3_produce_figures.py                    │
│  → figures/                              │
└──────────────────────────────────────────┘
```

---

## Onset Definition

The onset definition is fully configurable from the yml `options.onset_definition` block. Two dry-spell veto modes are supported.

### Trigger (both modes)

A candidate day `d` passes the trigger if:
- `trigger_rule: all_days_wet` requires every day in `[d, d+win-1]` to be wet;
  `trigger_rule: first_day_wet` requires only day `d` to be wet.
- `wet_day_comparison` selects inclusive `gte` or strict `gt` comparison with
  `wet_day_min_mm`.
- The rolling sum over those `win` days exceeds `thresh` (per-cell value from `thresholds_df.csv`)

### Dry-spell veto modes

**`consecutive_dry`** (new definition): no run of `>= min_dry_days` consecutive dry days (`rain < dry_day_min_mm`) in the configured follow-up interval.

**`window_sum`** (original Moron-Robertson dry-spell check): no rolling window of `sum_window` days with total rainfall `< sum_min_mm` in the configured follow-up interval. `followup_anchor: after_trigger` starts the interval after the trigger window; `followup_anchor: onset_day` starts it on candidate day `d`.

The first candidate day that passes both the trigger and the veto is returned as the onset date. If the first candidate is vetoed, the search continues to the next trigger candidate (option A behaviour). Setting `follow_days: 0` disables the veto entirely.

### yml configuration

```yaml
options:
  window: 3                        # trigger window length (days)
  onset_definition:
    wet_day_min_mm: 1.0            # minimum mm/day to count as wet
    trigger_rule: all_days_wet     # "all_days_wet" | "first_day_wet"
    wet_day_comparison: gte        # "gte" | "gt"
    follow_days: 21                # days after trigger window to check
    followup_anchor: after_trigger # "after_trigger" | "onset_day"

    dry_spell:
      mode: consecutive_dry        # "consecutive_dry" | "window_sum"

      # --- consecutive_dry parameters ---
      min_dry_days: 7              # consecutive dry days needed to veto
      dry_day_min_mm: 1.0          # mm/day below this = dry day
                                   # (defaults to wet_day_min_mm if omitted)

      # --- window_sum parameters (original definition) ---
      # mode: window_sum
      # sum_window: 10             # rolling window length for dry-spell check
      # sum_min_mm: 5.0            # window sum below this = dry spell
```

To reproduce the **original Moron-Robertson definition** exactly:
```yaml
options:
  window: 5
  onset_definition:
    wet_day_min_mm: 1.0
    trigger_rule: first_day_wet
    wet_day_comparison: gt
    follow_days: 30
    followup_anchor: onset_day
    dry_spell:
      mode: window_sum
      sum_window: 10
      sum_min_mm: 5.0
```

All parameters are optional. If `onset_definition` is omitted, the backward-compatible Python defaults are `all_days_wet`, `gte`, `consecutive_dry`, `follow_days=21`, `followup_anchor=after_trigger`, and `min_dry_days=7`. Setting `follow_days: 0` disables the veto.

### Short series behaviour

If the series ends before the full follow-up window is available, the veto is checked over however many days remain. A candidate is only rejected if a dry spell is actually found in the available data. To reject any candidate whose follow-up window is incomplete, pass `reject_if_short_followup=True` in the `find_onset()` call.

---

## Onset Start-Date Variants

Three variants control the **earliest calendar date that can count as an onset**. This is *not* a filter on issue dates — forecasts may be (and are) issued before the cutoff. The onset search simply will not return a date earlier than the cutoff; it looks for the first valid onset from the cutoff onward. In the code this is the `start_day` argument to `find_onset` (producing `onset_raw`, `onset_fixed_cutoff`, `onset_ref`).

| Variant | Spec suffix | Earliest date that can count as an onset |
|---------|-------------|-------------------------------------------|
| **fixed_cutoff** | `_fixed_cutoff` | A fixed climatological cutoff date each year (default June 2, via `options.fixed_cutoff_month_day`) |
| **ref_onset** | _(default)_ | The observed per-year reference onset date |
| **no_ref_filter** | `_no_ref_filter` | No restriction — onsets count from the season start (`options.cutoff_month_day`) onward |

Each variant has its own connect and CV spec in `specs/2025_blend/`.

**Configuring the `ref_onset` date** (the `ref_onset:` block in the raw-data spec) has three forms:
- **A specific date**, same every year and unit: `ref_onset: { constant_month_day: "06-01" }`.
- **A per-year file** (one date per year, all units): `file` + `year_col` + `day_col` + `base_date` (the reference onset date is `base_date` + `day_col` days).
- **A per-unit file** (unit-specific dates): `file` + `unit_col` + `date_col` (a parseable date), or `unit_col` + `year_col`/`day_col`/`base_date` for per-unit-per-year.

## Domain / spatial filtering

The modelling domain is set in the raw-data spec's `filter:` block, with two composable restrictions:
- **`dissemination_cells_file`** — a CSV keyed by configured `filter.id_col`, `id`, legacy `adm3_name`, or separate `lat`/`lon`; keeps only those units (the base domain).
- **`bbox`** — an optional *further* restriction `{lat_min, lat_max, lon_min, lon_max}` (any subset of keys), applied on top. Defaults to no bbox restriction (all dissemination cells). Unit lat/lon is taken from `lat`/`lon` columns, an id of the form `<lat>_<lon>` (grid-cell units), or a `filter.centroids_file` (`target_id` + lat/lon; legacy `adm3_name` is accepted). For admin (shapefile) units, the regrid step (0) writes a suitable centroids CSV automatically — set `filter.centroids_file` to that file.

---

## Rain Predictors

Rain-based predictors are computed in `connect_utils.py` from the `rain_predictors` list under each model in `specs/2025_blend/connect_*.yml`. Two formats are supported:

**Legacy string format** (still supported for backward compatibility):
```yaml
rain_predictors: [diff_5day, min_10day, max_5day]
```

**Dict format** (preferred — generalises to any window size without code changes):
```yaml
rain_predictors:
  - { agg: diff, window: 3 }
  - { agg: min,  window: 7 }   # optional, commented out by default
  - { agg: max,  window: 3 }   # optional, commented out by default
```

**Symbolic windows tied to the onset definition.** Instead of a hardcoded integer, `window` may be a token that resolves from the onset definition, so predictors track it automatically when the definition changes. Set `onset_spec: <raw_data spec id>` at the top of the connect spec and use:

| Token | Resolves to |
|-------|-------------|
| `trigger` / `window` | onset trigger window (`options.window`) |
| `dry_spell` / `sum_window` | dry-spell window (`onset_definition.dry_spell.sum_window`) |
| `follow` | `follow_days` |
| `min_dry` | `min_dry_days` |

```yaml
onset_spec: ref_rain_fixed_cutoff_2026
forecast_models:
  - name: aifs
    rain_predictors:
      - { agg: diff, window: trigger }     # follows options.window
      - { agg: min,  window: dry_spell }   # follows sum_window
```

Explicit integers always take precedence. Output column names use the *resolved* integer (e.g. `min_aifs_10day_week1`), so formula terms and the `_qx` contract are unchanged. A symbolic token with no `onset_spec` (and no inline onset definition) raises a clear error.

The current default configuration uses only `diff` with a 3-day window, matching the 3-day trigger window:

```yaml
forecast_models:
  - name: ngcm
    variants: [fixed_cutoff]
    rain_predictors:
      - { agg: diff, window: 3 }
  - name: aifs
    variants: [fixed_cutoff]
    rain_predictors:
      - { agg: diff, window: 3 }
```

Three aggregation types are available:

| `agg` | Output column name | Description |
|-------|--------------------|-------------|
| `diff` | `diff_{model}_week{w}` | Max rolling sum over week minus per-cell onset threshold |
| `max`  | `max_{model}_{N}day_week{w}` | Max rolling N-day sum over week |
| `min`  | `min_{model}_{N}day_week{w}` | Min rolling N-day sum over week |

Output column names (e.g. `diff_ngcm_week1`, `min_ngcm_7day_week1`) are constructed at runtime from the model name and window size. In a formula, `_qx` expands across all configured bins and drops only terms whose rain features are marked structurally unavailable by the connector's `rain_day_max` metadata. Interactions containing an unavailable rain feature are dropped while their constructible main effects are retained. An explicitly written `_weekN` term remains required and raises an error if it is unavailable.

### Rain transform (feature-only)

An optional transform can be applied to the rain features **before they enter the blend**. It does **not** affect onset detection or climatology — those are computed upstream from untransformed rainfall. For the `diff` aggregation the transform is applied to both sides of the subtraction, i.e. `f(weekly_sum) - f(onset_threshold)`; for `max`/`min` it is applied to the aggregated value.

Set a spec-level default and/or a per-model override (`rain_transform` under a `forecast_models` entry wins over the spec-level key):

```yaml
rain_transform: fourth_root        # default for all models

forecast_models:
  - name: aifs
    rain_transform: { power: 0.25 } # per-model override; any exponent
    rain_predictors:
      - { agg: diff, window: 3 }
```

| Value | Transform |
|-------|-----------|
| `identity` / omitted | `f(x) = x` |
| `sqrt` | `f(x) = x^(1/2)` |
| `fourth_root` | `f(x) = x^(1/4)` |
| `log1p` | `f(x) = log(1 + x)` |
| `power:<p>` or `{ power: <p> }` or a bare number | `f(x) = x^p` |

Power/log transforms clip inputs at 0 (rainfall sums and thresholds are non-negative); NaNs (e.g. short-week sentinels) propagate unchanged.

To add or remove a predictor: edit `rain_predictors` in the connect spec **and** add or remove the corresponding `_qx` term in the formula in `cv_models*.yml`. The two must stay in sync — a predictor present in the connect spec but absent from the formula is computed but silently unused; a formula term without a matching column will cause a runtime error. Formula-resolution diagnostics record the original, expanded, and resolved formula plus any dropped horizon-limited terms.

---

## Portable Country Template Bundle

The following six specifications form one copyable, internally consistent
example. They use relative placeholder paths, one forecast key
(`forecast_model`), and matching artifact names across every stage:

- `specs/regrid/country_template.yml` — optional grid-to-grid or
  grid-to-administrative-unit alignment;
- `specs/raw_data/ground_truth_template.yml` — observed rainfall, onset truth,
  and paired conditional/unconditional climatologies;
- `specs/raw_data/forecast_template.yml` — one rainfall forecast family;
- `specs/combine/combine_country_template.yml` — truth/climatology/forecast
  join;
- `specs/2025_blend/connect_country_template.yml` — weekly probabilities and
  rainfall predictors;
- `specs/2025_blend/cv_models_country_template.yml` — CV, metrics, and final-fit
  contract.

Copy and rename the bundle rather than editing these examples in place. Replace
the country paths, NetCDF variable/dimension names, stable unit IDs, years, and
scientific onset settings. Keep the chosen model key and artifact handoffs
identical across the renamed combine, connect, and CV specs. No run manifest is
required and the existing specification directories/loaders are unchanged.

The active template settings reproduce the Indian/Moron-Robertson-style onset
rule (`first_day_wet`, strict `gt`, `window_sum`, and `onset_day` anchoring) as
an example, not as a universal country default. The raw templates also show the
alternative consecutive-dry rule and fixed versus file-backed thresholds. The
connector demonstrates fourth-root rainfall features, symbolic trigger/dry
windows, and `truncate` horizon resolution. Read the comments and make each
choice explicitly for the target country.

### Artifact handoff

With the filenames above and no wrapper overrides, the data path is:

```text
ground_truth_template_wide.pkl
  + ground_truth_template_clim_issue.pkl
  + ground_truth_template_clim_unc_issue.pkl
  + forecast_template_wide.pkl
    -> combine_country_template_combined_wide.pkl
    -> cv_data_fixed_cutoff_new_pipeline.pkl
    -> CV results / final coefficient bundle
```

The regrid step is optional. When used, it writes `*_adm3.nc` files next to the
raw files; the raw templates are already configured to read those names. When
the inputs are already on the target support, skip regridding and change the raw
spec folders/regexes accordingly. As an alternative that avoids writing
regridded NetCDF files, both raw templates show the direct sparse
`cell_transform_file` path (`source_id`, `target_id`, `weight`).

The connector and CV templates also show the boundary for a genuinely
rainfall-only short-horizon system: retain its constructible rainfall predictors
in the connector/formula, but omit that model from `extras.forecasts` and any
enabled MME because those consumers require its daily onset probabilities.

### One-click historical training and CV

After copying the specs and preparing the referenced support files:

```bash
python python/run_training.py \
    --forecast forecast_model forecast_template \
    --clim_spec ground_truth_template \
    --combine_spec combine_country_template \
    --connect_spec connect_country_template \
    --blend_spec cv_models_country_template \
    --work_dir Monsoon_Data/Processed_Data/example_country \
    --results_dir Monsoon_Data/results/example_country \
    --cores 4
```

`run_training.py` rewrites only runtime copies of the selected YAML files and
passes the exact connector output to the evaluator. Use `--dry_run` first to
inspect the resolved commands and handoffs.

### Final fit and operational application

Fit the selected formula on all configured training years except
`true_holdout_years`:

```bash
python predict/3_fit_final_model.py \
    --spec_id cv_models_country_template \
    --model blended_model \
    --input_path Monsoon_Data/Processed_Data/example_country/cv_data_fixed_cutoff_new_pipeline.pkl \
    --out_dir Monsoon_Data/results/example_country \
    --tag final
```

For a forecast year, copy the ground-truth/climatology, forecast, and combine
templates to year-specific spec IDs. Set both climatology `test_year_*` fields
and the combine source `years` to that forecast year, while retaining the same
model key and saved formula contract. For one forecast (`N=1`), the generic
operational path through model application is:

```bash
python predict/run_operational_pipeline.py \
    --year 2026 \
    --issue_date 2026-06-09 \
    --forecast forecast_model forecast_country_2026 \
    --clim_spec ground_truth_country_2026 \
    --combine_spec combine_country_2026 \
    --connect_spec connect_country_template \
    --blend_spec cv_models_country_template \
    --coef_dir Monsoon_Data/results/example_country \
    --coef_tag final \
    --work_dir Monsoon_Data/Processed_Data/example_country/2026 \
    --stop_at 5
```

The core generic workflow currently ends at saved-model application (`N+4`).
CSV/NetCDF export and maps after that point still contain Ethiopia-oriented
support-file and `adm3_name` assumptions. Generalizing those presentation
outputs is separate from configuring and running the blending algorithm.

### Migration note (changes since `a146b0d`)

- No mandatory manifest or specification-directory reorganization was added.
- The fixed AIFS command-line arguments remain as legacy-compatible aliases;
  repeatable `--forecast MODEL SPEC_ID` is the geography/model-neutral path.
- Previously hard-coded onset, forecast-name, horizon, rainfall-transform, and
  spatial-ID behavior is now explicit in the existing YAML contracts.
- Effective formulas and coefficient feature/class schemas are saved and
  checked when the fitted model is applied.
- Obsolete implicit assumptions remain accepted only where needed for legacy
  compatibility; new country specs should use the explicit fields shown here.
- Country-generic export and map support remains future work.

---

## Re-targeting to a New Geography

The core pipeline keys on an abstract `id` and never touches lat/lon, so moving to a new region is mostly a matter of supplying a boundary shapefile and gridded inputs.

### Optional pre-step: regrid a whole gridded dataset onto a shapefile

`python/pipelines/prepare_data/0_regrid_to_shapefile.py --spec_id <id>` (spec in `specs/regrid/`) regrids gridded rainfall onto a shapefile's admin units **before** the normal pipeline, and handles partial ground-truth coverage correctly:

1. builds area weights (grid cell → target unit);
2. finds the grid cells that actually have ground-truth data (a rain-gauge grid has no data over the ocean);
3. drops the no-data cells and renormalizes each unit's weights to **sum to 1** (a conservative area-weighted average over only the cells with data);
4. applies that footprint to **both the ground truth and every forecast family**, so forecast and ground truth share one spatial footprint.

**Target units** are set by the spec: with a `geometry.shapefile` the target is the shapefile's admin (political) units; **without `geometry.shapefile`** the target is the **ground-truth grid cells** themselves (id `"<lat>_<lon>"`), i.e. forecasts are regridded to match the ground-truth grid. **Different forecast grids:** if a forecast's grid matches the ground truth it reuses the same weights; if it differs, it is regridded onto "the unit **minus** the parts where ground-truth data didn't exist" (unit ∩ coverage).

For shapefile targets, set `geometry.region_id_col` to the stable join key (for example `adm3_pcode` or `OBJECTID`) and optionally set `geometry.region_name_col` to a display label. The older `geometry.region_key_col` remains accepted as an alias for `region_id_col`. For grid-cell targets, an existing weight-table ID convention is authoritative. Otherwise set `geometry.grid_id_decimal_digits` and optionally `geometry.grid_id_format: fixed|trimmed`; if neither is supplied, the pipeline infers the least precise collision-free convention from the coordinates. The same formatter is used for raw data, lat/lon threshold tables, and dissemination-cell inputs.

**`clip_to_coverage`** (default **true**) toggles steps 2–4; set it false for a plain per-dataset regrid (the older woreda behavior, where `remap_nc` renormalizes over non-NaN cells per timestep with no shared footprint). This step supersedes the standalone `utils/remap.py` / `remap_weights*.py` weight-builders.

**Supplying your own weights.** Instead of computing weights, you can point the spec at precomputed weight CSVs (columns `latitude, longitude, target_id, weight`; legacy `adm3_name` is also accepted): a top-level `weights_in:` applies to everything, or a per-target `weights_in:` (under `ground_truth` or a `forecasts[]` entry) overrides it. When all weights are supplied, no shapefile overlay / coverage scan / report runs — the step just applies them. (The low-level `utils/remap.py apply --weights <csv>` does the same for a single file.)

Regridding is applied to **rainfall** (raw NetCDF variables), never to onset probabilities — those are computed downstream. Each `<name>.nc` becomes `<name>_adm3.nc` alongside it; point the step-1 specs at those (`file_regex: '..._adm3\.nc$'`). A **coverage report** CSV lists, for all units and for the dissemination subset, how many units are affected by missing ground truth and the min/5/25/50/75/95/max quantiles of the missing-area fraction. When a shapefile is used, a **unit centroids** CSV (`target_id, lat, lon`) is also written (`centroids_out`), ready to use as `filter.centroids_file` for the bbox domain filter. Existing centroid files keyed by `adm3_name` remain readable.

### Regridding a grid onto admin units (low-level)

`utils/remap.py` builds and applies the grid→admin-unit weight table. Resolution is auto-detected from the NetCDF coordinates, and the admin key column / CRS are configurable (defaults reproduce the Ethiopia `adm3_name` / EPSG:4326 setup). It supersedes the older `utils/remap_weights*.py` scripts.

```bash
# 1. Build area-fraction weights from a boundary shapefile + a sample grid
python utils/remap.py weights \
    --shapefile data/shapefile/admin.shp \
    --sample-nc Monsoon_Data/raw_nc/aifs \
    --out Monsoon_Data/grid_to_district_mapping.csv \
    [--region-id adm3_pcode] [--region-name adm3_name] \
    [--parent-key adm2_name] [--crs EPSG:4326]

# 2. Aggregate gridded .nc files to *_adm3.nc using those weights
python utils/remap.py apply \
    --weights Monsoon_Data/grid_to_district_mapping.csv \
    --input-dir Monsoon_Data/raw_nc/aifs
```

A shapefile whose stable key is not literally `adm3_name` is handled via `--region-id`; `--region-name` can retain a separate display label. `--region-key` remains a legacy alias for `--region-id`. The stable ID is carried through the existing internal `adm3_name` coordinate for backward compatibility. The reusable helpers live in `python/prepare_data/geometry_utils.py`.

### Output maps

`predict/run_maps.py` accepts `--shapefile` (adm3, must contain `adm3_name`), `--adm2_shapefile` (optional zone overlays), and `--region`, so a new geography's boundaries can be swapped in without editing the map modules.

### Computed onset thresholds

Onset threshold *y* is either a single constant or provided per unit — it is **never** auto-computed inside the pipeline:
- **Constant** (e.g. Ethiopia): `thresholds: { constant: 20.0 }` — one value for every unit, no file.
- **Per-unit file**: `thresholds.file` pointing at a CSV (keyed by `adm3_name`), NetCDF, or `.mat`, optionally transformed by `rule: { type: scale, factor: …, offset: …, min: … }`.

If you *want* a rule-derived per-unit threshold (e.g. the q-quantile of the seasonal `window`-day accumulation), run the standalone, opt-in `utils/compute_thresholds.py` yourself and point `thresholds.file` at its output CSV. This is deliberately separate from the pipeline so thresholds are always explicit.

## Spec-Driven Design

All pipeline behaviour is controlled by YAML specs. Spec files define input paths, variable selection, modelling options, and output configuration. Output basenames are derived from the `spec_id` (the yml filename), not from a field inside the YAML. Pipeline scripts are thin orchestration layers: parse args → load spec → call helpers → write artifacts.

### Spec directories

| Directory | Purpose | Used by |
|-----------|---------|---------|
| `specs/raw_data/` | NetCDF input config (paths, variables, thresholds, reference onset, onset definition) | `1_process_raw_nc_files.py`, `2_build_climatology.py` |
| `specs/combine/` | Which processed datasets to merge into wide tables | `3_combine_datasets.py` |
| `specs/2025_blend/connect_*.yml` | Day-to-week conversion, forecast models, rain predictors | `0_connect_prepare_data_to_2025_pipeline.py` |
| `specs/2025_blend/cv_models*.yml` | Model formulas, MME config, forecast calibration | `1_blend_evaluation.py` |

### Key spec sections (`cv_models*.yml`)

- **`run.training_years` / `run.cv_holdout_years`**: Years used for cross-validated training and evaluation (currently 2019–2022).
- **`models.formulas`**: Named multinomial logistic regression formulas. Terms with `_qx` are expanded to the configured week bins and resolved against connector rainfall-horizon metadata before fitting. Explicit `_weekN` terms are always strict. Current active formulas:
  - `ngcm_blend`: climatology × ngcm diff
  - `int_all`: climatology × ngcm diff × aifs diff (interaction)
  - `add_blend`: climatology + ngcm diff + aifs diff (additive)
  - `blended_model`: climatology × ngcm diff × aifs diff (currently same as `int_all`; `min` predictors commented out)
- **`models.window_variants`**: Training-window climatology variants — currently disabled (`enabled: false`).
- **`mme`**: Multi-model ensemble weight optimisation — currently disabled (`enabled: false`). When enabled, `blend_models` lists which calibrated models enter the MME.
- **`extras.forecasts`**: Per-system Platt calibration and raw/calibrated scoring options.
- **`extras.clim_logits`**: Climatology baseline configurations.

### Changing a model name

The model `name` field in `connect_*.yml` is the single source of truth for column naming. It propagates automatically into all output column names (`{name}_p_onset_week1`, `diff_{name}_week1`, etc.) and the wide pickle. If you rename a model, update:
1. `name` in `specs/2025_blend/connect_*.yml`
2. Formula terms in `specs/2025_blend/cv_models*.yml` (e.g. `diff_ngcm_qx` → `diff_newname_qx`)
3. Regenerate all downstream pickle files from stage 1 onward

---

## Key Python Functions

### `onset_utils.py`

| Function | Description |
|----------|-------------|
| `read_onset_params(spec)` | Parses `options.onset_definition` from spec; returns an `OnsetParams` namedtuple with all onset definition parameters and defaults |
| `find_onset(series, thresh, params)` | Finds first valid onset day in a rainfall series under the configured definition |
| `find_onset_precomp(series, win, thresh, ..., params)` | Batch-optimised version used by `calc_onsets_rowwise`; legacy positional args retained for call-site compatibility |
| `read_ref_onset_dates(spec)` | Reads the reference onset date source: a constant month-day, a per-year file, or a per-unit file (keyed by `year` and/or `id`) |
| `read_thresholds(spec)` | Reads per-cell onset thresholds from CSV, NetCDF, or `.mat` |
| `roll_sum_na_rm_left(x, k)` | Left-aligned k-day rolling sum, NA treated as 0 |
| `roll_sum_na_propagate_left(x, k)` | Left-aligned k-day rolling sum, NA propagates |

### `nc_utils.py`

| Function | Description |
|----------|-------------|
| `run_single_pipeline(spec_id)` | Main driver: loads spec, processes all NetCDF years, writes output pickles |
| `calc_onsets_rowwise(df, day_cols, day_ints, win, params)` | Computes onset indices (raw, fixed_cutoff, ref_onset) for every row; passes `params` to `find_onset_precomp` |
| `process_rainfall_forecast_id(df, spec, ...)` | Forecast pipeline: reads onset params from spec, attaches thresholds and reference onset dates, computes ensemble onset probabilities |
| `process_ground_truth_rainfall_id(df, spec, ...)` | Ground truth pipeline: computes true onset dates per cell-year |
| `attach_thresholds_id(df, thr_df)` | Left-joins per-cell `onset_thresh` onto the main DataFrame by `id` |

### `connect_utils.py`

| Function | Description |
|----------|-------------|
| `make_cv_rds_from_daylevel(spec)` | Main converter: reads daily combined pickle, builds weekly bins, onset outcomes, climatology logits, rain predictors, writes wide pickle for blending |
| `roll_sums_mat(mat, k)` | Rolling k-day row sums over a rainfall matrix |
| `week_max_over_starts(roll_mat, week_start_days)` | Per-row max of rolling sums at specified week-start positions |
| `week_min_over_starts(roll_mat, week_start_days)` | Per-row min of rolling sums at specified week-start positions |

---

## Conventions

- Spatial key: `id = f"{lat}_{lon}"`
- Time columns: `time` (date), `year` (int)
- Outcome categories: `week1`..`weekN` plus `later`, where **N = `n_bins`** and each bin spans **`days_per_bin`** days (both set in the connect spec; default 4×7 = a 28-day horizon). **`n_bins` counts only the interval bins that carry a threshold date — it does NOT include the `earlier` or `later` bins.** So `n_bins: 4` means the four bins `week1`–`week4`, and the full outcome set is `earlier` (before the first bin, used by the unconditional climatology), `week1`…`week4`, and `later` (after the last bin) — i.e. `n_bins + 2` categories in total, or `n_bins + 1` for the multinomial outcome (which omits `earlier`). The labels keep the `week` prefix for data-contract stability even when a bin is not 7 days wide.
  The bin structure is honored end-to-end through the data and model pipeline: onset outcome binning, bin-probability aggregation, climatology logits, rain predictors, formula `_qx` expansion, CV, Platt calibration, metrics, RPS, and the operational export. The plotting layer now renders **all** bins present in the data.
- All intermediate data stored as pandas DataFrames serialised to `.pkl` (replacing `.rds` from the R version)
- Forecast probabilities stored in wide format with system-specific prefixes
- All scripts must be run from the repository root
