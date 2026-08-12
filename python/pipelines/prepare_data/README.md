# prepare_data Pipeline

Converts raw NetCDF rainfall data (IMD ground truth, NeuralGCM and AIFS forecasts) into modeling-ready wide tables with onset probabilities, climatology forecasts, and combined features.

## Stages

### Stage 0 (optional): `0_regrid_to_shapefile.py`

Regrids gridded rainfall onto a shapefile's admin units before Stage 1 (spec in `specs/regrid/`). Builds grid→unit area weights, restricts them to cells that actually have ground-truth data (renormalized to sum to 1 per unit), and applies those **same** weights to both the ground truth and the forecasts so they share one spatial footprint. Regridding is applied to rainfall, not to probabilities. Source-to-target weights are compiled into a sparse matrix and applied in bounded value chunks. Writes `<name>_adm3.nc` beside each input; point the Stage-1 specs at those. See "Re-targeting to a New Geography" in the top-level README.

### Stage 1: `1_process_raw_nc_files.py`

Reads raw NetCDF files, detects monsoon onset per grid cell using the configured onset definition, and writes wide-format pickle tables.

- **Forecast systems** (e.g., `--spec_id ngcm`): Produces per-ensemble-member onset probabilities across lead days.
- **Ground truth** (e.g., `--spec_id ref_rain_fixed_cutoff`): Produces observed onset dates and a wide rainfall table, plus an optional annotated daily-long table.

When a spec selects multiple NetCDF files, `input.parallel: true` processes
independent files in separate processes using up to `input.workers` workers.
Results are collected in deterministic year/file order. Parallel execution is
skipped for a single file or one worker; `input.parallel: false` retains the
serial path.

The onset definition is fully configurable from the spec under `options.onset_definition`. Two dry-spell veto modes are supported: `consecutive_dry` (new definition) and `window_sum` (original Moron-Robertson definition). All numerical parameters — trigger window, wet-day threshold, accumulation threshold, follow-up period, and dry-spell check — are set in the yml. See [Onset Definition](#onset-definition) below.

### Stage 2: `2_build_climatology.py`

Fits per-cell KDE climatology models from historical IMD onset dates and produces issue-date probability forecasts over lead days. Supports multiple training windows defined in the spec. One convention to note is that for unconditional climatology (`clim_unc`) a "day 0" forecast stores the probability that onset occurred before the forecast date.

### Stage 3: `3_combine_datasets.py`

Joins ground truth, climatology, and forecast system outputs into a single wide table per combination template. Aligns time/space grids, adds "plus" remainder bins, and optionally trims post-onset forecasts.

---

## Running

All commands run from the repository root:

```bash
# Ground truth
python python/pipelines/prepare_data/1_process_raw_nc_files.py --spec_id ref_rain_fixed_cutoff
python python/pipelines/prepare_data/2_build_climatology.py --spec_id ref_rain_fixed_cutoff

# Forecast systems
python python/pipelines/prepare_data/1_process_raw_nc_files.py --spec_id ngcm
python python/pipelines/prepare_data/1_process_raw_nc_files.py --spec_id aifs

# Combine
python python/pipelines/prepare_data/3_combine_datasets.py --spec_id combine_template_fixed_cutoff_2025
```

---

## Spec Files

- **`specs/raw_data/*.yml`**: One per data source. Key fields: `type` (`ground_truth_rainfall` / `rainfall_forecast`), `input.nc_folder`, `output.out_dir`, optional `output.write_long`, `options.onset_definition`, `thresholds`, `ref_onset`, `paths.climatology_out_dir`, `climatologies`.
- **`specs/combine/*.yml`**: One per combination template. Key fields: `input.ground_truth_wide_rds`, `input.climatologies`, `forecasts`, `output.out_dir`.

---

## Inputs

- Raw NetCDF files in `Monsoon_Data/raw_nc/` (paths configured in `specs/raw_data/*.yml`):
  - `IMD_2by2/` — IMD ground truth at 2-degree resolution (files: `YYYY.nc`)
  - `ngcm/` — NeuralGCM ensemble forecast NetCDF files
  - `aifs/` — AIFS ensemble forecast NetCDF files
- Reference files in `Monsoon_Data/reference/`:
  - `thresholds_df.csv` — per-cell onset accumulation thresholds (`Unnamed: 0`, `lat`, `lon`, `onset_threshold`)
  - `MOK Onset May.csv` — observed MOK dates (`Unnamed: 0`, `Year`, `MOK` as days since May 1)
  - `dissemination_cells.csv` — grid cells used for evaluation

---

## Outputs

- `Monsoon_Data/Processed_Data/Models/*.pkl` — per-system wide onset tables
- `Monsoon_Data/Processed_Data/Climatology/*.pkl` — KDE climatology issue-date forecasts
- `Monsoon_Data/Processed_Data/Combined/*.pkl` — merged modeling-ready wide tables

---

## Onset Definition

The onset definition is controlled entirely by `options.onset_definition` in the raw data spec. No code changes are needed to switch definitions.

### Trigger (both modes)

A candidate day `d` passes the trigger if:
- All days in the window `[d, d+win-1]` are wet: `rain >= wet_day_min_mm`
- The rolling sum over those `win` days exceeds `thresh` (per-cell value from `thresholds_df.csv`)

The first qualifying candidate is returned as onset. If that candidate is vetoed by the dry-spell check, the search continues to the next trigger candidate.

### Dry-spell veto modes

**`consecutive_dry`**: no run of `>= min_dry_days` consecutive dry days within `follow_days` days after the trigger window.

**`window_sum`** (Moron-Robertson): no rolling window of `sum_window` days with total rainfall `< sum_min_mm` within `follow_days` days after the trigger window.

Setting `follow_days: 0` disables the veto entirely — only the trigger condition is checked. This is the current default across all specs.

### yml configuration

```yaml
options:
  window: 3                        # trigger window length (days)
  onset_definition:
    wet_day_min_mm: 1.0            # minimum mm/day to count as wet
    follow_days: 0                 # days after trigger to check for dry spell
                                   # set to 21 for Ethiopia/ICPAC definition
    dry_spell:
      mode: consecutive_dry        # "consecutive_dry" | "window_sum"

      # --- consecutive_dry ---
      min_dry_days: 7
      dry_day_min_mm: 1.0          # defaults to wet_day_min_mm if omitted

      # --- window_sum (Moron-Robertson) ---
      # mode: window_sum
      # sum_window: 10
      # sum_min_mm: 5.0
```

To reproduce the **original Moron-Robertson definition**:
```yaml
options:
  window: 5
  onset_definition:
    wet_day_min_mm: 1.0
    follow_days: 30
    dry_spell:
      mode: window_sum
      sum_window: 10
      sum_min_mm: 5.0
```

---

## NetCDF Input Requirements

### General Requirements (all files)

- **Format**: NetCDF-3 or NetCDF-4 (`.nc` / `.nc4`)
- **Filename convention**: Must contain a 4-digit year matching `(19|20)\d{2}` (e.g., `2000.nc`, `data_2000.nc`). The year is extracted from the filename, not the file contents.
- **Coordinate dimensions**: Must include latitude, longitude, and time. Common name variants are handled automatically; use `dimensions.rename` in the spec for any mismatches.
- **Time encoding**: Numeric time values must have a CF-compliant `units` attribute (e.g., `"days since 1900-01-01"`). Supported units: seconds, minutes, hours, days.
- **Rainfall variable**: A single variable specified by `input.value_col` in the spec (e.g., `"tp"`, `"precip"`).

### Forecast NetCDF Requirements (`type: "rainfall_forecast"`)

In addition to the general requirements:

- **Lead-day dimension**: A dimension for forecast lead days (e.g., `day`), specified via `input.wide_day_dim`. Values should be positive integers.
- **Ensemble dimension** (optional): A `number` dimension for ensemble members. The spec can filter to a maximum number of members via `filter.max_number`.
- **Day range**: The spec's `options.min_day` and `options.max_day` control which lead days are kept. The file should contain at least days in `[min_day, max_day + window - 1]`.
- **Fixed rainfall horizon** (optional): setting `options.rain_day_max: N` activates strict validation. By default, the NetCDF lead coordinate must be exactly `1..N`, and every retained issue date, ensemble member, and spatial cell must have finite rainfall throughout that range. Use the same `rain_day_max` for the model in the connect spec. Omitting it preserves legacy raw-input behavior. Set `options.rain_horizon_policy: truncate` only when later source days intentionally exist but features must stop at `N`; days `1..N` are still validated strictly.

Example for a forecast system with exactly 15 available days:

```yaml
options:
  min_day: 1
  max_day: 15
  rain_day_max: 15
```

### Ground Truth NetCDF Requirements (`type: "ground_truth_rainfall"`)

In addition to the general requirements:

- **No lead-day dimension**: The rainfall variable should be dimensioned by `(lat, lon, time)` only — one value per grid cell per calendar day.
- **Daily resolution**: Time steps must be daily.
- **Coverage**: Should cover the full monsoon season for each year (at least from `options.cutoff_month_day` onward).

### Optional: Cell Transform (Regridding)

When `options.cell_transform_enabled: true`, a CSV weights file is required with columns:
- `target_id`: Stable ID of the target grid cell or administrative unit
- `source_id`: Stable ID of the contributing source cell or unit
- `weight`: Finite, non-negative overlap weight; zero-weight rows are ignored, each source/target pair must be unique, and every target must have positive total weight

The transform is applied before target/dissemination filtering. For each target,
weights are renormalized over source cells with finite rainfall; if none are
observed, the transformed rainfall remains missing. The same sparse observed-weight
engine is used for forecast and ground-truth transforms. Thresholds must match
the target units when cell transform is enabled.

---

## Data Dictionary

### Stage 1 Outputs (`Processed_Data/Models/`)

#### Forecast wide table (`<spec_id>_wide.pkl`)

One row per (grid cell, initialization date, year). Ensemble members are aggregated.

| Column | Type | Description |
|--------|------|-------------|
| `id` | str | Grid cell ID (`"lat_lon"`, e.g., `"7.5_37.5"`) |
| `time` | date | Forecast initialization date |
| `year` | int | Year (from filename) |
| `onset_thresh` | float | Per-cell onset accumulation threshold (mm) |
| `ref_onset_date` | date | MOK date for this year (or NaT) |
| `forecast_rain_day_<k>` | float | Ensemble mean rainfall on lead day k |
| `forecast_rain_sd_day_<k>` | float | Ensemble std dev of rainfall on lead day k |
| `frac_raining_day_<k>` | float | Fraction of members with rainfall >= 1mm on day k |
| `predicted_prob_day_<k>` | float | Onset probability on day k (raw, no start restriction) |
| `predicted_prob_fixed_cutoff_day_<k>` | float | Onset probability on day k (restricted to after June 2) |
| `predicted_prob_ref_day_<k>` | float | Onset probability on day k (restricted to after MOK date) |

#### Ground truth wide table (`<spec_id>_wide.pkl`)

One row per (grid cell, year).

| Column | Type | Description |
|--------|------|-------------|
| `id` | str | Grid cell ID |
| `year` | int | Year |
| `onset_idx` | float | Index (position in daily series) of onset |
| `onset_date` | date | Calendar date of onset |
| `onset_day` | float | Days from `cutoff_month_day` to onset (e.g., days since May 1) |
| `cutoff_date` | date | Season start date for that year |

#### Ground truth long table (`<spec_id>_long.pkl`, optional)

Set `output.write_long: false` to skip constructing and writing this diagnostic
artifact. The default is `true` for backward compatibility. The wide onset
table used by climatology and blending is still written.

One row per (grid cell, day). Annotated daily rainfall series.

| Column | Type | Description |
|--------|------|-------------|
| `id` | str | Grid cell ID |
| `time` | date | Calendar date |
| `year` | int | Year |
| `<value_col>` | float | Daily rainfall (mm); column name from spec (e.g., `precip`) |
| `onset_thresh` | float | Per-cell onset threshold |
| `ref_onset_date` | date | MOK date for this year |
| `onset_date` | date | Onset date for this cell-year |
| `onset_flag` | bool | True on the onset date, False otherwise |

### Stage 2 Outputs (`Processed_Data/Climatology/`)

#### Climatology forecast table (`<out_stem>.pkl`)

One row per (grid cell, issue date, lead day).

| Column | Type | Description |
|--------|------|-------------|
| `lat`, `lon` | float | Grid cell coordinates |
| `time` | date | Issue (forecast) date |
| `day` | int | Lead day (1 to `forecast_window`) |
| `predicted_prob` | float | P(onset on lead day k \| not yet onset by issue date), or unconditional mass if `conditional: false` |
| `model` | str | Climatology model label (e.g., `"kde"`) |

### Stage 3 Outputs (`Processed_Data/Combined/`)

#### Combined wide table (`<spec_id>_combined_wide.pkl`)

One row per (grid cell, initialization date, year). All systems merged.

| Column | Type | Description |
|--------|------|-------------|
| `lat`, `lon` | float | Grid cell coordinates |
| `id` | str | Grid cell ID |
| `time` | date | Forecast initialization date |
| `year` | int | Year |
| `true_onset_day` | float | Days from season start to observed onset |
| `true_onset_date` | date | Observed onset date |
| `clim_p_onset_day_<k>` | float | Conditional climatology onset probability for day k |
| `clim_unc_p_onset_day_<k>` | float | Unconditional climatology onset probability for day k |
| `<system>_p_onset_day_<k>` | float | Forecast system onset probability for day k (e.g., `ngcm_p_onset_day_1`) |
| `<system>_rain_mean_day_<k>` | float | Forecast system mean rainfall for day k |
| `<system>_onset_thresh` | float | Per-cell onset threshold (constant per cell) |
| `<system>_ref_onset_date` | date | MOK date (constant per year) |
| `*_day_<max+1>plus` | float | Remainder bin probability (1 − sum of day bins) |

Where `<system>` is `ngcm`, `aifs`, etc. and `<k>` ranges from 1 to `max_day` (typically 28 or 40).

---

## Adding a New Data Source

1. Create a new spec in `specs/raw_data/` (copy an existing forecast spec as template).
2. Set `input.nc_folder`, `value_col`, and `dimensions.rename` for your NetCDF structure.
3. Configure `options.onset_definition` to match your desired onset definition.
4. Run Stage 1 with `--spec_id <your_spec>`.
5. Add the new source to the relevant `specs/combine/*.yml` under `forecasts`.
6. Re-run Stage 3.

## Adding a New Onset Filter Variant

1. Create a new IMD spec variant in `specs/raw_data/` (e.g., `ref_rain_new_variant.yml`) with the appropriate `ref_onset` configuration.
2. Run Stages 1–2 with the new spec.
3. Create a matching combine template in `specs/combine/`.
4. Run Stage 3 with the new combine spec.

## Testing the Onset Detection Logic

A standalone test script is provided at the repository root:

```bash
python test_onset.py
```

Runs 13 test cases covering both `consecutive_dry` and `window_sum` modes with various parameter combinations. Prints a PASS/FAIL summary and saves `test_onset_results.png` with annotated time-series plots (wet/dry periods highlighted, onset date marked with a vertical line).
