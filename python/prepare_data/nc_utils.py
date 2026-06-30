# ==============================================================================
# File: nc_utils.py
# ==============================================================================
# Purpose
#   Shared utilities for the single-pass NetCDF -> onset-output pipeline.
#   Supports both "rainfall_forecast" and "ground_truth_rainfall" pipeline modes.
#
# Key conventions
#   - Spatial key `id` = adm3_name string (replaces former lat_lon id).
#   - The NetCDF spatial dimension is named "adm3_name" (string coordinate).
#   - Years are extracted from filenames, not NetCDF contents.
#   - Rainfall variable name is specified by spec["input"]["value_col"].
# ==============================================================================

import os
import re
import pickle
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, date, timedelta
from itertools import product

from ..pipelines._shared.misc import coalesce
from ..pipelines._shared.read_spec import load_spec, validate_spec
from .onset_utils import (
    read_mok_dates, read_thresholds,
    read_onset_params,
    roll_sum_na_rm_left, roll_sum_na_propagate_left,
    precompute_onset, find_onset_from_precomputed,
    find_onset_precomp,
)


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------

def validate_spec_single(spec):
    """Validate the YAML spec for required sections and fields."""

    def _type_check(s):
        if s["type"] == "rainfall_forecast":
            for nm in ("min_day", "max_day", "window"):
                if s.get("options", {}).get(nm) is None:
                    raise ValueError(f"Missing options.{nm}")
        if s["type"] == "ground_truth_rainfall":
            if s.get("options", {}).get("window") is None:
                raise ValueError("Missing options.window")
            if s.get("options", {}).get("cutoff_month_day") is None:
                raise ValueError("Missing options.cutoff_month_day")
        if not os.path.isdir(s["input"]["nc_folder"]):
            raise ValueError(f"input.nc_folder does not exist: {s['input']['nc_folder']}")

    return validate_spec(
        spec,
        required_top=["input", "dimensions", "output", "options", "type"],
        required_paths=["input.nc_folder", "input.file_regex", "input.value_col", "output.out_dir"],
        checks=[_type_check],
    )


def get_value_var(spec):
    v = spec.get("input", {}).get("value_col")
    if v:
        return str(v)
    raise ValueError("Missing spec['input']['value_col'] in YAML.")


def rename_dimensions(df, rename_map):
    """Case-insensitive rename of column names according to rename_map (old->new)."""
    if not rename_map:
        return df
    rename = {}
    for old, new in rename_map.items():
        for col in df.columns:
            if col.lower() == old.lower():
                rename[col] = new
                break
    return df.rename(columns=rename)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def list_nc_files_with_year(spec):
    """
    List NetCDF files in input.nc_folder matching file_regex, extract year.

    Returns DataFrame: nc_path (str), year (int).
    """
    folder = spec["input"]["nc_folder"]
    regex = spec["input"]["file_regex"]
#    print(repr(regex))
#    print("folder exists:", os.path.exists(folder))
#    print("folder:", repr(folder))
#    print("RAW regex:", regex)
#    print("repr(regex):", repr(regex))
#	
#    all_files = os.listdir(folder)
#    print(all_files[:5])
#	
#    for f in all_files:
#	    matched = re.search(regex, f)
#	    print(repr(f), bool(matched))

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if re.search(regex, f)
    ]
    if not files:
        raise ValueError(f"No files matched file_regex '{regex}' in {folder}")

    rows = []
    for f in files:
        m = re.search(r"(19|20)\d{2}", os.path.basename(f))
        if not m:
            raise ValueError(f"Could not extract year from filename: {f}")
        rows.append({"nc_path": f, "year": int(m.group())})

    df = pd.DataFrame(rows)
    min_year = spec.get("options", {}).get("min_year")
    max_year = spec.get("options", {}).get("max_year")
    if min_year is not None:
        df = df[df["year"] >= int(min_year)]
    if max_year is not None:
        df = df[df["year"] <= int(max_year)]
    if df.empty:
        raise ValueError("After year filtering, no files remain.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# NetCDF coordinate + time utilities
# ---------------------------------------------------------------------------

def get_nc_adm3_names(ds):
    """
    Read the adm3_name string coordinate from a netCDF4 Dataset.
    Tries common capitalisation variants. Returns a list of strings.
    """
    import netCDF4 as nc4
    variants = ["adm3_name", "ADM3_NAME", "Adm3_Name", "adm3"]
    for v in variants:
        if v in ds.variables:
            var = ds.variables[v]
            # netCDF4 string variables (dtype=object / vlen str) must be
            # accessed via [:].tolist() — iterating the masked array directly
            # raises "memoryview: format O not supported".
            try:
                items = var[:].tolist()
            except (NotImplementedError, TypeError):
                # chararray fallback (fixed-length char dim)
                import netCDF4 as nc4
                items = nc4.chartostring(var[:]).tolist()
            
            names = []
            for item in items:
                if isinstance(item, (bytes, bytearray)):
                    names.append(item.decode("utf-8").strip())
                else:
                    names.append(str(item).strip())
            return names
    raise ValueError(
        "Could not find adm3_name variable/dimension in NetCDF file. "
        f"Available variables: {list(ds.variables.keys())}"
    )


def get_nc_time(ds):
    """
    Robust time getter from a netCDF4 Dataset.

    Returns dict: values, units, name, source
    """
    variants = ["TIME", "time", "Time", "t", "T"]
    for v in variants:
        if v in ds.variables:
            vals = np.array(ds[v][:]).flatten()
            units = getattr(ds[v], "units", None)
            return {"values": vals, "units": units, "name": v, "source": "var"}
    for v in variants:
        vl = v.lower()
        for name in ds.variables:
            if name.lower() == vl:
                vals = np.array(ds[name][:]).flatten()
                units = getattr(ds[name], "units", None)
                return {"values": vals, "units": units, "name": name, "source": "var_ci"}
    for v in variants:
        if v in ds.dimensions:
            if v in ds.variables:
                vals = np.array(ds[v][:]).flatten()
                units = getattr(ds[v], "units", None)
            else:
                vals = np.arange(ds.dimensions[v].size, dtype=float)
                units = None
            return {"values": vals, "units": units, "name": v, "source": "dim"}
    raise ValueError("Could not find time variable/dimension in NetCDF file")


def nc_time_to_dates(time_num, time_units):
    """
    Convert numeric NetCDF time values to pandas DatetimeSeries using CF units.
    """
    m = re.match(
        r"^\s*(seconds?|minutes?|hours?|days?)\s+since\s+(.+)\s*$",
        str(time_units),
        re.IGNORECASE,
    )
    if not m:
        raise ValueError(f"Unrecognized NetCDF time units: {time_units}")
    unit = m.group(1).lower().rstrip("s")  # second/minute/hour/day
    origin_str = m.group(2).strip().replace("T", " ")

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            origin = datetime.strptime(origin_str, fmt)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"Could not parse NetCDF time origin: {origin_str}")

    mult = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}[unit]
    timestamps = [origin + timedelta(seconds=float(t) * mult) for t in time_num]
    return pd.DatetimeIndex(timestamps)


# ---------------------------------------------------------------------------
# ID helpers  (adm3_name-based, replacing lat/lon)
# ---------------------------------------------------------------------------

def ensure_id_col(df, id_col="id"):
    """
    Ensure the DataFrame has an `id` column.
    For adm3-based data the column is already named `adm3_name`; this renames
    it to `id` (or leaves it if `id` already exists).
    """
    df = df.copy()
    if id_col in df.columns:
        return df
    if "adm3_name" in df.columns:
        df[id_col] = df["adm3_name"].astype(str)
    else:
        raise ValueError(
            "Cannot create id column: neither 'id' nor 'adm3_name' found in DataFrame."
        )
    return df


# Keep the old name as an alias so callers that import it directly still work.
def add_id_from_latlon(df, lat_col="lat", lon_col="lon", id_col="id"):
    """
    Backward-compat wrapper.  With adm3 data the lat/lon columns are gone;
    the id is already the adm3_name.  Delegates to ensure_id_col().
    """
    return ensure_id_col(df, id_col=id_col)


def prep_thresholds_id(thr_df):
    """
    Normalize thresholds DataFrame to be keyed by id (adm3_name).

    Accepts:
      - DataFrame with 'id' column  (new adm3 format)
      - DataFrame with 'adm3_name' column  (also new format)
      - Scalar float / int  (single global threshold)
    """
    if thr_df is None:
        return None
    # Scalar threshold — return as-is; callers handle it
    if isinstance(thr_df, (int, float, np.floating, np.integer)):
        return thr_df
    thr_df = thr_df.copy()
    thr_df.columns = thr_df.columns.str.lower()
    if "id" not in thr_df.columns:
        if "adm3_name" in thr_df.columns:
            thr_df["id"] = thr_df["adm3_name"].astype(str)
        else:
            raise ValueError(
                "Thresholds table must have either 'id' or 'adm3_name' column."
            )
    thr_df["onset_thresh"] = thr_df["onset_thresh"].astype(float)
    return thr_df[["id", "onset_thresh"]].drop_duplicates().set_index("id")


def attach_thresholds_id(df, thr_df):
    """Left-join onset_thresh from thr_df by id."""
    df = ensure_id_col(df)
    if thr_df is None:
        df["onset_thresh"] = np.nan
        return df
    # Scalar threshold
    if isinstance(thr_df, (int, float, np.floating, np.integer)):
        df["onset_thresh"] = float(thr_df)
        return df
    thr_idx = prep_thresholds_id(thr_df)
    if isinstance(thr_idx, (int, float)):
        df["onset_thresh"] = float(thr_idx)
        return df
    df = df.merge(thr_idx.reset_index(), on="id", how="left")
    return df


# ---------------------------------------------------------------------------
# Cell filtering by dissemination cells list
# ---------------------------------------------------------------------------

def filter_by_dissemination_cells(df, spec):
    """
    If spec['filter']['dissemination_cells_file'] is set, keep only rows whose
    id appears in that CSV's adm3_name column.
    Returns the (possibly filtered) DataFrame unchanged if no file is given.
    """
    filt = spec.get("filter") or {}
    dc_file = filt.get("dissemination_cells_file")
    if not dc_file:
        return df
    if not os.path.exists(dc_file):
        raise FileNotFoundError(f"dissemination_cells_file not found: {dc_file}")
    dc = pd.read_csv(dc_file)
    if "adm3_name" not in dc.columns:
        raise ValueError(
            f"dissemination_cells_file must contain an 'adm3_name' column. "
            f"Found: {dc.columns.tolist()}"
        )
    valid_ids = set(dc["adm3_name"].astype(str).str.strip())
    df = ensure_id_col(df)
    before = len(df)
    df = df[df["id"].isin(valid_ids)]
    print(f"  dissemination_cells filter: {before} → {len(df)} rows "
          f"({before - len(df)} removed)")
    return df


# ---------------------------------------------------------------------------
# Cell transform (optional regridding)  — kept for completeness but
# target_id / source_id are now adm3_name strings, not lat_lon strings.
# ---------------------------------------------------------------------------

def read_cell_transform(spec):
    """Read weights file for linear cell regridding. Returns DataFrame or None."""
    if not spec.get("options", {}).get("cell_transform_enabled", False):
        return None
    f = spec["options"].get("cell_transform_file")
    if not f:
        raise ValueError("cell_transform_enabled=True but options.cell_transform_file is empty.")
    if not os.path.exists(f):
        raise FileNotFoundError(f"cell_transform_file not found: {f}")
    w = pd.read_csv(f)
    for col in ("target_id", "source_id", "weight"):
        if col not in w.columns:
            raise ValueError(f"Transform file must have: target_id, source_id, weight")
    w["weight"] = w["weight"].astype(float)
    return w


def transform_forecast_wide(df, weights_df):
    """Apply cell transform to forecast wide-by-day DataFrame."""
    if weights_df is None:
        return df
    df = ensure_id_col(df)
    day_cols = [c for c in df.columns if re.search(r"_day_\d+$", c)]
    if not day_cols:
        raise ValueError("No forecast day columns found to transform.")

    meta_cols = [c for c in df.columns if c not in day_cols + ["id", "adm3_name"]]
    long = df.melt(id_vars=["id"] + meta_cols, value_vars=day_cols,
                   var_name="day_col", value_name="rain")
    long = long.merge(weights_df.rename(columns={"source_id": "id"}), on="id", how="inner")
    long["rain"] = long["weight"] * long["rain"]
    trans = long.groupby(meta_cols + ["target_id", "day_col"], as_index=False)["rain"].sum()
    trans = trans.rename(columns={"target_id": "id"})
    wide = trans.pivot_table(index=meta_cols + ["id"], columns="day_col", values="rain").reset_index()
    wide.columns.name = None
    return wide


def transform_groundtruth_long(df, weights_df, value_col):
    """Apply cell transform to ground-truth long DataFrame."""
    if weights_df is None:
        return df
    df = ensure_id_col(df)
    if value_col not in df.columns:
        raise ValueError(f"Value col not found: {value_col}")
    meta_cols = [c for c in df.columns if c not in ["id", value_col, "adm3_name"]]
    x = df.merge(weights_df.rename(columns={"source_id": "id"}), on="id", how="inner")
    x[value_col] = x["weight"] * x[value_col].astype(float)
    trans = x.groupby(meta_cols + ["target_id"], as_index=False)[value_col].sum()
    trans = trans.rename(columns={"target_id": "id"})
    return trans


# ---------------------------------------------------------------------------
# NetCDF readers
# ---------------------------------------------------------------------------

def nc_read_forecast_wide(nc_path, var_name, dim_rename_map, spec,
                          day_dim="day", prefix=None, add_year=True):
    """
    Read a forecast NetCDF (adm3_name-indexed) into a wide-by-lead-day DataFrame.

    The adm3_name dimension is read as strings and stored in an `id` column.
    Returns DataFrame or None if variable not found.
    """
    import netCDF4 as nc4
    prefix = prefix or var_name

    ds = nc4.Dataset(nc_path)
    try:
        if var_name not in ds.variables:
            return None

        v = ds.variables[var_name]
        dim_names = list(v.dimensions)

        # Build coordinate arrays; adm3_name handled specially as strings
        dim_vals = {}
        adm3_dim_name = None
        for d in dim_names:
            dl = d.lower()
            if dl in ("adm3_name", "adm3"):
                adm3_dim_name = d
                dim_vals[d] = get_nc_adm3_names(ds)
            elif dl == "time":
                tinfo = get_nc_time(ds)
                dim_vals[d] = tinfo["values"]
            else:
                if d in ds.variables:
                    dim_vals[d] = np.array(ds[d][:]).flatten()
                elif d in ds.dimensions:
                    dim_vals[d] = np.arange(ds.dimensions[d].size)
                else:
                    dim_vals[d] = np.arange(len(v))

        # Apply renames to identify day dimension
        dummy = pd.DataFrame({d: [dim_vals[d][0]] for d in dim_names})
        dummy_renamed = rename_dimensions(dummy, dim_rename_map or {})
        renamed_names = [c.lower() for c in dummy_renamed.columns]

        try:
            day_idx = renamed_names.index(day_dim.lower())
        except ValueError:
            raise ValueError(
                f"Could not identify day dimension '{day_dim}' (after rename). "
                f"Available: {renamed_names}"
            )

        day_vals = dim_vals[dim_names[day_idx]]
        day_vals_num = []
        for dv in day_vals:
            try:
                day_vals_num.append(int(dv))
            except (TypeError, ValueError):
                day_vals_num.append(None)

        min_day = spec.get("options", {}).get("min_day")
        max_day = spec.get("options", {}).get("max_day")
        keep = np.ones(len(day_vals), dtype=bool)
        if min_day is not None:
            keep &= np.array([(x is not None and x >= int(min_day)) for x in day_vals_num])
        if max_day is not None:
            keep &= np.array([(x is not None and x <= int(max_day)) for x in day_vals_num])
        if not np.any(keep):
            raise ValueError(f"Day filtering removed all columns for: {os.path.basename(nc_path)}")

        day_vals = [dv for dv, k in zip(day_vals, keep) if k]
        day_vals_num = [dv for dv, k in zip(day_vals_num, keep) if k]

        # Read and permute array so day is last
        arr = np.array(v[:])
        other_idx = [i for i in range(len(dim_names)) if i != day_idx]
        perm = other_idx + [day_idx]
        arr = np.transpose(arr, perm)
        arr = arr[..., np.where(keep)[0]]

        # Build grid of other dims
        other_dim_names = [dim_names[i] for i in other_idx]
        other_dim_vals = [dim_vals[dim_names[i]] for i in other_idx]
        grid_rows = list(product(*other_dim_vals))
        grid = pd.DataFrame(grid_rows, columns=other_dim_names)
        grid = rename_dimensions(grid, dim_rename_map or {})
        grid.columns = grid.columns.str.lower()

        # Rename the adm3 column to `id`
        adm3_col_renamed = adm3_dim_name.lower() if adm3_dim_name else None
        if adm3_col_renamed and adm3_col_renamed in grid.columns:
            grid = grid.rename(columns={adm3_col_renamed: "id"})
        # Also catch after rename_map has been applied
        if "adm3_name" in grid.columns:
            grid = grid.rename(columns={"adm3_name": "id"})

        n_rows = len(grid_rows)
        n_days = len(day_vals)
        mat = arr.reshape(n_rows, n_days)

        day_labels = [
            str(dv_num) if dv_num is not None else str(dv)
            for dv, dv_num in zip(day_vals, day_vals_num)
        ]
        day_col_names = [f"{prefix}_day_{lab}" for lab in day_labels]
        day_df = pd.DataFrame(mat, columns=day_col_names)

        out = pd.concat([grid.reset_index(drop=True), day_df], axis=1)

        if add_year and "time" in out.columns:
            tinfo = get_nc_time(ds)
            if pd.api.types.is_float_dtype(out["time"]) or pd.api.types.is_integer_dtype(out["time"]):
                dates = nc_time_to_dates(out["time"].values, tinfo["units"])
            else:
                dates = pd.to_datetime(out["time"])
            out["time"] = dates.date
            out["year"] = pd.DatetimeIndex(dates).year

        return out
    finally:
        ds.close()


def nc_read_groundtruth_long(nc_path, var_name, dim_rename_map, add_year=True):
    """
    Read a ground-truth NetCDF (adm3_name-indexed) into a long (tidy) DataFrame.

    Returns DataFrame or None if variable not found.
    """
    import netCDF4 as nc4

    ds = nc4.Dataset(nc_path)
    try:
        if var_name not in ds.variables:
            return None

        v = ds.variables[var_name]
        dim_names = list(v.dimensions)

        dim_vals = {}
        adm3_dim_name = None
        for d in dim_names:
            dl = d.lower()
            if dl in ("adm3_name", "adm3"):
                adm3_dim_name = d
                dim_vals[d] = get_nc_adm3_names(ds)
            elif dl == "time":
                tinfo = get_nc_time(ds)
                dim_vals[d] = tinfo["values"]
            else:
                if d in ds.variables:
                    dim_vals[d] = np.array(ds[d][:]).flatten()
                elif d in ds.dimensions:
                    dim_vals[d] = np.arange(ds.dimensions[d].size)
                else:
                    dim_vals[d] = np.arange(len(v))

        arr = np.array(v[:])
        grid_rows = list(product(*[dim_vals[d] for d in dim_names]))
        grid = pd.DataFrame(grid_rows, columns=dim_names)
        grid = rename_dimensions(grid, dim_rename_map or {})
        grid.columns = grid.columns.str.lower()

        # Rename adm3 column to `id`
        adm3_col_renamed = adm3_dim_name.lower() if adm3_dim_name else None
        if adm3_col_renamed and adm3_col_renamed in grid.columns:
            grid = grid.rename(columns={adm3_col_renamed: "id"})
        if "adm3_name" in grid.columns:
            grid = grid.rename(columns={"adm3_name": "id"})

        vcol = var_name.lower()
        grid[vcol] = arr.flatten()
        #grid = grid.dropna(subset=[vcol])

        if add_year and "time" in grid.columns:
            tinfo = get_nc_time(ds)
            if pd.api.types.is_float_dtype(grid["time"]) or pd.api.types.is_integer_dtype(grid["time"]):
                dates = nc_time_to_dates(grid["time"].values, tinfo["units"])
                grid["time"] = dates.date
            else:
                grid["time"] = pd.to_datetime(grid["time"]).dt.date
            grid["year"] = pd.to_datetime(grid["time"]).dt.year

        return grid
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# Stage-2 onset helpers
# ---------------------------------------------------------------------------

def order_day_cols(df, key_cols):
    """Return ordered day column names and their integer values."""
    cand = [c for c in df.columns if c not in key_cols]
    day_ints = []
    valid_cols = []
    for c in cand:
        try:
            day_ints.append(int(c))
            valid_cols.append(c)
        except (ValueError, TypeError):
            pass
    order = np.argsort(day_ints)
    return [valid_cols[i] for i in order], [day_ints[i] for i in order]


def calc_onsets_rowwise(df, day_cols, day_ints, win, params=None):
    """
    Compute per-row onset indices under three restriction rules:
      raw, clim_mok_date (after June 2), mok (after MOK date).
    """
    if params is not None and params.mode == "consecutive_dry":
        return calc_onsets_matrix_consecutive_dry(df, day_cols, day_ints, params)

    X = df[day_cols].values.astype(float)
    t0 = pd.to_datetime(df["time"]).values
    th = df["onset_thresh"].values.astype(float)
    yr = df["year"].values.astype(int)

    june2 = np.array([np.datetime64(f"{y}-06-02") for y in yr])
    need_clim_offset = (june2 - t0).astype("timedelta64[D]").astype(int)
    day_ints_arr = np.array(day_ints)
    need_clim = np.searchsorted(day_ints_arr, need_clim_offset - 1, side='right') + 1
    need_clim = np.where(need_clim > len(day_ints_arr), 9999, need_clim).astype(int)

    mok_dates = pd.to_datetime(df["mok_date"]).values if "mok_date" in df.columns else np.array([pd.NaT] * len(df))
    n = len(df)
    onset_raw = []
    onset_clim_mok_date = []
    onset_mok = []

    for i in range(n):
        s = X[i]
        wsum, aux1, aux2 = precompute_onset(s, params)

        onset_raw.append(
            find_onset_from_precomputed(s, th[i], wsum, aux1, aux2, params, start_day=0)
        )
        onset_clim_mok_date.append(
            find_onset_from_precomputed(
                s, th[i], wsum, aux1, aux2, params, start_day=int(need_clim[i])
            )
        )

        mk = mok_dates[i]
        if pd.isnull(mk):
            sd_mok = 1
        else:
            offset = int((mk - t0[i]).astype("timedelta64[D]").astype(int))
            sd_mok = int(np.searchsorted(day_ints_arr, offset - 1, side='right')) + 1
            if sd_mok > len(day_ints_arr):
                sd_mok = 9999
        onset_mok.append(
            find_onset_from_precomputed(s, th[i], wsum, aux1, aux2, params, start_day=sd_mok)
        )

    return {"onset_raw": onset_raw, "onset_clim_mok_date": onset_clim_mok_date, "onset_mok": onset_mok}


def _rolling_sum_na_rm_matrix(X, window):
    X0 = np.nan_to_num(X, nan=0.0)
    csum = np.concatenate(
        [np.zeros((X.shape[0], 1), dtype=float), np.cumsum(X0, axis=1)],
        axis=1,
    )
    return csum[:, window:] - csum[:, :-window]


def _rolling_count_matrix(mask, window):
    counts = mask.astype(np.int16)
    csum = np.concatenate(
        [np.zeros((mask.shape[0], 1), dtype=np.int32), np.cumsum(counts, axis=1)],
        axis=1,
    )
    return csum[:, window:] - csum[:, :-window]


def _dry_spell_prefix_matrix(X, params):
    dry = (~np.isnan(X)) & (X < params.dry_day_min_mm)
    dry_starts = np.zeros(X.shape, dtype=np.int8)
    min_dry_days = params.min_dry_days

    if X.shape[1] >= min_dry_days:
        run = np.zeros(X.shape[0], dtype=np.int16)
        for day_idx in range(X.shape[1]):
            run = (run + 1) * dry[:, day_idx]
            start_idx = day_idx - min_dry_days + 1
            if start_idx >= 0:
                dry_starts[:, start_idx] = run == min_dry_days

    return np.concatenate(
        [np.zeros((X.shape[0], 1), dtype=np.int32), np.cumsum(dry_starts, axis=1)],
        axis=1,
    )


def _first_onset_from_matrices(base_ok, pre_dry, start_days, win, follow_days, series_len):
    n_rows, max_candidate = base_ok.shape
    candidate_days = np.arange(1, max_candidate + 1, dtype=np.int16)
    min_days = np.maximum(1, np.ceil(start_days).astype(np.int32))
    start_ok = candidate_days[None, :] >= min_days[:, None]

    lower = candidate_days - 1 + win
    upper = np.minimum(series_len, candidate_days - 1 + win + follow_days)
    row_idx = np.arange(n_rows)[:, None]
    has_dry_spell = (pre_dry[row_idx, upper[None, :]] - pre_dry[row_idx, lower[None, :]]) > 0

    valid = base_ok & start_ok & ~has_dry_spell
    has_onset = valid.any(axis=1)
    onset = np.full(n_rows, np.nan)
    onset[has_onset] = candidate_days[valid.argmax(axis=1)[has_onset]]
    return onset


def calc_onsets_matrix_consecutive_dry(df, day_cols, day_ints, params):
    X = df[day_cols].values.astype(float)
    t0 = pd.to_datetime(df["time"]).values
    th = df["onset_thresh"].values.astype(float)
    yr = df["year"].values.astype(int)

    win = params.win
    series_len = X.shape[1]
    max_candidate = series_len - win + 1
    if max_candidate < 1:
        empty = [None] * len(df)
        return {"onset_raw": empty, "onset_clim_mok_date": empty, "onset_mok": empty}

    wsum = _rolling_sum_na_rm_matrix(X, win)
    wet_count = _rolling_count_matrix(X >= params.wet_day_min_mm, win)
    all_wet = wet_count == win
    acc_ok = wsum > th[:, None]
    first_day_present = ~np.isnan(X[:, :max_candidate])
    base_ok = all_wet & acc_ok & first_day_present & ~np.isnan(th[:, None])

    pre_dry = _dry_spell_prefix_matrix(X, params)

    day_ints_arr = np.array(day_ints)
    june2 = np.array([np.datetime64(f"{y}-06-02") for y in yr])
    need_clim_offset = (june2 - t0).astype("timedelta64[D]").astype(int)
    need_clim = np.searchsorted(day_ints_arr, need_clim_offset - 1, side="right") + 1
    need_clim = np.where(need_clim > len(day_ints_arr), 9999, need_clim).astype(int)

    mok_dates = pd.to_datetime(df["mok_date"]).values if "mok_date" in df.columns else np.array([pd.NaT] * len(df))
    need_mok = np.ones(len(df), dtype=np.int32)
    has_mok = ~pd.isnull(mok_dates)
    if np.any(has_mok):
        offsets = (mok_dates[has_mok] - t0[has_mok]).astype("timedelta64[D]").astype(int)
        need_mok[has_mok] = np.searchsorted(day_ints_arr, offsets - 1, side="right") + 1
        need_mok = np.where(need_mok > len(day_ints_arr), 9999, need_mok).astype(int)

    onset_raw = _first_onset_from_matrices(
        base_ok, pre_dry, np.zeros(len(df)), win, params.follow_days, series_len
    )
    onset_clim_mok_date = _first_onset_from_matrices(
        base_ok, pre_dry, need_clim, win, params.follow_days, series_len
    )
    onset_mok = _first_onset_from_matrices(
        base_ok, pre_dry, need_mok, win, params.follow_days, series_len
    )

    return {
        "onset_raw": onset_raw,
        "onset_clim_mok_date": onset_clim_mok_date,
        "onset_mok": onset_mok,
    }


def process_rainfall_forecast_id(df, spec, mok_dt=None, thr_dt=None):
    """
    Forecast pipeline: compute ensemble onset probabilities per (id, time, year).

    Returns dict: {"wide": DataFrame}
    """
    df = df.copy()
    df = ensure_id_col(df)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"]).dt.date
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    if "number" in df.columns:
        df["number"] = df["number"].astype(int)

    has_number = "number" in df.columns
    filter_cfg = spec.get("filter") or {}
    max_number = filter_cfg.get("max_number")

    df = attach_thresholds_id(df, thr_dt)
    if mok_dt is not None and "year" in df.columns:
        df = df.merge(mok_dt, on="year", how="left")
    else:
        df["mok_date"] = pd.NaT

    if has_number and max_number is not None:
        df = df[df["number"] <= int(max_number)]
        if df.empty:
            raise ValueError(f"After filtering number <= {max_number}, no rows remain.")

    wide_prefix = spec.get("input", {}).get("wide_prefix") or spec["input"]["value_col"].lower()
    day_pattern = re.compile(rf"^{re.escape(wide_prefix)}_day_(\d+)$")
    day_cols_pref = [c for c in df.columns if day_pattern.match(c)]
    if not day_cols_pref:
        raise ValueError(f"No wide day columns matched pattern '{wide_prefix}_day_<n>'")

    day_nums = {c: int(day_pattern.match(c).group(1)) for c in day_cols_pref}
    rename_map = {c: str(day_nums[c]) for c in day_cols_pref}
    df = df.rename(columns=rename_map)

    key_base = ["id", "time"]
    if "year" in df.columns:
        key_base.append("year")
    key_base += ["onset_thresh", "mok_date"]
    key_member = key_base + (["number"] if has_number else [])

    # Drop non-key, non-day columns to avoid merge collisions
    cols_to_drop = [c for c in df.columns
                    if c not in key_member
                    #and not any(c == str(i) for i in range(1, 100))]
                    and not any(c == str(i) for i in range(0, 100))] # <---NEW
    df = df.drop(columns=cols_to_drop)

    day_cols, day_ints = order_day_cols(df, key_member)
    onset_params = read_onset_params(spec)
    win = onset_params.win
    min_day = int(spec["options"]["min_day"])
    max_day = int(spec["options"]["max_day"])

    keep_days = [dc for dc, di in zip(day_cols, day_ints) if min_day <= di <= max_day + win - 1]
    keep_ints = [di for di in day_ints if min_day <= di <= max_day + win - 1]
    if not keep_days:
        raise ValueError("After min_day/max_day filtering, no day columns remain.")

    on = calc_onsets_rowwise(df, keep_days, keep_ints, win, params=onset_params)
    df["onset_raw"] = on["onset_raw"]
    df["onset_clim_mok_date"] = on["onset_clim_mok_date"]
    df["onset_mok"] = on["onset_mok"]

    D = len(keep_ints)

    grouped = df.groupby(key_base)
    group_sizes = grouped.size()

    # Preserve the legacy stats column labels, which used day_ints positions.
    stats_day_labels = day_ints[:len(keep_days)]
    mean_agg = grouped[keep_days].mean()
    mean_agg.columns = [f"forecast_rain_day_{d}" for d in stats_day_labels]

    sd_agg = grouped[keep_days].std()
    sd_agg.columns = [f"forecast_rain_sd_day_{d}" for d in stats_day_labels]

    rain_binary = df[keep_days].gt(1)
    rain_binary[key_base] = df[key_base]
    frac_agg = rain_binary.groupby(key_base)[keep_days].mean()
    frac_agg.columns = [f"frac_raining_day_{d}" for d in stats_day_labels]

    stats_agg = pd.concat([mean_agg, sd_agg, frac_agg], axis=1).reset_index()

    def _prob_agg(onset_col, out_prefix):
        valid = df[onset_col].notna() & df[onset_col].between(1, D)
        if valid.any():
            counts = (
                df.loc[valid, key_base + [onset_col]]
                  .groupby(key_base + [onset_col])
                  .size()
                  .unstack(fill_value=0)
            )
            counts = counts.reindex(columns=range(1, D + 1), fill_value=0)
        else:
            counts = pd.DataFrame(0, index=group_sizes.index, columns=range(1, D + 1))
        probs = counts.reindex(group_sizes.index, fill_value=0).div(group_sizes, axis=0)
        probs.columns = [f"{out_prefix}_day_{keep_ints[j]}" for j in range(D)]
        return probs

    prob_agg = pd.concat(
        [
            _prob_agg("onset_raw", "predicted_prob"),
            _prob_agg("onset_clim_mok_date", "predicted_prob_clim_mok_date"),
            _prob_agg("onset_mok", "predicted_prob_mok"),
        ],
        axis=1,
    ).reset_index()

    wide = stats_agg.merge(prob_agg, on=key_base)
    return {"wide": wide}


def process_ground_truth_rainfall_id(df, spec, mok_dt=None, thr_dt=None, value_col=None):
    """
    Ground-truth pipeline: compute onset dates per (id, year).

    Returns dict: {"wide": DataFrame, "long": DataFrame}
    """
    df = df.copy()
    df = ensure_id_col(df)
    df["time"] = pd.to_datetime(df["time"]).dt.date
    df["year"] = df["year"].astype(int)
    df[value_col] = df[value_col].astype(float)

    df = attach_thresholds_id(df, thr_dt)
    if mok_dt is not None:
        df = df.merge(mok_dt, on="year", how="left")
    else:
        df["mok_date"] = pd.NaT

    cutoff_md = spec["options"]["cutoff_month_day"]
    df["cutoff_date"] = df["year"].apply(lambda y: pd.Timestamp(f"{y}-{cutoff_md}").date())
    df["start_date"] = df["cutoff_date"]
    mask = ~df["mok_date"].isna()
    df.loc[mask, "start_date"] = df.loc[mask].apply(
        lambda r: max(r["cutoff_date"], r["mok_date"]), axis=1
    )

    df = df.sort_values(["id", "year", "time"])
    #win = int(spec["options"]["window"])
    onset_params = read_onset_params(spec)
    win = onset_params.win

    wide_rows = []
    long_rows = []

    for (cell_id, yr), g in df.groupby(["id", "year"]):
        series = g[value_col].values
        dates = g["time"].values
        sd = g["start_date"].iloc[0]
        th = g["onset_thresh"].iloc[0]

        start_pos_arr = np.where(pd.to_datetime(dates) >= pd.Timestamp(sd))[0]
        start_pos = int(start_pos_arr[0]) + 1 if len(start_pos_arr) > 0 else 9999

        if np.all(np.isnan(series)) or len(series) < win or np.isnan(th):
            mr_idx = None
            mr_date = None
        else:
            wsum_all = roll_sum_na_rm_left(series, win)
            if len(series) >= 10:
                sum10 = roll_sum_na_propagate_left(series, 10)
                bad10 = (~np.isnan(sum10)) & (sum10 < 5)
                pre_bad = np.concatenate([[0], np.cumsum(bad10.astype(int))])
                last10start = len(series) - 10 + 1
            else:
                pre_bad = np.array([0])
                last10start = 0

            #mr_idx = find_onset_precomp(series, win, th, wsum_all, pre_bad, last10start,
            #                            start_day=start_pos, reject_if_short_followup=True)
            mr_idx = find_onset_precomp(series, win, th, wsum_all, pre_bad, last10start,
                            start_day=start_pos, reject_if_short_followup=True,
                            params=onset_params)
            mr_date = dates[mr_idx - 1] if mr_idx is not None else None

        cutoff_date = g["cutoff_date"].iloc[0]
        if mr_date is not None:
            mr_onset_day = (pd.Timestamp(mr_date) - pd.Timestamp(cutoff_date)).days
        else:
            mr_onset_day = np.nan

        wide_rows.append({
            "id": cell_id,
            "year": yr,
            "mr_onset_idx": mr_idx if mr_idx is not None else np.nan,
            "mr_onset_date": mr_date,
            "mr_onset_day": mr_onset_day,
            "cutoff_date": cutoff_date,
        })

        mok_date = g["mok_date"].iloc[0]
        for _, row in g.iterrows():
            long_rows.append({
                "id": cell_id,
                "time": row["time"],
                "year": yr,
                value_col: row[value_col],
                "onset_thresh": th,
                "mok_date": mok_date,
                "mr_onset_date": mr_date,
                "mr_onset_flag": row["time"] == mr_date,
            })

    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows)
    return {"wide": wide, "long": long}


# ---------------------------------------------------------------------------
# Pipeline entrypoint
# ---------------------------------------------------------------------------

def _process_single_nc_file(row, spec, var_name, dim_rename_map, mok_dt, thr_dt, weights_df):
    nc_path = row["nc_path"]
    yr = int(row["year"])
    print(f"Processing year {yr}: {nc_path}")

    if spec["type"] == "rainfall_forecast":
        wide_prefix = spec["input"].get("wide_prefix") or var_name.lower()
        day_dim = spec["input"].get("wide_day_dim", "day")
        dt = nc_read_forecast_wide(nc_path, var_name, dim_rename_map, spec,
                                   day_dim=day_dim, prefix=wide_prefix)
        if dt is None:
            return {
                "year": yr,
                "wide": None,
                "long": None,
                "message": f"  Skipping {nc_path}: variable '{var_name}' not found.",
            }
        dt["year"] = yr
        dt = filter_by_dissemination_cells(dt, spec)
        if weights_df is not None:
            dt = transform_forecast_wide(dt, weights_df)
        result = process_rainfall_forecast_id(dt, spec, mok_dt=mok_dt, thr_dt=thr_dt)
        return {"year": yr, "wide": result["wide"], "long": None, "message": None}

    if spec["type"] == "ground_truth_rainfall":
        dt = nc_read_groundtruth_long(nc_path, var_name, dim_rename_map)
        if dt is None:
            return {
                "year": yr,
                "wide": None,
                "long": None,
                "message": f"  Skipping {nc_path}: variable '{var_name}' not found.",
            }
        dt["year"] = yr
        dt = filter_by_dissemination_cells(dt, spec)
        if weights_df is not None:
            dt = transform_groundtruth_long(dt, weights_df, var_name.lower())
        result = process_ground_truth_rainfall_id(dt, spec, mok_dt=mok_dt, thr_dt=thr_dt,
                                                  value_col=var_name.lower())
        return {"year": yr, "wide": result["wide"], "long": result["long"], "message": None}

    raise ValueError(f"Unsupported raw data type: {spec['type']}")


def _process_nc_files(files_df, spec, var_name, dim_rename_map, mok_dt, thr_dt, weights_df):
    rows = files_df.to_dict("records")
    parallel = bool(spec.get("input", {}).get("parallel", False))
    workers = int(spec.get("input", {}).get("workers") or 5)

    if not parallel or workers <= 1 or len(rows) <= 1:
        return [
            _process_single_nc_file(row, spec, var_name, dim_rename_map, mok_dt, thr_dt, weights_df)
            for row in rows
        ]

    worker_count = min(workers, len(rows))
    print(f"Processing {len(rows)} NetCDF files with {worker_count} workers.")
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                _process_single_nc_file,
                row,
                spec,
                var_name,
                dim_rename_map,
                mok_dt,
                thr_dt,
                weights_df,
            )
            for row in rows
        ]
        return [future.result() for future in futures]


def run_single_pipeline(spec_id):
    """
    Main driver: load spec, process all NetCDF years, write outputs.

    Parameters
    ----------
    spec_id : str
        Spec ID (loads specs/raw_data/<spec_id>.yml).
    """
    spec = load_spec(spec_id, "raw_data")
    spec["id"] = spec_id
    spec = validate_spec_single(spec)

    out_dir = spec["output"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    basename = spec["output"].get("basename", spec_id)

    var_name = get_value_var(spec)
    dim_rename_map = spec.get("dimensions", {}).get("rename") or {}
    mok_dt = read_mok_dates(spec)
    thr_dt = read_thresholds(spec)
    weights_df = read_cell_transform(spec)

    files_df = list_nc_files_with_year(spec)

    results = _process_nc_files(
        files_df,
        spec,
        var_name,
        dim_rename_map,
        mok_dt,
        thr_dt,
        weights_df,
    )
    results = sorted(results, key=lambda result: result["year"])

    for result in results:
        if result["message"]:
            print(result["message"])

    wide_all = [result["wide"] for result in results if result["wide"] is not None]
    long_all = [result["long"] for result in results if result["long"] is not None]

    if wide_all:
        wide_out = pd.concat(wide_all, ignore_index=True)
        wide_path = os.path.join(out_dir, f"{basename}_wide.pkl")
        with open(wide_path, "wb") as f:
            pickle.dump(wide_out, f)
        print(f"Wrote wide: {wide_path}")

    if long_all:
        long_out = pd.concat(long_all, ignore_index=True)
        long_path = os.path.join(out_dir, f"{basename}_long.pkl")
        with open(long_path, "wb") as f:
            pickle.dump(long_out, f)
        print(f"Wrote long: {long_path}")
