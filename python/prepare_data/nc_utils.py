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
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from itertools import product

from ..pipelines._shared.misc import coalesce
from ..pipelines._shared.read_spec import load_spec, validate_spec
from ..rain_horizon_utils import (
    resolve_rain_day_max,
    validate_day_coordinate,
    validate_rain_horizon_frame,
)
from .onset_utils import (
    read_ref_onset_dates, read_thresholds,
    read_onset_params,
    roll_sum_na_rm_left, roll_sum_na_propagate_left,
    find_onset_precomp, find_onsets_batch,
)
from .spatial_id_utils import (
    GridIdConvention,
    ensure_spatial_id_col,
    format_grid_ids,
    normalize_id_series,
    resolve_grid_id_convention,
    validate_id_coordinate_consistency,
    validate_expected_source_ids,
)
from .sparse_transform_utils import (
    compile_sparse_cell_transform,
    sparse_observed_weighted_mean,
    sparse_target_support,
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
            configured_horizon = s.get("options", {}).get("rain_day_max")
            if configured_horizon is not None:
                rain_day_max, _ = resolve_rain_day_max(
                    {
                        "rain_day_max": configured_horizon,
                        "rain_horizon_policy": s["options"].get(
                            "rain_horizon_policy"
                        ),
                    },
                    probability_day_max=1,
                )
                if int(s["options"]["min_day"]) != 1:
                    raise ValueError(
                        "Strict options.rain_day_max requires options.min_day: 1."
                    )
                if int(s["options"]["max_day"]) < rain_day_max:
                    raise ValueError(
                        "options.max_day must be at least options.rain_day_max "
                        f"({rain_day_max}) in strict horizon mode."
                    )
        if s["type"] == "ground_truth_rainfall":
            if s.get("options", {}).get("window") is None:
                raise ValueError("Missing options.window")
            if s.get("options", {}).get("cutoff_month_day") is None:
                raise ValueError("Missing options.cutoff_month_day")
            if not isinstance(s.get("output", {}).get("write_long", True), bool):
                raise ValueError("output.write_long must be true or false")
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
# ID helpers  (adm3_name or legacy lat/lon grids)
# ---------------------------------------------------------------------------

def _format_coord_component(value):
    """Format one coordinate for callers of the former private helper."""
    value = float(value)
    if not np.isfinite(value):
        return None
    return np.format_float_positional(value, trim="-")


def _latlon_ids(lat, lon, spec=None, convention=None):
    convention = convention or resolve_grid_id_convention(
        spec=spec, lat=lat, lon=lon, context="latitude/longitude IDs"
    )
    return format_grid_ids(lat, lon, convention)

def ensure_id_col(df, id_col="id", spec=None, convention=None,
                  force_latlon=False, context="spatial data"):
    """
    Ensure the DataFrame has an `id` column.
    For adm3-based data, rename `adm3_name` to `id`. For legacy gridded data,
    construct the same `<lat>_<lon>` key used by the R pipeline.
    """
    return ensure_spatial_id_col(
        df,
        id_col=id_col,
        spec=spec,
        convention=convention,
        force_latlon=force_latlon,
        context=context,
    )


# Keep the old name as an alias so callers that import it directly still work.
def add_id_from_latlon(df, lat_col="lat", lon_col="lon", id_col="id",
                       spec=None, convention=None):
    """Backward-compatible explicit lat/lon-to-id helper."""
    if id_col in df.columns:
        return df.copy()
    if lat_col not in df.columns or lon_col not in df.columns:
        return ensure_id_col(df, id_col=id_col, spec=spec,
                             convention=convention)
    out = df.copy()
    coords = out[[lat_col, lon_col]].rename(
        columns={lat_col: "lat", lon_col: "lon"}
    )
    coords = ensure_spatial_id_col(
        coords,
        id_col=id_col,
        spec=spec,
        convention=convention,
        force_latlon=True,
        context="latitude/longitude IDs",
    )
    out[id_col] = coords[id_col].to_numpy()
    return out


def prep_thresholds_id(thr_df, spec=None, convention=None):
    """
    Normalize thresholds DataFrame to be keyed by id (adm3_name).

    Accepts:
      - DataFrame with 'id' column
      - DataFrame with 'adm3_name' column
      - DataFrame with legacy 'lat' and 'lon' columns
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
            thr_df["id"] = thr_df["adm3_name"]
        elif "lat" in thr_df.columns and "lon" in thr_df.columns:
            thr_df = add_id_from_latlon(
                thr_df, spec=spec, convention=convention
            )
        else:
            raise ValueError(
                "Thresholds table must have 'id', 'adm3_name', or ('lat', 'lon')."
            )
    thr_df["id"] = normalize_id_series(thr_df["id"], context="threshold IDs")
    thr_df["onset_thresh"] = thr_df["onset_thresh"].astype(float)
    thresholds = thr_df[["id", "onset_thresh"]].drop_duplicates()
    conflicting_ids = thresholds["id"].duplicated(keep=False)
    if conflicting_ids.any():
        sample = ", ".join(thresholds.loc[conflicting_ids, "id"].unique()[:10])
        raise ValueError(f"Threshold IDs have conflicting values: {sample}")
    return thresholds.set_index("id")


def attach_thresholds_id(df, thr_df, spec=None, convention=None):
    """Left-join onset_thresh from thr_df by id."""
    df = ensure_id_col(df, spec=spec, convention=convention)
    convention = convention or resolve_grid_id_convention(
        spec=spec,
        authoritative_ids=df["id"],
        context="rainfall/threshold ID handoff",
    )
    if thr_df is None:
        df["onset_thresh"] = np.nan
        return df
    # Scalar threshold
    if isinstance(thr_df, (int, float, np.floating, np.integer)):
        df["onset_thresh"] = float(thr_df)
        return df
    thr_idx = prep_thresholds_id(thr_df, spec=spec, convention=convention)
    if isinstance(thr_idx, (int, float)):
        df["onset_thresh"] = float(thr_idx)
        return df
    df = df.merge(thr_idx.reset_index(), on="id", how="left")
    return df


# ---------------------------------------------------------------------------
# Cell filtering by dissemination cells list
# ---------------------------------------------------------------------------

def _resolve_unit_latlon(df, filt):
    """
    Return (lat, lon) arrays aligned to df rows for the domain bbox filter, or
    (None, None) if they cannot be determined. Sources, in order:
      1. explicit 'lat'/'lon' columns on df (gridded/legacy data),
      2. an id of the form '<lat>_<lon>' (grid-cell units),
      3. filter.centroids_file: CSV with adm3_name + lat/lon (or center_lat/
         center_lon) giving a representative point per unit (admin units).
    """
    if "lat" in df.columns and "lon" in df.columns:
        return df["lat"].astype(float).values, df["lon"].astype(float).values

    ids = df["id"].astype(str)
    parsed = ids.str.extract(r"^(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$")
    if len(df) and parsed.notna().all(axis=None):
        return parsed[0].astype(float).values, parsed[1].astype(float).values

    cf = (filt or {}).get("centroids_file")
    if cf:
        if not os.path.exists(cf):
            raise FileNotFoundError(f"filter.centroids_file not found: {cf}")
        c = pd.read_csv(cf, dtype=str)
        keyc = filt.get("centroids_id_col")
        if keyc and keyc not in c.columns:
            raise ValueError(
                f"filter.centroids_id_col '{keyc}' not found. "
                f"Found: {c.columns.tolist()}"
            )
        if not keyc:
            keyc = next(
                (x for x in ("id", "adm3_name") if x in c.columns),
                c.columns[0],
            )
        latc = next((x for x in ("lat", "center_lat", "latitude") if x in c.columns), None)
        lonc = next((x for x in ("lon", "center_lon", "longitude") if x in c.columns), None)
        if not latc or not lonc:
            raise ValueError("filter.centroids_file must have lat/lon (or center_lat/center_lon).")
        c = c.rename(columns={keyc: "id", latc: "_lat", lonc: "_lon"})
        c["id"] = normalize_id_series(c["id"], context="centroid IDs")
        m = df[["id"]].merge(c[["id", "_lat", "_lon"]].drop_duplicates("id"), on="id", how="left")
        return m["_lat"].astype(float).values, m["_lon"].astype(float).values

    return None, None


def attach_ref_onset(df, ref):
    """
    Attach a `ref_onset_date` column to df from the value returned by
    read_ref_onset_dates: None (NaT), a constant-month-day dict (a fixed date
    each year), or a DataFrame keyed by 'year' and/or 'id' (per-year and/or
    per-unit dates). Merges on whichever keys are present in both.
    """
    df = df.copy()
    if ref is None:
        df["ref_onset_date"] = pd.NaT
        return df
    if isinstance(ref, dict) and ref.get("mode") == "constant_month_day":
        md = ref["month_day"]
        if "year" in df.columns:
            df["ref_onset_date"] = pd.to_datetime(
                df["year"].astype(int).astype(str) + "-" + md, errors="coerce"
            ).dt.date
        else:
            df["ref_onset_date"] = pd.NaT
        return df
    # DataFrame keyed by year and/or id
    if "id" in ref.columns:
        df = ensure_id_col(df)
    keys = [k for k in ("id", "year") if k in ref.columns and k in df.columns]
    if not keys:
        df["ref_onset_date"] = pd.NaT
        return df
    return df.merge(ref, on=keys, how="left")


def filter_by_dissemination_cells(df, spec):
    """
    Restrict the domain to the modelling units of interest. Two independent,
    composable restrictions read from spec['filter']:

      - dissemination_cells_file : keep only ids listed in a configured `id_col`,
        canonical `id`, legacy `adm3_name`, or `lat`/`lon` columns.
      - bbox : an optional FURTHER restriction {lat_min, lat_max, lon_min,
        lon_max} (any subset of keys), applied on top of the dissemination set.
        Defaults to no bbox restriction (i.e. all dissemination cells).

    Returns df unchanged if neither is configured.
    """
    filt = spec.get("filter") or {}
    df = ensure_id_col(df, spec=spec, context="rainfall spatial IDs")

    dc_file = filt.get("dissemination_cells_file")
    if dc_file:
        if not os.path.exists(dc_file):
            raise FileNotFoundError(f"dissemination_cells_file not found: {dc_file}")
        dc = pd.read_csv(dc_file, dtype=str)
        configured_id_col = filt.get("id_col") or filt.get(
            "dissemination_id_col"
        )
        if configured_id_col:
            if configured_id_col not in dc.columns:
                raise ValueError(
                    f"Configured dissemination id_col '{configured_id_col}' "
                    f"not found. Found: {dc.columns.tolist()}"
                )
            dc = dc.rename(columns={configured_id_col: "id"})
        elif not any(
            column in dc.columns for column in ("id", "adm3_name")
        ) and not {"lat", "lon"}.issubset(dc.columns):
            fallback_col = dc.columns[0]
            print(
                "  WARNING: dissemination_cells_file has no explicit spatial "
                f"key; using first column '{fallback_col}' as a legacy fallback."
            )
            dc = dc.rename(columns={fallback_col: "id"})
        convention = resolve_grid_id_convention(
            spec=spec,
            authoritative_ids=df["id"],
            context="rainfall/dissemination ID handoff",
        )
        dc = ensure_id_col(
            dc,
            spec=spec,
            convention=convention,
            context="dissemination cell IDs",
        )
        validate_id_coordinate_consistency(dc, context="dissemination cell IDs")
        dc = dc.drop_duplicates("id")
        valid_ids = set(dc["id"])
        before = len(df)
        df = df[df["id"].isin(valid_ids)]
        print(f"  dissemination_cells filter: {before} -> {len(df)} rows "
              f"({before - len(df)} removed)")

    bbox = filt.get("bbox")
    if bbox:
        lat, lon = _resolve_unit_latlon(df, filt)
        if lat is None:
            raise ValueError(
                "filter.bbox requested but unit lat/lon could not be determined: "
                "df has no 'lat'/'lon' columns, ids are not '<lat>_<lon>', and no "
                "filter.centroids_file was provided."
            )
        mask = np.ones(len(df), dtype=bool)
        if bbox.get("lat_min") is not None:
            mask &= lat >= float(bbox["lat_min"])
        if bbox.get("lat_max") is not None:
            mask &= lat <= float(bbox["lat_max"])
        if bbox.get("lon_min") is not None:
            mask &= lon >= float(bbox["lon_min"])
        if bbox.get("lon_max") is not None:
            mask &= lon <= float(bbox["lon_max"])
        before = len(df)
        df = df[mask]
        print(f"  bbox filter {bbox}: {before} -> {len(df)} rows ({before - len(df)} removed)")

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
    w = pd.read_csv(f, dtype={"source_id": str, "target_id": str})
    for col in ("target_id", "source_id", "weight"):
        if col not in w.columns:
            raise ValueError(f"Transform file must have: target_id, source_id, weight")
    if w.empty:
        raise ValueError("Cell-transform weights file must not be empty.")
    w["source_id"] = normalize_id_series(
        w["source_id"], context="cell-transform source IDs"
    )
    w["target_id"] = normalize_id_series(
        w["target_id"], context="cell-transform target IDs"
    )
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce")
    if not np.isfinite(w["weight"]).all() or (w["weight"] < 0).any():
        raise ValueError("Cell-transform weights must be finite and non-negative.")
    if w.duplicated(["source_id", "target_id"]).any():
        raise ValueError("Cell-transform source_id/target_id pairs must be unique.")
    zero_targets = w.groupby("target_id")["weight"].sum()
    zero_targets = zero_targets[zero_targets <= 0]
    if len(zero_targets):
        sample = ", ".join(zero_targets.index.astype(str)[:10])
        raise ValueError(f"Cell-transform targets have zero total weight: {sample}")
    w = w.loc[w["weight"] > 0].copy()
    convention = resolve_grid_id_convention(
        spec=spec,
        authoritative_ids=w["source_id"].drop_duplicates(),
        context=f"cell-transform file {os.path.basename(f)}",
    )
    if convention is not None:
        w.attrs["source_id_convention"] = convention.as_dict()
        print(
            "  cell-transform source ID convention: "
            f"digits={convention.decimal_digits}, "
            f"format={convention.number_format} "
            f"({convention.source})"
        )
    w.attrs["sparse_transform"] = compile_sparse_cell_transform(
        w,
        source_ids=tuple(sorted(w["source_id"].unique())),
        target_ids=tuple(sorted(w["target_id"].unique())),
    )
    return w


def _prepare_transform_source_ids(df, weights_df, spec, context):
    convention_data = weights_df.attrs.get("source_id_convention")
    convention = (
        GridIdConvention.from_dict(convention_data)
        if convention_data else None
    )
    df, extra_ids, _ = validate_expected_source_ids(
        df,
        weights_df["source_id"].drop_duplicates(),
        convention=convention,
        spec=spec,
        context=context,
    )
    if extra_ids and not weights_df.attrs.get("reported_extra_source_ids"):
        print(
            f"  cell-transform source coverage: {len(extra_ids)} raw grid "
            "IDs are outside the supplied weights and will be ignored."
        )
        weights_df.attrs["reported_extra_source_ids"] = True
    return df


def _observed_weighted_mean(df, value_col, group_cols):
    """Aggregate over finite source values and renormalize observed weights."""
    values = pd.to_numeric(df[value_col], errors="coerce")
    valid = np.isfinite(values) & np.isfinite(df["weight"])
    work = df.assign(
        _weighted_value=np.where(valid, values * df["weight"], 0.0),
        _observed_weight=np.where(valid, df["weight"], 0.0),
    )
    out = work.groupby(group_cols, as_index=False, dropna=False)[
        ["_weighted_value", "_observed_weight"]
    ].sum()
    out[value_col] = out["_weighted_value"].div(out["_observed_weight"])
    out.loc[out["_observed_weight"] <= 0, value_col] = np.nan
    return out[group_cols + [value_col]]


def _get_sparse_transform(weights_df):
    transform = weights_df.attrs.get("sparse_transform")
    if transform is None:
        transform = compile_sparse_cell_transform(
            weights_df,
            source_ids=tuple(sorted(weights_df["source_id"].unique())),
            target_ids=tuple(sorted(weights_df["target_id"].unique())),
        )
        weights_df.attrs["sparse_transform"] = transform
    return transform


def _transform_forecast_wide_pandas(df, weights_df, day_cols, meta_cols):
    """Trusted pandas reference path for forecast cell transforms."""
    long = df.melt(id_vars=["id"] + meta_cols, value_vars=day_cols,
                   var_name="day_col", value_name="rain")
    long = long.merge(weights_df.rename(columns={"source_id": "id"}), on="id", how="inner")
    trans = _observed_weighted_mean(
        long, "rain", meta_cols + ["target_id", "day_col"]
    )
    trans = trans.rename(columns={"target_id": "id"})
    wide = trans.pivot(
        index=meta_cols + ["id"], columns="day_col", values="rain"
    ).reset_index()
    wide.columns.name = None
    return wide


def _transform_groundtruth_long_pandas(df, weights_df, value_col, meta_cols):
    """Trusted pandas reference path for ground-truth cell transforms."""
    x = df.merge(weights_df.rename(columns={"source_id": "id"}), on="id", how="inner")
    trans = _observed_weighted_mean(
        x, value_col, meta_cols + ["target_id"]
    )
    trans = trans.rename(columns={"target_id": "id"})
    return trans


def _transform_values_sparse(df, weights_df, value_cols, meta_cols,
                             group_chunk_size=16):
    """Transform one or more rainfall columns in bounded metadata-group chunks."""
    transform = _get_sparse_transform(weights_df)
    work = df[df["id"].isin(transform.source_index)].copy()
    work["id"] = work["id"].astype(str)
    group_id_col = "_transform_group_id"
    if meta_cols:
        work[group_id_col] = work.groupby(
            meta_cols, sort=True, dropna=False
        ).ngroup()
        group_meta = work[[group_id_col] + meta_cols].drop_duplicates(group_id_col)
    else:
        work[group_id_col] = 0
        group_meta = pd.DataFrame({group_id_col: [0]})
    group_meta = group_meta.sort_values(group_id_col).reset_index(drop=True)

    chunks = []
    group_ids = group_meta[group_id_col].to_numpy()
    n_source = len(transform.source_ids)
    n_target = len(transform.target_ids)
    n_value = len(value_cols)
    for start in range(0, len(group_ids), int(group_chunk_size)):
        chunk_ids = group_ids[start:start + int(group_chunk_size)]
        rows = work[work[group_id_col].isin(chunk_ids)]
        group_pos = rows[group_id_col].map(
            {group_id: pos for pos, group_id in enumerate(chunk_ids)}
        ).to_numpy(int)
        source_pos = rows["id"].map(transform.source_index).to_numpy(int)
        values = rows[value_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)

        source_values = np.full((n_source, len(chunk_ids) * n_value), np.nan)
        for value_pos in range(n_value):
            output_pos = group_pos * n_value + value_pos
            source_values[source_pos, output_pos] = values[:, value_pos]
        transformed = sparse_observed_weighted_mean(transform, source_values)

        present_sources = np.zeros((n_source, len(chunk_ids)), dtype=bool)
        present_sources[source_pos, group_pos] = True
        supported = sparse_target_support(transform, present_sources)

        chunk_meta = group_meta.set_index(group_id_col).loc[chunk_ids].reset_index()
        if meta_cols:
            out = chunk_meta.loc[
                chunk_meta.index.repeat(n_target), meta_cols
            ].reset_index(drop=True)
        else:
            out = pd.DataFrame(index=np.arange(len(chunk_ids) * n_target))
        out["id"] = np.tile(transform.target_ids, len(chunk_ids))
        for value_pos, value_col in enumerate(value_cols):
            out[value_col] = transformed[:, value_pos::n_value].T.reshape(-1)
        chunks.append(out.loc[supported.T.reshape(-1)].reset_index(drop=True))

    if not chunks:
        return pd.DataFrame(columns=meta_cols + ["id"] + value_cols)
    return pd.concat(chunks, ignore_index=True)[meta_cols + ["id"] + value_cols]


def transform_forecast_wide(df, weights_df, spec=None):
    """Apply a sparse cell transform to a forecast wide-by-day DataFrame."""
    if weights_df is None:
        return df
    df = _prepare_transform_source_ids(
        df, weights_df, spec, "forecast cell transform"
    )
    day_cols = [c for c in df.columns if re.search(r"_day_\d+$", c)]
    if not day_cols:
        raise ValueError("No forecast day columns found to transform.")
    spatial_cols = {"id", "adm3_name", "lat", "lon", "latitude", "longitude"}
    meta_cols = [c for c in df.columns if c not in day_cols and c not in spatial_cols]
    relevant = df[df["id"].isin(set(weights_df["source_id"]))]
    if relevant.duplicated(meta_cols + ["id"]).any():
        warnings.warn(
            "Duplicate source IDs within a forecast metadata group; using the "
            "pandas cell-transform reference path.",
            RuntimeWarning,
        )
        return _transform_forecast_wide_pandas(df, weights_df, day_cols, meta_cols)
    out = _transform_values_sparse(df, weights_df, day_cols, meta_cols)
    return out.sort_values(
        meta_cols + ["id"], na_position="first"
    ).reset_index(drop=True)


def transform_groundtruth_long(df, weights_df, value_col, spec=None):
    """Apply a sparse cell transform to a ground-truth long DataFrame."""
    if weights_df is None:
        return df
    df = _prepare_transform_source_ids(
        df, weights_df, spec, "ground-truth cell transform"
    )
    if value_col not in df.columns:
        raise ValueError(f"Value col not found: {value_col}")
    spatial_cols = {"id", "adm3_name", "lat", "lon", "latitude", "longitude"}
    meta_cols = [c for c in df.columns if c != value_col and c not in spatial_cols]
    relevant = df[df["id"].isin(set(weights_df["source_id"]))]
    if relevant.duplicated(meta_cols + ["id"]).any():
        warnings.warn(
            "Duplicate source IDs within a ground-truth metadata group; using "
            "the pandas cell-transform reference path.",
            RuntimeWarning,
        )
        return _transform_groundtruth_long_pandas(
            df, weights_df, value_col, meta_cols
        )
    return _transform_values_sparse(df, weights_df, [value_col], meta_cols)


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

        strict_rain_day_max = spec.get("options", {}).get("rain_day_max")
        if strict_rain_day_max is not None:
            _, horizon_policy = resolve_rain_day_max(
                {
                    "rain_day_max": strict_rain_day_max,
                    "rain_horizon_policy": spec["options"].get(
                        "rain_horizon_policy"
                    ),
                },
                probability_day_max=1,
            )
            validation_day_vals = [
                value for value, day in zip(day_vals, day_vals_num)
                if day is not None and (min_day is None or day >= int(min_day))
            ]
            validate_day_coordinate(
                validation_day_vals,
                strict_rain_day_max,
                context=f"NetCDF file {os.path.basename(nc_path)}",
                allow_extra=horizon_policy == "truncate",
            )

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
        raw = v[:]
        arr = np.ma.filled(raw, np.nan) if np.ma.isMaskedArray(raw) else np.asarray(raw)
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


def nc_read_groundtruth_long(nc_path, var_name, dim_rename_map, add_year=True,
                             missing_rain_policy="keep"):
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

        raw = v[:]
        arr = np.ma.filled(raw, np.nan) if np.ma.isMaskedArray(raw) else np.asarray(raw)
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
        missing_rain_policy = str(missing_rain_policy).lower()
        if missing_rain_policy == "drop":
            grid = grid.dropna(subset=[vcol]).reset_index(drop=True)
        elif missing_rain_policy == "zero":
            grid[vcol] = grid[vcol].fillna(0.0)
        elif missing_rain_policy != "keep":
            raise ValueError(
                "missing_rain_policy must be 'keep', 'drop', or 'zero', "
                f"got '{missing_rain_policy}'"
            )

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


def calc_onsets_rowwise(df, day_cols, day_ints, win, params=None,
                        fixed_cutoff_month_day="06-02"):
    """
    Compute per-row onset indices under three restriction rules:
      raw, fixed_cutoff (after a fixed climatological cutoff date), ref_onset
      (after the per-year reference onset date).

    fixed_cutoff_month_day : str "MM-DD"
        Fixed climatological cutoff (default "06-02"); the fixed_cutoff variant
        only considers onsets on/after this month-day each year. Configurable
        via options.fixed_cutoff_month_day in the spec so a new geography can set
        its own climatological season cutoff.
    """
    if params is None:
        params = read_onset_params({"options": {"window": win}})

    t0 = pd.to_datetime(df["time"]).values
    th = df["onset_thresh"].values.astype(float)
    yr = df["year"].values.astype(int)

    cutoff_dates = np.array([np.datetime64(f"{y}-{fixed_cutoff_month_day}") for y in yr])
    need_clim_offset = (cutoff_dates - t0).astype("timedelta64[D]").astype(int)
    day_ints_arr = np.array(day_ints)
    need_clim = np.searchsorted(day_ints_arr, need_clim_offset - 1, side='right') + 1
    need_clim = np.where(need_clim > len(day_ints_arr), 9999, need_clim).astype(int)

    ref_onset_dates = (
        pd.to_datetime(df["ref_onset_date"]).values
        if "ref_onset_date" in df.columns
        else np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")
    )
    start_ref = np.ones(len(df), dtype=int)
    has_ref = ~pd.isnull(ref_onset_dates)
    ref_offsets = np.zeros(len(df), dtype=int)
    ref_offsets[has_ref] = (
        ref_onset_dates[has_ref] - t0[has_ref]
    ).astype("timedelta64[D]").astype(int)
    start_ref[has_ref] = (
        np.searchsorted(
            day_ints_arr, ref_offsets[has_ref] - 1, side="right"
        ) + 1
    )
    start_ref[start_ref > len(day_ints_arr)] = 9999

    onset = np.full((3, len(df)), np.nan)
    valid_threshold = ~np.isnan(th)
    if np.any(valid_threshold):
        onset[:, valid_threshold] = find_onsets_batch(
            df.loc[valid_threshold, day_cols].to_numpy(dtype=float),
            th[valid_threshold],
            np.vstack((
                np.ones(valid_threshold.sum(), dtype=int),
                need_clim[valid_threshold],
                start_ref[valid_threshold],
            )),
            params,
        )

    return {
        "onset_raw": onset[0],
        "onset_fixed_cutoff": onset[1],
        "onset_ref": onset[2],
    }


def _aggregate_forecast_members(df, key_cols, rain_day_cols, rain_day_ints,
                                probability_day_ints):
    """Aggregate member rainfall summaries and onset-day probabilities."""
    grouped = df.groupby(key_cols, dropna=False, sort=True)
    group_index = grouped.size().index
    group_codes = grouped.ngroup().to_numpy(dtype=np.intp)
    n_groups = len(group_index)

    rain = df[rain_day_cols]
    rain_mean = rain.groupby(group_codes, sort=True).mean()
    rain_sd = rain.groupby(group_codes, sort=True).std()
    frac_raining = rain.gt(1).groupby(group_codes, sort=True).mean()

    rain_mean.columns = [f"forecast_rain_day_{day}" for day in rain_day_ints]
    rain_sd.columns = [f"forecast_rain_sd_day_{day}" for day in rain_day_ints]
    frac_raining.columns = [f"frac_raining_day_{day}" for day in rain_day_ints]

    group_sizes = np.bincount(group_codes, minlength=n_groups).astype(float)
    probability_frames = []
    D = len(probability_day_ints)
    for onset_col, prefix in (
        ("onset_raw", "predicted_prob_day_"),
        ("onset_fixed_cutoff", "predicted_prob_fixed_cutoff_day_"),
        ("onset_ref", "predicted_prob_ref_day_"),
    ):
        onset = pd.to_numeric(df[onset_col], errors="coerce").to_numpy(float)
        onset_pos = np.zeros(len(onset), dtype=np.intp)
        finite = np.isfinite(onset)
        onset_pos[finite] = onset[finite].astype(np.intp)
        valid = finite & (onset >= 1) & (onset <= D)

        counts = np.zeros((n_groups, D), dtype=float)
        np.add.at(
            counts,
            (group_codes[valid], onset_pos[valid] - 1),
            1.0,
        )
        probability_frames.append(pd.DataFrame(
            counts / group_sizes[:, None],
            columns=[f"{prefix}{day}" for day in probability_day_ints],
        ))

    parts = [rain_mean, rain_sd, frac_raining, *probability_frames]
    for part in parts:
        part.index = group_index
    return pd.concat(parts, axis=1).reset_index()


def process_rainfall_forecast_id(df, spec, ref_onset_dt=None, thr_dt=None):
    """
    Forecast pipeline: compute ensemble onset probabilities per (id, time, year).

    Returns dict: {"wide": DataFrame}
    """
    df = df.copy()
    df = ensure_id_col(df, spec=spec, context="forecast spatial IDs")
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"]).dt.date
    if "year" in df.columns:
        df["year"] = df["year"].astype(int)
    if "number" in df.columns:
        df["number"] = df["number"].astype(int)

    has_number = "number" in df.columns
    filter_cfg = spec.get("filter") or {}
    max_number = filter_cfg.get("max_number")

    df = attach_thresholds_id(df, thr_dt, spec=spec)
    df = attach_ref_onset(df, ref_onset_dt)

    if has_number and max_number is not None:
        df = df[df["number"] <= int(max_number)]
        if df.empty:
            raise ValueError(f"After filtering number <= {max_number}, no rows remain.")

    wide_prefix = spec.get("input", {}).get("wide_prefix") or spec["input"]["value_col"].lower()
    strict_rain_day_max = spec.get("options", {}).get("rain_day_max")
    if strict_rain_day_max is not None:
        _, horizon_policy = resolve_rain_day_max(
            {
                "rain_day_max": strict_rain_day_max,
                "rain_horizon_policy": spec["options"].get(
                    "rain_horizon_policy"
                ),
            },
            probability_day_max=1,
        )
        validate_rain_horizon_frame(
            df,
            day_prefix=f"{wide_prefix}_day_",
            rain_day_max=strict_rain_day_max,
            model_name=spec.get("id", wide_prefix),
            strict=True,
            allow_extra=horizon_policy == "truncate",
            key_columns=("id", "time", "year", "number"),
            context="raw forecast before ensemble aggregation",
        )

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
    key_base += ["onset_thresh", "ref_onset_date"]
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

    fixed_cutoff_md = spec.get("options", {}).get("fixed_cutoff_month_day", "06-02")
    on = calc_onsets_rowwise(df, keep_days, keep_ints, win, params=onset_params,
                             fixed_cutoff_month_day=fixed_cutoff_md)
    df["onset_raw"] = on["onset_raw"]
    df["onset_fixed_cutoff"] = on["onset_fixed_cutoff"]
    df["onset_ref"] = on["onset_ref"]

    # Preserve the historical rainfall-column labels based on day_ints. In the
    # normal pipeline the NetCDF reader has already applied min/max-day
    # filtering, so these align with keep_ints.
    rain_day_ints = day_ints[:len(keep_days)]
    wide = _aggregate_forecast_members(
        df,
        key_cols=key_base,
        rain_day_cols=keep_days,
        rain_day_ints=rain_day_ints,
        probability_day_ints=keep_ints,
    )
    return {"wide": wide}


def process_ground_truth_rainfall_id(df, spec, ref_onset_dt=None, thr_dt=None, value_col=None):
    """
    Ground-truth pipeline: compute onset dates per (id, year).

    Returns dict: {"wide": DataFrame, "long": DataFrame or None}. The
    annotated daily-long table is omitted when output.write_long is false.
    """
    df = df.copy()
    df = ensure_id_col(df, spec=spec, context="ground-truth spatial IDs")
    df["time"] = pd.to_datetime(df["time"]).dt.date
    df["year"] = df["year"].astype(int)
    df[value_col] = df[value_col].astype(float)

    df = attach_thresholds_id(df, thr_dt, spec=spec)
    df = attach_ref_onset(df, ref_onset_dt)

    cutoff_md = spec["options"]["cutoff_month_day"]
    df["cutoff_date"] = df["year"].apply(lambda y: pd.Timestamp(f"{y}-{cutoff_md}").date())
    df["start_date"] = df["cutoff_date"]
    mask = ~df["ref_onset_date"].isna()
    df.loc[mask, "start_date"] = df.loc[mask].apply(
        lambda r: max(r["cutoff_date"], r["ref_onset_date"]), axis=1
    )

    df = df.sort_values(["id", "year", "time"])
    #win = int(spec["options"]["window"])
    onset_params = read_onset_params(spec)
    win = onset_params.win
    write_long = spec.get("output", {}).get("write_long", True)

    wide_rows = []
    long_rows = [] if write_long else None

    for (cell_id, yr), g in df.groupby(["id", "year"]):
        series = g[value_col].values
        dates = g["time"].values
        sd = g["start_date"].iloc[0]
        th = g["onset_thresh"].iloc[0]

        start_pos_arr = np.where(pd.to_datetime(dates) >= pd.Timestamp(sd))[0]
        start_pos = int(start_pos_arr[0]) + 1 if len(start_pos_arr) > 0 else 9999

        if np.all(np.isnan(series)) or len(series) < win or np.isnan(th):
            onset_idx = None
            onset_date = None
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

            #onset_idx = find_onset_precomp(series, win, th, wsum_all, pre_bad, last10start,
            #                            start_day=start_pos, reject_if_short_followup=True)
            onset_idx = find_onset_precomp(series, win, th, wsum_all, pre_bad, last10start,
                            start_day=start_pos, reject_if_short_followup=True,
                            params=onset_params)
            onset_date = dates[onset_idx - 1] if onset_idx is not None else None

        cutoff_date = g["cutoff_date"].iloc[0]
        if onset_date is not None:
            onset_day = (pd.Timestamp(onset_date) - pd.Timestamp(cutoff_date)).days
        else:
            onset_day = np.nan

        wide_rows.append({
            "id": cell_id,
            "year": yr,
            "onset_idx": onset_idx if onset_idx is not None else np.nan,
            "onset_date": onset_date,
            "onset_day": onset_day,
            "cutoff_date": cutoff_date,
        })

        if write_long:
            ref_onset_date = g["ref_onset_date"].iloc[0]
            for _, row in g.iterrows():
                long_rows.append({
                    "id": cell_id,
                    "time": row["time"],
                    "year": yr,
                    value_col: row[value_col],
                    "onset_thresh": th,
                    "ref_onset_date": ref_onset_date,
                    "onset_date": onset_date,
                    "onset_flag": row["time"] == onset_date,
                })

    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows) if write_long else None
    return {"wide": wide, "long": long}


# ---------------------------------------------------------------------------
# Pipeline entrypoint
# ---------------------------------------------------------------------------

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
    if spec["type"] == "ground_truth_rainfall" and not spec["output"].get("write_long", True):
        print("Ground-truth long output disabled by output.write_long: false")

    var_name = get_value_var(spec)
    dim_rename_map = spec.get("dimensions", {}).get("rename") or {}
    ref_onset_dt = read_ref_onset_dates(spec)
    thr_dt = read_thresholds(spec)
    weights_df = read_cell_transform(spec)

    files_df = list_nc_files_with_year(spec)

    wide_all = []
    long_all = []

    for _, row in files_df.iterrows():
        nc_path = row["nc_path"]
        yr = row["year"]
        print(f"Processing year {yr}: {nc_path}")

        if spec["type"] == "rainfall_forecast":
            wide_prefix = spec["input"].get("wide_prefix") or var_name.lower()
            day_dim = spec["input"].get("wide_day_dim", "day")
            dt = nc_read_forecast_wide(nc_path, var_name, dim_rename_map, spec,
                                       day_dim=day_dim, prefix=wide_prefix)
            if dt is None:
                print(f"  Skipping {nc_path}: variable '{var_name}' not found.")
                continue
            dt["year"] = yr
            if weights_df is not None: # in spec yml if cell_transform_enabled: false, it does nothing
                dt = transform_forecast_wide(dt, weights_df, spec=spec)
            dt = filter_by_dissemination_cells(dt, spec)
            result = process_rainfall_forecast_id(dt, spec, ref_onset_dt=ref_onset_dt, thr_dt=thr_dt)
            wide_all.append(result["wide"])

        elif spec["type"] == "ground_truth_rainfall":
            missing_rain_policy = (
                spec.get("options", {}).get("missing_rain_policy")
                or spec.get("input", {}).get("missing_rain_policy")
                or "keep"
            )
            dt = nc_read_groundtruth_long(
                nc_path, var_name, dim_rename_map,
                missing_rain_policy=missing_rain_policy,
            )
            if dt is None:
                print(f"  Skipping {nc_path}: variable '{var_name}' not found.")
                continue
            dt["year"] = yr
            if weights_df is not None: # in spec yml if cell_transform_enabled: false, it does nothing
                dt = transform_groundtruth_long(
                    dt, weights_df, var_name.lower(), spec=spec
                )
            dt = filter_by_dissemination_cells(dt, spec)
            result = process_ground_truth_rainfall_id(dt, spec, ref_onset_dt=ref_onset_dt, thr_dt=thr_dt,
                                                       value_col=var_name.lower())
            wide_all.append(result["wide"])
            if result["long"] is not None:
                long_all.append(result["long"])

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
