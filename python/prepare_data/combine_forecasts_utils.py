# ==============================================================================
# File: combine_forecasts_utils.py
# ==============================================================================
# Purpose
#   Utilities for combining forecast products into a single wide table
#   suitable for modeling.
#
# Functions
#   parse_years(x)
#   detect_plus_bin(day_cols)
#   normalize_plus_bin(df, prefix)
#   read_ground_truth_wide(path)
#   read_and_format_climatology_wide(path, out_prefix)
#   read_one_forecast_source(source, daily_cols, const_cols)
#   format_forecast_family(forecast_name, conf)
# ==============================================================================

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd

from ..pipelines._shared.misc import coalesce


# ---------------------------------------------------------------------------
# Year parsing
# ---------------------------------------------------------------------------

def parse_years(x):
    """
    Parse a years spec string into a list of integers.
    Accepts comma-separated years and ranges like "2001,2003:2007".
    Returns empty list if input is None/empty (meaning no filter).
    """
    if x is None or x == "":
        return []
    x = str(x).replace(" ", "")
    parts = x.split(",")
    out = []
    for p in parts:
        if not p:
            continue
        if ":" in p:
            a, b = p.split(":", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return list(set(out))


# ---------------------------------------------------------------------------
# Plus-bin normalization
# ---------------------------------------------------------------------------

def detect_plus_bin(day_cols):
    """Find columns ending with 'plus'. Error if more than one found."""
    plus = [c for c in day_cols if c.endswith("plus")]
    if len(plus) > 1:
        raise ValueError(f"Multiple plus-bin columns found: {', '.join(plus)}")
    return plus


def normalize_plus_bin(df, prefix):
    """
    Ensure a plus-bin column exists and compute its value as max(0, 1 - sum(days)).
    """
    day_pat = re.compile(rf"^{re.escape(prefix)}_day_")
    day_cols = [c for c in df.columns if day_pat.match(c)]
    if not day_cols:
        return df

    plus_candidates = detect_plus_bin(day_cols)

    if not plus_candidates:
        nums = []
        for c in day_cols:
            m = re.search(r"_day_(\d+)$", c)
            if m:
                nums.append(int(m.group(1)))
        if not nums:
            raise ValueError(f"Cannot infer day numbers for prefix {prefix}")
        max_day = max(nums)
        plus_col = f"{prefix}_day_{max_day + 1}plus"
        df = df.copy()
        df[plus_col] = np.nan
        day_cols = day_cols + [plus_col]
        plus_candidates = [plus_col]

    plus_col = plus_candidates[0]
    base_cols = [c for c in day_cols if c != plus_col]

    df = df.copy()
    row_sums = df[base_cols].sum(axis=1, skipna=True)
    df[plus_col] = np.maximum(0.0, 1.0 - row_sums)
    return df


# ---------------------------------------------------------------------------
# Ground truth (id-based)
# ---------------------------------------------------------------------------

def read_ground_truth_wide(path):
    """Read stage-2 ground-truth wide pickle. Returns DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    with open(path, "rb") as f:
        dt = pickle.load(f)
    if not isinstance(dt, pd.DataFrame):
        dt = pd.DataFrame(dt)

    req = ["id", "year"]
    miss = [c for c in req if c not in dt.columns]
    if miss:
        raise ValueError(f"Ground truth missing columns: {', '.join(miss)}")

    true_onset_day = pd.to_numeric(dt["onset_day"], errors="coerce") if "onset_day" in dt.columns else np.nan
    true_onset_date = pd.to_datetime(dt["onset_date"]).dt.date if "onset_date" in dt.columns else None

    result = pd.DataFrame({
        "id": dt["id"].astype(str),
        "year": dt["year"].astype(int),
        "true_onset_day": true_onset_day,
        "true_onset_date": true_onset_date if true_onset_date is not None else pd.NaT,
    })
    return result


# ---------------------------------------------------------------------------
# Climatology WIDE (id-based)
# ---------------------------------------------------------------------------

def read_and_format_climatology_wide(path, out_prefix="clim_p_onset"):
    """
    Read climatology wide pickle. Rename predicted_prob_day_<k> -> <out_prefix>_day_<k>.
    Adds normalized plus-bin column.
    Returns DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Climatology file not found: {path}")
    with open(path, "rb") as f:
        dt = pickle.load(f)
    if not isinstance(dt, pd.DataFrame):
        dt = pd.DataFrame(dt)

    key_req = ["id", "time", "year"]
    miss_keys = [c for c in key_req if c not in dt.columns]
    if miss_keys:
        raise ValueError(f"Climatology missing key columns: {', '.join(miss_keys)}")

    if "day" in dt.columns:
        raise ValueError(f"Climatology must be WIDE (no 'day' column). Found 'day' in: {path}")

    prob_cols = [c for c in dt.columns if re.match(r"^predicted_prob_day_\d+$", c)]
    if not prob_cols:
        raise ValueError(f"No predicted_prob_day_<k> columns found in: {path}")

    prob_cols = sorted(prob_cols, key=lambda c: int(re.search(r"\d+$", c).group()))

    dt = dt.copy()
    dt["id"] = dt["id"].astype(str)
    dt["time"] = pd.to_datetime(dt["time"]).dt.date
    dt["year"] = dt["year"].astype(int)

    for cc in prob_cols:
        dt[cc] = pd.to_numeric(dt[cc], errors="coerce")

    new_names = {c: c.replace("predicted_prob_day_", f"{out_prefix}_day_") for c in prob_cols}
    dt = dt.rename(columns=new_names)

    keep = key_req + list(new_names.values())
    dt = dt[keep]
    dt = normalize_plus_bin(dt, out_prefix)
    return dt


# ---------------------------------------------------------------------------
# Forecast sources (id-based)
# ---------------------------------------------------------------------------

def read_one_forecast_source(source, daily_cols, const_cols):
    """
    Read a single forecast wide pickle and optionally filter by years.

    Returns DataFrame.
    """
    file_path = str(source["file"])
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Forecast file not found: {file_path}")

    yrs = parse_years(source.get("years", ""))

    with open(file_path, "rb") as f:
        dt = pickle.load(f)
    if not isinstance(dt, pd.DataFrame):
        dt = pd.DataFrame(dt)

    key_req = ["id", "time", "year"]
    miss_keys = [c for c in key_req if c not in dt.columns]
    if miss_keys:
        raise ValueError(f"Forecast file missing key columns: {', '.join(miss_keys)}\nFile: {file_path}")

    dt["id"] = dt["id"].astype(str)
    dt["time"] = pd.to_datetime(dt["time"]).dt.date
    dt["year"] = dt["year"].astype(int)

    if yrs:
        dt = dt[dt["year"].isin(yrs)]

    miss_const = [c for c in const_cols if c not in dt.columns]
    if miss_const:
        raise ValueError(f"Forecast wide file missing constant columns: {', '.join(miss_const)}\nFile: {file_path}")

    all_day_cols = []
    for dv in daily_cols:
        pat = re.compile(rf"^{re.escape(dv)}_day_\d+$")
        dcols = [c for c in dt.columns if pat.match(c)]
        if not dcols:
            raise ValueError(f"No day columns for daily var '{dv}' in {file_path}")
        all_day_cols.extend(dcols)

    keep = list(dict.fromkeys(key_req + list(const_cols) + all_day_cols))
    dt = dt[[c for c in keep if c in dt.columns]]

    for cc in all_day_cols:
        dt[cc] = pd.to_numeric(dt[cc], errors="coerce")

    return dt


# ---------------------------------------------------------------------------
# Forecast family formatter (id-based)
# ---------------------------------------------------------------------------

def format_forecast_family(forecast_name, conf, plus_label_template=None):
    """
    Read and bind all WIDE sources for one forecast family.

    Returns dict: {"daily": DataFrame, "constants": DataFrame}
    """
    max_day = int(conf["max_day"])
    if max_day <= 0:
        raise ValueError(f"Forecast '{forecast_name}' must define a positive max_day.")

    if not conf.get("daily"):
        raise ValueError(f"Forecast '{forecast_name}' must provide a non-empty daily list.")

    daily = [
        {"col": str(x["col"]), "out": str(x["out"]), "add_plus": bool(x.get("add_plus", False))}
        for x in conf["daily"]
    ]
    constants = [
        {"col": str(x["col"]), "out": str(x["out"])}
        for x in (conf.get("constants") or [])
    ]

    cols_daily = list(dict.fromkeys(d["col"] for d in daily))
    cols_const = list(dict.fromkeys(c["col"] for c in constants))

    parts = [
        read_one_forecast_source(src, daily_cols=cols_daily, const_cols=cols_const)
        for src in conf["sources"]
    ]
    dt = pd.concat(parts, ignore_index=True)

    if "day" in dt.columns:
        raise ValueError(f"Forecast '{forecast_name}' produced a 'day' column; forecasts must be wide-only.")

    key_cols = ["id", "time", "year"]

    dup_key = dt.duplicated(subset=key_cols)
    if dup_key.any():
        warnings.warn(f"Forecast '{forecast_name}' has overlapping rows; keeping first occurrence.")
        dt = dt.drop_duplicates(subset=key_cols, keep="first")

    # Constants
    if constants:
        const_cols_names = [c["col"] for c in constants]
        const_dt = dt[key_cols + const_cols_names].drop_duplicates()
        const_rename = {c["col"]: f"{forecast_name}_{c['out']}" for c in constants}
        const_dt = const_dt.rename(columns=const_rename)
    else:
        const_dt = dt[key_cols].drop_duplicates()

    # Daily outputs
    daily_wide = dt[key_cols].drop_duplicates()

    for d in daily:
        src = d["col"]
        out = d["out"]
        #print(f"    Processing variable: {src} -> {out}")
        pat = re.compile(rf"^{re.escape(src)}_day_")
        dcols = [c for c in dt.columns if pat.match(c)]
        if not dcols:
            raise ValueError(f"Forecast '{forecast_name}' missing day columns for {src}")

        bad = [c for c in dcols if re.search(r"_day_\d+$", c)]
        if bad:
            ks = [int(re.search(r"_day_(\d+)$", c).group(1)) for c in bad]
            #print("d = ", d)
            #print("ks = ", ks)
            #print("max_day - ", max_day)
            if max(ks) > max_day:
                raise ValueError(f"Forecast '{forecast_name}' exceeds max_day for {src}")

        tmp = dt[key_cols + dcols].copy()
        new_names = {c: c.replace(f"{src}_day_", f"{forecast_name}_{out}_day_") for c in dcols}
        tmp = tmp.rename(columns=new_names)

        daily_wide = daily_wide.merge(tmp, on=key_cols, how="outer")

        if d["add_plus"]:
            daily_wide = normalize_plus_bin(daily_wide, f"{forecast_name}_{out}")

    return {"daily": daily_wide, "constants": const_dt}
