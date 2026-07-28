# ==============================================================================
# File: climatology_utils.py
# ==============================================================================
# Purpose
#   Utilities for fitting per-cell KDE climatology models of monsoon onset day
#   and producing issue-date probability forecasts over lead days.
#
# Workflow
#   1) get_paths_clim(spec)
#   2) get_climatology_options_from_run(co)
#   3) read_gt_onset_from_tbl(gt_tbl, onset_col)
#   4) filter_gt_training(gt, y_min, y_max)
#   5) season_dates_for_year(year, start_md, end_md)
#   6) build_issue_grid(test_year_min, test_year_max, season_start_md, issue_end_md)
#   7) resolve_forecast_window_by_time(time, forecast_window, horizons)
#   8) fit_kde(x)
#   9) compute_d0(time, season_start_md)
#  10) predict_from_kde(dens, d0, forecast_window, conditional, include_day0)
#  11) fit_kdes_by_cell(gt_train)
#  12) compute_forecasts_for_cell(...)
#  13) compute_all_forecasts(...)
# ==============================================================================

import os
from functools import lru_cache
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import date
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

from python.pipelines._shared.misc import coalesce


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_paths_clim(spec):
    """
    Derive input ground-truth path and output directory/stem for climatology.

    Returns dict: gt_path, out_dir, out_stem
    """
    # Use explicit gt_path from yml if provided, otherwise derive from spec_id
    gt_path = (
        spec.get("input", {}).get("gt_path")
        or os.path.join(spec["output"]["out_dir"], f"{spec['id']}_wide.pkl")
    )
    #gt_path = os.path.join(spec["output"]["out_dir"], f"{spec['id']}_wide.pkl")
    out_dir = (
        spec.get("paths", {}).get("climatology_out_dir")
        or os.path.join(os.path.dirname(spec["output"]["out_dir"]), "Climatology")
    )
    out_stem = spec.get("paths", {}).get("climatology_out_stem") or "climatology_issue"
    os.makedirs(out_dir, exist_ok=True)
    return {"gt_path": gt_path, "out_dir": out_dir, "out_stem": out_stem}


def get_climatology_options_from_run(co):
    """Extract climatology options for a single run entry."""
    return {
        "train_year_min": int(co["train_year_min"]),
        "train_year_max": int(co["train_year_max"]),
        "test_year_min": int(co["test_year_min"]),
        "test_year_max": int(co["test_year_max"]),
        "season_start_md": str(co["season_start_md"]),
        "issue_end_md": str(co["issue_end_md"]),
        "onset_col": str(co.get("onset_col") or "onset_day"),
        "forecast_window": int(co["forecast_window"]) if co.get("forecast_window") is not None else None,
        "horizons": co.get("horizons"),
        "conditional": bool(co["conditional"]) if co.get("conditional") is not None else True,
        "cv_by_year": bool(co["cv_by_year"]) if co.get("cv_by_year") is not None else True,
    }


# ---------------------------------------------------------------------------
# Ground-truth IO
# ---------------------------------------------------------------------------

def read_gt_onset_from_tbl(gt_tbl, onset_col="onset_day", na_sentinel=None):
    """
    Read and standardize ground-truth onset data from a loaded wide table.

    Returns DataFrame: id (str), year (int), onset_day (int or NaN).
    """
    if onset_col not in gt_tbl.columns:
        raise ValueError(f"Missing onset column '{onset_col}' in ground-truth table.")
    if "id" not in gt_tbl.columns:
        raise ValueError("Missing required column 'id' in ground-truth table.")
    if "year" not in gt_tbl.columns:
        raise ValueError("Missing required column 'year' in ground-truth table.")

    out = pd.DataFrame({
        "id": gt_tbl["id"].astype(str),
        "year": gt_tbl["year"].astype(int),
        "onset_day": pd.to_numeric(gt_tbl[onset_col], errors="coerce"),
    })
    if na_sentinel is not None:
        out["onset_day"] = out["onset_day"].fillna(int(na_sentinel))
    return out


def filter_gt_training(gt, y_min, y_max):
    """Filter ground-truth table to [y_min, y_max] and drop missing onset_day."""
    return gt[gt["onset_day"].notna() & (gt["year"] >= y_min) & (gt["year"] <= y_max)].copy()


# ---------------------------------------------------------------------------
# Issue-date grid
# ---------------------------------------------------------------------------

def season_dates_for_year(year, start_md, end_md):
    """Return list of dates from YYYY-start_md to YYYY-end_md inclusive."""
    start = pd.Timestamp(f"{year}-{start_md}")
    end = pd.Timestamp(f"{year}-{end_md}")
    if pd.isna(start) or pd.isna(end):
        raise ValueError(f"Bad season dates; expected 'MM-DD'. Got: {start_md} / {end_md}")
    if end < start:
        raise ValueError(f"issue_end_md is before season_start_md for year {year}")
    return pd.date_range(start, end, freq="D").date.tolist()


def build_issue_grid(test_year_min, test_year_max, season_start_md, issue_end_md):
    """Build full grid of issue dates across test years."""
    rows = []
    for y in range(int(test_year_min), int(test_year_max) + 1):
        for d in season_dates_for_year(y, season_start_md, issue_end_md):
            rows.append({"year": int(y), "time": d})
    df = pd.DataFrame(rows).drop_duplicates().sort_values(["year", "time"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Horizon H(time)
# ---------------------------------------------------------------------------

def max_forecast_window(forecast_window, horizons):
    if horizons is not None:
        return max(int(h["forecast_window"]) for h in horizons)
    return int(forecast_window)


def resolve_forecast_window_by_time(t, forecast_window, horizons):
    """Return integer forecast horizon for a given issue date."""
    if horizons is None:
        return int(forecast_window)
    yr = pd.Timestamp(t).year
    for h in horizons:
        start = pd.Timestamp(f"{yr}-{h['start_md']}").date()
        end = pd.Timestamp(f"{yr}-{h['end_md']}").date()
        if start <= t <= end:
            return int(h["forecast_window"])
    return None


# ---------------------------------------------------------------------------
# KDE + forecast math
# ---------------------------------------------------------------------------

def fit_kde(x):
    """Fit a 1D KDE to onset_day samples using Scott/SJ bandwidth."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return None
    try:
        kde = gaussian_kde(x, bw_method="scott")
        return kde

        # Sheather-Jones bandwidth, matching R's bw="SJ"
        #from statsmodels.nonparametric.bandwidths import bw_silverman, select_bandwidth
        #bw = select_bandwidth(x, bw="sheather jones", kernel=None)
        #kde = gaussian_kde(x, bw_method=bw / np.std(x, ddof=1))
        #return kde

    except Exception:
        return None



#def fit_kde(x):
#    from KDEpy import FFTKDE
#    x = np.asarray(x, dtype=float)
#    x = x[~np.isnan(x)]
#    if len(x) < 10:
#        return None
#    try:
#        # 1. Fit using KDEpy
#        estimator = FFTKDE(bw="ISJ").fit(x)
#        # 2. Evaluate on a grid
#        grid, points = estimator.evaluate(1024)
#
#        # 3. Return a lambda that mimics SciPy's callability
#        # This uses interpolation so you can still call it like kde(5.5)
#        return lambda query_points: np.interp(query_points, grid, points, left=0, right=0)
#    except:
#        return None


#def _bw_sj(x):
#    from scipy.optimize import brentq
#    n = len(x)
#    std_x = np.std(x, ddof=1)
#    iqr_x = (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349
#    scale = min(std_x, iqr_x) if iqr_x > 0 else std_x
#    nb = 1000
#    bin_width = (x.max() - x.min()) * 1.01 / nb
#    i_idx, j_idx = np.triu_indices(n, k=1)
#    diffs = np.abs(x[i_idx] - x[j_idx])
#    v = diffs / bin_width
#    k = np.floor(v).astype(int)
#    frac = v - k
#    cnt = np.zeros(nb, dtype=float)
#    for i in range(len(diffs)):
#        k0 = k[i]
#        f = frac[i]
#        if k0 < nb:
#            cnt[k0] += (1.0 - f)
#        if k0 + 1 < nb:
#            cnt[k0 + 1] += f
#    d = np.arange(nb, dtype=float) * bin_width
#    def _phi4(h):
#        t = d / h
#        phi = (t**4 - 6*t**2 + 3) * np.exp(-0.5 * t**2)
#        return (3.0 * n + 2.0 * np.dot(cnt, phi)) / (n**2 * h**5 * np.sqrt(2 * np.pi))
#    def _phi6(h):
#        t = d / h
#        phi = (t**6 - 15*t**4 + 45*t**2 - 15) * np.exp(-0.5 * t**2)
#        return (-15.0 * n + 2.0 * np.dot(cnt, phi)) / (n**2 * h**7 * np.sqrt(2 * np.pi))
#    a = 1.24 * scale * n**(-1/7)
#    b = 1.23 * scale * n**(-1/9)
#    c1 = 1.0 / (2 * np.sqrt(np.pi) * n)
#    hmax = 1.144 * scale * n**(-1/5)
#    TD = -_phi6(b)
#    if not np.isfinite(TD) or TD <= 0:
#        raise ValueError(f"TD failed: {TD}")
#    SDh_a = _phi4(a)
#    alph2 = 1.357 * (SDh_a / TD) ** (1/7)
#    if not np.isfinite(alph2):
#        raise ValueError(f"alph2 failed: {alph2}")
#    def fSD(h):
#        sdh = _phi4(alph2 * h**(5/7))
#        if sdh <= 0 or not np.isfinite(sdh):
#            return -h
#        return (c1 / sdh) ** 0.2 - h
#    lower = 0.1 * hmax
#    upper = hmax
#    for _ in range(99):
#        if fSD(lower) * fSD(upper) <= 0:
#            break
#        upper *= 1.2
#        lower /= 1.2
#    else:
#        raise ValueError("No solution in bandwidth search interval.")
#    return brentq(fSD, lower, upper, xtol=0.1 * lower)
#
#
#def fit_kde(x):
#    """Fit a 1D KDE using SJ bandwidth matching R's density(bw='SJ')."""
#    x = np.asarray(x, dtype=float)
#    x = x[~np.isnan(x)]
#    if len(x) < 10:
#        return None
#    try:
#        bw_h = _bw_sj(x)
#        std_x = np.std(x, ddof=1)
#        return gaussian_kde(x, bw_method=bw_h / std_x)
#    except Exception as e:
#        warnings.warn(f"SJ bandwidth failed ({e}), falling back to Silverman.")
#        try:
#            return gaussian_kde(x, bw_method="silverman")
#        except Exception:
#            return None
#
#
#def _kde_cdf(kde, x_vals):
#    """
#    Approximate CDF matching R's stats::density() + approxfun(rule=2).
#    512 grid points, domain = data +/- 3*bw, boundary fill (not 0/1).
#    """
#    data = kde.dataset.flatten()
#    bw = kde.factor * np.std(data, ddof=1)
#    x_min = data.min() - 3 * bw
#    x_max = data.max() + 3 * bw
#    grid = np.linspace(x_min, x_max, 512)
#    pdf_vals = kde.evaluate(grid)
#    cdf_vals = np.cumsum(pdf_vals) / np.sum(pdf_vals)
#    # rule=2: use boundary values for out-of-range (matches R's approxfun)
#    return interp1d(grid, cdf_vals, bounds_error=False,
#                    fill_value=(cdf_vals[0], cdf_vals[-1]))


def compute_d0(t, season_start_md):
    """Convert issue date to integer offset d0 from season start."""
    yr = pd.Timestamp(t).year
    season_start = pd.Timestamp(f"{yr}-{season_start_md}").date()
    return (pd.Timestamp(t).date() - season_start).days


@lru_cache(maxsize=512)
def _kde_cdf(kde, x_vals=None):
    """Approximate R density()+approxfun(rule=2) CDF construction."""
    data = kde.dataset.flatten()
    bandwidth = kde.factor * np.std(data, ddof=1)
    x_min = data.min() - 3 * bandwidth
    x_max = data.max() + 3 * bandwidth
    grid = np.linspace(x_min, x_max, 512)
    pdf_vals = kde.evaluate(grid)
    cdf_vals = np.cumsum(pdf_vals) / np.sum(pdf_vals)
    cdf_fn = interp1d(
        grid,
        cdf_vals,
        bounds_error=False,
        fill_value=(cdf_vals[0], cdf_vals[-1]),
    )
    return cdf_fn



MIN_PROB = 5e-7
EPS = 1e-12


def _enforce_floor_with_target_sum(p_raw, target_sum, lb):
    H = len(p_raw)
    if not np.isfinite(target_sum) or target_sum < 0:
        target_sum = 0.0
    if target_sum > 1:
        target_sum = 1.0
    p_raw = np.maximum(p_raw, 0.0)
    p_raw = np.where(np.isfinite(p_raw), p_raw, 0.0)

    if target_sum <= EPS:
        return np.zeros(H)
    if target_sum < H * lb:
        return np.full(H, target_sum / H)

    p = np.maximum(p_raw, lb)
    excess = np.sum(p) - target_sum
    if excess <= EPS:
        return p
    reducible = p - lb
    reducible_sum = np.sum(reducible)
    if reducible_sum <= EPS:
        return np.full(H, target_sum / H)
    p = p - excess * (reducible / reducible_sum)
    return np.maximum(p, lb)


def predict_from_kde(kde, d0, forecast_window, conditional=True, include_day0=False,
                     min_prob=MIN_PROB, eps=EPS):
    """
    Produce a length-H probability vector from a fitted KDE.

    conditional=True:  P(onset on d0+k | onset > d0)
    conditional=False: unconditional day-mass values
    """
    out_len = forecast_window + (1 if include_day0 and not conditional else 0)
    if kde is None:
        return np.full(out_len, np.nan)

    cdf = _kde_cdf(kde, None)
    days = np.arange(1, forecast_window + 1)

    num = cdf(d0 + days) - cdf(d0 + days - 1)
    num = np.maximum(num, 0.0)
    num = np.where(np.isfinite(num), num, 0.0)

    if not conditional:
        target_sum = np.sum(num)
        p_adj = _enforce_floor_with_target_sum(num, target_sum, lb=min_prob)
        if include_day0:
            day0_mass = float(cdf(d0))
            return np.concatenate([[day0_mass], p_adj])
        return p_adj

    base_prob = float(cdf(d0))
    denom = 1.0 - base_prob

    if not np.isfinite(denom) or denom <= eps:
        target_sum = np.sum(num)
        return _enforce_floor_with_target_sum(num, target_sum, lb=min_prob)

    p_raw = num / denom
    p_raw = np.maximum(p_raw, 0.0)
    p_raw = np.where(np.isfinite(p_raw), p_raw, 0.0)

    target_sum = np.sum(p_raw)
    if not np.isfinite(target_sum) or target_sum <= eps:
        return np.zeros(forecast_window)

    return _enforce_floor_with_target_sum(p_raw, target_sum, lb=min_prob)


# ---------------------------------------------------------------------------
# KDEs by cell
# ---------------------------------------------------------------------------

def fit_kdes_by_cell(gt_train):
    """Fit KDE for each cell id in gt_train. Returns dict {id: kde}."""
    kdes = {}
    for cell_id, g in gt_train.groupby("id"):
        kdes[str(cell_id)] = fit_kde(g["onset_day"].values)
    return kdes


# ---------------------------------------------------------------------------
# Per-cell forecasts
# ---------------------------------------------------------------------------

def compute_forecasts_for_cell(cell_id, issue_grid, kdes,
                                season_start_md, forecast_window, horizons,
                                conditional=True, cv_by_year=False, gt_train=None):
    """
    Compute predicted probabilities for a single cell over all issue dates.

    Returns DataFrame: time, year, id, model, predicted_prob_day_1..N
    """
    cell_key = str(cell_id)
    max_H = max_forecast_window(forecast_window, horizons)
    include_day0 = not conditional
    max_cols = max_H + (1 if include_day0 else 0)

    dens_static = None
    dens_by_year = {}

    if not cv_by_year:
        dens_static = kdes.get(cell_key) if kdes else None
    else:
        if gt_train is None:
            raise ValueError("cv_by_year=True requires gt_train.")
        years_needed = sorted(issue_grid["year"].unique())
        gt_id = gt_train[gt_train["id"] == cell_key]
        for y in years_needed:
            x = gt_id[gt_id["year"] != y]["onset_day"].values
            dens_by_year[str(y)] = fit_kde(x)

    if cv_by_year and conditional:
        model_label = "clim_kde_cv"
    elif cv_by_year and not conditional:
        model_label = "clim_kde_unc_cv"
    elif not cv_by_year and conditional:
        model_label = "clim_kde"
    else:
        model_label = "clim_kde_unc"

    rows = []
    for _, row in issue_grid.iterrows():
        t = row["time"]
        yr = int(row["year"])
        d0 = compute_d0(t, season_start_md)
        H = resolve_forecast_window_by_time(t, forecast_window, horizons)

        probs = np.full(max_cols, np.nan)
        if H is not None and H > 0:
            dens_use = dens_static if not cv_by_year else dens_by_year.get(str(yr))
            p = predict_from_kde(dens_use, d0, H, conditional=conditional, include_day0=include_day0)
            length = min(len(p), max_cols)
            probs[:length] = p[:length]

        entry = {"time": t, "year": yr, "id": cell_key, "model": model_label}
        if include_day0:
            entry["predicted_prob_day_0"] = probs[0]
            for k in range(1, max_H + 1):
                entry[f"predicted_prob_day_{k}"] = probs[k] if k < max_cols else np.nan
        else:
            for k in range(1, max_H + 1):
                entry[f"predicted_prob_day_{k}"] = probs[k - 1] if k - 1 < max_cols else np.nan
        rows.append(entry)

    return pd.DataFrame(rows)


def compute_all_forecasts(gt_train, issue_grid, season_start_md,
                          forecast_window, horizons,
                          conditional=True, cv_by_year=True):
    """
    Fit KDEs for all cells and compute forecasts.

    Returns dict: forecasts (DataFrame), kdes (dict)
    """
    kdes = fit_kdes_by_cell(gt_train) if not cv_by_year else None
    cell_ids = gt_train["id"].unique()

    parts = []
    for cell_id in cell_ids:
        part = compute_forecasts_for_cell(
            cell_id, issue_grid, kdes,
            season_start_md, forecast_window, horizons,
            conditional=conditional, cv_by_year=cv_by_year, gt_train=gt_train,
        )
        parts.append(part)

    forecasts = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return {"forecasts": forecasts, "kdes": kdes}
