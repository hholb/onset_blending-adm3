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
import gc
from functools import lru_cache
import pickle
import tempfile
import warnings
from multiprocessing import get_context
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


def _enforce_floor_with_target_sum_rows(p_raw, target_sum, lb=MIN_PROB,
                                         eps=EPS):
    """Vectorized form of ``_enforce_floor_with_target_sum`` by row."""
    p_raw = np.asarray(p_raw, dtype=float)
    target_sum = np.asarray(target_sum, dtype=float).copy()
    if p_raw.ndim != 2:
        raise ValueError("p_raw must be a two-dimensional array.")
    if len(target_sum) != len(p_raw):
        raise ValueError("target_sum must have one value per probability row.")

    p_raw = np.maximum(p_raw, 0.0)
    p_raw = np.where(np.isfinite(p_raw), p_raw, 0.0)
    target_sum = np.where(
        np.isfinite(target_sum) & (target_sum >= 0.0), target_sum, 0.0
    )
    target_sum = np.minimum(target_sum, 1.0)

    out = np.empty_like(p_raw)
    zero = target_sum <= eps
    small = (~zero) & (target_sum < p_raw.shape[1] * lb)
    normal = ~(zero | small)
    out[zero] = 0.0
    out[small] = target_sum[small, None] / p_raw.shape[1]

    if normal.any():
        p = np.maximum(p_raw[normal], lb)
        excess = p.sum(axis=1) - target_sum[normal]
        no_reduction = excess <= eps
        adjusted = np.empty_like(p)
        adjusted[no_reduction] = p[no_reduction]

        reduce = ~no_reduction
        if reduce.any():
            p_reduce = p[reduce]
            reducible = p_reduce - lb
            reducible_sum = reducible.sum(axis=1)
            degenerate = reducible_sum <= eps
            reduced = np.empty_like(p_reduce)
            targets = target_sum[normal][reduce]
            reduced[degenerate] = (
                targets[degenerate, None] / p_raw.shape[1]
            )
            regular = ~degenerate
            reduced[regular] = p_reduce[regular] - (
                excess[reduce][regular, None]
                * reducible[regular]
                / reducible_sum[regular, None]
            )
            adjusted[reduce] = np.maximum(reduced, lb)
        out[normal] = adjusted

    return out


def _predict_pair_for_rows(kde, d0, windows, max_h):
    """Predict compatible conditional/unconditional rows from one KDE."""
    d0 = np.asarray(d0, dtype=int)
    windows = np.asarray(windows, dtype=int)
    conditional = np.full((len(d0), max_h), np.nan)
    unconditional = np.full((len(d0), max_h + 1), np.nan)
    if kde is None:
        return conditional, unconditional

    cdf = _kde_cdf(kde, None)
    for horizon in np.unique(windows[windows > 0]):
        rows = np.flatnonzero(windows == horizon)
        days = np.arange(1, int(horizon) + 1)
        num = cdf(d0[rows, None] + days) - cdf(
            d0[rows, None] + days - 1
        )
        num = np.maximum(num, 0.0)
        num = np.where(np.isfinite(num), num, 0.0)

        unc = _enforce_floor_with_target_sum_rows(num, num.sum(axis=1))
        base_prob = cdf(d0[rows])
        unconditional[rows, 0] = base_prob
        unconditional[rows, 1:int(horizon) + 1] = unc

        denom = 1.0 - base_prob
        cond = np.empty_like(num)
        fallback = (~np.isfinite(denom)) | (denom <= EPS)
        if fallback.any():
            cond[fallback] = _enforce_floor_with_target_sum_rows(
                num[fallback], num[fallback].sum(axis=1)
            )

        regular = ~fallback
        if regular.any():
            p_raw = num[regular] / denom[regular, None]
            p_raw = np.maximum(p_raw, 0.0)
            p_raw = np.where(np.isfinite(p_raw), p_raw, 0.0)
            target_sum = p_raw.sum(axis=1)
            empty = (~np.isfinite(target_sum)) | (target_sum <= EPS)
            cond_regular = np.empty_like(p_raw)
            cond_regular[empty] = 0.0
            if (~empty).any():
                cond_regular[~empty] = _enforce_floor_with_target_sum_rows(
                    p_raw[~empty], target_sum[~empty]
                )
            cond[regular] = cond_regular
        conditional[rows, :int(horizon)] = cond

    return conditional, unconditional


def _build_issue_plan(issue_grid, season_start_md, forecast_window, horizons):
    issue_grid = issue_grid.reset_index(drop=True)
    windows = np.array([
        resolve_forecast_window_by_time(t, forecast_window, horizons)
        for t in issue_grid["time"]
    ], dtype=object)
    windows = np.array([
        -1 if value is None else int(value) for value in windows
    ], dtype=int)
    return {
        "time": issue_grid["time"].to_numpy(copy=False),
        "year": issue_grid["year"].to_numpy(dtype=int, copy=False),
        "d0": np.array([
            compute_d0(t, season_start_md) for t in issue_grid["time"]
        ], dtype=int),
        "window": windows,
        "max_h": max_forecast_window(forecast_window, horizons),
    }


def _pair_matrices_for_cell(gt_years, onset_days, issue_plan,
                            cv_by_year, static_kde=None):
    max_h = issue_plan["max_h"]
    conditional = np.full((len(issue_plan["year"]), max_h), np.nan)
    unconditional = np.full((len(issue_plan["year"]), max_h + 1), np.nan)

    if not cv_by_year:
        return _predict_pair_for_rows(
            static_kde,
            issue_plan["d0"],
            issue_plan["window"],
            max_h,
        )

    for year in np.unique(issue_plan["year"]):
        rows = np.flatnonzero(issue_plan["year"] == year)
        kde = fit_kde(onset_days[gt_years != year])
        cond_part, unc_part = _predict_pair_for_rows(
            kde,
            issue_plan["d0"][rows],
            issue_plan["window"][rows],
            max_h,
        )
        conditional[rows] = cond_part
        unconditional[rows] = unc_part
    return conditional, unconditional


def _forecast_frame(cell_id, issue_plan, probabilities, conditional,
                    cv_by_year):
    if cv_by_year:
        model = "clim_kde_cv" if conditional else "clim_kde_unc_cv"
    else:
        model = "clim_kde" if conditional else "clim_kde_unc"

    data = {
        "time": issue_plan["time"],
        "year": issue_plan["year"],
        "id": np.full(len(issue_plan["year"]), str(cell_id), dtype=object),
        "model": np.full(len(issue_plan["year"]), model, dtype=object),
    }
    if not conditional:
        data["predicted_prob_day_0"] = probabilities[:, 0]
        for day in range(1, issue_plan["max_h"] + 1):
            data[f"predicted_prob_day_{day}"] = probabilities[:, day]
    else:
        for day in range(1, issue_plan["max_h"] + 1):
            data[f"predicted_prob_day_{day}"] = probabilities[:, day - 1]
    return pd.DataFrame(data)


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
                                conditional=True, cv_by_year=False, gt_train=None,
                                issue_plan=None):
    """
    Compute predicted probabilities for a single cell over all issue dates.

    Returns DataFrame: time, year, id, model, predicted_prob_day_1..N
    """
    cell_key = str(cell_id)
    if issue_plan is None:
        issue_plan = _build_issue_plan(
            issue_grid, season_start_md, forecast_window, horizons
        )

    if cv_by_year:
        if gt_train is None:
            raise ValueError("cv_by_year=True requires gt_train.")
        gt_id = gt_train[gt_train["id"] == cell_key]
        static_kde = None
    else:
        gt_id = None
        static_kde = kdes.get(cell_key) if kdes else None

    if gt_id is None:
        gt_years = np.empty(0, dtype=int)
        onset_days = np.empty(0, dtype=float)
    else:
        gt_years = gt_id["year"].to_numpy(dtype=int, copy=False)
        onset_days = gt_id["onset_day"].to_numpy(dtype=float, copy=False)

    cond_probs, unc_probs = _pair_matrices_for_cell(
        gt_years,
        onset_days,
        issue_plan,
        cv_by_year,
        static_kde=static_kde,
    )
    probabilities = cond_probs if conditional else unc_probs
    return _forecast_frame(
        cell_key, issue_plan, probabilities, conditional, cv_by_year
    )


def compute_all_forecasts(gt_train, issue_grid, season_start_md,
                          forecast_window, horizons,
                          conditional=True, cv_by_year=True):
    """
    Fit KDEs for all cells and compute forecasts.

    Returns dict: forecasts (DataFrame), kdes (dict)
    """
    kdes = fit_kdes_by_cell(gt_train) if not cv_by_year else None
    cell_ids = gt_train["id"].unique()
    issue_plan = _build_issue_plan(
        issue_grid, season_start_md, forecast_window, horizons
    )

    parts = []
    for cell_id in cell_ids:
        part = compute_forecasts_for_cell(
            cell_id, issue_grid, kdes,
            season_start_md, forecast_window, horizons,
            conditional=conditional, cv_by_year=cv_by_year, gt_train=gt_train,
            issue_plan=issue_plan,
        )
        parts.append(part)

    forecasts = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return {"forecasts": forecasts, "kdes": kdes}


def _compute_paired_chunk(gt_chunk, issue_grid, season_start_md,
                          forecast_window, horizons, cv_by_year):
    issue_plan = _build_issue_plan(
        issue_grid, season_start_md, forecast_window, horizons
    )
    conditional_parts = []
    unconditional_parts = []
    for cell_id, gt_id in gt_chunk.groupby("id", sort=False):
        gt_years = gt_id["year"].to_numpy(dtype=int, copy=False)
        onset_days = gt_id["onset_day"].to_numpy(dtype=float, copy=False)
        static_kde = None if cv_by_year else fit_kde(onset_days)
        cond_probs, unc_probs = _pair_matrices_for_cell(
            gt_years,
            onset_days,
            issue_plan,
            cv_by_year,
            static_kde=static_kde,
        )
        conditional_parts.append(
            _forecast_frame(cell_id, issue_plan, cond_probs, True, cv_by_year)
        )
        unconditional_parts.append(
            _forecast_frame(cell_id, issue_plan, unc_probs, False, cv_by_year)
        )

    return (
        pd.concat(conditional_parts, ignore_index=True),
        pd.concat(unconditional_parts, ignore_index=True),
    )


def _write_paired_chunk(task):
    (chunk_index, gt_chunk, issue_grid, season_start_md, forecast_window,
     horizons, cv_by_year, temp_dir) = task
    conditional, unconditional = _compute_paired_chunk(
        gt_chunk,
        issue_grid,
        season_start_md,
        forecast_window,
        horizons,
        cv_by_year,
    )
    conditional_path = os.path.join(
        temp_dir, f"conditional_{chunk_index:04d}.pkl"
    )
    unconditional_path = os.path.join(
        temp_dir, f"unconditional_{chunk_index:04d}.pkl"
    )
    with open(conditional_path, "wb") as f:
        pickle.dump(conditional, f)
    with open(unconditional_path, "wb") as f:
        pickle.dump(unconditional, f)
    result = (
        conditional_path,
        len(conditional),
        unconditional_path,
        len(unconditional),
    )
    _kde_cdf.cache_clear()
    return result


def _assemble_pickle_parts(paths, row_counts):
    total_rows = int(sum(row_counts))
    arrays = None
    columns = None
    offset = 0
    for path, row_count in zip(paths, row_counts):
        with open(path, "rb") as f:
            part = pickle.load(f)
        if arrays is None:
            columns = list(part.columns)
            arrays = {
                name: np.empty(total_rows, dtype=part[name].to_numpy().dtype)
                for name in columns
            }
        end = offset + int(row_count)
        for name in columns:
            arrays[name][offset:end] = part[name].to_numpy(copy=False)
        offset = end
        del part
        os.unlink(path)
    return pd.DataFrame(arrays, columns=columns)


def write_paired_climatologies(gt_train, issue_grid, season_start_md,
                               forecast_window, horizons, cv_by_year,
                               conditional_path, unconditional_path,
                               workers=1):
    """Write a compatible conditional/unconditional pair with shared work."""
    workers = int(workers)
    os.makedirs(os.path.dirname(conditional_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(unconditional_path) or ".", exist_ok=True)

    cell_ids = gt_train["id"].unique()
    ids_per_chunk = max(1, 200_000 // max(1, len(issue_grid)))
    id_chunks = [
        cell_ids[start:start + ids_per_chunk]
        for start in range(0, len(cell_ids), ids_per_chunk)
    ]

    temp_parent = os.path.dirname(conditional_path) or "."
    with tempfile.TemporaryDirectory(
        prefix=".climatology_tmp-", dir=temp_parent
    ) as temp_dir:
        tasks = []
        for chunk_index, ids in enumerate(id_chunks):
            gt_chunk = gt_train[gt_train["id"].isin(ids)].copy()
            tasks.append((
                chunk_index,
                gt_chunk,
                issue_grid,
                season_start_md,
                forecast_window,
                horizons,
                cv_by_year,
                temp_dir,
            ))

        worker_count = min(max(1, workers), len(tasks))
        if worker_count > 1:
            print(
                f"Building paired climatologies in {len(tasks)} chunks with "
                f"{worker_count} workers."
            )
            with get_context("spawn").Pool(worker_count) as pool:
                chunk_results = list(pool.imap(_write_paired_chunk, tasks))
        else:
            chunk_results = [_write_paired_chunk(task) for task in tasks]
        del tasks

        conditional = _assemble_pickle_parts(
            [result[0] for result in chunk_results],
            [result[1] for result in chunk_results],
        )
        with open(conditional_path, "wb") as f:
            pickle.dump(conditional, f)
        del conditional
        gc.collect()

        unconditional = _assemble_pickle_parts(
            [result[2] for result in chunk_results],
            [result[3] for result in chunk_results],
        )
        with open(unconditional_path, "wb") as f:
            pickle.dump(unconditional, f)
        del unconditional
        gc.collect()

    return {
        "conditional": conditional_path,
        "unconditional": unconditional_path,
    }
