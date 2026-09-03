# ==============================================================================
# File: evaluation_2025_utils.py
# ==============================================================================
# Purpose
#   Helper functions for 2_2025_evaluation.py. Provides date parsing, onset
#   reading, Brier/AUC/RPS scoring, Platt calibration lookup and application,
#   and onset-label-to-path/suffix mappings.
# ==============================================================================

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit

from ..pipelines._shared.misc import interval_bins, rps_bins, assign_lead_bin
from ..prepare_data.spatial_id_utils import ensure_spatial_id_col

# Default 4-week layout; the functions below accept n_bins / explicit bins to
# generalize to any number of forecast bins.
INTERVAL_BINS_5 = interval_bins(4)
RPS_BINS = rps_bins(4)


def skill_score(score_model, score_ref):
    return 1.0 - (score_model / score_ref)


def yearly_suffix_from_onset(on_lbl):
    """Map onset label to yearly output file suffix."""
    mapping = {
        "obs_ref_year": "",
        "obs_fixed_cutoff": "_fixed_cutoff",
        "obs_season_start": "_no_ref_filter",
    }
    if on_lbl not in mapping:
        raise ValueError(f"Unknown onset label: {on_lbl}")
    return mapping[on_lbl]


def parse_any_date(x):
    """Flexibly parse a date string or return a Date as-is."""
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(x).date()
    return pd.to_datetime(x, dayfirst=False, yearfirst=True, errors="coerce").date()


def diff_to_bin(diff_days, days_per_bin=7, n_bins=4):
    """Map a positive lead-day difference to a forecast bin (week1..weekN/later).
    Non-positive lead days return None; those rows are filtered out downstream."""
    return assign_lead_bin(diff_days, days_per_bin, n_bins, allow_earlier=False)


def ensure_id(df):
    """
    Attach a consistent spatial `id` column, matching the adm3-based convention
    used across the pipeline (see nc_utils.ensure_id_col):

      1. keep an existing `id` column,
      2. else derive it from `adm3_name` (current Ethiopia data), or finally
      3. fall back to `lat`_`lon` (legacy gridded data).

    Raises if none of those are available.
    """
    return ensure_spatial_id_col(df, context="evaluation spatial IDs")


def read_onsets(path):
    """
    Read onset CSV file. Returns DataFrame: id (str), true_onset (date).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Onset file missing: {path}")
    df = pd.read_csv(path, dtype=str)
    df["true_onset"] = pd.to_datetime(
        df["true_onset"].astype(str).replace("NaT", pd.NA),
        errors="coerce",
        dayfirst=True,
    )
    df = df.dropna(subset=["true_onset"])
    df = ensure_id(df)
    df["true_onset"] = df["true_onset"].dt.date
    return df[["id", "true_onset"]].drop_duplicates()


def ngcm_path_from_onset(on_lbl, forecast_files):
    """Choose NGCM forecast file based on ground-truth label."""
    if on_lbl == "obs_season_start":
        return forecast_files["ngcm_no_ref_filter"]
    return forecast_files["ngcm"]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def brier_auc(df, prob_col="prob"):
    """Brier score and AUC for a binary outcome column."""
    d = df.dropna(subset=[prob_col, "outcome", "bin"])
    y = (d["bin"] == d["outcome"]).astype(float)
    bs = float(np.mean((d[prob_col].values - y.values) ** 2))
    auc_val = np.nan
    if len(y.unique()) == 2:
        try:
            auc_val = roc_auc_score(y.values, d[prob_col].values)
        except Exception:
            pass
    return pd.DataFrame([{"brier": bs, "auc": auc_val}])


def rps_row(prob_vec, outcome_bin, bins=None):
    """Ranked Probability Score for a single row (over the given RPS bins)."""
    if bins is None:
        bins = RPS_BINS
    if outcome_bin is None or any(np.isnan(v) for v in prob_vec):
        return np.nan
    s = sum(prob_vec)
    if not np.isfinite(s) or s <= 0:
        return np.nan
    prob_vec = np.array(prob_vec) / s
    k = bins.index(outcome_bin)
    ok = np.array([1.0 if i >= k else 0.0 for i in range(len(bins))])
    ck = np.cumsum(prob_vec)
    return float(np.sum((ck - ok) ** 2))


def rps_frame(df_probs, outcome_vec, bins=None):
    """Compute RPS for each row of df_probs against outcome_vec (over `bins`)."""
    if bins is None:
        bins = RPS_BINS
    results = []
    for i in range(len(df_probs)):
        row_probs = [df_probs.iloc[i][b] for b in bins]
        results.append(rps_row(row_probs, outcome_vec.iloc[i] if hasattr(outcome_vec, "iloc") else outcome_vec[i], bins=bins))
    return np.array(results)


def make_probs_for_rps(df, family="forecast", n_bins=4):
    """Build the (n_bins+2)-bin probability DataFrame for RPS computation."""
    wk = [f"week{w}" for w in range(1, int(n_bins) + 1)] + ["later"]
    if family == "forecast":
        src = {b: b for b in wk}
    elif family == "clim":
        src = {b: f"clim_{b}" for b in wk}
    elif family == "clim_con":
        src = {b: f"clim_con_{b}" for b in wk}
    else:
        raise ValueError(f"Unknown family: {family}")

    out = pd.DataFrame({b: df[src[b]] for b in wk})
    out["earlier"] = np.maximum(0.0, 1.0 - out[wk].sum(axis=1))
    return out[rps_bins(n_bins)]


# ---------------------------------------------------------------------------
# Platt calibration helpers
# ---------------------------------------------------------------------------

def platt_config_from_onset(on_lbl):
    """Return Platt weight config for a given ground-truth variant."""
    mapping = {
        "obs_ref_year": {"label": "ngcm_fixed_cutoff", "cutoff_tag": ""},
        "obs_fixed_cutoff": {"label": "ngcm_fixed_cutoff", "cutoff_tag": "_fixed_cutoff"},
        "obs_season_start": {"label": "ngcm", "cutoff_tag": "_no_ref_filter"},
    }
    if on_lbl not in mapping:
        raise ValueError(f"Unknown onset label: {on_lbl}")
    return mapping[on_lbl]


def ngcm_yearly_name_from_onset(on_lbl):
    """Model name for NGCM in yearly metrics, matching 1_blend_evaluation convention."""
    mapping = {
        "obs_ref_year": "ngcm_calibrated_fixed_cutoff",
        "obs_fixed_cutoff": "ngcm_calibrated_fixed_cutoff",
        "obs_season_start": "ngcm_calibrated",
    }
    if on_lbl not in mapping:
        raise ValueError(f"Unknown onset label: {on_lbl}")
    return mapping[on_lbl]


def apply_platt_5(prob_mat, weights_df):
    """
    Apply saved Platt weights to a 5-bin probability matrix and renormalize.

    Parameters
    ----------
    prob_mat : ndarray (n x 5)
    weights_df : DataFrame with columns: bin, intercept, slope

    Returns
    -------
    ndarray (n x 5) renormalized
    """
    cal = prob_mat.copy().astype(float)
    bins = interval_bins((prob_mat.shape[1] - 1))   # N week bins + 'later'
    for i, b in enumerate(bins):
        row = weights_df[weights_df["bin"] == b]
        if row.empty:
            continue
        intercept = float(row["intercept"].iloc[0])
        slope = float(row["slope"].iloc[0])
        p_raw = np.clip(prob_mat[:, i], 1e-6, 1 - 1e-6)
        lp = logit(p_raw)
        cal[:, i] = expit(intercept + slope * lp)
    rs = cal.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return cal / rs


def read_platt_weights(platt_dir, platt_label, cutoff_tag=""):
    """Read Platt weights pickle exported by 1_blend_evaluation.py."""
    output_tag_cv = f"{cutoff_tag}_2000_2024"
    platt_path = os.path.join(
        platt_dir,
        f"platt_weights_{platt_label}_calibrated_df{output_tag_cv}.pkl"
    )
    if not os.path.exists(platt_path):
        return {"ok": False, "path": platt_path, "df": None}
    with open(platt_path, "rb") as f:
        df = pickle.load(f)
    return {"ok": True, "path": platt_path, "df": df}
