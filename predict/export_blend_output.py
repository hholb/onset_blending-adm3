"""
export_blend_output.py
======================
For a given issue date, extract blended model and climatology weekly
probabilities from the blending pipeline outputs and write a CSV in the
format of blend_output_summary_YYYYMMDD.csv:

    lat, lon, time,
    week1, week2, week3, week4, later,          ← blended model (cv_* cols)
    clim_week1, clim_week2, clim_week3, clim_week4, clim_later

Sources
-------
Blended model  : cv_preds_<model>_global_*.pkl  (from 1_blend_evaluation.py)
Climatology    : the weekly-bin wide pickle      (from 0_connect_...py)
                 configured climatology week columns are converted from logits

Usage (run from repo root)
--------------------------
Standard mode (two input files derived from spec):
    python predict/export_blend_output.py \\
        --issue_date  2021-06-15 \\
        --spec_id     cv_models_fixed_cutoff \\
        --model       blended_model \\
        [--method     global] \\
        [--coef_dir   Monsoon_Data/results/2025_model_evaluation/] \\
        [--cv_preds_file  Monsoon_Data/results/2025_model_evaluation/cv_preds_blended_model_global_clim_mok_date_2026.pkl] \\
        [--pipeline_input_dir Monsoon_Data/Processed_Data/2025_pipeline_input] \\
        [--input_path Monsoon_Data/Processed_Data/2026_pipeline_input/cv_data_clim_mok_date_new_pipeline.pkl] \\
        [--out_dir    Monsoon_Data/results/2025_model_evaluation/exports/]

Operational mode (single combined preds pkl containing both cv_* and climatology columns):
    python predict/export_blend_output.py \\
        --issue_date  2026-06-09 \\
        --spec_id     cv_models_fixed_cutoff_2026 \\
        --preds_file  Monsoon_Data/results/2026/blended_model_global_year2026_preds.pkl \\
        [--out_dir    Monsoon_Data/results/2026/exports/]

Notes
-----
--cv_preds_file   overrides --coef_dir for locating the cv_preds pickle.
--input_path  overrides --pipeline_input_dir for locating the wide pipeline pickle.
--preds_file  activates operational mode: skips loading two separate files and reads
              everything from a single combined pkl.
"""

import os
import sys
import pickle
import argparse
import warnings
import re
import numpy as np
import pandas as pd
from scipy.special import expit


def _week_nums(df, prefix):
    """Sorted week numbers present as `<prefix>week<n>` columns (bin count is
    inferred from the data, so any n_bins flows through)."""
    pat = re.compile(rf"^{re.escape(prefix)}week(\d+)$")
    return sorted(int(m.group(1)) for c in df.columns for m in [pat.match(c)] if m)

#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#
#from python.blending_process.blend_evaluation_utils import (
#    input_rds_from_cutoff,
#    make_cutoff_tag,
#    make_year_tag,
#)
#from python.pipelines._shared.read_spec import load_spec

# This points to the parent directory (the repo root)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#sys.path.insert(0, REPO_ROOT)
# Add the repo root to sys.path so the 'python' package can be found
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.chdir(REPO_ROOT)

try:
    from python.blending_process.blend_evaluation_utils import (
        input_rds_from_cutoff,
        make_cutoff_tag,
        make_year_tag,
        resolve_climatology_weeks,
    )
    from python.pipelines._shared.read_spec import load_spec
except ImportError:
    print(f"Error: Could not import internal modules from {REPO_ROOT}")
    print("Ensure this script is located exactly one folder deep from the root.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_cv_preds_file(coef_dir, model, method, output_tag):
    """Locate the cv_preds pickle for the given model."""
    fname = f"cv_preds_{model}_{method}{output_tag}.pkl"
    path = os.path.join(coef_dir, fname)
    if not os.path.exists(path):
        # List available cv_preds files to help user
        candidates = [f for f in os.listdir(coef_dir)
                      if f.startswith("cv_preds_") and f.endswith(".pkl")]
        raise FileNotFoundError(
            f"cv_preds file not found: {path}\n"
            f"Available cv_preds files in {coef_dir}:\n"
            + "\n".join(f"  {c}" for c in sorted(candidates))
        )
    return path


def load_pkl(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, pd.DataFrame):
        obj = pd.DataFrame(obj)
    return obj


def parse_issue_date(date_str):
    """Parse issue date string flexibly."""
    for fmt in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y", "%Y%m%d"):
        try:
            return pd.Timestamp(date_str).date()
        except Exception:
            pass
    raise ValueError(f"Cannot parse issue date: {date_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export blended model + climatology probabilities for one issue date.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--issue_date",  required=True,
                        help="Issue date to export, e.g. 2021-06-15")
    parser.add_argument("--spec_id",     required=True,
                        help="CV spec ID, e.g. cv_models_fixed_cutoff")
    parser.add_argument("--model",       default="blended_model",
                        help="Blended model name (default: blended_model)")
    parser.add_argument("--method",      default="global",
                        help="CV method (default: global)")
    parser.add_argument("--coef_dir", default=None,
                        help="Directory containing cv_preds_*.pkl files. "
                             "Default: Monsoon_Data/results/2025_model_evaluation/")
    parser.add_argument("--out_dir",     default=None,
                        help="Output directory. "
                             "Default: Monsoon_Data/results/2025_model_evaluation/exports/")
    parser.add_argument("--pipeline_input_dir", default=None,
                        help="Optional: Subdirectory under Monsoon_Data/Processed_Data/ "
                             "containing the wide pipeline pickle.")
    parser.add_argument("--input_path", default=None,
                        help="Optional: Direct path to the wide pipeline pickle file. "
                             "Overrides --pipeline_input_dir and the default derived path.")
    parser.add_argument("--cv_preds_file", default=None,
                        help="Optional: Direct path to the cv_preds pickle file. "
                             "Overrides --coef_dir and the auto-derived cv_preds filename.")
    parser.add_argument("--preds_file", default=None,
                        help="Optional: Operational mode. Direct path to a combined preds pkl "
                             "(e.g. blended_model_global_year2026_preds.pkl) that contains both "
                             "cv_* and configured climatology columns. When provided, --cv_preds_file, "
                             "--coef_dir, --input_path, and --pipeline_input_dir are all ignored.")
    args = parser.parse_args()

    # ── Parse date ────────────────────────────────────────────────────
    issue_date = pd.Timestamp(args.issue_date).date()
    date_str_fmt = pd.Timestamp(args.issue_date).strftime("%Y%m%d")
    print(f"\nIssue date : {issue_date}")
    print(f"Repo root  : {REPO_ROOT}")

    # ── Load spec ─────────────────────────────────────────────────────
    spec = load_spec(args.spec_id, "2025_blend")
    cutoff_mode   = spec["run"]["cutoff_mode"]
    holdout_years = [int(y) for y in spec["run"]["cv_holdout_years"]]
    cutoff_tag    = make_cutoff_tag(cutoff_mode)
    year_tag      = make_year_tag(holdout_years)
    output_tag    = f"{cutoff_tag}{year_tag}"

    #coef_dir = args.coef_dir or "Monsoon_Data/results/2025_model_evaluation"
    default_results = os.path.join(REPO_ROOT, "Monsoon_Data/results/2025_model_evaluation")
    coef_dir = args.coef_dir or default_results
    out_dir     = args.out_dir or os.path.join(coef_dir, "exports")
    os.makedirs(out_dir, exist_ok=True)

    # ── Operational mode: single combined preds pkl ───────────────────
    if args.preds_file:
        preds_path = args.preds_file
        if not os.path.exists(preds_path):
            raise FileNotFoundError(f"--preds_file not found: {preds_path}")
        print(f"Operational mode — loading combined preds: {preds_path}")
        combined = load_pkl(preds_path)
        combined["time"] = pd.to_datetime(combined["time"]).dt.date
        clim_day = combined[combined["time"] == issue_date].copy()
        if clim_day.empty:
            available = sorted(combined["time"].unique())
            raise ValueError(
                f"No rows for issue_date={issue_date} in preds file.\n"
                f"Available dates: {available}"
            )
        print(f"  Rows for {issue_date}: {len(clim_day)}")
        bins = [f"week{w}" for w in _week_nums(clim_day, "cv_")] + ["later"]
        cv_col_map = {f"cv_{b}": b for b in bins}
        blend_day = clim_day.rename(columns=cv_col_map)
    else:
        # ── Load cv_preds (blended model predictions) ─────────────────
        if args.cv_preds_file:
            cv_path = args.cv_preds_file
            if not os.path.exists(cv_path):
                raise FileNotFoundError(f"--cv_preds_file not found: {cv_path}")
        else:
            cv_path = find_cv_preds_file(coef_dir, args.model, args.method, output_tag)
        print(f"Loading blended predictions : {cv_path}")
        cv_preds = load_pkl(cv_path)
        cv_preds["time"] = pd.to_datetime(cv_preds["time"]).dt.date

        # Filter to issue date
        blend_day = cv_preds[cv_preds["time"] == issue_date].copy()
        if blend_day.empty:
            available = sorted(cv_preds["time"].unique())
            raise ValueError(
                f"No blended model rows for issue_date={issue_date}.\n"
                f"Available dates range: {available[0]} to {available[-1]}"
            )
        print(f"  Blended rows for {issue_date}: {len(blend_day)}")

        # Rename cv_* columns to week*, later (bin count inferred from the data)
        bins = [f"week{w}" for w in _week_nums(cv_preds, "cv_")] + ["later"]
        cv_col_map = {f"cv_{b}": b for b in bins}
        blend_day = blend_day.rename(columns=cv_col_map)

        # Keep only needed columns — include id for the merge
        blend_cols = ["id", "time"] + bins
        missing_bins = [b for b in bins if b not in blend_day.columns]
        if missing_bins:
            raise ValueError(f"Missing cv columns in blended predictions: {missing_bins}")
        blend_day = blend_day[blend_cols].copy()

        # ── Load wide_df (for climatology) ────────────────────────────
        input_file = input_rds_from_cutoff(cutoff_mode)
        # ── Resolve Input Path (Wide Pickle) ──────────────────────────
        if args.input_path:
            input_path = args.input_path
        elif args.pipeline_input_dir:
            input_path = os.path.join(REPO_ROOT, args.pipeline_input_dir, input_file)
        else:
            input_path = os.path.join(REPO_ROOT, "Monsoon_Data/Processed_Data/2025_pipeline_input", input_file)

        print(f"Loading wide pickle         : {input_path}")
        wide_df = load_pkl(input_path)
        wide_df["time"] = pd.to_datetime(wide_df["time"]).dt.date

        clim_day = wide_df[wide_df["time"] == issue_date].copy()
        if clim_day.empty:
            raise ValueError(f"No wide_df rows for issue_date={issue_date}.")
        print(f"  Climatology rows for {issue_date}: {len(clim_day)}")

    # ── Extract climatology probabilities ─────────────────────────────
    # The configured climatology week columns are stored in logit scale.
    clim_prefix, clim_weeks = resolve_climatology_weeks(spec, clim_day)
    clim_logit_cols = {
        f"{clim_prefix}_week{w}": f"clim_week{w}" for w in clim_weeks
    }

    clim_out = clim_day[["id", "time"]].copy()
    for logit_col, out_col in clim_logit_cols.items():
        clim_out[out_col] = expit(clim_day[logit_col].values.astype(float))

    # clim_later = max(0, 1 - sum(clim_week1..N))
    clim_week_cols = list(clim_logit_cols.values())
    clim_out["clim_later"] = np.maximum(
        0.0, 1.0 - clim_out[clim_week_cols].sum(axis=1)
    )

    # ── Merge blended + climatology on id string ─────────────────────
    # Merge on the id string ("10.00_34.00") rather than float lat/lon,
    # to avoid any decimal digit mismatch (str(10.0)="10.0" != "10.00").
    # Both DataFrames carry id unchanged from nc_utils through connect_utils.
    merged = clim_out.merge(
        blend_day[["id"] + bins],
        on="id",
        how="left"
    )

    # ── Final column order (matches target format) ─────────────────────
    col_order = (
        ["id", "time"]
        + bins                              # week1..later (blended)
        + [f"clim_{b}" for b in bins]      # clim_week1..clim_later
    )
    out = merged[col_order].copy()

    # Format time as M/D/YY to match sample
    out["time"] = pd.to_datetime(out["time"]).dt.strftime("%-m/%-d/%y")

    # ── Save ──────────────────────────────────────────────────────────
    out_fname = f"blend_output_summary_{date_str_fmt}.csv"
    out_path  = os.path.join(out_dir, out_fname)
    out.to_csv(out_path, index=False)

    print(f"\nOutput rows  : {len(out)}")
    print(f"Blended non-NaN rows : {out['week1'].notna().sum()}")
    print(f"Saved → {out_path}")
    print()
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
