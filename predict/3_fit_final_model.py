"""
3_fit_final_model.py
====================
Fits the blending model on ALL available training years (no holdout),
and saves the resulting coefficient bundle for use in future-year prediction.

This is the "production" model — unlike the CV models in 1_blend_evaluation.py
which each hold out one year, this uses every year of data to fit the best
possible model for forecasting future (unseen) years.

The saved bundle is identical in format to what 1_blend_evaluation.py saves
per holdout year, so apply_blend_model.py can load it directly via --coef_tag.

Usage (run from repo root)
--------------------------
    python 3_fit_final_model.py \\
        --spec_id  cv_models_fixed_cutoff \\
        --model    blended_model \\
        [--input_path Monsoon_Data/Processed_Data/2025_pipeline_input/cv_data_fixed_cutoff_new_pipeline.pkl] \\
        [--method  global] \\
        [--tag     final] \\
        [--out_dir Monsoon_Data/results/2025_model_evaluation/]

Output
------
    coefs_blended_model_global_final.pkl   <- coef bundle (coefs, scaler, features, formula)
    coefs_blended_model_global_final.csv   <- human-readable coefficient table

Then to apply to a future year:
    python apply_blend_model.py \\
        --spec_id    cv_models_fixed_cutoff \\
        --model      blended_model \\
        --year       2026 \\
        --coef_tag   final \\
        --input_path Monsoon_Data/Processed_Data/2025_pipeline_input/wide_2026.pkl
"""

import os
import sys
# --- MODIFICATION FOR SUBFOLDER SUPPORT ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
# ------------------------------------------
import pickle
import argparse
import warnings
import itertools
import re
import numpy as np
import pandas as pd
import patsy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from python.blending_process.blend_evaluation_utils import (
    apply_formula_sample_support,
    build_formulas_from_spec,
    expand_formula_str,
    input_rds_from_cutoff,
    make_cutoff_tag,
    restrict_to_allowed,
    _make_multinom_clf,
)
from python.pipelines._shared.misc import coalesce, interval_bins
from python.prepare_data.nc_utils import ensure_id_col
from python.prepare_data.spatial_id_utils import (
    resolve_grid_id_convention,
    validate_id_coordinate_consistency,
)
from python.pipelines._shared.read_spec import load_spec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_final_model(formula_str, train_df):
    """
    Fit multinomial logistic regression on all training rows.

    Parameters
    ----------
    formula_str : str  e.g. "outcome ~ a + b + a:b"
    train_df    : DataFrame — all training rows (all years, allowed cells)

    Returns
    -------
    bundle : dict with keys coefs, scaler, features, formula
    """
    # Build design matrix via patsy (handles interaction terms)
    rhs  = formula_str.split("~", 1)[1].strip() if "~" in formula_str else formula_str
    X_df = patsy.dmatrix(
        rhs, train_df, return_type="dataframe",
        NA_action=patsy.NAAction(NA_types=[])
    )
    X_df = X_df.drop(columns=["Intercept"], errors="ignore")
    feature_cols = list(X_df.columns)

    # Build outcome vector — keep only string outcomes (drop NaN)
    y = train_df["outcome"].values
    valid = np.array([isinstance(v, str) for v in y])
    X_vals = X_df.values.astype(float)

    # Also drop rows with NaN/Inf features
    finite = np.isfinite(X_vals).all(axis=1)
    mask   = valid & finite

    X_fit = X_vals[mask]
    y_fit = y[mask]

    if len(np.unique(y_fit)) < 2:
        raise ValueError("Fewer than 2 outcome classes in training data — cannot fit model.")

    print(f"  Training rows used : {mask.sum()} / {len(train_df)}")
    print(f"  Outcome classes    : {sorted(np.unique(y_fit))}")
    print(f"  Feature columns    : {len(feature_cols)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_fit)

    # Fit model
    clf = _make_multinom_clf()
    clf.fit(X_scaled, y_fit)
    clf.feature_names = feature_cols
    clf.scaler_       = scaler

    # Build coef DataFrame
    rows = []
    for i, cls in enumerate(clf.classes_):
        for j, feat in enumerate(feature_cols):
            rows.append({
                "class":       cls,
                "feature":     feat,
                "coefficient": float(clf.coef_[i, j]),
                "intercept":   float(clf.intercept_[i]),
            })
    coef_df = pd.DataFrame(rows)

    return {
        "coefs":    coef_df,
        "scaler":   scaler,
        "features": feature_cols,
        "formula":  formula_str,
    }


def print_coef_table(coef_df, model_name, label):
    """Pretty-print the coefficient table."""
    print(f"\n{'='*65}")
    print(f"Coefficients: {model_name}  |  {label}")
    print(f"{'='*65}")

    classes  = sorted(coef_df["class"].unique())
    features = coef_df["feature"].unique()

    header = f"  {'feature':<32}" + "".join(f"{c:>12}" for c in classes)
    print(header)
    print("  " + "-" * (32 + 12 * len(classes)))

    for feat in features:
        row = f"  {feat:<32}"
        for cls in classes:
            val = coef_df[
                (coef_df["class"] == cls) & (coef_df["feature"] == feat)
            ]["coefficient"]
            row += f"{float(val.iloc[0]):>12.4f}" if len(val) > 0 else f"{'N/A':>12}"
        print(row)

    row = f"  {'[intercept]':<32}"
    for cls in classes:
        val = coef_df[coef_df["class"] == cls]["intercept"].iloc[0]
        row += f"{float(val):>12.4f}"
    print(row)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit blending model on all years and save coef bundle for future prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--spec_id", required=True,
                        help="CV spec ID (e.g. cv_models_fixed_cutoff)")
    parser.add_argument("--model",   required=True,
                        help="Model name in cv_models*.yml (e.g. blended_model)")
    parser.add_argument("--method",  default="global",
                        help="CV method (default: global)")
    parser.add_argument("--tag",     default="final",
                        help="Tag for output filenames (default: final). "
                             "Output: coefs_{model}_{method}_{tag}.pkl")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory. "
                             "Defaults to Monsoon_Data/results/2025_model_evaluation/")
    parser.add_argument("--input_path", default=None,
                        help="Read the connector pickle at this exact path "
                             "(highest priority).")
    parser.add_argument("--dissem_file", default=None,
                        help="Override the dissemination CSV configured by "
                             "cell.dissemination")
    parser.add_argument("--pipeline_input_dir", default="Monsoon_Data/Processed_Data/2025_pipeline_input",
                        help="Directory containing the wide processed data")
    args = parser.parse_args()

    # ── Load spec ──────────────────────────────────────────────────────
    spec = load_spec(args.spec_id, "2025_blend")
    run_cfg = spec.get("run", {})
    cutoff_mode = coalesce(run_cfg.get("cutoff_mode"), "ref_onset")
    training_years = list(map(
        int,
        coalesce(run_cfg.get("training_years"), list(range(2000, 2025))),
    ))
    true_holdout_years = set(map(
        int, coalesce(run_cfg.get("true_holdout_years"), [])
    ))
    fit_years = [
        year for year in training_years if year not in true_holdout_years
    ]
    pipeline_input_dir = run_cfg.get("pipeline_input_dir", "")

    out_dir = args.out_dir or "Monsoon_Data/results/2025_model_evaluation"
    os.makedirs(out_dir, exist_ok=True)

    # ── Load input data ────────────────────────────────────────────────
    input_file = input_rds_from_cutoff(cutoff_mode)
    input_override = run_cfg.get("input_rds_override")
    if args.input_path:
        input_path = args.input_path
    elif input_override:
        input_path = input_override
    else:
        input_path = os.path.join(
            args.pipeline_input_dir, pipeline_input_dir, input_file
        )
    print(f"\nLoading wide_df from: {input_path}")
    with open(input_path, "rb") as f:
        wide_df = pickle.load(f)
    print(f"  Total rows in wide_df : {len(wide_df)}")

    # ── Normalize input and dissemination IDs ──────────────────────
    wide_df = ensure_id_col(wide_df, spec=spec, context="blend input IDs")
    id_convention = resolve_grid_id_convention(
        spec=spec,
        authoritative_ids=wide_df["id"],
        context="blend input/dissemination ID handoff",
    )

    cell_cfg = spec["cell"]
    dissemination_csv = (
        args.dissem_file
        or cell_cfg.get("dissemination")
        or "Monsoon_Data/dissemination_cells.csv"
    )
    dissemination_input = pd.read_csv(dissemination_csv, dtype=str)
    dissemination_id_col = cell_cfg.get("id_col") or cell_cfg.get(
        "dissemination_id_col"
    )
    if dissemination_id_col:
        if dissemination_id_col not in dissemination_input.columns:
            raise ValueError(
                f"cell.id_col '{dissemination_id_col}' not found. "
                f"Found: {dissemination_input.columns.tolist()}"
            )
        dissemination_input = dissemination_input.rename(
            columns={dissemination_id_col: "id"}
        )
    dissemination_cells = ensure_id_col(
        dissemination_input,
        spec=spec,
        convention=id_convention,
        context="blend dissemination IDs",
    )
    validate_id_coordinate_consistency(
        dissemination_cells, context="blend dissemination IDs"
    )
    dissemination_cells = dissemination_cells.drop_duplicates("id")

    n_bins = max([
        int(match.group(1))
        for column in wide_df.columns
        for match in [re.match(r"^prob_clim_week(\d+)$", column)]
        if match
    ] or [4])

    # ── Get formula ────────────────────────────────────────────────────
    formulas = build_formulas_from_spec(spec, cutoff_mode, data=wide_df)
    if args.model not in formulas:
        raise ValueError(
            f"Model '{args.model}' not found in spec formulas. "
            f"Available: {list(formulas.keys())}"
        )
    formula_str = formulas[args.model]
    wide_df, support = apply_formula_sample_support(wide_df, formulas)
    print(
        "Formula sample support "
        f"({support['policy']}): {support['rows_before']} -> "
        f"{support['rows_after']} rows "
        f"({support['rows_excluded']} excluded)"
    )

    wide_df_allowed = restrict_to_allowed(wide_df, dissemination_cells)
    train_df = wide_df_allowed[wide_df_allowed["year"].isin(fit_years)].copy()
    if train_df.empty:
        raise ValueError(
            "Final fit has no rows in the configured training years and "
            "dissemination cells."
        )
    required_classes = interval_bins(n_bins)
    present_classes = {
        value for value in train_df["outcome"].unique()
        if isinstance(value, str)
    }
    missing_classes = [
        value for value in required_classes if value not in present_classes
    ]
    if missing_classes:
        raise ValueError(
            "Final-fit training rows are missing outcome classes: "
            + ", ".join(missing_classes)
        )

    print(f"  Training years        : {sorted(fit_years)}")
    print(f"  Training rows (allowed cells) : {len(train_df)}")
    print(f"\nFormula : {formula_str}")

    # ── Fit model on all training years ───────────────────────────────
    print(f"\nFitting {args.model} on all {len(fit_years)} training years...")
    bundle = fit_final_model(formula_str, train_df)

    # ── Print and save ────────────────────────────────────────────────
    print_coef_table(
        bundle["coefs"],
        args.model,
        f"all years ({min(fit_years)}-{max(fit_years)})",
    )

    stem     = f"coefs_{args.model}_{args.method}_{args.tag}"
    pkl_path = os.path.join(out_dir, f"{stem}.pkl")
    csv_path = os.path.join(out_dir, f"{stem}.csv")

    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    bundle["coefs"].to_csv(csv_path, index=False)

    print(f"Saved:")
    print(f"  Coef bundle (pkl) -> {pkl_path}")
    print(f"  Coef table  (csv) -> {csv_path}")
    print(f"\nTo apply to a future year:")
    print(f"  python apply_blend_model.py \\")
    print(f"      --spec_id    {args.spec_id} \\")
    print(f"      --model      {args.model} \\")
    print(f"      --year       <future_year> \\")
    print(f"      --coef_tag   {args.tag} \\")
    print(f"      --input_path <path_to_future_wide_df.pkl>")


if __name__ == "__main__":
    main()
