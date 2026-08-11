#!/usr/bin/env python3
# ==============================================================================
# Script: 1_blend_evaluation.py
# ==============================================================================
# Purpose
#   Cross-validate weekly-bin multinomial onset models using a YAML spec.
#   Saves metrics, reliability data, Platt weights, and MME blend weights.
#
# Usage (run from MO_Forecast_Code/ directory)
#   python pipelines/blending_process/1_blend_evaluation.py --spec_id cv_models
#   python pipelines/blending_process/1_blend_evaluation.py --spec_id hindcast_1965_1978
# ==============================================================================

import argparse
import json
import os
import sys
import pickle
import yaml
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import re
from python.pipelines._shared.misc import coalesce, interval_bins
from python.prepare_data.nc_utils import ensure_id_col
from python.prepare_data.spatial_id_utils import (
    resolve_grid_id_convention,
    validate_id_coordinate_consistency,
)
from python.blending_process.blend_evaluation_utils import (
    _present_weeks,
    build_formulas_from_spec,
    print_formula_summary,
    apply_formula_sample_support,
    compute_cv_global,
    compute_cv_local,
    compute_cv_neighbors,
    compute_cv_clusters,
    compute_cell_metrics_fast,
    summarize_models_pooled,
    summarize_maps_compare,
    make_raw_preds_from_wide,
    make_raw_preds_from_wide_logit,
    make_raw_preds_from_wide_logit_window,
    make_calibrated_preds_from_wide,
    fit_platt_weights_export,
    platt_cv_multibin,
    input_rds_from_cutoff,
    make_cutoff_tag,
    make_year_tag,
    restrict_to_allowed,
    forecast_label,
    clean_probs5,
    pooled_rps5,
    optimize_mme_weights,
    apply_mme_weights,
)


def main():
    parser = argparse.ArgumentParser(description="Cross-validate weekly-bin blending models.")
    parser.add_argument("--spec_id", default="cv_models",
                        help="Spec file name (without .yml) in specs/2025_blend/")
    parser.add_argument("--cores", type=int, default=None,
                        help="Override number of parallel cores")
    parser.add_argument("--work_dir", default=None,
                        help="If provided, input pkl is read directly from this directory, "
                             "overriding pipeline_input_dir in the spec.")
    parser.add_argument("--input_path", default=None,
                        help="Read the connector pickle at this exact path (highest priority).")
    parser.add_argument("--results_dir", default=None,
                        help="If provided, all outputs are written to this directory, "
                             "overriding pipeline_output_dir in the spec.")
    args = parser.parse_args()

    spec_path = os.path.join("specs", "2025_blend", f"{args.spec_id}.yml")
    with open(spec_path, "r") as f:
        spec = yaml.safe_load(f)

    #path_box = "Monsoon_Data/results/2025_model_evaluation"
    #pipeline_output_dir = spec["run"].get("pipeline_output_dir", "")   # default: no subfolder
    #path_box = os.path.join("Monsoon_Data/results", pipeline_output_dir)
    if args.results_dir:
        path_box = args.results_dir
    else:
        pipeline_output_dir = spec["run"].get("pipeline_output_dir", "")
        path_box = os.path.join("Monsoon_Data/results", pipeline_output_dir)

    os.makedirs(path_box, exist_ok=True)

    cutoff_mode = coalesce(spec.get("run", {}).get("cutoff_mode"), "ref_onset")
    training_years = list(map(int, coalesce(spec.get("run", {}).get("training_years"), list(range(2000, 2025)))))
    true_holdout_years = list(map(int, coalesce(spec.get("run", {}).get("true_holdout_years"), [])))
    cv_holdout_years = list(map(int, coalesce(spec.get("run", {}).get("cv_holdout_years"), [])))
    holdout_years = true_holdout_years + cv_holdout_years

    cv_methods = coalesce(spec.get("cv", {}).get("methods"), ["global"])
    cutoff_tag = make_cutoff_tag(cutoff_mode)
    year_tag = make_year_tag(holdout_years)
    output_tag = f"{cutoff_tag}{year_tag}"

    # Load data
    input_file = input_rds_from_cutoff(cutoff_mode)
    input_override = spec.get("run", {}).get("input_rds_override")
    #input_path = os.path.join("Monsoon_Data/Processed_Data/2025_pipeline_input", input_file)
    # pipeline_input_dir = spec["run"].get("pipeline_input_dir", "")   # default: no subfolder
    # input_path = os.path.join(
    #     "Monsoon_Data/Processed_Data/pipeline_input",
    #     pipeline_input_dir,
    #     input_file,
    # )

    if args.input_path:
        input_path = args.input_path
    elif input_override:
        input_path = input_override
    elif args.work_dir:
        input_path = os.path.join(args.work_dir, input_file)
    else:
        pipeline_input_dir = spec["run"].get("pipeline_input_dir", "")
        input_path = os.path.join(
            "Monsoon_Data/Processed_Data/pipeline_input",
            pipeline_input_dir,
            input_file,
        )

    with open(input_path, "rb") as f:
        wide_df = pickle.load(f)

    wide_df = ensure_id_col(wide_df, spec=spec, context="blend input IDs")
    id_convention = resolve_grid_id_convention(
        spec=spec,
        authoritative_ids=wide_df["id"],
        context="blend input/dissemination ID handoff",
    )
    cell_cfg = spec["cell"]
    dissemination_csv = cell_cfg.get("dissemination", "")
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

    # Number of forecast week-bins, inferred from the connect-stage output
    # (prob_clim_week<n> columns). Drives all bin lists below so a spec with a
    # different days_per_bin / n_bins flows through unchanged.
    N_BINS = max([int(m.group(1)) for c in wide_df.columns
                   for m in [re.match(r"^prob_clim_week(\d+)$", c)] if m] or [4])


#    # --- DIAGNOSTIC
#    mask = wide_df["prob_clim_week1"].isna() & wide_df["ngcm_p_onset_fixed_cutoff_week1"].notna()
#    print(f"Before outcome filter — NaN clim but valid ngcm: {mask.sum()}")
#    print(f"  of which outcome is notna: {(mask & wide_df['outcome'].notna()).sum()}")
#    print(f"  of which outcome is NA:    {(mask & wide_df['outcome'].isna()).sum()}")
#    print(f"  unique ids affected: {wide_df[mask]['id'].nunique()}")
#    print(f"  unique years affected: {sorted(wide_df[mask]['year'].unique())}")
#
#
#    # Load ground truth to count onset samples per cell
#    gt_path = "Monsoon_Data/Processed_Data/Models/imd_clim_mok_date_wide.pkl"
#    with open(gt_path, "rb") as f:
#        gt_wide = pickle.load(f)
#    
#    # Get the 570 ghost cells (NaN clim but valid ngcm, before outcome filter)
#    mask = wide_df["prob_clim_week1"].isna() & wide_df["ngcm_p_onset_fixed_cutoff_week1"].notna()
#    ghost_ids = set(wide_df[mask]["id"].unique())
#    
#    # Count non-NaN onset years per cell in ground truth
#    gt_counts = (
#        gt_wide[gt_wide["id"].isin(ghost_ids)]
#        .groupby("id")
#        .apply(lambda g: pd.Series({
#            "n_years_total": len(g),
#            "n_years_with_onset": g["onset_day"].notna().sum(),
#            "lat": g.name.split("_")[0],
#            "lon": g.name.split("_")[1],
#        }))
#        .reset_index()
#        .sort_values("n_years_with_onset")
#    )
#
#    print(f"\nGround truth sample counts for {len(ghost_ids)} ghost cells:")
#    print(gt_counts.to_string(index=False))
#    # --- DIAGNOSTIC END


#    wide_df = wide_df[wide_df["outcome"].notna()].copy()
#
#    # --- DIAGNOSTIC
#    mask = wide_df["prob_clim_week1"].isna() & wide_df["ngcm_p_onset_fixed_cutoff_week1"].notna()
#    #print(f"\nRows where prob_clim_week1 is NaN but ngcm has a value: {mask.sum()}")
#    #print(wide_df[mask][["id", "year", "time", "outcome", "prob_clim_week1", "ngcm_p_onset_fixed_cutoff_week1"]].to_string())
#
#    clim_cols = [c for c in wide_df.columns if c.startswith("prob_clim")]
#    raw_col = next((c for c in wide_df.columns if "ngcm" in c and "p_onset" in c and "week" in c), None)
#    if clim_cols and raw_col:
#        has_raw   = wide_df[raw_col].notna()
#        no_clim   = wide_df[clim_cols[0]].isna()
#        ghost_ids = wide_df.loc[has_raw & no_clim, "id"].unique()
#        print(f"\nCells with raw ngcm score but no blend output (NaN clim): {len(ghost_ids)}")
#        print(sorted(ghost_ids)[:10])
#    # --- DIAGNOSTIC END


#    # --- DIAGNOSTIC: find the 74 rows ---
#    holdout_mask = wide_df["year"].isin(training_years)  # or holdout_years
#    
#    # Check which columns have NaN in holdout rows
#    clim_cols = [c for c in wide_df.columns if c.startswith("prob_clim")]
#    ngcm_cols = [c for c in wide_df.columns if "ngcm_p_onset" in c and "week" in c]
#    aifs_cols = [c for c in wide_df.columns if "aifs_p_onset" in c and "week" in c]
#    
#    sub = wide_df[wide_df["year"].isin([2019, 2020, 2021, 2022])]
#    print(f"\nTotal holdout rows: {len(sub)}")
#    print(f"\nNaN counts in climatology columns:")
#    print(sub[clim_cols].isna().sum())
#    print(f"\nNaN counts in ngcm columns (sample):")
#    print(sub[ngcm_cols[:4]].isna().sum())
#    print(f"\nNaN counts in aifs columns (sample):")
#    print(sub[aifs_cols[:4]].isna().sum())
#    
#    # Show the rows that have NaN clim but non-NaN ngcm
#    if clim_cols and ngcm_cols:
#        nan_clim = sub[clim_cols[0]].isna()
#        nan_ngcm = sub[ngcm_cols[0]].isna()
#        mismatch = sub[nan_clim & ~nan_ngcm]
#        print(f"\nRows with NaN clim but valid ngcm: {len(mismatch)}")
#        if len(mismatch) > 0:
#            print(mismatch[["id", "lat", "lon", "year", "time", "outcome"] + clim_cols[:2] + ngcm_cols[:2]].head(10))
#            print("\nUnique cells affected:", mismatch["id"].nunique())
#            print("Years affected:", sorted(mismatch["year"].unique()))
#            print("Time range:", mismatch["time"].min(), "to", mismatch["time"].max())
#    # --- END DIAGNOSTIC ---


    # Build formulas before selecting sample support. Only predictors actually
    # referenced by enabled formulas may determine the common evaluation rows.
    formulas = build_formulas_from_spec(spec, cutoff_mode)
    wide_df, support = apply_formula_sample_support(wide_df, formulas)
    print(
        "Formula sample support "
        f"({support['policy']}): {support['rows_before']} -> "
        f"{support['rows_after']} rows "
        f"({support['rows_excluded']} excluded)"
    )
    if support["nonfinite_by_column"]:
        print(f"Non-finite formula predictors: {support['nonfinite_by_column']}")
    support_path = os.path.join(
        path_box, f"formula_sample_support{output_tag}.json"
    )
    with open(support_path, "w") as handle:
        json.dump(support, handle, indent=2)
    print(f"Wrote formula sample-support diagnostics: {support_path}")

    print(f"Loaded {len(wide_df)} supported rows from {input_path}")
    print(f"Formulas: {list(formulas.keys())}")
    print_formula_summary(spec, cutoff_mode)

    # FIX Bug 1: restrict training to allowed cells; predict on all cells
    wide_df_allowed = restrict_to_allowed(wide_df, dissemination_cells)

    all_cells_list = []
    yearly_metrics_list = []

    # Storage for downstream MME blending (matches R's clim_cv_preds_list / cal_preds_store)
    clim_cv_preds_list = {}   # Section 2
    cal_preds_store = {}       # Section 3

    # --------------------------------------------------------------------------
    # Section 1: CV for each multinomial formula
    # --------------------------------------------------------------------------
    for model_name, formula_str in formulas.items():
        print(f"\nCV: {model_name}")
        for method in cv_methods:
            print(f"  Method: {method}")
            try:
                if method == "global":
                    # FIX Bug 1: train on allowed cells only, predict on all cells
                    # OLD:
                    #cv_preds = compute_cv_global(
                    #    formula_str,
                    #    wide_df_allowed,          # <-- training restricted to allowed
                    #    holdout_years,
                    #    true_holdout_years=true_holdout_years,
                    #    data_pred=wide_df,         # <-- predict on all cells
                    #)
                    # NEW:
                    cv_preds, coefs_by_year = compute_cv_global(             # ← NEW (unpack tuple)
                        formula_str,
                        wide_df_allowed,
                        holdout_years,
                        training_years=training_years,
                        true_holdout_years=true_holdout_years,
                        data_pred=wide_df,
                        save_coefs=True,                                     # ← NEW
                        required_classes=interval_bins(N_BINS),
                    )
                    cv_preds = cv_preds.reset_index(drop=True)

                elif method == "local":
                    cv_preds = compute_cv_local(formula_str, wide_df, holdout_years)
                    coefs_by_year = {}
                elif method == "neighbors":
                    cv_preds = compute_cv_neighbors(formula_str, wide_df, holdout_years)
                    coefs_by_year = {}
                elif method.startswith("cluster"):
                    cluster_var = spec.get("cv", {}).get("cluster_var", "cluster")
                    cv_preds = compute_cv_clusters(formula_str, wide_df, holdout_years, cluster_var)
                    coefs_by_year = {}
                else:
                    warnings.warn(f"Unknown CV method: {method}. Skipping.")
                    continue

                if cv_preds is None or cv_preds.empty:
                    raise ValueError(f"No predictions for {model_name}/{method}.")

#                print("\n cv_pred = ", cv_preds)
                # Compute metrics
                cell_metrics = compute_cell_metrics_fast(cv_preds, allowed_cells=dissemination_cells)
                cell_metrics["model"] = model_name
                cell_metrics["cv_method"] = method
                all_cells_list.append(cell_metrics)

                # Yearly metrics
                bins = interval_bins(N_BINS)
                cv_cols = [f"cv_{b}" for b in bins]
                sub = cv_preds.dropna(subset=["outcome"] + cv_cols)
                for yr, g in sub.groupby("year"):
                    P = g[cv_cols].values.astype(float)
                    Y = np.column_stack([(g["outcome"] == b).astype(float) for b in bins])
                    brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
                    rps = pooled_rps5(P, Y)
                    y_long = Y.flatten()
                    p_long = P.flatten()
                    try:
                        from sklearn.metrics import roc_auc_score
                        auc = roc_auc_score(y_long, p_long) if len(np.unique(y_long)) == 2 else np.nan
                    except Exception:
                        auc = np.nan
                    yearly_metrics_list.append({
                        "year": yr, "model": model_name, "cv_method": method,
                        "brier": brier, "rps": rps, "auc": auc,
                    })

                # Save predictions
                pred_path = os.path.join(path_box, f"cv_preds_{model_name}_{method}{output_tag}.pkl")
                with open(pred_path, "wb") as f:
                    pickle.dump(cv_preds, f)

                # OLD — nothing after saving cv_preds

                # NEW — added block:
                if method == "global" and coefs_by_year:                     # ← NEW
                    for yr, coef_df in coefs_by_year.items():                # ← NEW
                        coef_path = os.path.join(                            # ← NEW
                            path_box,                                        # ← NEW
                            f"coefs_{model_name}_{method}{output_tag}_year{yr}.pkl"  # ← NEW
                        )                                                    # ← NEW
                        # BEFORE
                        #with open(coef_path, "wb") as f:                     # ← NEW
                        #    pickle.dump(coef_df, f)                          # ← NEW
                        # AFTER
                        with open(coef_path, "wb") as f:
                            pickle.dump(coefs_by_year[yr], f)  # now a dict with coefs/scaler/features
                    print(f"  Saved coefficients for {len(coefs_by_year)} test years")  # ← NEW

            #except Exception as e:
            #    warnings.warn(f"Error in {model_name}/{method}: {e}")
            #    continue

            except Exception as e:
                raise RuntimeError(f"Error in {model_name}/{method}: {e}") from e

    # --------------------------------------------------------------------------
    # Section 2: Climatology logit baselines
    # --------------------------------------------------------------------------
    extras = spec.get("extras") or {}
    clim_logit_tasks = extras.get("clim_logits") or []
#    print("\n clim_logit_tasks  ,  ", clim_logit_tasks)
#    import sys
#    sys.exit()

    for task in clim_logit_tasks:
        nm = task.get("name") or (lambda: (_ for _ in ()).throw(ValueError("clim_logits task missing 'name'")))(  )
        base_prefix = task.get("base_col_prefix") or (lambda: (_ for _ in ()).throw(ValueError("clim_logits task missing 'base_col_prefix'")))(  )

        win_starts = [int(y) for y in (task.get("window_start_years") or [])]
        win_end = task.get("window_end_year")

        def handle_one_clim(model_nm, start_year=None, end_year=None):
            """Process one clim model variant and store for blending."""
            try:
                if model_nm == "unc_clim_raw":
                    earlier_col = task.get("earlier_col")
                    if not earlier_col:
                        raise ValueError("unc_clim_raw requires extras.clim_logits.earlier_col in YAML")
                    cvp = make_raw_preds_from_wide_logit(
                        wide_df=wide_df,
                        base_col_prefix=base_prefix,
                        holdout_years=holdout_years,
                        earlier_col=earlier_col,
                        earlier_is_logit=bool(task.get("earlier_is_logit", True)),
                        add_cv_earlier=True,
                        renormalize_6=True,
                    )
                else:
                    cvp = make_raw_preds_from_wide_logit_window(
                        wide_df=wide_df,
                        base_col_prefix=base_prefix,
                        holdout_years=holdout_years,
                        start_year=start_year,
                        end_year=end_year,
                    )

#                print("\n cvp = ", cvp)
                if cvp is None or cvp.empty:
                    return

                clim_cv_preds_list[model_nm] = cvp

                cell_metrics = compute_cell_metrics_fast(cvp, allowed_cells=dissemination_cells)
                cell_metrics["model"] = model_nm
                cell_metrics["cv_method"] = "raw"
#                print("cell_metrics = ", cell_metrics)
#                print("dissemination_cells =  ", dissemination_cells)
#                print("wide_df = ", wide_df)
#                import sys
#                exit()
                all_cells_list.append(cell_metrics)

            except Exception as e:
                warnings.warn(f"Error in clim logit '{model_nm}': {e}")

#        handle_one_clim(nm, start_year=None, end_year=None)
#        print("nm   ,  " , nm)
        if not win_starts or win_end is None or nm != "clim_raw":
            handle_one_clim(nm, start_year=None, end_year=None)
        else:
            for sy in win_starts:
                model_nm = nm if sy == 1900 else f"{nm}_{sy}_{win_end}"
                handle_one_clim(model_nm, start_year=sy, end_year=win_end)

    # --------------------------------------------------------------------------
    # Section 3: Forecast tasks (raw + calibrated)
    # --------------------------------------------------------------------------
    for fc_cfg in (extras.get("forecasts") or []):
        fc_name = fc_cfg["name"]
        variant = coalesce(fc_cfg.get("variant"), "base")
        base_label = forecast_label(fc_name, variant)

        if fc_cfg.get("raw", False):
            try:
                preds = make_raw_preds_from_wide(wide_df, fc_name, variant, holdout_years, spec)
                if preds is not None and not preds.empty:
                    lbl = f"{base_label}_raw"
                    cell_metrics = compute_cell_metrics_fast(preds, allowed_cells=dissemination_cells)
                    cell_metrics["model"] = lbl
                    cell_metrics["cv_method"] = "raw"
                    all_cells_list.append(cell_metrics)
            except Exception as e:
                warnings.warn(f"Error in raw forecast '{fc_name}': {e}")

        if fc_cfg.get("calibrated", False):
            try:
                preds = make_calibrated_preds_from_wide(
                    wide_df, fc_name, variant, training_years, holdout_years,
                    true_holdout_years, dissemination_cells, spec,
                )
                if preds is not None and not preds.empty:
                    # Store for MME blending — key matches R's cal_preds_store
                    key = fc_name if variant == "base" else f"{fc_name}_calibrated_{variant}"
                    cal_preds_store[key] = preds

                    model_nm = (f"{fc_name}_calibrated" if variant == "base"
                                else f"{fc_name}_calibrated_{variant}")
                    cell_metrics = compute_cell_metrics_fast(preds, allowed_cells=dissemination_cells)
                    cell_metrics["model"] = model_nm
                    cell_metrics["cv_method"] = "calibrated"
                    all_cells_list.append(cell_metrics)

                    # Save Platt weights if requested
                    if fc_cfg.get("export_platt_weights", False):
                        from python.blending_process.blend_evaluation_utils import (
                            get_forecast_variant_suffix, forecast_prob_cols
                        )
                        suf = get_forecast_variant_suffix(spec, variant)
                        base = f"{fc_name}_p_onset{suf}"
                        weeks = _present_weeks(wide_df, base)
                        if weeks:
                            platt_df_full = wide_df[wide_df["year"].isin(training_years + holdout_years)].copy()
                            wk_labels = [f"week{w}" for w in weeks]
                            for w in weeks:
                                platt_df_full[f"week{w}"] = platt_df_full[f"{base}_week{w}"]
                            platt_df_full["later"] = np.maximum(
                                0.0, 1.0 - platt_df_full[wk_labels].sum(axis=1)
                            )
                            platt_result = fit_platt_weights_export(
                                platt_df_full, wk_labels + ["later"],
                                training_years, year_col="year"
                            )
                            platt_path = os.path.join(
                                path_box,
                                f"platt_weights_{base_label}_calibrated_df{output_tag}.pkl"
                            )
                            with open(platt_path, "wb") as fp:
                                pickle.dump(platt_result["weights_df"], fp)
                            print(f"  Saved Platt weights: {platt_path}")

            except Exception as e:
                warnings.warn(f"Error in calibrated forecast '{fc_name}': {e}")

    # --------------------------------------------------------------------------
    # Section 4: MME blend optimization
    # --------------------------------------------------------------------------
    mme_cfg = spec.get("mme") or {}
    if mme_cfg.get("enabled", False) and clim_cv_preds_list:

        mme_variants = list(mme_cfg.get("variants") or ["fixed_cutoff"])
        blend_models = mme_cfg.get("blend_models") or []
        mc_cores = args.cores or mme_cfg.get("mc_cores", 1)

        bins5 = interval_bins(N_BINS)
        cols5 = [f"cv_{b}" for b in bins5]
        id_vars = ["time", "id", "lat", "lon", "year", "outcome"]

        n_blend = len(blend_models)
        blend_names = [bm["name"] for bm in blend_models]
        w_col_names = [f"w_{nm}" for nm in blend_names]

        do_opt_this_run = (
            bool(mme_cfg.get("optimize_if_holdout_is_full_2000_2024", True)) and
            sorted(holdout_years) == list(range(2000, 2025))
        )

        for mme_variant in mme_variants:
            print(f"\n=== MME optimization (RPS-only) variant: {mme_variant} ===")

            # Resolve sources from blend_models
            mme_sources = {}
            skip = False
            for bm in blend_models:
                if bm.get("source") == "clim":
                    src = clim_cv_preds_list.get(bm["name"])
                    if src is None:
                        warnings.warn(f"MME blend_model '{bm['name']}' (source: clim) not found.")
                        skip = True
                        break
                elif bm.get("source") == "forecast":
                    cal_variant = bm.get("cal_variant", "base")
                    key = bm["name"] if cal_variant == "base" else f"{bm['name']}_calibrated_{cal_variant}"
                    src = cal_preds_store.get(key)
                    if src is None:
                        warnings.warn(f"MME blend_model '{bm['name']}' (key: {key}) not found.")
                        skip = True
                        break
                else:
                    warnings.warn(f"Unknown MME blend_model source: {bm.get('source')}")
                    skip = True
                    break
                mme_sources[bm["name"]] = src

            if skip:
                continue

            # --- Join all sources ---
            join_list = []
            for nm in blend_names:
                src = mme_sources[nm].copy()
                rename_map = {c: f"cv_{nm}_{c[3:]}" for c in cols5 if c in src.columns}
                keep = [v for v in id_vars if v in src.columns] + [c for c in cols5 if c in src.columns]
                join_list.append(src[keep].rename(columns=rename_map))

            mme_base_full = join_list[0]
            for j in range(1, n_blend):
                mme_base_full = mme_base_full.merge(join_list[j], on=id_vars, how="inner")

            mme_base_allowed = restrict_to_allowed(mme_base_full, dissemination_cells)

            # --- Build P matrices ---
            P_full_list = []
            P_allowed_list = []
            for nm in blend_names:
                sel_cols = [f"cv_{nm}_{b}" for b in bins5]
                P_full_list.append(mme_base_full[sel_cols].values.astype(float))
                P_allowed_list.append(mme_base_allowed[sel_cols].values.astype(float))

            Y_allowed = np.column_stack(
                [(mme_base_allowed["outcome"] == b).astype(float) for b in bins5]
            )
            P_allowed_clean = [clean_probs5(P) for P in P_allowed_list]

            # --- Weights file paths ---
            clim_model_name = next(
                (bm["name"] for bm in blend_models if bm.get("source") == "clim"),
                blend_names[0]
            )
            mme_prefix = f"mme_{mme_variant}_{clim_model_name}_opt"
            mme_opt_file_write = os.path.join(
                path_box,
                f"mme_optimized_weights{cutoff_tag}_{clim_model_name}_{mme_variant}{year_tag}.pkl"
            )
            weights_read_tag = mme_cfg.get("weights_year_tag") or year_tag
            mme_opt_file_read = os.path.join(
                path_box,
                f"mme_optimized_weights{cutoff_tag}_{clim_model_name}_{mme_variant}{weights_read_tag}.pkl"
            )

            if do_opt_this_run:
                print("  Running MME optimization...")
                opt_result = optimize_mme_weights(P_allowed_clean, Y_allowed, w_col_names, mc_cores)
                opt_w = opt_result["weights"]
                opt_rps = opt_result["rps"]
                with open(mme_opt_file_write, "wb") as fp:
                    pickle.dump(opt_result["weights_df"], fp)
                print(f"  Saved MME weights: {mme_opt_file_write}")
            else:
                print("  Reading saved MME weights...")
                if not os.path.exists(mme_opt_file_read):
                    warnings.warn(f"MME weights file not found: {mme_opt_file_read}. Skipping.")
                    continue
                with open(mme_opt_file_read, "rb") as fp:
                    saved_weights_df = pickle.load(fp)
                row = saved_weights_df[saved_weights_df["objective"] == "rps"].iloc[0]
                opt_w = np.array([row[w] for w in w_col_names])
                # Recompute RPS with saved weights
                P_mix = sum(wi * Pi for wi, Pi in zip(opt_w, P_allowed_clean))
                P_mix = clean_probs5(P_mix)
                opt_rps = pooled_rps5(P_mix, Y_allowed)

            print(f"  MME weights: {dict(zip(blend_names, np.round(opt_w, 4)))}")
            print(f"  RPS (pooled, allowed): {opt_rps:.6f}")

            # Apply weights to full dataset
            P_mix_full = sum(wi * clean_probs5(Pi) for wi, Pi in zip(opt_w, P_full_list))
            P_mix_full = clean_probs5(P_mix_full)

            cv_preds_mme = mme_base_full[[v for v in id_vars if v in mme_base_full.columns]].copy()
            for j, b in enumerate(bins5):
                cv_preds_mme[f"cv_{b}"] = P_mix_full[:, j]

            mme_model_name = f"{mme_prefix}_rps"
            cell_metrics = compute_cell_metrics_fast(cv_preds_mme, allowed_cells=dissemination_cells)
            cell_metrics["model"] = mme_model_name
            cell_metrics["cv_method"] = "mme"
            all_cells_list.append(cell_metrics)
            print(f"  Added MME model: {mme_model_name}")

    # --------------------------------------------------------------------------
    # Save combined outputs
    # --------------------------------------------------------------------------
    if all_cells_list:
        all_cells = pd.concat(all_cells_list, ignore_index=True)

        summary_path = os.path.join(path_box, f"summary_models{output_tag}.pkl")
        with open(summary_path, "wb") as f:
            pickle.dump(all_cells, f)
        all_cells.to_csv(summary_path.replace(".pkl", ".csv"), index=False)
        print(f"Saved cell metrics: {summary_path}")

#        print("\n all_cells")
        # Pooled summary
        try:
            pooled = summarize_models_pooled(all_cells)
            pooled_path = os.path.join(path_box, f"summary_models_pooled{output_tag}.csv")
            pooled.to_csv(pooled_path, index=False)
            print(f"Saved pooled summary: {pooled_path}")
        except Exception as e:
            warnings.warn(f"Could not compute pooled summary: {e}")

    if yearly_metrics_list:
        yearly_df = pd.DataFrame(yearly_metrics_list)
        yearly_path = os.path.join(path_box, f"yearly_metrics_global{output_tag}.pkl")
        with open(yearly_path, "wb") as f:
            pickle.dump(yearly_df, f)
        yearly_df.to_csv(yearly_path.replace(".pkl", ".csv"), index=False)
        print(f"Saved yearly metrics: {yearly_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
