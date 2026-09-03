#!/usr/bin/env python3
# ==============================================================================
# Script: 3_produce_figures.py
# ==============================================================================
# Purpose
#   Reads pre-computed model evaluation summaries (pkl) and calibration chart
#   data (CSV) and produces publication-ready figures (PDF + SVG/PNG) for the
#   blended monsoon onset forecast paper.
#
# Inputs
#   - Monsoon_Data/results/2025_model_evaluation/summary_models_*.pkl
#   - Monsoon_Data/results/2025_model_evaluation/yearly_metrics_global_*.pkl
#   - Monsoon_Data/results/2025_model_evaluation/evaluation/model_metrics_*.pkl
#   - Monsoon_Data/results/2025_model_evaluation/calibration plots/*.csv
#   - Monsoon_Data/results/2025_model_evaluation/cell_metrics_*.pkl
#   - Monsoon_Data/maps/india_boundary.csv
#
# Outputs
#   - figures/*.pdf, figures/*.svg / figures/*.png
#
# Usage (run from MO_Forecast_Code/ directory)
#   python pipelines/blending_process/3_produce_figures.py
# ==============================================================================

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from python.blending_process.blend_figure_utils import (
    safe_read_pkl,
    save_plot,
    plot_fig4,
    make_weekly_bins_plot,
    make_yearly_plot,
    plot_reliability_3panel,
    plot_metric_map,
    plot_fig2_combined,
)
from python.blending_process.blend_evaluation_utils import (
    summarize_maps_compare,
    build_polygons_for_mapping,
)


# ---------------------- 0. USER SETTINGS ----------------------

OUTPUT_DIR = os.path.join(os.getcwd(), "figures")
FONT_SIZE  = 16
VECTOR_FORMATS_DEFAULT = ["pdf", "svg"]
RASTER_FORMAT_DEFAULT = None
RASTER_DPI_DEFAULT = 300


# ---------------------- 1. LABELS & COLORS ----------------------

MODEL_LABELS = {
    "clim_raw":                         "Evolving Expectations Model",
    "unc_clim_raw":                     "Static Climatology",
    "blended_model":                    "Blended Model",
    "ngcm_raw":                         "NGCM (raw)",
    "aifs_calibrated":                  "AIFS (calibrated)",
    "aifs_calibrated_fixed_cutoff":    "AIFS (calibrated)",
    "ngcm_calibrated":                  "NGCM (calibrated)",
    "ngcm_blend":                       "NGCM : EE",
    "int_all":                          "NGCM : AIFS : EE",
    "ngcm_fixed_cutoff_raw":           "NGCM (raw)",
    "ngcm_calibrated_fixed_cutoff":    "NGCM (calibrated)",
    "mme_fixed_cutoff_clim_raw_opt_rps": "Best MME",
    "mme_no_ref_filter_clim_raw_opt_rps": "Best MME",
}

METRIC_LABELS = {
    "brier_skill": "BSS",
    "rps_skill":   "RPSS",
    "auc":         "AUC",
}

PERIOD_LABELS = {
    "2000-2024":              "2000-2024",
    "1965-1978":              "1965-1978",
    "2000-2024-fixed_cutoff": "2000-2024 (Climatological MOK Date)",
    "2025_MR":                 "2025",
}

OKABE_ITO = {
    "black":         "#000000",
    "orange":        "#E69F00",
    "sky":           "#56B4E9",
    "bluish_green":  "#009E73",
    "yellow":        "#F0E442",
    "blue":          "#0072B2",
    "vermillion":    "#D55E00",
    "reddish_purple": "#CC79A7",
}

MODEL_COLORS_CORE = {
    "unc_clim_raw":                   "#237192",
    "clim_raw":                       "#70c8d4",
    "ngcm_calibrated_fixed_cutoff":  "#eb7900",
    "ngcm_calibrated_ref":            "#eb7900",
    "ngcm_calibrated":                "#eb7900",
    "blended_model":                  "#ca1b00",
    "ngcm_cal":                       "#eb7900",
}

METRIC_COLORS = {
    "brier_skill": "#eb5e71",
    "rps_skill":   "#93c1dc",
    "auc":         "#2b7d87",
}

RELIABILITY_COLORS = {
    "ngcm_fixed_cutoff_raw":          "#c49a2c",
    "ngcm_raw":                        "#c49a2c",
    "ngcm_calibrated_fixed_cutoff":   "#eb7900",
    "ngcm_calibrated":                 "#eb7900",
    "blended_model":                   "#ca1b00",
}

MODEL_COLORS_WEEKLY = {
    "unc_clim_raw":                   "#237192",
    "clim_raw":                       "#70c8d4",
    "ngcm_calibrated_fixed_cutoff":  "#eb7900",
    "ngcm_calibrated_ref":            "#eb7900",
    "blended_model":                  "#ca1b00",
}

FIG2_MODEL_COLORS = {**MODEL_COLORS_CORE}


# ---------------------- 2. FILE LOCATIONS ----------------------

PATHS = {
    "summary_1965_1978":              "Monsoon_Data/results/2025_model_evaluation/summary_models_1965_1978.pkl",
    "summary_1965_1978_fixed_cutoff": "Monsoon_Data/results/2025_model_evaluation/summary_models_1965_1978_clim_mok_date.pkl",
    "summary_1965_1978_no_ref_filter": "Monsoon_Data/results/2025_model_evaluation/summary_models_1965_1978_no_mok_filter.pkl",
    "summary_2000_2024":              "Monsoon_Data/results/2025_model_evaluation/summary_models_2000_2024.pkl",
    "summary_2000_2024_fixed_cutoff": "Monsoon_Data/results/2025_model_evaluation/summary_models_2000_2024_clim_mok_date.pkl",
    "summary_2000_2024_no_ref_filter": "Monsoon_Data/results/2025_model_evaluation/summary_models_2000_2024_no_mok_filter.pkl",
    "summary_2025_mr":                "Monsoon_Data/results/2025_model_evaluation/evaluation/model_metrics_sent_vs_mr_mok_year.pkl",
    "yearly_1965_1978":               "Monsoon_Data/results/2025_model_evaluation/yearly_metrics_global_1965_1978.pkl",
    "yearly_1965_1978_fixed_cutoff": "Monsoon_Data/results/2025_model_evaluation/yearly_metrics_global_clim_mok_date_1965_1978.pkl",
    "yearly_1965_1978_no_ref_filter": "Monsoon_Data/results/2025_model_evaluation/yearly_metrics_global_no_mok_filter_1965_1978.pkl",
    "yearly_2000_2024":               "Monsoon_Data/results/2025_model_evaluation/yearly_metrics_global_2000_2024.pkl",
    "yearly_2000_2024_fixed_cutoff": "Monsoon_Data/results/2025_model_evaluation/yearly_metrics_global_clim_mok_date_2000_2024.pkl",
    "yearly_2000_2024_no_ref_filter": "Monsoon_Data/results/2025_model_evaluation/yearly_metrics_global_no_mok_filter_2000_2024.pkl",
    "yearly_2025":                    "Monsoon_Data/results/2025_model_evaluation/yearly_metrics_2025.pkl",
    "yearly_2025_fixed_cutoff":      "Monsoon_Data/results/2025_model_evaluation/yearly_metrics_2025_clim_mok_date.pkl",
    "yearly_2025_no_ref_filter":      "Monsoon_Data/results/2025_model_evaluation/yearly_metrics_2025_no_mok_filter.pkl",
    "reliability_dir":                "Monsoon_Data/results/2025_model_evaluation/calibration plots",
    "cell_metrics_2000_2024":         "Monsoon_Data/results/2025_model_evaluation/cell_metrics_2000_2024.pkl",
    "boundary_path":                  "Monsoon_Data/maps/india_boundary.csv",
}

# Retarget without editing code: the results base dir and boundary file can be
# overridden via env vars. Defaults above are region-specific example paths
# (values may be region-named; the keys/knobs are generic).
_RESULTS_BASE = "Monsoon_Data/results/2025_model_evaluation"
_results_dir = os.environ.get("BLEND_RESULTS_DIR", _RESULTS_BASE)
if _results_dir != _RESULTS_BASE:
    PATHS = {k: (v.replace(_RESULTS_BASE, _results_dir) if isinstance(v, str) else v)
             for k, v in PATHS.items()}
if os.environ.get("BLEND_BOUNDARY_PATH"):
    PATHS["boundary_path"] = os.environ["BLEND_BOUNDARY_PATH"]


# ---------------------- 3. HELPERS ----------------------

def optional_read_pkl(path):
    if path and os.path.exists(path):
        return safe_read_pkl(path)
    return None


def check_required_cols(df, cols, label):
    if df is None:
        return
    missing = [c for c in cols if c not in df.columns]
    if missing:
        warnings.warn(f"{label} missing columns: {missing}")


def add_period_tag(df, period):
    if df is None:
        return None
    d = df.copy()
    d["period"] = period
    return d


def infer_train_window(model_name):
    """Try to extract training window tag from model name like 'clim_raw_1965_2024'."""
    import re
    m = re.search(r"(\d{4})_(\d{4})$", model_name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return None


def overall_theme(ax, base_size=12):
    """Apply a minimal theme to a matplotlib Axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#ebebeb", linewidth=0.5)
    ax.grid(axis="x", visible=False)
    ax.tick_params(labelsize=base_size - 2)


def make_fig4_variant(summ_df, model_list, title_suffix, file_stem, out_dir):
    if summ_df is None:
        return None
    try:
        fig = plot_fig4(
            df=summ_df,
            model_list=model_list,
            title_suffix=title_suffix,
            model_labels=MODEL_LABELS,
            metric_labels=METRIC_LABELS,
            metric_colors=METRIC_COLORS,
            font_size=FONT_SIZE,
        )
        save_plot(file_stem, fig, out_dir,
                  width=10, height=6,
                  vector_formats=VECTOR_FORMATS_DEFAULT,
                  raster_format=RASTER_FORMAT_DEFAULT,
                  raster_dpi=RASTER_DPI_DEFAULT)
        return fig
    except Exception as e:
        warnings.warn(f"Skipping fig4 ({file_stem}): {e}")
        return None


def make_overall_metrics_figure(pool_df, models_list, period_levels, period_labels_map,
                                file_stem_v, file_stem_h, out_dir):
    """Build vertical (ncol=1) and horizontal (ncol=3) overall-metrics bar plots."""
    metric_order = ["BSS", "RPSS", "AUC"]
    metric_cols  = ["brier_skill", "rps_skill", "auc"]

    long = pool_df[pool_df["model"].isin(models_list)][
        ["model", "period"] + metric_cols
    ].copy()
    long = long.melt(id_vars=["model", "period"], var_name="metric", value_name="value")
    long["metric_display"] = long["metric"].map(METRIC_LABELS)
    long["period_display"] = long["period"].map(period_labels_map).fillna(long["period"])

    pl_pretty = [period_labels_map.get(p, p) for p in period_levels]
    n_periods = len(pl_pretty)
    n_models  = len(models_list)
    bar_width = 0.65 / n_models
    offsets   = np.linspace(-(n_models - 1) * bar_width / 2, (n_models - 1) * bar_width / 2, n_models)

    def _make_panel(metric_display_lbl, ax):
        sub = long[long["metric_display"] == metric_display_lbl]
        for mi, model in enumerate(models_list):
            md = sub[sub["model"] == model]
            xs = []
            ys = []
            for pi, pl in enumerate(period_levels):
                row = md[md["period"] == pl]
                xs.append(pi + offsets[mi])
                ys.append(float(row["value"].iloc[0]) if not row.empty else np.nan)
            color = MODEL_COLORS_CORE.get(model, "#888888")
            label = MODEL_LABELS.get(model, model)
            ax.bar(xs, ys, width=bar_width * 0.9, color=color, label=label)
        ax.set_xticks(range(n_periods))
        ax.set_xticklabels(pl_pretty, fontsize=9)
        ax.set_ylabel(metric_display_lbl, fontsize=11)
        if metric_display_lbl == "AUC":
            ax.set_ylim(bottom=0.5)
        overall_theme(ax)

    # Vertical (3 panels stacked)
    fig_v, axes_v = plt.subplots(3, 1, figsize=(7, 6))
    for i, (mc, ml) in enumerate(zip(metric_cols, metric_order)):
        _make_panel(METRIC_LABELS[mc], axes_v[i])
    handles, labels = axes_v[0].get_legend_handles_labels()
    axes_v[0].legend(handles, labels, fontsize=8, loc="upper left", ncol=2)
    fig_v.tight_layout()
    save_plot(file_stem_v, fig_v, out_dir, width=7, height=6,
              vector_formats=VECTOR_FORMATS_DEFAULT,
              raster_format=RASTER_FORMAT_DEFAULT,
              raster_dpi=RASTER_DPI_DEFAULT)
    plt.close(fig_v)

    # Horizontal (3 panels side-by-side)
    fig_h, axes_h = plt.subplots(1, 3, figsize=(14, 4.5))
    for i, (mc, ml) in enumerate(zip(metric_cols, metric_order)):
        _make_panel(METRIC_LABELS[mc], axes_h[i])
    handles, labels = axes_h[0].get_legend_handles_labels()
    axes_h[0].legend(handles, labels, fontsize=7, loc="upper left", ncol=2)
    fig_h.tight_layout()
    save_plot(file_stem_h, fig_h, out_dir, width=14, height=4.5,
              vector_formats=VECTOR_FORMATS_DEFAULT,
              raster_format=RASTER_FORMAT_DEFAULT,
              raster_dpi=RASTER_DPI_DEFAULT)
    plt.close(fig_h)


def build_overall_variant(summ_main, summ_hist, summ_2025,
                          variant_suffix, ngcm_pref_order=None):
    """Build overall metrics plots for one data variant."""
    if ngcm_pref_order is None:
        ngcm_pref_order = {"ngcm_calibrated_fixed_cutoff": 1, "ngcm_calibrated": 2}

    ngcm_variants = list(ngcm_pref_order.keys())
    pool_parts = [add_period_tag(summ_main, "2000-2024"),
                  add_period_tag(summ_hist, "1965-1978")]
    if summ_2025 is not None:
        pool_parts.append(add_period_tag(summ_2025, "2025_MR"))
    pool = pd.concat([p for p in pool_parts if p is not None], ignore_index=True)

    # Harmonize NGCM variants → ngcm_cal
    pool["_ngcm_rank"] = pool["model"].apply(lambda m: ngcm_pref_order.get(m, 0))
    pool["model"] = pool["model"].apply(lambda m: "ngcm_cal" if m in ngcm_variants else m)
    pool = (pool.sort_values("_ngcm_rank")
            .groupby(["model", "period"], as_index=False).first()
            .drop(columns=["_ngcm_rank"]))

    mf = [m for m in ["unc_clim_raw", "clim_raw", "ngcm_cal", "blended_model"]
          if m in pool["model"].unique()]
    pl_raw = ["2000-2024", "1965-1978"] + (["2025_MR"] if summ_2025 is not None else [])
    pl_raw = [p for p in pl_raw if p in pool["period"].unique()]

    MODEL_LABELS["ngcm_cal"] = "NGCM (calibrated)"
    MODEL_COLORS_CORE["ngcm_cal"] = "#eb7900"

    make_overall_metrics_figure(
        pool_df=pool,
        models_list=mf,
        period_levels=pl_raw,
        period_labels_map=PERIOD_LABELS,
        file_stem_v=f"overall_metrics{variant_suffix}",
        file_stem_h=f"overall_metrics_horizontal{variant_suffix}",
        out_dir=OUTPUT_DIR,
    )


def make_chartdata_path(rel_dir, model, period_tag, bins_tag="10bins"):
    return os.path.join(rel_dir, f"chartdata_{model}_{period_tag}_{bins_tag}.csv")


def read_chartdata_one(path, model, model_pretty, period_pretty):
    """Read one calibration chart-data CSV and tag with model metadata."""
    df = pd.read_csv(path)
    df["model"] = model
    df["model_pretty"] = model_pretty
    df["period"] = period_pretty
    return df


def legend_title_for_metric(metric_col):
    mapping = {
        "brier_skill": "BSS",
        "rps_skill":   "RPSS",
        "auc_diff":    "ΔAUC",
    }
    return mapping.get(metric_col, metric_col)


def read_india_boundary(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------- 4. READ DATA ----------------------
    summ_1965_1978              = safe_read_pkl(PATHS["summary_1965_1978"])
    summ_1965_1978_fixed_cutoff = optional_read_pkl(PATHS["summary_1965_1978_fixed_cutoff"])
    summ_1965_1978_no_ref_filter = optional_read_pkl(PATHS["summary_1965_1978_no_ref_filter"])
    summ_2000_2024              = safe_read_pkl(PATHS["summary_2000_2024"])
    summ_2000_2024_fixed_cutoff = safe_read_pkl(PATHS["summary_2000_2024_fixed_cutoff"])
    summ_2000_2024_no_ref_filter = safe_read_pkl(PATHS["summary_2000_2024_no_ref_filter"])
    summ_2025_mr                = optional_read_pkl(PATHS["summary_2025_mr"])

    yearly_1965_1978              = safe_read_pkl(PATHS["yearly_1965_1978"])
    yearly_1965_1978_fixed_cutoff = optional_read_pkl(PATHS["yearly_1965_1978_fixed_cutoff"])
    yearly_1965_1978_no_ref_filter = optional_read_pkl(PATHS["yearly_1965_1978_no_ref_filter"])
    yearly_2000_2024              = safe_read_pkl(PATHS["yearly_2000_2024"])
    yearly_2000_2024_fixed_cutoff = safe_read_pkl(PATHS["yearly_2000_2024_fixed_cutoff"])
    yearly_2000_2024_no_ref_filter = safe_read_pkl(PATHS["yearly_2000_2024_no_ref_filter"])

    yearly_2025              = optional_read_pkl(PATHS["yearly_2025"])
    yearly_2025_fixed_cutoff = optional_read_pkl(PATHS["yearly_2025_fixed_cutoff"])
    yearly_2025_no_ref_filter = optional_read_pkl(PATHS["yearly_2025_no_ref_filter"])

    # Append 2025 to yearly series if available
    if yearly_2025 is not None and yearly_2000_2024 is not None:
        yearly_2000_2024 = pd.concat([yearly_2000_2024, yearly_2025], ignore_index=True)
    if yearly_2025_fixed_cutoff is not None and yearly_2000_2024_fixed_cutoff is not None:
        yearly_2000_2024_fixed_cutoff = pd.concat(
            [yearly_2000_2024_fixed_cutoff, yearly_2025_fixed_cutoff], ignore_index=True
        )
    if yearly_2025_no_ref_filter is not None and yearly_2000_2024_no_ref_filter is not None:
        yearly_2000_2024_no_ref_filter = pd.concat(
            [yearly_2000_2024_no_ref_filter, yearly_2025_no_ref_filter], ignore_index=True
        )

    required_cols = ["model", "brier_skill", "rps_skill", "auc"]
    for df, lbl in [
        (summ_1965_1978, "summ_1965_1978"),
        (summ_2000_2024, "summ_2000_2024"),
        (summ_2000_2024_fixed_cutoff, "summ_2000_2024_fixed_cutoff"),
        (summ_2000_2024_no_ref_filter, "summ_2000_2024_no_ref_filter"),
    ]:
        check_required_cols(df, required_cols, lbl)

    # ---------------------- 5. OVERALL METRICS BY PERIOD ----------------------
    pool_parts = [add_period_tag(summ_2000_2024, "2000-2024"),
                  add_period_tag(summ_1965_1978, "1965-1978")]
    if summ_2025_mr is not None:
        pool_parts.append(add_period_tag(summ_2025_mr, "2025_MR"))
    first_pool = pd.concat(pool_parts, ignore_index=True)

    ngcm_variants = ["ngcm_calibrated_fixed_cutoff", "ngcm_calibrated"]
    ngcm_pref     = {"ngcm_calibrated_fixed_cutoff": 1, "ngcm_calibrated": 2}
    first_pool["_ngcm_rank"] = first_pool["model"].apply(lambda m: ngcm_pref.get(m, 0))
    first_pool["model"] = first_pool["model"].apply(
        lambda m: "ngcm_cal" if m in ngcm_variants else m
    )
    first_pool = (first_pool.sort_values("_ngcm_rank")
                  .groupby(["model", "period"], as_index=False).first()
                  .drop(columns=["_ngcm_rank"]))

    MODEL_LABELS["ngcm_cal"] = "NGCM (calibrated)"
    MODEL_COLORS_CORE["ngcm_cal"] = "#eb7900"

    models_first = [m for m in ["unc_clim_raw", "clim_raw", "ngcm_cal", "blended_model"]
                    if m in first_pool["model"].unique()]
    period_levels_raw = ["2000-2024", "1965-1978"] + (["2025_MR"] if summ_2025_mr is not None else [])

    make_overall_metrics_figure(
        pool_df=first_pool,
        models_list=models_first,
        period_levels=period_levels_raw,
        period_labels_map=PERIOD_LABELS,
        file_stem_v="overall_metrics",
        file_stem_h="overall_metrics_horizontal",
        out_dir=OUTPUT_DIR,
    )

    # Variant: fixed_cutoff
    if summ_2000_2024_fixed_cutoff is not None:
        build_overall_variant(
            summ_2000_2024_fixed_cutoff,
            summ_1965_1978_fixed_cutoff if summ_1965_1978_fixed_cutoff is not None else summ_1965_1978,
            summ_2025_mr,
            "_fixed_cutoff",
        )

    # Variant: no_ref_filter
    if summ_2000_2024_no_ref_filter is not None:
        build_overall_variant(
            summ_2000_2024_no_ref_filter,
            summ_1965_1978_no_ref_filter if summ_1965_1978_no_ref_filter is not None else summ_1965_1978,
            summ_2025_mr,
            "_no_ref_filter",
            ngcm_pref_order={"ngcm_calibrated": 1},
        )

    # ---------------------- 7. MODEL COMPARISONS (FIG 4 STYLE) ----------------------
    models_fixed_cutoff_bars = [
        "clim_raw", "ngcm_fixed_cutoff_raw", "aifs_calibrated_fixed_cutoff",
        "ngcm_calibrated_fixed_cutoff", "ngcm_blend", "int_all",
        "mme_fixed_cutoff_clim_raw_opt_rps", "blended_model",
    ]
    models_no_ref_filter = [
        "clim_raw", "ngcm_raw", "aifs_calibrated", "ngcm_calibrated",
        "ngcm_blend", "int_all", "mme_no_ref_filter_clim_raw_opt_rps", "blended_model",
    ]

    for summ_df, model_list, title_suffix, file_stem in [
        (summ_2000_2024,              models_fixed_cutoff_bars, "2000-2024",                    "skill_comparison_2000_2024"),
        (summ_2000_2024_fixed_cutoff, models_fixed_cutoff_bars, "2000-2024, Climatological MOK Filter", "skill_comparison_2000_2024_fixed_cutoff_filter"),
        (summ_2000_2024_no_ref_filter, models_no_ref_filter,      "2000-2024, No MOK Filter",   "skill_comparison_2000_2024_no_ref_filter"),
        (summ_1965_1978,              models_fixed_cutoff_bars, "1965-1978",                    "skill_comparison_1965_1978"),
        (summ_1965_1978_fixed_cutoff, models_fixed_cutoff_bars, "1965-1978, Climatological MOK Filter", "skill_comparison_1965_1978_fixed_cutoff_filter"),
        (summ_1965_1978_no_ref_filter, models_no_ref_filter,      "1965-1978, No MOK Filter",   "skill_comparison_1965_1978_no_ref_filter"),
    ]:
        make_fig4_variant(summ_df, model_list, title_suffix, file_stem, OUTPUT_DIR)

    # ---------------------- 8. WEEKLY PERFORMANCE ----------------------
    for summ_df, title_suffix, variant, file_stem in [
        (summ_1965_1978,              "1965-1978",                            "standard",       "skill_by_week_1965_1978"),
        (summ_2000_2024,              "2000-2024",                            "standard",       "skill_by_week_2000_2024"),
        (summ_2000_2024_fixed_cutoff, "2000-2024 (Climatological MOK Filter)", "fixed_cutoff", "skill_by_week_2000_2024_fixed_cutoff_filter"),
        (summ_2000_2024_no_ref_filter, "2000-2024 (No MOK Filter)",            "no_ref_filter", "skill_by_week_2000_2024_no_ref_filter"),
    ]:
        if summ_df is None:
            continue
        try:
            fig = make_weekly_bins_plot(
                df=summ_df,
                title_suffix=title_suffix,
                variant=variant,
                include_later=False,
                model_labels=MODEL_LABELS,
                metric_labels=METRIC_LABELS,
                model_colors=MODEL_COLORS_WEEKLY,
                font_size=FONT_SIZE,
            )
            save_plot(file_stem, fig, OUTPUT_DIR, width=12, height=7.5,
                      vector_formats=VECTOR_FORMATS_DEFAULT,
                      raster_format=RASTER_FORMAT_DEFAULT,
                      raster_dpi=RASTER_DPI_DEFAULT)
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"Skipping weekly plot ({file_stem}): {e}")

    # ---------------------- 9. YEARLY PERFORMANCE ----------------------
    for yearly_df, title_suffix, file_stem, ngcm_model in [
        (yearly_1965_1978,              "1965-1978",   "yearly_metrics_by_model_1965_1978",   None),
        (yearly_2000_2024,              "2000-2025" if (yearly_2000_2024 is not None and 2025 in yearly_2000_2024.get("year", pd.Series()).values) else "2000-2024",
                                                        "yearly_metrics_by_model_2000_2024",   None),
        (yearly_2000_2024_fixed_cutoff, "2000-2025 (Climatological MOK Filter)" if (yearly_2000_2024_fixed_cutoff is not None and 2025 in yearly_2000_2024_fixed_cutoff.get("year", pd.Series()).values) else "2000-2024 (Climatological MOK Filter)",
                                                        "yearly_metrics_by_model_2000_2024_fixed_cutoff", None),
        (yearly_2000_2024_no_ref_filter, "2000-2025 (No MOK Filter)" if (yearly_2000_2024_no_ref_filter is not None and 2025 in yearly_2000_2024_no_ref_filter.get("year", pd.Series()).values) else "2000-2024 (No MOK Filter)",
                                                        "yearly_metrics_by_model_2000_2024_no_ref_filter", "ngcm_calibrated"),
    ]:
        if yearly_df is None:
            continue
        try:
            kwargs = dict(
                yearly_df=yearly_df,
                title_suffix=title_suffix,
                model_labels=MODEL_LABELS,
                metric_labels=METRIC_LABELS,
                okabe_ito=OKABE_ITO,
                font_size=FONT_SIZE,
            )
            if ngcm_model:
                kwargs["ngcm_model"] = ngcm_model
            fig = make_yearly_plot(**kwargs)
            save_plot(file_stem, fig, OUTPUT_DIR, width=12, height=8,
                      vector_formats=VECTOR_FORMATS_DEFAULT,
                      raster_format=RASTER_FORMAT_DEFAULT,
                      raster_dpi=RASTER_DPI_DEFAULT)
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"Skipping yearly plot ({file_stem}): {e}")

    # ---------------------- 9b. COMBINED FIG 2 ----------------------
    for summ_df, yearly_df, variant, file_stem in [
        (summ_2000_2024,              yearly_2000_2024,              "standard",       "fig2_combined_2000_2024"),
        (summ_2000_2024_fixed_cutoff, yearly_2000_2024_fixed_cutoff, "fixed_cutoff", "fig2_combined_2000_2024_fixed_cutoff"),
        (summ_2000_2024_no_ref_filter, yearly_2000_2024_no_ref_filter, "no_ref_filter", "fig2_combined_2000_2024_no_ref_filter"),
        (summ_1965_1978,              yearly_1965_1978,              "standard",       "fig2_combined_1965_1978"),
        (summ_1965_1978_fixed_cutoff, yearly_1965_1978_fixed_cutoff, "fixed_cutoff", "fig2_combined_1965_1978_fixed_cutoff"),
        (summ_1965_1978_no_ref_filter, yearly_1965_1978_no_ref_filter, "no_ref_filter", "fig2_combined_1965_1978_no_ref_filter"),
    ]:
        if summ_df is None or yearly_df is None:
            continue
        try:
            fig = plot_fig2_combined(
                summ_df=summ_df,
                yearly_df=yearly_df,
                variant=variant,
                model_labels=MODEL_LABELS,
                model_colors=FIG2_MODEL_COLORS,
            )
            save_plot(file_stem, fig, OUTPUT_DIR, width=6.5, height=5.8,
                      vector_formats=VECTOR_FORMATS_DEFAULT,
                      raster_format=RASTER_FORMAT_DEFAULT,
                      raster_dpi=RASTER_DPI_DEFAULT)
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"Skipping fig2 ({file_stem}): {e}")

    # ---------------------- 10. RELIABILITY DIAGRAMS ----------------------
    rel_out_dir = os.path.join(OUTPUT_DIR, "reliability")
    os.makedirs(rel_out_dir, exist_ok=True)

    reliability_configs = [
        {"period_tag": "2000_2024",          "label": "2000_2024",             "models": ["ngcm_fixed_cutoff_raw", "ngcm_calibrated_fixed_cutoff", "blended_model"]},
        {"period_tag": "fixed_cutoff_2000_2024", "label": "2000_2024_fixed_cutoff", "models": ["ngcm_fixed_cutoff_raw", "ngcm_calibrated_fixed_cutoff", "blended_model"]},
        {"period_tag": "no_ref_filter_2000_2024", "label": "2000_2024_no_ref_filter", "models": ["ngcm_raw", "ngcm_calibrated", "blended_model"]},
        {"period_tag": "1965_1978",          "label": "1965_1978",             "models": ["ngcm_fixed_cutoff_raw", "ngcm_calibrated_fixed_cutoff", "blended_model"]},
        {"period_tag": "fixed_cutoff_1965_1978", "label": "1965_1978_fixed_cutoff", "models": ["ngcm_fixed_cutoff_raw", "ngcm_calibrated_fixed_cutoff", "blended_model"]},
        {"period_tag": "no_ref_filter_1965_1978", "label": "1965_1978_no_ref_filter", "models": ["ngcm_raw", "ngcm_calibrated", "blended_model"]},
    ]

    for cfg in reliability_configs:
        rel_rows = []
        for model in cfg["models"]:
            path = make_chartdata_path(PATHS["reliability_dir"], model, cfg["period_tag"], "10bins")
            if not os.path.exists(path):
                continue
            model_pretty = MODEL_LABELS.get(model, model)
            rel_rows.append(read_chartdata_one(path, model, model_pretty, cfg["label"]))

        if not rel_rows:
            print(f"Skipping reliability 3-panel for {cfg['label']}: no chart-data files found.")
            continue

        rel_data = pd.concat(rel_rows, ignore_index=True)
        try:
            fig = plot_reliability_3panel(
                rel_data=rel_data,
                model_order=cfg["models"],
                model_labels=MODEL_LABELS,
                reliability_colors=RELIABILITY_COLORS,
                font_size=FONT_SIZE,
            )
            save_plot(
                f"reliability_3panel_{cfg['label']}", fig, rel_out_dir,
                width=7.2, height=2.36,
                vector_formats=VECTOR_FORMATS_DEFAULT,
                raster_format=RASTER_FORMAT_DEFAULT,
                raster_dpi=RASTER_DPI_DEFAULT,
            )
            plt.close(fig)
        except Exception as e:
            warnings.warn(f"Skipping reliability plot ({cfg['label']}): {e}")

    # ---------------------- 11. MAPS ----------------------
    maps_out_dir = os.path.join(OUTPUT_DIR, "maps_2000_2024")
    os.makedirs(maps_out_dir, exist_ok=True)

    cell_2000_2024 = safe_read_pkl(PATHS["cell_metrics_2000_2024"])
    if cell_2000_2024 is not None:
        check_required_cols(cell_2000_2024, ["id", "lat", "lon", "model"], "cell_2000_2024")

        cell_2000_2024 = cell_2000_2024[cell_2000_2024["id"] != "ALL"].copy()
        cell_2000_2024["lat"] = pd.to_numeric(cell_2000_2024["lat"], errors="coerce")
        cell_2000_2024["lon"] = pd.to_numeric(cell_2000_2024["lon"], errors="coerce")

#        boundary_map = read_boundary(PATHS["boundary_path"])
#        if india_map is not None and "id" in india_map.columns:
#            india_map = india_map[india_map["id"] == 253]

        grid_centers = cell_2000_2024[["id", "lat", "lon"]].drop_duplicates()
        allowed_cells = grid_centers

        poly_obj    = build_polygons_for_mapping(grid_centers=grid_centers, allowed_cells=allowed_cells)
        polygons_df = poly_obj["polygons_df"]

        # Dissemination cell overlay
        dissem_cells_path = "Monsoon_Data/dissemination_cells.csv"
        dissem_poly_df = None
        if os.path.exists(dissem_cells_path):
            dissem_raw = pd.read_csv(dissem_cells_path)
            dissem_matched = grid_centers.merge(dissem_raw, on=["lat", "lon"], how="inner")
            if not dissem_matched.empty:
                d_poly_obj = build_polygons_for_mapping(
                    grid_centers=dissem_matched, allowed_cells=dissem_matched
                )
                dissem_poly_df = d_poly_obj["polygons_df"]
                print(f"Dissemination cells matched: {dissem_matched['id'].nunique()} cells")

        # Per-cell comparison (blended_model vs clim_raw)
        CLIM_MODEL  = "clim_raw"
        FINAL_MODEL = "blended_model"
        method_col = "cv_method" if "cv_method" in cell_2000_2024.columns else None
        if method_col:
            method_val = cell_2000_2024[method_col].iloc[0]
            cell_sub = cell_2000_2024[cell_2000_2024[method_col] == method_val]
            cell_comp = summarize_maps_compare(cell_sub, method_val, CLIM_MODEL, FINAL_MODEL)
        else:
            cell_comp = summarize_maps_compare(cell_2000_2024, "default", CLIM_MODEL, FINAL_MODEL)

        map_specs = [
            ("brier_skill", "Brier Skill (Blended Model vs Evolving Expectations Model)", "map_brier_skill_2000_2024"),
            ("rps_skill",   "RPS Skill (Blended Model vs Evolving Expectations Model)",   "map_rps_skill_2000_2024"),
            ("auc_diff",    "AUC Difference (Blended Model - Evolving Expectations Model)", "map_auc_diff_2000_2024"),
        ]
        for metric_col, title_text, file_stem in map_specs:
            try:
                fig = plot_metric_map(
                    cell_metrics=cell_comp,
                    polygons_df=polygons_df,
                    metric_col=metric_col,
                    title_text=title_text,
#                    india_map=india_map,
                    legend_title=legend_title_for_metric(metric_col),
                    dissem_poly_df=dissem_poly_df,
                )
                save_plot(file_stem, fig, maps_out_dir,
                          width=10, height=10,
                          vector_formats=VECTOR_FORMATS_DEFAULT,
                          raster_format=RASTER_FORMAT_DEFAULT,
                          raster_dpi=RASTER_DPI_DEFAULT)
                plt.close(fig)
            except Exception as e:
                warnings.warn(f"Skipping map ({file_stem}): {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
