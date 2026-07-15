# ==============================================================================
# File: blend_figure_utils.py
# ==============================================================================
# Purpose
#   Helper functions for 3_produce_figures.py. Provides safe pickle reading,
#   column checks, plot saving, model label inference, and specialized plot
#   constructors for skill comparisons, weekly bins, yearly time series,
#   reliability diagrams, and metric maps.
# ==============================================================================

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def safe_read_pkl(path):
    """Read pickle with informative error on failure."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def optional_read_pkl(path):
    """Read pickle if file exists, else return None."""
    if not os.path.exists(path):
        return None
    return safe_read_pkl(path)


def check_required_cols(df, required_cols, name_for_error):
    """Stop with error if df is missing any required columns."""
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name_for_error} missing columns: {', '.join(miss)}")


def add_period_tag(df, period_label):
    """Add a 'period' column with a fixed label."""
    df = df.copy()
    df["period"] = period_label
    return df


def fmt_num(x, digits=2):
    """Format a number to a fixed number of decimal places."""
    return f"{x:.{digits}f}"


# ---------------------------------------------------------------------------
# Plot saving
# ---------------------------------------------------------------------------

def save_plot(file_stem, fig, out_dir, width=10, height=7, formats=("pdf", "svg"), dpi=300):
    """Save a matplotlib figure to one or more formats."""
    os.makedirs(out_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(out_dir, f"{file_stem}.{fmt}")
        try:
            fig.savefig(path, bbox_inches="tight", dpi=dpi)
            print(f"Saved: {path}")
        except Exception as e:
            warnings.warn(f"Could not save {path}: {e}")


# ---------------------------------------------------------------------------
# Model label helpers
# ---------------------------------------------------------------------------

def infer_train_window(model):
    """Parse training window label from model name string."""
    m = re.search(r"(\d{4})_(\d{4})", str(model))
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return "all"


def pretty_model_name(m, model_labels):
    """Look up display name for a model string."""
    return model_labels.get(str(m), str(m))


# ---------------------------------------------------------------------------
# Secondary axis helpers
# ---------------------------------------------------------------------------

def sec_axis_inv(x, auc_scale, auc_min):
    """Inverse transformation for AUC secondary axis."""
    return (x - auc_min) * auc_scale


# ---------------------------------------------------------------------------
# Long-format helpers
# ---------------------------------------------------------------------------

def long_format_for_fig4(df, models_keep, auc_scale=1.0, auc_min=0.5, model_labels=None, **kwargs):
    """Pivot model metrics to long format for grouped bar charts."""
    if model_labels is None:
        model_labels = {}
    sub = df[df["model"].isin(models_keep)].copy()
    sub["model_label"] = sub["model"].map(lambda m: model_labels.get(m, m))
    sub["auc_scaled"] = (sub["auc"] - auc_min) * auc_scale
    return sub


# ---------------------------------------------------------------------------
# Figure 4: overall skill bar chart
# ---------------------------------------------------------------------------

def plot_fig4(df, year_label_text="", auc_scale=1.0, auc_min=0.5,
              model_labels=None, figsize=(10, 5)):
    """Horizontal bar chart of Brier skill, RPS skill, and AUC."""
    if model_labels is None:
        model_labels = {}
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, metric, xlabel in zip(
        axes,
        ["brier_skill", "rps_skill", "auc_scaled"],
        ["Brier Skill Score", "RPS Skill Score", "AUC (scaled)"],
    ):
        df["model_label"] = df["model"].map(lambda m: model_labels.get(m, m))
        ax.barh(df["model_label"], df[metric])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(year_label_text)

    fig.tight_layout()
    return fig


def make_fig4_variant(summ_df, models_keep, label_text, file_stem, out_dir,
                       model_labels=None, auc_scale=1.0, auc_min=0.5):
    """Wrapper: create fig4 variant from summary DataFrame."""
    long_df = long_format_for_fig4(summ_df, models_keep, auc_scale=auc_scale,
                                    auc_min=auc_min, model_labels=model_labels)
    fig = plot_fig4(long_df, year_label_text=label_text, auc_scale=auc_scale,
                    auc_min=auc_min, model_labels=model_labels)
    save_plot(file_stem, fig, out_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Weekly bins plot
# ---------------------------------------------------------------------------

def make_weekly_bins_plot(df, title_suffix="", variant="", include_later=True,
                           model_labels=None, figsize=(12, 6)):
    """Faceted bar chart of per-week Brier skill and AUC by model.

    The week bins are inferred from the data (any number of week<n> values),
    so this adapts to the configured n_bins. NOTE: some other figures in this
    module use fixed hand-designed panel layouts (e.g. the 3-panel reliability
    diagram combines week3+4) and assume the 4-week structure.
    """
    if model_labels is None:
        model_labels = {}
    fig, ax = plt.subplots(figsize=figsize)
    _key = "week" if "week" in df.columns else ("bin" if "bin" in df.columns else None)
    if _key is not None:
        wk_nums = sorted({int(m.group(1)) for v in df[_key].astype(str)
                          for m in [re.match(r"^week(\d+)$", v)] if m})
        weeks = [f"week{n}" for n in wk_nums] or ["week1", "week2", "week3", "week4"]
    else:
        weeks = ["week1", "week2", "week3", "week4"]
    if include_later:
        weeks.append("later")

    for w in weeks:
        sub = df[df.get("week", df.get("bin")) == w] if "week" in df.columns or "bin" in df.columns else df
        if sub.empty:
            continue
        ax.bar(sub["model"], sub.get("brier_skill", 0), label=w, alpha=0.7)

    ax.set_title(f"Weekly Brier Skill {title_suffix}")
    ax.set_ylabel("Brier Skill")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Yearly time series plot
# ---------------------------------------------------------------------------

def make_yearly_plot(yearly_df, title_suffix="", model_labels=None, figsize=(12, 5)):
    """Faceted line plot of yearly Brier skill, RPSS, and AUC by model."""
    if model_labels is None:
        model_labels = {}
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)

    for model, g in yearly_df.groupby("model"):
        label = model_labels.get(model, model)
        axes[0].plot(g["year"], g["brier_skill"], label=label)
        axes[1].plot(g["year"], g.get("rpss", g.get("rps_skill", g["brier_skill"])), label=label)
        axes[2].plot(g["year"], g["auc"], label=label)

    for ax, title in zip(axes, ["Brier Skill", "RPSS", "AUC"]):
        ax.set_title(f"{title} {title_suffix}")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_3panel(rel_data, model_order=None, model_labels=None, figsize=(14, 5)):
    """Reliability diagram with one panel per bin present in the data (plots ALL bins)."""
    if model_labels is None:
        model_labels = {}
    _key = "panel" if "panel" in rel_data.columns else ("week" if "week" in rel_data.columns else None)
    if _key is not None:
        panels = sorted(rel_data[_key].dropna().astype(str).unique(),
                        key=lambda v: (int(m.group(1)) if (m := re.match(r"^week(\d+)", v)) else 999))
    else:
        panels = ["week1", "week2", "week3", "week4"]
    n = max(1, len(panels))
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] / 3 * n, figsize[1]))
    axes = np.atleast_1d(axes)

    for ax, panel in zip(axes, panels):
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect")
        sub = rel_data[rel_data.get("panel", rel_data.get("week", "")) == panel] if "panel" in rel_data.columns else rel_data
        for model, g in sub.groupby("model"):
            label = model_labels.get(model, model)
            ax.plot(g["forecast_prob"], g["obs_freq"], marker="o", markersize=4, label=label)
        ax.set_title(panel)
        ax.set_xlabel("Forecast Probability")
        ax.set_ylabel("Observed Frequency")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Map plot
# ---------------------------------------------------------------------------

def read_india_boundary(path):
    """Read India boundary. Returns DataFrame with long/lat/group columns."""
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    return pd.read_csv(path)


def legend_title_for_metric(metric_col):
    if re.search(r"skill|rpss", str(metric_col), re.IGNORECASE):
        return "Skill"
    return "Difference"


def plot_metric_map(cell_metrics, polygons_df, metric_col, title_text="",
                    cmap="RdYlGn", vmin=None, vmax=None, figsize=(8, 8)):
    """Choropleth map of a per-cell metric."""
    merged = polygons_df.merge(cell_metrics[["id", metric_col]], on="id", how="left")

    fig, ax = plt.subplots(figsize=figsize)
    if vmin is None:
        vmin = merged[metric_col].quantile(0.05)
    if vmax is None:
        vmax = merged[metric_col].quantile(0.95)

    for cell_id, g in merged.groupby("id"):
        val = g[metric_col].iloc[0]
        if pd.isna(val):
            color = "lightgray"
        else:
            norm_val = (val - vmin) / max(vmax - vmin, 1e-10)
            norm_val = np.clip(norm_val, 0, 1)
            color = plt.cm.get_cmap(cmap)(norm_val)
        ax.fill(g["lon"], g["lat"], color=color, edgecolor="white", linewidth=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=legend_title_for_metric(metric_col))
    ax.set_title(title_text)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 combined
# ---------------------------------------------------------------------------

def plot_fig2_combined(summ_df, yearly_df, variant="", model_labels=None, figsize=(14, 10)):
    """4-panel combined figure: AUC bars, Brier bars, AUC by year, Brier skill by year."""
    if model_labels is None:
        model_labels = {}
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel A: AUC bar
    ax = axes[0, 0]
    summ_df["model_label"] = summ_df["model"].map(lambda m: model_labels.get(m, m))
    ax.barh(summ_df["model_label"], summ_df["auc"])
    ax.set_title("AUC")
    ax.set_xlabel("AUC")

    # Panel B: Brier skill bar
    ax = axes[0, 1]
    ax.barh(summ_df["model_label"], summ_df.get("brier_skill", 0))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Brier Skill")
    ax.set_xlabel("Brier Skill Score")

    # Panel C: AUC by year
    ax = axes[1, 0]
    for model, g in yearly_df.groupby("model"):
        ax.plot(g["year"], g["auc"], label=model_labels.get(model, model))
    ax.set_title("AUC by Year")
    ax.legend(fontsize=7)

    # Panel D: Brier skill by year
    ax = axes[1, 1]
    for model, g in yearly_df.groupby("model"):
        ax.plot(g["year"], g.get("brier_skill", 0), label=model_labels.get(model, model))
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Brier Skill by Year")
    ax.legend(fontsize=7)

    fig.suptitle(f"Model Performance {variant}", fontsize=14)
    fig.tight_layout()
    return fig
