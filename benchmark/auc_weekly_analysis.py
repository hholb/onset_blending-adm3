import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# ── 0. Parse arguments ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot AUC by forecast week.")
parser.add_argument(
    "--case",
    default="dry_spell_aifs_gencast_box6",
    help="Case sub-directory name (default: dry_spell_aifs_gencast_box6)",
)
parser.add_argument(
    "--dir_in",
    default="Monsoon_Data/results/",
    help="Root input directory (default: Monsoon_Data/results/)",
)
parser.add_argument(
    "--data_in",
    default="summary_models_pooled_clim_mok_date_2000_2022.csv",
    help="CSV filename inside <dir_in>/<case>/ (default: summary_models_pooled_clim_mok_date_2000_2022.csv)",
)
parser.add_argument(
    "--input_file",
    default=None,
    help="Full path to input CSV. If given, overrides --dir_in and --data_in.",
)
parser.add_argument(
    "--dir_out",
    default="benchmark/figure/",
    help="Output directory for figures (default: benchmark/figure/)",
)
parser.add_argument(
    "--model",
    default="ngcm",
    help="Model prefix used in CSV key, e.g. 'ngcm', 'gencast', 'aifs_ens' (default: ngcm)",
)
args = parser.parse_args()

# ── Model prefix → display label mapping ─────────────────────────────────────
MODEL_LABELS = {
    "ngcm":    "NGCM",
    "gencast": "GenCast",
    "aifs_ens": "AIFS_ENS",
    "aifs_ens_v2": "AIFS_ENS_v2",
    "aifs": "AIFS",
    "aifs_v2": "AIFS_v2",
}
model_prefix = args.model.lower()
model_key    = f"{model_prefix}_fixed_cutoff_raw"
model_label  = MODEL_LABELS.get(model_prefix, model_prefix.upper())

case   = args.case
dir_out = args.dir_out

# ── 1. Load data ──────────────────────────────────────────────────────────────
if args.input_file is not None:
    file_path = args.input_file
else:
    file_path = os.path.join(args.dir_in, case, args.data_in)

df = pd.read_csv(file_path)
os.makedirs(dir_out, exist_ok=True)

# ── 2. Models to plot ─────────────────────────────────────────────────────────
plot_bins   = ["week1", "week2", "week3", "week4"]
auc_models  = {
    "unc_clim_raw":  "Unc Clim Raw",
    "clim_raw":      "Clim Raw",
    "blended_model": "Blended Model",
    model_key:       model_label,
}
colors = {
    "Unc Clim Raw":  "#8172B2",
    "Clim Raw":      "#4C72B0",
    "Blended Model": "#DD8452",
    model_label:     "#55A868",
}

# ── 3. Bar plot ───────────────────────────────────────────────────────────────
x        = np.arange(len(plot_bins))
width    = 0.19
gap      = 0.02
offsets  = np.array([-1.5, -0.5, 0.5, 1.5]) * (width + gap)

fig, ax = plt.subplots(figsize=(11, 5.5))

for i, (model_key, label) in enumerate(auc_models.items()):
    row  = df[df["model"] == model_key].iloc[0]
    vals = [row[f"auc_{b}"] for b in plot_bins]
    bars = ax.bar(x + offsets[i], vals, width=width,
                  color=colors[label], edgecolor="white", linewidth=0.8,
                  zorder=3, label=label)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=7, fontweight="medium", color="#333333")

ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#cccccc")
ax.set_xticks(x)
ax.set_xticklabels(["Week 1", "Week 2", "Week 3", "Week 4"], fontsize=11)
ax.set_ylabel("AUC", fontsize=11)
ax.set_title("AUC by Forecast Week", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10, framealpha=0.9, edgecolor="#cccccc")
ax.spines[["top", "right"]].set_visible(False)

# Zoom y-axis so bars are distinguishable
ymin = min(df[df["model"].isin(auc_models)][f"auc_{b}"].min() for b in plot_bins) - 0.04
ax.set_ylim(bottom=max(0, ymin))

plt.tight_layout()
fout = os.path.join(dir_out, f"auc_barplot_{case}.png")
plt.savefig(fout, dpi=150, bbox_inches="tight")
plt.show()
