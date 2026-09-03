#!/usr/bin/env python3
"""
Plot per-adm3 forecast probability histograms from weekly_probs netCDF files.

For each adm3_name in --cells_file, produces a bar chart with 4 bins:
  Week 1 (Forecast_p_1), Week 2 (Forecast_p_2),
  Week 3 (Forecast_p_3), Week 4+ (Forecast_p_4 + Forecast_p_later)

Usage:
  python plot_forecast_histograms.py --input_dir /data/nc --output_dir /data/plots \
      --cells_file /data/all_cells.csv
  python plot_forecast_histograms.py --input_dir /data/nc --output_dir /data/plots \
      --cells_file /data/all_cells.csv \
      --input_file /other/path/weekly_probs_20260526_icpac.nc
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import xarray as xr

# ── Bin labels & colours ──────────────────────────────────────────────────────
BIN_LABELS = ["Week 1", "Week 2", "Week 3", "Week 4+"]
#BIN_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
BIN_COLORS = ["grey", "grey", "grey", "grey"]


def load_dataset(input_dir: str, input_file: str | None) -> xr.Dataset:
    """Resolve which .nc file to open and return an xarray Dataset."""
    if input_file:
        path = Path(input_file)
        if not path.is_file():
            sys.exit(f"ERROR: --input_file not found: {input_file}")
        return xr.open_dataset(path)

    # Auto-detect weekly_probs_*.nc files in input_dir
    nc_files = sorted(Path(input_dir).glob("weekly_probs_*.nc"))
    if not nc_files:
        sys.exit(
            f"ERROR: No 'weekly_probs_*.nc' files found in --input_dir: {input_dir}"
        )
    if len(nc_files) > 1:
        sys.exit(
            f"ERROR: Multiple 'weekly_probs_*.nc' files found in {input_dir}:\n"
            + "\n".join(f"  {f}" for f in nc_files)
            + "\nSpecify one with --input_file."
        )
    print(f"Auto-detected: {nc_files[0].name}")
    return xr.open_dataset(nc_files[0])


def load_cells(cells_file: str) -> set[str]:
    """Load the set of adm3_names to plot from a CSV file."""
    path = Path(cells_file)
    if not path.is_file():
        sys.exit(f"ERROR: --cells_file not found: {cells_file}")
    df = pd.read_csv(path)
    if "adm3_name" not in df.columns:
        sys.exit(f"ERROR: --cells_file has no 'adm3_name' column. Found: {df.columns.tolist()}")
    names = set(df["adm3_name"].dropna().str.strip())
    print(f"Loaded {len(names)} adm3 names from {path.name}")
    return names


def plot_adm3(name: str, values: dict[str, float], output_dir: Path,
              issue_date: str | None) -> None:
    """Render and save a single histogram for one adm3_name (plots ALL bins)."""
    bin_nums = sorted(int(m.group(1)) for k in values
                      for m in [re.match(r"^Forecast_p_(\d+)$", k)] if m)
    bin_labels = [f"Week {i}" for i in bin_nums] + ["Later"]
    bin_colors = ["grey"] * len(bin_labels)
    heights = [values[f"Forecast_p_{i}"] for i in bin_nums] + [values["Forecast_p_later"]]

    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(bin_labels)), 4))

    x = np.arange(len(bin_labels))
    bars = ax.bar(x, heights, width=1.0, color=bin_colors,
                  edgecolor="white", linewidth=0.5)

    # Value labels on each bar
    for bar, h in zip(bars, heights):
        if h > 0.01:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.005,
                f"{h:.2f}",
                ha="center", va="bottom", fontsize=8, color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_xlim(-0.5, len(bin_labels) - 0.5)
    ax.set_ylim(0, min(1.0, max(heights) * 1.25 + 0.05))
    ax.set_ylabel("Probability", fontsize=9)
    ax.set_xlabel("Forecast week", fontsize=9)

    title = f"{name}"
    if issue_date:
        title += f"\n(issued {issue_date})"
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()

    # Sanitise filename
    safe_name = (
        name.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("'", "")
    )
    out_path = output_dir / f"{safe_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-adm3 forecast probability histograms."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory containing the weekly_probs .nc file.",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where PNG histograms will be saved.",
    )
    parser.add_argument(
        "--cells_file", required=True,
        help="Path to all_cells.csv; only adm3_names listed here will be plotted.",
    )
    parser.add_argument(
        "--input_file", default=None,
        help="(Optional) Explicit path to a .nc file; overrides --input_dir.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_names = load_cells(args.cells_file)

    print("Loading dataset …")
    ds = load_dataset(args.input_dir, args.input_file)

    issue_date: str | None = ds.attrs.get("issue_date", None)
    adm3_names: list[str] = ds["adm3_name"].values.tolist()
    print(f"Found {len(adm3_names)} adm3 regions in dataset. Issue date: {issue_date or 'unknown'}")

    # Detect however many Forecast_p_<n> bins the file contains (plot all of them).
    bin_nums = sorted(int(m.group(1)) for v in ds.data_vars
                      for m in [re.match(r"^Forecast_p_(\d+)$", str(v))] if m)
    required_vars = [f"Forecast_p_{i}" for i in bin_nums] + ["Forecast_p_later"]
    missing = [v for v in required_vars if v not in ds.data_vars]
    if missing:
        sys.exit(f"ERROR: Missing variables in dataset: {missing}")

    # Filter to only names present in cells_file
    filtered = [(i, name) for i, name in enumerate(adm3_names) if name in allowed_names]
    not_found = allowed_names - {name for _, name in filtered}
    if not_found:
        print(f"WARNING: {len(not_found)} name(s) in cells_file not found in dataset: "
              + ", ".join(sorted(not_found)))
    print(f"Plotting {len(filtered)} regions …")

    # Pre-load all arrays to numpy for fast indexing
    data = {v: ds[v].values for v in required_vars}

    for count, (i, name) in enumerate(filtered, 1):
        vals = {v: float(data[v][i]) for v in required_vars}
        plot_adm3(name, vals, output_dir, issue_date)

        if count % 50 == 0 or count == len(filtered):
            print(f"  {count}/{len(filtered)} done …")

    print(f"\nAll {len(filtered)} histograms saved to: {output_dir}")


if __name__ == "__main__":
    main()
