#!/usr/bin/env python3
# ==============================================================================
# Script: utils/compute_thresholds.py
# ==============================================================================
# Compute per-unit onset accumulation thresholds ("threshold y") from a rule,
# writing a CSV that the pipeline consumes via `thresholds.file` in the spec.
# This keeps data-driven threshold rules decoupled from the main pipeline: run
# it once against your ground-truth rainfall, point the spec at the output.
#
# Currently implements the `quantile_accumulation` rule: for each unit, the
# q-quantile of the `window`-day rolling rainfall accumulation, pooled over the
# seasonal days across all years present.
#
# Input rainfall may be:
#   --long-csv    a tidy CSV with columns: <id-col>, time, <value-col>
#                 (e.g. the *_long.pkl exported as CSV, or any daily table)
#
# Usage
#   python utils/compute_thresholds.py \
#       --long-csv rainfall_daily.csv \
#       --id-col adm3_name --value-col precip \
#       --window 3 --q 0.9 \
#       --season-start 05-01 --season-end 07-31 \
#       --out Monsoon_Data/reference/thresholds_df.csv
#
# Output CSV columns: <id-col>, onset_threshold
# ==============================================================================

import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from python.prepare_data.onset_utils import threshold_quantile_accumulation


def _md_to_doy_mask(times, season_start, season_end):
    """Boolean mask selecting rows whose month-day falls in [start, end]."""
    if not season_start and not season_end:
        return np.ones(len(times), dtype=bool)
    t = pd.to_datetime(times)
    md = t.dt.strftime("%m-%d")
    lo = season_start or "01-01"
    hi = season_end or "12-31"
    return (md >= lo) & (md <= hi)


def compute_thresholds(long_df, id_col, value_col, window, q,
                       season_start=None, season_end=None):
    """Return DataFrame [id_col, onset_threshold] via the quantile rule."""
    df = long_df[[id_col, "time", value_col]].copy()
    df = df[df[value_col].notna() | df[value_col].isna()]  # keep all; NaN handled downstream
    mask = _md_to_doy_mask(df["time"], season_start, season_end)
    df = df[mask]
    df = df.sort_values([id_col, "time"])

    series_by_id = {
        uid: g[value_col].to_numpy(dtype=float)
        for uid, g in df.groupby(id_col)
    }
    thr = threshold_quantile_accumulation(series_by_id, window=window, q=q)
    out = pd.DataFrame({id_col: list(thr.keys()), "onset_threshold": list(thr.values())})
    return out.sort_values(id_col).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="Compute per-unit onset thresholds from a rule.")
    p.add_argument("--long-csv", required=True, help="Tidy daily rainfall CSV.")
    p.add_argument("--id-col", default="adm3_name", help="Unit id column (default adm3_name).")
    p.add_argument("--value-col", default="precip", help="Rainfall column (default precip).")
    p.add_argument("--window", type=int, required=True, help="Accumulation window (days).")
    p.add_argument("--q", type=float, required=True, help="Quantile in [0,1], e.g. 0.9.")
    p.add_argument("--season-start", default=None, help="Season start month-day, e.g. 05-01.")
    p.add_argument("--season-end", default=None, help="Season end month-day, e.g. 07-31.")
    p.add_argument("--out", required=True, help="Output thresholds CSV path.")
    args = p.parse_args()

    long_df = pd.read_csv(args.long_csv)
    for c in (args.id_col, "time", args.value_col):
        if c not in long_df.columns:
            raise SystemExit(f"Input missing required column '{c}'. Found: {long_df.columns.tolist()}")

    out = compute_thresholds(
        long_df, args.id_col, args.value_col, args.window, args.q,
        season_start=args.season_start, season_end=args.season_end,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} thresholds to {args.out}")


if __name__ == "__main__":
    main()
