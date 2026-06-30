#!/usr/bin/env python3
"""
Time and fingerprint the raw forecast preprocessing path.

This is a local development harness, not part of the blending pipeline. It uses
the package's existing raw-NetCDF flow:

1. copy one gridded forecast NetCDF into an isolated run directory
2. aggregate it to ADM3 with utils.remap_nc
3. read the generated *_adm3.nc with prepare_data.nc_utils
4. run process_rainfall_forecast_id()
5. save output and a JSON summary for before/after comparisons
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from python.prepare_data.nc_utils import (
    filter_by_dissemination_cells,
    nc_read_forecast_wide,
    process_rainfall_forecast_id,
)
from python.prepare_data.onset_utils import read_mok_dates, read_thresholds
from utils.remap_nc import batch_aggregate_to_adm3_matrix


DEFAULT_FORECAST_NC = "/Users/hayden/code/ROMP/data/ethiopia/aifs/2019.nc"
DEFAULT_MAPPING_CSV = "Monsoon_Data/grid_to_district_mapping.csv"
DEFAULT_DISSEMINATION_CSV = "Monsoon_Data/dissemination_cells.csv"
DEFAULT_THRESHOLDS_CSV = "Monsoon_Data/reference/thresholds_df.csv"
DEFAULT_MOK_CSV = "Monsoon_Data/reference/MOK Onset May.csv"


@contextmanager
def timed(timings: dict[str, float], name: str):
    start = time.perf_counter()
    yield
    timings[name] = time.perf_counter() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time raw NetCDF -> ADM3 -> forecast preprocessing and save comparable outputs."
    )
    parser.add_argument("--forecast-nc", default=DEFAULT_FORECAST_NC)
    parser.add_argument("--mapping-csv", default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--dissemination-csv", default=DEFAULT_DISSEMINATION_CSV)
    parser.add_argument("--thresholds-csv", default=DEFAULT_THRESHOLDS_CSV)
    parser.add_argument("--mok-csv", default=DEFAULT_MOK_CSV)
    parser.add_argument("--variable", default="tp")
    parser.add_argument("--day-dim", default="day")
    parser.add_argument("--rain-prefix", default="rain")
    parser.add_argument("--min-day", type=int, default=1)
    parser.add_argument("--max-day", type=int, default=45)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--wet-day-min-mm", type=float, default=1.0)
    parser.add_argument("--follow-days", type=int, default=21)
    parser.add_argument("--min-dry-days", type=int, default=5)
    parser.add_argument("--dry-day-min-mm", type=float, default=1.0)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--out-dir", default=".preprocess_timing")
    parser.add_argument(
        "--compare-output",
        default=None,
        help="Optional baseline output pickle to compare against this run.",
    )
    return parser.parse_args()


def build_spec(args: argparse.Namespace, raw_dir: Path, output_dir: Path) -> dict[str, Any]:
    return {
        "id": "preprocess_timing_harness",
        "type": "rainfall_forecast",
        "input": {
            "nc_folder": str(raw_dir),
            "file_regex": r".*_adm3\.nc$",
            "value_col": args.variable,
            "wide_day_dim": args.day_dim,
            "wide_prefix": args.rain_prefix,
        },
        "dimensions": {
            "rename": {
                "time": "time",
                "day": "day",
                "number": "number",
            }
        },
        "output": {"out_dir": str(output_dir), "basename": "forecast_preprocess"},
        "options": {
            "min_day": args.min_day,
            "max_day": args.max_day,
            "window": args.window,
            "cell_transform_enabled": False,
            "onset_definition": {
                "wet_day_min_mm": args.wet_day_min_mm,
                "follow_days": args.follow_days,
                "dry_spell": {
                    "mode": "consecutive_dry",
                    "min_dry_days": args.min_dry_days,
                    "dry_day_min_mm": args.dry_day_min_mm,
                },
            },
        },
        "thresholds": {
            "file": args.thresholds_csv,
            "adm3_col": "adm3_name",
            "thresh_col": "onset_threshold",
        },
        "mok": {
            "file": args.mok_csv,
            "year_col": "Year",
            "day_col": "MOK",
            "base_date": "05-01",
        },
        "filter": {
            "dissemination_cells_file": args.dissemination_csv,
        },
    }


def stable_table_hash(df: pd.DataFrame) -> str:
    sort_cols = [c for c in ["id", "time", "year"] if c in df.columns]
    ordered = df.sort_values(sort_cols) if sort_cols else df
    ordered = ordered.reindex(sorted(ordered.columns), axis=1)
    values = pd.util.hash_pandas_object(ordered, index=False).values
    return hashlib.sha256(values.tobytes()).hexdigest()


def table_summary(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number])
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "hash": stable_table_hash(df),
        "numeric_column_sums": {
            col: float(value) for col, value in numeric.sum(axis=0, skipna=True).items()
        },
        "numeric_column_nan_counts": {
            col: int(value) for col, value in numeric.isna().sum(axis=0).items()
        },
    }


def compare_outputs(current: pd.DataFrame, baseline_path: str) -> dict[str, Any]:
    with open(baseline_path, "rb") as f:
        baseline = pickle.load(f)
    if not isinstance(baseline, pd.DataFrame):
        baseline = pd.DataFrame(baseline)

    try:
        pd.testing.assert_frame_equal(
            baseline.reset_index(drop=True),
            current.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        return {"matches": True, "reason": None}
    except AssertionError as exc:
        return {"matches": False, "reason": str(exc)}


def copy_raw_nc(source: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / Path(source).name
    shutil.copy2(source, destination)
    return destination


def run_forecast_preprocess(args: argparse.Namespace, run_dir: Path) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
    timings: dict[str, float] = {}
    raw_dir = run_dir / "raw"
    output_dir = run_dir / "outputs"
    spec = build_spec(args, raw_dir=raw_dir, output_dir=output_dir)

    with timed(timings, "copy_raw"):
        local_nc = copy_raw_nc(args.forecast_nc, raw_dir)

    with timed(timings, "remap_to_adm3"):
        batch_aggregate_to_adm3_matrix(
            input_dir=str(raw_dir),
            mapping_csv_path=args.mapping_csv,
            input_file=str(local_nc),
        )

    adm3_nc = local_nc.with_name(f"{local_nc.stem}_adm3{local_nc.suffix}")
    if not adm3_nc.exists():
        raise FileNotFoundError(f"Expected remapped file not found: {adm3_nc}")

    mok_dt = read_mok_dates(spec)
    thr_dt = read_thresholds(spec)

    with timed(timings, "read_adm3_forecast"):
        forecast_df = nc_read_forecast_wide(
            str(adm3_nc),
            var_name=args.variable,
            dim_rename_map=spec["dimensions"]["rename"],
            spec=spec,
            day_dim=args.day_dim,
            prefix=args.rain_prefix,
        )

    if forecast_df is None:
        raise ValueError(f"Variable '{args.variable}' not found in {adm3_nc}")

    with timed(timings, "filter_cells"):
        forecast_df = filter_by_dissemination_cells(forecast_df, spec)

    with timed(timings, "process_forecast"):
        output_df = process_rainfall_forecast_id(
            forecast_df,
            spec,
            mok_dt=mok_dt,
            thr_dt=thr_dt,
        )["wide"]

    return output_df, timings, {
        "local_nc": str(local_nc),
        "adm3_nc": str(adm3_nc),
        "input_shape_after_filter": list(forecast_df.shape),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    run_dir = out_dir / f"{args.label}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    output_df, timings, run_metadata = run_forecast_preprocess(args, run_dir)
    output_path = run_dir / "forecast_preprocess_output.pkl"

    with timed(timings, "write_artifacts"):
        with open(output_path, "wb") as f:
            pickle.dump(output_df, f)

        summary = {
            "args": vars(args),
            "timings_seconds": timings,
            "run_metadata": run_metadata,
            "output": table_summary(output_df),
            "output_path": str(output_path),
        }
        if args.compare_output:
            summary["comparison"] = compare_outputs(output_df, args.compare_output)

        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary.get("comparison", {}).get("matches") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
