#!/usr/bin/env python3
"""
Run compact preprocessing timing checks across multiple raw forecast files.

This is a local development harness. It reuses preprocess_timing_harness.py but
summarizes each run instead of printing the full output fingerprint JSON.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from preprocess_timing_harness import (
    DEFAULT_DISSEMINATION_CSV,
    DEFAULT_MAPPING_CSV,
    DEFAULT_MOK_CSV,
    DEFAULT_THRESHOLDS_CSV,
    compare_outputs,
    run_forecast_preprocess,
    table_summary,
)


DEFAULT_MODEL_ROOT = "/Users/hayden/code/ROMP/data/ethiopia"
DEFAULT_MODELS = ["aifs", "gencast"]
DEFAULT_YEARS = [2019, 2020, 2021]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Time preprocessing across multiple models and years."
    )
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
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
    parser.add_argument("--out-dir", default=".preprocess_timing")
    parser.add_argument(
        "--compare-first-model",
        action="store_true",
        help="Compare non-first-model outputs to the same year from the first model.",
    )
    return parser.parse_args()


def build_run_args(args: argparse.Namespace, forecast_nc: Path) -> argparse.Namespace:
    return argparse.Namespace(
        forecast_nc=str(forecast_nc),
        mapping_csv=args.mapping_csv,
        dissemination_csv=args.dissemination_csv,
        thresholds_csv=args.thresholds_csv,
        mok_csv=args.mok_csv,
        variable=args.variable,
        day_dim=args.day_dim,
        rain_prefix=args.rain_prefix,
        min_day=args.min_day,
        max_day=args.max_day,
        window=args.window,
        wet_day_min_mm=args.wet_day_min_mm,
        follow_days=args.follow_days,
        min_dry_days=args.min_dry_days,
        dry_day_min_mm=args.dry_day_min_mm,
        label="scale",
        out_dir=args.out_dir,
        compare_output=None,
    )


def run_one(args: argparse.Namespace, model: str, year: int, run_root: Path) -> dict:
    forecast_nc = Path(args.model_root) / model / f"{year}.nc"
    if not forecast_nc.exists():
        raise FileNotFoundError(f"Missing forecast NetCDF: {forecast_nc}")

    run_dir = run_root / model / str(year)
    run_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    output_df, timings, metadata = run_forecast_preprocess(
        build_run_args(args, forecast_nc),
        run_dir,
    )
    total_seconds = time.perf_counter() - started

    output_path = run_dir / "forecast_preprocess_output.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(output_df, f)

    return {
        "model": model,
        "year": year,
        "forecast_nc": str(forecast_nc),
        "output_path": str(output_path),
        "output": table_summary(output_df),
        "run_metadata": metadata,
        "timings_seconds": timings,
        "total_seconds": total_seconds,
    }


def summarize(records: list[dict]) -> dict:
    rows = []
    for record in records:
        row = {
            "model": record["model"],
            "year": record["year"],
            "total_seconds": record["total_seconds"],
            "rows": record["output"]["shape"][0],
            "cols": record["output"]["shape"][1],
        }
        row.update(record["timings_seconds"])
        rows.append(row)

    df = pd.DataFrame(rows)
    phase_cols = [
        col
        for col in df.columns
        if col not in {"model", "year", "rows", "cols"}
    ]
    return {
        "runs": rows,
        "phase_totals_seconds": {
            col: float(df[col].sum()) for col in phase_cols if col in df
        },
        "phase_means_seconds": {
            col: float(df[col].mean()) for col in phase_cols if col in df
        },
        "by_model_totals_seconds": (
            df.groupby("model")[phase_cols].sum().to_dict(orient="index")
            if not df.empty
            else {}
        ),
    }


def add_cross_model_comparisons(records: list[dict], models: list[str]) -> None:
    baseline_model = models[0]
    baseline_by_year = {
        record["year"]: record["output_path"]
        for record in records
        if record["model"] == baseline_model
    }
    for record in records:
        if record["model"] == baseline_model:
            continue
        baseline_path = baseline_by_year.get(record["year"])
        if not baseline_path:
            continue
        with open(record["output_path"], "rb") as f:
            current = pickle.load(f)
        record["comparison_to_first_model_same_year"] = compare_outputs(
            current,
            baseline_path,
        )


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(args.out_dir) / f"scale_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    records = []
    for model in args.models:
        for year in args.years:
            print(f"Timing {model} {year}...")
            record = run_one(args, model, year, run_root)
            records.append(record)
            timings = record["timings_seconds"]
            print(
                f"  total={record['total_seconds']:.3f}s "
                f"remap={timings.get('remap_to_adm3', 0.0):.3f}s "
                f"process={timings.get('process_forecast', 0.0):.3f}s "
                f"hash={record['output']['hash'][:12]}"
            )

    if args.compare_first_model:
        add_cross_model_comparisons(records, args.models)

    summary = {
        "args": vars(args),
        "records": records,
        "summary": summarize(records),
    }
    summary_path = run_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
