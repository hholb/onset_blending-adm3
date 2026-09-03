#!/usr/bin/env python3
# ==============================================================================
# Script: 2_build_climatology.py
# ==============================================================================
# Purpose
#   Fit per-cell KDE climatology models from historical onset dates and
#   produce issue-date probability forecasts over within-season date windows.
#   Supports MULTIPLE climatology runs from a single YAML spec.
#
# Usage (run from MO_Forecast_Code/ directory)
#   python pipelines/prepare_data/2_build_climatology.py --spec_id ref_rain
#   python pipelines/prepare_data/2_build_climatology.py --spec_id ref_rain --run clim_1965_2024
# ==============================================================================

import argparse
import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from python.pipelines._shared.read_spec import load_spec, validate_spec
from python.prepare_data.climatology_utils import (
    get_paths_clim,
    get_climatology_options_from_run,
    read_gt_onset_from_tbl,
    filter_gt_training,
    build_issue_grid,
    compute_all_forecasts,
    write_paired_climatologies,
)


def _pair_key(run):
    opt = run["options"]
    horizons = tuple(
        tuple(sorted(horizon.items())) for horizon in (opt["horizons"] or [])
    )
    return (
        opt["onset_col"],
        opt["train_year_min"],
        opt["train_year_max"],
        run["min_onset_years"],
        opt["test_year_min"],
        opt["test_year_max"],
        opt["season_start_md"],
        opt["issue_end_md"],
        opt["forecast_window"],
        horizons,
        opt["cv_by_year"],
    )


def main():
    parser = argparse.ArgumentParser(description="Build KDE climatology forecasts.")
    parser.add_argument("-i", "--spec_id", required=True,
                        help="Spec ID (loads specs/raw_data/<id>.yml)")
    parser.add_argument("-r", "--run", default=None,
                        help="Optional: run only one climatology key")
    args = parser.parse_args()

    spec = load_spec(args.spec_id, "raw_data")
    spec["id"] = args.spec_id

    def _validate_clim(s):
        if not s.get("climatologies"):
            raise ValueError("Missing or empty 'climatologies' in spec.")
        required = ["train_year_min", "train_year_max", "test_year_min", "test_year_max",
                    "season_start_md", "issue_end_md"]
        for nm, co in s["climatologies"].items():
            miss = [f for f in required if f not in co]
            if miss:
                raise ValueError(f"Run '{nm}' missing fields: {', '.join(miss)}")
            has_fw = co.get("forecast_window") is not None
            has_hz = co.get("horizons") is not None
            if not has_fw and not has_hz:
                raise ValueError(f"Run '{nm}': provide forecast_window or horizons.")
            if has_fw and has_hz:
                raise ValueError(f"Run '{nm}': provide only one of forecast_window or horizons.")

    spec = validate_spec(spec, required_paths=["output.out_dir", "climatologies"], checks=[_validate_clim])

    workers_value = os.environ.get(
        "CLIM_WORKERS", spec.get("climatology_runtime", {}).get("workers", 1)
    )
    try:
        workers = int(workers_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("climatology_runtime.workers must be an integer.") from exc
    if workers < 1:
        raise ValueError("climatology_runtime.workers must be at least 1.")

    paths = get_paths_clim(spec)
    gt_wide_path = paths["gt_path"]
    if not os.path.exists(gt_wide_path):
        raise FileNotFoundError(f"Ground-truth file not found: {gt_wide_path}")

    with open(gt_wide_path, "rb") as f:
        gt_wide = pickle.load(f)

    run_keys = list(spec["climatologies"].keys())
    if args.run is not None:
        if args.run not in run_keys:
            raise ValueError(f"Requested --run '{args.run}' not found. Available: {', '.join(run_keys)}")
        run_keys = [args.run]

    runs = []
    for run_key in run_keys:
        co = spec["climatologies"][run_key]
        opt = get_climatology_options_from_run(co)

        gt = read_gt_onset_from_tbl(gt_wide, onset_col=opt["onset_col"])
        gt_train = filter_gt_training(gt, opt["train_year_min"], opt["train_year_max"])

        # Filter out cells with too few onset observations to fit a reliable KDE
        min_onset_years = co.get("min_onset_years", 10)  # configurable per run in the spec
        onset_counts = gt_train.groupby("id")["onset_day"].count()
        valid_ids = onset_counts[onset_counts >= min_onset_years].index
        excluded = onset_counts[onset_counts < min_onset_years]
        if len(excluded) > 0:
            print(f"[{run_key}] Excluding {len(excluded)} cells with fewer than {min_onset_years} onset years:")
#            print(excluded.sort_values().to_string())
        gt_train = gt_train[gt_train["id"].isin(valid_ids)]
        if gt_train.empty:
            raise ValueError(f"[{run_key}] No cells remain after min_onset_years filter.")
        # End of bug fix

        issue_grid = build_issue_grid(
            opt["test_year_min"], opt["test_year_max"],
            opt["season_start_md"], opt["issue_end_md"],
        )

        out_stem = co.get("out_stem") or f"{paths['out_stem']}_{run_key}"
        runs.append({
            "key": run_key,
            "options": opt,
            "min_onset_years": min_onset_years,
            "gt_train": gt_train,
            "issue_grid": issue_grid,
            "out_pkl": os.path.join(paths["out_dir"], f"{out_stem}.pkl"),
        })

    results = {}
    completed = set()
    for index, run in enumerate(runs):
        if run["key"] in completed:
            continue

        pair = next((
            other for other in runs[index + 1:]
            if other["key"] not in completed
            and other["options"]["conditional"]
            != run["options"]["conditional"]
            and _pair_key(other) == _pair_key(run)
        ), None)

        if pair is not None:
            conditional = run if run["options"]["conditional"] else pair
            unconditional = pair if run["options"]["conditional"] else run
            opt = run["options"]
            write_paired_climatologies(
                gt_train=run["gt_train"],
                issue_grid=run["issue_grid"],
                season_start_md=opt["season_start_md"],
                forecast_window=opt["forecast_window"],
                horizons=opt["horizons"],
                cv_by_year=opt["cv_by_year"],
                conditional_path=conditional["out_pkl"],
                unconditional_path=unconditional["out_pkl"],
                workers=workers,
            )
            for item in (conditional, unconditional):
                print(f"[{item['key']}] Wrote: {item['out_pkl']}")
                results[item["key"]] = {
                    "run": item["key"], "out_pkl": item["out_pkl"]
                }
                completed.add(item["key"])
            continue

        out = compute_all_forecasts(
            gt_train=run["gt_train"],
            issue_grid=run["issue_grid"],
            season_start_md=run["options"]["season_start_md"],
            forecast_window=run["options"]["forecast_window"],
            horizons=run["options"]["horizons"],
            conditional=run["options"]["conditional"],
            cv_by_year=run["options"]["cv_by_year"],
        )

        forecast_tbl = out["forecasts"]
        with open(run["out_pkl"], "wb") as f:
            pickle.dump(forecast_tbl, f)

        print(f"[{run['key']}] Wrote: {run['out_pkl']}")
        results[run["key"]] = {
            "run": run["key"], "out_pkl": run["out_pkl"]
        }
        completed.add(run["key"])

    return [results[run_key] for run_key in run_keys]


if __name__ == "__main__":
    main()
