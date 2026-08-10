#!/usr/bin/env python3
# ==============================================================================
# Script: 0_regrid_to_shapefile.py   (OPTIONAL pre-step, run before step 1)
# ==============================================================================
# Purpose
#   Regrid gridded rainfall NetCDFs onto a shapefile's admin units before the
#   normal prepare_data pipeline. Regridding is applied to RAINFALL (the raw
#   NetCDF variables), never to onset probabilities.
#
# Ground-truth-coverage weighting
#   Grid cells overlapping an admin unit may lack ground-truth data (e.g. a
#   rain-gauge grid has no values over the ocean). This script:
#     1. builds area-fraction weights (grid cell -> admin unit) from the shapefile,
#     2. finds the grid cells that actually have ground-truth data and unions them
#        into a "coverage" geometry,
#     3. for the ground truth, drops the no-data cells and renormalizes each
#        unit's weights to sum to 1,
#     4. for each forecast family:
#          - if it shares the ground-truth grid, reuses the SAME weights (identical
#            footprint);
#          - if its grid differs, regrids it onto "the political unit minus the
#            parts where ground-truth data did not exist" (unit  intersect  coverage),
#            normalized per unit to sum to 1.
#
#   Either way, forecast and ground truth cover the same footprint. A coverage
#   report is written listing how many units are affected by missing ground-truth
#   data and the quantiles of the missing-area fraction (overall and for the
#   dissemination area).
#
# Output
#   For each input <name>.nc, writes <name>_adm3.nc alongside it. Also writes the
#   shared/forecast weight tables and the coverage report CSV.
#
# Usage
#   python python/pipelines/prepare_data/0_regrid_to_shapefile.py --spec_id <id>
#     (loads specs/regrid/<id>.yml)
# ==============================================================================

import os
import re
import sys
import glob
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import numpy as np
import pandas as pd
import yaml

from python.prepare_data.geometry_utils import (
    get_geometry_cfg, load_admin_geometry, build_grid_to_admin_weights,
    grid_valid_cells, restrict_weights_to_valid, grid_coords_of, get_half_delta,
    grids_equal, build_coverage_geom, coverage_missing_fraction,
    build_weights_to_coverage, build_grid_cell_units, unit_centroids, CANON_REGION_KEY,
)
from python.prepare_data.spatial_id_utils import (
    ensure_spatial_id_col,
    resolve_grid_id_convention,
)
from utils.remap_nc import batch_aggregate_to_adm3_matrix

_QLABELS = ["min", "p05", "p25", "p50", "p75", "p95", "max"]
_QS = [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]


def _list_nc(folder, regex):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"nc_folder does not exist: {folder}")
    pat = re.compile(regex) if regex else None
    return sorted(
        f for f in glob.glob(os.path.join(folder, "*.nc"))
        if not f.endswith("_adm3.nc") and (pat is None or pat.search(os.path.basename(f)))
    )


def _read_dissemination_ids(path, spec=None, convention=None):
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[regrid] WARNING: dissemination_cells_file not found: {path}")
        return None
    dc = pd.read_csv(path, dtype=str)
    configured_id_col = (spec or {}).get("dissemination_id_col")
    if configured_id_col:
        if configured_id_col not in dc.columns:
            raise ValueError(
                f"dissemination_id_col '{configured_id_col}' not found. "
                f"Found: {dc.columns.tolist()}"
            )
        dc = dc.rename(columns={configured_id_col: "id"})
    elif not any(column in dc.columns for column in ("id", "adm3_name")) \
            and not {"lat", "lon"}.issubset(dc.columns):
        fallback_col = dc.columns[0]
        print(
            "[regrid] WARNING: dissemination file has no explicit spatial key; "
            f"using first column '{fallback_col}' as a legacy fallback."
        )
        dc = dc.rename(columns={fallback_col: "id"})
    dc = ensure_spatial_id_col(
        dc,
        spec=spec,
        convention=convention,
        context="regrid dissemination IDs",
    )
    return set(dc["id"])


def write_coverage_report(missing_df, dissem_ids, out_path, eps=1e-6):
    """Write missing-ground-truth coverage quantiles (overall + dissemination)."""
    def summarize(df, label):
        v = df["missing_frac"].dropna().values
        row = {"scope": label, "n_units": int(len(df)),
               "n_affected": int((v > eps).sum())}
        qq = np.quantile(v, _QS) if len(v) else [np.nan] * len(_QS)
        for lab, val in zip(_QLABELS, qq):
            row[f"missing_frac_{lab}"] = float(val)
        return row

    rows = [summarize(missing_df, "overall")]
    if dissem_ids is not None:
        sub = missing_df[missing_df[CANON_REGION_KEY].astype(str).isin(dissem_ids)]
        rows.append(summarize(sub, "dissemination"))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    # also persist the per-unit fractions for inspection
    per_unit_path = out_path.replace(".csv", "_per_unit.csv")
    missing_df.to_csv(per_unit_path, index=False)
    return rows, per_unit_path


def main(spec_id):
    spec_path = os.path.join("specs", "regrid", f"{spec_id}.yml")
    if not os.path.exists(spec_path):
        raise FileNotFoundError(f"Regrid spec not found: {spec_path}")
    with open(spec_path, encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    cfg = get_geometry_cfg(spec)
    # clip_to_coverage (default True): restrict/renormalize to the ground-truth
    # footprint and apply that same footprint to the forecasts. Set false to get
    # the plain per-dataset regrid (old woreda behavior: remap_nc renormalizes
    # over non-NaN cells per timestep, no shared footprint).
    clip = bool(spec.get("clip_to_coverage", True))

    gt = spec.get("ground_truth") or {}
    if not gt.get("nc_folder"):
        raise ValueError("ground_truth.nc_folder is required in the regrid spec.")
    gt_regex = gt.get("file_regex", r"\.nc$")
    gt_files = _list_nc(gt["nc_folder"], gt_regex)
    if not gt_files:
        raise ValueError(f"No ground-truth .nc matched {gt_regex} in {gt['nc_folder']}")

    forecasts = spec.get("forecasts") or []
    top_win = spec.get("weights_in")               # a single supplied weight CSV for everything
    gt_win = gt.get("weights_in") or top_win       # supplied GT weights (skip computing them)

    def _fam_win(fam):
        return fam.get("weights_in") or top_win

    # We only need the (geometry) computation if some weight table must be built.
    # If you supply all weight files, no shapefile overlay / coverage scan runs.
    need_compute = (gt_win is None) or any(_fam_win(f) is None for f in forecasts)

    ref_dir = os.path.dirname(spec.get("weights_out") or os.path.join("Monsoon_Data", "reference", "x")) or "."
    valid = coverage = units = None

    if need_compute:
        if not gt.get("value_col"):
            raise ValueError("ground_truth.value_col is required when weights are computed.")
        gt_lats, gt_lons = grid_coords_of(gt_files[0], cfg)
        grid_id_convention = None
        if not cfg.get("shapefile"):
            established_target_ids = None
            if gt_win:
                existing_weights = pd.read_csv(
                    gt_win, usecols=[CANON_REGION_KEY], dtype=str
                )
                established_target_ids = existing_weights[CANON_REGION_KEY]
            grid_id_convention = resolve_grid_id_convention(
                spec=cfg,
                lat=np.repeat(gt_lats, len(gt_lons)),
                lon=np.tile(gt_lons, len(gt_lats)),
                authoritative_ids=established_target_ids,
                context="ground-truth grid",
            )
        valid = grid_valid_cells(gt_files, gt["value_col"], cfg.get("grid_lat_var"), cfg.get("grid_lon_var"))
        coverage = build_coverage_geom(valid, get_half_delta(gt_lats), get_half_delta(gt_lons))
        if cfg.get("shapefile"):
            units = load_admin_geometry(cfg)
            mode = f"shapefile ({os.path.basename(cfg['shapefile'])})"
        else:
            units = build_grid_cell_units(
                gt_files[0],
                cfg,
                valid_cells=(valid if clip else None),
                convention=grid_id_convention,
            )
            mode = "ground-truth grid cells (no shapefile)"
            print(
                "[regrid] grid ID convention: "
                f"digits={grid_id_convention.decimal_digits}, "
                f"format={grid_id_convention.number_format} "
                f"({grid_id_convention.source})"
            )
        print(f"[regrid] target units: {mode} | clip_to_coverage={clip}")

    # --- Ground-truth weights (supplied or computed) ---
    if gt_win:
        if not os.path.exists(gt_win):
            raise FileNotFoundError(f"ground_truth.weights_in not found: {gt_win}")
        gt_weights_out = gt_win
        print(f"[regrid] ground-truth weights: using supplied file {gt_win}")
    else:
        print(f"[regrid] building weights from {os.path.basename(gt_files[0])} ...")
        base = build_grid_to_admin_weights(gt_files[0], cfg, admin_gdf=units)
        gt_weights, dropped = (restrict_weights_to_valid(base, valid, renormalize=True) if clip else (base, []))
        if gt_weights.empty:
            raise ValueError("No unit has any ground-truth grid cell - check shapefile/grid overlap.")
        print(f"[regrid] {base[CANON_REGION_KEY].nunique()} units overlap the grid; "
              f"{gt_weights[CANON_REGION_KEY].nunique()} retained{' (ground-truth coverage)' if clip else ''}.")
        if dropped:
            print(f"[regrid] WARNING: {len(dropped)} unit(s) dropped (overlap grid, NO ground-truth cell): "
                  f"{', '.join(map(str, dropped[:20]))}{' ...' if len(dropped) > 20 else ''}")
        gt_weights_out = spec.get("weights_out") or os.path.join(ref_dir, f"shared_regrid_weights_{spec_id}.csv")
        os.makedirs(os.path.dirname(gt_weights_out) or ".", exist_ok=True)
        gt_weights.to_csv(gt_weights_out, index=False)
        print(f"[regrid] ground-truth weight table: {gt_weights_out} ({len(gt_weights)} rows)")

    # --- Per-unit centroids (usable as filter.centroids_file for the bbox filter) ---
    cent_units = units
    if cent_units is None and cfg.get("shapefile"):
        cent_units = load_admin_geometry(cfg)   # weights supplied but shapefile available
    if cent_units is not None:
        cents = unit_centroids(cent_units)
        cent_out = spec.get("centroids_out") or os.path.join(ref_dir, f"unit_centroids_{spec_id}.csv")
        os.makedirs(os.path.dirname(cent_out) or ".", exist_ok=True)
        cents.to_csv(cent_out, index=False)
        print(f"[regrid] unit centroids: {cent_out} ({len(cents)} units) - set as filter.centroids_file for bbox")

    # --- Missing-fraction diagnostics (only when geometry was computed) ---
    if coverage is not None and units is not None:
        missing_df = coverage_missing_fraction(units, coverage)
        dissem_ids = _read_dissemination_ids(
            spec.get("dissemination_cells_file"),
            spec=spec,
            convention=grid_id_convention,
        )
        report_out = spec.get("report_out") or os.path.join(ref_dir, f"regrid_coverage_report_{spec_id}.csv")
        rows, per_unit = write_coverage_report(missing_df, dissem_ids, report_out)
        print(f"[regrid] coverage report: {report_out} (per-unit: {per_unit})")
        for r in rows:
            print(f"    {r['scope']:13s}: {r['n_affected']}/{r['n_units']} units missing some GT; "
                  f"median missing frac {r['missing_frac_p50']:.3f}, p95 {r['missing_frac_p95']:.3f}, max {r['missing_frac_max']:.3f}")
    else:
        print("[regrid] all weights supplied -> skipping weight computation and coverage report.")

    # --- Regrid ground truth ---
    print(f"[regrid] ground_truth: regridding {len(gt_files)} file(s)")
    for fpath in gt_files:
        batch_aggregate_to_adm3_matrix(None, gt_weights_out, input_file=fpath)

    # --- Regrid each forecast family ---
    for fam in forecasts:
        fam_regex = fam.get("file_regex", r"\.nc$")
        files = _list_nc(fam["nc_folder"], fam_regex)
        if not files:
            print(f"[regrid] WARNING: no files matched {fam_regex} in {fam['nc_folder']}; skipping.")
            continue
        fw = _fam_win(fam)
        if fw:
            if not os.path.exists(fw):
                raise FileNotFoundError(f"forecast weights_in not found: {fw}")
            fam_weights_out = fw
            print(f"[regrid] {fam['nc_folder']}: using supplied weights {fw}")
        elif grids_equal(files[0], gt_files[0], cfg):
            fam_weights_out = gt_weights_out
            print(f"[regrid] {fam['nc_folder']}: same grid as ground truth -> reuse weights")
        else:
            fam_weights_out = os.path.join(ref_dir, f"regrid_weights_{spec_id}_{fam.get('value_col', 'fc')}.csv")
            if clip:
                build_weights_to_coverage(files[0], cfg, coverage, out_csv=fam_weights_out, admin_gdf=units)
                print(f"[regrid] {fam['nc_folder']}: different grid -> regridded to unit-intersect-GT-coverage")
            else:
                build_grid_to_admin_weights(files[0], cfg, out_csv=fam_weights_out, admin_gdf=units)
                print(f"[regrid] {fam['nc_folder']}: different grid -> plain regrid to units (no coverage clip)")
        for fpath in files:
            batch_aggregate_to_adm3_matrix(None, fam_weights_out, input_file=fpath)

    print("[regrid] done. Point step-1 specs at the *_adm3.nc outputs (file_regex '..._adm3\\.nc$').")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Optional: regrid gridded rainfall onto a shapefile before prepare_data.")
    ap.add_argument("--spec_id", required=True, help="Regrid spec id (loads specs/regrid/<spec_id>.yml).")
    a = ap.parse_args()
    main(a.spec_id)
