#!/usr/bin/env python3
# ==============================================================================
# Script: utils/remap.py
# ==============================================================================
# One spec/CLI-driven entry point for regridding gridded NetCDF onto admin units.
# Replaces the hand-edited utils/remap_weights.py and utils/remap_weights_ngcm.py
# (both had hardcoded __main__ paths and, in one case, a hardcoded 0.125 grid
# half-cell). Resolution is now auto-detected and geometry is fully configurable.
#
# Two subcommands:
#
#   weights  -- build a grid->admin area-fraction weight table from a boundary
#               shapefile + a sample gridded NetCDF.
#
#     python utils/remap.py weights \
#         --shapefile data/shapefile/admin.shp \
#         --sample-nc Monsoon_Data/raw_nc/aifs \
#         --out Monsoon_Data/grid_to_district_mapping.csv \
#         [--region-id adm3_pcode] [--region-name adm3_name] \
#         [--parent-key adm2_name] [--crs EPSG:4326]
#
#     # or drive everything from a spec yml that contains a `geometry:` block:
#     python utils/remap.py weights --geometry-spec specs/geometry/eth.yml \
#         --sample-nc Monsoon_Data/raw_nc/aifs --out mapping.csv
#
#   apply    -- aggregate gridded NetCDF files to *_adm3.nc using a weight table
#               (thin wrapper over utils/remap_nc.py).
#
#     python utils/remap.py apply \
#         --weights Monsoon_Data/grid_to_district_mapping.csv \
#         --input-dir Monsoon_Data/raw_nc/aifs
# ==============================================================================

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from python.prepare_data.geometry_utils import build_grid_to_admin_weights, get_geometry_cfg


def _cfg_from_args(args):
    """Merge a --geometry-spec yml block (if given) with explicit CLI flags.
    CLI flags win over the spec block; both fall back to canonical defaults."""
    base = {}
    if getattr(args, "geometry_spec", None):
        import yaml
        with open(args.geometry_spec) as f:
            spec = yaml.safe_load(f) or {}
        base = get_geometry_cfg(spec)
    else:
        base = get_geometry_cfg({})

    if getattr(args, "shapefile", None):
        base["shapefile"] = args.shapefile
    if getattr(args, "region_id", None):
        base["region_id_col"] = args.region_id
        base["region_key_col"] = args.region_id
    elif getattr(args, "region_key", None):
        base["region_id_col"] = args.region_key
        base["region_key_col"] = args.region_key
    if getattr(args, "region_name", None):
        base["region_name_col"] = args.region_name
    if getattr(args, "parent_key", None):
        base["parent_key_col"] = args.parent_key
    if getattr(args, "crs", None):
        base["crs"] = args.crs
    if getattr(args, "grid_lat_var", None):
        base["grid_lat_var"] = args.grid_lat_var
    if getattr(args, "grid_lon_var", None):
        base["grid_lon_var"] = args.grid_lon_var
    return base


def cmd_weights(args):
    cfg = _cfg_from_args(args)
    if not cfg.get("shapefile"):
        raise SystemExit("A shapefile is required (via --shapefile or geometry.shapefile in --geometry-spec).")
    build_grid_to_admin_weights(args.sample_nc, cfg, out_csv=args.out)


def cmd_apply(args):
    from utils.remap_nc import batch_aggregate_to_adm3_matrix
    batch_aggregate_to_adm3_matrix(args.input_dir, args.weights, input_file=args.input_file)


def build_parser():
    p = argparse.ArgumentParser(description="Regrid gridded NetCDF onto admin units.")
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("weights", help="Build grid->admin weight table from a shapefile.")
    w.add_argument("--geometry-spec", default=None, help="YAML spec with a `geometry:` block.")
    w.add_argument("--shapefile", default=None, help="Admin boundary shapefile.")
    w.add_argument("--sample-nc", required=True, help="Sample gridded .nc file or a directory of them.")
    w.add_argument("--out", required=True, help="Output weight CSV path.")
    w.add_argument("--region-id", default=None, help="Shapefile stable ID column (default adm3_name).")
    w.add_argument("--region-name", default=None, help="Optional shapefile display-name column.")
    w.add_argument("--region-key", default=None, help="Legacy alias for --region-id.")
    w.add_argument("--parent-key", default=None, help="Optional parent admin column (e.g. adm2_name).")
    w.add_argument("--crs", default=None, help="CRS (default EPSG:4326).")
    w.add_argument("--grid-lat-var", dest="grid_lat_var", default=None, help="NetCDF latitude coord name.")
    w.add_argument("--grid-lon-var", dest="grid_lon_var", default=None, help="NetCDF longitude coord name.")
    w.set_defaults(func=cmd_weights)

    a = sub.add_parser("apply", help="Aggregate gridded .nc to *_adm3.nc using a weight table.")
    a.add_argument("--weights", required=True, help="Weight CSV from `weights` (lat, lon, target_id or adm3_name, weight).")
    a.add_argument("--input-dir", default=None, help="Directory of .nc files to aggregate.")
    a.add_argument("--input-file", default=None, help="Single .nc file (instead of --input-dir).")
    a.set_defaults(func=cmd_apply)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
