#!/usr/bin/env python3
# ==============================================================================
# Script: 1_process_raw_nc_files.py
# ==============================================================================
# Purpose
#   Process raw NetCDF rainfall files into monsoon-onset summary outputs.
#   Reads a YAML spec (specs/raw_data/<spec_id>.yml) and processes each
#   NetCDF file year-by-year, writing per-year results to combined outputs.
#
# Usage (run from MO_Forecast_Code/ directory)
#   python pipelines/prepare_data/1_process_raw_nc_files.py --spec_id ref_rain
#   python pipelines/prepare_data/1_process_raw_nc_files.py --spec_id ngcm
#   python pipelines/prepare_data/1_process_raw_nc_files.py --spec_id aifs
# ==============================================================================

import argparse
import sys
import os

# Allow running from MO_Forecast_Code/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from python.prepare_data.nc_utils import run_single_pipeline


def main():
    parser = argparse.ArgumentParser(description="Process raw NetCDF files into onset tables.")
    parser.add_argument("-i", "--spec_id", required=True,
                        help="Spec ID (loads specs/raw_data/<id>.yml)")
    args = parser.parse_args()

    run_single_pipeline(args.spec_id)


if __name__ == "__main__":
    main()
