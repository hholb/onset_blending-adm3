#!/usr/bin/env python3
# ==============================================================================
# Script: 0_connect_prepare_data_to_2025_pipeline.py
# ==============================================================================
# Purpose
#   Convert day-level wide pickle from the prepare_data pipeline into a
#   weekly-bin pickle that 1_blend_evaluation.py expects.
#
# Usage (run from MO_Forecast_Code/ directory)
#   python pipelines/blending_process/0_connect_prepare_data_to_2025_pipeline.py --spec_id connect_ref
#   python pipelines/blending_process/0_connect_prepare_data_to_2025_pipeline.py --spec_id connect_fixed_cutoff
#   python pipelines/blending_process/0_connect_prepare_data_to_2025_pipeline.py --spec_id connect_no_ref_filter
# ==============================================================================

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from python.blending_process.connect_utils import make_cv_rds_from_daylevel


def main():
    parser = argparse.ArgumentParser(
        description="Convert daily combined data to weekly-bin RDS for blending."
    )
    parser.add_argument("--spec_id", default="connect_ref",
                        help="Spec file name (without .yml) in specs/2025_blend/")
    parser.add_argument(
        "--required_probability_prefixes",
        nargs="*",
        default=None,
        help=(
            "Internal wrapper contract listing forecast probability series "
            "needed downstream. Omit to preserve strict legacy behavior."
        ),
    )
    args = parser.parse_args()

    spec_path = os.path.join("specs", "2025_blend", f"{args.spec_id}.yml")
    with open(spec_path, "r") as f:
        spec = yaml.safe_load(f)

    wide_df = make_cv_rds_from_daylevel(
        spec=spec,
        required_probability_prefixes=args.required_probability_prefixes,
    )

    print(f"Wrote: {spec['output_rds']}")
    print(f"Rows: {len(wide_df)}, Cols: {len(wide_df.columns)}")
    print(list(wide_df.columns))


if __name__ == "__main__":
    main()
