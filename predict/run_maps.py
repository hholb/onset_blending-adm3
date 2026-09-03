import argparse
import pandas as pd
from pathlib import Path
import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.chdir(REPO_ROOT)

#from utils.maps import make_maps
#from utils.maps_new import make_maps
from utils.maps_new_zone import make_maps


def main():
    parser = argparse.ArgumentParser(
        description="Generate forecast maps from a blended output summary CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage (run from repo root)
--------------------------
    python predict/run_maps.py \\
        --input_file Monsoon_Data/results/wet_spell_aifs_aifs_ens/exports/blend_output_summary_20220617.csv \\
        [--output_path predict/output/] \\
        [--region Ethiopia] \\
        [--ref_onset] \\
        [--all_cells_file Monsoon_Data/dissemination_cells.csv] \\
        [--zoom_to_data]
        """,
    )
    parser.add_argument("--input_file", default=None,
                        help="Path to the blended output summary CSV "
                             "(e.g. blend_output_summary_YYYYMMDD.csv). "
                             "Default: Monsoon_Data/results/wet_spell_aifs_aifs_ens/exports/"
                             "blend_output_summary_20220617.csv")
    parser.add_argument("--output_path", default="predict/output/",
                        help="Directory to write map output files. "
                             "Default: predict/output/")
    parser.add_argument("--region", default="Ethiopia",
                        help="Region name passed to make_maps. Default: Ethiopia")
    parser.add_argument("--ref_onset", action="store_true",
                        help="If set, overlay MOK date on maps.")
    parser.add_argument("--all_cells_file", default=None,
                        help="Optional path to all-cells CSV for background plotting.")
    parser.add_argument("--shapefile", default=None,
                        help="adm3 boundary shapefile (must contain 'adm3_name'). "
                             "Default: predict/data/shapefile/manual_zones_woredas.shp")
    parser.add_argument("--adm2_shapefile", default=None,
                        help="Optional adm2 shapefile for zone overlays/labels "
                             "(must contain 'adm2_name'). Defaults to the adm3 shapefile.")
    parser.add_argument("--zoom_to_data", action="store_true",
                        help="If set, zoom map extent to data bounds.")
    args = parser.parse_args()

    # ── Resolve input file ────────────────────────────────────────────
    if args.input_file:
        fname = args.input_file
    else:
        fname = os.path.join(
            "Monsoon_Data/results/wet_spell_aifs_aifs_ens/exports",
            "blend_output_summary_20220617.csv"
        )

    if not os.path.exists(fname):
        raise FileNotFoundError(f"Input file not found: {fname}")

    print(f"Input file  : {fname}")
    print(f"Output path : {args.output_path}")
    print(f"Region      : {args.region}")

    summary = pd.read_csv(fname, parse_dates=["time"])

    make_maps(
        summary,
        output_path=Path(args.output_path),
        ref_onset=args.ref_onset,
        all_cells_file=args.all_cells_file,
        woreda_shapefile=args.shapefile,
        adm2_shapefile=args.adm2_shapefile,
        use_cartopy=True,
        region=args.region,
        zoom_to_data=args.zoom_to_data,
    )


if __name__ == "__main__":
    main()
