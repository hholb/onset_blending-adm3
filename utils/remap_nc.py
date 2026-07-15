import argparse
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr


def batch_aggregate_to_adm3_matrix(input_dir, mapping_csv_path, input_file=None):
    # 1. Load mapping
    mapping = pd.read_csv(mapping_csv_path)
    mapping = mapping.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

    # 2. Prepare Weights Matrix
    weights_xr = mapping.set_index(['lat', 'lon', 'adm3_name'])['weight'].to_xarray().fillna(0)
    weights_matrix = weights_xr.stack(pixel=['lat', 'lon'])

    # Per-adm3 weight sum (shape: adm3_name) — used for normalization.
    # Must be computed per-adm3, NOT summed over everything (old bug: summed
    # over pixel AND adm3_name together, giving a scalar that made all results NaN).
    weight_sum_per_adm3 = weights_matrix.sum(dim='pixel')  # shape: (adm3_name,)

    # 3. Process Files
    if input_file is not None:
        nc_files = [input_file]
    else:
        nc_files = [f for f in glob.glob(os.path.join(input_dir, "*.nc")) if not f.endswith("_adm3.nc")]

    if not nc_files:
        print("No new .nc files found to process.")
        return

    for file_path in nc_files:
        print(f"Processing: {os.path.basename(file_path)}...")

        # Open with mask_and_scale=True so _FillValue=-99 is automatically
        # masked to NaN before any arithmetic. This was the main data bug:
        # -99 fill values were being included in the weighted sum.
        with xr.open_dataset(file_path, mask_and_scale=True) as ds:
            # Standardize spatial coord names so we always work in lat/lon
            # (input files may use latitude/longitude, e.g. the IMD grids).
            rename_ll = {}
            if 'latitude' in ds.dims or 'latitude' in ds.coords:
                rename_ll['latitude'] = 'lat'
            if 'longitude' in ds.dims or 'longitude' in ds.coords:
                rename_ll['longitude'] = 'lon'
            if rename_ll:
                ds = ds.rename(rename_ll)

            processed_vars = {}

            for var_name in ds.data_vars:
                if 'lat' in ds[var_name].dims and 'lon' in ds[var_name].dims:
                    da = ds[var_name]

                    # Extra safety: replace any remaining sentinel values with NaN
                    fill_val = da.encoding.get('_FillValue', None) or \
                               da.attrs.get('_FillValue', None) or \
                               da.attrs.get('missing_value', None)
                    if fill_val is not None:
                        da = da.where(da != fill_val)

                    # Stack spatial dims
                    da_stacked = da.stack(pixel=['lat', 'lon'])  # (time, pixel)

                    # For each pixel/adm3 pair, set weight to 0 where data is NaN
                    # so NaN pixels don't contribute to either the sum or the
                    # effective weight denominator.
                    valid_mask = da_stacked.notnull()  # (time, pixel)

#                    # Weighted sum of valid data: (time, adm3_name)
#                    da_filled = da_stacked.fillna(0.0)
#                    dist_sum = xr.dot(da_filled, weights_matrix, dims='pixel')
#
#                    # Effective weight sum per (time, adm3_name) — excludes NaN pixels
#                    # Broadcast weights_matrix to (time, pixel) by multiplying with valid_mask
#                    effective_weights = weights_matrix * valid_mask  # (time, pixel, adm3_name)
#                    effective_weight_sum = effective_weights.sum(dim='pixel')  # (time, adm3_name)
#
#                    # Weighted average; NaN where no valid pixels contributed
#                    result = xr.where(effective_weight_sum > 0,
#                                      dist_sum / effective_weight_sum,
#                                      np.nan)

                    # 1. Weighted sum of valid data (You already do this efficiently)
                    da_filled = da_stacked.fillna(0.0)
                    dist_sum = xr.dot(da_filled, weights_matrix, dims='pixel')
                    
                    # 2. OPTIMIZED: Effective weight sum using xr.dot
                    # We treat the boolean mask as 1s and 0s and dot it with the weights
                    effective_weight_sum = xr.dot(valid_mask.astype(float), weights_matrix, dims='pixel')
                    
                    # 3. Weighted average
                    result = xr.where(effective_weight_sum > 0,
                                      dist_sum / effective_weight_sum,
                                      np.nan)


                    # Restore adm3_name coordinate (xr.where can drop it)
                    result['adm3_name'] = weight_sum_per_adm3['adm3_name']
                    processed_vars[var_name] = result

            # 4. Reconstruct Dataset and save
            adm3_ds = xr.Dataset(processed_vars)

            base_name = os.path.splitext(file_path)[0]
            output_path = f"{base_name}_adm3.nc"
            adm3_ds.to_netcdf(output_path)
            print(f"Saved: {os.path.basename(output_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch aggregate gridded .nc files to ADM3 districts using a pixel-to-district weight mapping."
    )
    parser.add_argument(
        "--input_dir",
        #required=True,
        default=None,
        help="Directory containing input .nc files to process.",
    )
    parser.add_argument(
        "--weight_file",
        required=True,
        help="Path to the CSV mapping file (columns: lat, lon, adm3_name, weight).",
    )
    parser.add_argument(
        "--input_file",
        default=None,
        help="Optional single .nc file to process instead of all files in input_dir.",
    )
    args = parser.parse_args()

    batch_aggregate_to_adm3_matrix(args.input_dir, args.weight_file, input_file=args.input_file)
