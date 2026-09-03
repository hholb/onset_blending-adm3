import argparse
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

from python.prepare_data.geometry_utils import normalize_regrid_weights
from python.prepare_data.sparse_transform_utils import (
    compile_sparse_cell_transform,
    sparse_observed_weighted_mean,
)


def batch_aggregate_to_adm3_matrix(input_dir, mapping_csv_path, input_file=None):
    # 1. Load mapping
    mapping = normalize_regrid_weights(
        pd.read_csv(
            mapping_csv_path,
            dtype={"target_id": str, "adm3_name": str},
        ),
        context=f"regrid weights {mapping_csv_path}",
    )
    mapping = mapping.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    for column in ("lat", "lon", "weight"):
        mapping[column] = pd.to_numeric(mapping[column], errors="raise")

    mapping['source_id'] = list(zip(mapping['lat'], mapping['lon']))
    mapping['target_id'] = mapping['target_id'].astype(str)

    # 2. Process files
    if input_file is not None:
        nc_files = [input_file]
    else:
        nc_files = [f for f in glob.glob(os.path.join(input_dir, "*.nc")) if not f.endswith("_adm3.nc")]

    if not nc_files:
        print("No new .nc files found to process.")
        return

    transforms_by_grid = {}
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

            source_ids = tuple(
                (float(lat), float(lon))
                for lat in ds['lat'].values
                for lon in ds['lon'].values
            )
            transform = transforms_by_grid.get(source_ids)
            if transform is None:
                transform = compile_sparse_cell_transform(
                    mapping,
                    source_ids=source_ids,
                    target_ids=tuple(sorted(mapping['target_id'].unique())),
                )
                transforms_by_grid[source_ids] = transform
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

                    other_dims = [dim for dim in da.dims if dim not in ('lat', 'lon')]
                    ordered = da.transpose(*other_dims, 'lat', 'lon')
                    other_shape = tuple(ordered.sizes[dim] for dim in other_dims)
                    source_values = np.asarray(ordered.values, dtype=float).reshape(
                        (-1, len(source_ids))
                    ).T
                    transformed = sparse_observed_weighted_mean(
                        transform, source_values
                    ).T.reshape(other_shape + (len(transform.target_ids),))
                    result = xr.DataArray(
                        transformed,
                        dims=other_dims + ['adm3_name'],
                        coords={
                            **{dim: ordered.coords[dim] for dim in other_dims},
                            'adm3_name': np.asarray(transform.target_ids, dtype=object),
                        },
                        attrs=da.attrs,
                        name=var_name,
                    )
                    processed_vars[var_name] = result

            # 3. Reconstruct dataset and save
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
        help="Path to the CSV mapping file (lat, lon, target_id or adm3_name, weight).",
    )
    parser.add_argument(
        "--input_file",
        default=None,
        help="Optional single .nc file to process instead of all files in input_dir.",
    )
    args = parser.parse_args()

    batch_aggregate_to_adm3_matrix(args.input_dir, args.weight_file, input_file=args.input_file)
