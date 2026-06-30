import argparse
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse


def _coord_key(lat, lon, precision=10):
    return (round(float(lat), precision), round(float(lon), precision))


def _build_sparse_weights(mapping, lat_values, lon_values):
    pixel_lookup = {
        _coord_key(lat, lon): idx
        for idx, (lat, lon) in enumerate(
            (lat, lon) for lat in lat_values for lon in lon_values
        )
    }

    adm3_names = pd.Index(mapping["adm3_name"].astype(str).drop_duplicates())
    adm3_lookup = {name: idx for idx, name in enumerate(adm3_names)}

    rows = []
    cols = []
    data = []
    missing_pixels = 0
    for rec in mapping.itertuples(index=False):
        pixel_idx = pixel_lookup.get(_coord_key(rec.lat, rec.lon))
        if pixel_idx is None:
            missing_pixels += 1
            continue
        rows.append(pixel_idx)
        cols.append(adm3_lookup[str(rec.adm3_name)])
        data.append(float(rec.weight))

    if not data:
        raise ValueError("No mapping rows matched the input NetCDF lat/lon coordinates.")
    if missing_pixels:
        print(f"  Skipped {missing_pixels} mapping rows with no matching input pixel.")

    weights = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(lat_values) * len(lon_values), len(adm3_names)),
    )
    return weights, adm3_names.to_numpy()


def _aggregate_data_array_to_adm3(da, weights, adm3_names):
    spatial_dims = ["lat", "lon"]
    other_dims = [dim for dim in da.dims if dim not in spatial_dims]
    ordered = da.transpose(*other_dims, *spatial_dims)

    other_shape = tuple(ordered.sizes[dim] for dim in other_dims)
    values = np.asarray(ordered.values)
    flat = values.reshape((-1, ordered.sizes["lat"] * ordered.sizes["lon"]))

    valid = np.isfinite(flat)
    weighted_sum = weights.T.dot(np.nan_to_num(flat, nan=0.0).T).T
    effective_weight = weights.T.dot(valid.astype(float).T).T

    with np.errstate(invalid="ignore", divide="ignore"):
        out = weighted_sum / effective_weight
    out[effective_weight <= 0] = np.nan
    out = np.asarray(out).reshape(other_shape + (len(adm3_names),))

    coords = {dim: da.coords[dim] for dim in other_dims if dim in da.coords}
    coords["adm3_name"] = adm3_names
    return xr.DataArray(out, dims=other_dims + ["adm3_name"], coords=coords, name=da.name)


def batch_aggregate_to_adm3_matrix(input_dir, mapping_csv_path, input_file=None):
    # 1. Load mapping
    mapping = pd.read_csv(mapping_csv_path)
    mapping = mapping.rename(columns={'latitude': 'lat', 'longitude': 'lon'})

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
            processed_vars = {}
            weights_cache = {}

            for var_name in ds.data_vars:
                if 'lat' in ds[var_name].dims and 'lon' in ds[var_name].dims:
                    da = ds[var_name]

                    # Extra safety: replace any remaining sentinel values with NaN
                    fill_val = da.encoding.get('_FillValue', None) or \
                               da.attrs.get('_FillValue', None) or \
                               da.attrs.get('missing_value', None)
                    if fill_val is not None:
                        da = da.where(da != fill_val)

                    grid_key = (tuple(da["lat"].values), tuple(da["lon"].values))
                    if grid_key not in weights_cache:
                        weights_cache[grid_key] = _build_sparse_weights(
                            mapping, da["lat"].values, da["lon"].values
                        )
                    weights, adm3_names = weights_cache[grid_key]
                    processed_vars[var_name] = _aggregate_data_array_to_adm3(
                        da, weights, adm3_names
                    )

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
