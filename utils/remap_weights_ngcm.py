# DEPRECATED: superseded by utils/remap.py + python/prepare_data/geometry_utils.py,
# which generalize this file's dynamic-resolution logic and make the shapefile,
# admin key column, and CRS configurable. Prefer:
#     python utils/remap.py weights --shapefile <shp> --sample-nc <dir> --out <csv>
import os
import glob
import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
from shapely.geometry import box

def normalize_adm2_name(adm2_id_series):
    return adm2_id_series.astype(str).str.lower().str.replace(" ", "_")

def standardize_ref_dataset(ds):
    rename_dict = {}
    if 'lat' in ds.coords: rename_dict['lat'] = 'latitude'
    if 'lon' in ds.coords: rename_dict['lon'] = 'longitude'
    if rename_dict:
        ds = ds.rename(rename_dict)
    return ds

def build_grid_to_district_mapping(dirs, shp_file):
    if not os.path.exists(shp_file):
        raise FileNotFoundError(f"Could not find shapefile at {shp_file}")

    gdf = gpd.read_file(shp_file)
    # Ensure we keep the district names
    india = gdf[["adm3_name", "geometry"]].reset_index(drop=True)

    ref_rain_path = dirs["raw"]
    sample_files = sorted(glob.glob(os.path.join(ref_rain_path, "*.nc")))
    
    if not sample_files:
        raise FileNotFoundError(f"No NetCDF files found in {ref_rain_path}")
        
    ds_sample = xr.open_dataset(sample_files[0])
    ds_sample = standardize_ref_dataset(ds_sample)
    lats = ds_sample.latitude.values
    lons = ds_sample.longitude.values
    ds_sample.close()

    # --- DYNAMIC GRID CALCULATION ---
    # Instead of hardcoded 0.125, we calculate half the distance between points
    # For your lon (33.75 - 30.9375 = 2.8125), half_lon will be ~1.406
    # For your lat (~2.79 difference), half_lat will be ~1.395
    def get_half_delta(arr):
        if len(arr) > 1:
            return np.abs(arr[1] - arr[0]) / 2.0
        else:
            return 0.125 # Fallback for single-point datasets
            
    half_lat = get_half_delta(lats)
    half_lon = get_half_delta(lons)

    cell_records = []
    print(f"Building grid polygons (Resolution: {half_lat*2:.4f} lat x {half_lon*2:.4f} lon)...")
    
    for lat in lats:
        for lon in lons:
            # We increase rounding to 4 decimals to support your specific coords 
            # (e.g., 1.395306... needs more than 2 digits)
            lat_val = float(lat)
            lon_val = float(lon)
            
            # Create a box using the calculated spacing
            cell_poly = box(
                lon_val - half_lon, 
                lat_val - half_lat, 
                lon_val + half_lon, 
                lat_val + half_lat
            )
            
            cell_records.append({
                "latitude": lat_val,
                "longitude": lon_val,
                "geometry": cell_poly,
                "cell_area": cell_poly.area,
            })
            
    grid_gdf = gpd.GeoDataFrame(cell_records, crs="EPSG:4326")

    print("Computing intersections...")
    overlaid = gpd.overlay(grid_gdf, india, how="intersection")

    overlaid["intersection_area"] = overlaid.geometry.area
    overlaid["weight"] = overlaid["intersection_area"] / overlaid["cell_area"]
    
    # Filter out negligible overlaps
    overlaid = overlaid[overlaid["weight"] > 1e-5]

    mapping = overlaid[["latitude", "longitude", "adm3_name", "weight"]].reset_index(drop=True)

    mapping_path = os.path.join(dirs["processed"], "grid_to_district_mapping_ngcm.csv")
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    mapping.to_csv(mapping_path, index=False)
    
    print(f"Success! Saved {len(mapping)} mappings to {mapping_path}")
    return mapping

if __name__ == "__main__":
    # Update these paths to your environment
    base_path = "/Users/bodong/Code/project/forecast"
    shp_file = "/Users/bodong/Downloads/eth_admin_boundaries/eth_admin3.shp"
    
    my_dirs = {
        "raw": os.path.join(base_path, "ngcm"),
        "processed": os.path.join(base_path, "processed")
    }

    try:
        df = build_grid_to_district_mapping(my_dirs, shp_file)
        print("\nFirst 5 mappings:")
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")
