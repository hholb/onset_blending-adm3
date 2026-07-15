# DEPRECATED: superseded by utils/remap.py + python/prepare_data/geometry_utils.py.
# This version hardcodes the grid half-cell to 0.125 (0.25-deg grids only) and
# has hardcoded paths in __main__. Prefer:
#     python utils/remap.py weights --shapefile <shp> --sample-nc <dir> --out <csv>
# which auto-detects resolution and reads the admin key column / CRS from config.
import os
import glob
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box

def normalize_adm2_name(adm2_id_series):
    """
    CIL convention: lowercase and underscores.
    Adjust this logic if your adm2_id is numerical or a different string.
    """
    return adm2_id_series.astype(str).str.lower().str.replace(" ", "_")

def standardize_ref_dataset(ds):
    """
    Ensures the dataset has 'latitude' and 'longitude' names.
    IMD files sometimes use 'lat' and 'lon'.
    """
    rename_dict = {}
    if 'lat' in ds.coords: rename_dict['lat'] = 'latitude'
    if 'lon' in ds.coords: rename_dict['lon'] = 'longitude'
    if rename_dict:
        ds = ds.rename(rename_dict)
    return ds

# --- THE MAIN FUNCTION ---

def build_grid_to_district_mapping(dirs, shp_file):
    #shp_file = os.path.join(dirs["raw"], "hultgren_cil_yield_data", "shapes", "all_countries.shp")
    
    if not os.path.exists(shp_file):
        raise FileNotFoundError(f"Could not find shapefile at {shp_file}")

    gdf = gpd.read_file(shp_file)
    #india = gdf[gdf["iso"] == "IND"].copy()
    india = gdf.copy()

    # Note: Using 'adm2_name' or 'adm2_id' depending on your shapefile columns
    #india["adm2_name"] = normalize_adm2_name(india["adm2_id"])
    india = india[["adm3_name", "geometry"]].reset_index(drop=True)

    #ref_rain_path = os.path.join(dirs["raw"], "aifs")
    ref_rain_path = dirs["raw"]
    sample_files = sorted(glob.glob(os.path.join(ref_rain_path, "*.nc"))) # Adjusted to catch all .nc
    
    if not sample_files:
        raise FileNotFoundError(f"No IMD files found in {ref_rain_path}")
        
    ds_sample = xr.open_dataset(sample_files[0])
    ds_sample = standardize_ref_dataset(ds_sample)
    lats = ds_sample.latitude.values
    lons = ds_sample.longitude.values
    ds_sample.close()

    half = 0.125
    cell_records = []
    print("Building grid polygons...")
    for lat in lats:
        for lon in lons:
            lat_r = round(float(lat), 2)
            lon_r = round(float(lon), 2)
            cell_poly = box(lon_r - half, lat_r - half, lon_r + half, lat_r + half)
            cell_records.append({
                "latitude": lat_r,
                "longitude": lon_r,
                "geometry": cell_poly,
                "cell_area": cell_poly.area,
            })
    grid_gdf = gpd.GeoDataFrame(cell_records, crs="EPSG:4326")

    print("Computing intersections (this may take a few minutes)...")
    overlaid = gpd.overlay(grid_gdf, india, how="intersection")

    overlaid["intersection_area"] = overlaid.geometry.area
    overlaid["weight"] = overlaid["intersection_area"] / overlaid["cell_area"]
    overlaid = overlaid[overlaid["weight"] > 1e-5]

    mapping = overlaid[["latitude", "longitude", "adm3_name", "weight"]].reset_index(drop=True)

    mapping_path = os.path.join(dirs["processed"], "grid_to_district_mapping.csv")
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    mapping.to_csv(mapping_path, index=False)
    
    print(f"Success! Saved to {mapping_path}")
    return mapping

# --- EXECUTION BLOCK ---

if __name__ == "__main__":
    # Update these paths to the actual folders on your computer
    #base_path = os.getcwd() # Or use an absolute path like "C:/Data"
    base_path = "/Users/bodong/Code/project/forecast"
    shp_file = "/Users/bodong/Downloads/eth_admin_boundaries/eth_admin3.shp"
    
    my_dirs = {
        "raw": os.path.join(base_path, "aifs"),
        "processed": os.path.join(base_path, "processed")
    }

    # Run the function
    try:
        df = build_grid_to_district_mapping(my_dirs, shp_file)
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")
