# ==============================================================================
# File: geometry_utils.py
# ==============================================================================
# Purpose
#   Spec-driven geometry / regridding helpers so the pipeline can be re-targeted
#   to a new geography by supplying only a boundary shapefile + gridded NetCDF.
#
#   Consolidates the two hand-written weight builders (utils/remap_weights.py and
#   utils/remap_weights_ngcm.py) into one configurable function that:
#     - auto-detects grid resolution from the NetCDF coordinates (no hardcoded
#       half-cell size),
#     - reads the admin-unit key column, optional parent column, and CRS from a
#       `geometry` config block (defaults reproduce the current Ethiopia setup),
#     - renames the admin key to the canonical internal name `adm3_name`.
#
# `geometry` config block (all keys optional except `shapefile`):
#   geometry:
#     shapefile:       path/to/admin.shp
#     region_key_col:  adm3_name      # renamed to canonical `adm3_name` on read
#     parent_key_col:  adm2_name      # optional (zone overlays / labels)
#     crs:             EPSG:4326
#     region_label:    Ethiopia       # used by the map/plot utilities
#     grid_lat_var:    latitude       # NetCDF latitude coord (lat/latitude auto)
#     grid_lon_var:    longitude
#
# Function index
#   get_geometry_cfg(spec)
#   get_half_delta(arr)
#   standardize_grid_coords(ds, lat_var, lon_var)
#   load_admin_geometry(cfg)
#   build_grid_to_admin_weights(sample_nc, cfg, out_csv=None)
# ==============================================================================

import os
import glob
import numpy as np
import pandas as pd

# Canonical internal names - the rest of the pipeline keys on these.
CANON_REGION_KEY = "adm3_name"
CANON_PARENT_KEY = "adm2_name"
DEFAULT_CRS = "EPSG:4326"


def get_geometry_cfg(spec):
    """
    Extract the `geometry` config block from a spec dict, applying defaults.

    Returns a dict with keys: shapefile, region_key_col, parent_key_col, crs,
    region_label, grid_lat_var, grid_lon_var. `shapefile` may be None if the
    spec has no geometry block (callers that need it should check).
    """
    g = dict((spec or {}).get("geometry") or {})
    return {
        "shapefile":      g.get("shapefile"),
        "region_key_col": g.get("region_key_col", CANON_REGION_KEY),
        "parent_key_col": g.get("parent_key_col"),  # optional
        "crs":            g.get("crs", DEFAULT_CRS),
        "region_label":   g.get("region_label"),
        "grid_lat_var":   g.get("grid_lat_var"),   # None -> auto-detect
        "grid_lon_var":   g.get("grid_lon_var"),
    }


def get_half_delta(arr):
    """
    Half the spacing between the first two coordinate values (grid half-cell).
    Falls back to 0.125 for single-point axes. Ported from remap_weights_ngcm.py
    so resolution is inferred from the data instead of hardcoded.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size > 1:
        return float(np.abs(arr[1] - arr[0]) / 2.0)
    return 0.125


def standardize_grid_coords(ds, lat_var=None, lon_var=None):
    """
    Return (lats, lons) arrays from an xarray Dataset, tolerant of common
    coordinate names. Explicit lat_var/lon_var win; otherwise tries
    latitude/lat and longitude/lon.
    """
    def _pick(explicit, candidates):
        if explicit is not None:
            if explicit not in ds.coords and explicit not in ds.variables:
                raise ValueError(f"Grid coord '{explicit}' not found in dataset.")
            return explicit
        for c in candidates:
            if c in ds.coords or c in ds.variables:
                return c
        raise ValueError(
            f"Could not find a coordinate among {candidates}. "
            f"Available: {list(ds.coords)}"
        )

    lat_name = _pick(lat_var, ["latitude", "lat", "y", "Y"])
    lon_name = _pick(lon_var, ["longitude", "lon", "x", "X"])
    return np.asarray(ds[lat_name].values, dtype=float), np.asarray(ds[lon_name].values, dtype=float)


def load_admin_geometry(cfg):
    """
    Read the admin-boundary shapefile named in cfg and return a GeoDataFrame
    with the canonical `adm3_name` column (and `adm2_name` if parent_key_col is
    given), reprojected to cfg["crs"].

    cfg : dict from get_geometry_cfg (or a raw geometry block).
    """
    import geopandas as gpd

    cfg = {**get_geometry_cfg({"geometry": cfg})} if "region_key_col" not in cfg else cfg
    shp = cfg.get("shapefile")
    if not shp:
        raise ValueError("geometry.shapefile is required to load admin boundaries.")
    if not os.path.exists(shp):
        raise FileNotFoundError(f"Shapefile not found: {shp}")

    gdf = gpd.read_file(shp)

    region_col = cfg.get("region_key_col", CANON_REGION_KEY)
    if region_col not in gdf.columns:
        raise ValueError(
            f"Shapefile missing region key column '{region_col}'. "
            f"Set geometry.region_key_col. Available: {gdf.columns.tolist()}"
        )

    keep = {region_col: CANON_REGION_KEY}
    parent_col = cfg.get("parent_key_col")
    if parent_col:
        if parent_col not in gdf.columns:
            raise ValueError(
                f"Shapefile missing parent key column '{parent_col}'. "
                f"Available: {gdf.columns.tolist()}"
            )
        keep[parent_col] = CANON_PARENT_KEY

    cols = list(keep.keys()) + ["geometry"]
    gdf = gdf[cols].rename(columns=keep).reset_index(drop=True)
    gdf[CANON_REGION_KEY] = gdf[CANON_REGION_KEY].astype(str).str.strip()

    crs = cfg.get("crs", DEFAULT_CRS)
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    elif crs is not None and str(gdf.crs) != str(crs):
        gdf = gdf.to_crs(crs)
    return gdf


def _latlon_dim_names(ds, lat_var=None, lon_var=None):
    """Return the (lat, lon) coordinate names present in the dataset."""
    def _pick(explicit, cands):
        if explicit is not None:
            return explicit
        for c in cands:
            if c in ds.coords or c in ds.variables or c in ds.dims:
                return c
        raise ValueError(f"Could not find a coordinate among {cands}. Available: {list(ds.coords)}")
    return (_pick(lat_var, ["latitude", "lat", "y", "Y"]),
            _pick(lon_var, ["longitude", "lon", "x", "X"]))


def grid_valid_cells(nc_paths, value_col, lat_var=None, lon_var=None, round_dp=5):
    """
    Return the set of (round(lat), round(lon)) grid cells that have at least one
    finite value of `value_col` across the given NetCDF files.

    Used to restrict regridding to cells with real ground-truth coverage - e.g.
    a rain-gauge grid has no data over the ocean, so those cells must not
    contribute to (or dilute) a district's regridded value.
    """
    import xarray as xr
    valid = set()
    for p in nc_paths:
        with xr.open_dataset(p) as ds:
            if value_col not in ds.variables:
                # tolerate case differences
                match = [v for v in ds.variables if v.lower() == value_col.lower()]
                if not match:
                    continue
                vcol = match[0]
            else:
                vcol = value_col
            latn, lonn = _latlon_dim_names(ds, lat_var, lon_var)
            da = ds[vcol]
            other = [d for d in da.dims if d not in (latn, lonn)]
            finite_any = np.isfinite(da).any(dim=other) if other else np.isfinite(da)
            finite_any = finite_any.transpose(latn, lonn)
            lats = np.asarray(ds[latn].values, dtype=float)
            lons = np.asarray(ds[lonn].values, dtype=float)
            arr = np.asarray(finite_any.values)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if arr[i, j]:
                        valid.add((round(float(lats[i]), round_dp), round(float(lons[j]), round_dp)))
    return valid


def restrict_weights_to_valid(weights_df, valid_cells, renormalize=True, round_dp=5):
    """
    Drop grid->admin weight rows whose cell is not in `valid_cells`, then (by
    default) renormalize the surviving weights within each admin unit so they
    sum to 1. This makes each unit's regridded value a weighted average over
    ONLY the cells that have ground-truth data - and the returned table is meant
    to be applied to BOTH the ground truth and the forecasts, so both share the
    same spatial footprint.

    Returns (restricted_weights_df, dropped_units) where dropped_units is the
    list of admin units that had overlapping grid cells but none with data.
    """
    w = weights_df.copy()
    keys = list(zip(w["latitude"].round(round_dp), w["longitude"].round(round_dp)))
    w = w[[k in valid_cells for k in keys]].copy()

    all_units = set(weights_df[CANON_REGION_KEY].unique())
    kept_units = set(w[CANON_REGION_KEY].unique())
    dropped_units = sorted(all_units - kept_units)

    if renormalize and not w.empty:
        denom = w.groupby(CANON_REGION_KEY)["weight"].transform("sum")
        w["weight"] = w["weight"] / denom
    return w.reset_index(drop=True), dropped_units


def _resolve_sample_nc(sample_nc):
    """Resolve a sample NetCDF path (accept a dir; skip *_adm3.nc)."""
    if os.path.isdir(sample_nc):
        cand = [f for f in sorted(glob.glob(os.path.join(sample_nc, "*.nc"))) if not f.endswith("_adm3.nc")]
        if not cand:
            raise FileNotFoundError(f"No .nc files found in {sample_nc}")
        return cand[0]
    if not os.path.exists(sample_nc):
        raise FileNotFoundError(f"Sample NetCDF not found: {sample_nc}")
    return sample_nc


def grid_coords_of(sample_nc, cfg):
    """Return (lats, lons) for a dataset's grid, honoring cfg grid_lat/lon_var."""
    import xarray as xr
    with xr.open_dataset(_resolve_sample_nc(sample_nc)) as ds:
        return standardize_grid_coords(ds, cfg.get("grid_lat_var"), cfg.get("grid_lon_var"))


def grids_equal(nc_a, nc_b, cfg, tol=1e-6):
    """True if two datasets share the same lat/lon grid (order-independent)."""
    la1, lo1 = grid_coords_of(nc_a, cfg)
    la2, lo2 = grid_coords_of(nc_b, cfg)
    return (la1.shape == la2.shape and lo1.shape == lo2.shape
            and np.allclose(np.sort(la1), np.sort(la2), atol=tol)
            and np.allclose(np.sort(lo1), np.sort(lo2), atol=tol))


def _grid_cell_gdf(sample_nc, cfg):
    """GeoDataFrame of grid-cell square polygons for a dataset's grid."""
    import geopandas as gpd
    from shapely.geometry import box
    lats, lons = grid_coords_of(sample_nc, cfg)
    hlat, hlon = get_half_delta(lats), get_half_delta(lons)
    recs = []
    for la in lats:
        for lo in lons:
            laf, lof = float(la), float(lo)
            recs.append({"latitude": laf, "longitude": lof,
                         "geometry": box(lof - hlon, laf - hlat, lof + hlon, laf + hlat)})
    return gpd.GeoDataFrame(recs, crs=cfg.get("crs", DEFAULT_CRS)), hlat, hlon


def build_coverage_geom(valid_cells, half_lat, half_lon):
    """
    Union of the ground-truth-valid cell boxes -> a single geometry describing
    the area where ground-truth data actually exists. `valid_cells` is the set
    of (lat, lon) centers from grid_valid_cells; half_lat/half_lon size the boxes.
    """
    from shapely.geometry import box
    from shapely.ops import unary_union
    boxes = [box(lo - half_lon, la - half_lat, lo + half_lon, la + half_lat)
             for (la, lo) in valid_cells]
    return unary_union(boxes) if boxes else None


def coverage_missing_fraction(admin_gdf, coverage_geom):
    """
    Per admin unit, the fraction of its area NOT covered by ground-truth data.
    Returns DataFrame [adm3_name, area, missing_frac] (missing_frac in [0, 1]).
    """
    rows = []
    for _, r in admin_gdf.iterrows():
        g = r.geometry
        area = float(g.area) if g is not None and not g.is_empty else 0.0
        if area <= 0 or coverage_geom is None:
            mf = np.nan if area <= 0 else 1.0
        else:
            covered = float(g.intersection(coverage_geom).area)
            mf = max(0.0, min(1.0, 1.0 - covered / area))
        rows.append({CANON_REGION_KEY: r[CANON_REGION_KEY], "area": area, "missing_frac": mf})
    return pd.DataFrame(rows)


def unit_centroids(units_gdf):
    """
    Per-unit centroid as a DataFrame [adm3_name, lat, lon] (in the units' CRS,
    normally EPSG:4326). Produced during regridding from the shapefile so it can
    be reused as a `filter.centroids_file` for the bbox domain filter on admin
    units (which otherwise carry no lat/lon).
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # geographic-CRS centroid warning is fine here
        cen = units_gdf.geometry.centroid
    return pd.DataFrame({
        CANON_REGION_KEY: units_gdf[CANON_REGION_KEY].values,
        "lat": np.asarray(cen.y.values, dtype=float),
        "lon": np.asarray(cen.x.values, dtype=float),
    })


def build_grid_cell_units(sample_nc, cfg, valid_cells=None, round_dp=5):
    """
    Build "admin units" that ARE the ground-truth grid cells (id = "{lat}_{lon}").
    Used when no shapefile is provided: forecasts are then regridded to match the
    ground-truth grid rather than political boundaries. If `valid_cells` is given,
    only those (data-bearing) cells become units.
    """
    grid_gdf, _, _ = _grid_cell_gdf(sample_nc, cfg)
    g = grid_gdf.copy()
    g[CANON_REGION_KEY] = [f"{round(la, round_dp)}_{round(lo, round_dp)}"
                           for la, lo in zip(g["latitude"], g["longitude"])]
    if valid_cells is not None:
        keep = [(round(la, round_dp), round(lo, round_dp)) in valid_cells
                for la, lo in zip(g["latitude"], g["longitude"])]
        g = g[keep]
    return g[[CANON_REGION_KEY, "geometry"]].reset_index(drop=True)


def build_weights_to_coverage(sample_nc, cfg, coverage_geom, out_csv=None, admin_gdf=None):
    """
    Area weights to regrid a dataset (on ITS OWN grid) onto each admin unit
    CLIPPED to the ground-truth coverage geometry, normalized per unit to sum 1.

    Use this for forecasts whose grid differs from the ground-truth grid: the
    forecast is integrated only over "the political unit minus the parts where
    ground-truth data did not exist", so it matches the ground-truth footprint.
    """
    import geopandas as gpd
    if coverage_geom is None:
        raise ValueError("coverage_geom is None (no ground-truth cells).")
    admin = (admin_gdf if admin_gdf is not None else load_admin_geometry(cfg)).copy()
    admin["geometry"] = admin.geometry.intersection(coverage_geom)
    admin = admin[admin.geometry.notna() & (~admin.geometry.is_empty)]
    if admin.empty:
        raise ValueError("No admin unit intersects the ground-truth coverage area.")

    grid_gdf, _, _ = _grid_cell_gdf(sample_nc, cfg)
    ov = gpd.overlay(grid_gdf, admin, how="intersection")
    ov["w"] = ov.geometry.area
    ov = ov[ov["w"] > 0]
    denom = ov.groupby(CANON_REGION_KEY)["w"].transform("sum")
    ov["weight"] = ov["w"] / denom
    mapping = ov[["latitude", "longitude", CANON_REGION_KEY, "weight"]].reset_index(drop=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        mapping.to_csv(out_csv, index=False)
    return mapping


def build_grid_to_admin_weights(sample_nc, cfg, out_csv=None, admin_gdf=None):
    """
    Build area-fraction weights mapping gridded cells -> admin units.

    Overlays each grid cell (a square polygon sized by the auto-detected
    resolution) with the admin polygons and computes
    weight = intersection_area / cell_area (in the config CRS). Reproduces the
    behavior of remap_weights*.py but fully parameterized.

    Parameters
    ----------
    sample_nc : str
        Path to a representative gridded NetCDF (or a directory; the first *.nc
        that is not an *_adm3.nc is used).
    cfg : dict
        A `geometry` block (see get_geometry_cfg).
    out_csv : str or None
        If given, write the mapping CSV (columns: latitude, longitude,
        adm3_name, weight).

    Returns
    -------
    DataFrame with columns: latitude, longitude, adm3_name, weight.
    """
    import geopandas as gpd
    import xarray as xr
    from shapely.geometry import box

    cfg = get_geometry_cfg({"geometry": cfg}) if "region_key_col" not in cfg else cfg

    # Resolve the sample NetCDF file
    if os.path.isdir(sample_nc):
        cand = [f for f in sorted(glob.glob(os.path.join(sample_nc, "*.nc")))
                if not f.endswith("_adm3.nc")]
        if not cand:
            raise FileNotFoundError(f"No .nc files found in {sample_nc}")
        sample_nc = cand[0]
    if not os.path.exists(sample_nc):
        raise FileNotFoundError(f"Sample NetCDF not found: {sample_nc}")

    admin = admin_gdf if admin_gdf is not None else load_admin_geometry(cfg)

    with xr.open_dataset(sample_nc) as ds:
        lats, lons = standardize_grid_coords(ds, cfg.get("grid_lat_var"), cfg.get("grid_lon_var"))

    half_lat = get_half_delta(lats)
    half_lon = get_half_delta(lons)
    print(f"Building grid polygons (resolution {half_lat*2:.4f} lat x {half_lon*2:.4f} lon)...")

    cell_records = []
    for lat in lats:
        for lon in lons:
            latf, lonf = float(lat), float(lon)
            poly = box(lonf - half_lon, latf - half_lat, lonf + half_lon, latf + half_lat)
            cell_records.append({
                "latitude": latf, "longitude": lonf,
                "geometry": poly, "cell_area": poly.area,
            })
    grid_gdf = gpd.GeoDataFrame(cell_records, crs=cfg.get("crs", DEFAULT_CRS))

    print("Computing intersections...")
    overlaid = gpd.overlay(grid_gdf, admin, how="intersection")
    overlaid["intersection_area"] = overlaid.geometry.area
    overlaid["weight"] = overlaid["intersection_area"] / overlaid["cell_area"]
    overlaid = overlaid[overlaid["weight"] > 1e-5]

    mapping = overlaid[["latitude", "longitude", CANON_REGION_KEY, "weight"]].reset_index(drop=True)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        mapping.to_csv(out_csv, index=False)
        print(f"Wrote {len(mapping)} grid->admin weights to {out_csv}")
    return mapping
