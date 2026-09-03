"""
plot_clim_onset_map.py
──────────────────────
Plot a spatial map of the climatological median monsoon onset day-of-year
for each woreda, using pre-computed median onset indices from a pickle file
and the manual_zones_woredas shapefile.

Map settings follow maps_new.py exactly:
  - Country boundary via Natural Earth (cartopy-style)
  - Extent from country_gdf.total_bounds (minx, miny, maxx, maxy)
  - figsize=(6, 6), colorbar pad=0.12
  - Piecewise seasonal ListedColormap (May–Sep) with BoundaryNorm
  - Luminance-based edge colours (black/white)
  - Grey #d3d3d3 for missing values
  - Dashed grid, Longitude/Latitude axis labels

Usage
-----
  python plot_clim_onset_map.py \
      --pkl Monsoon_Data/Processed_Data/Models/onset_idx_median_by_id.pkl \
      --shp manual_zones_woredas.shp \
      --out clim_median_onset.png
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib import colors as mcolors

import sys
import os

# Identify the repository root (one level up from this script's directory)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

def doy_to_mmm_dd(doy, date_filter_year=2024):
    """Convert day of year to 'MMM DD' format"""
    # Use 2018 (not a leap year) as reference to handle all possible DOYs
    date = pd.to_datetime(f"{date_filter_year}-{int(doy):03d}", format="%Y-%j")
    return date.strftime("%b %d")

def cbar_season():
    """ Create the custom monthly colormap (May to September) """
    from matplotlib.colors import ListedColormap, BoundaryNorm

    month_cmaps = {
    "May":    plt.cm.YlOrBr,
    "Jun":    plt.cm.Greens,
    "Jul":    plt.cm.YlGnBu,
    "Aug":    plt.cm.Purples,
    "Sep":    plt.cm.RdPu,
    }

    # Day-of-year ranges (only until September)
    month_doys = {
        "May": (121, 151),  # May 15 - May 31
        "Jun": (152, 181),  # Jun 1 - Jun 30
        "Jul": (182, 212),  # Jul 1 - Jul 31
        "Aug": (213, 243),  # Aug 1 - Aug 31
        "Sep": (244, 273),  # Sep 1 - Sep 30
    }
    colors = []
    bounds = []

    N_per_month = 8  # smoothness inside each month

    for month, cmap in month_cmaps.items():
        d0, d1 = month_doys[month]

        # Sequential slice of each colormap
        ramp = cmap(np.linspace(0.35, 0.95, N_per_month))

        colors.extend(ramp)
        bounds.extend(np.linspace(d0, d1, N_per_month, endpoint=False))

    # Add final bound
    bounds.append(list(month_doys.values())[-1][1])

    cmap_jjas = ListedColormap(colors, name="JJAS_piecewise")
    norm_jjas = BoundaryNorm(bounds, cmap_jjas.N)

    return cmap_jjas, norm_jjas, bounds

def cbar_diff():
    """Colormap for difference map (days): RdBu_r centred at 0."""
    bounds = [-45, -35, -25, -15, -5,0, 5, 15, 25, 35, 45]
    #bounds = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
    cmap = ListedColormap(plt.get_cmap('RdBu_r')(np.linspace(0, 1, len(bounds) - 1)))
    norm = BoundaryNorm(bounds, ncolors=len(bounds) - 1, clip=True)
    return cmap, norm

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--pkl',  default='Monsoon_Data/Processed_Data/Models/mr_onset_idx_median_by_id.pkl')
parser.add_argument('--shp', default='/Users/bodong/Code/project/onset_blending-adm3/predict/data/shapefile/manual_zones_woredas.shp')
parser.add_argument('--boundary', default=None,
                    help='Path to Country_Boundary.shp. If not provided, '
                         'falls back to Natural Earth 110m built-in.')
parser.add_argument('--out', default='/Users/bodong/Code/project/onset_blending-adm3/Monsoon_Data/results/clim_median_onset.png')
parser.add_argument('--pkl2', default=None,
                    help='Second pkl for difference map (pkl - pkl2).')
parser.add_argument('--diff_map', action='store_true',
                    help='Plot difference map (pkl minus pkl2) with RdBu colormap.')

args = parser.parse_args()

# ── Country boundary — mirrors maps_new.py load_boundary() ───────────────────
# Prefers Country_Boundary.shp if provided; falls back to Natural Earth built-in
if args.boundary and Path(args.boundary).exists():
    country_gdf = gpd.read_file(args.boundary).to_crs('EPSG:4326')
else:
    try:
        import cartopy.io.shapereader as shpreader
        ne_path = shpreader.natural_earth(
            resolution='10m', category='cultural', name='admin_0_countries'
        )
        geom = None
        for country in shpreader.Reader(ne_path).records():
            if country.attributes['NAME'] == 'Ethiopia':
                geom = country.geometry
                break
        country_gdf = gpd.GeoDataFrame(geometry=[geom], crs='EPSG:4326')
    except Exception:
        # Final fallback: pyogrio naturalearth_lowres (110m, always available)
        import pyogrio
        ne_lowres = Path(pyogrio.__file__).parent / 'tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp'
        world = gpd.read_file(ne_lowres)
        country_gdf = world[world['name'] == 'Ethiopia'].to_crs('EPSG:4326')

# ── Map extent — exactly as maps_new.py ──────────────────────────────────────
minx, miny, maxx, maxy = country_gdf.total_bounds

# ── Woreda geometries — maps_new.py load_woreda_geometries() ─────────────────
woredas = gpd.read_file(args.shp).to_crs('EPSG:4326')[['adm3_name', 'geometry']].copy()

# ── NetCDF ────────────────────────────────────────────────────────────────────
#ds = xr.open_dataset(args.nc)
##tp_slice = ds['tp'].isel(day=args.day, time=args.time_idx)
#tp_slice = ds[args.var].isel(time=args.time_idx)
#
#if "number" in ds.dims:
#   tp_slice = tp_slice.isel(number=args.number)
#
#time_val  = pd.Timestamp(ds['time'].values[args.time_idx])
#
#if "day" in ds.dims:
#    tp_slice = tp_slice.isel(day=args.day)
#    day_val   = int(ds['day'].values[args.day])
#    plot_date = (time_val + pd.Timedelta(days=day_val)).strftime('%Y-%m-%d')
#else:
#    plot_date = time_val.strftime('%Y-%m-%d')
#
#
#df = pd.DataFrame({'adm3_name': ds['adm3_name'].values, 'tp': tp_slice.values})

df = pd.read_pickle(args.pkl)
df = df.rename(columns={"median_onset_idx" : "tp"})
df["tp"] = df["tp"] + 120

if args.diff_map:
    if args.pkl2 is None:
        raise ValueError("--diff_map requires --pkl2 to be specified.")
    df2 = pd.read_pickle(args.pkl2)
    df2 = df2.rename(columns={"median_onset_idx": "tp"})
    df2["tp"] = df2["tp"] + 120
    df = df.merge(df2[["adm3_name", "tp"]], on="adm3_name", suffixes=("", "_2"))
    df["tp"] = df["tp_2"] - df["tp"]

merged = woredas.merge(df, on='adm3_name', how='left')

# ── Colour scheme — maps_new.py plasma_r / BoundaryNorm pattern ──────────────
if args.diff_map:
    tp_cmap, tp_norm = cbar_diff()
    cbar_label = 'difference (days)'
    plot_title = 'Onset difference'
else:
    #tp_bins = [135, 150, 165, 180, 195, 210]
    tp_bins = list(range(135, 231, 5))
    #tp_bins = np.arange(30, 150, 3)
    #tp_bins = np.arange(135, 245, 3)
    #tp_cmap = ListedColormap(plt.get_cmap('plasma_r')(np.linspace(0, 1, len(tp_bins) - 1)))
    #tp_cmap = ListedColormap(plt.get_cmap('Blues_r')(np.linspace(0, 1, len(tp_bins) - 1)))
    tp_cmap = ListedColormap(plt.get_cmap('plasma')(np.linspace(0, 1, len(tp_bins) - 1)))
    tp_norm = BoundaryNorm(tp_bins, ncolors=len(tp_bins) - 1, clip=True)

    cbar_label = 'day of year'
    plot_title = 'Climatology median onset'

cbar_ssn = True

if cbar_ssn:
    cmap_jjas, norm_jjas, bounds = cbar_season()
else:
    # Use a colormap (RdYlBu_r or similar) also 'RdYlGn_r', 'Spectral_r', or 'coolwarm'
    cmap_jjas = plt.cm.Spectral
    norm_jjas = mcolors.BoundaryNorm(tp_bins, cmap_jjas.N, extend='max')
    tp_norm = norm_jjas

#tp_cmap = cmap_jjas

# ── Figure — figsize=(6,6) as maps_new.py single-panel ───────────────────────
fig, ax = plt.subplots(figsize=(6, 6))

# maps_new.py: country_gdf.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')
country_gdf.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')

merged_na = merged[merged['tp'].isna()]
merged_ok = merged[merged['tp'].notna()]

# Grey for missing — maps_new.py period_colors['none'] = '#d3d3d3'
if not merged_na.empty:
    merged_na.plot(ax=ax, color='#d3d3d3', edgecolor='none', zorder=1)

# Filled polygons with luminance-based edge colour — maps_new.py lines 139-145
if not merged_ok.empty:
    face_colors = tp_cmap(tp_norm(merged_ok['tp']))
    lum = (0.299 * face_colors[:, 0] +
           0.587 * face_colors[:, 1] +
           0.114 * face_colors[:, 2])
    edge_colors = ['black' if l > 0.5 else 'white' for l in lum]
    merged_ok.plot(ax=ax, column='tp', cmap=tp_cmap, norm=tp_norm,
                   edgecolor=edge_colors, linewidth=0.2, zorder=2)

# maps_new.py styling — minx/maxx/miny/maxy, pad=0.12
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)
#ax.set_title(f'Total Precipitation — {plot_date} (day index {args.day})',
#ax.set_title(f'Climatology median onset', fontsize=10, pad=6)
ax.set_title(plot_title, fontsize=10, pad=6)

sm = plt.cm.ScalarMappable(norm=tp_norm, cmap=tp_cmap)
sm.set_array([])

#fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04, pad=0.12, label='day of year')
fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04, pad=0.12, label=cbar_label)

# Add colorbar with MMM DD labels for every other tick
#cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.6, aspect=20)
#
#if cbar_ssn:
#    # Create tick positions - use every other bound for labeling
#    tick_positions = bounds[::2]  # Every other boundary
#    tick_labels = [doy_to_mmm_dd(doy) for doy in tick_positions[:-1]]  # Exclude last boundary
#
#    # Set all boundaries as minor ticks (for visual separation)
#    cbar.set_ticks(bounds, minor=True)
#    # Set every other boundary as major ticks (with labels)
#    cbar.set_ticks(tick_positions[:-1])  # Exclude last boundary
#else:
#    # Create custom tick labels in MMM DD format
#    tick_levels = levels[::4]  # Use every other level to avoid crowding
#    tick_labels = [doy_to_mmm_dd(doy) for doy in tick_levels]
#    cbar.set_ticks(tick_levels)
#
#cbar.set_ticklabels(tick_labels)

#cbar.set_label('Mean onset date', fontsize=12, fontweight='normal')
#cbar.ax.tick_params(labelsize=10)


plt.tight_layout()
plt.savefig(args.out, dpi=150)
plt.close()
print(f"Map saved: {args.out}")
