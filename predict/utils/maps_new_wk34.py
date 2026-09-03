# ==============================================================================
# File: maps.py (3-Panel Probability & Single-Panel Max-Period Edition)
# ==============================================================================
# NOTE: This is an UNUSED legacy variant that combines weeks 3+4 into one panel
# and assumes the 4-week structure. It is not imported by run_maps.py (which uses
# maps_new_zone). For the general "plot all bins" behavior use maps_new.py /
# maps_new_zone.py, which size the panels to the number of bins in the data.
# ==============================================================================
from datetime import timedelta
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm, ListedColormap
import geopandas as gpd
from pathlib import Path
import logging
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_boundary(base, use_cartopy=False, region='country',
                  resolution='10m', category='cultural',
                  name='admin_0_countries'):
    if use_cartopy:
        import cartopy.io.shapereader as shpreader
        ne_path = shpreader.natural_earth(resolution=resolution, category=category, name=name)
        geom = None
        for country in shpreader.Reader(ne_path).records():
            if country.attributes['NAME'] == region:
                geom = country.geometry
                break
        if geom is None:
            raise ValueError(f"Region '{region}' not found in cartopy Natural Earth.")
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    else:
        shp_path = base / "data" / "shapefile" / "Country_Boundary.shp"
        gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
    return gdf

def _period_labels(t):
    """4-window system labels"""
    fmt = '%m/%d/%Y'
    def date(offset): return (t + timedelta(days=offset)).strftime(fmt)
    return {
        'just_week1':  f"{date(1)} - {date(7)}",
        'week2':       f"{date(8)} - {date(14)}",
        'weeks34':     f"{date(15)} - {date(28)}",
        'later':       f"{date(29)}+",
    }

def load_woreda_geometries(base, shapefile_path=None):
    if shapefile_path is None:
        shapefile_path = base / "data" / "shapefile" / "manual_zones_woredas.shp"
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    return gdf[["adm3_name", "geometry"]].copy()

def max_period(vf):
    """4-window logic (W1, W2, W3+4, Later)"""
    if vf[0] >= 0.65: return 'just_week1'
    if vf[4] >= 0.65: return 'later'
    # Aggregate Weeks 3 and 4
    w1, w2, w34, later = vf[0], vf[1], (vf[2] + vf[3]), vf[4]
    categories = ['just_week1', 'week2', 'weeks34', 'later']
    values = [w1, w2, w34, later]
    return categories[int(np.argmax(values))]

# ---------------------------------------------------------------------------
# Main map generator
# ---------------------------------------------------------------------------

def make_maps(summary, output_path, ref_onset=False, all_cells_file=None,
              woreda_shapefile=None, use_cartopy=False, region='country',
              resolution='10m', category='cultural', name='admin_0_countries', 
              zoom_to_data=False):
    
    base = Path(__file__).resolve().parent.parent
    country_gdf = load_boundary(base, use_cartopy=use_cartopy, region=region,
                                resolution=resolution, category=category, name=name)

    if all_cells_file is None:
        all_cells_file = base / "data" / "support" / "all_cells.csv"
    all_cells = pd.read_csv(all_cells_file)
    valid_ids = set(all_cells["adm3_name"].astype(str).str.strip())

    woredas = load_woreda_geometries(base, woreda_shapefile)
    woredas = woredas[woredas["adm3_name"].isin(valid_ids)]

    first_date = pd.to_datetime(summary["time"].iloc[0])
    date_str_fmt = first_date.strftime("%Y%m%d")
    output_dir = output_path / ("maps_ref" if ref_onset else "maps")
    os.makedirs(output_dir, exist_ok=True)

    preds_df = summary.copy()
    if "id" not in preds_df.columns: preds_df["id"] = preds_df["adm3_name"]

    for i in range(1, 5):
        preds_df[f"Forecast_p_{i}"] = preds_df[f"week{i}"]
        preds_df[f"Climatology_p_{i}"] = preds_df[f"clim_week{i}"]
    
    # Combined data for the 3-panel Weekly Map
    preds_df["Forecast_p_34"] = preds_df["Forecast_p_3"] + preds_df["Forecast_p_4"]
    preds_df["Climatology_p_34"] = preds_df["Climatology_p_3"] + preds_df["Climatology_p_4"]
    preds_df["Forecast_p_later"] = preds_df.get("later", 1 - preds_df[[f"Forecast_p_{i}" for i in range(1, 5)]].sum(axis=1))

    # Extent logic
    minx, miny, maxx, maxy = country_gdf.total_bounds
    if zoom_to_data:
        wb = woredas.total_bounds
        minx, maxx, miny, maxy = wb[0], wb[2], wb[1], wb[3]

    # Color settings
    period_order = ['just_week1', 'week2', 'weeks34', 'later']
    plasma_cmap = plt.get_cmap('plasma')
    stops = np.linspace(0.2, 1.0, len(period_order))
    period_colors = {k: plasma_cmap(s) for k, s in zip(period_order, stops)}
    period_colors['none'] = '#d3d3d3'

    #prob_bins = [0, 0.1, 0.2, 0.3, 0.4, 1.0]
    prob_bins = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    #prob_cmap = ListedColormap(plt.get_cmap('plasma_r')(np.linspace(0, 1, len(prob_bins) - 1)))
    #prob_cmap = ListedColormap(plt.get_cmap('PuBu')(np.linspace(0, 1, len(prob_bins) - 1)))
    prob_cmap = ListedColormap(plt.get_cmap('Blues')(np.linspace(0, 1, len(prob_bins) - 1)))
    prob_norm = BoundaryNorm(prob_bins, ncolors=len(prob_bins) - 1, clip=True)

    # 1. 3-PANEL WEEKLY PROBABILITY MAPS
    for t, grp in preds_df.groupby('time'):
        merged = woredas.merge(grp, left_on="adm3_name", right_on="id", how="left")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True, gridspec_kw={'wspace': 0.1})
        
        display_cols = [("Forecast_p_1", "Climatology_p_1", "Week 1"), 
                        ("Forecast_p_2", "Climatology_p_2", "Week 2"), 
                        ("Forecast_p_34", "Climatology_p_34", "Week 3 & 4")]

        for i, (f_col, c_col, title) in enumerate(display_cols):
            ax = axes[i]
            country_gdf.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')
            merged_ok = merged[merged[f_col].notna()]
            
            if not merged_ok.empty:
                # Contrast Borders
                face_colors = prob_cmap(prob_norm(merged_ok[f_col]))
                lum = (0.299 * face_colors[:, 0] + 0.587 * face_colors[:, 1] + 0.114 * face_colors[:, 2])
                edge_colors = ['black' if l > 0.5 else 'white' for l in lum]

                merged_ok.plot(ax=ax, column=f_col, cmap=prob_cmap, norm=prob_norm, 
                               edgecolor=edge_colors, linewidth=0.2, zorder=2)
                
                # Climatology Borders
                hi = merged_ok[merged_ok[f_col] >= merged_ok[c_col] + 0.10]
                lo = merged_ok[merged_ok[f_col] <= merged_ok[c_col] - 0.10]
                if not hi.empty: hi.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=0.2, zorder=3)
                if not lo.empty: lo.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=0.05, zorder=3)

            # Styling
            ax.set_title(title, fontsize=10, pad=6)
            ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
            ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
            ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)

        sm = plt.cm.ScalarMappable(norm=prob_norm, cmap=prob_cmap)
        fig.colorbar(sm, ax=list(axes), orientation='horizontal', fraction=0.04, pad=0.12, label='Probability')
        legend_handles = [Patch(facecolor='none', edgecolor=c, linewidth=1.5, label=l)
                         for c, l in [('red', '≥10% lower than climatology'), ('green', '≥10% higher than climatology')]]
        axes[-1].legend(handles=legend_handles, loc='lower right', fontsize=7, framealpha=0.8)

        plt.savefig(output_dir / f"prob_weeks_3panel_{pd.Timestamp(t).strftime('%Y%m%d')}_wk34.png", dpi=150)
        plt.close(fig)

    # 2. SINGLE-PANEL MAX-PERIOD MAP (Restored)
    for t, grp in preds_df.groupby('time'):
        merged = woredas.merge(grp, left_on="adm3_name", right_on="id", how="left")
        merged["period"] = merged.apply(lambda r: max_period([r.get(f"Forecast_p_{i}", 0) for i in range(1, 5)] + [r.get("Forecast_p_later", 0)]), axis=1)
        
        labels = _period_labels(pd.Timestamp(t))
        handles = [Patch(facecolor=period_colors[k], edgecolor='none', label=labels[k]) for k in period_order]
        handles.append(Patch(facecolor=period_colors['none'], edgecolor='none', label='No forecast / onset declared'))

        fig, ax = plt.subplots(figsize=(6, 6))
        country_gdf.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')
        
        for period_key, color in period_colors.items():
            subset = merged[merged["period"] == period_key]
            if not subset.empty:
                rgb = plt.cm.colors.to_rgb(color)
                l = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
                subset.plot(ax=ax, color=color, edgecolor='black' if l > 0.5 else 'white', linewidth=0.2, zorder=1)

        # Styling
        ax.legend(handles=handles, title='Period with Max Probability of Onset', loc='lower left', ncol=2, fontsize=7, title_fontsize=7)
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / f"map_max_period_{pd.Timestamp(t).strftime('%Y%m%d')}_wk34.png", dpi=150)
        plt.close(fig)
