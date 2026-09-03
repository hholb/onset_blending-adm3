# ==============================================================================
# File: maps.py  (adm3_name edition)
# ==============================================================================
# Purpose
#   Generate forecast maps using adm3-level polygon geometries instead of
#   lat/lon grid rectangles.  Each dissemination cell is a named woreda
#   (adm3_name) and is rendered as a filled polygon joined from a shapefile.
#
# Key changes from the lat/lon version
#   - all_cells.csv has a single `adm3_name` column (no lat/lon).
#   - exclude_cells.csv has `adm3_name` + `flag` columns.
#   - summary DataFrame must have `id` (== adm3_name) instead of lat/lon.
#   - Spatial rendering uses geopandas polygon joins, not Rectangle patches.
#   - NetCDF export now uses adm3_name as a 1-D string dimension.
#   - _infer_resolution() is removed (not applicable to polygon data).
# ==============================================================================

from datetime import timedelta
import os
import re
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
# Boundary loader (unchanged)
# ---------------------------------------------------------------------------

def load_boundary(base, use_cartopy=False, region='country',
                  resolution='10m', category='cultural',
                  name='admin_0_countries'):
    if use_cartopy:
        import cartopy.io.shapereader as shpreader
        ne_path = shpreader.natural_earth(
            resolution=resolution, category=category, name=name
        )
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


# ---------------------------------------------------------------------------
# Period label helper (unchanged)
# ---------------------------------------------------------------------------

def _period_labels(t):
    fmt = '%m/%d/%Y'
    def date(offset): return (t + timedelta(days=offset)).strftime(fmt)
    #return {
    #    'just_week1':  f"{date(1)} - {date(7)}",
    #    'weeks12':     f"{date(1)} - {date(14)}",
    #    'weeks23':     f"{date(8)} - {date(21)}",
    #    'weeks34':     f"{date(15)} - {date(28)}",
    #    'weeks4later': f"{date(22)}+",
    #    'later':       f"{date(29)}+",
    #}

    return {
        'week1': f"{date(1)} - {date(7)}",
        'week2': f"{date(8)} - {date(14)}",
        'week3': f"{date(15)} - {date(21)}",
        'week4': f"{date(22)} - {date(28)}",
        'later': f"{date(29)}+",
    }

# ---------------------------------------------------------------------------
# Woreda geometry loader
# ---------------------------------------------------------------------------

def load_woreda_geometries(base, shapefile_path=None):
    """
    Load the adm3-level woreda shapefile.

    Returns a GeoDataFrame with at least columns: adm3_name, geometry.
    Falls back to data/shapefile/manual_zones_woredas.shp if no path given.
    """
    if shapefile_path is None:
        shapefile_path = base / "data" / "shapefile" / "manual_zones_woredas.shp"
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    if "adm3_name" not in gdf.columns:
        raise ValueError(
            f"Woreda shapefile must contain an 'adm3_name' column. "
            f"Found: {gdf.columns.tolist()}"
        )
    return gdf[["adm3_name", "geometry"]].copy()


# ---------------------------------------------------------------------------
# Main map generator
# ---------------------------------------------------------------------------

def make_maps(summary, output_path, ref_onset=False, all_cells_file=None,
              woreda_shapefile=None,
              use_cartopy=False, region='country',
              resolution='10m', category='cultural',
              name='admin_0_countries', zoom_to_data=False):
    """
    Generate forecast maps from a blended summary DataFrame (adm3_name-based).

    Parameters
    ----------
    summary : pd.DataFrame
        Blended forecast output with columns:
          id (adm3_name), time, week1-week4, clim_week1-clim_week4,
          later (optional).
    output_path : Path
        Directory where output PNGs will be saved.
    ref_onset : bool
        If True, save into maps_mok/ subfolder with _mok suffix.
    all_cells_file : Path or None
        Path to all_cells.csv (adm3_name column).  Defaults to
        data/support/all_cells.csv.
    woreda_shapefile : Path or None
        Path to the adm3-level shapefile.  Defaults to
        data/shapefile/manual_zones_woredas.shp.
    """
    base = Path(__file__).resolve().parent.parent

    country_gdf = load_boundary(
        base, use_cartopy=use_cartopy, region=region,
        resolution=resolution, category=category, name=name
    )

    if all_cells_file is None:
        all_cells_file = base / "data" / "support" / "all_cells.csv"
    all_cells = pd.read_csv(all_cells_file)
    if "adm3_name" not in all_cells.columns:
        raise ValueError("all_cells.csv must have an 'adm3_name' column.")
    valid_ids = set(all_cells["adm3_name"].astype(str).str.strip())

    exclude_cells_file = base / "data" / "support" / "exclude_cells.csv"
    df_exclude = pd.read_csv(exclude_cells_file, dtype=str)
    exclude_set = set(
        df_exclude.loc[df_exclude.get("flag", pd.Series()) == "exclude", "adm3_name"]
        if "adm3_name" in df_exclude.columns else []
    )
    logging.info(f"Excluding {len(exclude_set)} cells from map generation.")

    # Load woreda polygons and restrict to dissemination cells
    woredas = load_woreda_geometries(base, woreda_shapefile)
    woredas = woredas[woredas["adm3_name"].isin(valid_ids - exclude_set)]
    logging.info(f"Woreda polygons loaded: {len(woredas)} dissemination cells.")

    first_date = pd.to_datetime(summary["time"].iloc[0])
    date_str_fmt = first_date.strftime("%Y%m%d")

    output_dir = output_path / ("maps_ref" if ref_onset else "maps")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare forecast data — normalise id column
    preds_df = summary.copy()
    if "id" not in preds_df.columns and "adm3_name" in preds_df.columns:
        preds_df["id"] = preds_df["adm3_name"]
    preds_df = preds_df[~preds_df["id"].isin(exclude_set)]

    # Bins present in the data (any number) -- plot ALL of them.
    bin_nums = sorted(int(m.group(1)) for c in preds_df.columns
                      for m in [re.match(r"^week(\d+)$", c)] if m) or [1, 2, 3, 4]
    NB = len(bin_nums)
    for i in bin_nums:
        preds_df[f"Climatology_p_{i}"] = preds_df[f"clim_week{i}"]
        preds_df[f"Forecast_p_{i}"]    = preds_df[f"week{i}"]
    preds_df["Forecast_p_later"] = preds_df.get(
        "later", 1 - preds_df[[f"Forecast_p_{i}" for i in bin_nums]].sum(axis=1)
    )
    preds_df["Climatology_p_later"] = (
        1 - preds_df[[f"Climatology_p_{i}" for i in bin_nums]].sum(axis=1)
    )

    # Map extent
    minx, miny, maxx, maxy = country_gdf.total_bounds
    x_min, x_max, y_min, y_max = minx, maxx, miny, maxy
    if zoom_to_data:
        wb = woredas.total_bounds
        x_min, x_max, y_min, y_max = wb[0], wb[2], wb[1], wb[3]
        logging.info("Map extent: zoomed to woreda bounding box")
    else:
        logging.info("Map extent: full country boundary")

    # Color schemes
    #period_order  = ['just_week1', 'weeks12', 'weeks23', 'weeks34', 'weeks4later', 'later']
    period_order  = [f'week{i}' for i in bin_nums] + ['later']
    #plasma_cmap   = plt.get_cmap('plasma')
    plasma_cmap   = plt.get_cmap('Blues_r')
    stops         = np.linspace(0.2, 1.0, len(period_order))
    period_colors = {k: plasma_cmap(s) for k, s in zip(period_order, stops)}
    period_colors['none'] = '#d3d3d3'

    #prob_bins = [0, 0.1, 0.2, 0.3, 0.4, 1.0]
    #prob_bins = [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    prob_bins = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0]
    #prob_cmap = ListedColormap(plt.get_cmap('plasma_r')(np.linspace(0, 1, len(prob_bins) - 1)))
    prob_cmap = ListedColormap(plt.get_cmap('Blues')(np.linspace(0, 1, len(prob_bins) - 1)))
    prob_norm = BoundaryNorm(prob_bins, ncolors=len(prob_bins) - 1, clip=True)

    def max_period(vf):
        #if vf[0] >= 0.65: return 'just_week1'
        #if vf[4] >= 0.65: return 'later'
        #sums = [vf[0]+vf[1], vf[1]+vf[2], vf[2]+vf[3], vf[3]+vf[4]]
        #return ['weeks12', 'weeks23', 'weeks34', 'weeks4later'][int(np.argmax(sums))]

        period_keys = [f'week{i}' for i in bin_nums] + ['later']
        return period_keys[int(np.argmax(vf))]

    # ------------------------------------------------------------------
    # SECTION: Weekly Probability Maps
    # ------------------------------------------------------------------
    for t, grp in preds_df.groupby('time'):
        ds_str = pd.Timestamp(t).strftime('%Y-%m-%d')
        week_titles = {
            i: (f"{(pd.Timestamp(t) + timedelta(days=(i-1)*7+1)).strftime('%m/%d/%Y')} - "
                f"{(pd.Timestamp(t) + timedelta(days=i*7)).strftime('%m/%d/%Y')}")
            for i in bin_nums
        }

        # Join forecast probs onto woreda geometries
        merged = woredas.merge(grp[["id"] + [f"Forecast_p_{i}" for i in bin_nums] +
                                    [f"Climatology_p_{i}" for i in bin_nums]],
                               left_on="adm3_name", right_on="id", how="left")

        fig, axes = plt.subplots(1, NB, figsize=(4.5 * NB, 5), sharex=True, sharey=True,
                                 gridspec_kw={'wspace': 0.1})
        axes = np.atleast_1d(axes)
        for i, ax in zip(bin_nums, axes):
            country_gdf.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')

            # Fill with forecast prob colour
            col = f"Forecast_p_{i}"
            clim_col = f"Climatology_p_{i}"
            merged[col] = pd.to_numeric(merged[col], errors='coerce')
            merged[clim_col] = pd.to_numeric(merged[clim_col], errors='coerce')

            # Grey for missing, colourmap for present
            merged_na  = merged[merged[col].isna()]
            merged_ok  = merged[merged[col].notna()]

            if not merged_na.empty:
                merged_na.plot(ax=ax, color=period_colors['none'], edgecolor='none', zorder=1)
            if not merged_ok.empty:
                merged_ok.plot(ax=ax, column=col, cmap=prob_cmap, norm=prob_norm,
                               edgecolor='none', zorder=2)
                # 1. Get the face colors by applying the cmap and norm to the data
                face_colors = prob_cmap(prob_norm(merged_ok[col]))
            
                # 2. Calculate luminance: 0.299*R + 0.587*G + 0.114*B
                # face_colors is an array of (R, G, B, A)
                luminance = (0.299 * face_colors[:, 0] +
                             0.587 * face_colors[:, 1] +
                             0.114 * face_colors[:, 2])
            
                # 3. Choose edge color: black for bright backgrounds, white for dark
                edge_colors = ['black' if l > 0.5 else 'white' for l in luminance]

                merged_ok.plot(ax=ax, column=col, cmap=prob_cmap, norm=prob_norm,
                               edgecolor=edge_colors, linewidth=0.2,zorder=2)

            # Climatology border overlay
            lo = merged_ok[merged_ok[col] <= merged_ok[clim_col] - 0.10]
            hi = merged_ok[merged_ok[col] >= merged_ok[clim_col] + 0.10]
            if not lo.empty:
                lo.plot(ax=ax, facecolor='none', edgecolor='red',   linewidth=0.05, zorder=3)
            if not hi.empty:
                hi.plot(ax=ax, facecolor='none', edgecolor='green', linewidth=0.2, zorder=3)

            ax.set_title(week_titles[i], fontsize=10, pad=6)
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            #ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
            ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)

        sm = plt.cm.ScalarMappable(norm=prob_norm, cmap=prob_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=list(axes), orientation='horizontal', fraction=0.04, pad=0.08)
        cbar.set_label('Probability')
        cbar_ticks = prob_bins
        cbar.set_ticks(cbar_ticks)
        cbar_tick_labels = [str(v) for v in cbar_ticks]
        cbar_tick_labels[-2] = '>0.8'
        cbar_tick_labels[-1] = ''
        cbar.set_ticklabels(cbar_tick_labels)
        legend_handles = [
            Patch(facecolor='none', edgecolor=c, linewidth=1.5, label=l)
            for c, l in [('red',   '≥10% lower than climatology'),
                         ('green', '≥10% higher than climatology')]
        ]
#        axes[-1].legend(handles=legend_handles, loc='lower right',
#                        fontsize=7, handlelength=1.5, handleheight=1.5,
#                        borderpad=0.5, labelspacing=0.3, framealpha=0.8)
        fig.subplots_adjust(left=0.03, right=0.98, top=0.90, bottom=0.22)

        suffix = '_ref' if ref_onset else ''
        fname = output_dir / f"prob_bins_{ds_str}{suffix}.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        logging.info(f"Saved weekly probability map: {fname}")

        # NetCDF export
        prob_cols = [f"Forecast_p_{i}" for i in bin_nums] + ["Forecast_p_later"]
        nc_path = os.path.join(output_dir, f"weekly_probs_{date_str_fmt}{suffix}.nc")
        export_to_netcdf(grp, prob_cols, nc_path, pd.Timestamp(t).date())

        plt.close(fig)

    # ------------------------------------------------------------------
    # SECTION: Max-Period Map
    # ------------------------------------------------------------------
    for t, grp in preds_df.groupby('time'):
        ds_str = pd.Timestamp(t).strftime('%Y-%m-%d')

        merged = woredas.merge(
            grp[["id"] + [f"Forecast_p_{i}" for i in bin_nums] + ["Forecast_p_later"]],
            left_on="adm3_name", right_on="id", how="left"
        )

        def _row_max_period(row):
            vf = [row.get(f"Forecast_p_{i}", np.nan) for i in bin_nums] + \
                 [row.get("Forecast_p_later", np.nan)]
            if any(pd.isna(vf)):
                return 'none'
            return max_period(vf)

        merged["period"] = merged.apply(_row_max_period, axis=1)

        handles = [
            Patch(facecolor=period_colors[k], edgecolor='none', label=k)
            for k in period_order
        ]
        handles.append(Patch(facecolor=period_colors['none'], edgecolor='none',
                             label='No forecast/onset'))

        fig, ax = plt.subplots(figsize=(6, 6))
        country_gdf.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')
        for period_key, color in period_colors.items():
            subset = merged[merged["period"] == period_key]
            if not subset.empty:
                #subset.plot(ax=ax, color=color, edgecolor='none', zorder=1)
                # Convert hex/named color to RGB for luminance check
                from matplotlib.colors import to_rgb
                rgb = to_rgb(color)
                l = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
                contrast_edge = 'black' if l > 0.5 else 'white'

                subset.plot(ax=ax, color=color, edgecolor=contrast_edge, linewidth=0.2, zorder=1)

        ax.legend(handles=handles, title='Period with Max Probability of Onset',
                  loc='lower left', ncol=2, fontsize=9, title_fontsize=7,
                  handlelength=1.5, handleheight=1.5,
                  borderpad=0.5, labelspacing=0.3, framealpha=0.8)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)

        suffix = '_ref' if ref_onset else ''
        fname = output_dir / f"map_max_period_{ds_str}{suffix}.png"
        plt.tight_layout(); plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.close(fig)
        logging.info(f"Saved max-period map: {fname}")

        # NetCDF export for max period
        def _identify_max_period_idx(row):
            probs = [row.get(f"Forecast_p_{i}", np.nan) for i in bin_nums] + \
                    [row.get("Forecast_p_later", np.nan)]
            if any(pd.isna(probs)):
                return np.nan
            return float(np.argmax(probs) + 1)

        grp = grp.copy()
        grp["max_period_index"] = grp.apply(_identify_max_period_idx, axis=1)
        nc_max_path = os.path.join(output_dir, f"max_period_index_{date_str_fmt}{suffix}.nc")
        export_to_netcdf(grp, ["max_period_index"], nc_max_path, pd.Timestamp(t).date())

    logging.info(f"All maps saved under {output_dir}")


# ---------------------------------------------------------------------------
# NetCDF export (adm3_name as string dimension)
# ---------------------------------------------------------------------------

def export_to_netcdf(df, columns, output_path, issue_date):
    """
    Export a forecast DataFrame to NetCDF with adm3_name as the spatial dimension.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'id' (adm3_name) and the columns listed in `columns`.
    columns : list of str
    output_path : str or Path
    issue_date : date
    """
    df = df.copy()
    if "id" not in df.columns and "adm3_name" in df.columns:
        df["id"] = df["adm3_name"]

    keep = ["id"] + [c for c in columns if c in df.columns]
    df = df[keep].drop_duplicates(subset=["id"])

    # Build xarray Dataset with adm3_name as 1-D string coordinate
    adm3_names = df["id"].astype(str).values
    data_vars = {}
    for col in columns:
        if col in df.columns:
            data_vars[col] = xr.DataArray(
                df[col].values.astype(float),
                dims=["adm3_name"],
                attrs={"issue_date": str(issue_date)},
            )

    ds = xr.Dataset(data_vars, coords={"adm3_name": adm3_names})
    ds.attrs["issue_date"] = str(issue_date)
    ds.to_netcdf(output_path)
    logging.info(f"NetCDF saved: {output_path}")


def save_to_netcdf(df, columns, output_path):
    """Convenience wrapper — calls export_to_netcdf with no issue_date."""
    export_to_netcdf(df, columns, output_path, issue_date="unknown")
