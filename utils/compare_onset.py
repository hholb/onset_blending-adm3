"""
compare_onset.py
================
Compare monsoon onset between an observation file and a model-forecast file
for the same (year, lat, lon) cell.  Produces a single figure with two line
subplots sharing the same x-axis so the daily rainfall time series and their
detected onset dates can be compared directly.

Layout
------
  Top panel    : Observation rainfall (line) + rolling-sum (dashed, right axis)
                 Detected onset marked with a vertical red line.
  Bottom panel : Forecast rainfall (line, ensemble member or mean) +
                 rolling-sum (dashed, right axis).
                 Detected onset marked with a vertical orange line.
  Shared x-axis: weekly date ticks covering the plot window.
  Footer annotation: onset dates / difference in days.

Usage (run from repo root)
--------------------------
    python compare_onset.py \\
        --obs_file     /path/to/2000.nc \\
        --fct_file     /path/to/2003.nc \\
        --obs_spec     ref_rain_fixed_cutoff \\
        --fct_spec     ref_rain_fixed_cutoff \\
        --obs_year     2000 \\
        --fct_year     2003 \\
        --lat          9.0 \\
        --lon          39.0 \\
        --thresh       20.0 \\
        --obs_col      precip \\
        --fct_col      tp \\
        --member       1 \\
        --overlap      nearest \\
        [--out         compare_onset.png]

Notes
-----
- --obs_spec and --fct_spec can be the same if both files share the same
  onset definition (common case).
- --obs_year and --fct_year default to the same value if only --year is given.
- All onset-definition parameters come from the spec yml; individual fields
  can be overridden with --win / --wet_day_min_mm / etc.
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Path setup — allow running from the repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from python.prepare_data.onset_utils import (
    find_onset, read_onset_params, read_thresholds,
    roll_sum_na_rm_left,
)
from python.prepare_data.nc_utils import (
    get_nc_coord, get_nc_time, nc_time_to_dates,
)
from python.pipelines._shared.read_spec import load_spec


# ===========================================================================
# Extraction helpers  (shared with diagnose_onset.py logic)
# ===========================================================================

def _detect_format(ds):
    dim_lower = {d.lower() for d in ds.dimensions}
    has_member = any(d in dim_lower for d in ("number", "member", "ensemble"))
    has_day    = any(d in dim_lower for d in ("day", "lead", "step"))
    return "forecast" if (has_member and has_day) else "obs"


def _find_dim_idx(candidates, dim_names):
    lower = [d.lower() for d in dim_names]
    for c in candidates:
        for i, d in enumerate(lower):
            if d == c.lower():
                return i, dim_names[i]
    return None, None


def _resolve_dim_name(ds, candidates):
    for c in candidates:
        for name in list(ds.dimensions) + list(ds.variables):
            if name.lower() == c.lower():
                return name
    return None


def _extract_obs(ds, lat_idx, lon_idx, var_name, dates, year, cutoff_md, max_days):
    raw       = ds.variables[var_name]
    dim_names = list(raw.dimensions)
    ndim      = len(dim_names)

    t_pos,   _ = _find_dim_idx(["time", "t"], dim_names)
    lat_pos, _ = _find_dim_idx(["lat", "latitude"], dim_names)
    lon_pos, _ = _find_dim_idx(["lon", "longitude", "long"], dim_names)

    if None in (t_pos, lat_pos, lon_pos):
        raise ValueError(
            f"[obs] Cannot identify time/lat/lon in dimensions {dim_names}."
        )

    idx = [slice(None)] * ndim
    idx[lat_pos] = lat_idx
    idx[lon_pos] = lon_idx
    series = np.array(raw[tuple(idx)], dtype=float).flatten()

    cutoff_date = pd.Timestamp(f"{year}-{cutoff_md}")
    mask        = (dates >= cutoff_date) & (dates <= pd.Timestamp(f"{year}-12-31"))
    dates_out, series_out = dates[mask], series[mask]

    if max_days is not None:
        dates_out, series_out = dates_out[:max_days], series_out[:max_days]
    return dates_out, series_out


def _extract_forecast(ds, lat_idx, lon_idx, var_name, year,
                      cutoff_md, max_days, member, overlap,
                      dim_time, dim_day, dim_number, dim_lat, dim_lon):
    import netCDF4 as nc4

    raw       = ds.variables[var_name]
    dim_names = list(raw.dimensions)
    ndim      = len(dim_names)

    def _pos(override, candidates):
        name = override or _resolve_dim_name(ds, candidates)
        if name is None:
            raise ValueError(f"Cannot find dimension {candidates} in {dim_names}.")
        lower = [d.lower() for d in dim_names]
        for i, d in enumerate(lower):
            if d == name.lower():
                return i, dim_names[i]
        raise ValueError(f"Dimension '{name}' not found in {dim_names}")

    t_pos,   t_name   = _pos(dim_time,   ["time", "t"])
    day_pos, day_name = _pos(dim_day,    ["day", "lead", "step"])
    num_pos, num_name = _pos(dim_number, ["number", "member", "ensemble"])
    lat_pos, _        = _pos(dim_lat,    ["lat", "latitude"])
    lon_pos, _        = _pos(dim_lon,    ["lon", "longitude", "long"])

    init_dates = nc_time_to_dates(
        np.array(ds.variables[t_name][:]).flatten(),
        getattr(ds.variables[t_name], "units", None),
    )
    day_vals = np.array(ds.variables[day_name][:]).flatten().astype(int)
    num_vals = np.array(ds.variables[num_name][:]).flatten().astype(int)

    if member == 0:
        member_slice   = slice(None)
        do_member_mean = True
    else:
        matches = np.where(num_vals == member)[0]
        if len(matches) == 0:
            warnings.warn(
                f"Member {member} not in '{num_name}' coordinate; "
                f"using positional index {member - 1}."
            )
            member_slice = member - 1
        else:
            member_slice = int(matches[0])
        do_member_mean = False

    base_idx = [slice(None)] * ndim
    base_idx[lat_pos] = lat_idx
    base_idx[lon_pos] = lon_idx

    date_data = {}
    for ti in range(len(init_dates)):
        init = init_dates[ti]
        idx  = list(base_idx)
        idx[t_pos] = ti

        if do_member_mean:
            chunk = np.array(raw[tuple(idx)], dtype=float)
            free_dims = [(i, n) for i, n in enumerate(dim_names)
                         if i not in (lat_pos, lon_pos, t_pos)]
            chunk_num_axis = next(j for j, (i, _) in enumerate(free_dims) if i == num_pos)
            values_per_day = np.nanmean(chunk, axis=chunk_num_axis).flatten()
        else:
            idx[num_pos] = member_slice
            values_per_day = np.array(raw[tuple(idx)], dtype=float).flatten()

        for di, day_offset in enumerate(day_vals):
            actual_date = init + pd.Timedelta(days=int(day_offset) - 1)
            lead        = int(day_offset)
            date_data.setdefault(actual_date, []).append((lead, values_per_day[di]))

    all_dates = sorted(date_data.keys())
    all_vals  = []
    for d in all_dates:
        entries = date_data[d]
        if len(entries) == 1:
            all_vals.append(entries[0][1])
        elif overlap == "nearest":
            all_vals.append(min(entries, key=lambda x: x[0])[1])
        elif overlap == "longest":
            all_vals.append(max(entries, key=lambda x: x[0])[1])
        elif overlap == "mean":
            all_vals.append(float(np.nanmean([v for _, v in entries])))
        elif overlap == "first":
            all_vals.append(max(entries, key=lambda x: x[0])[1])
        else:
            raise ValueError(f"Unknown --overlap: {overlap!r}")

    idx_all   = pd.DatetimeIndex(all_dates)
    arr_all   = np.array(all_vals, dtype=float)
    cutoff_ts = pd.Timestamp(f"{year}-{cutoff_md}")
    mask      = (idx_all >= cutoff_ts) & (idx_all <= pd.Timestamp(f"{year}-12-31"))
    dates_out, series_out = idx_all[mask], arr_all[mask]

    if max_days is not None:
        dates_out, series_out = dates_out[:max_days], series_out[:max_days]
    return dates_out, series_out


def extract_cell_series(nc_file, year, lat, lon, value_col,
                        cutoff_md="01-01", max_days=None,
                        member=1, overlap="nearest",
                        dim_time=None, dim_day=None, dim_number=None,
                        dim_lat=None, dim_lon=None):
    """Load one cell's daily series; auto-detects obs vs forecast format."""
    import netCDF4 as nc4
    ds  = nc4.Dataset(nc_file)
    fmt = _detect_format(ds)
    print(f"  [format detected: {fmt}]")

    var_name = next((v for v in ds.variables if v.lower() == value_col.lower()), None)
    if var_name is None:
        raise ValueError(
            f"Variable '{value_col}' not found in {nc_file}. "
            f"Available: {list(ds.variables.keys())}"
        )

    lats    = get_nc_coord(ds, "lat")
    lons    = get_nc_coord(ds, "lon")
    lat_idx = int(np.argmin(np.abs(lats - lat)))
    lon_idx = int(np.argmin(np.abs(lons - lon)))
    actual_lat, actual_lon = float(lats[lat_idx]), float(lons[lon_idx])

    if abs(actual_lat - lat) > 1.0 or abs(actual_lon - lon) > 1.0:
        warnings.warn(
            f"Requested ({lat}, {lon}), nearest point is ({actual_lat}, {actual_lon})."
        )

    if fmt == "obs":
        t_info = get_nc_time(ds)
        if t_info["units"] is None:
            raise ValueError("Obs NetCDF time variable has no 'units' attribute.")
        dates = nc_time_to_dates(t_info["values"], t_info["units"])
        dates_out, series_out = _extract_obs(
            ds, lat_idx, lon_idx, var_name, dates, year, cutoff_md, max_days
        )
    else:
        print(f"  [forecast] member={member}, overlap='{overlap}'")
        dates_out, series_out = _extract_forecast(
            ds, lat_idx, lon_idx, var_name, year,
            cutoff_md, max_days, member, overlap,
            dim_time, dim_day, dim_number, dim_lat, dim_lon,
        )

    ds.close()
    return dates_out, series_out, actual_lat, actual_lon, fmt


# ===========================================================================
# Threshold / MOK helpers
# ===========================================================================

def resolve_threshold(spec, lat, lon, thresh_override):
    if thresh_override is not None:
        return thresh_override
    thr_dt = read_thresholds(spec)
    if thr_dt is None:
        raise ValueError("No --thresh provided and no thresholds file in spec.")
    if isinstance(thr_dt, (int, float)):
        return float(thr_dt)
    dists   = np.sqrt((thr_dt["lat"] - lat) ** 2 + (thr_dt["lon"] - lon) ** 2)
    nearest = thr_dt.iloc[dists.argmin()]
    print(f"  threshold from file (nearest cell {nearest['lat']}, {nearest['lon']}): "
          f"{nearest['onset_thresh']:.2f} mm")
    return float(nearest["onset_thresh"])


def resolve_ref(spec, year):
    ref_onset_cfg = spec.get("ref_onset")
    if not ref_onset_cfg or not ref_onset_cfg.get("file"):
        return None
    mok_file = ref_onset_cfg["file"]
    if not os.path.exists(mok_file):
        print(f"  MOK file not found: {mok_file}")
        return None
    mok_df      = pd.read_csv(mok_file)
    ycol        = ref_onset_cfg.get("year_col", "Year")
    dcol        = ref_onset_cfg.get("day_col",  "MOK")
    base_md     = ref_onset_cfg.get("base_date", "01-01")
    row         = mok_df[mok_df[ycol] == year]
    if row.empty:
        print(f"  MOK date: year {year} not found.")
        return None
    mok_day = int(row.iloc[0][dcol])
    ref_onset_ts  = pd.Timestamp(f"{year}-{base_md}") + pd.Timedelta(days=mok_day)
    print(f"  MOK date for {year}: {ref_onset_ts.date()} ({mok_day} days after {base_md})")
    return ref_onset_ts


def mok_start_day(ref_onset_date, dates):
    if ref_onset_date is None:
        return 0
    matches = np.where(dates == pd.Timestamp(ref_onset_date))[0]
    return int(matches[0]) + 1 if len(matches) > 0 else 0


# ===========================================================================
# Onset param override
# ===========================================================================

def apply_overrides(params, args):
    d = params._asdict()
    mapping = dict(
        win=args.win, wet_day_min_mm=args.wet_day_min_mm,
        follow_days=args.follow_days, mode=args.mode,
        min_dry_days=args.min_dry_days, dry_day_min_mm=args.dry_day_min_mm,
        sum_window=args.sum_window, sum_min_mm=args.sum_min_mm,
    )
    changed = {k: v for k, v in mapping.items() if v is not None}
    d.update(changed)
    if changed:
        print(f"  [onset overrides]: {changed}")
    return type(params)(**d)


# ===========================================================================
# Shared x-axis tick builder
# ===========================================================================

def _build_xticks(all_dates_list, plot_start, plot_end, tick_every=7):
    """
    Return (positions, labels) for weekly ticks over the union date range.

    positions : list of pd.Timestamp  (used as x-values in both panels)
    labels    : list of str  "Mon DD"
    """
    # All unique dates across both series within the plot window
    dates_in_window = sorted(
        d for d in all_dates_list if plot_start <= d <= plot_end
    )
    if not dates_in_window:
        return [], []

    first = dates_in_window[0]
    ticks, labels = [], []
    for d in dates_in_window:
        delta = (d - first).days
        if delta == 0 or delta % tick_every == 0:
            ticks.append(d)
            labels.append(d.strftime("%b %d"))
    return ticks, labels


# ===========================================================================
# The comparison plot
# ===========================================================================

def plot_comparison(
    obs_dates,  obs_series,  obs_onset_idx,  obs_params,  obs_thresh,
    fct_dates,  fct_series,  fct_onset_idx,  fct_params,  fct_thresh,
    actual_lat, actual_lon,
    obs_year,   fct_year,
    obs_label,  fct_label,
    ref_onset_date=None,
    plot_start_md="05-01", plot_end_md="07-31",
    member=1, overlap="nearest",
    out_path="compare_onset.png",
):
    """
    Single-panel comparison figure.

    Both observation and forecast rainfall lines are drawn on one axes,
    with a shared rolling-sum right axis.  The x-axis starts at the MOK date
    (or plot_start_md if no MOK date is available) and ends at plot_end_md.
    """
    # ---- colour palette ----
    OBS_LINE   = "#2166ac"   # blue
    OBS_ROLL   = "#4393c3"   # lighter blue (dashed)
    OBS_ONSET  = "#d6212b"   # red
    FCT_LINE   = "#1a7a3c"   # green
    FCT_ROLL   = "#5aad6f"   # lighter green (dashed)
    FCT_ONSET  = "#e87722"   # orange
    MOK_COLOR  = "#666666"
    THRESH_CLR = "#8e44ad"
    BG         = "#f8f9fa"

    # ---- align forecast year onto obs year for the x-axis ----
    year_offset = obs_year - fct_year
    if year_offset != 0:
        fct_dates_aligned = fct_dates + pd.DateOffset(years=year_offset)
        print(f"  [year alignment] shifting forecast dates by {year_offset} year(s) "
              f"to align with observation year {obs_year}.")
    else:
        fct_dates_aligned = fct_dates

    # ---- plot window: start at MOK date (or plot_start_md), end at plot_end_md ----
    plot_end = pd.Timestamp(f"{obs_year}-{plot_end_md}")
    if ref_onset_date is not None:
        plot_start = pd.Timestamp(ref_onset_date).replace(year=obs_year)
    else:
        plot_start = pd.Timestamp(f"{obs_year}-{plot_start_md}")

    # ---- onset calendar dates ----
    def _onset_date(dates_arr, onset_idx):
        return dates_arr[onset_idx - 1] if onset_idx is not None else None

    obs_onset_date = _onset_date(obs_dates,         obs_onset_idx)
    fct_onset_date = _onset_date(fct_dates_aligned, fct_onset_idx)

    # ---- onset difference text ----
    if obs_onset_date is not None and fct_onset_date is not None:
        diff_days = (fct_onset_date - obs_onset_date).days
        diff_str  = (f"Forecast onset {abs(diff_days)}d "
                     + ("earlier" if diff_days < 0 else
                        "later"   if diff_days > 0 else "same day")
                     + " than observation")
    else:
        diff_str = "Onset comparison unavailable (one or both series: no onset detected)"

    # ---- slice both series to the plot window ----
    def _window(dates_arr, vals):
        m = (dates_arr >= plot_start) & (dates_arr <= plot_end)
        return dates_arr[m], np.array(vals, dtype=float)[m]

    obs_pd, obs_pv = _window(obs_dates,         obs_series)
    fct_pd, fct_pv = _window(fct_dates_aligned, fct_series)

    # ================================================================
    # Single-panel figure
    # ================================================================
    fig, ax = plt.subplots(figsize=(17, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax2 = ax.twinx()   # right axis shared by both rolling sums

    def _draw_series(pd_, pv_, dates_full, onset_idx,
                     params, line_col, roll_col, onset_col, label):
        if len(pd_) == 0:
            return
        # rainfall line + shaded fill
        ax.plot(pd_, pv_, color=line_col, lw=1.6, alpha=0.85,
                label=f"{label} — daily rainfall", zorder=3)
        ax.fill_between(pd_, 0, pv_, color=line_col, alpha=0.08, zorder=2)
        # rolling sum on right axis — roll_sum_na_rm_left returns n-win+1 values
        wsum = roll_sum_na_rm_left(pv_, params.win)
        ax2.plot(pd_[:len(wsum)], wsum,
                 color=roll_col, lw=1.2, ls="--", alpha=0.75,
                 label=f"{label} — {params.win}-day sum")
        # onset vertical line
        if onset_idx is not None:
            od = dates_full[onset_idx - 1]
            if plot_start <= od <= plot_end:
                ax.axvline(od, color=onset_col, lw=2.2, zorder=6,
                           label=f"Onset {label}: {od.strftime('%b %d')}")
                ax.text(od, 0, f"  {od.strftime('%b %d')}",
                        color=onset_col, fontsize=8, va="bottom",
                        fontweight="bold", rotation=90, zorder=7)
        else:
            ypos = 0.99 if line_col == OBS_LINE else 0.91
            ax.text(0.99, ypos, f"No onset — {label}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color=onset_col, fontweight="bold",
                    bbox=dict(fc="white", ec=onset_col, pad=2, alpha=0.85))

    _draw_series(obs_pd, obs_pv, obs_dates,         obs_onset_idx,
                 obs_params, OBS_LINE, OBS_ROLL, OBS_ONSET, obs_label)
    _draw_series(fct_pd, fct_pv, fct_dates_aligned, fct_onset_idx,
                 fct_params, FCT_LINE, FCT_ROLL, FCT_ONSET, fct_label)

    # ---- shared threshold line ----
    thresh_label = (f"Threshold {obs_thresh:.1f} mm"
                    + (f" (obs) / {fct_thresh:.1f} mm (fct)"
                       if fct_thresh != obs_thresh else ""))
    ax2.axhline(obs_thresh, color=THRESH_CLR, lw=1.0, ls=":", alpha=0.8,
                label=thresh_label)

    # ---- MOK date line ----
    if ref_onset_date is not None:
        ref_onset_ts = pd.Timestamp(ref_onset_date).replace(year=obs_year)
        ax.axvline(ref_onset_ts, color=MOK_COLOR, lw=1.4, ls="--", zorder=5,
                   label=f"MOK date: {ref_onset_ts.strftime('%b %d')}")

    # ---- right-axis decoration ----
    ax2.set_ylabel(f"{obs_params.win}-day rolling sum (mm)", fontsize=9, color="#555555")
    ax2.tick_params(axis="y", labelsize=8, labelcolor="#555555")
    ax2.spines["right"].set_color("#aaaaaa")

    # ---- left-axis decoration ----
    ax.set_ylabel("Daily rainfall (mm)", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)

    # ---- x-axis weekly ticks ----
    all_plot_dates = sorted(set(obs_pd.tolist()) | set(fct_pd.tolist()))
    tick_dates, tick_labels = _build_xticks(all_plot_dates, plot_start, plot_end)
    ax.set_xticks(tick_dates)
    ax.set_xticklabels(tick_labels, fontsize=7.5, rotation=40, ha="right")
    ax.set_xlabel("Date", fontsize=10)
    if tick_dates:
        ax.set_xlim(
            tick_dates[0]  - pd.Timedelta(days=1),
            tick_dates[-1] + pd.Timedelta(days=1),
        )

    # ---- legend ----
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right",
              framealpha=0.88, ncol=2)

    # ---- title ----
    mode_str   = (f"consecutive_dry (min_dry={obs_params.min_dry_days}d)"
                  if obs_params.mode == "consecutive_dry"
                  else f"window_sum (sum_win={obs_params.sum_window}d)")
    member_str = "ens. mean" if member == 0 else f"member {member}"
    year_note  = (f"[obs={obs_year}, fct={fct_year}]"
                  if obs_year != fct_year else f"[year={obs_year}]")
    ax.set_title(
        f"Onset comparison  —  lat={actual_lat:.2f}, lon={actual_lon:.2f}  {year_note}\n"
        f"Trigger: {obs_params.win}-day window \u2265 {obs_thresh:.1f} mm  |  "
        f"Dry-spell: {mode_str}  |  Forecast: {member_str}, overlap={overlap}",
        fontsize=9, pad=8,
    )

    # ---- footer: onset dates + difference ----
    obs_str = obs_onset_date.strftime("%b %d") if obs_onset_date else "none"
    fct_str = fct_onset_date.strftime("%b %d") if fct_onset_date else "none"
    footer  = f"Obs onset: {obs_str}    Forecast onset: {fct_str}    \u2192  {diff_str}"
    fig.text(0.5, 0.005, footer, ha="center", va="bottom",
             fontsize=9, color="#333333",
             bbox=dict(fc="#eef3f8", ec="#aac4df", pad=4, alpha=0.9))

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved \u2192 {out_path}")
    plt.close(fig)



# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare monsoon onset between an observation file and a model\n"
            "forecast file for the same (lat, lon) cell.\n"
            "Produces a dual-panel line-plot figure."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Files ----
    parser.add_argument("--obs_file", required=True,
                        help="Path to the observation / reanalysis NetCDF file.")
    parser.add_argument("--fct_file", required=True,
                        help="Path to the model forecast NetCDF file.")

    # ---- Specs ----
    parser.add_argument("--obs_spec", default=None,
                        help="Spec ID for the obs file (specs/raw_data/<id>.yml). "
                             "Falls back to --spec if not given.")
    parser.add_argument("--fct_spec", default=None,
                        help="Spec ID for the forecast file. "
                             "Falls back to --spec if not given.")
    parser.add_argument("--spec",     default=None,
                        help="Shared spec ID used for both files when "
                             "--obs_spec / --fct_spec are not separately set.")

    # ---- Year ----
    parser.add_argument("--year",     type=int, default=None,
                        help="Year for both files (used when obs and fct share the same year).")
    parser.add_argument("--obs_year", type=int, default=None,
                        help="Year for the observation file (overrides --year).")
    parser.add_argument("--fct_year", type=int, default=None,
                        help="Year for the forecast file (overrides --year).")

    # ---- Spatial ----
    parser.add_argument("--lat", required=True, type=float, help="Target latitude.")
    parser.add_argument("--lon", required=True, type=float, help="Target longitude.")

    # ---- Variable names ----
    parser.add_argument("--obs_col",   default=None,
                        help="NetCDF variable name in the obs file (default: from spec).")
    parser.add_argument("--fct_col",   default=None,
                        help="NetCDF variable name in the forecast file (default: from spec).")

    # ---- Threshold ----
    parser.add_argument("--thresh",     type=float, default=None,
                        help="Shared onset threshold (mm). "
                             "Reads from thresholds_df.csv if omitted.")
    parser.add_argument("--obs_thresh", type=float, default=None,
                        help="Per-file threshold override for obs (overrides --thresh).")
    parser.add_argument("--fct_thresh", type=float, default=None,
                        help="Per-file threshold override for forecast (overrides --thresh).")

    # ---- Cutoff ----
    parser.add_argument("--cutoff_md", default=None,
                        help="Season start as MM-DD (default: from spec or 01-01).")
    parser.add_argument("--max_days",  type=int, default=None,
                        help="Truncate both series to this many days after cutoff.")

    # ---- Forecast options ----
    fgrp = parser.add_argument_group("forecast options")
    fgrp.add_argument("--member",  type=int, default=1,
                      help="Ensemble member (1-indexed). 0 = ensemble mean. Default: 1.")
    fgrp.add_argument("--overlap", default="nearest",
                      choices=["nearest", "longest", "mean", "first"],
                      help="Overlap resolution strategy for forecast dates. Default: nearest.")

    # ---- Dimension name overrides ----
    dgrp = parser.add_argument_group("forecast dimension name overrides")
    dgrp.add_argument("--dim_time",   default=None)
    dgrp.add_argument("--dim_day",    default=None)
    dgrp.add_argument("--dim_number", default=None)
    dgrp.add_argument("--dim_lat",    default=None)
    dgrp.add_argument("--dim_lon",    default=None)

    # ---- Labels ----
    parser.add_argument("--obs_label", default=None,
                        help="Legend label for the observation panel "
                             "(default: 'Observation <year>').")
    parser.add_argument("--fct_label", default=None,
                        help="Legend label for the forecast panel "
                             "(default: 'Forecast <year>').")

    # ---- Output ----
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--out",     default=None,
                        help="Full output path (overrides --out_dir).")
    parser.add_argument("--plot_start_md", default="05-01",
                        help="Plot window start MM-DD (default: 05-01).")
    parser.add_argument("--plot_end_md",   default="07-31",
                        help="Plot window end MM-DD (default: 07-31).")

    # ---- Onset definition overrides (applied to both files) ----
    ogrp = parser.add_argument_group("onset definition overrides (applied to both files)")
    ogrp.add_argument("--win",            type=int,   default=None)
    ogrp.add_argument("--wet_day_min_mm", type=float, default=None)
    ogrp.add_argument("--follow_days",    type=int,   default=None)
    ogrp.add_argument("--mode",           type=str,   default=None,
                      choices=["consecutive_dry", "window_sum"])
    ogrp.add_argument("--min_dry_days",   type=int,   default=None)
    ogrp.add_argument("--dry_day_min_mm", type=float, default=None)
    ogrp.add_argument("--sum_window",     type=int,   default=None)
    ogrp.add_argument("--sum_min_mm",     type=float, default=None)

    args = parser.parse_args()

    # ---- resolve spec IDs ----
    obs_spec_id = args.obs_spec or args.spec
    fct_spec_id = args.fct_spec or args.spec
    if obs_spec_id is None or fct_spec_id is None:
        parser.error("Provide --spec (shared) or both --obs_spec and --fct_spec.")

    # ---- resolve years ----
    obs_year = args.obs_year or args.year
    fct_year = args.fct_year or args.year
    if obs_year is None or fct_year is None:
        parser.error("Provide --year (shared) or both --obs_year and --fct_year.")

    # ---- resolve files ----
    obs_file = os.path.abspath(args.obs_file)
    fct_file = os.path.abspath(args.fct_file)
    for f in (obs_file, fct_file):
        if not os.path.exists(f):
            raise FileNotFoundError(f"NetCDF file not found: {f}")

    # ---- output path ----
    if args.out:
        out_path = os.path.abspath(args.out)
    else:
        out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"compare_onset_{obs_year}_{fct_year}_{args.lat}_{args.lon}.png",
        )

    # ---- load specs ----
    obs_spec = load_spec(obs_spec_id, "raw_data")
    fct_spec = load_spec(fct_spec_id, "raw_data")

    obs_params = apply_overrides(read_onset_params(obs_spec), args)
    fct_params = apply_overrides(read_onset_params(fct_spec), args)

    # ---- cutoff ----
    cutoff_md = args.cutoff_md or obs_spec.get("options", {}).get("cutoff_month_day", "01-01")
    print(f"\n  cutoff_month_day : {cutoff_md}")

    # ---- variable names ----
    obs_col = args.obs_col or obs_spec["input"]["value_col"]
    fct_col = args.fct_col or fct_spec["input"]["value_col"]

    # ---- thresholds ----
    obs_thresh = resolve_threshold(obs_spec, args.lat, args.lon,
                                   args.obs_thresh or args.thresh)
    fct_thresh = resolve_threshold(fct_spec, args.lat, args.lon,
                                   args.fct_thresh or args.thresh)
    print(f"  obs threshold    : {obs_thresh:.2f} mm")
    print(f"  fct threshold    : {fct_thresh:.2f} mm")

    # ---- labels ----
    obs_label = args.obs_label or f"Observation {obs_year}"
    fct_label = (args.fct_label or
                 f"Forecast {fct_year}"
                 + (f" (member {args.member})" if args.member != 0 else " (ens. mean)"))

    # ---- extract observation series ----
    print(f"\n--- Observation ---")
    print(f"  file: {obs_file}")
    obs_dates, obs_series, actual_lat, actual_lon, _ = extract_cell_series(
        nc_file   = obs_file,
        year      = obs_year,
        lat       = args.lat,
        lon       = args.lon,
        value_col = obs_col,
        cutoff_md = cutoff_md,
        max_days  = args.max_days,
    )
    print(f"  extracted {len(obs_series)} days: "
          f"{obs_dates[0].date()} → {obs_dates[-1].date()}")

    # ---- extract forecast series ----
    print(f"\n--- Forecast ---")
    print(f"  file: {fct_file}")
    fct_dates, fct_series, _, _, _ = extract_cell_series(
        nc_file    = fct_file,
        year       = fct_year,
        lat        = args.lat,
        lon        = args.lon,
        value_col  = fct_col,
        cutoff_md  = cutoff_md,
        max_days   = args.max_days,
        member     = args.member,
        overlap    = args.overlap,
        dim_time   = args.dim_time,
        dim_day    = args.dim_day,
        dim_number = args.dim_number,
        dim_lat    = args.dim_lat,
        dim_lon    = args.dim_lon,
    )
    print(f"  extracted {len(fct_series)} days: "
          f"{fct_dates[0].date()} → {fct_dates[-1].date()}")

    # ---- MOK date (from obs spec, obs year) ----
    ref_onset_date = resolve_ref(obs_spec, obs_year)

    # ---- detect onset ----
    obs_start = mok_start_day(ref_onset_date, obs_dates)
    fct_start = mok_start_day(ref_onset_date, fct_dates)

    obs_onset_idx = find_onset(obs_series, thresh=obs_thresh,
                               params=obs_params, start_day=obs_start)
    fct_onset_idx = find_onset(fct_series, thresh=fct_thresh,
                               params=fct_params, start_day=fct_start)

    def _fmt_onset(dates_arr, idx, label):
        if idx is None:
            print(f"  {label}: no onset detected")
        else:
            print(f"  {label}: day {idx} → {dates_arr[idx - 1].date()}")

    print()
    _fmt_onset(obs_dates, obs_onset_idx, obs_label)
    _fmt_onset(fct_dates, fct_onset_idx, fct_label)

    # ---- plot ----
    plot_comparison(
        obs_dates      = obs_dates,
        obs_series     = obs_series,
        obs_onset_idx  = obs_onset_idx,
        obs_params     = obs_params,
        obs_thresh     = obs_thresh,
        fct_dates      = fct_dates,
        fct_series     = fct_series,
        fct_onset_idx  = fct_onset_idx,
        fct_params     = fct_params,
        fct_thresh     = fct_thresh,
        actual_lat     = actual_lat,
        actual_lon     = actual_lon,
        obs_year       = obs_year,
        fct_year       = fct_year,
        obs_label      = obs_label,
        fct_label      = fct_label,
        ref_onset_date       = ref_onset_date,
        plot_start_md  = args.plot_start_md,
        plot_end_md    = args.plot_end_md,
        member         = args.member,
        overlap        = args.overlap,
        out_path       = out_path,
    )


if __name__ == "__main__":
    main()
