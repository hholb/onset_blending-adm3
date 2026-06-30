# ==============================================================================
# File: onset_utils.py
# ==============================================================================
# Purpose
#   Low-level utilities for computing monsoon onset indices from
#   rainfall time series.
#
# Functions
#   read_mok_dates(spec)
#   read_thresholds(spec)
#   roll_sum_na_rm_left(x, k)
#   roll_sum_na_propagate_left(x, k)
#   find_onset(series, win, thresh, reject_if_short_followup, start_day)
#   find_onset_precomp(series, win, thresh, wsum_all, pre_bad, last10start, ...)
# ==============================================================================

import os
import numpy as np
import pandas as pd
from datetime import date


def read_mok_dates(spec):
    """
    Read Monsoon Onset Kerala (MOK) dates from the file specified in spec.

    Returns a DataFrame with columns: year (int), mok_date (datetime.date).
    Returns None if spec["mok"]["file"] is not set.
    """
    if spec.get("mok") is None or spec["mok"].get("file") is None:
        return None

    f = spec["mok"]["file"]
    if not os.path.exists(f):
        raise FileNotFoundError(f"MOK file not found: {f}")

    mok = pd.read_csv(f)
    ycol = spec["mok"]["year_col"]
    dcol = spec["mok"]["day_col"]
    if ycol not in mok.columns or dcol not in mok.columns:
        raise ValueError(f"MOK file must contain columns '{ycol}' and '{dcol}'")

    base_md = spec["mok"]["base_date"]  # e.g. "01-01"
    mok["mok_date"] = mok.apply(
        lambda row: pd.to_datetime(f"{int(row[ycol])}-{base_md}") + pd.Timedelta(days=int(row[dcol])),
        axis=1
    ).dt.date
    return mok[[ycol, "mok_date"]].rename(columns={ycol: "year"}).assign(year=lambda d: d["year"].astype(int))


def read_thresholds(spec):
    """
    Read per-grid-cell onset thresholds.

    Returns a DataFrame with columns: lat, lon, onset_thresh.
    Returns None if spec["thresholds"]["file"] is not set.
    """
    if spec.get("thresholds") is None or spec["thresholds"].get("file") is None:
        return None

    f = spec["thresholds"]["file"]

    # Case 1: scalar threshold (int or float)
    if isinstance(f, (int, float)):
        return f

    # Optional: handle numeric strings like "0.5"
    try:
        val = float(f)
        return val
    except (TypeError, ValueError):
        pass

    if not os.path.exists(f):
        raise FileNotFoundError(f"Threshold file not found: {f}")

    ext = os.path.splitext(f)[1].lower()

    if ext in (".nc", ".nc4", ".netcdf"):
        import netCDF4 as nc4
        ds = nc4.Dataset(f)
        lat_name = spec["thresholds"].get("lat_var") or spec["thresholds"].get("lat_col") or "lat"
        lon_name = spec["thresholds"].get("lon_var") or spec["thresholds"].get("lon_col") or "lon"
        thr_name = spec["thresholds"].get("thresh_var") or spec["thresholds"].get("thresh_col") or "MWmean"
        lats = np.array(ds[lat_name][:]).flatten()
        lons = np.array(ds[lon_name][:]).flatten()
        thresh = np.array(ds[thr_name][:]).flatten()
        ds.close()
        return pd.DataFrame({"lat": lats, "lon": lons, "onset_thresh": thresh}).drop_duplicates()

    if ext == ".mat":
        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError("scipy is required to read .mat threshold files.")
        lat_name = spec["thresholds"].get("mat_lat_name", "lat")
        lon_name = spec["thresholds"].get("mat_lon_name", "lon")
        thr_name = spec["thresholds"].get("mat_thresh_name", "onset.day.thres")
        md = sio.loadmat(f)
        lats = np.array(md[lat_name]).flatten()
        lons = np.array(md[lon_name]).flatten()
        thresh = np.array(md[thr_name])
        grid = pd.DataFrame(
            [(la, lo) for la in lats for lo in lons],
            columns=["lat", "lon"]
        )
        grid["onset_thresh"] = thresh.flatten()
        return grid.drop_duplicates()

    # CSV/TSV
    th = pd.read_csv(f)
    tc = spec["thresholds"].get("thresh_col", "onset_thresh")

    # adm3_name-based thresholds (new format)
    adm3_col = spec["thresholds"].get("adm3_col", None)
    if adm3_col and adm3_col in th.columns:
        th = th.rename(columns={adm3_col: "id", tc: "onset_thresh"})
        return th[["id", "onset_thresh"]].drop_duplicates()
    if "adm3_name" in th.columns:
        th = th.rename(columns={"adm3_name": "id", tc: "onset_thresh"})
        return th[["id", "onset_thresh"]].drop_duplicates()
    if "id" in th.columns:
        th = th.rename(columns={tc: "onset_thresh"})
        return th[["id", "onset_thresh"]].drop_duplicates()

    # Legacy lat/lon-based thresholds
    lc = spec["thresholds"].get("lat_col", "lat")
    oc = spec["thresholds"].get("lon_col", "lon")
    th = th.rename(columns={lc: "lat", oc: "lon", tc: "onset_thresh"})
    return th[["lat", "lon", "onset_thresh"]].drop_duplicates()


# ---------------------------------------------------------------------------
# Fast rolling helpers
# ---------------------------------------------------------------------------

def roll_sum_na_rm_left(x, k):
    """
    Left-aligned rolling sum of length k with NA treated as 0 (na.rm=TRUE).
    Returns array of length max(0, n - k + 1).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if k <= 0 or n < k:
        return np.array([], dtype=float)
    x0 = np.where(np.isnan(x), 0.0, x)
    cs = np.concatenate([[0.0], np.cumsum(x0)])
    return cs[k:] - cs[:n - k + 1]


def roll_sum_na_propagate_left(x, k):
    """
    Left-aligned rolling sum where any NA in the window propagates to the sum.
    Returns array of length max(0, n - k + 1).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if k <= 0 or n < k:
        return np.array([], dtype=float)
    na = np.isnan(x).astype(float)
    x0 = np.where(np.isnan(x), 0.0, x)
    cs = np.concatenate([[0.0], np.cumsum(x0)])
    cna = np.concatenate([[0.0], np.cumsum(na)])
    s = cs[k:] - cs[:n - k + 1]
    na_ct = cna[k:] - cna[:n - k + 1]
    s[na_ct > 0] = np.nan
    return s


def read_onset_params(spec):
    """
    Parse onset definition parameters from spec["options"]["onset_definition"].

    All parameters have defaults that reproduce the *new* consecutive-dry
    definition when onset_definition is omitted entirely. Set
    dry_spell.mode = "window_sum" to use the original definition.

    Returns an OnsetParams namedtuple consumed by find_onset /
    find_onset_precomp / calc_onsets_rowwise.

    yml layout (all fields optional; defaults shown):
    ------------------------------------------------
    options:
      window: 5                       # trigger rolling-window length (days)
      onset_definition:
        wet_day_min_mm: 1.0           # rain >= this => wet day
        follow_days: 21               # days after trigger window to check for dry spell
        dry_spell:
          mode: "consecutive_dry"     # "consecutive_dry" | "window_sum"

          # --- consecutive_dry params ---
          min_dry_days: 7             # run of >= this many dry days = dry spell
          dry_day_min_mm: 1.0         # rain < this => dry day (defaults to wet_day_min_mm)

          # --- window_sum params ---
          sum_window: 10              # rolling window size for dry-spell check
          sum_min_mm: 5               # window sum below this => dry spell
    """
    import collections
    OnsetParams = collections.namedtuple("OnsetParams", [
        "win",           # trigger window (days)
        "wet_day_min_mm",
        "follow_days",
        "mode",          # "consecutive_dry" | "window_sum"
        # consecutive_dry
        "min_dry_days",
        "dry_day_min_mm",
        # window_sum
        "sum_window",
        "sum_min_mm",
    ])

    opts = spec.get("options", {})
    win = int(opts.get("window", 5))

    od = opts.get("onset_definition") or {}
    wet_day_min_mm = float(od.get("wet_day_min_mm", 1.0))
    follow_days    = int(od.get("follow_days", 21))

    ds = od.get("dry_spell") or {}
    mode           = str(ds.get("mode", "consecutive_dry"))
    min_dry_days   = int(ds.get("min_dry_days", 7))
    dry_day_min_mm = float(ds.get("dry_day_min_mm", wet_day_min_mm))
    sum_window     = int(ds.get("sum_window", 10))
    sum_min_mm     = float(ds.get("sum_min_mm", 5.0))

    if mode not in ("consecutive_dry", "window_sum"):
        raise ValueError(
            f"onset_definition.dry_spell.mode must be 'consecutive_dry' or "
            f"'window_sum', got '{mode}'"
        )

    return OnsetParams(
        win=win,
        wet_day_min_mm=wet_day_min_mm,
        follow_days=follow_days,
        mode=mode,
        min_dry_days=min_dry_days,
        dry_day_min_mm=dry_day_min_mm,
        sum_window=sum_window,
        sum_min_mm=sum_min_mm,
    )


def _precompute_onset(series, params):
    """
    Precompute trigger and veto arrays for a single rainfall series.

    Trigger
    -------
    Left-aligned rolling sum of length params.win (NA treated as 0).
    Length: max(0, n - win + 1).

    Veto arrays (one of two modes, controlled by params.mode)
    ----------------------------------------------------------
    "consecutive_dry"
        pre_dry : int ndarray, length n+1
            Prefix count of positions where a consecutive dry run first reaches
            params.min_dry_days.  O(1) range query: dry spell of >= min_dry_days
            exists starting in 0-based [a, b] iff pre_dry[b+1] - pre_dry[a] > 0.

    "window_sum"
        pre_bad : int ndarray
            Prefix count of 'bad' rolling windows (sum < sum_min_mm, NA
            propagating).  Matches the original _apply_followup logic.
        last_win_start : int
            0-based index of the last position where a full sum_window fits.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    win = params.win

    # --- trigger sums ---
    wsum = roll_sum_na_rm_left(series, win)   # length max(0, n - win + 1)

    # --- veto precomputation ---
    if params.mode == "consecutive_dry":
        dry = (~np.isnan(series)) & (series < params.dry_day_min_mm)
        dry_starts = np.zeros(n, dtype=int)
        k = params.min_dry_days
        if n >= k:
            run = np.zeros(n, dtype=int)
            run[0] = int(dry[0])
            for i in range(1, n):
                run[i] = (run[i - 1] + 1) * int(dry[i])
            # Mark the start of each run the first time it reaches length k
            for e in np.where(run == k)[0]:
                dry_starts[e - k + 1] = 1
        pre_dry = np.concatenate([[0], np.cumsum(dry_starts)])
        return wsum, pre_dry, None

    else:  # window_sum
        sw = params.sum_window
        if n >= sw:
            sum_w = roll_sum_na_propagate_left(series, sw)
            bad = (~np.isnan(sum_w)) & (sum_w < params.sum_min_mm)
            pre_bad = np.concatenate([[0], np.cumsum(bad.astype(int))])
            last_win_start = n - sw   # 0-based index of last valid window start
        else:
            pre_bad = np.array([0], dtype=int)
            last_win_start = 0
        return wsum, pre_bad, last_win_start


def precompute_onset(series, params):
    """Precompute onset trigger and veto arrays for repeated onset searches."""
    return _precompute_onset(series, params)


# ---------------------------------------------------------------------------
# Unified onset finder
# ---------------------------------------------------------------------------

def find_onset(series, win=None, thresh=None, reject_if_short_followup=False,
               start_day=0, params=None):
    """
    Find the first rainy-season onset day in a rainfall series.

    Parameters
    ----------
    series : array-like
        Daily rainfall values (index 0 = day 1, 1-based return value).
    win : int
        Trigger rolling-window length (days). Overridden by params.win if
        params is supplied.
    thresh : float
        Trigger accumulation threshold (mm) — per-cell value from
        thresholds_df.csv.
    reject_if_short_followup : bool
        Reject candidates where the follow-up window extends past the end of
        the series.
    start_day : float
        Skip candidates before this 1-based index.
    params : OnsetParams (from read_onset_params)
        Full definition parameters. If None, a default consecutive_dry params
        object is constructed using win (default 5).

    Returns
    -------
    int or None  (1-based onset day index, or None)
    """
    if params is None:
        import collections
        OnsetParams = collections.namedtuple("OnsetParams", [
            "win", "wet_day_min_mm", "follow_days", "mode",
            "min_dry_days", "dry_day_min_mm", "sum_window", "sum_min_mm",
        ])
        params = OnsetParams(
            win=int(win) if win is not None else 5,
            wet_day_min_mm=1.0, follow_days=21,
            mode="consecutive_dry", min_dry_days=7, dry_day_min_mm=1.0,
            sum_window=10, sum_min_mm=5.0,
        )

    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < params.win or thresh is None or np.isnan(thresh):
        return None

    wsum, aux1, aux2 = _precompute_onset(series, params)
    return _find_onset_core(series, n, wsum, aux1, aux2, thresh,
                            params, start_day, reject_if_short_followup)


def find_onset_precomp(series, win, thresh, wsum_all, pre_bad, last10start,
                       reject_if_short_followup=False, start_day=0, params=None):
    """
    Onset finder compatible with batch call sites in nc_utils.py.

    When params is supplied (passed from calc_onsets_rowwise), all definition
    parameters come from the yml via read_onset_params().  The legacy
    positional arguments wsum_all / pre_bad / last10start are ignored and
    recomputed internally from params; they are retained only so existing
    call sites need no signature changes.

    When params is None (e.g. direct calls in tests), the function falls back
    to the same defaults as find_onset().

    Parameters
    ----------
    series, win, thresh : as in find_onset
    wsum_all, pre_bad, last10start : ignored (legacy compatibility)
    reject_if_short_followup, start_day : as in find_onset
    params : OnsetParams or None
    """
    if params is None:
        import collections
        OnsetParams = collections.namedtuple("OnsetParams", [
            "win", "wet_day_min_mm", "follow_days", "mode",
            "min_dry_days", "dry_day_min_mm", "sum_window", "sum_min_mm",
        ])
        params = OnsetParams(
            win=int(win) if win is not None else 5,
            wet_day_min_mm=1.0, follow_days=21,
            mode="consecutive_dry", min_dry_days=7, dry_day_min_mm=1.0,
            sum_window=10, sum_min_mm=5.0,
        )

    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < params.win or thresh is None or np.isnan(thresh):
        return None

    wsum, aux1, aux2 = _precompute_onset(series, params)
    return _find_onset_core(series, n, wsum, aux1, aux2, thresh,
                            params, start_day, reject_if_short_followup)


def find_onset_from_precomputed(series, thresh, wsum, aux1, aux2, params,
                                reject_if_short_followup=False, start_day=0):
    """
    Find onset using arrays returned by _precompute_onset().

    This is intended for call sites that need to evaluate the same rainfall
    series under several start-day restrictions.
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < params.win or thresh is None or np.isnan(thresh):
        return None
    return _find_onset_core(series, n, wsum, aux1, aux2, thresh,
                            params, start_day, reject_if_short_followup)


def _find_onset_core(series, n, wsum, aux1, aux2, thresh,
                     params, start_day, reject_if_short_followup):
    """
    Internal vectorised onset search given precomputed arrays.

    Trigger (both modes)
    --------------------
    - All days in the trigger window [d, d+win-1] are wet
      (rain >= wet_day_min_mm).
    - Rolling sum over that window > thresh.

    Veto — "consecutive_dry"
    ------------------------
    No run of >= min_dry_days consecutive dry days (rain < dry_day_min_mm)
    starts within follow_days days *after* the trigger window,
    i.e. 0-based positions [d-1+win, d-1+win+follow_days).
    aux1 = pre_dry  (prefix count of dry-run starts of length min_dry_days)
    aux2 = None

    Veto — "window_sum"
    -------------------
    No sum_window-day rolling sum < sum_min_mm within follow_days days after
    the trigger window. Mirrors the original _apply_followup logic.
    aux1 = pre_bad  (prefix count of bad rolling windows)
    aux2 = last_win_start (last valid 0-based window start)
    """
    win        = params.win
    wmm        = params.wet_day_min_mm
    follow     = params.follow_days

    # Candidates: days 1 .. n-win+1  (need full trigger window to fit)
    max_candidate = len(wsum)    # = n - win + 1
    if max_candidate < 1:
        return None

    min_i = max(1, int(np.ceil(start_day)))
    if min_i > max_candidate:
        return None

    idx = np.arange(min_i, max_candidate + 1)   # 1-based

    # --- trigger: all days in [d, d+win-1] wet AND rolling sum > thresh ---
    # Build a (len(idx), win) boolean matrix; all columns must be True
    wet_mat = np.stack(
        [series[idx - 1 + k] >= wmm for k in range(win)],
        axis=1
    )
    all_wet = wet_mat.all(axis=1)
    acc_ok  = wsum[idx - 1] > thresh
    base_ok = all_wet & acc_ok
    base_ok = np.where(np.isnan(series[idx - 1]), False, base_ok)

    cand = idx[base_ok]
    if len(cand) == 0:
        return None

    # --- reject_if_short_followup: need d + win - 1 + follow <= n ---
    full_end = cand + win - 1 + follow   # last day of follow-up (1-based)
    if reject_if_short_followup:
        cand = cand[full_end <= n]
        if len(cand) == 0:
            return None
        full_end = full_end[full_end <= n]  # keep in sync after filter
    else:
        full_end = np.minimum(n, full_end)

    # --- veto ---
    if params.mode == "consecutive_dry":
        pre_dry = aux1
        # Follow-up window starts right after the trigger: 0-based [d-1+win, ...]
        lower = cand - 1 + win                       # 0-based start of follow-up
        upper = np.minimum(n, cand - 1 + win + follow)  # pre_dry exclusive upper
        has_dry_spell = np.where(
            lower < upper,
            (pre_dry[upper] - pre_dry[lower]) > 0,
            False
        )

    else:  # window_sum
        pre_bad        = aux1
        last_win_start = aux2   # 0-based index of last valid sum_window start
        # Follow-up starts right after the trigger window (0-based: cand-1+win)
        # and ends at full_end (already clamped to n). We check rolling windows
        # of length sum_window whose start falls in that range.
        sw = params.sum_window
        c_lower = cand - 1 + win                              # 0-based start of follow-up
        c_upper = np.minimum(last_win_start + 1,              # pre_bad is length last_win_start+2
                             np.maximum(0, full_end - sw + 1))  # exclusive upper for pre_bad query
        # Clamp both to valid pre_bad indices [0, len(pre_bad)-1]
        max_idx = len(pre_bad) - 1
        c_lower = np.minimum(c_lower, max_idx)
        c_upper = np.minimum(c_upper, max_idx)
        has_dry_spell = np.where(
            c_lower < c_upper,
            (pre_bad[c_upper] - pre_bad[c_lower]) > 0,
            False
        )

    ok = ~has_dry_spell
    return int(cand[np.where(ok)[0][0]]) if np.any(ok) else None
