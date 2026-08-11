# ==============================================================================
# File: connect_utils.py
# ==============================================================================
# Purpose
#   Helper functions for 0_connect_prepare_data_to_2025_pipeline.py.
#   Provides day-to-week aggregation, logit-winsorization, rolling-window
#   rain summaries, and the main make_cv_rds_from_daylevel() converter.
#
# Function index
#   winsor_weekp(p, lo, hi)
#   logit_winsor(p, lo, hi)
#   sum_week_probs_from_dayprefix(df, day_prefix, out_prefix, ...)
#   sum_week_probs(df, prefix, ...)
#   sum_week_probs_with_day0(df, prefix, ...)
#   make_clim_logits_from_prefix(raw, input_prefix, output_tag, ...)
#   make_rain_transform(spec_value)
#   roll_sums_mat(mat, k)
#   week_max_over_starts(roll_mat, week_start_days)
#   week_min_over_starts(roll_mat, week_start_days)
#   make_cv_rds_from_daylevel(spec)
# ==============================================================================

import os
import pickle
import numpy as np
import pandas as pd
from scipy.special import logit, expit

from ..pipelines._shared.misc import coalesce, assign_lead_bin
from ..pipelines._shared.read_spec import load_spec
from ..prepare_data.onset_utils import read_onset_params
from ..prepare_data.spatial_id_utils import ensure_spatial_id_col
from ..rain_horizon_utils import (
    resolve_rain_day_max,
    validate_rain_horizon_frame,
    valid_week_start_days,
)


def resolve_onset_window_map(spec):
    """
    Build a map of symbolic rain-predictor window tokens -> integer day counts,
    derived from the onset definition so predictors track it automatically.

    Source, in priority order:
      1. spec['onset_spec']  -> loads specs/raw_data/<id>.yml (single source of
         truth; the same definition drives onset detection).
      2. spec itself, if it carries options.window / options.onset_definition inline.
    Returns {} if no onset source is configured (symbolic tokens then raise a
    helpful error in _resolve_window).

    Tokens: trigger/window -> trigger window; dry_spell/sum_window -> dry-spell
    window; follow -> follow_days; min_dry -> min_dry_days.
    """
    onset_src = None
    onset_spec_id = spec.get("onset_spec")
    if onset_spec_id:
        onset_src = load_spec(onset_spec_id, "raw_data")
    else:
        opts = spec.get("options") or {}
        if "window" in opts or opts.get("onset_definition") is not None:
            onset_src = spec
    if onset_src is None:
        return {}
    p = read_onset_params(onset_src)
    return {
        "trigger":    p.win,
        "window":     p.win,
        "dry_spell":  p.sum_window,
        "sum_window": p.sum_window,
        "follow":     p.follow_days,
        "min_dry":    p.min_dry_days,
    }


def _resolve_window(w, window_map):
    """Resolve a rain-predictor window to an int: explicit int / numeric string
    passes through; a symbolic token is looked up in window_map (from the onset
    definition)."""
    if isinstance(w, bool):
        raise ValueError(f"rain_predictor window must be an int or token, got bool {w!r}")
    if isinstance(w, (int, np.integer)):
        return int(w)
    s = str(w).strip()
    if s.isdigit():
        return int(s)
    tok = s.lower()
    if tok in window_map:
        return int(window_map[tok])
    raise ValueError(
        f"rain_predictor window '{w}' is not an integer and not a known onset token "
        f"{sorted(window_map) if window_map else '(none available)'}. Set the connect "
        f"spec's 'onset_spec' to the raw_data spec id, or use an explicit integer window."
    )


def winsor_weekp(p, lo=0.0001, hi=0.9999):
    """Winsorize probability to [lo, hi]."""
    return np.clip(p, lo, hi)


def logit_winsor(p, lo=0.0001, hi=0.9999):
    """Winsorize then apply logit transform."""
    return logit(winsor_weekp(np.asarray(p, dtype=float), lo, hi))


def sum_week_probs_from_dayprefix(df, day_prefix, out_prefix,
                                  day_max=28, days_per_bin=7, n_bins=4):
    """
    Sum daily columns <day_prefix>1..<day_prefix>N into weekly bins.
    Returns DataFrame with week columns named <out_prefix>_week1..N.
    """
    # day_0 is a separate "earlier" bin, not part of week 1.  This mirrors
    # the R connector, which always aggregates forecast days 1..day_max.
    cols = [f"{day_prefix}{k}" for k in range(1, day_max + 1)]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns for {out_prefix}: {', '.join(miss)}")
    mat = df[cols].values

    result = {}
    for w in range(1, n_bins + 1):
        lo = (w - 1) * days_per_bin
        hi = w * days_per_bin
        result[f"{out_prefix}_week{w}"] = mat[:, lo:hi].sum(axis=1)
    return pd.DataFrame(result, index=df.index)


def sum_week_probs(df, prefix, day_max=28, days_per_bin=7, n_bins=4):
    """Sum <prefix>_p_onset_day_1..<day_max> into weekly bins.

    A day_0 column, when present for unconditional climatology, represents the
    separate "earlier" category and must not be folded into week 1.
    """
    cols = [f"{prefix}_p_onset_day_{k}" for k in range(1, day_max + 1)]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns for {prefix}: {', '.join(miss)}")
    mat = df[cols].values

    result = {}
    for w in range(1, n_bins + 1):
        lo = (w - 1) * days_per_bin
        hi = w * days_per_bin
        result[f"{prefix}_p_onset_week{w}"] = mat[:, lo:hi].sum(axis=1)
    return pd.DataFrame(result, index=df.index)


def sum_week_probs_with_day0(df, prefix, day_max=28, days_per_bin=7, n_bins=4):
    """Like sum_week_probs but also extracts day_0 column for 'earlier' bin."""
    col0 = f"{prefix}_p_onset_day_0"
    if col0 not in df.columns:
        raise ValueError(f"Missing required column: {col0}")
    week_tbl = sum_week_probs(df, prefix, day_max=day_max, days_per_bin=days_per_bin, n_bins=n_bins)
    return {"day0": df[col0].values.copy(), "week": week_tbl}


def make_clim_logits_from_prefix(raw, input_prefix, output_tag,
                                  day_max, days_per_bin, n_bins,
                                  output_base_prefix="prob_clim"):
    """Aggregate climatology day probs to weeks and apply logit_winsor."""
    wk = sum_week_probs(raw, prefix=input_prefix, day_max=day_max,
                        days_per_bin=days_per_bin, n_bins=n_bins)
    result = {}
    for w in range(1, n_bins + 1):
        col = f"{input_prefix}_p_onset_week{w}"
        result[f"{output_base_prefix}_{output_tag}_week{w}"] = logit_winsor(wk[col].values)
    return pd.DataFrame(result, index=raw.index)


def make_rain_transform(spec_value):
    """
    Build a vectorized, non-negative rainfall transform from a spec value.

    The returned function is applied ONLY when constructing rain-based blend
    features (the diff/max/min predictors). It does NOT affect onset detection
    or climatology, which operate on untransformed rainfall upstream.

    Accepted spec values
    ---------------------
      - None / "identity" / "none" / ""   -> f(x) = x                (default)
      - "sqrt"                            -> f(x) = x ** 0.5
      - "fourth_root"                     -> f(x) = x ** 0.25
      - "log1p"                           -> f(x) = log(1 + x)
      - "power:<p>"                       -> f(x) = x ** p
      - {"power": <p>}                    -> f(x) = x ** p
      - a bare number <p>                 -> f(x) = x ** p

    Power and log transforms clip inputs at 0 first. Rainfall accumulations and
    onset thresholds are non-negative, so this only guards against tiny negative
    round-off (which would otherwise produce NaN under a fractional power).
    NaNs propagate unchanged so missing-week sentinels are preserved.
    """
    def _power(p):
        p = float(p)
        def f(x):
            x = np.asarray(x, dtype=float)
            return np.where(np.isnan(x), x, np.clip(x, 0.0, None) ** p)
        return f

    def _identity(x):
        return np.asarray(x, dtype=float)

    if spec_value is None:
        return _identity

    if isinstance(spec_value, dict):
        if "power" in spec_value:
            return _power(spec_value["power"])
        raise ValueError(f"Unknown rain_transform dict {spec_value!r} (expected key 'power').")

    if isinstance(spec_value, (int, float)) and not isinstance(spec_value, bool):
        return _power(spec_value)

    name = str(spec_value).strip().lower()
    if name in ("identity", "none", ""):
        return _identity
    if name == "sqrt":
        return _power(0.5)
    if name == "fourth_root":
        return _power(0.25)
    if name == "log1p":
        def f_log1p(x):
            x = np.asarray(x, dtype=float)
            return np.where(np.isnan(x), x, np.log1p(np.clip(x, 0.0, None)))
        return f_log1p
    if name.startswith("power:"):
        return _power(name.split(":", 1)[1])
    raise ValueError(
        f"Unknown rain_transform '{spec_value}'. Expected one of: identity, sqrt, "
        f"fourth_root, log1p, 'power:<p>', {{power: <p>}}, or a number."
    )


def roll_sums_mat(mat, k):
    """
    Rolling k-day row sums from a matrix [nrow x ncols].
    Returns [nrow x (ncols - k + 1)].
    """
    n = mat.shape[1]
    if n < k:
        return np.empty((mat.shape[0], 0))
    out = np.stack(
        [mat[:, s:s + k].sum(axis=1) for s in range(n - k + 1)],
        axis=1
    )
    return out


def week_max_over_starts(roll_mat, week_start_days):
    """Per-row max of rolling sums at specified start days (1-based)."""
    if roll_mat.shape[1] == 0:
        return np.full(roll_mat.shape[0], np.nan)
    ok = [s for s in week_start_days if 1 <= s <= roll_mat.shape[1]]
    if not ok:
        return np.full(roll_mat.shape[0], np.nan)
    idx = [s - 1 for s in ok]
    # np.fmax/fmin implement R's na.rm=TRUE behaviour across candidate starts:
    # a partially unavailable horizon can still contribute a valid summary.
    return np.fmax.reduce(roll_mat[:, idx], axis=1)


def week_min_over_starts(roll_mat, week_start_days):
    """Per-row min of rolling sums at specified start days (1-based)."""
    if roll_mat.shape[1] == 0:
        return np.full(roll_mat.shape[0], np.nan)
    ok = [s for s in week_start_days if 1 <= s <= roll_mat.shape[1]]
    if not ok:
        return np.full(roll_mat.shape[0], np.nan)
    idx = [s - 1 for s in ok]
    return np.fmin.reduce(roll_mat[:, idx], axis=1)


def make_cv_rds_from_daylevel(spec):
    """
    Main converter: reads daily combined pickle, builds weekly bins, onset
    outcomes, climatology logits, rain-based predictors, and writes wide
    pickle for the 2025 blending pipeline.

    Parameters
    ----------
    spec : dict  parsed YAML spec with mode, input_rds, output_rds, etc.

    Returns
    -------
    DataFrame  (also saved to spec["output_rds"])
    """
    input_rds = spec["input_rds"]
    output_rds = spec["output_rds"]
    day_max = coalesce(spec.get("day_max"), 28)
    # Accept both the Python names and the maintained R spec names.
    days_per_bin = coalesce(spec.get("days_per_bin"),
                            coalesce(spec.get("days_per_week"), 7))
    n_bins = coalesce(spec.get("n_bins"), coalesce(spec.get("n_weeks"), 4))

    with open(input_rds, "rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, pd.DataFrame):
        raw = pd.DataFrame(raw)

    raw = ensure_spatial_id_col(
        raw, spec=spec, context="connector spatial IDs"
    )
    raw["time"] = pd.to_datetime(raw["time"]).dt.date
    raw["year"] = pd.to_datetime(raw["time"]).dt.year

    # The maintained R connector exposes numeric lat/lon downstream, deriving
    # them from the legacy "lat_lon" id when the combined table lacks columns.
    id_text = raw["id"].astype(str)
    id_parts = id_text.str.split("_", n=1, expand=True)
    parsed_lat = pd.to_numeric(id_parts[0], errors="coerce")
    parsed_lon = (pd.to_numeric(id_parts[1], errors="coerce")
                  if id_parts.shape[1] > 1
                  else pd.Series(np.nan, index=raw.index))
    is_grid_id = (
        (id_text.str.count("_") == 1)
        & parsed_lat.between(-90, 90)
        & parsed_lon.between(-180, 360)
    )
    parsed_lat = parsed_lat.where(is_grid_id)
    parsed_lon = parsed_lon.where(is_grid_id)
    if "lat" in raw.columns:
        raw["lat"] = pd.to_numeric(raw["lat"], errors="coerce").fillna(parsed_lat)
    else:
        raw["lat"] = parsed_lat
    if "lon" in raw.columns:
        raw["lon"] = pd.to_numeric(raw["lon"], errors="coerce").fillna(parsed_lon)
    else:
        raw["lon"] = parsed_lon

    # Onset threshold
    first_model = spec["forecast_models"][0]["name"]
    thresh_col = f"{first_model}_onset_thresh"
    if thresh_col not in raw.columns:
        raise ValueError(f"Missing {thresh_col} (onset threshold).")
    raw["onset_threshold"] = raw[thresh_col]

    # Outcome: bin true onset date relative to forecast init date
    if "true_onset_date" not in raw.columns:
        raise ValueError("Missing true_onset_date.")
    raw["true_onset_date"] = pd.to_datetime(raw["true_onset_date"]).dt.date
    raw["lead_day"] = (pd.to_datetime(raw["true_onset_date"]) - pd.to_datetime(raw["time"])).dt.days

    raw["outcome"] = raw["lead_day"].apply(
        lambda ld: assign_lead_bin(ld, days_per_bin, n_bins, allow_earlier=False)
    )

    # Climatology base prefix
    base_prefix = coalesce(spec.get("climatology", {}).get("base_prefix"), "clim")
    unc_prefix = coalesce(spec.get("climatology", {}).get("unconditional_prefix"), "clim_unc")
    clim_output_prefix = coalesce(
        spec.get("climatology", {}).get("output_prefix"), "prob_clim"
    )

    clim_week_probs = sum_week_probs(raw, base_prefix, day_max=day_max,
                                     days_per_bin=days_per_bin, n_bins=n_bins)

    # Climatology window variants
    window_tags = spec.get("climatology", {}).get("window_tags") or []
    if window_tags:
        variant_parts = []
        for tag in window_tags:
            pref = f"{base_prefix}_{tag}"
            variant_parts.append(
                make_clim_logits_from_prefix(raw, pref, tag,
                                              day_max=day_max, days_per_bin=days_per_bin,
                                              n_bins=n_bins,
                                              output_base_prefix=clim_output_prefix)
            )
        clim_variant_logits = pd.concat(variant_parts, axis=1)
    else:
        clim_variant_logits = pd.DataFrame(index=raw.index)

    # Forecast model week probabilities
    model_week_cols_list = {}
    for fm in spec["forecast_models"]:
        model_name = fm["name"]
        model_week_cols_list[model_name] = sum_week_probs(
            raw, model_name, day_max=day_max, days_per_bin=days_per_bin, n_bins=n_bins
        )
        for variant in (fm.get("variants") or []):
            variant_key = f"{model_name}_{variant}"
            model_week_cols_list[variant_key] = sum_week_probs_from_dayprefix(
                raw,
                day_prefix=f"{model_name}_p_onset_{variant}_day_",
                out_prefix=f"{model_name}_p_onset_{variant}",
                day_max=day_max,
                days_per_bin=days_per_bin,
                n_bins=n_bins,
            )
    model_week_cols = pd.concat(model_week_cols_list.values(), axis=1)

    # Unconditional climatology (has day_0 -> "earlier")
    unc = sum_week_probs_with_day0(raw, unc_prefix, day_max=day_max,
                                    days_per_bin=days_per_bin, n_bins=n_bins)
    unc_day0 = unc["day0"]
    unc_week_probs = unc["week"]

    # Build logit features (one per configured week bin, + unconditional 'earlier')
    clim_feats = {f"{clim_output_prefix}_unc_earlier": logit_winsor(unc_day0)}
    for w in range(1, n_bins + 1):
        clim_feats[f"{clim_output_prefix}_week{w}"] = logit_winsor(
            clim_week_probs[f"{base_prefix}_p_onset_week{w}"].values)
        clim_feats[f"{clim_output_prefix}_unc_week{w}"] = logit_winsor(
            unc_week_probs[f"{unc_prefix}_p_onset_week{w}"].values)
    clim_logits = pd.DataFrame(clim_feats, index=raw.index)

    # Rain-based predictors
    week_start_days_list = [
        list(range((w - 1) * days_per_bin + 1, w * days_per_bin + 1))
        for w in range(1, n_bins + 1)
    ]
    rain_predictors_dict = {}
    rain_horizon_metadata = {}
    unavailable_rain_predictors = {}

    # Symbolic window tokens (e.g. window: trigger / dry_spell) resolve from the
    # onset definition so predictors track it. Explicit ints always win.
    onset_window_map = resolve_onset_window_map(spec)

    for fm in spec["forecast_models"]:
        model_name = fm["name"]
        rain_preds = fm.get("rain_predictors") or []
        if not rain_preds:
            continue

        # Optional transform applied to rain features before they enter the
        # blend. Per-model `rain_transform` wins; otherwise fall back to a
        # spec-level default; otherwise identity. Does not affect onset or
        # climatology (computed upstream from untransformed rainfall).
        rain_tf = make_rain_transform(
            coalesce(fm.get("rain_transform"),
                     coalesce(spec.get("rain_transform"), "identity"))
        )

        rain_day_max, horizon_policy = resolve_rain_day_max(fm, day_max)
        strict_horizon = horizon_policy != "legacy"
        need_rain = validate_rain_horizon_frame(
            raw,
            day_prefix=f"{model_name}_rain_mean_day_",
            rain_day_max=rain_day_max,
            model_name=model_name,
            strict=strict_horizon,
            allow_extra=horizon_policy == "truncate",
            key_columns=("id", "time", "year"),
            context="connect input",
        )
        rain_mat = (
            raw[need_rain]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )
        rain_horizon_metadata[model_name] = {
            "rain_day_max": rain_day_max,
            "strict": strict_horizon,
            "policy": horizon_policy,
        }

        # Parse rain_predictors. Supports:
        #   - dicts:   { agg: diff, window: 5 }  or  { agg: min, window: dry_spell }
        #   - legacy strings: "diff_5day", "min_10day", "max_5day"
        # window may be an int, a numeric string, or a symbolic onset token
        # (trigger / dry_spell / follow / min_dry) resolved from the onset definition.
        def _parse_pred(p):
            if isinstance(p, dict):
                return str(p["agg"]).lower(), _resolve_window(p["window"], onset_window_map)
            parts = str(p).split("_")
            agg = parts[0]
            tok = parts[-1]
            digits = "".join(filter(str.isdigit, tok))
            window = int(digits) if digits else _resolve_window(tok, onset_window_map)
            return agg, window

        parsed_preds = [_parse_pred(p) for p in rain_preds]

        # Pre-compute only the distinct rolling windows actually needed
        needed_windows = set(window for _, window in parsed_preds)
        roll_cache = {w: roll_sums_mat(rain_mat, w) for w in needed_windows}

        # Pre-compute per-window, per-week aggregations (cached to avoid recomputation
        # when multiple predictors share the same window)
        agg_cache = {}  # (agg, window, week_index) -> array
        for agg, window in parsed_preds:
            roll_mat = roll_cache[window]
            for wi, sd in enumerate(week_start_days_list):
                key = (agg, window, wi)
                if key not in agg_cache:
                    valid_starts = valid_week_start_days(
                        rain_day_max, window, sd
                    )
                    if agg in ("diff", "max"):
                        agg_cache[key] = week_max_over_starts(
                            roll_mat, valid_starts
                        )
                    elif agg == "min":
                        agg_cache[key] = week_min_over_starts(
                            roll_mat, valid_starts
                        )
                    else:
                        raise ValueError(f"Unknown rain predictor agg '{agg}' in model '{model_name}'")
                    if not valid_starts:
                        week = wi + 1
                        col_name = (
                            f"diff_{model_name}_week{week}"
                            if agg == "diff"
                            else f"{agg}_{model_name}_{window}day_week{week}"
                        )
                        unavailable_rain_predictors[col_name] = {
                            "model": model_name,
                            "rain_day_max": rain_day_max,
                            "window": window,
                            "week": week,
                        }

        for agg, window in parsed_preds:
            for w in range(1, n_bins + 1):
                wi = w - 1
                agg_vals = agg_cache[(agg, window, wi)]
                if agg == "diff":
                    col_name = f"diff_{model_name}_week{w}"
                    rain_predictors_dict[col_name] = (
                        rain_tf(agg_vals) - rain_tf(raw["onset_threshold"].values)
                    )
                else:
                    col_name = f"{agg}_{model_name}_{window}day_week{w}"
                    rain_predictors_dict[col_name] = rain_tf(agg_vals)

    rain_predictors = pd.DataFrame(rain_predictors_dict, index=raw.index)

    base_cols = ["id", "time", "year", "lat", "lon", "onset_threshold", "outcome"]
    wide_df = pd.concat(
        [raw[base_cols].reset_index(drop=True),
         clim_logits.reset_index(drop=True),
         clim_variant_logits.reset_index(drop=True),
         model_week_cols.reset_index(drop=True),
         rain_predictors.reset_index(drop=True)],
        axis=1
    )
    wide_df["outcome"] = wide_df["outcome"].astype(str).where(wide_df["outcome"].notna(), None)
    wide_df.attrs.update(raw.attrs)
    wide_df.attrs["rain_horizons"] = rain_horizon_metadata
    wide_df.attrs["unavailable_rain_predictors"] = unavailable_rain_predictors

    out_dir = os.path.dirname(output_rds)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_rds, "wb") as f:
        pickle.dump(wide_df, f)

    return wide_df
