# ==============================================================================
# File: misc.py
# ==============================================================================
# Purpose
#   Small utility functions used across multiple pipeline stages.
#
# Function index
#   coalesce(x, y)
#     Null-coalescing helper. Returns x if not None, else y.
#
#   softmax_row(x)
#     Numerically stable softmax for a single numeric array.
#
#   row_logsumexp(mat_NK)
#     Row-wise log-sum-exp for an N x K matrix.
#
#   inv_logit(x)
#     Numerically stable inverse logit (sigmoid).
#
#   Bin-structure helpers (single source of truth for the forecast bin layout,
#   controlled by n_bins and days_per_bin):
#     week_labels(n_bins)          -> ["week1", ..., "weekN"]
#     interval_bins(n_bins)        -> week labels + ["later"]  (multinomial outcome)
#     rps_bins(n_bins)             -> ["earlier"] + week labels + ["later"]
#     assign_lead_bin(lead_day, days_per_bin, n_bins, allow_earlier=False)
# ==============================================================================

import math
import numpy as np


def week_labels(n_bins):
    """Bin labels week1..weekN for the configured number of bins."""
    return [f"week{w}" for w in range(1, int(n_bins) + 1)]


def interval_bins(n_bins):
    """Multinomial outcome bins: week1..weekN + 'later' (N+1 bins)."""
    return week_labels(n_bins) + ["later"]


def rps_bins(n_bins):
    """RPS bins: 'earlier' + week1..weekN + 'later' (N+2 bins)."""
    return ["earlier"] + week_labels(n_bins) + ["later"]


def assign_lead_bin(lead_day, days_per_bin, n_bins, allow_earlier=False):
    """
    Map a lead time (onset date minus reference date, in days) to a forecast bin,
    using the configured bin width (days_per_bin) and count (n_bins):
      - lead_day <= 0  -> "earlier" if allow_earlier else None
      - (W-1)*dpw < lead_day <= W*dpw  -> "weekW"  for W in 1..n_bins
      - beyond n_bins*dpw            -> "later"
      - NaN/None                      -> None
    """
    if lead_day is None:
        return None
    try:
        if math.isnan(float(lead_day)):
            return None
    except (TypeError, ValueError):
        return None
    dpw, n = int(days_per_bin), int(n_bins)
    if lead_day <= 0:
        return "earlier" if allow_earlier else None
    for w in range(1, n + 1):
        if lead_day <= w * dpw:
            return f"week{w}"
    return "later"


def coalesce(x, y):
    """Return x if not None, else y. Equivalent to R's %||% operator."""
    return x if x is not None else y


def softmax_row(x):
    """Numerically stable softmax for a 1D numeric array."""
    x = np.asarray(x, dtype=float)
    z = x - np.nanmax(x)
    ez = np.exp(z)
    return ez / np.sum(ez)


def row_logsumexp(mat_NK):
    """Row-wise log-sum-exp for an N x K matrix."""
    mat = np.asarray(mat_NK, dtype=float)
    m = mat.max(axis=1)
    return m + np.log(np.sum(np.exp(mat - m[:, None]), axis=1))


def inv_logit(x):
    """Numerically stable inverse logit (sigmoid)."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (np.exp(x) + 1.0))
