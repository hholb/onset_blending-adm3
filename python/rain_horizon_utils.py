"""Shared validation for fixed per-model rainfall forecast horizons."""

import re

import numpy as np
import pandas as pd


def _positive_int(value, field_name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a positive integer, got {value!r}."
        ) from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}.")
    return int(numeric)


def resolve_rain_day_max(model_spec, probability_day_max):
    """
    Resolve a model's rainfall horizon and validation policy.

    An omitted rain_day_max preserves the legacy day_max + 10 minimum. An
    explicit value activates exact fixed-horizon validation unless the spec
    explicitly requests ``rain_horizon_policy: truncate``.
    """
    explicit = model_spec.get("rain_day_max") is not None
    value = (
        model_spec["rain_day_max"]
        if explicit
        else _positive_int(probability_day_max, "day_max") + 10
    )
    policy = model_spec.get("rain_horizon_policy")
    if policy is None:
        policy = "exact" if explicit else "legacy"
    policy = str(policy).strip().lower()
    if policy not in {"legacy", "exact", "truncate"}:
        raise ValueError(
            "rain_horizon_policy must be one of: legacy, exact, truncate."
        )
    if explicit and policy == "legacy":
        raise ValueError(
            "An explicit rain_day_max cannot use rain_horizon_policy: legacy; "
            "use truncate to allow later columns while validating days 1..N."
        )
    if not explicit and policy != "legacy":
        raise ValueError(
            f"rain_horizon_policy: {policy} requires an explicit rain_day_max."
        )
    return _positive_int(value, "rain_day_max"), policy


def _parse_day_value(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or not numeric.is_integer():
        return None
    return int(numeric)


def validate_day_coordinate(
    day_values, rain_day_max, context, *, allow_extra=False
):
    """Validate a NetCDF lead coordinate against the integers 1..N."""
    rain_day_max = _positive_int(rain_day_max, "rain_day_max")
    parsed = [_parse_day_value(value) for value in day_values]
    invalid = [repr(value) for value, day in zip(day_values, parsed) if day is None]
    valid = [day for day in parsed if day is not None]
    duplicates = sorted({day for day in valid if valid.count(day) > 1})
    actual = set(valid)
    required = set(range(1, rain_day_max + 1))
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    nonpositive = [day for day in extra if day < 1]
    later = [day for day in extra if day > rain_day_max]

    problems = []
    if invalid:
        problems.append(f"non-integer day values={invalid[:10]}")
    if duplicates:
        problems.append(f"duplicate days={duplicates[:10]}")
    if missing:
        problems.append(f"missing days={missing[:10]}")
    if nonpositive:
        problems.append(f"nonpositive days={nonpositive[:10]}")
    if later and not allow_extra:
        problems.append(f"extra days={later[:10]}")
    if problems:
        requirement = (
            f"days 1..{rain_day_max}, with later days permitted"
            if allow_extra
            else f"exactly days 1..{rain_day_max}"
        )
        raise ValueError(
            f"{context}: strict rainfall horizon requires {requirement}; "
            + "; ".join(problems)
        )
    return valid


def _day_column_map(columns, day_prefix):
    pattern = re.compile(rf"^{re.escape(day_prefix)}(-?\d+)$")
    mapping = {}
    for column in list(columns):
        match = pattern.match(str(column))
        if match:
            mapping.setdefault(int(match.group(1)), []).append(column)
    return mapping


def validate_rain_horizon_frame(
    frame,
    *,
    day_prefix,
    rain_day_max,
    model_name,
    strict,
    allow_extra=False,
    key_columns=(),
    context="rainfall data",
    sample_size=5,
):
    """
    Validate rainfall day columns and, in strict mode, every required value.

    Legacy mode requires columns 1..N and permits later columns and missing
    values. Strict mode requires finite rainfall for every row and required
    day; exact mode rejects later columns, while truncate mode permits them.
    """
    rain_day_max = _positive_int(rain_day_max, "rain_day_max")
    mapping = _day_column_map(frame.columns, day_prefix)
    required_days = list(range(1, rain_day_max + 1))
    missing = [day for day in required_days if day not in mapping]
    duplicates = {
        day: names for day, names in mapping.items() if len(names) > 1
    }
    required_set = set(required_days)
    extra = sorted(day for day in mapping if day not in required_set)
    nonpositive = [day for day in extra if day < 1]
    later = [day for day in extra if day > rain_day_max]

    problems = []
    if missing:
        problems.append(f"missing day columns={missing[:10]}")
    if duplicates:
        duplicate_summary = {
            day: [str(name) for name in names]
            for day, names in list(duplicates.items())[:10]
        }
        problems.append(f"duplicate day columns={duplicate_summary}")
    if strict and nonpositive:
        problems.append(f"nonpositive day columns={nonpositive[:10]}")
    if strict and later and not allow_extra:
        problems.append(f"extra day columns={later[:10]}")
    if problems:
        mode = "truncate" if strict and allow_extra else (
            "strict" if strict else "legacy"
        )
        raise ValueError(
            f"{context}: rainfall horizon validation failed for model "
            f"'{model_name}' (rain_day_max={rain_day_max}, mode={mode}): "
            + "; ".join(problems)
        )

    required_columns = [mapping[day][0] for day in required_days]
    if not strict or frame.empty:
        return required_columns

    numeric = frame.loc[:, required_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    bad = ~np.isfinite(values)
    bad_rows = np.flatnonzero(bad.any(axis=1))
    if not len(bad_rows):
        return required_columns

    available_keys = [column for column in key_columns if column in frame.columns]
    examples = []
    for position in bad_rows[:sample_size]:
        row = frame.iloc[position]
        key_text = ", ".join(
            f"{column}={row[column]!r}" for column in available_keys
        )
        invalid_days = [
            required_days[index] for index in np.flatnonzero(bad[position])
        ]
        if len(invalid_days) > 12:
            invalid_text = f"{invalid_days[:12]}..."
        else:
            invalid_text = str(invalid_days)
        prefix = f"{key_text}, " if key_text else ""
        examples.append(f"({prefix}invalid_days={invalid_text})")

    raise ValueError(
        f"{context}: model '{model_name}' has {len(bad_rows)} of {len(frame)} "
        f"forecast rows with non-finite rainfall inside the fixed horizon "
        f"1..{rain_day_max}. Examples: {'; '.join(examples)}"
    )


def valid_week_start_days(rain_day_max, window, week_start_days):
    """Return week starts whose complete rolling window fits within 1..N."""
    rain_day_max = _positive_int(rain_day_max, "rain_day_max")
    window = _positive_int(window, "rain predictor window")
    last_start = rain_day_max - window + 1
    if last_start < 1:
        return []
    return [
        int(start)
        for start in week_start_days
        if 1 <= int(start) <= last_start
    ]
