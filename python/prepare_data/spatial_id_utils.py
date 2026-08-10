"""Shared construction and validation of spatial identifiers."""

from dataclasses import dataclass, replace
import re

import numpy as np
import pandas as pd


_GRID_ID_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))_"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))$"
)


@dataclass(frozen=True)
class GridIdConvention:
    decimal_digits: int
    number_format: str = "trimmed"
    source: str = "inferred"

    def __post_init__(self):
        if not 0 <= int(self.decimal_digits) <= 15:
            raise ValueError("grid_id_decimal_digits must be between 0 and 15.")
        if self.number_format not in ("trimmed", "fixed"):
            raise ValueError("grid_id_format must be 'trimmed' or 'fixed'.")

    def as_dict(self):
        return {
            "decimal_digits": int(self.decimal_digits),
            "number_format": self.number_format,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            decimal_digits=int(value["decimal_digits"]),
            number_format=str(value.get("number_format", "trimmed")),
            source=str(value.get("source", "inferred")),
        )


def _geometry_cfg(spec):
    if not spec:
        return {}
    if "geometry" in spec:
        return spec.get("geometry") or {}
    return spec


def explicit_grid_id_convention(spec):
    """Return the grid-ID convention declared in ``geometry``, if any."""
    cfg = _geometry_cfg(spec)
    digits = cfg.get("grid_id_decimal_digits")
    number_format = cfg.get("grid_id_format")
    if digits is None:
        if number_format is not None:
            raise ValueError(
                "geometry.grid_id_format requires geometry.grid_id_decimal_digits."
            )
        return None
    return GridIdConvention(
        decimal_digits=int(digits),
        number_format=str(number_format or "trimmed").lower(),
        source="spec",
    )


def parse_grid_id(value):
    """Parse a ``<lat>_<lon>`` ID, returning ``None`` for non-grid IDs."""
    if value is None or pd.isna(value):
        return None
    match = _GRID_ID_RE.fullmatch(str(value).strip())
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def format_coord_component(value, convention):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Grid coordinates must be finite.")

    digits = int(convention.decimal_digits)
    rounded = round(value, digits)
    if rounded == 0:
        rounded = 0.0
    text = f"{rounded:.{digits}f}"
    if convention.number_format == "trimmed" and "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "-0.0") else text


def format_grid_ids(lat, lon, convention):
    lat = np.asarray(lat).reshape(-1)
    lon = np.asarray(lon).reshape(-1)
    if len(lat) != len(lon):
        raise ValueError("Latitude and longitude arrays must have equal length.")
    return [
        f"{format_coord_component(la, convention)}_"
        f"{format_coord_component(lo, convention)}"
        for la, lo in zip(lat, lon)
    ]


def _minimum_positive_spacing(values):
    values = np.unique(np.asarray(values, dtype=float))
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return None
    spacing = np.diff(np.sort(values))
    spacing = spacing[spacing > 0]
    return float(spacing.min()) if len(spacing) else None


def infer_grid_id_convention(lat, lon, max_digits=12,
                             max_error_fraction=1e-4):
    """Infer the least precise collision-free trimmed grid-ID convention."""
    lat = np.asarray(lat, dtype=float).reshape(-1)
    lon = np.asarray(lon, dtype=float).reshape(-1)
    if len(lat) != len(lon) or not len(lat):
        raise ValueError("Cannot infer grid IDs from empty or unequal coordinates.")
    if not np.isfinite(lat).all() or not np.isfinite(lon).all():
        raise ValueError("Cannot infer grid IDs from non-finite coordinates.")

    pairs = np.unique(np.column_stack([lat, lon]), axis=0)
    spacing_values = [
        x for x in (
            _minimum_positive_spacing(pairs[:, 0]),
            _minimum_positive_spacing(pairs[:, 1]),
        ) if x is not None
    ]
    if spacing_values:
        error_tolerance = min(spacing_values) * float(max_error_fraction)
    else:
        # A single-cell input has no grid spacing. Permit only small
        # floating-point noise relative to the coordinate magnitude.
        scale = max(1.0, float(np.max(np.abs(pairs))))
        error_tolerance = scale * 1e-6

    for digits in range(int(max_digits) + 1):
        convention = GridIdConvention(digits, "trimmed", "inferred")
        ids = format_grid_ids(pairs[:, 0], pairs[:, 1], convention)
        if len(set(ids)) != len(pairs):
            continue
        rounded_lat = np.round(pairs[:, 0], digits)
        rounded_lon = np.round(pairs[:, 1], digits)
        max_error = max(
            float(np.max(np.abs(rounded_lat - pairs[:, 0]))),
            float(np.max(np.abs(rounded_lon - pairs[:, 1]))),
        )
        if max_error <= error_tolerance:
            return convention

    raise ValueError(
        "Could not infer a collision-free grid-ID precision within "
        f"0..{max_digits} decimal digits. Set geometry.grid_id_decimal_digits."
    )


def infer_grid_id_convention_from_ids(ids):
    """Infer a convention that reproduces established grid-ID strings exactly."""
    values = pd.Series(ids, dtype="string").dropna().str.strip().unique().tolist()
    if not values:
        return None
    parsed = [parse_grid_id(value) for value in values]
    if all(value is None for value in parsed):
        return None
    if any(value is None for value in parsed):
        return None

    lat = [value[0] for value in parsed]
    lon = [value[1] for value in parsed]
    max_text_digits = max(
        len(component.partition(".")[2])
        for value in values
        for component in value.split("_", 1)
    )
    if max_text_digits > 15:
        raise ValueError("Established grid IDs exceed 15 decimal digits.")
    for digits in range(16):
        for number_format in ("trimmed", "fixed"):
            convention = GridIdConvention(digits, number_format, "established")
            if format_grid_ids(lat, lon, convention) == values:
                return convention

    raise ValueError(
        "Established grid IDs do not follow one consistent decimal convention."
    )


def _convention_reproduces_ids(convention, ids):
    values = pd.Series(ids, dtype="string").dropna().str.strip().unique().tolist()
    parsed = [parse_grid_id(value) for value in values]
    if not values or any(value is None for value in parsed):
        return False
    return format_grid_ids(
        [value[0] for value in parsed],
        [value[1] for value in parsed],
        convention,
    ) == values


def resolve_grid_id_convention(spec=None, lat=None, lon=None,
                               authoritative_ids=None, context="grid"):
    """Resolve established -> explicit spec -> inferred grid-ID precedence."""
    explicit = explicit_grid_id_convention(spec)
    established = None
    if authoritative_ids is not None:
        established = infer_grid_id_convention_from_ids(authoritative_ids)

    if established is not None:
        if explicit is not None and not _convention_reproduces_ids(
            explicit, authoritative_ids
        ):
            raise ValueError(
                f"{context}: the explicit grid-ID convention conflicts with "
                "the established source IDs."
            )
        convention = replace(
            explicit or established,
            source="established",
        )
    elif explicit is not None:
        convention = explicit
    elif lat is not None and lon is not None:
        convention = infer_grid_id_convention(lat, lon)
    else:
        return None

    return convention


def normalize_id_series(values, context="spatial IDs"):
    """Normalize string IDs without coercing codes through numeric types."""
    values = pd.Series(values, copy=True)
    missing = values.isna()
    normalized = values.astype("string").str.strip()
    invalid = missing | normalized.isna() | normalized.eq("")
    if invalid.any():
        raise ValueError(f"{context} contain missing or empty values.")
    return normalized.astype(str)


def validate_id_coordinate_consistency(df, id_col="id", context="spatial IDs"):
    """Reject one ID assigned to multiple coordinate pairs; allow exact repeats."""
    if id_col not in df.columns or not {"lat", "lon"}.issubset(df.columns):
        return
    pairs = df[[id_col, "lat", "lon"]].copy()
    pairs["lat"] = pd.to_numeric(pairs["lat"], errors="raise")
    pairs["lon"] = pd.to_numeric(pairs["lon"], errors="raise")
    pairs = pairs.drop_duplicates()
    conflicts = pairs[id_col].duplicated(keep=False)
    if conflicts.any():
        sample = ", ".join(pairs.loc[conflicts, id_col].astype(str).unique()[:10])
        raise ValueError(
            f"{context}: an ID is assigned to multiple coordinate pairs: {sample}"
        )


def ensure_spatial_id_col(df, id_col="id", spec=None, convention=None,
                          force_latlon=False, context="spatial data"):
    """Return a copy with a normalized ID, deriving grid IDs when necessary."""
    out = df.copy()
    if id_col in out.columns and not force_latlon:
        out[id_col] = normalize_id_series(out[id_col], context=context)
        return out
    if not force_latlon and "adm3_name" in out.columns:
        out[id_col] = normalize_id_series(out["adm3_name"], context=context)
        return out
    if "lat" not in out.columns or "lon" not in out.columns:
        raise ValueError(
            f"Cannot create {id_col!r}: need {id_col!r}, 'adm3_name', or "
            "both 'lat' and 'lon'."
        )

    convention = convention or resolve_grid_id_convention(
        spec=spec,
        lat=out["lat"].values,
        lon=out["lon"].values,
        context=context,
    )
    print(
        f"  {context}: grid IDs use {convention.decimal_digits} decimal "
        f"digits, {convention.number_format} format ({convention.source})."
    )
    lat = pd.to_numeric(out["lat"], errors="raise").to_numpy(dtype=float)
    lon = pd.to_numeric(out["lon"], errors="raise").to_numpy(dtype=float)
    out[id_col] = format_grid_ids(lat, lon, convention)
    unique_coords = pd.DataFrame({"lat": lat, "lon": lon, id_col: out[id_col]})
    unique_coords = unique_coords.drop_duplicates()
    collisions = unique_coords.groupby(id_col).size()
    collisions = collisions[collisions > 1]
    if len(collisions):
        raise ValueError(
            f"{context}: {len(collisions)} grid IDs represent multiple "
            "coordinates after applying the selected decimal convention."
        )
    return out


def validate_expected_source_ids(df, expected_ids, convention=None,
                                 spec=None, context="cell transform"):
    """Build source IDs from coordinates and validate expected IDs one-to-one."""
    expected = normalize_id_series(expected_ids, context=f"{context} source IDs")
    convention = convention or resolve_grid_id_convention(
        spec=spec,
        lat=df["lat"].values if "lat" in df.columns else None,
        lon=df["lon"].values if "lon" in df.columns else None,
        authoritative_ids=expected,
        context=context,
    )
    has_coordinates = {"lat", "lon"}.issubset(df.columns)
    if convention is None or not has_coordinates:
        out = ensure_spatial_id_col(df, spec=spec, context=context)
    else:
        out = ensure_spatial_id_col(
            df,
            spec=spec,
            convention=convention,
            force_latlon=True,
            context=context,
        )

    raw_ids = set(out["id"].astype(str).unique())
    expected_set = set(expected.astype(str))
    missing = sorted(expected_set - raw_ids)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"{context}: {len(missing)} expected weight source IDs are absent "
            f"from the raw NetCDF coordinates: {preview}"
        )
    return out, sorted(raw_ids - expected_set), convention
