"""Shared sparse execution for source-to-target rainfall transformations."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class SparseCellTransform:
    """Compiled target-by-source weight and support matrices."""

    weighted_matrix: sparse.csr_matrix
    support_matrix: sparse.csr_matrix
    source_ids: tuple
    target_ids: tuple
    source_index: dict


def compile_sparse_cell_transform(
    weights_df,
    *,
    source_ids=None,
    target_ids=None,
    source_col="source_id",
    target_col="target_id",
    weight_col="weight",
):
    """Compile a long source/target weight table into reusable CSR matrices."""
    required = {source_col, target_col, weight_col}
    missing = required - set(weights_df.columns)
    if missing:
        raise ValueError(f"Sparse cell transform is missing columns: {sorted(missing)}")
    if weights_df.empty:
        raise ValueError("Sparse cell-transform weights must not be empty.")

    values = pd.to_numeric(weights_df[weight_col], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Compiled cell-transform weights must be finite and non-negative.")
    target_totals = pd.Series(values).groupby(weights_df[target_col].to_numpy()).sum()
    if (target_totals <= 0).any():
        raise ValueError("Every compiled cell-transform target must have positive weight.")
    positive = values > 0
    positive_weights = weights_df.loc[positive]
    source_ids = tuple(
        pd.unique(positive_weights[source_col]) if source_ids is None else source_ids
    )
    target_ids = tuple(
        pd.unique(positive_weights[target_col]) if target_ids is None else target_ids
    )
    source_index = {source_id: index for index, source_id in enumerate(source_ids)}
    target_index = {target_id: index for index, target_id in enumerate(target_ids)}
    source_pos = positive_weights[source_col].map(source_index)
    target_pos = positive_weights[target_col].map(target_index)
    if source_pos.isna().any() or target_pos.isna().any():
        raise ValueError("Cell-transform weights contain IDs outside the compiled axes.")

    shape = (len(target_ids), len(source_ids))
    weighted = sparse.csr_matrix(
        (
            values[positive],
            (
                target_pos.to_numpy(int),
                source_pos.to_numpy(int),
            ),
        ),
        shape=shape,
    )
    support = weighted.copy()
    support.data = np.ones_like(support.data)
    return SparseCellTransform(
        weighted_matrix=weighted,
        support_matrix=support,
        source_ids=source_ids,
        target_ids=target_ids,
        source_index=source_index,
    )


def sparse_observed_weighted_mean(transform, source_values, column_chunk_size=256):
    """Apply ``W @ filled / W @ valid`` to a source-by-column value matrix."""
    source_values = np.asarray(source_values, dtype=float)
    if source_values.ndim != 2:
        raise ValueError("Sparse transform values must be a two-dimensional matrix.")
    if source_values.shape[0] != len(transform.source_ids):
        raise ValueError(
            "Sparse transform source axis does not match the compiled weight matrix."
        )
    if column_chunk_size < 1:
        raise ValueError("column_chunk_size must be positive.")

    out = np.full((len(transform.target_ids), source_values.shape[1]), np.nan)
    for start in range(0, source_values.shape[1], int(column_chunk_size)):
        stop = min(source_values.shape[1], start + int(column_chunk_size))
        block = source_values[:, start:stop]
        valid = np.isfinite(block)
        numerator = np.asarray(transform.weighted_matrix @ np.where(valid, block, 0.0))
        denominator = np.asarray(transform.weighted_matrix @ valid.astype(float))
        np.divide(
            numerator,
            denominator,
            out=out[:, start:stop],
            where=denominator > 0,
        )
    return out


def sparse_target_support(transform, present_sources):
    """Return target support for each group from a source-by-group presence mask."""
    present_sources = np.asarray(present_sources, dtype=bool)
    if present_sources.ndim != 2 or present_sources.shape[0] != len(transform.source_ids):
        raise ValueError("Source-presence mask does not match the compiled weight matrix.")
    return np.asarray(transform.support_matrix @ present_sources.astype(float)) > 0
