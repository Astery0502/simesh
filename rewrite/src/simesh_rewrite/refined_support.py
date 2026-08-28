"""Validated STO-004 balanced refined support planning."""

from __future__ import annotations

import numpy as np

from ._refined_support import (
    maximum_balanced_refined_support_slots_unchecked,
    plan_balanced_refined_support_prefix_unchecked,
    validate_refined_support_rows_unchecked,
)
from .primary import _require_primary_ids
from .workspace import _require_nonnegative_integer


def _require_source_counts(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("source_counts must be a NumPy array")
    if value.dtype != np.dtype(np.uint8):
        raise TypeError("source_counts must have dtype uint8")
    if value.ndim != 2 or not value.flags.c_contiguous:
        raise ValueError("source_counts must be a C-contiguous matrix")
    return value


def _require_source_leaf_ids(
    value: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("source_leaf_ids must be a NumPy array")
    if value.dtype != np.dtype(np.int64):
        raise TypeError("source_leaf_ids must have dtype int64")
    if value.shape != shape:
        raise ValueError(
            f"source_leaf_ids must have shape {shape}, got {value.shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError("source_leaf_ids must be C-contiguous")
    return value


def _validate_relation_rows(
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
    leaf_count: int,
) -> None:
    status, row, direction, source = validate_refined_support_rows_unchecked(
        source_counts,
        source_leaf_ids,
        leaf_count,
    )
    if status == 1:
        raise ValueError(
            f"source count exceeds four at row {row}, direction {direction}"
        )
    if status == 2:
        raise ValueError(
            "active source leaf ID is out of range at "
            f"row {row}, direction {direction}, source {source}"
        )
    if status == 3:
        raise ValueError(
            "unused source leaf ID must be -1 at "
            f"row {row}, direction {direction}, source {source}"
        )
    if status != 0:
        raise RuntimeError(f"unexpected refined support validation status {status}")


def _require_relation_inputs(
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_counts = _require_source_counts(source_counts)
    source_leaf_ids = _require_source_leaf_ids(
        source_leaf_ids,
        (*source_counts.shape, 4),
    )
    return source_counts, source_leaf_ids


def plan_balanced_refined_support_prefix(
    first_primary_id: int,
    leaf_count: int,
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
    selected_leaf_ids: np.ndarray,
) -> tuple[int, int]:
    """Fill maximal dense refined primaries plus unique REL source support."""
    first_primary_id = _require_nonnegative_integer(
        "first_primary_id", first_primary_id
    )
    leaf_count = _require_nonnegative_integer("leaf_count", leaf_count)
    source_counts, source_leaf_ids = _require_relation_inputs(
        source_counts,
        source_leaf_ids,
    )
    selected_leaf_ids = _require_primary_ids(
        "selected_leaf_ids", selected_leaf_ids
    )
    if first_primary_id > leaf_count:
        raise ValueError("first_primary_id exceeds leaf count")
    expected_rows = min(
        selected_leaf_ids.shape[0], leaf_count - first_primary_id
    )
    if source_counts.shape[0] != expected_rows:
        raise ValueError(
            f"source rows must equal min(capacity, remaining)={expected_rows}"
        )
    if np.shares_memory(selected_leaf_ids, source_counts) or np.shares_memory(
        selected_leaf_ids, source_leaf_ids
    ):
        raise ValueError("selected_leaf_ids must not overlap relation inputs")
    _validate_relation_rows(source_counts, source_leaf_ids, leaf_count)
    if first_primary_id < leaf_count and selected_leaf_ids.shape[0] == 0:
        raise ValueError("selected capacity must be positive before the end")
    return tuple(
        int(value)
        for value in plan_balanced_refined_support_prefix_unchecked(
            first_primary_id,
            source_counts,
            source_leaf_ids,
            selected_leaf_ids,
        )
    )


def maximum_balanced_refined_support_slots(
    first_primary_id: int,
    leaf_count: int,
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
) -> int:
    """Return exact maximum one-primary closure in bounded REL rows."""
    first_primary_id = _require_nonnegative_integer(
        "first_primary_id", first_primary_id
    )
    leaf_count = _require_nonnegative_integer("leaf_count", leaf_count)
    source_counts, source_leaf_ids = _require_relation_inputs(
        source_counts,
        source_leaf_ids,
    )
    if first_primary_id > leaf_count:
        raise ValueError("first_primary_id exceeds leaf count")
    if source_counts.shape[0] > leaf_count - first_primary_id:
        raise ValueError("source rows exceed remaining leaf count")
    _validate_relation_rows(source_counts, source_leaf_ids, leaf_count)
    return int(
        maximum_balanced_refined_support_slots_unchecked(
            first_primary_id,
            source_counts,
            source_leaf_ids,
        )
    )
