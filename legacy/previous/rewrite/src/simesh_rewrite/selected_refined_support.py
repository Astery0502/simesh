"""Validated SPR-001 explicit selected-primary refined support planning."""

from __future__ import annotations

import numpy as np

from ._refined_support import (
    invalid_selected_primary_order_unchecked,
    maximum_selected_refined_support_slots_unchecked,
    plan_selected_refined_support_prefix_unchecked,
)
from .foundation import INDEX_DTYPE
from .primary import _require_primary_ids
from .refined_support import (
    _require_relation_inputs,
    _validate_relation_rows,
)
from .workspace import _require_nonnegative_integer


def _require_primary_selection(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("primary_leaf_ids must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError("primary_leaf_ids must have dtype int64")
    if value.ndim != 1 or not value.flags.c_contiguous:
        raise ValueError("primary_leaf_ids must be a C-contiguous vector")
    return value


def _validate_primary_selection(
    primary_leaf_ids: np.ndarray,
    leaf_count: int,
) -> None:
    invalid = int(
        invalid_selected_primary_order_unchecked(
            primary_leaf_ids, leaf_count
        )
    )
    if invalid >= 0:
        raise ValueError(
            "primary_leaf_ids must be strictly increasing in range; "
            f"first invalid entry is {invalid}"
        )


def plan_selected_refined_support_prefix(
    primary_leaf_ids: np.ndarray,
    first_primary_position: int,
    leaf_count: int,
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
    selected_leaf_ids: np.ndarray,
) -> tuple[int, int]:
    """Plan a bounded prefix of a sparse ascending primary selection."""
    primary_leaf_ids = _require_primary_selection(primary_leaf_ids)
    first_primary_position = _require_nonnegative_integer(
        "first_primary_position", first_primary_position
    )
    leaf_count = _require_nonnegative_integer("leaf_count", leaf_count)
    source_counts, source_leaf_ids = _require_relation_inputs(
        source_counts,
        source_leaf_ids,
    )
    selected_leaf_ids = _require_primary_ids(
        "selected_leaf_ids", selected_leaf_ids
    )

    primary_count_total = primary_leaf_ids.shape[0]
    if first_primary_position > primary_count_total:
        raise ValueError("first_primary_position exceeds primary selection")
    expected_rows = min(
        selected_leaf_ids.shape[0],
        primary_count_total - first_primary_position,
    )
    if source_counts.shape[0] != expected_rows:
        raise ValueError(
            f"source rows must equal min(capacity, remaining)={expected_rows}"
        )
    if any(
        np.shares_memory(selected_leaf_ids, value)
        for value in (primary_leaf_ids, source_counts, source_leaf_ids)
    ):
        raise ValueError("selected_leaf_ids must not overlap inputs")

    _validate_primary_selection(primary_leaf_ids, leaf_count)
    _validate_relation_rows(source_counts, source_leaf_ids, leaf_count)
    if (
        first_primary_position < primary_count_total
        and selected_leaf_ids.shape[0] == 0
    ):
        raise ValueError("selected capacity must be positive before the end")

    stop = first_primary_position + expected_rows
    candidates = primary_leaf_ids[first_primary_position:stop]
    return tuple(
        int(value)
        for value in plan_selected_refined_support_prefix_unchecked(
            candidates,
            source_counts,
            source_leaf_ids,
            selected_leaf_ids,
        )
    )


def maximum_selected_refined_support_slots(
    candidate_leaf_ids: np.ndarray,
    leaf_count: int,
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
) -> int:
    """Return the largest one-primary closure in explicit candidate rows."""
    candidate_leaf_ids = _require_primary_selection(candidate_leaf_ids)
    leaf_count = _require_nonnegative_integer("leaf_count", leaf_count)
    source_counts, source_leaf_ids = _require_relation_inputs(
        source_counts,
        source_leaf_ids,
    )
    if source_counts.shape[0] != candidate_leaf_ids.shape[0]:
        raise ValueError("source rows must equal candidate leaf count")
    _validate_primary_selection(candidate_leaf_ids, leaf_count)
    _validate_relation_rows(source_counts, source_leaf_ids, leaf_count)
    return int(
        maximum_selected_refined_support_slots_unchecked(
            candidate_leaf_ids,
            source_counts,
            source_leaf_ids,
        )
    )
