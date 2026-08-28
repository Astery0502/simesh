"""Validated RSL-001 accepted REL source-slot resolution."""

from __future__ import annotations

import numpy as np

from ._refined_support import validate_refined_support_rows_unchecked
from ._relation_slots import (
    duplicate_selected_leaf_id_unchecked,
    missing_refined_relation_source_unchecked,
    resolve_refined_relation_source_slots_unchecked,
)
from ._storage import validate_indices_unchecked
from .foundation import INDEX_DTYPE
from .refined_support import _require_source_counts, _require_source_leaf_ids
from .workspace import _require_nonnegative_integer


def _require_selected_leaf_ids(value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("selected_leaf_ids must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError("selected_leaf_ids must have dtype int64")
    if value.ndim != 1 or not value.flags.c_contiguous:
        raise ValueError("selected_leaf_ids must be a C-contiguous vector")
    return value


def _require_source_slots(
    value: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("source_slots must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError("source_slots must have dtype int64")
    if value.shape != shape:
        raise ValueError(f"source_slots must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError("source_slots must be C-contiguous")
    if not value.flags.writeable:
        raise ValueError("source_slots must be writable")
    return value


def _validate_relation_values(
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
        raise RuntimeError(f"unexpected relation-source validation status {status}")


def resolve_refined_relation_source_slots(
    leaf_count: int,
    selected_leaf_ids: np.ndarray,
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
    source_slots: np.ndarray,
) -> None:
    """Map every active accepted REL source ID to its selected slot."""
    leaf_count = _require_nonnegative_integer("leaf_count", leaf_count)
    selected_leaf_ids = _require_selected_leaf_ids(selected_leaf_ids)
    source_counts = _require_source_counts(source_counts)
    source_leaf_ids = _require_source_leaf_ids(
        source_leaf_ids,
        (*source_counts.shape, 4),
    )
    source_slots = _require_source_slots(
        source_slots,
        (*source_counts.shape, 4),
    )

    primary_count = source_counts.shape[0]
    if primary_count > selected_leaf_ids.shape[0]:
        raise ValueError("accepted relation rows exceed selected slots")

    invalid_selected = int(
        validate_indices_unchecked(selected_leaf_ids, leaf_count)
    )
    if invalid_selected >= 0:
        raise ValueError(
            f"selected_leaf_ids entry {invalid_selected} is out of range"
        )
    duplicate = int(
        duplicate_selected_leaf_id_unchecked(selected_leaf_ids)
    )
    if duplicate >= 0:
        raise ValueError(
            f"selected_leaf_ids entry {duplicate} duplicates an earlier ID"
        )

    _validate_relation_values(source_counts, source_leaf_ids, leaf_count)

    if any(
        np.shares_memory(source_slots, value)
        for value in (
            selected_leaf_ids,
            source_counts,
            source_leaf_ids,
        )
    ):
        raise ValueError("source_slots must not overlap inputs")

    missing = int(
        missing_refined_relation_source_unchecked(
            selected_leaf_ids,
            source_counts,
            source_leaf_ids,
        )
    )
    if missing >= 0:
        direction_count = source_counts.shape[1]
        primary, remainder = divmod(missing, direction_count * 4)
        direction, source = divmod(remainder, 4)
        leaf_id = int(source_leaf_ids[primary, direction, source])
        raise ValueError(
            "active source leaf ID is absent from selected slots at "
            f"row {primary}, direction {direction}, source {source}: {leaf_id}"
        )

    resolve_refined_relation_source_slots_unchecked(
        selected_leaf_ids,
        source_counts,
        source_leaf_ids,
        source_slots,
    )
