"""Validated HPL-001 level-1 halo relation/source-slot plans."""

from __future__ import annotations

import numpy as np

from ._halos import (
    duplicate_block_id_index_unchecked,
    fill_level1_halo_relation_plan_unchecked,
    missing_halo_relation_plan_entry_unchecked,
)
from ._storage import validate_indices_unchecked
from .foundation import INDEX_DTYPE
from .halos import (
    _require_face_table,
    _require_index_vector,
    _require_primary_count,
)


def _require_plan_output(
    name: str,
    value: np.ndarray,
    dtype: np.dtype,
    primary_count: int,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype.name}")
    if value.shape != (primary_count, 27):
        raise ValueError(
            f"{name} must have shape {(primary_count, 27)}, got {value.shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def fill_level1_halo_relation_plan(
    block_ids: np.ndarray,
    primary_count: int,
    face_neighbor_ids: np.ndarray,
    source_slots: np.ndarray,
    physical_masks: np.ndarray,
) -> None:
    """Fill exact selected-slot and physical-mask plans for all primaries."""
    block_ids = _require_index_vector("block_ids", block_ids)
    primary_count = _require_primary_count(primary_count)
    face_neighbor_ids = _require_face_table(face_neighbor_ids)
    if primary_count > block_ids.shape[0]:
        raise ValueError("primary_count exceeds selected slot count")
    source_slots = _require_plan_output(
        "source_slots",
        source_slots,
        INDEX_DTYPE,
        primary_count,
    )
    physical_masks = _require_plan_output(
        "physical_masks",
        physical_masks,
        np.dtype(np.uint8),
        primary_count,
    )

    inputs = (block_ids, face_neighbor_ids)
    for output in (source_slots, physical_masks):
        if any(np.shares_memory(output, value) for value in inputs):
            raise ValueError("plan outputs must not overlap inputs")
    if np.shares_memory(source_slots, physical_masks):
        raise ValueError("plan outputs must not overlap each other")

    invalid_block = int(
        validate_indices_unchecked(block_ids, face_neighbor_ids.shape[0])
    )
    if invalid_block >= 0:
        raise ValueError(f"block_ids entry {invalid_block} is out of range")
    duplicate = int(duplicate_block_id_index_unchecked(block_ids))
    if duplicate >= 0:
        raise ValueError(f"block_ids entry {duplicate} duplicates an earlier ID")
    missing_entry = int(
        missing_halo_relation_plan_entry_unchecked(
            primary_count,
            block_ids,
            face_neighbor_ids,
        )
    )
    if missing_entry >= 0:
        primary, issue = divmod(missing_entry, 54)
        if issue >= 27:
            direction = issue - 27
            raise ValueError(
                f"primary slot {primary} direction {direction} traverses a "
                "face relation outside [-1, block_count)"
            )
        direction = issue
        raise ValueError(
            f"primary slot {primary} direction {direction} lacks selected closure"
        )

    fill_level1_halo_relation_plan_unchecked(
        primary_count,
        block_ids,
        face_neighbor_ids,
        source_slots,
        physical_masks,
    )


def level1_halo_relation_plan(
    block_ids: np.ndarray,
    primary_count: int,
    face_neighbor_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate and return the exact HPL-001 plan arrays."""
    primary_count = _require_primary_count(primary_count)
    source_slots = np.empty((primary_count, 27), dtype=np.int64)
    physical_masks = np.empty((primary_count, 27), dtype=np.uint8)
    fill_level1_halo_relation_plan(
        block_ids,
        primary_count,
        face_neighbor_ids,
        source_slots,
        physical_masks,
    )
    return source_slots, physical_masks
