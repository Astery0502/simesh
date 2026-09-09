"""Validated STO-001 in-memory block gather and scatter."""

from __future__ import annotations

from typing import Final

import numpy as np

from ._storage import (
    gather_blocks_into_unchecked,
    scatter_blocks_from_unchecked,
    validate_indices_unchecked,
)
from .foundation import INDEX_DTYPE, _require_index_triplet, _require_payload


_INDEX_MAX: Final = int(np.iinfo(np.int64).max)


def _require_region(
    name: str,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = _require_index_triplet(f"{name}_lower", lower)
    upper = _require_index_triplet(f"{name}_upper", upper)
    if np.any(lower < 0):
        raise ValueError(f"{name} lower bound must be non-negative")
    if np.any(lower > upper):
        raise ValueError(f"{name} lower bound exceeds its upper bound")
    return lower, upper


def _require_index_vector(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.ndim != 1:
        raise ValueError(f"{name} must have rank one, got shape {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _validate_source_region(
    name: str,
    lower: np.ndarray,
    upper: np.ndarray,
    spatial_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower, upper = _require_region(name, lower, upper)
    shape = np.asarray(spatial_shape, dtype=np.int64)
    if np.any(upper > shape):
        raise ValueError(f"{name} exceeds the source spatial shape")
    return lower, upper, upper - lower


def _validate_translated_region(
    name: str,
    lower: np.ndarray,
    extent: np.ndarray,
    spatial_shape: tuple[int, int, int],
) -> np.ndarray:
    lower = _require_index_triplet(f"{name}_lower", lower)
    if np.any(lower < 0):
        raise ValueError(f"{name} lower bound must be non-negative")
    if any(
        int(size) > _INDEX_MAX - int(start)
        for start, size in zip(lower, extent, strict=True)
    ):
        raise OverflowError(f"{name} upper bound does not fit in int64")
    shape = np.asarray(spatial_shape, dtype=np.int64)
    if np.any(lower > shape) or np.any(extent > shape - lower):
        raise ValueError(f"{name} exceeds the output spatial shape")
    return lower


def _validate_selectors(
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    block_count: int,
    field_count: int,
) -> None:
    invalid_block = int(validate_indices_unchecked(block_ids, block_count))
    if invalid_block >= 0:
        raise ValueError(f"block_ids entry {invalid_block} is out of range")
    invalid_field = int(validate_indices_unchecked(field_ids, field_count))
    if invalid_field >= 0:
        raise ValueError(f"field_ids entry {invalid_field} is out of range")


def gather_blocks_into(
    backing: np.ndarray,
    backing_valid_lower: np.ndarray,
    backing_valid_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    """Gather selected backing data into a translated caller-owned box."""
    backing = _require_payload("backing", backing, writable=False)
    destination = _require_payload("destination", destination, writable=True)
    block_ids = _require_index_vector("block_ids", block_ids)
    field_ids = _require_index_vector("field_ids", field_ids)
    if destination.shape[:2] != (block_ids.shape[0], field_ids.shape[0]):
        raise ValueError(
            "destination slot/field shape must match block_ids and field_ids"
        )
    source_lower, _, extent = _validate_source_region(
        "backing_valid",
        backing_valid_lower,
        backing_valid_upper,
        backing.shape[2:],
    )
    destination_lower = _validate_translated_region(
        "destination",
        destination_lower,
        extent,
        destination.shape[2:],
    )
    _validate_selectors(
        block_ids,
        field_ids,
        backing.shape[0],
        backing.shape[1],
    )
    if any(
        np.shares_memory(destination, source)
        for source in (
            backing,
            backing_valid_lower,
            backing_valid_upper,
            block_ids,
            field_ids,
            destination_lower,
        )
    ):
        raise ValueError("gather destination must not overlap inputs")

    gather_blocks_into_unchecked(
        backing,
        source_lower,
        extent,
        block_ids,
        field_ids,
        destination,
        destination_lower,
    )


def scatter_blocks_from(
    source: np.ndarray,
    source_valid_lower: np.ndarray,
    source_valid_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    backing: np.ndarray,
    backing_lower: np.ndarray,
) -> None:
    """Scatter a valid source box into selected resident backing cells."""
    source = _require_payload("source", source, writable=False)
    backing = _require_payload("backing", backing, writable=True)
    block_ids = _require_index_vector("block_ids", block_ids)
    field_ids = _require_index_vector("field_ids", field_ids)
    if source.shape[:2] != (block_ids.shape[0], field_ids.shape[0]):
        raise ValueError("source slot/field shape must match block_ids and field_ids")
    source_lower, _, extent = _validate_source_region(
        "source_valid",
        source_valid_lower,
        source_valid_upper,
        source.shape[2:],
    )
    backing_lower = _validate_translated_region(
        "backing",
        backing_lower,
        extent,
        backing.shape[2:],
    )
    _validate_selectors(
        block_ids,
        field_ids,
        backing.shape[0],
        backing.shape[1],
    )
    if any(
        np.shares_memory(backing, input_array)
        for input_array in (
            source,
            source_valid_lower,
            source_valid_upper,
            block_ids,
            field_ids,
            backing_lower,
        )
    ):
        raise ValueError("scatter backing must not overlap inputs")

    scatter_blocks_from_unchecked(
        source,
        source_lower,
        extent,
        block_ids,
        field_ids,
        backing,
        backing_lower,
    )
