"""Validated OPR-003 Cartesian 3D refined curl."""

from __future__ import annotations

import numpy as np

from ._curl import cartesian_curl_unchecked
from .foundation import INDEX_DTYPE, PAYLOAD_DTYPE, _require_index_triplet, _require_payload


_INDEX_MAX = int(np.iinfo(np.int64).max)


def _require_field_triplet(
    name: str,
    value: np.ndarray,
    field_count: int,
    *,
    distinct: bool,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.shape != (3,) or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous triplet")
    for position, field in enumerate(value):
        if int(field) < 0 or int(field) >= field_count:
            raise ValueError(f"{name} entry {position} is outside the field axis")
    if distinct and len({int(field) for field in value}) != 3:
        raise ValueError("destination_field_positions must be pairwise distinct")
    return value


def _require_slot_spacing(value: np.ndarray, slot_count: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("slot_cell_spacing must be a NumPy array")
    if value.dtype != PAYLOAD_DTYPE:
        raise TypeError("slot_cell_spacing must have dtype float64")
    if value.shape != (slot_count, 3):
        raise ValueError(
            f"slot_cell_spacing must have shape {(slot_count, 3)}, got {value.shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError("slot_cell_spacing must be C-contiguous")
    if not np.all(np.isfinite(value)) or np.any(
        value < np.finfo(np.float64).tiny
    ):
        raise ValueError(
            "slot_cell_spacing entries must be finite, positive, and normal"
        )
    return value


def cartesian_curl_into(
    source: np.ndarray,
    source_valid_lower: np.ndarray,
    source_valid_upper: np.ndarray,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    source_field_positions: np.ndarray,
    slot_cell_spacing: np.ndarray,
    destination: np.ndarray,
    destination_field_positions: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    """Compute the fixed three-component curl over one common source box."""
    source = _require_payload("source", source, writable=False)
    destination = _require_payload("destination", destination, writable=True)
    source_valid_lower = _require_index_triplet(
        "source_valid_lower", source_valid_lower
    )
    source_valid_upper = _require_index_triplet(
        "source_valid_upper", source_valid_upper
    )
    output_lower = _require_index_triplet("output_lower", output_lower)
    output_upper = _require_index_triplet("output_upper", output_upper)
    destination_lower = _require_index_triplet(
        "destination_lower", destination_lower
    )
    source_field_positions = _require_field_triplet(
        "source_field_positions",
        source_field_positions,
        source.shape[1],
        distinct=False,
    )
    destination_field_positions = _require_field_triplet(
        "destination_field_positions",
        destination_field_positions,
        destination.shape[1],
        distinct=True,
    )
    slot_cell_spacing = _require_slot_spacing(
        slot_cell_spacing, source.shape[0]
    )

    if source.shape[0] != destination.shape[0]:
        raise ValueError("source and destination slot extents must match")
    source_shape = np.asarray(source.shape[2:], dtype=np.int64)
    destination_shape = np.asarray(destination.shape[2:], dtype=np.int64)
    if np.any(source_valid_lower < 0) or np.any(
        source_valid_lower > source_valid_upper
    ):
        raise ValueError("source valid region must be ordered and nonnegative")
    if np.any(source_valid_upper > source_shape):
        raise ValueError("source valid region exceeds source spatial shape")
    if np.any(output_lower < 0) or np.any(output_lower > output_upper):
        raise ValueError("output region must be ordered and nonnegative")
    if np.any(output_upper > source_shape):
        raise ValueError("output region exceeds source spatial shape")

    extent = output_upper - output_lower
    empty = bool(np.any(extent == 0))
    if not empty:
        for axis in range(3):
            if (
                int(output_lower[axis]) < 1
                or int(output_upper[axis]) > _INDEX_MAX - 1
                or int(source_valid_lower[axis]) > int(output_lower[axis]) - 1
                or int(source_valid_upper[axis]) < int(output_upper[axis]) + 1
            ):
                raise ValueError(
                    "source valid region does not support aggregate curl reach"
                )

    if np.any(destination_lower < 0):
        raise ValueError("destination_lower must be nonnegative")
    for axis in range(3):
        if int(destination_lower[axis]) > _INDEX_MAX - int(extent[axis]):
            raise OverflowError("translated destination region does not fit in int64")
    if np.any(destination_lower > destination_shape) or np.any(
        extent > destination_shape - destination_lower
    ):
        raise ValueError("translated region exceeds destination spatial shape")

    metadata = (
        source_valid_lower,
        source_valid_upper,
        output_lower,
        output_upper,
        source_field_positions,
        slot_cell_spacing,
        destination_field_positions,
        destination_lower,
    )
    if np.shares_memory(source, destination) or any(
        np.shares_memory(destination, value) for value in metadata
    ):
        raise ValueError("destination must not overlap source or curl metadata")
    if empty or source.shape[0] == 0:
        return

    cartesian_curl_unchecked(
        source,
        output_lower,
        output_upper,
        source_field_positions,
        slot_cell_spacing,
        destination,
        destination_field_positions,
        destination_lower,
    )
