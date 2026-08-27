"""Validated Python boundary for the FND-001 Cython primitives."""

from __future__ import annotations

from typing import Final

import numpy as np

from ._foundation import (
    copy_region_into_unchecked,
    ravel_cell_unchecked,
    unravel_cell_unchecked,
)


AXIS_NAMES: Final = ("x", "y", "z")
INDEX_DTYPE: Final = np.dtype(np.int64)
PAYLOAD_DTYPE: Final = np.dtype(np.float64)
_INDEX_MAX: Final = int(np.iinfo(np.int64).max)


def _require_index_triplet(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _require_payload(name: str, value: np.ndarray, *, writable: bool) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != PAYLOAD_DTYPE:
        raise TypeError(f"{name} must have dtype float64")
    if value.ndim != 5:
        raise ValueError(
            f"{name} must have layout (slot, field, x, y, z), got rank {value.ndim}"
        )
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def interior_region(
    interior_shape: np.ndarray,
    lower_halo: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the half-open interior box in padded-storage coordinates."""
    interior_shape = _require_index_triplet("interior_shape", interior_shape)
    lower_halo = _require_index_triplet("lower_halo", lower_halo)
    if np.any(interior_shape < 0):
        raise ValueError("interior_shape entries must be non-negative")
    if np.any(lower_halo < 0):
        raise ValueError("lower_halo entries must be non-negative")
    if any(
        int(size) > _INDEX_MAX - int(lower)
        for size, lower in zip(interior_shape, lower_halo, strict=True)
    ):
        raise OverflowError("interior upper bound does not fit in int64")
    lower = lower_halo.copy()
    upper = lower + interior_shape
    return lower, upper


def ravel_cell(index: np.ndarray, shape: np.ndarray) -> int:
    """Map a zero-based ``(x, y, z)`` cell index to C-order spatial offset."""
    index = _require_index_triplet("index", index)
    shape = _require_index_triplet("shape", shape)
    if np.any(shape <= 0):
        raise ValueError("shape entries must be positive")
    if np.any(index < 0) or np.any(index >= shape):
        raise IndexError(f"index {tuple(index)} is outside shape {tuple(shape)}")
    volume = int(shape[0]) * int(shape[1]) * int(shape[2])
    if volume > _INDEX_MAX:
        raise OverflowError("spatial volume does not fit in int64")
    return int(ravel_cell_unchecked(index, shape))


def unravel_cell(offset: int, shape: np.ndarray) -> tuple[int, int, int]:
    """Invert :func:`ravel_cell` for a C-order spatial payload."""
    shape = _require_index_triplet("shape", shape)
    if np.any(shape <= 0):
        raise ValueError("shape entries must be positive")
    if not isinstance(offset, (int, np.integer)):
        raise TypeError("offset must be an integer")
    offset = int(offset)
    volume = int(shape[0]) * int(shape[1]) * int(shape[2])
    if volume > _INDEX_MAX:
        raise OverflowError("spatial volume does not fit in int64")
    if offset < 0 or offset >= volume:
        raise IndexError(f"offset {offset} is outside payload volume {volume}")
    result = unravel_cell_unchecked(offset, shape)
    return int(result[0]), int(result[1]), int(result[2])


def copy_region_into(
    source: np.ndarray,
    source_lower: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
    extent: np.ndarray,
) -> None:
    """Copy one exact spatial box between non-overlapping canonical payloads."""
    source = _require_payload("source", source, writable=False)
    destination = _require_payload("destination", destination, writable=True)
    source_lower = _require_index_triplet("source_lower", source_lower)
    destination_lower = _require_index_triplet(
        "destination_lower", destination_lower
    )
    extent = _require_index_triplet("extent", extent)

    if source.shape[:2] != destination.shape[:2]:
        raise ValueError(
            "source and destination must have the same slot and field extents"
        )
    if np.any(source_lower < 0) or np.any(destination_lower < 0):
        raise ValueError("region lower bounds must be non-negative")
    if np.any(extent < 0):
        raise ValueError("region extent entries must be non-negative")

    source_shape = np.asarray(source.shape[2:], dtype=np.int64)
    destination_shape = np.asarray(destination.shape[2:], dtype=np.int64)
    if np.any(source_lower > source_shape) or np.any(
        extent > source_shape - source_lower
    ):
        raise ValueError("source region exceeds the source spatial extent")
    if np.any(destination_lower > destination_shape) or np.any(
        extent > destination_shape - destination_lower
    ):
        raise ValueError("destination region exceeds the destination spatial extent")
    if np.shares_memory(source, destination):
        raise ValueError("source and destination must not overlap")

    copy_region_into_unchecked(
        source,
        source_lower,
        destination,
        destination_lower,
        extent,
    )
