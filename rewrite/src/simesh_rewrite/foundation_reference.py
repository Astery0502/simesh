"""Independent NumPy reference for the FND-001 semantic primitives."""

from __future__ import annotations

import numpy as np


_INDEX_MAX = int(np.iinfo(np.int64).max)


def _triplet(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.int64):
        raise TypeError(f"{name} must be an int64 NumPy array")
    if value.shape != (3,) or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous triplet")
    return value


def interior_region(
    interior_shape: np.ndarray,
    lower_halo: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    interior_shape = _triplet("interior_shape", interior_shape)
    lower_halo = _triplet("lower_halo", lower_halo)
    if np.any(interior_shape < 0) or np.any(lower_halo < 0):
        raise ValueError("shape and halo entries must be non-negative")
    if any(
        int(size) > _INDEX_MAX - int(lower)
        for size, lower in zip(interior_shape, lower_halo, strict=True)
    ):
        raise OverflowError("interior upper bound does not fit in int64")
    return lower_halo.copy(), lower_halo + interior_shape


def ravel_cell(index: np.ndarray, shape: np.ndarray) -> int:
    index = _triplet("index", index)
    shape = _triplet("shape", shape)
    if np.any(shape <= 0) or np.any(index < 0) or np.any(index >= shape):
        raise IndexError("cell index is outside the spatial shape")
    i, j, k = (int(value) for value in index)
    nx, ny, nz = (int(value) for value in shape)
    if nx * ny * nz > _INDEX_MAX:
        raise OverflowError("spatial volume does not fit in int64")
    return (i * ny + j) * nz + k


def unravel_cell(offset: int, shape: np.ndarray) -> tuple[int, int, int]:
    shape = _triplet("shape", shape)
    nx, ny, nz = (int(value) for value in shape)
    if nx * ny * nz > _INDEX_MAX:
        raise OverflowError("spatial volume does not fit in int64")
    if nx <= 0 or ny <= 0 or nz <= 0 or offset < 0 or offset >= nx * ny * nz:
        raise IndexError("cell offset is outside the spatial shape")
    i, remainder = divmod(int(offset), ny * nz)
    j, k = divmod(remainder, nz)
    return i, j, k


def copy_region_into(
    source: np.ndarray,
    source_lower: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
    extent: np.ndarray,
) -> None:
    source_lower = _triplet("source_lower", source_lower)
    destination_lower = _triplet("destination_lower", destination_lower)
    extent = _triplet("extent", extent)
    if source.dtype != np.float64 or destination.dtype != np.float64:
        raise TypeError("payloads must have dtype float64")
    if source.ndim != 5 or destination.ndim != 5:
        raise ValueError("payloads must have rank five")
    if not source.flags.c_contiguous or not destination.flags.c_contiguous:
        raise ValueError("payloads must be C-contiguous")
    if source.shape[:2] != destination.shape[:2]:
        raise ValueError("slot and field extents must agree")
    if np.any(source_lower < 0) or np.any(destination_lower < 0) or np.any(extent < 0):
        raise ValueError("region entries must be non-negative")
    source_shape = np.asarray(source.shape[2:], dtype=np.int64)
    destination_shape = np.asarray(destination.shape[2:], dtype=np.int64)
    if np.any(source_lower > source_shape) or np.any(
        extent > source_shape - source_lower
    ):
        raise ValueError("source region is out of bounds")
    if np.any(destination_lower > destination_shape) or np.any(
        extent > destination_shape - destination_lower
    ):
        raise ValueError("destination region is out of bounds")
    if np.shares_memory(source, destination):
        raise ValueError("source and destination must not overlap")

    source_slices = tuple(
        slice(int(lower), int(lower + size))
        for lower, size in zip(source_lower, extent, strict=True)
    )
    destination_slices = tuple(
        slice(int(lower), int(lower + size))
        for lower, size in zip(destination_lower, extent, strict=True)
    )
    destination[(slice(None), slice(None), *destination_slices)] = source[
        (slice(None), slice(None), *source_slices)
    ]
