"""Independent arbitrary-precision reference for MOR-001."""

from __future__ import annotations

import numpy as np


_INDEX_MAX = int(np.iinfo(np.int64).max)


def _validate_root_shape(root_shape: np.ndarray) -> tuple[int, int, int]:
    if not isinstance(root_shape, np.ndarray) or root_shape.dtype != np.int64:
        raise TypeError("root_shape must be an int64 NumPy array")
    if root_shape.shape != (3,) or not root_shape.flags.c_contiguous:
        raise ValueError("root_shape must be a C-contiguous triplet")
    if np.any(root_shape <= 0):
        raise ValueError("root_shape entries must be positive")
    shape = tuple(int(value) for value in root_shape)
    volume = shape[0] * shape[1] * shape[2]
    if volume > _INDEX_MAX:
        raise OverflowError("root-grid volume does not fit in int64")
    return shape


def morton_key(x: int, y: int, z: int) -> int:
    """Return the arbitrary-precision mathematical key used only as evidence."""
    x, y, z = int(x), int(y), int(z)
    if x < 0 or y < 0 or z < 0:
        raise ValueError("Morton reference coordinates must be non-negative")
    key = 0
    bit = 0
    while x or y or z:
        key |= (x & 1) << (3 * bit)
        key |= (y & 1) << (3 * bit + 1)
        key |= (z & 1) << (3 * bit + 2)
        x >>= 1
        y >>= 1
        z >>= 1
        bit += 1
    return key


def level1_morton_reference(
    root_shape: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shape = _validate_root_shape(root_shape)
    coordinates = list(np.ndindex(shape))
    coordinates.sort(key=lambda coordinate: morton_key(*coordinate))
    rank_to_coord = np.asarray(coordinates, dtype=np.int64)
    coord_to_rank = np.empty(shape, dtype=np.int64)
    coord_to_rank[
        rank_to_coord[:, 0],
        rank_to_coord[:, 1],
        rank_to_coord[:, 2],
    ] = np.arange(len(coordinates), dtype=np.int64)
    return coord_to_rank, rank_to_coord
