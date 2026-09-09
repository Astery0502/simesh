"""Validated MOR-001 dense level-1 Morton mappings."""

from __future__ import annotations

from typing import Final

import numpy as np

from ._morton import fill_level1_morton_unchecked
from .foundation import INDEX_DTYPE, _require_index_triplet


_INDEX_MAX: Final = int(np.iinfo(np.int64).max)


def _root_volume(root_shape: np.ndarray) -> int:
    root_shape = _require_index_triplet("root_shape", root_shape)
    if np.any(root_shape <= 0):
        raise ValueError("root_shape entries must be positive")
    volume = 1
    for extent in root_shape:
        extent = int(extent)
        if volume > _INDEX_MAX // extent:
            raise OverflowError("root-grid volume does not fit in int64")
        volume *= extent
    return volume


def _require_mapping_output(
    name: str,
    value: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def fill_level1_morton(
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
) -> None:
    """Fill caller-owned exact forward and inverse dense Morton mappings."""
    volume = _root_volume(root_shape)
    forward_shape = tuple(int(value) for value in root_shape)
    coord_to_rank = _require_mapping_output(
        "coord_to_rank",
        coord_to_rank,
        forward_shape,
    )
    rank_to_coord = _require_mapping_output(
        "rank_to_coord",
        rank_to_coord,
        (volume, 3),
    )
    if np.shares_memory(root_shape, coord_to_rank) or np.shares_memory(
        root_shape, rank_to_coord
    ):
        raise ValueError("root_shape must not overlap either output")
    if np.shares_memory(coord_to_rank, rank_to_coord):
        raise ValueError("Morton mapping outputs must not overlap")

    fill_level1_morton_unchecked(root_shape, coord_to_rank, rank_to_coord)


def level1_morton(root_shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Allocate and return exact forward and inverse dense Morton mappings."""
    volume = _root_volume(root_shape)
    forward_shape = tuple(int(value) for value in root_shape)
    coord_to_rank = np.empty(forward_shape, dtype=np.int64)
    rank_to_coord = np.empty((volume, 3), dtype=np.int64)
    fill_level1_morton(root_shape, coord_to_rank, rank_to_coord)
    return coord_to_rank, rank_to_coord
