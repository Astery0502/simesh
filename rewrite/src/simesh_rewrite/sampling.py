"""Validated sampling and native-grid placement boundaries."""

from __future__ import annotations

import numpy as np

from ._sampling import (
    place_level1_blocks_unchecked,
    validate_selected_level1_placement_unchecked,
)
from .foundation import (
    INDEX_DTYPE,
    PAYLOAD_DTYPE,
    _require_index_triplet,
    _require_payload,
)
from .morton import _root_volume


def _require_index_vector(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.ndim != 1 or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous vector")
    return value


def _require_mapping(
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
    return value


def _require_uniform_grid(uniform_grid: np.ndarray) -> np.ndarray:
    if not isinstance(uniform_grid, np.ndarray):
        raise TypeError("uniform_grid must be a NumPy array")
    if uniform_grid.dtype != PAYLOAD_DTYPE:
        raise TypeError("uniform_grid must have dtype float64")
    if uniform_grid.ndim != 4:
        raise ValueError("uniform_grid must have layout (field, x, y, z)")
    if not uniform_grid.flags.c_contiguous:
        raise ValueError("uniform_grid must be C-contiguous")
    if not uniform_grid.flags.writeable:
        raise ValueError("uniform_grid must be writable")
    return uniform_grid


def place_level1_blocks(
    payload: np.ndarray,
    payload_valid_lower: np.ndarray,
    payload_valid_upper: np.ndarray,
    block_ids: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
    uniform_grid: np.ndarray,
) -> None:
    """Place selected level-1 block interiors on the exact native grid."""
    payload = _require_payload("payload", payload, writable=False)
    payload_valid_lower = _require_index_triplet(
        "payload_valid_lower", payload_valid_lower
    )
    payload_valid_upper = _require_index_triplet(
        "payload_valid_upper", payload_valid_upper
    )
    block_ids = _require_index_vector("block_ids", block_ids)
    domain_cell_counts = _require_index_triplet(
        "domain_cell_counts", domain_cell_counts
    )
    block_cell_counts = _require_index_triplet(
        "block_cell_counts", block_cell_counts
    )
    uniform_grid = _require_uniform_grid(uniform_grid)

    if payload.shape[0] != block_ids.shape[0]:
        raise ValueError("payload slot axis must match block_ids")
    if np.any(domain_cell_counts <= 0) or np.any(block_cell_counts <= 0):
        raise ValueError("cell counts must be positive")
    if np.any(domain_cell_counts % block_cell_counts != 0):
        raise ValueError("domain_cell_counts must be divisible by block_cell_counts")
    root_shape = np.ascontiguousarray(
        domain_cell_counts // block_cell_counts,
        dtype=np.int64,
    )
    volume = _root_volume(root_shape)
    coord_to_rank = _require_mapping(
        "coord_to_rank",
        coord_to_rank,
        tuple(int(value) for value in root_shape),
    )
    rank_to_coord = _require_mapping(
        "rank_to_coord",
        rank_to_coord,
        (volume, 3),
    )

    spatial_shape = np.asarray(payload.shape[2:], dtype=np.int64)
    if np.any(payload_valid_lower < 0) or np.any(
        payload_valid_lower >= payload_valid_upper
    ):
        raise ValueError("payload valid region must be nonempty and nonnegative")
    if np.any(payload_valid_upper > spatial_shape):
        raise ValueError("payload valid region exceeds payload spatial shape")
    if not np.array_equal(
        payload_valid_upper - payload_valid_lower,
        block_cell_counts,
    ):
        raise ValueError("payload valid extent must equal block_cell_counts")

    expected_output_shape = (
        payload.shape[1],
        *(int(value) for value in domain_cell_counts),
    )
    if uniform_grid.shape != expected_output_shape:
        raise ValueError(
            f"uniform_grid must have shape {expected_output_shape}, "
            f"got {uniform_grid.shape}"
        )

    readonly_inputs = (
        payload,
        payload_valid_lower,
        payload_valid_upper,
        block_ids,
        domain_cell_counts,
        block_cell_counts,
        coord_to_rank,
        rank_to_coord,
    )
    if any(np.shares_memory(uniform_grid, value) for value in readonly_inputs):
        raise ValueError("uniform_grid must not overlap placement inputs")

    invalid_slot = int(
        validate_selected_level1_placement_unchecked(
            root_shape,
            coord_to_rank,
            rank_to_coord,
            block_ids,
        )
    )
    if invalid_slot >= 0:
        raise ValueError(f"invalid selected block placement at slot {invalid_slot}")

    place_level1_blocks_unchecked(
        payload,
        payload_valid_lower,
        block_ids,
        block_cell_counts,
        rank_to_coord,
        uniform_grid,
    )
