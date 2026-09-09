"""Validated GEO-001 selected level-1 Cartesian geometry."""

from __future__ import annotations

import numpy as np

from simesh._kernels.primitives._geometry import (
    fill_level1_block_geometry_unchecked,
    validate_selected_geometry_unchecked,
)
from simesh._amr.foundation import INDEX_DTYPE, PAYLOAD_DTYPE, _require_index_triplet
from simesh._amr.morton import _root_volume


def _require_float_triplet(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != PAYLOAD_DTYPE:
        raise TypeError(f"{name} must have dtype float64")
    if value.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


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


def _require_mapping_input(
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


def _require_geometry_outputs(
    block_bounds: np.ndarray,
    cell_spacing: np.ndarray,
    slot_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(block_bounds, np.ndarray):
        raise TypeError("block_bounds must be a NumPy array")
    if block_bounds.dtype != PAYLOAD_DTYPE:
        raise TypeError("block_bounds must have dtype float64")
    if block_bounds.shape != (slot_count, 2, 3):
        raise ValueError(
            f"block_bounds must have shape {(slot_count, 2, 3)}, "
            f"got {block_bounds.shape}"
        )
    if not block_bounds.flags.c_contiguous:
        raise ValueError("block_bounds must be C-contiguous")
    if not block_bounds.flags.writeable:
        raise ValueError("block_bounds must be writable")
    cell_spacing = _require_float_triplet("cell_spacing", cell_spacing)
    if not cell_spacing.flags.writeable:
        raise ValueError("cell_spacing must be writable")
    return block_bounds, cell_spacing


def fill_level1_block_geometry(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
    block_ids: np.ndarray,
    block_bounds: np.ndarray,
    cell_spacing: np.ndarray,
) -> None:
    """Fill selected block bounds and one global spacing triplet."""
    domain_lower = _require_float_triplet("domain_lower", domain_lower)
    domain_upper = _require_float_triplet("domain_upper", domain_upper)
    domain_cell_counts = _require_index_triplet(
        "domain_cell_counts", domain_cell_counts
    )
    block_cell_counts = _require_index_triplet(
        "block_cell_counts", block_cell_counts
    )
    block_ids = _require_index_vector("block_ids", block_ids)

    if not np.all(np.isfinite(domain_lower)) or not np.all(np.isfinite(domain_upper)):
        raise ValueError("domain bounds must be finite")
    if np.any(domain_upper <= domain_lower):
        raise ValueError("domain_upper must be greater than domain_lower")
    if np.any(domain_cell_counts <= 0) or np.any(block_cell_counts <= 0):
        raise ValueError("cell counts must be positive")
    if np.any(domain_cell_counts % block_cell_counts != 0):
        raise ValueError("domain_cell_counts must be divisible by block_cell_counts")

    root_shape = np.ascontiguousarray(
        domain_cell_counts // block_cell_counts,
        dtype=np.int64,
    )
    volume = _root_volume(root_shape)
    forward_shape = tuple(int(value) for value in root_shape)
    coord_to_rank = _require_mapping_input(
        "coord_to_rank",
        coord_to_rank,
        forward_shape,
    )
    rank_to_coord = _require_mapping_input(
        "rank_to_coord",
        rank_to_coord,
        (volume, 3),
    )
    block_bounds, cell_spacing = _require_geometry_outputs(
        block_bounds,
        cell_spacing,
        block_ids.shape[0],
    )

    spacing_values = np.empty(3, dtype=np.float64)
    for axis in range(3):
        extent = float(domain_upper[axis]) - float(domain_lower[axis])
        spacing_values[axis] = extent / int(domain_cell_counts[axis])
    if (
        not np.all(np.isfinite(spacing_values))
        or np.any(spacing_values < np.finfo(np.float64).tiny)
    ):
        raise ValueError("cell spacing must be finite, positive, and normal")
    for axis in range(3):
        first_center = float(domain_lower[axis]) + 0.5 * spacing_values[axis]
        last_index = int(domain_cell_counts[axis]) - 1
        last_center = float(domain_lower[axis]) + (
            float(last_index) + 0.5
        ) * spacing_values[axis]
        if not (
            np.isfinite(first_center)
            and np.isfinite(last_center)
            and float(domain_lower[axis]) < first_center
            and first_center <= last_center
            and last_center < float(domain_upper[axis])
        ):
            raise ValueError(
                "first and last cell centers must be representable inside the domain"
            )

    outputs = (block_bounds, cell_spacing)
    inputs = (
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        coord_to_rank,
        rank_to_coord,
        block_ids,
    )
    if np.shares_memory(block_bounds, cell_spacing) or any(
        np.shares_memory(output, source)
        for output in outputs
        for source in inputs
    ):
        raise ValueError("geometry outputs must not overlap inputs or each other")

    invalid_slot = int(
        validate_selected_geometry_unchecked(
            domain_lower,
            domain_upper,
            domain_cell_counts,
            block_cell_counts,
            coord_to_rank,
            rank_to_coord,
            block_ids,
            spacing_values,
        )
    )
    if invalid_slot >= 0:
        raise ValueError(f"invalid selected block geometry at slot {invalid_slot}")

    fill_level1_block_geometry_unchecked(
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        rank_to_coord,
        block_ids,
        spacing_values,
        block_bounds,
        cell_spacing,
    )


def level1_block_geometry(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
    block_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate and return selected bounds and global cell spacing."""
    block_ids = _require_index_vector("block_ids", block_ids)
    block_bounds = np.empty((block_ids.shape[0], 2, 3), dtype=np.float64)
    cell_spacing = np.empty(3, dtype=np.float64)
    fill_level1_block_geometry(
        domain_lower,
        domain_upper,
        domain_cell_counts,
        block_cell_counts,
        coord_to_rank,
        rank_to_coord,
        block_ids,
        block_bounds,
        cell_spacing,
    )
    return block_bounds, cell_spacing
