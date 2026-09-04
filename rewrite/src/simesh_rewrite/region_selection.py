"""Validated ROI-001 refined physical-region cell-center windows."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ._region_selection import (
    fill_refined_region_windows_unchecked,
    validate_and_count_refined_region_windows_unchecked,
)
from .forest_conformance import _require_index_array
from .foundation import INDEX_DTYPE
from .geometry import _require_float_triplet, _require_index_vector
from .morton import _root_volume


_INDEX_MAX = int(np.iinfo(np.int64).max)
_NORMAL_MIN = float(np.finfo(np.float64).tiny)


class RefinedRegionSelection(NamedTuple):
    """Exact ascending leaf IDs and their nonempty local cell windows."""

    leaf_ids: np.ndarray
    cell_lower: np.ndarray
    cell_upper: np.ndarray


def _validated_region_inputs(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    region_lower: np.ndarray,
    region_upper: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, int]:
    domain_lower = _require_float_triplet("domain_lower", domain_lower)
    domain_upper = _require_float_triplet("domain_upper", domain_upper)
    region_lower = _require_float_triplet("region_lower", region_lower)
    region_upper = _require_float_triplet("region_upper", region_upper)
    root_shape = _require_index_vector("root_shape", root_shape)
    domain_cell_counts = _require_index_vector(
        "domain_cell_counts", domain_cell_counts
    )
    block_cell_counts = _require_index_vector(
        "block_cell_counts", block_cell_counts
    )
    for name, value in (
        ("root_shape", root_shape),
        ("domain_cell_counts", domain_cell_counts),
        ("block_cell_counts", block_cell_counts),
    ):
        if value.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), got {value.shape}")

    if not np.all(np.isfinite(domain_lower)) or not np.all(
        np.isfinite(domain_upper)
    ):
        raise ValueError("domain bounds must be finite")
    if np.any(domain_upper <= domain_lower):
        raise ValueError("domain_upper must be greater than domain_lower")
    if not np.all(np.isfinite(region_lower)) or not np.all(
        np.isfinite(region_upper)
    ):
        raise ValueError("region bounds must be finite")
    if np.any(region_lower > region_upper):
        raise ValueError("region_lower must not exceed region_upper")
    if np.any(region_lower < domain_lower) or np.any(region_upper > domain_upper):
        raise ValueError("region bounds must be contained in the domain")
    if (
        np.any(root_shape <= 0)
        or np.any(domain_cell_counts <= 0)
        or np.any(block_cell_counts <= 0)
    ):
        raise ValueError("root shape and cell counts must be positive")
    _root_volume(root_shape)
    for axis in range(3):
        root_extent = int(root_shape[axis])
        block_extent = int(block_cell_counts[axis])
        if root_extent > _INDEX_MAX // block_extent:
            raise OverflowError("root block-cell product does not fit in int64")
        if int(domain_cell_counts[axis]) != root_extent * block_extent:
            raise ValueError(
                "domain_cell_counts must equal root_shape * block_cell_counts"
            )

    node_levels = _require_index_vector("node_levels", node_levels)
    if node_levels.shape[0] == 0:
        raise ValueError("node_levels must contain at least one node")
    node_coords = _require_index_array(
        "node_coords", node_coords, (node_levels.shape[0], 3)
    )
    leaf_node_ids = _require_index_vector("leaf_node_ids", leaf_node_ids)
    if leaf_node_ids.shape[0] == 0:
        raise ValueError("leaf_node_ids must contain at least one leaf")

    base_spacing = np.empty(3, dtype=np.float64)
    for axis in range(3):
        extent = float(domain_upper[axis]) - float(domain_lower[axis])
        base_spacing[axis] = extent / float(domain_cell_counts[axis])
    if (
        not np.all(np.isfinite(base_spacing))
        or np.any(base_spacing < _NORMAL_MIN)
    ):
        raise ValueError("base cell spacing must be finite, positive, and normal")

    inputs = (
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        region_lower,
        region_upper,
    )
    status, bad_leaf, bad_axis, selected_count = (
        validate_and_count_refined_region_windows_unchecked(
            domain_lower,
            domain_upper,
            root_shape,
            domain_cell_counts,
            block_cell_counts,
            node_levels,
            node_coords,
            leaf_node_ids,
            region_lower,
            region_upper,
            base_spacing,
        )
    )
    if status == 1:
        raise ValueError(
            "invalid refined forest geometry at "
            f"leaf {bad_leaf}, axis {bad_axis}"
        )
    if status == 2:
        raise OverflowError(
            "refined global-cell arithmetic overflows int64 at "
            f"leaf {bad_leaf}, axis {bad_axis}"
        )
    if status == 3:
        raise ValueError(
            "refined cell centers must be finite, ordered, and strictly "
            f"inside leaf faces; first invalid leaf is {bad_leaf}, axis {bad_axis}"
        )
    if status != 0:
        raise RuntimeError(f"unexpected region selection status {status}")
    return inputs, base_spacing, int(selected_count)


def _require_output_vector(
    name: str,
    value: np.ndarray,
    selected_count: int,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.shape != (selected_count,):
        raise ValueError(
            f"{name} must have shape {(selected_count,)}, got {value.shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def _require_output_boxes(
    name: str,
    value: np.ndarray,
    selected_count: int,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    expected = (selected_count, 3)
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def _validated_outputs(
    leaf_ids: np.ndarray,
    cell_lower: np.ndarray,
    cell_upper: np.ndarray,
    selected_count: int,
    inputs: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs = (
        _require_output_vector("leaf_ids", leaf_ids, selected_count),
        _require_output_boxes("cell_lower", cell_lower, selected_count),
        _require_output_boxes("cell_upper", cell_upper, selected_count),
    )
    for index, output in enumerate(outputs):
        if any(np.shares_memory(output, value) for value in inputs):
            raise ValueError("region selection outputs must not overlap inputs")
        if any(np.shares_memory(output, prior) for prior in outputs[:index]):
            raise ValueError("region selection outputs must not overlap each other")
    return outputs


def count_refined_region_windows(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    region_lower: np.ndarray,
    region_upper: np.ndarray,
) -> int:
    """Return the number of nonempty refined cell-center ROI windows."""
    _, _, selected_count = _validated_region_inputs(
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        region_lower,
        region_upper,
    )
    return selected_count


def fill_refined_region_windows(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    region_lower: np.ndarray,
    region_upper: np.ndarray,
    leaf_ids: np.ndarray,
    cell_lower: np.ndarray,
    cell_upper: np.ndarray,
) -> None:
    """Fill exact ascending leaf IDs and local windows after full preflight."""
    inputs, base_spacing, selected_count = _validated_region_inputs(
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        region_lower,
        region_upper,
    )
    leaf_ids, cell_lower, cell_upper = _validated_outputs(
        leaf_ids,
        cell_lower,
        cell_upper,
        selected_count,
        inputs,
    )
    fill_refined_region_windows_unchecked(
        inputs[0],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        inputs[8],
        inputs[9],
        base_spacing,
        leaf_ids,
        cell_lower,
        cell_upper,
    )


def refined_region_windows(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    region_lower: np.ndarray,
    region_upper: np.ndarray,
) -> RefinedRegionSelection:
    """Allocate and return the exact sparse refined cell-center ROI plan."""
    inputs, base_spacing, selected_count = _validated_region_inputs(
        domain_lower,
        domain_upper,
        root_shape,
        domain_cell_counts,
        block_cell_counts,
        node_levels,
        node_coords,
        leaf_node_ids,
        region_lower,
        region_upper,
    )
    leaf_ids = np.empty(selected_count, dtype=np.int64)
    cell_lower = np.empty((selected_count, 3), dtype=np.int64)
    cell_upper = np.empty((selected_count, 3), dtype=np.int64)
    fill_refined_region_windows_unchecked(
        inputs[0],
        inputs[3],
        inputs[4],
        inputs[5],
        inputs[6],
        inputs[7],
        inputs[8],
        inputs[9],
        base_spacing,
        leaf_ids,
        cell_lower,
        cell_upper,
    )
    return RefinedRegionSelection(leaf_ids, cell_lower, cell_upper)
