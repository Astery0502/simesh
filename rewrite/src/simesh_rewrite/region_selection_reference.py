"""Independent exhaustive leaf/cell reference for ROI-001."""

from __future__ import annotations

import math

import numpy as np


def _center(
    lower: float,
    spacing: float,
    global_cell: int,
) -> float:
    factor = float(global_cell) + 0.5
    offset = factor * spacing
    return lower + offset


def refined_region_windows_reference(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate every canonical local cell rather than using lower bounds.

    Inputs are expected to satisfy the ROI-001 representation contract.  The
    reference deliberately does not call production validation or search code.
    """
    base_spacing = np.empty(3, dtype=np.float64)
    for axis in range(3):
        extent = float(domain_upper[axis]) - float(domain_lower[axis])
        base_spacing[axis] = extent / float(domain_cell_counts[axis])

    selected_ids: list[int] = []
    selected_lower: list[tuple[int, int, int]] = []
    selected_upper: list[tuple[int, int, int]] = []
    for leaf_id, node_value in enumerate(leaf_node_ids):
        node_id = int(node_value)
        shift = int(node_levels[node_id]) - 1
        lowers: list[int] = []
        uppers: list[int] = []
        for axis in range(3):
            spacing = math.ldexp(float(base_spacing[axis]), -shift)
            first_global = (
                int(node_coords[node_id, axis])
                * int(block_cell_counts[axis])
            )
            included = [
                local
                for local in range(int(block_cell_counts[axis]))
                if float(region_lower[axis])
                <= _center(
                    float(domain_lower[axis]),
                    spacing,
                    first_global + local,
                )
                < float(region_upper[axis])
            ]
            if included:
                lowers.append(included[0])
                uppers.append(included[-1] + 1)
            else:
                lowers.append(0)
                uppers.append(0)
        if all(lower < upper for lower, upper in zip(lowers, uppers, strict=True)):
            selected_ids.append(leaf_id)
            selected_lower.append(tuple(lowers))
            selected_upper.append(tuple(uppers))

    return (
        np.asarray(selected_ids, dtype=np.int64),
        np.asarray(selected_lower, dtype=np.int64).reshape(-1, 3),
        np.asarray(selected_upper, dtype=np.int64).reshape(-1, 3),
    )
