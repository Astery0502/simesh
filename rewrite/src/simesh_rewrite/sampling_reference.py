"""Independent references for the SAM capabilities."""

from __future__ import annotations

import numpy as np


def place_level1_blocks_reference(
    payload: np.ndarray,
    payload_valid_lower: np.ndarray,
    block_ids: np.ndarray,
    block_cell_counts: np.ndarray,
    rank_to_coord: np.ndarray,
    uniform_grid: np.ndarray,
) -> None:
    source = tuple(
        slice(int(lower), int(lower + extent))
        for lower, extent in zip(
            payload_valid_lower,
            block_cell_counts,
            strict=True,
        )
    )
    for slot, block_id in enumerate(block_ids):
        coordinate = rank_to_coord[int(block_id)]
        global_lower = coordinate * block_cell_counts
        destination = tuple(
            slice(int(lower), int(lower + extent))
            for lower, extent in zip(
                global_lower,
                block_cell_counts,
                strict=True,
            )
        )
        uniform_grid[(slice(None), *destination)] = payload[
            (slot, slice(None), *source)
        ]


def _canonical_face_reference(
    domain_lower: float,
    domain_upper: float,
    native_spacing: float,
    face_index: int,
    domain_cells: int,
) -> float:
    if face_index == 0:
        return domain_lower
    if face_index == domain_cells:
        return domain_upper
    return float(np.float64(domain_lower) + np.float64(face_index) * native_spacing)


def _source_cell_reference(
    center: float,
    domain_lower: float,
    domain_upper: float,
    native_spacing: float,
    domain_cells: int,
) -> int:
    lower = 0
    upper = domain_cells - 1
    while lower < upper:
        middle = lower + (upper - lower + 1) // 2
        if _canonical_face_reference(
            domain_lower,
            domain_upper,
            native_spacing,
            middle,
            domain_cells,
        ) <= center:
            lower = middle
        else:
            upper = middle - 1
    return lower


def sample_level1_zero_order_reference(
    payload: np.ndarray,
    payload_valid_lower: np.ndarray,
    block_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    coord_to_rank: np.ndarray,
    sample_lower: np.ndarray,
    sample_upper: np.ndarray,
    uniform_grid: np.ndarray,
) -> None:
    native_spacing = (domain_upper - domain_lower) / domain_cell_counts
    output_counts = np.asarray(uniform_grid.shape[1:], dtype=np.int64)
    output_spacing = (sample_upper - sample_lower) / output_counts
    for slot, block_id in enumerate(block_ids):
        for output_index in np.ndindex(uniform_grid.shape[1:]):
            global_index = []
            for axis in range(3):
                factor = np.float64(output_index[axis]) + np.float64(0.5)
                center = float(
                    np.float64(sample_lower[axis])
                    + factor * np.float64(output_spacing[axis])
                )
                global_index.append(
                    _source_cell_reference(
                        center,
                        float(domain_lower[axis]),
                        float(domain_upper[axis]),
                        float(native_spacing[axis]),
                        int(domain_cell_counts[axis]),
                    )
                )
            block_coordinate = tuple(
                global_index[axis] // int(block_cell_counts[axis])
                for axis in range(3)
            )
            if int(coord_to_rank[block_coordinate]) != int(block_id):
                continue
            local_index = tuple(
                global_index[axis] % int(block_cell_counts[axis])
                for axis in range(3)
            )
            source_index = tuple(
                int(payload_valid_lower[axis]) + local_index[axis]
                for axis in range(3)
            )
            uniform_grid[(slice(None), *output_index)] = payload[
                (slot, slice(None), *source_index)
            ]


def _lerp_reference(left: np.float64, right: np.float64, weight: float) -> np.float64:
    one_minus = np.float64(1.0) - np.float64(weight)
    left_term = np.float64(left) * one_minus
    right_term = np.float64(right) * np.float64(weight)
    return np.float64(left_term + right_term)


def sample_level1_trilinear_reference(
    payload: np.ndarray,
    interior_lower: np.ndarray,
    block_ids: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    coord_to_rank: np.ndarray,
    rank_to_coord: np.ndarray,
    sample_lower: np.ndarray,
    sample_upper: np.ndarray,
    uniform_grid: np.ndarray,
) -> None:
    native_spacing = (domain_upper - domain_lower) / domain_cell_counts
    output_counts = np.asarray(uniform_grid.shape[1:], dtype=np.int64)
    output_spacing = (sample_upper - sample_lower) / output_counts
    for slot, block_id in enumerate(block_ids):
        block_coordinate = rank_to_coord[int(block_id)]
        block_global_lower = block_coordinate * block_cell_counts
        block_lower = np.asarray(
            [
                _canonical_face_reference(
                    float(domain_lower[axis]),
                    float(domain_upper[axis]),
                    float(native_spacing[axis]),
                    int(block_global_lower[axis]),
                    int(domain_cell_counts[axis]),
                )
                for axis in range(3)
            ]
        )
        for output_index in np.ndindex(uniform_grid.shape[1:]):
            global_index = []
            center = []
            for axis in range(3):
                factor = np.float64(output_index[axis]) + np.float64(0.5)
                center.append(
                    float(
                        np.float64(sample_lower[axis])
                        + factor * np.float64(output_spacing[axis])
                    )
                )
                global_index.append(
                    _source_cell_reference(
                        center[axis],
                        float(domain_lower[axis]),
                        float(domain_upper[axis]),
                        float(native_spacing[axis]),
                        int(domain_cell_counts[axis]),
                    )
                )
            owner_coordinate = tuple(
                global_index[axis] // int(block_cell_counts[axis])
                for axis in range(3)
            )
            if int(coord_to_rank[owner_coordinate]) != int(block_id):
                continue

            left = []
            weight = []
            for axis in range(3):
                delta = np.float64(center[axis]) - np.float64(block_lower[axis])
                ratio = delta / np.float64(native_spacing[axis])
                normalized = ratio - np.float64(0.5)
                left_index = int(np.floor(normalized))
                left.append(int(interior_lower[axis]) + left_index)
                weight.append(float(normalized - np.float64(left_index)))
            right = [value + 1 for value in left]

            for field in range(payload.shape[1]):
                c00 = _lerp_reference(
                    payload[slot, field, left[0], left[1], left[2]],
                    payload[slot, field, left[0], left[1], right[2]],
                    weight[2],
                )
                c01 = _lerp_reference(
                    payload[slot, field, left[0], right[1], left[2]],
                    payload[slot, field, left[0], right[1], right[2]],
                    weight[2],
                )
                c10 = _lerp_reference(
                    payload[slot, field, right[0], left[1], left[2]],
                    payload[slot, field, right[0], left[1], right[2]],
                    weight[2],
                )
                c11 = _lerp_reference(
                    payload[slot, field, right[0], right[1], left[2]],
                    payload[slot, field, right[0], right[1], right[2]],
                    weight[2],
                )
                c0 = _lerp_reference(c00, c01, weight[1])
                c1 = _lerp_reference(c10, c11, weight[1])
                uniform_grid[(field, *output_index)] = _lerp_reference(
                    c0,
                    c1,
                    weight[0],
                )
