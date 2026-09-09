"""Exact-rational independent reference for GEO-001."""

from __future__ import annotations

from fractions import Fraction

import numpy as np


def level1_block_geometry_reference(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    rank_to_coord: np.ndarray,
    block_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    spacing_fraction = [
        (
            Fraction.from_float(float(domain_upper[axis]))
            - Fraction.from_float(float(domain_lower[axis]))
        )
        / int(domain_cell_counts[axis])
        for axis in range(3)
    ]
    spacing = np.asarray([float(value) for value in spacing_fraction], dtype=np.float64)
    bounds = np.empty((block_ids.shape[0], 2, 3), dtype=np.float64)
    for slot, block_id in enumerate(block_ids):
        coordinate = rank_to_coord[int(block_id)]
        for axis in range(3):
            lower_index = int(coordinate[axis]) * int(block_cell_counts[axis])
            upper_index = lower_index + int(block_cell_counts[axis])
            lower_fraction = Fraction.from_float(float(domain_lower[axis]))
            for side, face_index in enumerate((lower_index, upper_index)):
                if face_index == 0:
                    value = float(domain_lower[axis])
                elif face_index == int(domain_cell_counts[axis]):
                    value = float(domain_upper[axis])
                else:
                    value = float(
                        lower_fraction + face_index * spacing_fraction[axis]
                    )
                bounds[slot, side, axis] = value
    return bounds, spacing


def cell_center_reference(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    domain_cell_counts: np.ndarray,
    global_cell_index: np.ndarray,
) -> np.ndarray:
    result = np.empty(3, dtype=np.float64)
    for axis in range(3):
        lower = Fraction.from_float(float(domain_lower[axis]))
        extent = (
            Fraction.from_float(float(domain_upper[axis]))
            - lower
        )
        spacing = extent / int(domain_cell_counts[axis])
        result[axis] = float(
            lower + (Fraction(int(global_cell_index[axis])) + Fraction(1, 2)) * spacing
        )
    return result
