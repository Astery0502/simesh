"""Independent per-axis Python reference for TGT-001."""

from __future__ import annotations

import numpy as np


def directed_halo_target_boxes_reference(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    requested_lower: np.ndarray,
    requested_upper: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate the exact target box for every explicit direction row."""
    lower_rows: list[list[int]] = []
    upper_rows: list[list[int]] = []
    for direction in directions:
        lower_row: list[int] = []
        upper_row: list[int] = []
        for axis in range(3):
            component = int(direction[axis])
            if component < 0:
                lower_row.append(int(requested_lower[axis]))
                upper_row.append(int(interior_lower[axis]))
            elif component == 0:
                lower_row.append(int(interior_lower[axis]))
                upper_row.append(int(interior_upper[axis]))
            else:
                lower_row.append(int(interior_upper[axis]))
                upper_row.append(int(requested_upper[axis]))
        lower_rows.append(lower_row)
        upper_rows.append(upper_row)

    direction_count = int(directions.shape[0])
    return (
        np.asarray(lower_rows, dtype=np.int64).reshape(direction_count, 3),
        np.asarray(upper_rows, dtype=np.int64).reshape(direction_count, 3),
    )
