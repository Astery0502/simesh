"""Independent Python-integer reference for SLB-001."""

from __future__ import annotations

import numpy as np


def same_level_source_boxes_reference(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    reduced_directions: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate exact ratio-one source boxes for explicit target rows."""
    widths = [
        int(interior_upper[axis]) - int(interior_lower[axis])
        for axis in range(3)
    ]
    lower_rows: list[list[int]] = []
    upper_rows: list[list[int]] = []
    for row, direction in enumerate(reduced_directions):
        source_lower: list[int] = []
        source_upper: list[int] = []
        for axis in range(3):
            offset = int(direction[axis]) * widths[axis]
            source_lower.append(int(target_lower[row, axis]) - offset)
            source_upper.append(int(target_upper[row, axis]) - offset)
        lower_rows.append(source_lower)
        upper_rows.append(source_upper)

    row_count = int(reduced_directions.shape[0])
    return (
        np.asarray(lower_rows, dtype=np.int64).reshape(row_count, 3),
        np.asarray(upper_rows, dtype=np.int64).reshape(row_count, 3),
    )
