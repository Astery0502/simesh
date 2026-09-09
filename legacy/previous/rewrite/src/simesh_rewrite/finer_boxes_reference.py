"""Independent integer reference for FRP-001."""

from __future__ import annotations

import numpy as np


def finer_restriction_boxes_reference(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    reduced_directions: np.ndarray,
    phase_codes: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate exact active-FINER target and source boxes."""
    placed_lower = np.empty_like(reduced_directions)
    placed_upper = np.empty_like(reduced_directions)
    source_lower = np.empty_like(reduced_directions)
    source_upper = np.empty_like(reduced_directions)
    for row, direction in enumerate(reduced_directions):
        phase = int(phase_codes[row])
        for axis in range(3):
            lower = int(interior_lower[axis])
            upper = int(interior_upper[axis])
            extent = upper - lower
            component = int(direction[axis])
            bit = (phase >> axis) & 1
            if component == 0:
                half = extent // 2
                placed_lower[row, axis] = lower + bit * half
                placed_upper[row, axis] = lower + (bit + 1) * half
                source_lower[row, axis] = lower
                source_upper[row, axis] = upper
            elif component < 0:
                width = int(target_upper[row, axis] - target_lower[row, axis])
                placed_lower[row, axis] = target_lower[row, axis]
                placed_upper[row, axis] = target_upper[row, axis]
                source_lower[row, axis] = upper - 2 * width
                source_upper[row, axis] = upper
            else:
                width = int(target_upper[row, axis] - target_lower[row, axis])
                placed_lower[row, axis] = target_lower[row, axis]
                placed_upper[row, axis] = target_upper[row, axis]
                source_lower[row, axis] = lower
                source_upper[row, axis] = lower + 2 * width
    return placed_lower, placed_upper, source_lower, source_upper
