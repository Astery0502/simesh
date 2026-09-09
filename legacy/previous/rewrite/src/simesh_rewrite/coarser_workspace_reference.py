"""Independent integer reference for CWP-001."""

from __future__ import annotations

import numpy as np


def coarser_workspace_boxes_reference(
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    reduced_directions: np.ndarray,
    phase_codes: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Allocate normalized COARSER source/workspace geometry."""
    outputs = tuple(np.empty_like(reduced_directions) for _ in range(7))
    for row, direction in enumerate(reduced_directions):
        if any(target_lower[row, axis] == target_upper[row, axis] for axis in range(3)):
            for output in outputs:
                output[row] = 0
            continue
        for axis in range(3):
            lower = int(interior_lower[axis])
            upper = int(interior_upper[axis])
            extent = upper - lower
            component = int(direction[axis])
            phase_bit = (int(phase_codes[row]) >> axis) & 1
            delta = (phase_bit + component) // 2
            logical_origin = lower + phase_bit * (extent // 2) - delta * extent
            center_lower = logical_origin + (
                int(target_lower[row, axis]) - lower
            ) // 2
            center_upper = logical_origin + (
                int(target_upper[row, axis]) - 1 - lower
            ) // 2 + 1
            required_lower = center_lower - 1
            required_upper = center_upper + 1
            source_lower = max(required_lower, lower)
            source_upper = min(required_upper, upper)
            base = min(required_lower, logical_origin)
            outputs[0][row, axis] = source_lower
            outputs[1][row, axis] = source_upper
            outputs[2][row, axis] = source_lower - base
            outputs[3][row, axis] = source_upper - base
            outputs[4][row, axis] = required_lower - base
            outputs[5][row, axis] = required_upper - base
            outputs[6][row, axis] = logical_origin - base
    return outputs
