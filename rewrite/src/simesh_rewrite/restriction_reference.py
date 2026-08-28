"""Independent scalar fixed-order reference for RST-001."""

from __future__ import annotations

import numpy as np


def restrict_cartesian_2to1_reference(
    fine_payload: np.ndarray,
    fine_lower: np.ndarray,
    fine_upper: np.ndarray,
    coarse_payload: np.ndarray,
    coarse_lower: np.ndarray,
) -> None:
    """Apply the exact eight-value Cartesian restriction arithmetic."""
    coarse_extent = tuple(
        (int(fine_upper[axis]) - int(fine_lower[axis])) // 2
        for axis in range(3)
    )
    factor = np.float64(0.125)
    for slot in range(fine_payload.shape[0]):
        for field in range(fine_payload.shape[1]):
            for qx in range(coarse_extent[0]):
                i = int(fine_lower[0]) + 2 * qx
                I = int(coarse_lower[0]) + qx
                for qy in range(coarse_extent[1]):
                    j = int(fine_lower[1]) + 2 * qy
                    J = int(coarse_lower[1]) + qy
                    for qz in range(coarse_extent[2]):
                        k = int(fine_lower[2]) + 2 * qz
                        K = int(coarse_lower[2]) + qz
                        total = fine_payload[slot, field, i, j, k]
                        total = total + fine_payload[slot, field, i + 1, j, k]
                        total = total + fine_payload[slot, field, i, j + 1, k]
                        total = total + fine_payload[
                            slot, field, i + 1, j + 1, k
                        ]
                        total = total + fine_payload[slot, field, i, j, k + 1]
                        total = total + fine_payload[
                            slot, field, i + 1, j, k + 1
                        ]
                        total = total + fine_payload[
                            slot, field, i, j + 1, k + 1
                        ]
                        total = total + fine_payload[
                            slot, field, i + 1, j + 1, k + 1
                        ]
                        coarse_payload[slot, field, I, J, K] = total * factor
