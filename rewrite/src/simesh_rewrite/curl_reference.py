"""Independent scalar reference for OPR-003."""

from __future__ import annotations

import numpy as np


def _derivative(
    plus_value: np.float64,
    minus_value: np.float64,
    spacing: np.float64,
) -> np.float64:
    inverse = np.float64(0.5) / np.float64(spacing)
    difference = np.float64(plus_value) - np.float64(minus_value)
    return np.float64(difference * inverse)


def _difference(positive: np.float64, negative: np.float64) -> np.float64:
    product = np.float64(1.0) * np.float64(negative)
    return np.float64(np.float64(positive) - product)


def cartesian_curl_reference(
    source: np.ndarray,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    source_field_positions: np.ndarray,
    slot_cell_spacing: np.ndarray,
    destination: np.ndarray,
    destination_field_positions: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    """Apply the contracted scalar curl tree to one translated common box."""
    bx, by, bz = (int(value) for value in source_field_positions)
    jx, jy, jz = (int(value) for value in destination_field_positions)
    for slot in range(source.shape[0]):
        spacing = slot_cell_spacing[slot]
        for i in range(int(output_lower[0]), int(output_upper[0])):
            di = int(destination_lower[0]) + i - int(output_lower[0])
            for j in range(int(output_lower[1]), int(output_upper[1])):
                dj = int(destination_lower[1]) + j - int(output_lower[1])
                for k in range(int(output_lower[2]), int(output_upper[2])):
                    dk = int(destination_lower[2]) + k - int(output_lower[2])
                    positive = _derivative(
                        source[slot, bz, i, j + 1, k],
                        source[slot, bz, i, j - 1, k],
                        spacing[1],
                    )
                    negative = _derivative(
                        source[slot, by, i, j, k + 1],
                        source[slot, by, i, j, k - 1],
                        spacing[2],
                    )
                    destination[slot, jx, di, dj, dk] = _difference(
                        positive, negative
                    )

                    positive = _derivative(
                        source[slot, bx, i, j, k + 1],
                        source[slot, bx, i, j, k - 1],
                        spacing[2],
                    )
                    negative = _derivative(
                        source[slot, bz, i + 1, j, k],
                        source[slot, bz, i - 1, j, k],
                        spacing[0],
                    )
                    destination[slot, jy, di, dj, dk] = _difference(
                        positive, negative
                    )

                    positive = _derivative(
                        source[slot, by, i + 1, j, k],
                        source[slot, by, i - 1, j, k],
                        spacing[0],
                    )
                    negative = _derivative(
                        source[slot, bx, i, j + 1, k],
                        source[slot, bx, i, j - 1, k],
                        spacing[1],
                    )
                    destination[slot, jz, di, dj, dk] = _difference(
                        positive, negative
                    )
