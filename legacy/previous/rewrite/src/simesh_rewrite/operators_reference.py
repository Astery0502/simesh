"""Independent references for concrete rewrite operators."""

from __future__ import annotations

import numpy as np


def scaled_difference_into_reference(
    source: np.ndarray,
    source_valid_lower: np.ndarray,
    source_valid_upper: np.ndarray,
    left_field_position: int,
    right_field_position: int,
    scale: float,
    destination: np.ndarray,
    destination_field_position: int,
    destination_lower: np.ndarray,
) -> None:
    extent = source_valid_upper - source_valid_lower
    source_region = tuple(
        slice(int(lower), int(upper))
        for lower, upper in zip(
            source_valid_lower,
            source_valid_upper,
            strict=True,
        )
    )
    destination_region = tuple(
        slice(int(lower), int(lower + size))
        for lower, size in zip(destination_lower, extent, strict=True)
    )
    product = np.multiply(
        source[(slice(None), right_field_position, *source_region)],
        np.float64(scale),
    )
    np.subtract(
        source[(slice(None), left_field_position, *source_region)],
        product,
        out=destination[
            (slice(None), destination_field_position, *destination_region)
        ],
    )


def central_difference_into_reference(
    source: np.ndarray,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    source_field_position: int,
    axis: int,
    cell_spacing: np.ndarray,
    destination: np.ndarray,
    destination_field_position: int,
    destination_lower: np.ndarray,
) -> None:
    extent = output_upper - output_lower
    destination_region = tuple(
        slice(int(lower), int(lower + size))
        for lower, size in zip(destination_lower, extent, strict=True)
    )
    lower_neighbor = output_lower.copy()
    upper_neighbor = output_upper.copy()
    lower_neighbor[axis] -= 1
    upper_neighbor[axis] -= 1
    lower_region = [
        slice(int(lower), int(upper))
        for lower, upper in zip(
            lower_neighbor,
            upper_neighbor,
            strict=True,
        )
    ]
    upper_region = list(lower_region)
    lower_region[axis] = slice(
        int(output_lower[axis] - 1),
        int(output_upper[axis] - 1),
    )
    upper_region[axis] = slice(
        int(output_lower[axis] + 1),
        int(output_upper[axis] + 1),
    )
    difference = np.subtract(
        source[(slice(None), source_field_position, *upper_region)],
        source[(slice(None), source_field_position, *lower_region)],
    )
    inverse_two_spacing = np.float64(0.5) / cell_spacing[axis]
    np.multiply(
        difference,
        inverse_two_spacing,
        out=destination[
            (slice(None), destination_field_position, *destination_region)
        ],
    )
