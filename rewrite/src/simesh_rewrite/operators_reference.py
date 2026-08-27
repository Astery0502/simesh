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
