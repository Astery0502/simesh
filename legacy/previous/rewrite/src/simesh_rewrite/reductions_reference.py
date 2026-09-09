"""Independent references for streaming rewrite reductions."""

from __future__ import annotations

import numpy as np


def accumulate_field_sum_reference(
    payload: np.ndarray,
    valid_lower: np.ndarray,
    valid_upper: np.ndarray,
    field_position: int,
    accumulator: np.ndarray,
) -> None:
    total = np.float64(accumulator[0])
    for slot in range(payload.shape[0]):
        for i in range(int(valid_lower[0]), int(valid_upper[0])):
            for j in range(int(valid_lower[1]), int(valid_upper[1])):
                for k in range(int(valid_lower[2]), int(valid_upper[2])):
                    total = np.float64(
                        total + payload[slot, field_position, i, j, k]
                    )
    accumulator[0] = total


def merge_field_sums_reference(
    accumulator: np.ndarray,
    partial: np.ndarray,
) -> None:
    accumulator[0] = np.float64(accumulator[0] + partial[0])


def finalize_field_sum_reference(accumulator: np.ndarray) -> float:
    return float(accumulator[0])
