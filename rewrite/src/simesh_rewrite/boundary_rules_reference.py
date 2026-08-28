"""Independent Python reference for PBC-001 scalar rules."""

from __future__ import annotations

import numpy as np


def physical_halo_source_index_reference(
    target: int,
    lower: int,
    upper: int,
    face: int,
    mode: int,
) -> int:
    if face % 2 == 0:
        layer = lower - target
        return lower + layer - 1 if mode in (1, 2) else lower
    layer = target - upper + 1
    return upper - layer if mode in (1, 2) else upper - 1


def transform_physical_halo_value_reference(
    value: np.float64,
    field_position: int,
    normal_field_slot: int,
    face: int,
    mode: int,
) -> np.float64:
    if mode == 2:
        return np.negative(value)
    if mode == 3 and field_position == normal_field_slot:
        if face % 2 == 0 and value > 0.0:
            return np.float64(0.0)
        if face % 2 == 1 and value < 0.0:
            return np.float64(0.0)
    return value
