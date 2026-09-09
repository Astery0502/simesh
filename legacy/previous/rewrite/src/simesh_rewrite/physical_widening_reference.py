"""Independent direct-cell reference for PWA-001."""

from __future__ import annotations

import itertools

import numpy as np

from .boundary_rules_reference import (
    physical_halo_source_index_reference,
    transform_physical_halo_value_reference,
)


def apply_cartesian_physical_widening_reference(
    payload: np.ndarray,
    target_slot: int,
    logical_interior_lower: np.ndarray,
    logical_interior_upper: np.ndarray,
    storage_logical_offsets: np.ndarray,
    directions: np.ndarray,
    physical_masks: np.ndarray,
    base_lower: np.ndarray,
    base_upper: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> None:
    """Apply the exact PWA mapping without production validation helpers."""
    del base_lower, base_upper
    with np.errstate(invalid="ignore"):
        for row in range(directions.shape[0]):
            mask = int(physical_masks[row])
            for field in range(payload.shape[1]):
                for target in itertools.product(
                    *(
                        range(
                            int(target_lower[row, axis]),
                            int(target_upper[row, axis]),
                        )
                        for axis in range(3)
                    )
                ):
                    source = list(target)
                    operations: list[tuple[int, int, int]] = []
                    for axis in range(3):
                        if (mask & (1 << axis)) == 0:
                            continue
                        face = 2 * axis + (
                            1 if int(directions[row, axis]) > 0 else 0
                        )
                        mode = int(boundary_modes[field, face])
                        logical_target = (
                            target[axis]
                            + int(storage_logical_offsets[row, axis])
                        )
                        logical_source = physical_halo_source_index_reference(
                            logical_target,
                            int(logical_interior_lower[row, axis]),
                            int(logical_interior_upper[row, axis]),
                            face,
                            mode,
                        )
                        source[axis] = (
                            logical_source
                            - int(storage_logical_offsets[row, axis])
                        )
                        operations.append((axis, face, mode))

                    value = payload[(target_slot, field, *source)]
                    for axis, face, mode in operations:
                        value = transform_physical_halo_value_reference(
                            value,
                            field,
                            int(normal_field_slots[axis]),
                            face,
                            mode,
                        )
                    payload[(target_slot, field, *target)] = value
