"""Independent direct-cell reference for HAX-001."""

from __future__ import annotations

import numpy as np


def apply_level1_same_level_halo_plan_reference(
    payload: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    source_slots: np.ndarray,
    physical_masks: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> None:
    for primary in range(source_slots.shape[0]):
        for field in range(payload.shape[1]):
            for target in np.ndindex(payload.shape[2:]):
                direction = [0, 0, 0]
                for axis in range(3):
                    if target[axis] < interior_lower[axis]:
                        direction[axis] = -1
                    elif target[axis] >= interior_upper[axis]:
                        direction[axis] = 1
                column = (
                    (direction[2] + 1) * 9
                    + (direction[1] + 1) * 3
                    + direction[0]
                    + 1
                )
                source_slot = int(source_slots[primary, column])
                if source_slot < 0:
                    continue
                mask = int(physical_masks[primary, column])
                source = list(target)
                physical_operations: list[tuple[int, int, int]] = []
                for axis, step in enumerate(direction):
                    if step == 0:
                        continue
                    face = 2 * axis + (1 if step > 0 else 0)
                    if (mask & (1 << axis)) == 0:
                        if step < 0:
                            layer = int(interior_lower[axis]) - target[axis]
                            source[axis] = int(interior_upper[axis]) - layer
                        else:
                            offset = target[axis] - int(interior_upper[axis])
                            source[axis] = int(interior_lower[axis]) + offset
                        continue
                    mode = int(boundary_modes[field, face])
                    if mode in (1, 2):
                        if step < 0:
                            layer = int(interior_lower[axis]) - target[axis]
                            source[axis] = int(interior_lower[axis]) + layer - 1
                        else:
                            layer = target[axis] - int(interior_upper[axis]) + 1
                            source[axis] = int(interior_upper[axis]) - layer
                    elif step < 0:
                        source[axis] = int(interior_lower[axis])
                    else:
                        source[axis] = int(interior_upper[axis]) - 1
                    physical_operations.append((axis, face, mode))

                value = payload[(source_slot, field, *source)]
                for axis, face, mode in physical_operations:
                    if mode == 2:
                        value = -value
                    elif mode == 3 and field == normal_field_slots[axis]:
                        if face % 2 == 0 and value > 0.0:
                            value = np.float64(0.0)
                        elif face % 2 == 1 and value < 0.0:
                            value = np.float64(0.0)
                payload[(primary, field, *target)] = value
