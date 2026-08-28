"""Independent Python reference and consumer for HPL-001."""

from __future__ import annotations

import numpy as np


def level1_halo_relation_plan_reference(
    block_ids: np.ndarray,
    primary_count: int,
    face_neighbor_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected_slots = {
        int(block_id): slot for slot, block_id in enumerate(block_ids)
    }
    source_slots = np.full((primary_count, 27), -1, dtype=np.int64)
    physical_masks = np.zeros((primary_count, 27), dtype=np.uint8)
    for primary in range(primary_count):
        primary_id = int(block_ids[primary])
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    column = (dz + 1) * 9 + (dy + 1) * 3 + dx + 1
                    current = primary_id
                    has_sibling = False
                    for axis, direction in enumerate((dx, dy, dz)):
                        if direction == 0:
                            continue
                        face = 2 * axis + (1 if direction > 0 else 0)
                        neighbor = int(face_neighbor_ids[current, face])
                        if neighbor < 0:
                            physical_masks[primary, column] |= np.uint8(1 << axis)
                        else:
                            current = neighbor
                            has_sibling = True
                    if has_sibling:
                        source_slots[primary, column] = selected_slots[current]
    return source_slots, physical_masks


def fill_same_level_halos_from_plan_reference(
    payload: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    source_slots: np.ndarray,
    physical_masks: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> None:
    """Apply a valid HPL plan as an independent immediate consumer."""
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
                            source[axis] = (
                                int(interior_upper[axis])
                                - int(interior_lower[axis])
                                + target[axis]
                            )
                        else:
                            source[axis] = (
                                int(interior_lower[axis])
                                + target[axis]
                                - int(interior_upper[axis])
                            )
                        continue
                    mode = int(boundary_modes[field, face])
                    if mode in (1, 2):
                        if step < 0:
                            source[axis] = (
                                2 * int(interior_lower[axis]) - target[axis] - 1
                            )
                        else:
                            source[axis] = (
                                2 * int(interior_upper[axis]) - target[axis] - 1
                            )
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
