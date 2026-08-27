"""Independent direct-cell reference for HAL-001."""

from __future__ import annotations

import numpy as np


def fill_physical_halos_reference(
    payload: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    block_ids: np.ndarray,
    face_neighbor_ids: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> None:
    for slot, block_id in enumerate(block_ids):
        for field in range(payload.shape[1]):
            for target in np.ndindex(payload.shape[2:]):
                source = list(target)
                operations = []
                valid = True
                for axis in range(3):
                    if target[axis] < interior_lower[axis]:
                        face = 2 * axis
                    elif target[axis] >= interior_upper[axis]:
                        face = 2 * axis + 1
                    else:
                        continue
                    if face_neighbor_ids[int(block_id), face] != -1:
                        valid = False
                        break
                    mode = int(boundary_modes[field, face])
                    if mode in (1, 2):
                        if face % 2 == 0:
                            source[axis] = 2 * int(interior_lower[axis]) - target[axis] - 1
                        else:
                            source[axis] = 2 * int(interior_upper[axis]) - target[axis] - 1
                    elif face % 2 == 0:
                        source[axis] = int(interior_lower[axis])
                    else:
                        source[axis] = int(interior_upper[axis]) - 1
                    operations.append((axis, face, mode))
                if not operations or not valid:
                    continue
                value = payload[(slot, field, *source)]
                for axis, face, mode in operations:
                    if mode == 2:
                        value = -value
                    elif mode == 3 and field == normal_field_slots[axis]:
                        if face % 2 == 0 and value > 0.0:
                            value = np.float64(0.0)
                        elif face % 2 == 1 and value < 0.0:
                            value = np.float64(0.0)
                payload[(slot, field, *target)] = value


def fill_same_level_halos_reference(
    payload: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    block_ids: np.ndarray,
    primary_count: int,
    face_neighbor_ids: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> None:
    for primary in range(primary_count):
        block_id = int(block_ids[primary])
        for field in range(payload.shape[1]):
            for target in np.ndindex(payload.shape[2:]):
                source = list(target)
                displacement = [0, 0, 0]
                physical_operations = []
                has_sibling = False
                for axis in range(3):
                    if target[axis] < interior_lower[axis]:
                        face = 2 * axis
                    elif target[axis] >= interior_upper[axis]:
                        face = 2 * axis + 1
                    else:
                        continue
                    if face_neighbor_ids[block_id, face] >= 0:
                        has_sibling = True
                        if face % 2 == 0:
                            displacement[axis] = -1
                            source[axis] = (
                                int(interior_upper[axis])
                                - int(interior_lower[axis])
                                + target[axis]
                            )
                        else:
                            displacement[axis] = 1
                            source[axis] = (
                                int(interior_lower[axis])
                                + target[axis]
                                - int(interior_upper[axis])
                            )
                        continue

                    mode = int(boundary_modes[field, face])
                    if mode in (1, 2):
                        if face % 2 == 0:
                            source[axis] = (
                                2 * int(interior_lower[axis]) - target[axis] - 1
                            )
                        else:
                            source[axis] = (
                                2 * int(interior_upper[axis]) - target[axis] - 1
                            )
                    elif face % 2 == 0:
                        source[axis] = int(interior_lower[axis])
                    else:
                        source[axis] = int(interior_upper[axis]) - 1
                    physical_operations.append((axis, face, mode))

                if not has_sibling:
                    continue

                source_block = block_id
                for axis, delta in enumerate(displacement):
                    if delta:
                        source_block = int(
                            face_neighbor_ids[
                                source_block,
                                2 * axis + (1 if delta > 0 else 0),
                            ]
                        )
                source_slot = next(
                    slot
                    for slot, selected_id in enumerate(block_ids)
                    if int(selected_id) == source_block
                )
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
