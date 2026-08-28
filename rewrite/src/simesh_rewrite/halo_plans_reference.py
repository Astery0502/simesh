"""Independent Python reference for HPL-001."""

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
