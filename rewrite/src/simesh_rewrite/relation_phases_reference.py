"""Independent Python-list and parity reference for RPH-001."""

from __future__ import annotations

import numpy as np


PHYSICAL = 1
COARSER = 2
SAME = 3
FINER = 4
NO_PHASE = np.uint8(255)


def refined_relation_phase_codes_reference(
    selected_leaf_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    directions: np.ndarray,
    relation_kinds: np.ndarray,
    physical_masks: np.ndarray,
    source_counts: np.ndarray,
    source_slots: np.ndarray,
) -> np.ndarray:
    """Allocate exact ratio-two phase codes from explicit accepted records."""
    selected = [int(value) for value in selected_leaf_ids]
    leaf_count = int(leaf_node_ids.shape[0])
    node_count = int(node_levels.shape[0])
    for slot, leaf_id in enumerate(selected):
        if leaf_id < 0 or leaf_id >= leaf_count:
            raise ValueError(f"selected leaf {slot} is out of range")
        for earlier in range(slot):
            if selected[earlier] == leaf_id:
                raise ValueError(f"selected leaf {slot} is duplicated")
        node = int(leaf_node_ids[leaf_id])
        if node < 0 or node >= node_count:
            raise ValueError(f"selected leaf {slot} has an invalid node")

    primary_count, direction_count = relation_kinds.shape
    if primary_count > len(selected):
        raise ValueError("accepted relation rows exceed selected slots")

    direction_rows: list[list[int]] = []
    for direction_index, direction in enumerate(directions):
        row = [int(direction[axis]) for axis in range(3)]
        if any(value < -1 or value > 1 for value in row):
            raise ValueError(f"direction {direction_index} is invalid")
        if row == [0, 0, 0]:
            raise ValueError(f"direction {direction_index} is center")
        direction_rows.append(row)

    output = np.full(
        (primary_count, direction_count, 4), NO_PHASE, dtype=np.uint8
    )
    for primary in range(primary_count):
        primary_leaf = selected[primary]
        primary_node = int(leaf_node_ids[primary_leaf])
        primary_level = int(node_levels[primary_node])
        primary_coord = [
            int(node_coords[primary_node, axis]) for axis in range(3)
        ]
        primary_phase = sum(
            (primary_coord[axis] & 1) << axis for axis in range(3)
        )

        for direction_index in range(direction_count):
            direction = direction_rows[direction_index]
            kind = int(relation_kinds[primary, direction_index])
            mask = int(physical_masks[primary, direction_index])
            count = int(source_counts[primary, direction_index])
            if count < 0 or count > 4:
                raise ValueError("source count exceeds four")
            if mask < 0 or mask > 7:
                raise ValueError("physical mask uses unsupported bits")
            for axis in range(3):
                if mask & (1 << axis) and direction[axis] == 0:
                    raise ValueError("physical mask marks a zero direction")
            reduced = [
                0 if mask & (1 << axis) else direction[axis]
                for axis in range(3)
            ]

            active_slots: list[int] = []
            for source in range(4):
                slot = int(source_slots[primary, direction_index, source])
                if source < count:
                    if slot < 0 or slot >= len(selected):
                        raise ValueError("active source slot is out of range")
                    active_slots.append(slot)
                elif slot != -1:
                    raise ValueError("inactive source slot is not -1")

            if kind == PHYSICAL:
                expected_mask = sum(
                    (1 << axis)
                    for axis in range(3)
                    if direction[axis] != 0
                )
                if mask != expected_mask or count != 0:
                    raise ValueError("invalid physical relation record")
                continue

            if kind not in (COARSER, SAME, FINER):
                raise ValueError("unknown relation kind")
            if reduced == [0, 0, 0]:
                raise ValueError("nonphysical relation has no active direction")

            if kind == SAME:
                if count != 1:
                    raise ValueError("same-level relation must have one source")
                source_leaf = selected[active_slots[0]]
                source_node = int(leaf_node_ids[source_leaf])
                if int(node_levels[source_node]) != primary_level:
                    raise ValueError("same-level source has the wrong level")
                continue

            if kind == COARSER:
                if count != 1:
                    raise ValueError("coarser relation must have one source")
                source_leaf = selected[active_slots[0]]
                source_node = int(leaf_node_ids[source_leaf])
                if int(node_levels[source_node]) != primary_level - 1:
                    raise ValueError("coarser source has the wrong level")
                for axis in range(3):
                    expected = (primary_coord[axis] + reduced[axis]) // 2
                    if int(node_coords[source_node, axis]) != expected:
                        raise ValueError("coarser source coordinate is incompatible")
                output[primary, direction_index, 0] = np.uint8(
                    primary_phase
                )
                continue

            neutral_axes = sum(
                direction[axis] == 0 or bool(mask & (1 << axis))
                for axis in range(3)
            )
            expected_count = 1 << neutral_axes
            if count != expected_count:
                raise ValueError("finer source count is incompatible")
            previous_phase = -1
            for source, slot in enumerate(active_slots):
                source_leaf = selected[slot]
                source_node = int(leaf_node_ids[source_leaf])
                if int(node_levels[source_node]) != primary_level + 1:
                    raise ValueError("finer source has the wrong level")
                phase = sum(
                    (int(node_coords[source_node, axis]) & 1) << axis
                    for axis in range(3)
                )
                for axis in range(3):
                    if reduced[axis] < 0 and not phase & (1 << axis):
                        raise ValueError("finer source has the wrong phase")
                    if reduced[axis] > 0 and phase & (1 << axis):
                        raise ValueError("finer source has the wrong phase")
                if phase <= previous_phase:
                    raise ValueError("finer source phases are out of order")
                output[primary, direction_index, source] = np.uint8(phase)
                previous_phase = phase

    return output
