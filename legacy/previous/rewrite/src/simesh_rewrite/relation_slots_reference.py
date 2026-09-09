"""Independent Python-list reference for RSL-001."""

from __future__ import annotations

import numpy as np


def resolve_refined_relation_source_slots_reference(
    leaf_count: int,
    selected_leaf_ids: np.ndarray,
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
    source_slots: np.ndarray,
) -> None:
    """Map active global source IDs through list uniqueness and ``.index``."""
    total_leaves = int(leaf_count)
    selected = [int(value) for value in selected_leaf_ids]
    for slot, leaf_id in enumerate(selected):
        if leaf_id < 0 or leaf_id >= total_leaves:
            raise ValueError(f"selected leaf {slot} is out of range")
        for earlier in range(slot):
            if selected[earlier] == leaf_id:
                raise ValueError(f"selected leaf {slot} is duplicated")

    primary_count, direction_count = source_counts.shape
    if primary_count > len(selected):
        raise ValueError("accepted relation rows exceed selected slots")

    for primary in range(primary_count):
        for direction in range(direction_count):
            count = int(source_counts[primary, direction])
            if count < 0 or count > 4:
                raise ValueError("source count exceeds four")
            for source in range(4):
                leaf_id = int(source_leaf_ids[primary, direction, source])
                if source < count:
                    if leaf_id < 0 or leaf_id >= total_leaves:
                        raise ValueError("active source leaf is out of range")
                    try:
                        selected.index(leaf_id)
                    except ValueError as error:
                        raise ValueError(
                            "active source leaf is absent from selected slots"
                        ) from error
                elif leaf_id != -1:
                    raise ValueError("unused source leaf must be -1")

    for primary in range(primary_count):
        for direction in range(direction_count):
            count = int(source_counts[primary, direction])
            for source in range(count):
                leaf_id = int(source_leaf_ids[primary, direction, source])
                source_slots[primary, direction, source] = selected.index(
                    leaf_id
                )
            for source in range(count, 4):
                source_slots[primary, direction, source] = -1
