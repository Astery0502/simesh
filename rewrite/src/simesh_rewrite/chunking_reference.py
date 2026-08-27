"""Independent list-based reference for STO-002 chunk planning."""

from __future__ import annotations

import numpy as np


def plan_level1_chunk_reference(
    first_primary_id: int,
    face_neighbor_ids: np.ndarray,
    include_face_closure: bool,
    capacity: int,
) -> tuple[list[int], int, int]:
    block_count = face_neighbor_ids.shape[0]
    if first_primary_id == block_count:
        return [], 0, 0
    if not include_face_closure:
        count = min(capacity, block_count - first_primary_id)
        values = list(range(first_primary_id, first_primary_id + count))
        return values, count, count

    primaries: list[int] = []
    support: list[int] = []
    while first_primary_id + len(primaries) < block_count:
        candidate = first_primary_id + len(primaries)
        trial_primaries = [*primaries, candidate]
        trial_support = [value for value in support if value != candidate]
        for neighbor in face_neighbor_ids[candidate]:
            neighbor = int(neighbor)
            if (
                neighbor >= 0
                and neighbor not in trial_primaries
                and neighbor not in trial_support
            ):
                trial_support.append(neighbor)
        if len(trial_primaries) + len(trial_support) > capacity:
            if not primaries:
                raise ValueError("chunk capacity cannot fit the first primary closure")
            break
        primaries = trial_primaries
        support = trial_support
    values = [*primaries, *support]
    return values, len(primaries), len(values)
