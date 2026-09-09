"""Independent Python-list reference for FCL-001."""

from __future__ import annotations

import numpy as np


def plan_direct_face_closed_prefix_reference(
    first_primary_id: int,
    face_neighbor_ids: np.ndarray,
    capacity: int,
) -> tuple[list[int], int, int]:
    """Allocate the maximal direct-face-closed primary/support plan."""
    first = int(first_primary_id)
    block_count = int(face_neighbor_ids.shape[0])
    slot_capacity = int(capacity)
    if first == block_count:
        return [], 0, 0
    if slot_capacity == 0:
        raise ValueError("selected capacity must be positive before the end")

    primaries: list[int] = []
    support: list[int] = []
    while first + len(primaries) < block_count:
        candidate = first + len(primaries)
        trial_primaries = [*primaries, candidate]
        trial_support = [value for value in support if value != candidate]
        for neighbor_value in face_neighbor_ids[candidate]:
            neighbor = int(neighbor_value)
            if (
                neighbor >= 0
                and neighbor not in trial_primaries
                and neighbor not in trial_support
            ):
                trial_support.append(neighbor)

        if len(trial_primaries) + len(trial_support) > slot_capacity:
            if not primaries:
                raise ValueError(
                    "chunk capacity cannot fit the first primary closure"
                )
            break
        primaries = trial_primaries
        support = trial_support

    selected = [*primaries, *support]
    return selected, len(primaries), len(selected)


def minimum_direct_face_closed_slots_reference(
    face_neighbor_ids: np.ndarray,
) -> int:
    """Return the largest raw single-row direct-face closure size."""
    maximum = 0
    for neighbors in face_neighbor_ids:
        count = 1 + sum(int(neighbor) >= 0 for neighbor in neighbors)
        maximum = max(maximum, count)
    return maximum
