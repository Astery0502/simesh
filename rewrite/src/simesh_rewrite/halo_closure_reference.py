"""Independent Python-list and face-walk reference for HCL-001."""

from __future__ import annotations

import numpy as np


def _walk_level1_direction_reference(
    block_id: int,
    dx: int,
    dy: int,
    dz: int,
    face_neighbor_ids: np.ndarray,
) -> int:
    current = int(block_id)
    for axis, delta in enumerate((dx, dy, dz)):
        if delta == 0:
            continue
        face = 2 * axis + (1 if delta > 0 else 0)
        current = int(face_neighbor_ids[current, face])
        if current < 0:
            return -1
    return current


def plan_level1_halo_closed_prefix_reference(
    first_primary_id: int,
    face_neighbor_ids: np.ndarray,
    capacity: int,
) -> tuple[list[int], int, int]:
    """Allocate the maximal complete one-block halo-closed plan."""
    first = int(first_primary_id)
    block_count = int(face_neighbor_ids.shape[0])
    slot_capacity = int(capacity)
    if first == block_count:
        return [], 0, 0

    primaries: list[int] = []
    support: list[int] = []
    while first + len(primaries) < block_count:
        candidate = first + len(primaries)
        trial_primaries = [*primaries, candidate]
        trial_support = [value for value in support if value != candidate]
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    neighbor = _walk_level1_direction_reference(
                        candidate,
                        dx,
                        dy,
                        dz,
                        face_neighbor_ids,
                    )
                    if (
                        neighbor >= 0
                        and neighbor not in trial_primaries
                        and neighbor not in trial_support
                    ):
                        trial_support.append(neighbor)

        if len(trial_primaries) + len(trial_support) > slot_capacity:
            if not primaries:
                raise ValueError(
                    "chunk capacity cannot fit the first halo closure"
                )
            break
        primaries = trial_primaries
        support = trial_support

    selected = [*primaries, *support]
    return selected, len(primaries), len(selected)


def minimum_level1_halo_closed_slots_reference(
    face_neighbor_ids: np.ndarray,
) -> int:
    """Return the largest clipped one-block closure from fixed face walks."""
    maximum = 0
    for block_id in range(face_neighbor_ids.shape[0]):
        count = 0
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if _walk_level1_direction_reference(
                        block_id,
                        dx,
                        dy,
                        dz,
                        face_neighbor_ids,
                    ) >= 0:
                        count += 1
        maximum = max(maximum, count)
    return maximum
