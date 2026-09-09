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


def _walk_direction(
    block_id: int,
    dx: int,
    dy: int,
    dz: int,
    face_neighbor_ids: np.ndarray,
) -> int:
    for axis, delta in enumerate((dx, dy, dz)):
        if delta:
            block_id = int(
                face_neighbor_ids[block_id, 2 * axis + (1 if delta > 0 else 0)]
            )
            if block_id < 0:
                return -1
    return block_id


def plan_level1_halo_chunk_reference(
    first_primary_id: int,
    face_neighbor_ids: np.ndarray,
    capacity: int,
) -> tuple[list[int], int, int]:
    block_count = face_neighbor_ids.shape[0]
    if first_primary_id == block_count:
        return [], 0, 0
    primaries: list[int] = []
    support: list[int] = []
    while first_primary_id + len(primaries) < block_count:
        candidate = first_primary_id + len(primaries)
        trial_primaries = [*primaries, candidate]
        trial_support = [value for value in support if value != candidate]
        for dz in range(-1, 2):
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == dy == dz == 0:
                        continue
                    neighbor = _walk_direction(
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
        if len(trial_primaries) + len(trial_support) > capacity:
            if not primaries:
                raise ValueError("chunk capacity cannot fit the first halo closure")
            break
        primaries = trial_primaries
        support = trial_support
    values = [*primaries, *support]
    return values, len(primaries), len(values)
