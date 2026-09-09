"""Independent Python-list reference for SPR-001."""

from __future__ import annotations

import numpy as np


def plan_selected_refined_support_prefix_reference(
    primary_leaf_ids: np.ndarray,
    first_primary_position: int,
    leaf_count: int,
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
    capacity: int,
) -> tuple[list[int], int, int]:
    """Allocate one maximal selected-primary/support prefix."""
    primary_ids = [int(value) for value in primary_leaf_ids]
    first = int(first_primary_position)
    total_leaves = int(leaf_count)
    slot_capacity = int(capacity)
    if any(value < 0 or value >= total_leaves for value in primary_ids):
        raise ValueError("primary selection is out of range")
    if any(left >= right for left, right in zip(primary_ids, primary_ids[1:])):
        raise ValueError("primary selection must be strictly increasing")
    if first < 0 or first > len(primary_ids):
        raise ValueError("first position is out of range")
    expected_rows = min(slot_capacity, len(primary_ids) - first)
    if source_counts.shape[0] != expected_rows:
        raise ValueError("source relation window has the wrong row count")
    if source_leaf_ids.shape != (*source_counts.shape, 4):
        raise ValueError("source arrays have incompatible shapes")
    if first == len(primary_ids):
        return [], 0, 0
    if slot_capacity == 0:
        raise ValueError("selected capacity must be positive before the end")

    primaries: list[int] = []
    support: list[int] = []
    for row in range(expected_rows):
        candidate = primary_ids[first + row]
        trial_primaries = [*primaries, candidate]
        trial_support = [value for value in support if value != candidate]
        for direction in range(source_counts.shape[1]):
            for source_column in range(int(source_counts[row, direction])):
                source = int(
                    source_leaf_ids[row, direction, source_column]
                )
                if (
                    source not in trial_primaries
                    and source not in trial_support
                ):
                    trial_support.append(source)
        if len(trial_primaries) + len(trial_support) > slot_capacity:
            if not primaries:
                raise ValueError(
                    "selected capacity cannot fit the first refined support closure"
                )
            break
        primaries = trial_primaries
        support = trial_support

    selected = [*primaries, *support]
    return selected, len(primaries), len(selected)


def maximum_selected_refined_support_slots_reference(
    candidate_leaf_ids: np.ndarray,
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
) -> int:
    """Return the maximum explicit one-primary closure."""
    maximum = 0
    for row, candidate_value in enumerate(candidate_leaf_ids):
        closure = {int(candidate_value)}
        for direction in range(source_counts.shape[1]):
            for source_column in range(int(source_counts[row, direction])):
                closure.add(
                    int(source_leaf_ids[row, direction, source_column])
                )
        maximum = max(maximum, len(closure))
    return maximum
