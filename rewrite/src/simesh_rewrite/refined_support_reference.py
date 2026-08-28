"""Independent Python-list reference for STO-004."""

from __future__ import annotations

import numpy as np


def _require_paired_relation_shape(
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
) -> tuple[int, int]:
    if source_counts.ndim != 2:
        raise ValueError("source_counts must have rank two")
    candidate_count, direction_count = source_counts.shape
    if source_leaf_ids.shape != (candidate_count, direction_count, 4):
        raise ValueError("source arrays must have paired candidate/direction shapes")
    return candidate_count, direction_count


def plan_balanced_refined_support_prefix_reference(
    first_primary_id: int,
    leaf_count: int,
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
    capacity: int,
) -> tuple[list[int], int, int]:
    """Allocate the maximal dense refined primary/support plan."""
    first = int(first_primary_id)
    total_leaves = int(leaf_count)
    slot_capacity = int(capacity)
    candidate_count, direction_count = _require_paired_relation_shape(
        source_counts,
        source_leaf_ids,
    )
    expected_candidates = min(slot_capacity, total_leaves - first)
    if candidate_count != expected_candidates:
        raise ValueError(
            "source relation window must equal min(capacity, remaining leaves)"
        )
    if first == total_leaves:
        return [], 0, 0
    if slot_capacity == 0:
        raise ValueError("selected capacity must be positive before the end")

    primaries: list[int] = []
    support: list[int] = []
    for row in range(candidate_count):
        candidate = first + row
        trial_primaries = [*primaries, candidate]
        trial_support = [value for value in support if value != candidate]
        for direction in range(direction_count):
            count = int(source_counts[row, direction])
            for source_column in range(count):
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


def maximum_balanced_refined_support_slots_reference(
    first_primary_id: int,
    leaf_count: int,
    source_counts: np.ndarray,
    source_leaf_ids: np.ndarray,
) -> int:
    """Return the largest unique one-primary closure in a bounded window."""
    first = int(first_primary_id)
    total_leaves = int(leaf_count)
    candidate_count, direction_count = _require_paired_relation_shape(
        source_counts,
        source_leaf_ids,
    )
    if candidate_count > total_leaves - first:
        raise ValueError("source relation window exceeds remaining leaves")

    maximum = 0
    for row in range(candidate_count):
        closure = {first + row}
        for direction in range(direction_count):
            count = int(source_counts[row, direction])
            closure.update(
                int(source_leaf_ids[row, direction, source_column])
                for source_column in range(count)
            )
        maximum = max(maximum, len(closure))
    return maximum
