"""Independent Python-integer reference for PRI-001."""

from __future__ import annotations


def ascending_primary_prefix_reference(
    first_primary_id: int,
    block_count: int,
    capacity: int,
) -> tuple[list[int], int]:
    """Allocate and return the maximal ascending primary-ID prefix."""
    first = int(first_primary_id)
    count_limit = int(block_count)
    slot_capacity = int(capacity)

    if first > count_limit:
        raise ValueError("first_primary_id exceeds block count")
    if first == count_limit:
        return [], 0
    if slot_capacity == 0:
        raise ValueError("primary capacity must be positive before the end")

    primary_count = min(slot_capacity, count_limit - first)
    return list(range(first, first + primary_count)), primary_count
