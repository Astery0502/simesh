"""Independent basic-slice reference for STO-001."""

from __future__ import annotations

import numpy as np


def _spatial_slices(lower: np.ndarray, upper: np.ndarray) -> tuple[slice, ...]:
    return tuple(
        slice(int(start), int(stop))
        for start, stop in zip(lower, upper, strict=True)
    )


def gather_blocks_into_reference(
    backing: np.ndarray,
    backing_lower: np.ndarray,
    backing_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    destination: np.ndarray,
    destination_lower: np.ndarray,
) -> None:
    extent = backing_upper - backing_lower
    destination_upper = destination_lower + extent
    source_slices = _spatial_slices(backing_lower, backing_upper)
    destination_slices = _spatial_slices(destination_lower, destination_upper)
    for slot, block_id in enumerate(block_ids):
        for field_slot, field_id in enumerate(field_ids):
            np.copyto(
                destination[(slot, field_slot, *destination_slices)],
                backing[(int(block_id), int(field_id), *source_slices)],
            )


def scatter_blocks_from_reference(
    source: np.ndarray,
    source_lower: np.ndarray,
    source_upper: np.ndarray,
    block_ids: np.ndarray,
    field_ids: np.ndarray,
    backing: np.ndarray,
    backing_lower: np.ndarray,
) -> None:
    extent = source_upper - source_lower
    backing_upper = backing_lower + extent
    source_slices = _spatial_slices(source_lower, source_upper)
    backing_slices = _spatial_slices(backing_lower, backing_upper)
    for slot, block_id in enumerate(block_ids):
        for field_slot, field_id in enumerate(field_ids):
            np.copyto(
                backing[(int(block_id), int(field_id), *backing_slices)],
                source[(slot, field_slot, *source_slices)],
            )
