"""Independent references for the SAM capabilities."""

from __future__ import annotations

import numpy as np


def place_level1_blocks_reference(
    payload: np.ndarray,
    payload_valid_lower: np.ndarray,
    block_ids: np.ndarray,
    block_cell_counts: np.ndarray,
    rank_to_coord: np.ndarray,
    uniform_grid: np.ndarray,
) -> None:
    source = tuple(
        slice(int(lower), int(lower + extent))
        for lower, extent in zip(
            payload_valid_lower,
            block_cell_counts,
            strict=True,
        )
    )
    for slot, block_id in enumerate(block_ids):
        coordinate = rank_to_coord[int(block_id)]
        global_lower = coordinate * block_cell_counts
        destination = tuple(
            slice(int(lower), int(lower + extent))
            for lower, extent in zip(
                global_lower,
                block_cell_counts,
                strict=True,
            )
        )
        uniform_grid[(slice(None), *destination)] = payload[
            (slot, slice(None), *source)
        ]
