"""Exact-rational independent reference for GEO-002."""

from __future__ import annotations

from fractions import Fraction

import numpy as np


def refined_leaf_geometry_reference(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
    root_shape: np.ndarray,
    domain_cell_counts: np.ndarray,
    block_cell_counts: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    leaf_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate exact-rational bounds and slot spacing for selected leaves.

    Inputs are expected to satisfy the GEO-002 contract and to belong to one
    validated FST artifact lifecycle.  This reference deliberately performs
    only the small compatibility check needed to interpret the integer cell
    lattice; production owns complete layout, overflow, and atomicity checks.
    """
    for axis in range(3):
        if int(domain_cell_counts[axis]) != (
            int(root_shape[axis]) * int(block_cell_counts[axis])
        ):
            raise ValueError(
                "domain_cell_counts must equal root_shape * block_cell_counts"
            )

    lower_fractions = [
        Fraction.from_float(float(domain_lower[axis])) for axis in range(3)
    ]
    base_spacing_fractions = [
        (
            Fraction.from_float(float(domain_upper[axis]))
            - lower_fractions[axis]
        )
        / int(domain_cell_counts[axis])
        for axis in range(3)
    ]

    slot_count = int(leaf_ids.shape[0])
    bounds = np.empty((slot_count, 2, 3), dtype=np.float64)
    spacing = np.empty((slot_count, 3), dtype=np.float64)

    for slot, leaf_id_value in enumerate(leaf_ids):
        node_id = int(leaf_node_ids[int(leaf_id_value)])
        level = int(node_levels[node_id])
        scale = 1 << (level - 1)
        coordinate = node_coords[node_id]

        for axis in range(3):
            level_domain_cells = int(domain_cell_counts[axis]) * scale
            level_spacing = base_spacing_fractions[axis] / scale
            lower_index = (
                int(coordinate[axis]) * int(block_cell_counts[axis])
            )
            upper_index = lower_index + int(block_cell_counts[axis])

            spacing[slot, axis] = float(level_spacing)
            for side, face_index in enumerate((lower_index, upper_index)):
                if face_index == 0:
                    value = float(domain_lower[axis])
                elif face_index == level_domain_cells:
                    value = float(domain_upper[axis])
                else:
                    value = float(
                        lower_fractions[axis] + face_index * level_spacing
                    )
                bounds[slot, side, axis] = value

    return bounds, spacing
