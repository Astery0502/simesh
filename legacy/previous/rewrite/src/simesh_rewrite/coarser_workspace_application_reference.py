"""Independent CWA-001 copy/RST/PWA reference composition."""

from __future__ import annotations

import numpy as np

from .coarser_support import SOURCE_FINE
from .physical_widening_reference import (
    apply_cartesian_physical_widening_reference,
)
from .restriction_reference import restrict_cartesian_2to1_reference


def apply_coarser_workspace_plan_reference(
    selected_payload: np.ndarray,
    coarse_workspace: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    workspace_required_lower: np.ndarray,
    workspace_required_upper: np.ndarray,
    transfer_count: int,
    record_count: int,
    plan_source_slots: np.ndarray,
    plan_source_is_fine: np.ndarray,
    plan_physical_masks: np.ndarray,
    plan_directions: np.ndarray,
    plan_source_lower: np.ndarray,
    plan_source_upper: np.ndarray,
    plan_base_lower: np.ndarray,
    plan_base_upper: np.ndarray,
    plan_target_lower: np.ndarray,
    plan_target_upper: np.ndarray,
    plan_logical_interior_lower: np.ndarray,
    plan_logical_interior_upper: np.ndarray,
    plan_storage_logical_offsets: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> None:
    """Apply a valid CWA plan using independent high-level/reference pieces."""
    del interior_lower, interior_upper
    del workspace_required_lower, workspace_required_upper

    for record in range(transfer_count):
        source_slot = int(plan_source_slots[record])
        source = selected_payload[source_slot : source_slot + 1]
        if int(plan_source_is_fine[record]) == int(SOURCE_FINE):
            restrict_cartesian_2to1_reference(
                source,
                plan_source_lower[record],
                plan_source_upper[record],
                coarse_workspace,
                plan_target_lower[record],
            )
            continue

        source_box = tuple(
            slice(
                int(plan_source_lower[record, axis]),
                int(plan_source_upper[record, axis]),
            )
            for axis in range(3)
        )
        target_box = tuple(
            slice(
                int(plan_target_lower[record, axis]),
                int(plan_target_upper[record, axis]),
            )
            for axis in range(3)
        )
        coarse_workspace[(0, slice(None), *target_box)] = selected_payload[
            (source_slot, slice(None), *source_box)
        ]

    physical_slice = slice(transfer_count, record_count)
    apply_cartesian_physical_widening_reference(
        coarse_workspace,
        0,
        plan_logical_interior_lower[physical_slice],
        plan_logical_interior_upper[physical_slice],
        plan_storage_logical_offsets[physical_slice],
        plan_directions[physical_slice],
        plan_physical_masks[physical_slice],
        plan_base_lower[physical_slice],
        plan_base_upper[physical_slice],
        plan_target_lower[physical_slice],
        plan_target_upper[physical_slice],
        boundary_modes,
        normal_field_slots,
    )
