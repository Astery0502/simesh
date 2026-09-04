"""Independent resident per-primary reference for RHE-001."""

from __future__ import annotations

import numpy as np

from .coarser_support_reference import coarser_slope_support_plan_reference
from .coarser_workspace_application_reference import (
    apply_coarser_workspace_plan_reference,
)
from .coarser_workspace_reference import coarser_workspace_boxes_reference
from .finer_boxes_reference import finer_restriction_boxes_reference
from .physical_widening_reference import (
    apply_cartesian_physical_widening_reference,
)
from .prolongation_reference import prolong_cartesian_2to1_reference
from .relation_phases_reference import refined_relation_phase_codes_reference
from .relation_slots_reference import resolve_refined_relation_source_slots_reference
from .relations_reference import (
    RELATION_COARSER,
    RELATION_FINER,
    RELATION_PHYSICAL,
    RELATION_SAME,
    balanced_refined_relations_reference,
)
from .restriction_reference import restrict_cartesian_2to1_reference
from .same_level_boxes_reference import same_level_source_boxes_reference
from .target_boxes_reference import directed_halo_target_boxes_reference
from .coarser_support import CANONICAL_DIRECTIONS


def _direction_index(direction: tuple[int, int, int]) -> int:
    column = (
        (direction[2] + 1) * 9
        + (direction[1] + 1) * 3
        + direction[0]
        + 1
    )
    return column if column < 13 else column - 1


def _nonempty(lower: np.ndarray, upper: np.ndarray) -> bool:
    return all(int(lower[axis]) < int(upper[axis]) for axis in range(3))


def fill_selected_refined_halos_reference(
    backing: np.ndarray,
    primary_leaf_ids: np.ndarray,
    field_ids: np.ndarray,
    root_shape: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    leaf_node_ids: np.ndarray,
    lower_halo: np.ndarray,
    upper_halo: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
    output: np.ndarray,
) -> None:
    """Fill selected output rows through an independent resident composition."""
    leaf_count = int(leaf_node_ids.shape[0])
    field_count = int(field_ids.shape[0])
    block_shape = np.asarray(backing.shape[2:], dtype=np.int64)
    interior_lower = np.asarray(lower_halo, dtype=np.int64)
    interior_upper = interior_lower + block_shape
    padded_shape = tuple(
        int(block_shape[axis] + lower_halo[axis] + upper_halo[axis])
        for axis in range(3)
    )
    zero = np.zeros(3, dtype=np.int64)
    requested_upper = np.asarray(padded_shape, dtype=np.int64)
    target_lower, target_upper = directed_halo_target_boxes_reference(
        interior_lower,
        interior_upper,
        zero,
        requested_upper,
        CANONICAL_DIRECTIONS,
    )
    interior_box = tuple(
        slice(int(interior_lower[axis]), int(interior_upper[axis]))
        for axis in range(3)
    )

    for primary_leaf_value in primary_leaf_ids:
        primary_leaf = int(primary_leaf_value)
        primary_vector = np.asarray([primary_leaf], dtype=np.int64)
        kinds, masks, counts, source_ids = balanced_refined_relations_reference(
            root_shape,
            node_levels,
            node_coords,
            leaf_node_ids,
            primary_vector,
            CANONICAL_DIRECTIONS,
        )

        selected_list = [primary_leaf]
        for direction_row in range(26):
            for source in range(int(counts[0, direction_row])):
                leaf_id = int(source_ids[0, direction_row, source])
                if leaf_id not in selected_list:
                    selected_list.append(leaf_id)
        selected_ids = np.asarray(selected_list, dtype=np.int64)
        source_slots = np.empty_like(source_ids)
        resolve_refined_relation_source_slots_reference(
            leaf_count, selected_ids, counts, source_ids, source_slots
        )
        phase_codes = refined_relation_phase_codes_reference(
            selected_ids,
            node_levels,
            node_coords,
            leaf_node_ids,
            CANONICAL_DIRECTIONS,
            kinds,
            masks,
            counts,
            source_slots,
        )

        payload = np.full(
            (selected_ids.shape[0], field_count, *padded_shape),
            np.nan,
            dtype=np.float64,
        )
        payload[(slice(None), slice(None), *interior_box)] = backing[
            selected_ids[:, None], field_ids[None, :]
        ]

        for direction_row, direction_array in enumerate(CANONICAL_DIRECTIONS):
            if int(masks[0, direction_row]) != 0:
                continue
            if not _nonempty(target_lower[direction_row], target_upper[direction_row]):
                continue
            direction = direction_array.reshape(1, 3)
            target_l = target_lower[direction_row].reshape(1, 3)
            target_u = target_upper[direction_row].reshape(1, 3)
            kind = int(kinds[0, direction_row])
            source_count = int(counts[0, direction_row])

            if kind == RELATION_SAME:
                source_l, source_u = same_level_source_boxes_reference(
                    interior_lower,
                    interior_upper,
                    direction,
                    target_l,
                    target_u,
                )
                source_slot = int(source_slots[0, direction_row, 0])
                source_box = tuple(
                    slice(int(source_l[0, axis]), int(source_u[0, axis]))
                    for axis in range(3)
                )
                destination_box = tuple(
                    slice(
                        int(target_lower[direction_row, axis]),
                        int(target_upper[direction_row, axis]),
                    )
                    for axis in range(3)
                )
                payload[(0, slice(None), *destination_box)] = payload[
                    (source_slot, slice(None), *source_box)
                ]
            elif kind == RELATION_FINER:
                phases = phase_codes[0, direction_row, :source_count]
                directions = np.repeat(direction, source_count, axis=0)
                targets_l = np.repeat(target_l, source_count, axis=0)
                targets_u = np.repeat(target_u, source_count, axis=0)
                placed_l, placed_u, source_l, source_u = (
                    finer_restriction_boxes_reference(
                        interior_lower,
                        interior_upper,
                        directions,
                        phases,
                        targets_l,
                        targets_u,
                    )
                )
                for source in range(source_count):
                    source_slot = int(source_slots[0, direction_row, source])
                    restrict_cartesian_2to1_reference(
                        payload[source_slot : source_slot + 1],
                        source_l[source],
                        source_u[source],
                        payload[:1],
                        placed_l[source],
                    )
            elif kind == RELATION_COARSER:
                phase = phase_codes[0, direction_row, :1]
                cwp = coarser_workspace_boxes_reference(
                    interior_lower,
                    interior_upper,
                    direction,
                    phase,
                    target_l,
                    target_u,
                )
                csp = coarser_slope_support_plan_reference(
                    interior_lower,
                    interior_upper,
                    int(selected_ids.shape[0]),
                    0,
                    int(phase[0]),
                    direction_array,
                    CANONICAL_DIRECTIONS,
                    kinds[0],
                    masks[0],
                    counts[0],
                    source_slots[0],
                    target_lower[direction_row],
                    target_upper[direction_row],
                    *(value[0] for value in cwp),
                )
                transfer_count, record_count, *plan = csp
                coarse_workspace = np.full(
                    (
                        1,
                        field_count,
                        int(block_shape[0]) + 1,
                        int(block_shape[1]) + 1,
                        int(block_shape[2]) + 1,
                    ),
                    np.nan,
                    dtype=np.float64,
                )
                apply_coarser_workspace_plan_reference(
                    payload,
                    coarse_workspace,
                    interior_lower,
                    interior_upper,
                    cwp[4][0],
                    cwp[5][0],
                    transfer_count,
                    record_count,
                    *plan,
                    boundary_modes,
                    normal_field_slots,
                )
                prolong_cartesian_2to1_reference(
                    coarse_workspace,
                    cwp[4][0],
                    cwp[5][0],
                    cwp[6][0],
                    payload[:1],
                    target_lower[direction_row],
                    target_upper[direction_row],
                    interior_lower,
                )
            elif kind == RELATION_PHYSICAL:
                raise AssertionError("unmasked noncenter reference action is physical")

        pwa_logical_lower: list[np.ndarray] = []
        pwa_logical_upper: list[np.ndarray] = []
        pwa_offsets: list[tuple[int, int, int]] = []
        pwa_directions: list[np.ndarray] = []
        pwa_masks: list[int] = []
        pwa_base_lower: list[np.ndarray] = []
        pwa_base_upper: list[np.ndarray] = []
        pwa_target_lower: list[np.ndarray] = []
        pwa_target_upper: list[np.ndarray] = []
        for direction_row, direction in enumerate(CANONICAL_DIRECTIONS):
            mask = int(masks[0, direction_row])
            if mask == 0 or not _nonempty(
                target_lower[direction_row], target_upper[direction_row]
            ):
                continue
            reduced = tuple(
                0 if mask & (1 << axis) else int(direction[axis])
                for axis in range(3)
            )
            if reduced == (0, 0, 0):
                base_l = interior_lower
                base_u = interior_upper
            else:
                reduced_row = _direction_index(reduced)
                base_l = target_lower[reduced_row]
                base_u = target_upper[reduced_row]
            pwa_logical_lower.append(interior_lower)
            pwa_logical_upper.append(interior_upper)
            pwa_offsets.append((0, 0, 0))
            pwa_directions.append(direction)
            pwa_masks.append(mask)
            pwa_base_lower.append(base_l)
            pwa_base_upper.append(base_u)
            pwa_target_lower.append(target_lower[direction_row])
            pwa_target_upper.append(target_upper[direction_row])

        row_count = len(pwa_masks)
        apply_cartesian_physical_widening_reference(
            payload,
            0,
            np.asarray(pwa_logical_lower, dtype=np.int64).reshape(row_count, 3),
            np.asarray(pwa_logical_upper, dtype=np.int64).reshape(row_count, 3),
            np.asarray(pwa_offsets, dtype=np.int64).reshape(row_count, 3),
            np.asarray(pwa_directions, dtype=np.int64).reshape(row_count, 3),
            np.asarray(pwa_masks, dtype=np.uint8),
            np.asarray(pwa_base_lower, dtype=np.int64).reshape(row_count, 3),
            np.asarray(pwa_base_upper, dtype=np.int64).reshape(row_count, 3),
            np.asarray(pwa_target_lower, dtype=np.int64).reshape(row_count, 3),
            np.asarray(pwa_target_upper, dtype=np.int64).reshape(row_count, 3),
            boundary_modes,
            normal_field_slots,
        )
        output[primary_leaf] = payload[0]
