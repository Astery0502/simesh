"""Exact-phase halo workspace, support binding and numerical application.

Owned internal primitives; source reads and product publication belong to the
analysis preparation adapters. Arithmetic is retained from the checked provider.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from simesh._kernels.primitives._coarser_support import fill_coarser_slope_support_plan_unchecked
from simesh._kernels.primitives._coarser_workspace import fill_coarser_workspace_boxes_unchecked
from simesh._kernels.primitives._finer_boxes import fill_finer_restriction_boxes_unchecked
from simesh._kernels.primitives._foundation import copy_region_into_unchecked
from simesh._kernels.primitives._physical_widening import apply_cartesian_physical_widening_unchecked
from simesh._kernels.primitives._prolongation import prolong_cartesian_2to1_into_unchecked
from simesh._kernels.primitives._refined_support import plan_selected_refined_support_prefix_unchecked
from simesh._kernels.primitives._relation_phases import fill_refined_relation_phase_codes_unchecked
from simesh._kernels.primitives._relation_slots import resolve_refined_relation_source_slots_unchecked
from simesh._kernels.primitives._relations import fill_balanced_refined_relations_unchecked
from simesh._kernels.primitives._restriction import restrict_cartesian_2to1_into_unchecked
from simesh._kernels.primitives._same_level_boxes import fill_same_level_source_boxes_unchecked
from simesh._amr.coarser_support import CANONICAL_DIRECTIONS, PLAN_CAPACITY, _validate_plan_geometry, _validate_relation_row, fill_coarser_slope_support_plan
from simesh._amr.coarser_workspace import fill_coarser_workspace_boxes
from simesh._amr.coarser_workspace_application import _apply_coarser_workspace_plan_unchecked
from simesh._amr.finer_boxes import fill_finer_restriction_boxes
from simesh._amr.physical_widening import _validate_physical_widening
from simesh._amr.prolongation import prolong_cartesian_2to1_into
from simesh._amr.relations import RELATION_COARSER, RELATION_FINER, RELATION_PHYSICAL, RELATION_SAME
from simesh._amr.same_level_boxes import fill_same_level_source_boxes


_DIRECTION_COUNT = 26


_FINER_SOURCE_CAPACITY = 4


@dataclass(frozen=True, slots=True)
class _RHEWorkspace:
    selected_leaf_ids: np.ndarray
    payload: np.ndarray
    relation_kinds: np.ndarray
    physical_masks: np.ndarray
    source_counts: np.ndarray
    source_leaf_ids: np.ndarray
    source_slots: np.ndarray
    phase_codes: np.ndarray
    coarse_workspace: np.ndarray
    target_lower: np.ndarray
    target_upper: np.ndarray
    action_directions: np.ndarray
    action_phases: np.ndarray
    action_target_lower: np.ndarray
    action_target_upper: np.ndarray
    same_source_lower: np.ndarray
    same_source_upper: np.ndarray
    copy_extent: np.ndarray
    finer_target_lower: np.ndarray
    finer_target_upper: np.ndarray
    finer_source_lower: np.ndarray
    finer_source_upper: np.ndarray
    cwp_outputs: tuple[np.ndarray, ...]
    csp_outputs: tuple[np.ndarray, ...]
    pwa_logical_lower: np.ndarray
    pwa_logical_upper: np.ndarray
    pwa_offsets: np.ndarray
    pwa_directions: np.ndarray
    pwa_masks: np.ndarray
    pwa_base_lower: np.ndarray
    pwa_base_upper: np.ndarray
    pwa_target_lower: np.ndarray
    pwa_target_upper: np.ndarray


def _direction_index(direction: tuple[int, int, int]) -> int:
    column = (
        (direction[2] + 1) * 9
        + (direction[1] + 1) * 3
        + direction[0]
        + 1
    )
    if column == 13:
        raise ValueError("center has no all-26 direction row")
    return column if column < 13 else column - 1


def _nonempty(lower: np.ndarray, upper: np.ndarray) -> bool:
    return all(int(lower[axis]) < int(upper[axis]) for axis in range(3))


def _fresh_csp_plan() -> tuple[np.ndarray, ...]:
    return (
        np.empty(PLAN_CAPACITY, dtype=np.int64),
        np.empty(PLAN_CAPACITY, dtype=np.uint8),
        np.empty(PLAN_CAPACITY, dtype=np.uint8),
        *(np.empty((PLAN_CAPACITY, 3), dtype=np.int64) for _ in range(10)),
    )


def _allocate_workspace(
    slot_capacity: int,
    field_count: int,
    block_shape: tuple[int, int, int],
    padded_shape: tuple[int, int, int],
) -> tuple[_RHEWorkspace, tuple[np.ndarray, ...]]:
    selected_leaf_ids = np.empty(slot_capacity, dtype=np.int64)
    payload = np.empty(
        (slot_capacity, field_count, *padded_shape), dtype=np.float64
    )
    relation_kinds = np.empty(
        (slot_capacity, _DIRECTION_COUNT), dtype=np.uint8
    )
    physical_masks = np.empty_like(relation_kinds)
    source_counts = np.empty_like(relation_kinds)
    source_leaf_ids = np.empty(
        (slot_capacity, _DIRECTION_COUNT, 4), dtype=np.int64
    )
    source_slots = np.empty_like(source_leaf_ids)
    phase_codes = np.empty_like(source_leaf_ids, dtype=np.uint8)
    coarse_shape = tuple(value + 1 for value in block_shape)
    coarse_workspace = np.empty(
        (1, field_count, *coarse_shape), dtype=np.float64
    )
    target_lower = np.empty((_DIRECTION_COUNT, 3), dtype=np.int64)
    target_upper = np.empty_like(target_lower)

    action_directions = np.empty((_FINER_SOURCE_CAPACITY, 3), dtype=np.int64)
    action_phases = np.empty(_FINER_SOURCE_CAPACITY, dtype=np.uint8)
    action_target_lower = np.empty_like(action_directions)
    action_target_upper = np.empty_like(action_directions)
    same_source_lower = np.empty((1, 3), dtype=np.int64)
    same_source_upper = np.empty((1, 3), dtype=np.int64)
    copy_extent = np.empty(3, dtype=np.int64)
    finer_target_lower = np.empty_like(action_directions)
    finer_target_upper = np.empty_like(action_directions)
    finer_source_lower = np.empty_like(action_directions)
    finer_source_upper = np.empty_like(action_directions)
    cwp_outputs = tuple(np.empty((1, 3), dtype=np.int64) for _ in range(7))
    csp_outputs = _fresh_csp_plan()

    pwa_logical_lower = np.empty((_DIRECTION_COUNT, 3), dtype=np.int64)
    pwa_logical_upper = np.empty_like(pwa_logical_lower)
    pwa_offsets = np.empty_like(pwa_logical_lower)
    pwa_directions = np.empty_like(pwa_logical_lower)
    pwa_masks = np.empty(_DIRECTION_COUNT, dtype=np.uint8)
    pwa_base_lower = np.empty_like(pwa_logical_lower)
    pwa_base_upper = np.empty_like(pwa_logical_lower)
    pwa_target_lower = np.empty_like(pwa_logical_lower)
    pwa_target_upper = np.empty_like(pwa_logical_lower)

    workspace = _RHEWorkspace(
        selected_leaf_ids,
        payload,
        relation_kinds,
        physical_masks,
        source_counts,
        source_leaf_ids,
        source_slots,
        phase_codes,
        coarse_workspace,
        target_lower,
        target_upper,
        action_directions,
        action_phases,
        action_target_lower,
        action_target_upper,
        same_source_lower,
        same_source_upper,
        copy_extent,
        finer_target_lower,
        finer_target_upper,
        finer_source_lower,
        finer_source_upper,
        cwp_outputs,
        csp_outputs,
        pwa_logical_lower,
        pwa_logical_upper,
        pwa_offsets,
        pwa_directions,
        pwa_masks,
        pwa_base_lower,
        pwa_base_upper,
        pwa_target_lower,
        pwa_target_upper,
    )
    arrays = (
        selected_leaf_ids,
        payload,
        relation_kinds,
        physical_masks,
        source_counts,
        source_leaf_ids,
        source_slots,
        phase_codes,
        coarse_workspace,
        target_lower,
        target_upper,
        action_directions,
        action_phases,
        action_target_lower,
        action_target_upper,
        same_source_lower,
        same_source_upper,
        copy_extent,
        finer_target_lower,
        finer_target_upper,
        finer_source_lower,
        finer_source_upper,
        *cwp_outputs,
        *csp_outputs,
        pwa_logical_lower,
        pwa_logical_upper,
        pwa_offsets,
        pwa_directions,
        pwa_masks,
        pwa_base_lower,
        pwa_base_upper,
        pwa_target_lower,
        pwa_target_upper,
    )
    return workspace, arrays


def _prepare_action_rows(
    workspace: _RHEWorkspace,
    direction_index: int,
    row_count: int,
) -> None:
    workspace.action_directions[:row_count] = CANONICAL_DIRECTIONS[
        direction_index
    ]
    workspace.action_target_lower[:row_count] = workspace.target_lower[
        direction_index
    ]
    workspace.action_target_upper[:row_count] = workspace.target_upper[
        direction_index
    ]


def _prepare_primary_pwa(
    workspace: _RHEWorkspace,
    primary: int,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
) -> int:
    count = 0
    for direction_row in range(_DIRECTION_COUNT):
        mask = int(workspace.physical_masks[primary, direction_row])
        target_lower = workspace.target_lower[direction_row]
        target_upper = workspace.target_upper[direction_row]
        if mask == 0 or not _nonempty(target_lower, target_upper):
            continue
        direction = CANONICAL_DIRECTIONS[direction_row]
        workspace.pwa_logical_lower[count] = interior_lower
        workspace.pwa_logical_upper[count] = interior_upper
        workspace.pwa_offsets[count] = 0
        workspace.pwa_directions[count] = direction
        workspace.pwa_masks[count] = mask
        workspace.pwa_target_lower[count] = target_lower
        workspace.pwa_target_upper[count] = target_upper
        reduced = tuple(
            0 if mask & (1 << axis) else int(direction[axis])
            for axis in range(3)
        )
        if reduced == (0, 0, 0):
            workspace.pwa_base_lower[count] = interior_lower
            workspace.pwa_base_upper[count] = interior_upper
        else:
            reduced_row = _direction_index(reduced)
            workspace.pwa_base_lower[count] = workspace.target_lower[reduced_row]
            workspace.pwa_base_upper[count] = workspace.target_upper[reduced_row]
        count += 1
    return count


def _guard_relation_source_slots_before_phase(
    workspace: _RHEWorkspace,
    primary_count: int,
    selected_count: int,
) -> None:
    """Protect the unchecked RPH dereference with bounded active-slot guards."""
    kind_names = {
        RELATION_COARSER: "COARSER",
        RELATION_FINER: "FINER",
        RELATION_PHYSICAL: "PHYSICAL",
        RELATION_SAME: "SAME",
    }
    for primary in range(primary_count):
        for direction in range(_DIRECTION_COUNT):
            count = int(workspace.source_counts[primary, direction])
            if count < 0 or count > 4:
                raise RuntimeError("relation source count is outside [0,4]")
            for source in range(count):
                slot = int(workspace.source_slots[primary, direction, source])
                if slot < 0 or slot >= selected_count:
                    kind = int(workspace.relation_kinds[primary, direction])
                    name = kind_names.get(kind, "relation")
                    raise RuntimeError(f"{name} action source slot is out of range")


def _preflight_chunk_actions_m1_reference(
    workspace: _RHEWorkspace,
    primary_count: int,
    selected_count: int,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
    *,
    _owned: bool = False,
) -> None:
    """Validate actual action geometry without repeating proven value kernels."""
    empty_fields = slice(0, 0)
    for primary in range(primary_count):
        relation_validated = False
        for direction_row in range(_DIRECTION_COUNT):
            if int(workspace.physical_masks[primary, direction_row]) != 0:
                continue
            target_lower = workspace.target_lower[direction_row]
            target_upper = workspace.target_upper[direction_row]
            if not _nonempty(target_lower, target_upper):
                continue
            kind = int(workspace.relation_kinds[primary, direction_row])
            source_count = int(workspace.source_counts[primary, direction_row])
            _prepare_action_rows(workspace, direction_row, max(source_count, 1))

            if kind == RELATION_SAME:
                if source_count != 1:
                    raise RuntimeError("SAME action has invalid source count")
                source_slot = int(
                    workspace.source_slots[primary, direction_row, 0]
                )
                if source_slot < 0 or source_slot >= selected_count:
                    raise RuntimeError("SAME action source slot is out of range")
                if source_slot == primary:
                    raise RuntimeError("SAME action source aliases its primary")
                fill_same_level_source_boxes(
                    interior_lower,
                    interior_upper,
                    workspace.action_directions[:1],
                    workspace.action_target_lower[:1],
                    workspace.action_target_upper[:1],
                    workspace.same_source_lower,
                    workspace.same_source_upper,
                )
            elif kind == RELATION_FINER:
                if source_count < 1 or source_count > 4:
                    raise RuntimeError("FINER action has invalid source count")
                workspace.action_phases[:source_count] = workspace.phase_codes[
                    primary, direction_row, :source_count
                ]
                fill_finer_restriction_boxes(
                    interior_lower,
                    interior_upper,
                    workspace.action_directions[:source_count],
                    workspace.action_phases[:source_count],
                    workspace.action_target_lower[:source_count],
                    workspace.action_target_upper[:source_count],
                    workspace.finer_target_lower[:source_count],
                    workspace.finer_target_upper[:source_count],
                    workspace.finer_source_lower[:source_count],
                    workspace.finer_source_upper[:source_count],
                )
                for source in range(source_count):
                    source_slot = int(
                        workspace.source_slots[
                            primary, direction_row, source
                        ]
                    )
                    if source_slot < 0 or source_slot >= selected_count:
                        raise RuntimeError(
                            "FINER action source slot is out of range"
                        )
                    if source_slot == primary:
                        raise RuntimeError(
                            "FINER action source aliases its primary"
                        )
            elif kind == RELATION_COARSER:
                if source_count != 1:
                    raise RuntimeError("COARSER action has invalid source count")
                source_slot = int(
                    workspace.source_slots[primary, direction_row, 0]
                )
                if source_slot < 0 or source_slot >= selected_count:
                    raise RuntimeError("COARSER action source slot is out of range")
                workspace.action_phases[0] = workspace.phase_codes[
                    primary, direction_row, 0
                ]
                fill_coarser_workspace_boxes(
                    interior_lower,
                    interior_upper,
                    workspace.action_directions[:1],
                    workspace.action_phases[:1],
                    workspace.action_target_lower[:1],
                    workspace.action_target_upper[:1],
                    *workspace.cwp_outputs,
                )
                if _owned:
                    if not relation_validated:
                        _validate_relation_row(
                            CANONICAL_DIRECTIONS,
                            workspace.relation_kinds[primary],
                            workspace.physical_masks[primary],
                            workspace.source_counts[primary],
                            workspace.source_slots[primary],
                            selected_count,
                        )
                        relation_validated = True
                    transfer_count, record_count = _fill_owned_coarser_plan(
                        workspace, primary, interior_lower, interior_upper
                    )
                else:
                    transfer_count, record_count = fill_coarser_slope_support_plan(
                        interior_lower,
                        interior_upper,
                        selected_count,
                        primary,
                        int(workspace.action_phases[0]),
                        workspace.action_directions[0],
                        CANONICAL_DIRECTIONS,
                        workspace.relation_kinds[primary],
                        workspace.physical_masks[primary],
                        workspace.source_counts[primary],
                        workspace.source_slots[primary],
                        workspace.action_target_lower[0],
                        workspace.action_target_upper[0],
                        *(value[0] for value in workspace.cwp_outputs),
                        *workspace.csp_outputs,
                    )
                if (
                    transfer_count < 1
                    or transfer_count > record_count
                    or record_count > PLAN_CAPACITY
                ):
                    raise RuntimeError("CSP action record counts are invalid")
                for record in range(transfer_count):
                    slot = int(workspace.csp_outputs[0][record])
                    flag = int(workspace.csp_outputs[1][record])
                    if slot < 0 or slot >= selected_count:
                        raise RuntimeError("CSP transfer source slot is out of range")
                    if flag not in (0, 1):
                        raise RuntimeError("CSP transfer source flag is invalid")

                physical = slice(transfer_count, record_count)
                _validate_physical_widening(
                    workspace.coarse_workspace,
                    0,
                    workspace.csp_outputs[10][physical],
                    workspace.csp_outputs[11][physical],
                    workspace.csp_outputs[12][physical],
                    workspace.csp_outputs[3][physical],
                    workspace.csp_outputs[2][physical],
                    workspace.csp_outputs[6][physical],
                    workspace.csp_outputs[7][physical],
                    workspace.csp_outputs[8][physical],
                    workspace.csp_outputs[9][physical],
                    boundary_modes,
                    normal_field_slots,
                )
                prolong_cartesian_2to1_into(
                    workspace.coarse_workspace[:, empty_fields],
                    workspace.cwp_outputs[4][0],
                    workspace.cwp_outputs[5][0],
                    workspace.cwp_outputs[6][0],
                    workspace.payload[primary : primary + 1, empty_fields],
                    target_lower,
                    target_upper,
                    interior_lower,
                )
            elif kind == RELATION_PHYSICAL:
                raise RuntimeError("unmasked noncenter action cannot be PHYSICAL")
            else:
                raise RuntimeError("unknown refined relation kind")

        pwa_count = _prepare_primary_pwa(
            workspace, primary, interior_lower, interior_upper
        )
        physical = slice(0, pwa_count)
        _validate_physical_widening(
            workspace.payload[:selected_count],
            primary,
            workspace.pwa_logical_lower[physical],
            workspace.pwa_logical_upper[physical],
            workspace.pwa_offsets[physical],
            workspace.pwa_directions[physical],
            workspace.pwa_masks[physical],
            workspace.pwa_base_lower[physical],
            workspace.pwa_base_upper[physical],
            workspace.pwa_target_lower[physical],
            workspace.pwa_target_upper[physical],
            boundary_modes,
            normal_field_slots,
        )


def _fill_owned_coarser_plan(workspace, primary, interior_lower, interior_upper):
    """Consume allocator, checked CWP and relation-row proofs; check geometry."""
    boxes = tuple(tuple(int(v) for v in box[0]) for box in workspace.cwp_outputs)
    _validate_plan_geometry(
        tuple(int(v) for v in interior_lower),
        tuple(int(v) for v in interior_upper),
        primary,
        int(workspace.action_phases[0]),
        tuple(int(v) for v in workspace.action_directions[0]),
        workspace.relation_kinds[primary],
        workspace.physical_masks[primary],
        workspace.source_slots[primary],
        *boxes,
    )
    return fill_coarser_slope_support_plan_unchecked(
        interior_lower,
        interior_upper,
        primary,
        int(workspace.action_phases[0]),
        workspace.action_directions[0],
        workspace.relation_kinds[primary],
        workspace.physical_masks[primary],
        workspace.source_slots[primary],
        *(box[0] for box in workspace.cwp_outputs),
        *workspace.csp_outputs,
    )


def _preflight_chunk_actions(*args):
    """Validate actions with invocation-local reuse of owned CSP invariants."""
    return _preflight_chunk_actions_m1_reference(*args, _owned=True)


def _apply_chunk_actions_unchecked(
    workspace: _RHEWorkspace,
    primary_count: int,
    selected_count: int,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
    *,
    _direct_applied: bool = False,
) -> None:
    for primary in range(primary_count):
        for direction_row in range(_DIRECTION_COUNT):
            if int(workspace.physical_masks[primary, direction_row]) != 0:
                continue
            target_lower = workspace.target_lower[direction_row]
            target_upper = workspace.target_upper[direction_row]
            if not _nonempty(target_lower, target_upper):
                continue
            kind = int(workspace.relation_kinds[primary, direction_row])
            if _direct_applied and kind in (RELATION_SAME, RELATION_FINER):
                continue
            source_count = int(workspace.source_counts[primary, direction_row])
            _prepare_action_rows(workspace, direction_row, max(source_count, 1))

            if kind == RELATION_SAME:
                fill_same_level_source_boxes_unchecked(
                    interior_lower,
                    interior_upper,
                    workspace.action_directions[:1],
                    workspace.action_target_lower[:1],
                    workspace.action_target_upper[:1],
                    workspace.same_source_lower,
                    workspace.same_source_upper,
                )
                source_slot = int(
                    workspace.source_slots[primary, direction_row, 0]
                )
                copy_region_into_unchecked(
                    workspace.payload[source_slot : source_slot + 1],
                    workspace.same_source_lower[0],
                    workspace.payload[primary : primary + 1],
                    target_lower,
                    np.subtract(
                        workspace.same_source_upper[0],
                        workspace.same_source_lower[0],
                        out=workspace.copy_extent,
                    ),
                )
            elif kind == RELATION_FINER:
                workspace.action_phases[:source_count] = workspace.phase_codes[
                    primary, direction_row, :source_count
                ]
                fill_finer_restriction_boxes_unchecked(
                    interior_lower,
                    interior_upper,
                    workspace.action_directions[:source_count],
                    workspace.action_phases[:source_count],
                    workspace.action_target_lower[:source_count],
                    workspace.action_target_upper[:source_count],
                    workspace.finer_target_lower[:source_count],
                    workspace.finer_target_upper[:source_count],
                    workspace.finer_source_lower[:source_count],
                    workspace.finer_source_upper[:source_count],
                )
                for source in range(source_count):
                    source_slot = int(
                        workspace.source_slots[
                            primary, direction_row, source
                        ]
                    )
                    restrict_cartesian_2to1_into_unchecked(
                        workspace.payload[source_slot : source_slot + 1],
                        workspace.finer_source_lower[source],
                        workspace.finer_source_upper[source],
                        workspace.payload[primary : primary + 1],
                        workspace.finer_target_lower[source],
                    )
            elif kind == RELATION_COARSER:
                workspace.action_phases[0] = workspace.phase_codes[
                    primary, direction_row, 0
                ]
                fill_coarser_workspace_boxes_unchecked(
                    interior_lower,
                    interior_upper,
                    workspace.action_directions[:1],
                    workspace.action_phases[:1],
                    workspace.action_target_lower[:1],
                    workspace.action_target_upper[:1],
                    *workspace.cwp_outputs,
                )
                transfer_count, record_count = (
                    fill_coarser_slope_support_plan_unchecked(
                        interior_lower,
                        interior_upper,
                        primary,
                        workspace.action_phases[0],
                        workspace.action_directions[0],
                        workspace.relation_kinds[primary],
                        workspace.physical_masks[primary],
                        workspace.source_slots[primary],
                        *(value[0] for value in workspace.cwp_outputs),
                        *workspace.csp_outputs,
                    )
                )
                _apply_coarser_workspace_plan_unchecked(
                    workspace.payload[:selected_count],
                    workspace.coarse_workspace,
                    interior_lower,
                    interior_upper,
                    workspace.cwp_outputs[4][0],
                    workspace.cwp_outputs[5][0],
                    int(transfer_count),
                    int(record_count),
                    *workspace.csp_outputs,
                    boundary_modes,
                    normal_field_slots,
                )
                prolong_cartesian_2to1_into_unchecked(
                    workspace.coarse_workspace,
                    workspace.cwp_outputs[6][0],
                    workspace.payload[primary : primary + 1],
                    target_lower,
                    target_upper,
                    interior_lower,
                )

    # Complete every primary's nonphysical base before widening any primary.
    for primary in range(primary_count):
        pwa_count = _prepare_primary_pwa(
            workspace, primary, interior_lower, interior_upper
        )
        physical = slice(0, pwa_count)
        apply_cartesian_physical_widening_unchecked(
            workspace.payload[:selected_count],
            primary,
            workspace.pwa_logical_lower[physical],
            workspace.pwa_logical_upper[physical],
            workspace.pwa_offsets[physical],
            workspace.pwa_directions[physical],
            workspace.pwa_masks[physical],
            workspace.pwa_base_lower[physical],
            workspace.pwa_base_upper[physical],
            workspace.pwa_target_lower[physical],
            workspace.pwa_target_upper[physical],
            boundary_modes,
            normal_field_slots,
        )


def _prepare_refined_halo_chunk(
    workspace: _RHEWorkspace,
    candidates: np.ndarray,
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    lower_halo: np.ndarray,
    interior_upper: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
    *,
    validate_actions: bool,
) -> tuple[int, int]:
    candidate_count = int(candidates.shape[0])
    fill_balanced_refined_relations_unchecked(
        root_shape,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        candidates,
        CANONICAL_DIRECTIONS,
        workspace.relation_kinds[:candidate_count],
        workspace.physical_masks[:candidate_count],
        workspace.source_counts[:candidate_count],
        workspace.source_leaf_ids[:candidate_count],
    )
    primary_count, selected_count = plan_selected_refined_support_prefix_unchecked(
        candidates,
        workspace.source_counts[:candidate_count],
        workspace.source_leaf_ids[:candidate_count],
        workspace.selected_leaf_ids,
    )
    primary_count = int(primary_count)
    selected_count = int(selected_count)
    resolve_refined_relation_source_slots_unchecked(
        workspace.selected_leaf_ids[:selected_count],
        workspace.source_counts[:primary_count],
        workspace.source_leaf_ids[:primary_count],
        workspace.source_slots[:primary_count],
    )
    _guard_relation_source_slots_before_phase(
        workspace,
        primary_count,
        selected_count,
    )
    fill_refined_relation_phase_codes_unchecked(
        workspace.selected_leaf_ids[:selected_count],
        node_coords,
        leaf_node_ids,
        workspace.relation_kinds[:primary_count],
        workspace.source_counts[:primary_count],
        workspace.source_slots[:primary_count],
        workspace.phase_codes[:primary_count],
    )
    if validate_actions:
        _preflight_chunk_actions(
            workspace,
            primary_count,
            selected_count,
            lower_halo,
            interior_upper,
            boundary_modes,
            normal_field_slots,
        )
    return primary_count, selected_count
