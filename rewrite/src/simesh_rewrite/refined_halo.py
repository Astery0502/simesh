"""RHE-001 bounded selected Cartesian 3D refined-halo execution."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import NamedTuple

import numpy as np

from ._coarser_support import fill_coarser_slope_support_plan_unchecked
from ._coarser_workspace import fill_coarser_workspace_boxes_unchecked
from ._finer_boxes import fill_finer_restriction_boxes_unchecked
from ._foundation import copy_region_into_unchecked
from ._physical_widening import apply_cartesian_physical_widening_unchecked
from ._prolongation import prolong_cartesian_2to1_into_unchecked
from ._refined_support import plan_selected_refined_support_prefix_unchecked
from ._relation_phases import fill_refined_relation_phase_codes_unchecked
from ._relation_slots import resolve_refined_relation_source_slots_unchecked
from ._relations import fill_balanced_refined_relations_unchecked
from ._restriction import restrict_cartesian_2to1_into_unchecked
from ._same_level_boxes import fill_same_level_source_boxes_unchecked
from .blockio import (
    BlockReader,
    BlockWriter,
    _require_block_reader,
    _require_block_writer,
    read_blocks_into,
    write_blocks_from,
)
from .coarser_support import (
    CANONICAL_DIRECTIONS,
    PLAN_CAPACITY,
    fill_coarser_slope_support_plan,
)
from .coarser_workspace import fill_coarser_workspace_boxes
from .coarser_workspace_application import (
    _apply_coarser_workspace_plan_unchecked,
    _validate_coarser_workspace_plan,
)
from .finer_boxes import fill_finer_restriction_boxes
from .foundation import _require_index_triplet, copy_region_into
from .halos import BoundaryMode, _require_modes
from .physical_widening import _validate_physical_widening
from .prolongation import prolong_cartesian_2to1_into
from .relation_phases import fill_refined_relation_phase_codes
from .relation_slots import resolve_refined_relation_source_slots
from .relations import (
    RELATION_COARSER,
    RELATION_FINER,
    RELATION_PHYSICAL,
    RELATION_SAME,
    fill_balanced_refined_relations,
)
from .restriction import restrict_cartesian_2to1_into
from .same_level_boxes import fill_same_level_source_boxes
from .selected_refined_support import (
    _require_primary_selection,
    _validate_primary_selection,
)
from .storage import _require_index_vector
from .target_boxes import fill_directed_halo_target_boxes
from .workspace import _require_nonnegative_integer


_INDEX_MAX = int(np.iinfo(np.int64).max)
_DIRECTION_COUNT = 26
_FINER_SOURCE_CAPACITY = 4


class RefinedHaloExecutionStats(NamedTuple):
    primary_count: int
    chunk_count: int
    reader_calls: int
    writer_calls: int
    selected_load_count: int
    maximum_selected_slots: int
    managed_array_bytes: int


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


def _checked_add(name: str, left: int, right: int) -> int:
    result = left + right
    if result < 0 or result > _INDEX_MAX:
        raise OverflowError(f"{name} does not fit in int64")
    return result


def _checked_product(name: str, values: tuple[int, int, int]) -> int:
    result = 1
    for value in values:
        if value <= 0:
            raise ValueError(f"{name} entries must be positive")
        if result > _INDEX_MAX // value:
            raise OverflowError(f"{name} volume does not fit in int64")
        result *= value
    return result


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


def _validate_boundary_configuration(
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
    field_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    boundary_modes = _require_modes(boundary_modes, field_count)
    normal_field_slots = _require_index_triplet(
        "normal_field_slots", normal_field_slots
    )
    for field in range(field_count):
        for face in range(6):
            if int(boundary_modes[field, face]) > int(BoundaryMode.NO_INFLOW):
                raise ValueError("boundary_modes contains an unknown mode code")
    for axis in range(3):
        normal = int(normal_field_slots[axis])
        if normal < -1 or normal >= field_count:
            raise ValueError(
                "normal_field_slots entries must be -1 or field positions"
            )
        if any(
            int(boundary_modes[field, 2 * axis])
            == int(BoundaryMode.NO_INFLOW)
            or int(boundary_modes[field, 2 * axis + 1])
            == int(BoundaryMode.NO_INFLOW)
            for field in range(field_count)
        ) and normal < 0:
            raise ValueError(
                "no-inflow mode requires an explicit normal field slot"
            )
    return boundary_modes, normal_field_slots


def _validate_external_nonoverlap(
    reader: BlockReader,
    writer: BlockWriter,
    metadata: tuple[np.ndarray, ...],
) -> None:
    inputs = (*reader.memory_arrays, *metadata)
    outputs = writer.memory_arrays
    for index, output in enumerate(outputs):
        if any(np.shares_memory(output, value) for value in inputs):
            raise ValueError("writer memory must not overlap reader or metadata")
        for other in outputs[index + 1 :]:
            if np.shares_memory(output, other):
                raise ValueError("writer memory arrays must be pairwise nonoverlapping")


def _validate_consumer_output_nonoverlap(
    reader: BlockReader,
    output_arrays: tuple[np.ndarray, ...],
    metadata: tuple[np.ndarray, ...],
) -> None:
    inputs = (*reader.memory_arrays, *metadata)
    for index, output in enumerate(output_arrays):
        if not isinstance(output, np.ndarray):
            raise TypeError("consumer output arrays must be NumPy arrays")
        if any(np.shares_memory(output, value) for value in inputs):
            raise ValueError("consumer output must not overlap reader or metadata")
        for other in output_arrays[index + 1 :]:
            if not isinstance(other, np.ndarray):
                raise TypeError("consumer output arrays must be NumPy arrays")
            if np.shares_memory(output, other):
                raise ValueError(
                    "consumer output arrays must be pairwise nonoverlapping"
                )


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


def _preflight_chunk_actions_checked_reference(
    workspace: _RHEWorkspace,
    primary_count: int,
    selected_count: int,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> None:
    empty_fields = slice(0, 0)
    for primary in range(primary_count):
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
                copy_region_into(
                    workspace.payload[source_slot : source_slot + 1, empty_fields],
                    workspace.same_source_lower[0],
                    workspace.payload[primary : primary + 1, empty_fields],
                    target_lower,
                    np.subtract(
                        workspace.same_source_upper[0],
                        workspace.same_source_lower[0],
                        out=workspace.copy_extent,
                    ),
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
                    if source_slot == primary:
                        raise RuntimeError("FINER action source aliases its primary")
                    restrict_cartesian_2to1_into(
                        workspace.payload[
                            source_slot : source_slot + 1, empty_fields
                        ],
                        workspace.finer_source_lower[source],
                        workspace.finer_source_upper[source],
                        workspace.payload[primary : primary + 1, empty_fields],
                        workspace.finer_target_lower[source],
                    )
            elif kind == RELATION_COARSER:
                if source_count != 1:
                    raise RuntimeError("COARSER action has invalid source count")
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
                _validate_coarser_workspace_plan(
                    workspace.payload[:selected_count],
                    workspace.coarse_workspace,
                    interior_lower,
                    interior_upper,
                    workspace.cwp_outputs[4][0],
                    workspace.cwp_outputs[5][0],
                    transfer_count,
                    record_count,
                    *workspace.csp_outputs,
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


def _preflight_chunk_actions(
    workspace: _RHEWorkspace,
    primary_count: int,
    selected_count: int,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> None:
    """Validate actual action geometry without repeating proven value kernels."""
    empty_fields = slice(0, 0)
    for primary in range(primary_count):
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


def _apply_chunk_actions_unchecked(
    workspace: _RHEWorkspace,
    primary_count: int,
    selected_count: int,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
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


def _execute_selected_refined_halos(
    reader: BlockReader,
    writer: BlockWriter | None,
    primary_leaf_ids: np.ndarray,
    field_ids: np.ndarray,
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    lower_halo: np.ndarray,
    upper_halo: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
    slot_capacity: int,
    *,
    completed_primary_consumer=None,
    consumer_output_arrays: tuple[np.ndarray, ...] = (),
    preflight_all_chunks: bool = False,
    additional_managed_array_bytes: int = 0,
) -> RefinedHaloExecutionStats:
    reader = _require_block_reader(reader)
    if completed_primary_consumer is None:
        writer = _require_block_writer(writer)
    else:
        if writer is not None:
            raise RuntimeError("private RHE execution has two terminal actions")
        if not callable(completed_primary_consumer):
            raise TypeError("completed primary consumer must be callable")
        if not isinstance(consumer_output_arrays, tuple):
            raise TypeError("consumer_output_arrays must be a tuple")
        if (
            type(additional_managed_array_bytes) is not int
            or additional_managed_array_bytes < 0
        ):
            raise TypeError("additional managed bytes must be a nonnegative int")
    primary_leaf_ids = _require_primary_selection(primary_leaf_ids)
    field_ids = _require_index_vector("field_ids", field_ids)
    lower_halo = _require_index_triplet("lower_halo", lower_halo)
    upper_halo = _require_index_triplet("upper_halo", upper_halo)
    slot_capacity = _require_nonnegative_integer("slot_capacity", slot_capacity)

    leaf_count = (
        int(leaf_node_ids.shape[0])
        if isinstance(leaf_node_ids, np.ndarray) and leaf_node_ids.ndim == 1
        else reader.shape[0]
    )
    if reader.shape[0] != leaf_count:
        raise ValueError("reader block axis must equal leaf count")
    if any(value > _INDEX_MAX for value in reader.shape):
        raise OverflowError("reader shape does not fit in int64")
    block_shape = tuple(int(value) for value in reader.shape[2:])
    for axis, block in enumerate(block_shape):
        if block < 4 or block % 2:
            raise ValueError("reader block extents must be even and at least four")
        if int(lower_halo[axis]) < 0 or int(upper_halo[axis]) < 0:
            raise ValueError("halo extents must be nonnegative")
        if (
            int(lower_halo[axis]) > block // 2
            or int(upper_halo[axis]) > block // 2
        ):
            raise ValueError("halo extents must not exceed half a block")
    padded_shape = tuple(
        _checked_add(
            "padded shape",
            _checked_add("padded shape", block_shape[axis], int(lower_halo[axis])),
            int(upper_halo[axis]),
        )
        for axis in range(3)
    )

    field_count = int(field_ids.shape[0])
    for position, field_id in enumerate(field_ids):
        if int(field_id) < 0 or int(field_id) >= reader.shape[1]:
            raise ValueError(f"field_ids entry {position} is out of range")
    if writer is not None and writer.shape != (
        leaf_count,
        field_count,
        *padded_shape,
    ):
        raise ValueError(
            "writer shape must be (leaf_count, selected fields, padded block)"
        )
    boundary_modes, normal_field_slots = _validate_boundary_configuration(
        boundary_modes, normal_field_slots, field_count
    )
    _validate_primary_selection(primary_leaf_ids, leaf_count)
    primary_total = int(primary_leaf_ids.shape[0])
    if slot_capacity > leaf_count:
        raise ValueError("slot_capacity exceeds leaf count")
    if primary_total and slot_capacity < min(57, leaf_count):
        raise ValueError("slot_capacity cannot fit the universal all-26 closure")

    metadata = tuple(
        value
        for value in (
            primary_leaf_ids,
            field_ids,
            root_shape,
            coord_to_rank,
            root_node_ids,
            node_levels,
            node_coords,
            child_node_ids,
            node_leaf_ids,
            leaf_node_ids,
            lower_halo,
            upper_halo,
            boundary_modes,
            normal_field_slots,
        )
        if isinstance(value, np.ndarray)
    )
    if writer is not None:
        _validate_external_nonoverlap(reader, writer, metadata)
    else:
        _validate_consumer_output_nonoverlap(
            reader,
            consumer_output_arrays,
            metadata,
        )

    padded_volume = _checked_product("padded shape", padded_shape)
    payload_bytes_per_slot = 8 * field_count * padded_volume
    capacity_bytes_per_slot = payload_bytes_per_slot + 1854
    if capacity_bytes_per_slot > _INDEX_MAX:
        raise OverflowError("managed bytes per slot do not fit in int64")
    if (
        slot_capacity
        and capacity_bytes_per_slot > _INDEX_MAX // slot_capacity
    ):
        raise OverflowError("capacity-dependent managed bytes do not fit in int64")
    if any(block == _INDEX_MAX for block in block_shape):
        raise OverflowError("coarse workspace shape does not fit in int64")
    coarse_shape = tuple(block + 1 for block in block_shape)
    coarse_volume = _checked_product("coarse workspace shape", coarse_shape)
    if field_count and coarse_volume > _INDEX_MAX // (8 * field_count):
        raise OverflowError("coarse workspace bytes do not fit in int64")
    workspace, workspace_arrays = _allocate_workspace(
        slot_capacity, field_count, block_shape, padded_shape
    )
    zero = np.zeros(3, dtype=np.int64)
    block_shape_array = np.asarray(block_shape, dtype=np.int64)
    padded_shape_array = np.asarray(padded_shape, dtype=np.int64)
    interior_upper = np.empty(3, dtype=np.int64)
    np.add(lower_halo, block_shape_array, out=interior_upper)
    local_field_ids = np.arange(field_count, dtype=np.int64)

    fill_directed_halo_target_boxes(
        lower_halo,
        interior_upper,
        zero,
        padded_shape_array,
        CANONICAL_DIRECTIONS,
        workspace.target_lower,
        workspace.target_upper,
    )

    empty_candidates = primary_leaf_ids[:0]
    fill_balanced_refined_relations(
        root_shape,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        empty_candidates,
        CANONICAL_DIRECTIONS,
        workspace.relation_kinds[:0],
        workspace.physical_masks[:0],
        workspace.source_counts[:0],
        workspace.source_leaf_ids[:0],
    )
    resolve_refined_relation_source_slots(
        leaf_count,
        workspace.selected_leaf_ids[:0],
        workspace.source_counts[:0],
        workspace.source_leaf_ids[:0],
        workspace.source_slots[:0],
    )
    fill_refined_relation_phase_codes(
        workspace.selected_leaf_ids[:0],
        node_levels,
        node_coords,
        leaf_node_ids,
        CANONICAL_DIRECTIONS,
        workspace.relation_kinds[:0],
        workspace.physical_masks[:0],
        workspace.source_counts[:0],
        workspace.source_slots[:0],
        workspace.phase_codes[:0],
    )

    if preflight_all_chunks:
        preflight_first = 0
        while preflight_first < primary_total:
            candidate_count = min(
                slot_capacity,
                primary_total - preflight_first,
            )
            candidates = primary_leaf_ids[
                preflight_first : preflight_first + candidate_count
            ]
            primary_count, _ = _prepare_refined_halo_chunk(
                workspace,
                candidates,
                root_shape,
                coord_to_rank,
                root_node_ids,
                node_levels,
                node_coords,
                child_node_ids,
                node_leaf_ids,
                leaf_node_ids,
                lower_halo,
                interior_upper,
                boundary_modes,
                normal_field_slots,
                validate_actions=True,
            )
            preflight_first += primary_count

        private_managed_bytes = sum(
            value.nbytes
            for value in (
                *workspace_arrays,
                zero,
                block_shape_array,
                padded_shape_array,
                interior_upper,
                local_field_ids,
            )
        )
        if private_managed_bytes > _INDEX_MAX - additional_managed_array_bytes:
            raise OverflowError("combined managed raw-array bytes do not fit in int64")

    empty_payload = workspace.payload[:0]
    read_blocks_into(
        reader,
        zero,
        block_shape_array,
        workspace.selected_leaf_ids[:0],
        field_ids,
        empty_payload,
        lower_halo,
    )
    if writer is not None:
        write_blocks_from(
            writer,
            empty_payload,
            zero,
            padded_shape_array,
            workspace.selected_leaf_ids[:0],
            local_field_ids,
            zero,
        )

    capacity_arrays = (
        workspace.selected_leaf_ids,
        workspace.payload,
        workspace.relation_kinds,
        workspace.physical_masks,
        workspace.source_counts,
        workspace.source_leaf_ids,
        workspace.source_slots,
        workspace.phase_codes,
    )
    expected_capacity_bytes = slot_capacity * (
        8 * field_count * padded_volume + 1854
    )
    actual_capacity_bytes = sum(value.nbytes for value in capacity_arrays)
    if actual_capacity_bytes != expected_capacity_bytes:
        raise RuntimeError("internal capacity-dependent byte accounting mismatch")
    managed_arrays = (
        *workspace_arrays,
        zero,
        block_shape_array,
        padded_shape_array,
        interior_upper,
        local_field_ids,
    )
    managed_array_bytes = sum(value.nbytes for value in managed_arrays)
    if managed_array_bytes > _INDEX_MAX:
        raise OverflowError("managed raw-array bytes do not fit in int64")

    first = 0
    chunk_count = 0
    selected_load_count = 0
    maximum_selected_slots = 0
    while first < primary_total:
        candidate_count = min(slot_capacity, primary_total - first)
        candidates = primary_leaf_ids[first : first + candidate_count]
        primary_count, selected_count = _prepare_refined_halo_chunk(
            workspace,
            candidates,
            root_shape,
            coord_to_rank,
            root_node_ids,
            node_levels,
            node_coords,
            child_node_ids,
            node_leaf_ids,
            leaf_node_ids,
            lower_halo,
            interior_upper,
            boundary_modes,
            normal_field_slots,
            validate_actions=not preflight_all_chunks,
        )

        read_blocks_into(
            reader,
            zero,
            block_shape_array,
            workspace.selected_leaf_ids[:selected_count],
            field_ids,
            workspace.payload[:selected_count],
            lower_halo,
        )
        _apply_chunk_actions_unchecked(
            workspace,
            primary_count,
            selected_count,
            lower_halo,
            interior_upper,
            boundary_modes,
            normal_field_slots,
        )
        if writer is not None:
            write_blocks_from(
                writer,
                workspace.payload[:primary_count],
                zero,
                padded_shape_array,
                workspace.selected_leaf_ids[:primary_count],
                local_field_ids,
                zero,
            )
        else:
            consumer_leaf_ids = workspace.selected_leaf_ids[:primary_count].view()
            consumer_payload = workspace.payload[:primary_count].view()
            consumer_valid_lower = zero.view()
            consumer_valid_upper = padded_shape_array.view()
            consumer_interior_lower = lower_halo.view()
            consumer_interior_upper = interior_upper.view()
            for value in (
                consumer_leaf_ids,
                consumer_payload,
                consumer_valid_lower,
                consumer_valid_upper,
                consumer_interior_lower,
                consumer_interior_upper,
            ):
                value.setflags(write=False)
            result = completed_primary_consumer(
                first,
                consumer_leaf_ids,
                consumer_payload,
                consumer_valid_lower,
                consumer_valid_upper,
                consumer_interior_lower,
                consumer_interior_upper,
            )
            if result is not None:
                raise TypeError("completed primary consumer must return None")

        first += primary_count
        chunk_count += 1
        selected_load_count += selected_count
        maximum_selected_slots = max(maximum_selected_slots, selected_count)

    return RefinedHaloExecutionStats(
        primary_total,
        chunk_count,
        chunk_count,
        chunk_count if writer is not None else 0,
        selected_load_count,
        maximum_selected_slots,
        managed_array_bytes,
    )


def execute_selected_refined_halos_from_blocks(
    reader: BlockReader,
    writer: BlockWriter,
    primary_leaf_ids: np.ndarray,
    field_ids: np.ndarray,
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    lower_halo: np.ndarray,
    upper_halo: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
    slot_capacity: int,
) -> RefinedHaloExecutionStats:
    """Execute one bounded selected refined-halo traversal."""
    return _execute_selected_refined_halos(
        reader,
        writer,
        primary_leaf_ids,
        field_ids,
        root_shape,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        lower_halo,
        upper_halo,
        boundary_modes,
        normal_field_slots,
        slot_capacity,
    )


def _execute_selected_refined_halos_with_consumer(
    reader: BlockReader,
    primary_leaf_ids: np.ndarray,
    field_ids: np.ndarray,
    root_shape: np.ndarray,
    coord_to_rank: np.ndarray,
    root_node_ids: np.ndarray,
    node_levels: np.ndarray,
    node_coords: np.ndarray,
    child_node_ids: np.ndarray,
    node_leaf_ids: np.ndarray,
    leaf_node_ids: np.ndarray,
    lower_halo: np.ndarray,
    upper_halo: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
    slot_capacity: int,
    completed_primary_consumer,
    *,
    consumer_output_arrays: tuple[np.ndarray, ...] = (),
    additional_managed_array_bytes: int = 0,
) -> RefinedHaloExecutionStats:
    """Run RHE privately with a synchronous completed-primary consumer."""
    return _execute_selected_refined_halos(
        reader,
        None,
        primary_leaf_ids,
        field_ids,
        root_shape,
        coord_to_rank,
        root_node_ids,
        node_levels,
        node_coords,
        child_node_ids,
        node_leaf_ids,
        leaf_node_ids,
        lower_halo,
        upper_halo,
        boundary_modes,
        normal_field_slots,
        slot_capacity,
        completed_primary_consumer=completed_primary_consumer,
        consumer_output_arrays=consumer_output_arrays,
        preflight_all_chunks=True,
        additional_managed_array_bytes=additional_managed_array_bytes,
    )
