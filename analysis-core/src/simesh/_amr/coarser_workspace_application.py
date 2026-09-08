"""Validated CWA-001 application of one explicit COARSER workspace plan."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from simesh._kernels.primitives._foundation import copy_region_into_unchecked
from simesh._kernels.primitives._physical_widening import apply_cartesian_physical_widening_unchecked
from simesh._kernels.primitives._restriction import restrict_cartesian_2to1_into_unchecked
from simesh._amr.coarser_support import NO_SOURCE, PLAN_CAPACITY, SOURCE_COARSE, SOURCE_FINE
from simesh._amr.foundation import INDEX_DTYPE, _require_index_triplet, _require_payload
from simesh._amr.halos import _require_modes
from simesh._amr.physical_widening import _validate_physical_widening
from simesh._amr.workspace import _require_nonnegative_integer


_UINT8_DTYPE = np.dtype(np.uint8)


def _require_plan_array(
    name: str,
    value: np.ndarray,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype.name}")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _box_overlap(
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
) -> bool:
    return all(
        max(int(left_lower[axis]), int(right_lower[axis]))
        < min(int(left_upper[axis]), int(right_upper[axis]))
        for axis in range(3)
    )


def _box_contains(
    outer_lower: np.ndarray,
    outer_upper: np.ndarray,
    inner_lower: np.ndarray,
    inner_upper: np.ndarray,
) -> bool:
    return all(
        int(outer_lower[axis]) <= int(inner_lower[axis])
        and int(inner_upper[axis]) <= int(outer_upper[axis])
        for axis in range(3)
    )


def _box_volume(lower: np.ndarray, upper: np.ndarray) -> int:
    result = 1
    for axis in range(3):
        result *= int(upper[axis]) - int(lower[axis])
    return result


def _validate_coarser_workspace_plan(
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
) -> tuple:
    """Validate and normalize one complete CWA call without writing values."""
    selected_payload = _require_payload(
        "selected_payload", selected_payload, writable=False
    )
    coarse_workspace = _require_payload(
        "coarse_workspace", coarse_workspace, writable=True
    )
    interior_lower = _require_index_triplet("interior_lower", interior_lower)
    interior_upper = _require_index_triplet("interior_upper", interior_upper)
    workspace_required_lower = _require_index_triplet(
        "workspace_required_lower", workspace_required_lower
    )
    workspace_required_upper = _require_index_triplet(
        "workspace_required_upper", workspace_required_upper
    )
    transfer_count = _require_nonnegative_integer(
        "transfer_count", transfer_count
    )
    record_count = _require_nonnegative_integer("record_count", record_count)

    plan_source_slots = _require_plan_array(
        "plan_source_slots", plan_source_slots, INDEX_DTYPE, (PLAN_CAPACITY,)
    )
    plan_source_is_fine = _require_plan_array(
        "plan_source_is_fine",
        plan_source_is_fine,
        _UINT8_DTYPE,
        (PLAN_CAPACITY,),
    )
    plan_physical_masks = _require_plan_array(
        "plan_physical_masks",
        plan_physical_masks,
        _UINT8_DTYPE,
        (PLAN_CAPACITY,),
    )
    matrix_shape = (PLAN_CAPACITY, 3)
    plan_directions = _require_plan_array(
        "plan_directions", plan_directions, INDEX_DTYPE, matrix_shape
    )
    plan_source_lower = _require_plan_array(
        "plan_source_lower", plan_source_lower, INDEX_DTYPE, matrix_shape
    )
    plan_source_upper = _require_plan_array(
        "plan_source_upper", plan_source_upper, INDEX_DTYPE, matrix_shape
    )
    plan_base_lower = _require_plan_array(
        "plan_base_lower", plan_base_lower, INDEX_DTYPE, matrix_shape
    )
    plan_base_upper = _require_plan_array(
        "plan_base_upper", plan_base_upper, INDEX_DTYPE, matrix_shape
    )
    plan_target_lower = _require_plan_array(
        "plan_target_lower", plan_target_lower, INDEX_DTYPE, matrix_shape
    )
    plan_target_upper = _require_plan_array(
        "plan_target_upper", plan_target_upper, INDEX_DTYPE, matrix_shape
    )
    plan_logical_interior_lower = _require_plan_array(
        "plan_logical_interior_lower",
        plan_logical_interior_lower,
        INDEX_DTYPE,
        matrix_shape,
    )
    plan_logical_interior_upper = _require_plan_array(
        "plan_logical_interior_upper",
        plan_logical_interior_upper,
        INDEX_DTYPE,
        matrix_shape,
    )
    plan_storage_logical_offsets = _require_plan_array(
        "plan_storage_logical_offsets",
        plan_storage_logical_offsets,
        INDEX_DTYPE,
        matrix_shape,
    )
    boundary_modes = _require_modes(boundary_modes, selected_payload.shape[1])
    normal_field_slots = _require_index_triplet(
        "normal_field_slots", normal_field_slots
    )

    if coarse_workspace.shape[0] != 1:
        raise ValueError("coarse_workspace must have exactly one slot")
    if coarse_workspace.shape[1] != selected_payload.shape[1]:
        raise ValueError(
            "selected_payload and coarse_workspace field extents must match"
        )
    if transfer_count > record_count or record_count > PLAN_CAPACITY:
        raise ValueError(
            "counts must satisfy 0 <= transfer_count <= record_count <= 18"
        )

    selected_shape = tuple(int(value) for value in selected_payload.shape[2:])
    workspace_shape = tuple(int(value) for value in coarse_workspace.shape[2:])
    for axis in range(3):
        lower = int(interior_lower[axis])
        upper = int(interior_upper[axis])
        if lower < 0 or lower >= upper or upper > selected_shape[axis]:
            raise ValueError(
                "interior must be a nonempty selected-payload-contained box"
            )
        required_lower = int(workspace_required_lower[axis])
        required_upper = int(workspace_required_upper[axis])
        if (
            required_lower < 0
            or required_lower >= required_upper
            or required_upper > workspace_shape[axis]
        ):
            raise ValueError(
                "workspace_required must be a nonempty workspace-contained box"
            )

    plan_arrays = (
        plan_source_slots,
        plan_source_is_fine,
        plan_physical_masks,
        plan_directions,
        plan_source_lower,
        plan_source_upper,
        plan_base_lower,
        plan_base_upper,
        plan_target_lower,
        plan_target_upper,
        plan_logical_interior_lower,
        plan_logical_interior_upper,
        plan_storage_logical_offsets,
    )
    metadata = (
        interior_lower,
        interior_upper,
        workspace_required_lower,
        workspace_required_upper,
        *plan_arrays,
        boundary_modes,
        normal_field_slots,
    )
    if np.shares_memory(coarse_workspace, selected_payload) or any(
        np.shares_memory(coarse_workspace, value) for value in metadata
    ):
        raise ValueError(
            "coarse_workspace must not overlap selected payload or metadata"
        )
    if any(
        np.shares_memory(left, right)
        for left, right in combinations(plan_arrays, 2)
    ):
        raise ValueError("CSP plan arrays must be pairwise nonoverlapping")

    required_volume = _box_volume(
        workspace_required_lower, workspace_required_upper
    )
    active_volume = 0
    for record in range(record_count):
        for axis in range(3):
            target_lower = int(plan_target_lower[record, axis])
            target_upper = int(plan_target_upper[record, axis])
            if (
                target_lower < int(workspace_required_lower[axis])
                or target_lower >= target_upper
                or target_upper > int(workspace_required_upper[axis])
            ):
                raise ValueError(
                    "active target boxes must be nonempty and contained in required"
                )
            direction = int(plan_directions[record, axis])
            if direction < -1 or direction > 1:
                raise ValueError("plan direction component is outside [-1,1]")
        active_volume += _box_volume(
            plan_target_lower[record], plan_target_upper[record]
        )

    for left in range(record_count):
        for right in range(left + 1, record_count):
            if _box_overlap(
                plan_target_lower[left],
                plan_target_upper[left],
                plan_target_lower[right],
                plan_target_upper[right],
            ):
                raise ValueError("active target boxes must be pairwise disjoint")
    if active_volume != required_volume:
        raise ValueError("active target boxes must exactly cover workspace_required")

    for record in range(transfer_count):
        slot = int(plan_source_slots[record])
        flag = int(plan_source_is_fine[record])
        if slot < 0 or slot >= selected_payload.shape[0]:
            raise ValueError("transfer source slot is out of range")
        if flag not in (int(SOURCE_COARSE), int(SOURCE_FINE)):
            raise ValueError("transfer source_is_fine must be zero or one")
        if int(plan_physical_masks[record]) != 0:
            raise ValueError("transfer physical mask must be zero")
        if not np.array_equal(
            plan_base_lower[record], plan_target_lower[record]
        ) or not np.array_equal(
            plan_base_upper[record], plan_target_upper[record]
        ):
            raise ValueError("transfer base must equal its target")
        if (
            np.any(plan_logical_interior_lower[record] != 0)
            or np.any(plan_logical_interior_upper[record] != 0)
            or np.any(plan_storage_logical_offsets[record] != 0)
        ):
            raise ValueError("transfer logical metadata must be zero")

        for axis in range(3):
            source_lower = int(plan_source_lower[record, axis])
            source_upper = int(plan_source_upper[record, axis])
            if (
                source_lower < int(interior_lower[axis])
                or source_lower >= source_upper
                or source_upper > int(interior_upper[axis])
            ):
                raise ValueError(
                    "transfer source box must be contained in the valid interior"
                )
            source_extent = source_upper - source_lower
            target_extent = int(plan_target_upper[record, axis]) - int(
                plan_target_lower[record, axis]
            )
            expected_extent = (2 if flag == int(SOURCE_FINE) else 1) * target_extent
            if source_extent != expected_extent:
                raise ValueError("transfer source/target extent ratio is invalid")

    for record in range(transfer_count, record_count):
        if int(plan_source_slots[record]) != -1:
            raise ValueError("physical source slot must be -1")
        if int(plan_source_is_fine[record]) != int(NO_SOURCE):
            raise ValueError("physical source flag must be 255")
        mask = int(plan_physical_masks[record])
        if mask == 0 or mask > 7:
            raise ValueError("physical mask must be nonzero and use bits 0..2")
        if np.any(plan_source_lower[record] != 0) or np.any(
            plan_source_upper[record] != 0
        ):
            raise ValueError("physical source boxes must be zero")
        if not any(
            _box_contains(
                plan_target_lower[owner],
                plan_target_upper[owner],
                plan_base_lower[record],
                plan_base_upper[record],
            )
            for owner in range(transfer_count)
        ):
            raise ValueError("physical base must lie in a completed transfer target")

    for record in range(record_count, PLAN_CAPACITY):
        if int(plan_source_slots[record]) != -1:
            raise ValueError("inactive source slot must be -1")
        if int(plan_source_is_fine[record]) != int(NO_SOURCE):
            raise ValueError("inactive source flag must be 255")
        if int(plan_physical_masks[record]) != 0:
            raise ValueError("inactive physical mask must be zero")
        if any(np.any(value[record] != 0) for value in plan_arrays[3:]):
            raise ValueError("inactive plan metadata must be zero")

    for physical_record in range(transfer_count, record_count):
        for base_record in range(record_count):
            if _box_overlap(
                plan_target_lower[physical_record],
                plan_target_upper[physical_record],
                plan_base_lower[base_record],
                plan_base_upper[base_record],
            ):
                raise ValueError("physical targets must not overlap any active base")

    physical_slice = slice(transfer_count, record_count)
    _validate_physical_widening(
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

    return (
        selected_payload,
        coarse_workspace,
        interior_lower,
        interior_upper,
        workspace_required_lower,
        workspace_required_upper,
        transfer_count,
        record_count,
        *plan_arrays,
        boundary_modes,
        normal_field_slots,
    )


def _apply_coarser_workspace_plan_unchecked(
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
    """Apply one already validated CWA plan without further checks."""
    del interior_lower, interior_upper
    del workspace_required_lower, workspace_required_upper

    for record in range(transfer_count):
        source_slot = int(plan_source_slots[record])
        source = selected_payload[source_slot : source_slot + 1]
        if int(plan_source_is_fine[record]) == int(SOURCE_FINE):
            restrict_cartesian_2to1_into_unchecked(
                source,
                plan_source_lower[record],
                plan_source_upper[record],
                coarse_workspace,
                plan_target_lower[record],
            )
        else:
            extent = plan_target_upper[record] - plan_target_lower[record]
            copy_region_into_unchecked(
                source,
                plan_source_lower[record],
                coarse_workspace,
                plan_target_lower[record],
                extent,
            )

    physical_slice = slice(transfer_count, record_count)
    apply_cartesian_physical_widening_unchecked(
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


def apply_coarser_workspace_plan(
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
    """Assemble one complete rectangular PRL workspace from an explicit plan."""
    normalized = _validate_coarser_workspace_plan(
        selected_payload,
        coarse_workspace,
        interior_lower,
        interior_upper,
        workspace_required_lower,
        workspace_required_upper,
        transfer_count,
        record_count,
        plan_source_slots,
        plan_source_is_fine,
        plan_physical_masks,
        plan_directions,
        plan_source_lower,
        plan_source_upper,
        plan_base_lower,
        plan_base_upper,
        plan_target_lower,
        plan_target_upper,
        plan_logical_interior_lower,
        plan_logical_interior_upper,
        plan_storage_logical_offsets,
        boundary_modes,
        normal_field_slots,
    )
    _apply_coarser_workspace_plan_unchecked(*normalized)
