"""Validated PWA-001 Cartesian physical widening application."""

from __future__ import annotations

from typing import Final

import numpy as np

from simesh._kernels.primitives._physical_widening import apply_cartesian_physical_widening_unchecked
from simesh._amr.foundation import _require_index_triplet, _require_payload
from simesh._amr.halos import BoundaryMode, _require_modes
from simesh._amr.same_level_boxes import _require_rows


_INDEX_MIN: Final = int(np.iinfo(np.int64).min)
_INDEX_MAX: Final = int(np.iinfo(np.int64).max)


def _require_integer_scalar(name: str, value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer scalar")
    result = int(value)
    if result < _INDEX_MIN or result > _INDEX_MAX:
        raise OverflowError(f"{name} does not fit in int64")
    return result


def _require_masks(value: np.ndarray, row_count: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("physical_masks must be a NumPy array")
    if value.dtype != np.dtype(np.uint8):
        raise TypeError("physical_masks must have dtype uint8")
    if value.shape != (row_count,):
        raise ValueError(
            f"physical_masks must have shape ({row_count},), got {value.shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError("physical_masks must be C-contiguous")
    return value


def _checked_index(name: str, value: int) -> int:
    if value < _INDEX_MIN or value > _INDEX_MAX:
        raise OverflowError(f"{name} does not fit in int64")
    return value


def _boxes_overlap(
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


def _validate_physical_widening(
    payload: np.ndarray,
    target_slot: int,
    logical_interior_lower: np.ndarray,
    logical_interior_upper: np.ndarray,
    storage_logical_offsets: np.ndarray,
    directions: np.ndarray,
    physical_masks: np.ndarray,
    base_lower: np.ndarray,
    base_upper: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> tuple:
    """Validate and normalize a complete PWA call without mutating payload."""
    payload = _require_payload("payload", payload, writable=True)
    target_slot = _require_integer_scalar("target_slot", target_slot)
    logical_interior_lower = _require_rows(
        "logical_interior_lower", logical_interior_lower
    )
    row_shape = logical_interior_lower.shape
    row_count = row_shape[0]
    logical_interior_upper = _require_rows(
        "logical_interior_upper", logical_interior_upper, row_shape
    )
    storage_logical_offsets = _require_rows(
        "storage_logical_offsets", storage_logical_offsets, row_shape
    )
    directions = _require_rows("directions", directions, row_shape)
    physical_masks = _require_masks(physical_masks, row_count)
    base_lower = _require_rows("base_lower", base_lower, row_shape)
    base_upper = _require_rows("base_upper", base_upper, row_shape)
    target_lower = _require_rows("target_lower", target_lower, row_shape)
    target_upper = _require_rows("target_upper", target_upper, row_shape)
    boundary_modes = _require_modes(boundary_modes, payload.shape[1])
    normal_field_slots = _require_index_triplet(
        "normal_field_slots", normal_field_slots
    )

    if target_slot < 0 or target_slot >= payload.shape[0]:
        raise ValueError("target_slot is outside the payload slot axis")

    metadata = (
        logical_interior_lower,
        logical_interior_upper,
        storage_logical_offsets,
        directions,
        physical_masks,
        base_lower,
        base_upper,
        target_lower,
        target_upper,
        boundary_modes,
        normal_field_slots,
    )
    if any(np.shares_memory(payload, value) for value in metadata):
        raise ValueError("payload must not overlap physical-widening metadata")

    field_count = payload.shape[1]
    for field in range(field_count):
        for face in range(6):
            if int(boundary_modes[field, face]) > BoundaryMode.NO_INFLOW:
                raise ValueError("boundary_modes contains an unknown mode code")
    if any(
        int(normal_field_slots[axis]) < -1
        or int(normal_field_slots[axis]) >= field_count
        for axis in range(3)
    ):
        raise ValueError("normal_field_slots entries must be -1 or field positions")
    for axis in range(3):
        has_noinflow = any(
            int(boundary_modes[field, 2 * axis]) == BoundaryMode.NO_INFLOW
            or int(boundary_modes[field, 2 * axis + 1])
            == BoundaryMode.NO_INFLOW
            for field in range(field_count)
        )
        if has_noinflow and int(normal_field_slots[axis]) < 0:
            raise ValueError("no-inflow mode requires an explicit normal field slot")

    spatial_shape = tuple(int(size) for size in payload.shape[2:])
    for row in range(row_count):
        mask = int(physical_masks[row])
        if mask == 0 or mask > 7:
            raise ValueError("physical mask must be a nonzero value using bits 0..2")

        direction_mask = 0
        spatially_empty = False
        translated_target_lower = [0, 0, 0]
        translated_target_upper = [0, 0, 0]
        for axis in range(3):
            interior_start = int(logical_interior_lower[row, axis])
            interior_stop = int(logical_interior_upper[row, axis])
            if interior_start < 0 or interior_start >= interior_stop:
                raise ValueError("logical interiors must be nonempty and nonnegative")

            direction = int(directions[row, axis])
            if direction < -1 or direction > 1:
                raise ValueError("direction component is outside [-1,1]")
            if direction != 0:
                direction_mask |= 1 << axis

            base_start = int(base_lower[row, axis])
            base_stop = int(base_upper[row, axis])
            target_start = int(target_lower[row, axis])
            target_stop = int(target_upper[row, axis])
            if not (0 <= base_start < base_stop <= spatial_shape[axis]):
                raise ValueError("base boxes must be nonempty and storage-contained")
            if not (0 <= target_start <= target_stop <= spatial_shape[axis]):
                raise ValueError("target boxes must be ordered and storage-contained")
            spatially_empty = spatially_empty or target_start == target_stop

            offset = int(storage_logical_offsets[row, axis])
            translated_target_lower[axis] = _checked_index(
                "forward target translation", target_start + offset
            )
            translated_target_upper[axis] = _checked_index(
                "forward target translation", target_stop + offset
            )

        if mask & ~direction_mask:
            raise ValueError("physical mask marks a zero direction component")

        for axis in range(3):
            bit = 1 << axis
            if (mask & bit) == 0:
                if not (
                    int(base_lower[row, axis])
                    <= int(target_lower[row, axis])
                    <= int(target_upper[row, axis])
                    <= int(base_upper[row, axis])
                ):
                    raise ValueError("unmasked target interval must lie in the base")
                continue
            direction = int(directions[row, axis])
            if direction < 0:
                if (
                    translated_target_upper[axis]
                    != int(logical_interior_lower[row, axis])
                ):
                    raise ValueError("lower physical target is not boundary-anchored")
            elif (
                translated_target_lower[axis]
                != int(logical_interior_upper[row, axis])
            ):
                raise ValueError("upper physical target is not boundary-anchored")

        if spatially_empty:
            continue

        for field in range(field_count):
            for axis in range(3):
                if (mask & (1 << axis)) == 0:
                    continue
                lower = int(logical_interior_lower[row, axis])
                upper = int(logical_interior_upper[row, axis])
                offset = int(storage_logical_offsets[row, axis])
                target_start = translated_target_lower[axis]
                target_stop = translated_target_upper[axis]
                direction = int(directions[row, axis])
                face = 2 * axis + (1 if direction > 0 else 0)
                mode = int(boundary_modes[field, face])
                width = target_stop - target_start
                if mode in (
                    BoundaryMode.SYMMETRIC,
                    BoundaryMode.ANTISYMMETRIC,
                ) and width > upper - lower:
                    raise ValueError(
                        "reflected target depth exceeds logical interior extent"
                    )

                if mode in (
                    BoundaryMode.SYMMETRIC,
                    BoundaryMode.ANTISYMMETRIC,
                ):
                    if direction < 0:
                        logical_source_start = lower
                        logical_source_stop = lower + width
                    else:
                        logical_source_start = upper - width
                        logical_source_stop = upper
                elif direction < 0:
                    logical_source_start = lower
                    logical_source_stop = lower + 1
                else:
                    logical_source_start = upper - 1
                    logical_source_stop = upper

                source_start = _checked_index(
                    "reverse source translation", logical_source_start - offset
                )
                source_stop = _checked_index(
                    "reverse source translation", logical_source_stop - offset
                )
                if not (
                    int(base_lower[row, axis])
                    <= source_start
                    <= source_stop
                    <= int(base_upper[row, axis])
                ):
                    raise ValueError("mapped source image lies outside the base")

    for left in range(row_count):
        for right in range(left + 1, row_count):
            if _boxes_overlap(
                target_lower[left],
                target_upper[left],
                target_lower[right],
                target_upper[right],
            ):
                raise ValueError("physical target boxes must be pairwise disjoint")
    for target_row in range(row_count):
        for base_row in range(row_count):
            if _boxes_overlap(
                target_lower[target_row],
                target_upper[target_row],
                base_lower[base_row],
                base_upper[base_row],
            ):
                raise ValueError("physical targets must not overlap any batch base")

    return (
        payload,
        target_slot,
        logical_interior_lower,
        logical_interior_upper,
        storage_logical_offsets,
        directions,
        physical_masks,
        base_lower,
        base_upper,
        target_lower,
        target_upper,
        boundary_modes,
        normal_field_slots,
    )


def apply_cartesian_physical_widening(
    payload: np.ndarray,
    target_slot: int,
    logical_interior_lower: np.ndarray,
    logical_interior_upper: np.ndarray,
    storage_logical_offsets: np.ndarray,
    directions: np.ndarray,
    physical_masks: np.ndarray,
    base_lower: np.ndarray,
    base_upper: np.ndarray,
    target_lower: np.ndarray,
    target_upper: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> None:
    """Widen complete valid bases into disjoint physical target boxes."""
    normalized = _validate_physical_widening(
        payload,
        target_slot,
        logical_interior_lower,
        logical_interior_upper,
        storage_logical_offsets,
        directions,
        physical_masks,
        base_lower,
        base_upper,
        target_lower,
        target_upper,
        boundary_modes,
        normal_field_slots,
    )
    apply_cartesian_physical_widening_unchecked(*normalized)
