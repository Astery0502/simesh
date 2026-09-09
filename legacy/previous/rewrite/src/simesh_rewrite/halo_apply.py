"""Validated HAX-001 application of explicit level-1 halo plans."""

from __future__ import annotations

import numpy as np

from ._halo_apply import (
    apply_level1_same_level_halo_plan_unchecked,
    invalid_level1_halo_plan_entry_unchecked,
)
from .foundation import INDEX_DTYPE, _require_index_triplet, _require_payload
from .halos import _require_modes


def _require_plan_input(
    name: str,
    value: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype.name}")
    if value.ndim != 2 or value.shape[1] != 27:
        raise ValueError(f"{name} must have shape (primary_count, 27)")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def apply_level1_same_level_halo_plan(
    payload: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    source_slots: np.ndarray,
    physical_masks: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> None:
    """Apply an HPL-001 plan to same-level-dependent primary halo cells."""
    payload = _require_payload("payload", payload, writable=True)
    interior_lower = _require_index_triplet("interior_lower", interior_lower)
    interior_upper = _require_index_triplet("interior_upper", interior_upper)
    source_slots = _require_plan_input(
        "source_slots",
        source_slots,
        INDEX_DTYPE,
    )
    physical_masks = _require_plan_input(
        "physical_masks",
        physical_masks,
        np.dtype(np.uint8),
    )
    boundary_modes = _require_modes(boundary_modes, payload.shape[1])
    normal_field_slots = _require_index_triplet(
        "normal_field_slots", normal_field_slots
    )
    if physical_masks.shape != source_slots.shape:
        raise ValueError("source_slots and physical_masks must have equal shapes")
    if source_slots.shape[0] > payload.shape[0]:
        raise ValueError("plan primary count exceeds payload selected slots")

    spatial_shape = np.asarray(payload.shape[2:], dtype=np.int64)
    if np.any(interior_lower < 0) or np.any(interior_lower >= interior_upper):
        raise ValueError("interior must be a nonempty nonnegative box")
    if np.any(interior_upper > spatial_shape):
        raise ValueError("interior exceeds payload spatial shape")
    interior_extent = interior_upper - interior_lower
    if np.any(interior_lower > interior_extent) or np.any(
        spatial_shape - interior_upper > interior_extent
    ):
        raise ValueError("allocated halo width exceeds interior extent")

    metadata = (
        interior_lower,
        interior_upper,
        source_slots,
        physical_masks,
        boundary_modes,
        normal_field_slots,
    )
    if any(np.shares_memory(payload, value) for value in metadata):
        raise ValueError("payload must not overlap halo-plan metadata")

    invalid_plan = int(
        invalid_level1_halo_plan_entry_unchecked(
            source_slots,
            physical_masks,
            payload.shape[0],
        )
    )
    if invalid_plan >= 0:
        primary, direction = divmod(invalid_plan, 27)
        raise ValueError(
            f"halo plan entry ({primary}, {direction}) is structurally invalid"
        )

    if np.any(normal_field_slots < -1) or np.any(
        normal_field_slots >= payload.shape[1]
    ):
        raise ValueError("normal_field_slots entries must be -1 or field positions")
    for field in range(payload.shape[1]):
        for face in range(6):
            if boundary_modes[field, face] > 3:
                raise ValueError("boundary_modes contains an unknown mode code")
    for axis in range(3):
        has_noinflow = False
        for field in range(payload.shape[1]):
            has_noinflow = has_noinflow or (
                boundary_modes[field, 2 * axis] == 3
            )
            has_noinflow = has_noinflow or (
                boundary_modes[field, 2 * axis + 1] == 3
            )
        if has_noinflow and normal_field_slots[axis] < 0:
            raise ValueError("no-inflow mode requires an explicit normal field slot")

    apply_level1_same_level_halo_plan_unchecked(
        payload,
        interior_lower,
        interior_upper,
        source_slots,
        physical_masks,
        boundary_modes,
        normal_field_slots,
    )
