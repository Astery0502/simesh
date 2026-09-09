"""Validated non-periodic level-1 halo provision."""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from simesh._kernels.primitives._halos import (
    common_physical_valid_region_unchecked,
    duplicate_block_id_index_unchecked,
    fill_physical_halos_unchecked,
    fill_same_level_halos_unchecked,
    missing_halo_closure_primary_unchecked,
)
from simesh._kernels.primitives._storage import validate_indices_unchecked
from simesh._amr.foundation import INDEX_DTYPE, _require_index_triplet, _require_payload


class BoundaryMode(IntEnum):
    CONTINUOUS = 0
    SYMMETRIC = 1
    ANTISYMMETRIC = 2
    NO_INFLOW = 3


_INDEX_MAX = int(np.iinfo(np.int64).max)


def _require_index_vector(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != INDEX_DTYPE:
        raise TypeError(f"{name} must have dtype int64")
    if value.ndim != 1 or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous vector")
    return value


def _require_face_table(face_neighbor_ids: np.ndarray) -> np.ndarray:
    if not isinstance(face_neighbor_ids, np.ndarray):
        raise TypeError("face_neighbor_ids must be a NumPy array")
    if face_neighbor_ids.dtype != INDEX_DTYPE:
        raise TypeError("face_neighbor_ids must have dtype int64")
    if face_neighbor_ids.ndim != 2 or face_neighbor_ids.shape[1] != 6:
        raise ValueError("face_neighbor_ids must have shape (block_count, 6)")
    if not face_neighbor_ids.flags.c_contiguous:
        raise ValueError("face_neighbor_ids must be C-contiguous")
    return face_neighbor_ids


def _require_modes(boundary_modes: np.ndarray, field_count: int) -> np.ndarray:
    if not isinstance(boundary_modes, np.ndarray):
        raise TypeError("boundary_modes must be a NumPy array")
    if boundary_modes.dtype != np.dtype(np.uint8):
        raise TypeError("boundary_modes must have dtype uint8")
    if boundary_modes.shape != (field_count, 6):
        raise ValueError(
            f"boundary_modes must have shape {(field_count, 6)}, "
            f"got {boundary_modes.shape}"
        )
    if not boundary_modes.flags.c_contiguous:
        raise ValueError("boundary_modes must be C-contiguous")
    return boundary_modes


def _validate_configuration(
    payload: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    block_ids: np.ndarray,
    face_neighbor_ids: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> None:
    spatial_shape = np.asarray(payload.shape[2:], dtype=np.int64)
    if np.any(interior_lower < 0) or np.any(interior_lower >= interior_upper):
        raise ValueError("interior must be a nonempty nonnegative box")
    if np.any(interior_upper > spatial_shape):
        raise ValueError("interior exceeds payload spatial shape")
    invalid_block = int(
        validate_indices_unchecked(block_ids, face_neighbor_ids.shape[0])
    )
    if invalid_block >= 0:
        raise ValueError(f"block_ids entry {invalid_block} is out of range")
    for field in range(payload.shape[1]):
        for face in range(6):
            if boundary_modes[field, face] > BoundaryMode.NO_INFLOW:
                raise ValueError("boundary_modes contains an unknown mode code")
    if np.any(normal_field_slots < -1) or np.any(
        normal_field_slots >= payload.shape[1]
    ):
        raise ValueError("normal_field_slots entries must be -1 or field positions")

    interior_extent = interior_upper - interior_lower
    lower_width = interior_lower
    upper_width = spatial_shape - interior_upper
    for axis in range(3):
        lower_reflected = False
        upper_reflected = False
        has_noinflow = False
        for field in range(payload.shape[1]):
            lower_mode = boundary_modes[field, 2 * axis]
            upper_mode = boundary_modes[field, 2 * axis + 1]
            lower_reflected = lower_reflected or lower_mode in (
                BoundaryMode.SYMMETRIC,
                BoundaryMode.ANTISYMMETRIC,
            )
            upper_reflected = upper_reflected or upper_mode in (
                BoundaryMode.SYMMETRIC,
                BoundaryMode.ANTISYMMETRIC,
            )
            has_noinflow = has_noinflow or lower_mode == BoundaryMode.NO_INFLOW
            has_noinflow = has_noinflow or upper_mode == BoundaryMode.NO_INFLOW
        if lower_reflected and lower_width[axis] > interior_extent[axis]:
            raise ValueError("reflected lower halo width exceeds interior extent")
        if upper_reflected and upper_width[axis] > interior_extent[axis]:
            raise ValueError("reflected upper halo width exceeds interior extent")
        if has_noinflow and normal_field_slots[axis] < 0:
            raise ValueError("no-inflow mode requires an explicit normal field slot")


def fill_physical_halos(
    payload: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    block_ids: np.ndarray,
    face_neighbor_ids: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> None:
    """Fill every cell in each slot's physical envelope in place."""
    payload = _require_payload("payload", payload, writable=True)
    interior_lower = _require_index_triplet("interior_lower", interior_lower)
    interior_upper = _require_index_triplet("interior_upper", interior_upper)
    block_ids = _require_index_vector("block_ids", block_ids)
    face_neighbor_ids = _require_face_table(face_neighbor_ids)
    boundary_modes = _require_modes(boundary_modes, payload.shape[1])
    normal_field_slots = _require_index_triplet(
        "normal_field_slots", normal_field_slots
    )
    if payload.shape[0] != block_ids.shape[0]:
        raise ValueError("payload slot axis must match block_ids")

    metadata = (
        interior_lower,
        interior_upper,
        block_ids,
        face_neighbor_ids,
        boundary_modes,
        normal_field_slots,
    )
    if any(np.shares_memory(payload, value) for value in metadata):
        raise ValueError("payload must not overlap halo metadata")
    _validate_configuration(
        payload,
        interior_lower,
        interior_upper,
        block_ids,
        face_neighbor_ids,
        boundary_modes,
        normal_field_slots,
    )
    fill_physical_halos_unchecked(
        payload,
        interior_lower,
        interior_upper,
        block_ids,
        face_neighbor_ids,
        boundary_modes,
        normal_field_slots,
    )


def _require_primary_count(primary_count) -> int:
    if isinstance(primary_count, bool) or not isinstance(
        primary_count, (int, np.integer)
    ):
        raise TypeError("primary_count must be an integer")
    primary_count = int(primary_count)
    if primary_count < 0:
        raise ValueError("primary_count must be non-negative")
    if primary_count > _INDEX_MAX:
        raise OverflowError("primary_count does not fit in int64")
    return primary_count


def fill_same_level_halos(
    payload: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    block_ids: np.ndarray,
    primary_count: int,
    face_neighbor_ids: np.ndarray,
    boundary_modes: np.ndarray,
    normal_field_slots: np.ndarray,
) -> None:
    """Fill same-level-dependent halo cells of the primary slot prefix."""
    payload = _require_payload("payload", payload, writable=True)
    interior_lower = _require_index_triplet("interior_lower", interior_lower)
    interior_upper = _require_index_triplet("interior_upper", interior_upper)
    block_ids = _require_index_vector("block_ids", block_ids)
    primary_count = _require_primary_count(primary_count)
    face_neighbor_ids = _require_face_table(face_neighbor_ids)
    boundary_modes = _require_modes(boundary_modes, payload.shape[1])
    normal_field_slots = _require_index_triplet(
        "normal_field_slots", normal_field_slots
    )
    if payload.shape[0] != block_ids.shape[0]:
        raise ValueError("payload slot axis must match block_ids")
    if primary_count > payload.shape[0]:
        raise ValueError("primary_count exceeds selected slot count")

    metadata = (
        interior_lower,
        interior_upper,
        block_ids,
        face_neighbor_ids,
        boundary_modes,
        normal_field_slots,
    )
    if any(np.shares_memory(payload, value) for value in metadata):
        raise ValueError("payload must not overlap halo metadata")
    _validate_configuration(
        payload,
        interior_lower,
        interior_upper,
        block_ids,
        face_neighbor_ids,
        boundary_modes,
        normal_field_slots,
    )
    spatial_shape = np.asarray(payload.shape[2:], dtype=np.int64)
    interior_extent = interior_upper - interior_lower
    if np.any(interior_lower > interior_extent) or np.any(
        spatial_shape - interior_upper > interior_extent
    ):
        raise ValueError("allocated halo width exceeds interior extent")
    duplicate = int(duplicate_block_id_index_unchecked(block_ids))
    if duplicate >= 0:
        raise ValueError(f"block_ids entry {duplicate} duplicates an earlier ID")
    missing_primary = int(
        missing_halo_closure_primary_unchecked(
            primary_count,
            block_ids,
            face_neighbor_ids,
        )
    )
    if missing_primary >= 0:
        raise ValueError(
            f"primary slot {missing_primary} lacks complete one-block halo closure"
        )

    fill_same_level_halos_unchecked(
        payload,
        interior_lower,
        interior_upper,
        block_ids,
        primary_count,
        face_neighbor_ids,
        boundary_modes,
        normal_field_slots,
    )


def common_physical_valid_region(
    spatial_shape: np.ndarray,
    interior_lower: np.ndarray,
    interior_upper: np.ndarray,
    block_ids: np.ndarray,
    face_neighbor_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the intersection of selected slots' physical envelopes."""
    spatial_shape = _require_index_triplet("spatial_shape", spatial_shape)
    interior_lower = _require_index_triplet("interior_lower", interior_lower)
    interior_upper = _require_index_triplet("interior_upper", interior_upper)
    block_ids = _require_index_vector("block_ids", block_ids)
    face_neighbor_ids = _require_face_table(face_neighbor_ids)
    if np.any(interior_lower < 0) or np.any(interior_lower >= interior_upper):
        raise ValueError("interior must be a nonempty nonnegative box")
    if np.any(interior_upper > spatial_shape):
        raise ValueError("interior exceeds spatial_shape")
    invalid_block = int(
        validate_indices_unchecked(block_ids, face_neighbor_ids.shape[0])
    )
    if invalid_block >= 0:
        raise ValueError(f"block_ids entry {invalid_block} is out of range")
    lower = interior_lower.copy()
    upper = interior_upper.copy()
    common_physical_valid_region_unchecked(
        spatial_shape,
        interior_lower,
        interior_upper,
        block_ids,
        face_neighbor_ids,
        lower,
        upper,
    )
    return lower, upper
