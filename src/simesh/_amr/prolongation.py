"""Validated PRL-001 Cartesian 3D ratio-two limited prolongation."""

from __future__ import annotations

import numpy as np

from simesh._kernels.primitives._prolongation import prolong_cartesian_2to1_into_unchecked
from simesh._amr.foundation import _require_index_triplet, _require_payload


_INDEX_MIN = int(np.iinfo(np.int64).min)
_INDEX_MAX = int(np.iinfo(np.int64).max)


def _require_box(
    name: str,
    lower: np.ndarray,
    upper: np.ndarray,
    spatial_shape: tuple[int, int, int],
) -> None:
    for axis in range(3):
        start = int(lower[axis])
        stop = int(upper[axis])
        if start < 0 or start > stop:
            raise ValueError(f"{name} must be ordered and nonnegative")
    for axis in range(3):
        if int(upper[axis]) > spatial_shape[axis]:
            raise ValueError(f"{name} exceeds its payload spatial shape")


def _require_int64(name: str, value: int) -> int:
    if value < _INDEX_MIN or value > _INDEX_MAX:
        raise OverflowError(f"{name} does not fit in int64")
    return value


def prolong_cartesian_2to1_into(
    coarse_payload: np.ndarray,
    coarse_valid_lower: np.ndarray,
    coarse_valid_upper: np.ndarray,
    coarse_origin: np.ndarray,
    fine_payload: np.ndarray,
    fine_lower: np.ndarray,
    fine_upper: np.ndarray,
    fine_origin: np.ndarray,
) -> None:
    """Reconstruct an explicit fine box from an aligned coarse workspace."""
    coarse_payload = _require_payload(
        "coarse_payload", coarse_payload, writable=False
    )
    coarse_valid_lower = _require_index_triplet(
        "coarse_valid_lower", coarse_valid_lower
    )
    coarse_valid_upper = _require_index_triplet(
        "coarse_valid_upper", coarse_valid_upper
    )
    coarse_origin = _require_index_triplet("coarse_origin", coarse_origin)
    fine_payload = _require_payload(
        "fine_payload", fine_payload, writable=True
    )
    fine_lower = _require_index_triplet("fine_lower", fine_lower)
    fine_upper = _require_index_triplet("fine_upper", fine_upper)
    fine_origin = _require_index_triplet("fine_origin", fine_origin)

    if coarse_payload.shape[:2] != fine_payload.shape[:2]:
        raise ValueError(
            "coarse and fine payloads must have equal slot and field extents"
        )
    _require_box(
        "coarse valid box",
        coarse_valid_lower,
        coarse_valid_upper,
        coarse_payload.shape[2:],
    )
    _require_box(
        "fine target box",
        fine_lower,
        fine_upper,
        fine_payload.shape[2:],
    )

    if any(
        np.shares_memory(fine_payload, value)
        for value in (
            coarse_payload,
            coarse_valid_lower,
            coarse_valid_upper,
            coarse_origin,
            fine_lower,
            fine_upper,
            fine_origin,
        )
    ):
        raise ValueError("fine_payload must not overlap inputs")

    spatially_empty = any(
        int(fine_lower[axis]) == int(fine_upper[axis]) for axis in range(3)
    )
    if spatially_empty:
        return

    required_lower = [0, 0, 0]
    required_upper = [0, 0, 0]
    for axis in range(3):
        first_relative = _require_int64(
            "fine-relative mapping",
            int(fine_lower[axis]) - int(fine_origin[axis]),
        )
        last_relative = _require_int64(
            "fine-relative mapping",
            int(fine_upper[axis]) - 1 - int(fine_origin[axis]),
        )
        first_quotient = first_relative // 2
        last_quotient = last_relative // 2
        first_center = _require_int64(
            "coarse-center mapping",
            int(coarse_origin[axis]) + first_quotient,
        )
        last_center = _require_int64(
            "coarse-center mapping",
            int(coarse_origin[axis]) + last_quotient,
        )
        required_lower[axis] = _require_int64(
            "coarse reach", first_center - 1
        )
        required_upper[axis] = _require_int64(
            "coarse reach", last_center + 2
        )

    for axis in range(3):
        if (
            required_lower[axis] < 0
            or required_upper[axis] > coarse_payload.shape[axis + 2]
        ):
            raise ValueError("mapped coarse reach exceeds coarse storage")
    for axis in range(3):
        if (
            int(coarse_valid_lower[axis]) > required_lower[axis]
            or int(coarse_valid_upper[axis]) < required_upper[axis]
        ):
            raise ValueError("coarse valid box does not contain required reach")

    prolong_cartesian_2to1_into_unchecked(
        coarse_payload,
        coarse_origin,
        fine_payload,
        fine_lower,
        fine_upper,
        fine_origin,
    )
