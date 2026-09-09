"""Validated RST-001 Cartesian 3D ratio-two restriction."""

from __future__ import annotations

import numpy as np

from ._restriction import restrict_cartesian_2to1_into_unchecked
from .foundation import _require_index_triplet, _require_payload


_INDEX_MAX = int(np.iinfo(np.int64).max)


def restrict_cartesian_2to1_into(
    fine_payload: np.ndarray,
    fine_lower: np.ndarray,
    fine_upper: np.ndarray,
    coarse_payload: np.ndarray,
    coarse_lower: np.ndarray,
) -> None:
    """Average an even fine box into a translated ratio-two coarse box."""
    fine_payload = _require_payload(
        "fine_payload", fine_payload, writable=False
    )
    fine_lower = _require_index_triplet("fine_lower", fine_lower)
    fine_upper = _require_index_triplet("fine_upper", fine_upper)
    coarse_payload = _require_payload(
        "coarse_payload", coarse_payload, writable=True
    )
    coarse_lower = _require_index_triplet("coarse_lower", coarse_lower)

    if fine_payload.shape[:2] != coarse_payload.shape[:2]:
        raise ValueError(
            "fine and coarse payloads must have equal slot and field extents"
        )
    fine_extent = [0, 0, 0]
    for axis in range(3):
        lower = int(fine_lower[axis])
        upper = int(fine_upper[axis])
        if lower < 0 or lower > upper:
            raise ValueError("fine region must be ordered and nonnegative")
        fine_extent[axis] = upper - lower
    for axis in range(3):
        if int(fine_upper[axis]) > fine_payload.shape[axis + 2]:
            raise ValueError("fine region exceeds the fine spatial shape")
    if any(extent % 2 != 0 for extent in fine_extent):
        raise ValueError("fine region extents must be even")
    coarse_extent = [extent // 2 for extent in fine_extent]
    for axis in range(3):
        if int(coarse_lower[axis]) < 0:
            raise ValueError("coarse_lower must be nonnegative")
    for axis in range(3):
        lower = int(coarse_lower[axis])
        if lower > _INDEX_MAX - coarse_extent[axis]:
            raise OverflowError(
                "translated coarse region does not fit in int64"
            )
    for axis in range(3):
        lower = int(coarse_lower[axis])
        shape = coarse_payload.shape[axis + 2]
        if lower > shape or coarse_extent[axis] > shape - lower:
            raise ValueError("translated region exceeds coarse spatial shape")

    if any(
        np.shares_memory(coarse_payload, value)
        for value in (
            fine_payload,
            fine_lower,
            fine_upper,
            coarse_lower,
        )
    ):
        raise ValueError("coarse_payload must not overlap inputs")

    restrict_cartesian_2to1_into_unchecked(
        fine_payload,
        fine_lower,
        fine_upper,
        coarse_payload,
        coarse_lower,
    )
