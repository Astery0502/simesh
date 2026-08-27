"""Validated FND-002 access-pattern and region vocabulary."""

from __future__ import annotations

from enum import IntEnum
from typing import Final

import numpy as np

from ._access import (
    required_input_region_unchecked,
    valid_output_region_unchecked,
)
from .foundation import _require_index_triplet


_INDEX_MAX: Final = int(np.iinfo(np.int64).max)


class AccessPattern(IntEnum):
    POINTWISE = 0
    LOCAL_STENCIL = 1
    STREAMING_REDUCTION = 2


def _require_region(
    name: str,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = _require_index_triplet(f"{name}_lower", lower)
    upper = _require_index_triplet(f"{name}_upper", upper)
    if np.any(lower < 0):
        raise ValueError(f"{name} lower bound must be non-negative")
    if np.any(lower > upper):
        raise ValueError(f"{name} lower bound exceeds its upper bound")
    return lower, upper


def _require_reaches(
    lower_reach: np.ndarray,
    upper_reach: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower_reach = _require_index_triplet("lower_reach", lower_reach)
    upper_reach = _require_index_triplet("upper_reach", upper_reach)
    if np.any(lower_reach < 0) or np.any(upper_reach < 0):
        raise ValueError("reach entries must be non-negative")
    return lower_reach, upper_reach


def _is_empty(lower: np.ndarray, upper: np.ndarray) -> bool:
    return bool(np.any(lower == upper))


def validate_access_requirement(
    pattern: AccessPattern,
    lower_reach: np.ndarray,
    upper_reach: np.ndarray,
) -> None:
    """Validate one canonical access tag and its per-side reach."""
    if not isinstance(pattern, AccessPattern):
        raise TypeError("pattern must be an AccessPattern")
    lower_reach, upper_reach = _require_reaches(lower_reach, upper_reach)
    has_reach = bool(np.any(lower_reach != 0) or np.any(upper_reach != 0))
    if pattern is AccessPattern.LOCAL_STENCIL:
        if not has_reach:
            raise ValueError("a local stencil must have nonzero reach")
    elif has_reach:
        raise ValueError(f"{pattern.name} must have zero reach")


def required_input_region(
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    lower_reach: np.ndarray,
    upper_reach: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the minimal input box enclosing all reads for an output box."""
    output_lower, output_upper = _require_region(
        "output", output_lower, output_upper
    )
    lower_reach, upper_reach = _require_reaches(lower_reach, upper_reach)
    if _is_empty(output_lower, output_upper):
        return output_lower.copy(), output_lower.copy()
    if np.any(lower_reach > output_lower):
        raise ValueError("required input region begins below storage coordinate zero")
    if any(
        int(reach) > _INDEX_MAX - int(upper)
        for upper, reach in zip(output_upper, upper_reach, strict=True)
    ):
        raise OverflowError("required input upper bound does not fit in int64")

    required_lower = np.empty(3, dtype=np.int64)
    required_upper = np.empty(3, dtype=np.int64)
    required_input_region_unchecked(
        output_lower,
        output_upper,
        lower_reach,
        upper_reach,
        required_lower,
        required_upper,
    )
    return required_lower, required_upper


def valid_output_region(
    valid_lower: np.ndarray,
    valid_upper: np.ndarray,
    lower_reach: np.ndarray,
    upper_reach: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the largest aligned output box supported by an input-valid box."""
    valid_lower, valid_upper = _require_region("valid", valid_lower, valid_upper)
    lower_reach, upper_reach = _require_reaches(lower_reach, upper_reach)
    extent = valid_upper - valid_lower
    if _is_empty(valid_lower, valid_upper) or any(
        int(size) <= int(lower) + int(upper)
        for size, lower, upper in zip(
            extent,
            lower_reach,
            upper_reach,
            strict=True,
        )
    ):
        return valid_lower.copy(), valid_lower.copy()

    output_lower = np.empty(3, dtype=np.int64)
    output_upper = np.empty(3, dtype=np.int64)
    valid_output_region_unchecked(
        valid_lower,
        valid_upper,
        lower_reach,
        upper_reach,
        output_lower,
        output_upper,
    )
    return output_lower, output_upper


def supports_output_region(
    valid_lower: np.ndarray,
    valid_upper: np.ndarray,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    lower_reach: np.ndarray,
    upper_reach: np.ndarray,
) -> bool:
    """Return whether an input-valid box covers every requested input read."""
    valid_lower, valid_upper = _require_region("valid", valid_lower, valid_upper)
    output_lower, output_upper = _require_region(
        "output", output_lower, output_upper
    )
    lower_reach, upper_reach = _require_reaches(lower_reach, upper_reach)
    if _is_empty(output_lower, output_upper):
        return True
    if any(
        int(reach) > _INDEX_MAX - int(upper)
        for upper, reach in zip(output_upper, upper_reach, strict=True)
    ):
        raise OverflowError("required input upper bound does not fit in int64")
    if np.any(lower_reach > output_lower):
        return False
    required_lower, required_upper = required_input_region(
        output_lower,
        output_upper,
        lower_reach,
        upper_reach,
    )
    return bool(
        np.all(valid_lower <= required_lower)
        and np.all(required_upper <= valid_upper)
    )
