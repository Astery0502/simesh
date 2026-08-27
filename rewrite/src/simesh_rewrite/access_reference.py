"""Independent Python/NumPy reference for FND-002 region arithmetic."""

from __future__ import annotations

import numpy as np


_INDEX_MAX = int(np.iinfo(np.int64).max)


def _triplet(name: str, value: np.ndarray) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.int64):
        raise TypeError(f"{name} must be an int64 NumPy array")
    if value.shape != (3,) or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be a C-contiguous triplet")
    return value


def _region(
    name: str,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower = _triplet(f"{name}_lower", lower)
    upper = _triplet(f"{name}_upper", upper)
    if np.any(lower < 0) or np.any(lower > upper):
        raise ValueError(f"{name} is not a valid half-open storage box")
    return lower, upper


def _reaches(
    lower_reach: np.ndarray,
    upper_reach: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lower_reach = _triplet("lower_reach", lower_reach)
    upper_reach = _triplet("upper_reach", upper_reach)
    if np.any(lower_reach < 0) or np.any(upper_reach < 0):
        raise ValueError("reach entries must be non-negative")
    return lower_reach, upper_reach


def _empty(lower: np.ndarray, upper: np.ndarray) -> bool:
    return bool(np.any(lower == upper))


def required_input_region(
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    lower_reach: np.ndarray,
    upper_reach: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    output_lower, output_upper = _region("output", output_lower, output_upper)
    lower_reach, upper_reach = _reaches(lower_reach, upper_reach)
    if _empty(output_lower, output_upper):
        return output_lower.copy(), output_lower.copy()

    required_lower = []
    required_upper = []
    for lower, upper, below, above in zip(
        output_lower,
        output_upper,
        lower_reach,
        upper_reach,
        strict=True,
    ):
        lower_value = int(lower) - int(below)
        upper_value = int(upper) + int(above)
        if lower_value < 0:
            raise ValueError("required input region begins below storage origin")
        if upper_value > _INDEX_MAX:
            raise OverflowError("required input upper bound does not fit in int64")
        required_lower.append(lower_value)
        required_upper.append(upper_value)
    return (
        np.asarray(required_lower, dtype=np.int64),
        np.asarray(required_upper, dtype=np.int64),
    )


def valid_output_region(
    valid_lower: np.ndarray,
    valid_upper: np.ndarray,
    lower_reach: np.ndarray,
    upper_reach: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid_lower, valid_upper = _region("valid", valid_lower, valid_upper)
    lower_reach, upper_reach = _reaches(lower_reach, upper_reach)
    if _empty(valid_lower, valid_upper):
        return valid_lower.copy(), valid_lower.copy()

    output_lower = []
    output_upper = []
    for lower, upper, below, above in zip(
        valid_lower,
        valid_upper,
        lower_reach,
        upper_reach,
        strict=True,
    ):
        if int(upper) - int(lower) <= int(below) + int(above):
            return valid_lower.copy(), valid_lower.copy()
        output_lower.append(int(lower) + int(below))
        output_upper.append(int(upper) - int(above))
    return (
        np.asarray(output_lower, dtype=np.int64),
        np.asarray(output_upper, dtype=np.int64),
    )


def supports_output_region(
    valid_lower: np.ndarray,
    valid_upper: np.ndarray,
    output_lower: np.ndarray,
    output_upper: np.ndarray,
    lower_reach: np.ndarray,
    upper_reach: np.ndarray,
) -> bool:
    valid_lower, valid_upper = _region("valid", valid_lower, valid_upper)
    output_lower, output_upper = _region("output", output_lower, output_upper)
    lower_reach, upper_reach = _reaches(lower_reach, upper_reach)
    if _empty(output_lower, output_upper):
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
