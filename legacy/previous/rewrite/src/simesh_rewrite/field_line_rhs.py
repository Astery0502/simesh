"""Validated FLN-001 normalized magnetic field-line RHS."""

from __future__ import annotations

from typing import Final

import numpy as np

from ._field_line_rhs import field_line_rhs_unchecked


RHS_OK: Final = 0
RHS_ZERO_FIELD: Final = 1
RHS_NONFINITE_FIELD: Final = 2
RHS_UNREPRESENTABLE_NORM: Final = 3

_FLOAT64 = np.dtype(np.float64)
_INT8 = np.dtype(np.int8)
_UINT8 = np.dtype(np.uint8)


def _require_matrix(
    name: str,
    value: np.ndarray,
    shape: tuple[int, int],
    *,
    writable: bool,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != _FLOAT64:
        raise TypeError(f"{name} must have dtype float64")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def _require_direction_signs(
    value: np.ndarray,
    row_count: int,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("direction_signs must be a NumPy array")
    if value.dtype != _INT8:
        raise TypeError("direction_signs must have dtype int8")
    if value.shape != (row_count,):
        raise ValueError(
            f"direction_signs must have shape {(row_count,)}, got {value.shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError("direction_signs must be C-contiguous")
    for row, sign in enumerate(value):
        if int(sign) not in (-1, 1):
            raise ValueError(
                "direction_signs entries must be -1 or +1; "
                f"first invalid entry is {row}"
            )
    return value


def _require_statuses(value: np.ndarray, row_count: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError("rhs_statuses must be a NumPy array")
    if value.dtype != _UINT8:
        raise TypeError("rhs_statuses must have dtype uint8")
    if value.shape != (row_count,):
        raise ValueError(
            f"rhs_statuses must have shape {(row_count,)}, got {value.shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError("rhs_statuses must be C-contiguous")
    if not value.flags.writeable:
        raise ValueError("rhs_statuses must be writable")
    return value


def field_line_rhs_into(
    field_vectors: np.ndarray,
    direction_signs: np.ndarray,
    rhs_values: np.ndarray,
    rhs_statuses: np.ndarray,
) -> None:
    """Fill normalized field-line RHS rows and exact status codes."""
    if not isinstance(field_vectors, np.ndarray):
        raise TypeError("field_vectors must be a NumPy array")
    if field_vectors.dtype != _FLOAT64:
        raise TypeError("field_vectors must have dtype float64")
    if field_vectors.ndim != 2 or field_vectors.shape[1:] != (3,):
        raise ValueError(
            "field_vectors must have shape (N,3), "
            f"got {field_vectors.shape}"
        )
    if not field_vectors.flags.c_contiguous:
        raise ValueError("field_vectors must be C-contiguous")
    row_count = int(field_vectors.shape[0])
    direction_signs = _require_direction_signs(direction_signs, row_count)
    rhs_values = _require_matrix(
        "rhs_values", rhs_values, (row_count, 4), writable=True
    )
    rhs_statuses = _require_statuses(rhs_statuses, row_count)

    inputs = (field_vectors, direction_signs)
    outputs = (rhs_values, rhs_statuses)
    if np.shares_memory(rhs_values, rhs_statuses):
        raise ValueError("FLN outputs must not overlap each other")
    for output in outputs:
        if any(np.shares_memory(output, source) for source in inputs):
            raise ValueError("FLN outputs must not overlap inputs")

    field_line_rhs_unchecked(
        field_vectors,
        direction_signs,
        rhs_values,
        rhs_statuses,
    )
