"""Validated fixed classical RK4 augmented-state arithmetic."""

from __future__ import annotations

import math

import numpy as np

from ._rk4 import (
    field_line_rk4_finish_unchecked,
    field_line_rk4_stage_unchecked,
)


_NORMAL_MIN = float(np.finfo(np.float64).tiny)


def _require_state_rows(
    name: str,
    value: np.ndarray,
    *,
    writable: bool,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have dtype float64")
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError(f"{name} must have shape (row, 4), got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


def _require_step_size(value: float) -> float:
    if not isinstance(value, (float, np.floating)):
        raise TypeError("step_size must be a Python or NumPy real floating scalar")
    result = float(value)
    if not math.isfinite(result) or result < _NORMAL_MIN:
        raise ValueError("step_size must be finite, positive, and normal")
    return result


def _require_equal_shape(
    reference_name: str,
    reference: np.ndarray,
    named_arrays: tuple[tuple[str, np.ndarray], ...],
) -> None:
    for name, value in named_arrays:
        if value.shape != reference.shape:
            raise ValueError(
                f"{name} must have the same shape as {reference_name}, "
                f"got {value.shape} and {reference.shape}"
            )


def _require_output_nonoverlap(
    output_name: str,
    output: np.ndarray,
    inputs: tuple[np.ndarray, ...],
) -> None:
    if any(np.shares_memory(output, value) for value in inputs):
        raise ValueError(f"{output_name} must not overlap any RK4 input")


def field_line_rk4_stage_into(
    base_states: np.ndarray,
    previous_rhs: np.ndarray,
    step_size: float,
    half_step: bool,
    stage_states: np.ndarray,
) -> None:
    """Construct one half- or full-step RK4 stage state."""
    if type(half_step) is not bool:
        raise TypeError("half_step must be an exact bool")
    step_size = _require_step_size(step_size)
    base_states = _require_state_rows("base_states", base_states, writable=False)
    previous_rhs = _require_state_rows(
        "previous_rhs", previous_rhs, writable=False
    )
    stage_states = _require_state_rows("stage_states", stage_states, writable=True)
    _require_equal_shape(
        "base_states",
        base_states,
        (("previous_rhs", previous_rhs), ("stage_states", stage_states)),
    )
    _require_output_nonoverlap(
        "stage_states", stage_states, (base_states, previous_rhs)
    )
    field_line_rk4_stage_unchecked(
        base_states, previous_rhs, step_size, half_step, stage_states
    )


def field_line_rk4_finish_into(
    base_states: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    k3: np.ndarray,
    k4: np.ndarray,
    step_size: float,
    candidate_states: np.ndarray,
) -> None:
    """Finalize one fixed classical RK4 candidate state."""
    step_size = _require_step_size(step_size)
    base_states = _require_state_rows("base_states", base_states, writable=False)
    k1 = _require_state_rows("k1", k1, writable=False)
    k2 = _require_state_rows("k2", k2, writable=False)
    k3 = _require_state_rows("k3", k3, writable=False)
    k4 = _require_state_rows("k4", k4, writable=False)
    candidate_states = _require_state_rows(
        "candidate_states", candidate_states, writable=True
    )
    _require_equal_shape(
        "base_states",
        base_states,
        (
            ("k1", k1),
            ("k2", k2),
            ("k3", k3),
            ("k4", k4),
            ("candidate_states", candidate_states),
        ),
    )
    _require_output_nonoverlap(
        "candidate_states", candidate_states, (base_states, k1, k2, k3, k4)
    )
    field_line_rk4_finish_unchecked(
        base_states, k1, k2, k3, k4, step_size, candidate_states
    )
