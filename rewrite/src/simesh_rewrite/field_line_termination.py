"""TRM-001 nonperiodic field-line termination policy."""

from __future__ import annotations

from enum import IntEnum
import math

import numpy as np


class FieldLineTermination(IntEnum):
    """Stable trajectory-state and termination codes."""

    ACTIVE = 0
    SEED_OUTSIDE = 1
    MAX_STEPS = 2
    DOMAIN_EXIT = 3
    ZERO_FIELD = 4
    NONFINITE_FIELD = 5
    UNREPRESENTABLE_NORM = 6
    UNREPRESENTABLE_SAMPLE = 7
    NONFINITE_STATE = 8
    NO_PROGRESS = 9


class FieldLineStage(IntEnum):
    """Stable markers for the control or RK stage that terminated a line."""

    CONTROL = 0
    K1 = 1
    K2 = 2
    K3 = 3
    K4 = 4
    CANDIDATE = 5


_RHS_STATUS_TO_TERMINATION = (
    int(FieldLineTermination.ACTIVE),
    int(FieldLineTermination.ZERO_FIELD),
    int(FieldLineTermination.NONFINITE_FIELD),
    int(FieldLineTermination.UNREPRESENTABLE_NORM),
)


def _require_float_vector(
    name: str,
    value: np.ndarray,
    length: int,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must have dtype float64")
    if value.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}, got {value.shape}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return value


def _require_domain(
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    domain_lower = _require_float_vector("domain_lower", domain_lower, 3)
    domain_upper = _require_float_vector("domain_upper", domain_upper, 3)
    for axis in range(3):
        lower = float(domain_lower[axis])
        upper = float(domain_upper[axis])
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("domain bounds must be finite")
        if upper <= lower:
            raise ValueError("domain_upper must be greater than domain_lower")
    return domain_lower, domain_upper


def _coordinates_inside(
    state: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
) -> bool:
    for axis in range(3):
        coordinate = float(state[axis])
        if not (
            float(domain_lower[axis])
            <= coordinate
            < float(domain_upper[axis])
        ):
            return False
    return True


def classify_field_line_seed(
    seed: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
) -> int:
    """Classify one finite seed under the exact half-open domain rule."""
    seed = _require_float_vector("seed", seed, 3)
    domain_lower, domain_upper = _require_domain(domain_lower, domain_upper)
    if any(not np.isfinite(float(seed[axis])) for axis in range(3)):
        raise ValueError("seed coordinates must be finite")
    if _coordinates_inside(seed, domain_lower, domain_upper):
        return int(FieldLineTermination.ACTIVE)
    return int(FieldLineTermination.SEED_OUTSIDE)


def classify_field_line_stage_state(
    stage_state: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
) -> int:
    """Classify one augmented RK stage before field sampling."""
    stage_state = _require_float_vector("stage_state", stage_state, 4)
    domain_lower, domain_upper = _require_domain(domain_lower, domain_upper)
    return _classify_field_line_stage_state_unchecked(
        stage_state,
        domain_lower,
        domain_upper,
    )


def _classify_field_line_stage_state_unchecked(
    stage_state: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
) -> int:
    """Classify one prevalidated augmented RK stage without allocation."""
    for component in range(4):
        if not math.isfinite(float(stage_state[component])):
            return int(FieldLineTermination.NONFINITE_STATE)
    if not _coordinates_inside(stage_state, domain_lower, domain_upper):
        return int(FieldLineTermination.DOMAIN_EXIT)
    return int(FieldLineTermination.ACTIVE)


def _require_rhs_status(rhs_status: int) -> int:
    if type(rhs_status) is int:
        return rhs_status
    if type(rhs_status) is np.uint8:
        return int(rhs_status)
    raise TypeError("rhs_status must be a Python int or NumPy uint8 scalar")


def _field_line_termination_from_rhs_status_unchecked(rhs_status: int) -> int:
    return _RHS_STATUS_TO_TERMINATION[rhs_status]


def field_line_termination_from_rhs_status(rhs_status: int) -> int:
    """Map one stable FLN-001 status byte to its trajectory outcome."""
    rhs_status = _require_rhs_status(rhs_status)
    if rhs_status < 0 or rhs_status >= len(_RHS_STATUS_TO_TERMINATION):
        raise ValueError("rhs_status is not a known FLN-001 status")
    return _field_line_termination_from_rhs_status_unchecked(rhs_status)


def _coordinates_bitwise_equal(
    current_state: np.ndarray,
    candidate_state: np.ndarray,
) -> bool:
    for axis in range(3):
        current = float(current_state[axis])
        candidate = float(candidate_state[axis])
        if current != candidate:
            return False
        if current == 0.0 and math.copysign(1.0, current) != math.copysign(
            1.0, candidate
        ):
            return False
    return True


def classify_field_line_candidate(
    current_state: np.ndarray,
    candidate_state: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
) -> int:
    """Classify one finished RK candidate before accepted-prefix commit."""
    current_state = _require_float_vector("current_state", current_state, 4)
    candidate_state = _require_float_vector("candidate_state", candidate_state, 4)
    domain_lower, domain_upper = _require_domain(domain_lower, domain_upper)

    if any(not np.isfinite(float(current_state[axis])) for axis in range(3)):
        raise ValueError("current_state coordinates must be finite")
    if not _coordinates_inside(current_state, domain_lower, domain_upper):
        raise ValueError("current_state coordinates must be inside the domain")

    return _classify_field_line_candidate_unchecked(
        current_state,
        candidate_state,
        domain_lower,
        domain_upper,
    )


def _classify_field_line_candidate_unchecked(
    current_state: np.ndarray,
    candidate_state: np.ndarray,
    domain_lower: np.ndarray,
    domain_upper: np.ndarray,
) -> int:
    """Classify one prevalidated RK candidate without allocation."""
    for component in range(4):
        if not math.isfinite(float(candidate_state[component])):
            return int(FieldLineTermination.NONFINITE_STATE)
    if _coordinates_bitwise_equal(current_state, candidate_state):
        return int(FieldLineTermination.NO_PROGRESS)
    if not _coordinates_inside(candidate_state, domain_lower, domain_upper):
        return int(FieldLineTermination.DOMAIN_EXIT)
    return int(FieldLineTermination.ACTIVE)
