"""Independent scalar operation-tree references for RKS-001."""

from __future__ import annotations

import numpy as np


def field_line_rk4_stage_reference(
    base_states: np.ndarray,
    previous_rhs: np.ndarray,
    step_size: float,
    half_step: bool,
    stage_states: np.ndarray,
) -> None:
    step = np.float64(step_size)
    half = np.float64(np.float64(0.5) * step)
    scale = half if half_step else step
    with np.errstate(all="ignore"):
        for row in range(base_states.shape[0]):
            for component in range(4):
                delta = np.float64(scale * previous_rhs[row, component])
                stage_states[row, component] = np.float64(
                    base_states[row, component] + delta
                )


def field_line_rk4_finish_reference(
    base_states: np.ndarray,
    k1: np.ndarray,
    k2: np.ndarray,
    k3: np.ndarray,
    k4: np.ndarray,
    step_size: float,
    candidate_states: np.ndarray,
) -> None:
    h6 = np.float64(np.float64(step_size) / np.float64(6.0))
    with np.errstate(all="ignore"):
        for row in range(base_states.shape[0]):
            for component in range(4):
                two_k2 = np.float64(np.float64(2.0) * k2[row, component])
                sum12 = np.float64(k1[row, component] + two_k2)
                two_k3 = np.float64(np.float64(2.0) * k3[row, component])
                sum123 = np.float64(sum12 + two_k3)
                sum1234 = np.float64(sum123 + k4[row, component])
                delta = np.float64(h6 * sum1234)
                candidate_states[row, component] = np.float64(
                    base_states[row, component] + delta
                )
