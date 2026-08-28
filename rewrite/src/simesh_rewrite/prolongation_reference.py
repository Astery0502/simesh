"""Independent scalar-loop reference for PRL-001."""

from __future__ import annotations

import numpy as np

from .limiter_reference import three_point_limited_slope_reference


def prolong_cartesian_2to1_reference(
    coarse_payload: np.ndarray,
    coarse_valid_lower: np.ndarray,
    coarse_valid_upper: np.ndarray,
    coarse_origin: np.ndarray,
    fine_payload: np.ndarray,
    fine_lower: np.ndarray,
    fine_upper: np.ndarray,
    fine_origin: np.ndarray,
) -> None:
    """Reconstruct an explicit fine box with exact ratio-two phase weights."""
    with np.errstate(all="ignore"):
        for slot in range(coarse_payload.shape[0]):
            for field in range(coarse_payload.shape[1]):
                for i in range(int(fine_lower[0]), int(fine_upper[0])):
                    rx = i - int(fine_origin[0])
                    qx = rx // 2
                    px = rx - 2 * qx
                    I = int(coarse_origin[0]) + qx
                    eta_x = np.float64(-0.25 if px == 0 else 0.25)
                    for j in range(int(fine_lower[1]), int(fine_upper[1])):
                        ry = j - int(fine_origin[1])
                        qy = ry // 2
                        py = ry - 2 * qy
                        J = int(coarse_origin[1]) + qy
                        eta_y = np.float64(-0.25 if py == 0 else 0.25)
                        for k in range(int(fine_lower[2]), int(fine_upper[2])):
                            rz = k - int(fine_origin[2])
                            qz = rz // 2
                            pz = rz - 2 * qz
                            K = int(coarse_origin[2]) + qz
                            eta_z = np.float64(-0.25 if pz == 0 else 0.25)

                            center = coarse_payload[slot, field, I, J, K]
                            slope_x = np.float64(
                                three_point_limited_slope_reference(
                                    coarse_payload[slot, field, I - 1, J, K],
                                    center,
                                    coarse_payload[slot, field, I + 1, J, K],
                                )
                            )
                            slope_y = np.float64(
                                three_point_limited_slope_reference(
                                    coarse_payload[slot, field, I, J - 1, K],
                                    center,
                                    coarse_payload[slot, field, I, J + 1, K],
                                )
                            )
                            slope_z = np.float64(
                                three_point_limited_slope_reference(
                                    coarse_payload[slot, field, I, J, K - 1],
                                    center,
                                    coarse_payload[slot, field, I, J, K + 1],
                                )
                            )
                            term_x = np.float64(slope_x * eta_x)
                            term_y = np.float64(slope_y * eta_y)
                            term_z = np.float64(slope_z * eta_z)
                            value = np.float64(center + term_x)
                            value = np.float64(value + term_y)
                            value = np.float64(value + term_z)
                            fine_payload[slot, field, i, j, k] = value
