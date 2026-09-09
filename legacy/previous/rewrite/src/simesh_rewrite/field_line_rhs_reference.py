"""Independent scalar operation-tree reference for FLN-001."""

from __future__ import annotations

import math

import numpy as np

from .field_line_rhs import (
    RHS_NONFINITE_FIELD,
    RHS_OK,
    RHS_UNREPRESENTABLE_NORM,
    RHS_ZERO_FIELD,
)


def field_line_rhs_reference(
    field_vectors: np.ndarray,
    direction_signs: np.ndarray,
    rhs_values: np.ndarray,
    rhs_statuses: np.ndarray,
) -> None:
    """Apply the written scalar tree without using the production kernel."""
    with np.errstate(all="ignore"):
        for row in range(field_vectors.shape[0]):
            bx = np.float64(field_vectors[row, 0])
            by = np.float64(field_vectors[row, 1])
            bz = np.float64(field_vectors[row, 2])
            if not (np.isfinite(bx) and np.isfinite(by) and np.isfinite(bz)):
                rhs_statuses[row] = RHS_NONFINITE_FIELD
                continue

            ax = np.float64(abs(bx))
            ay = np.float64(abs(by))
            az = np.float64(abs(bz))
            maximum = ax
            if ay > maximum:
                maximum = ay
            if az > maximum:
                maximum = az
            if maximum == np.float64(0.0):
                rhs_statuses[row] = RHS_ZERO_FIELD
                continue

            ux = np.float64(bx / maximum)
            uy = np.float64(by / maximum)
            uz = np.float64(bz / maximum)
            xx = np.float64(ux * ux)
            yy = np.float64(uy * uy)
            zz = np.float64(uz * uz)
            xy = np.float64(xx + yy)
            xyz = np.float64(xy + zz)
            radius = np.float64(math.sqrt(float(xyz)))
            magnitude = np.float64(maximum * radius)
            if not np.isfinite(magnitude):
                rhs_statuses[row] = RHS_UNREPRESENTABLE_NORM
                continue

            tx = np.float64(ux / radius)
            ty = np.float64(uy / radius)
            tz = np.float64(uz / radius)
            sign = np.float64(direction_signs[row])
            rhs_values[row, 0] = np.float64(sign * tx)
            rhs_values[row, 1] = np.float64(sign * ty)
            rhs_values[row, 2] = np.float64(sign * tz)
            rhs_values[row, 3] = np.float64(sign * magnitude)
            rhs_statuses[row] = RHS_OK
