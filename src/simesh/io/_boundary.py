"""Normalize explicit per-component physical halo rules at source creation."""

from collections.abc import Mapping

import numpy as np

from .._amr.halos import BoundaryMode
from .._validation import frozen_array
from ..fields import _field_index


_MODES = {
    "continuous": BoundaryMode.CONTINUOUS,
    "symmetric": BoundaryMode.SYMMETRIC,
    "asymmetric": BoundaryMode.ANTISYMMETRIC,
}


def boundary_configuration(mesh, fields, boundary):
    periodic = tuple(flag for flag in mesh.periodic for _ in range(2))

    def row(value):
        if isinstance(value, str):
            if value not in _MODES:
                raise ValueError("boundary must be continuous, symmetric or asymmetric")
            return tuple("periodic" if p else value for p in periodic)
        values = tuple(value)
        if len(values) != 6:
            raise ValueError("boundary rows require six faces: x-, x+, y-, y+, z-, z+")
        for mode, is_periodic in zip(values, periodic):
            if not isinstance(mode, str) or mode not in (*_MODES, "periodic"):
                raise ValueError("unknown boundary mode")
            if (mode == "periodic") != is_periodic:
                raise ValueError("boundary periodic faces must agree with Mesh.periodic")
        return values

    rows = [row("continuous")] * len(fields)
    if boundary is None:
        pass
    elif isinstance(boundary, str):
        rows = [row(boundary)] * len(fields)
    elif isinstance(boundary, Mapping):
        for name, value in boundary.items():
            if not isinstance(name, str):
                raise ValueError("boundary mapping keys must be field names")
            rows[_field_index(fields, name)] = row(value)
    else:
        rows = [row(value) for value in boundary]
        if len(rows) != len(fields):
            raise ValueError("boundary table requires one six-face row per source field")
    modes = [[_MODES.get(mode, BoundaryMode.CONTINUOUS) for mode in values] for values in rows]
    return tuple(rows), frozen_array(modes, np.uint8)
