import numpy as np


BOUNDARY_MODE_CODES = {
    "cont": 0,
    "symm": 1,
    "asymm": 2,
    "noinflow": 3,
}

SIDE_NAMES_BY_NDIM = {
    2: ("xlo", "xhi", "ylo", "yhi"),
    3: ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi"),
}

_VELOCITY_NAMES = (
    ("m1", "v1", "u1", "mom1", "rho_v1", "mx", "vx", "ux", "momx", "rho_vx"),
    ("m2", "v2", "u2", "mom2", "rho_v2", "my", "vy", "uy", "momy", "rho_vy"),
    ("m3", "v3", "u3", "mom3", "rho_v3", "mz", "vz", "uz", "momz", "rho_vz"),
)


def normalize_boundary_conditions(boundary_conditions, field_names, ndim):
    """
    Normalize AMRVAC-like boundary condition input to an integer table.
    """
    ndim = int(ndim)
    if ndim not in SIDE_NAMES_BY_NDIM:
        raise ValueError(f"Only 2D and 3D boundary conditions are supported, got ndim={ndim}")

    field_names = list(field_names)
    side_names = SIDE_NAMES_BY_NDIM[ndim]
    table = np.full((len(field_names), 2 * ndim), BOUNDARY_MODE_CODES["cont"], dtype=np.int32)

    if boundary_conditions is None:
        return table, _normal_velocity_fields(table, field_names, ndim)

    if isinstance(boundary_conditions, str):
        code = _mode_code(boundary_conditions)
        table[:, :] = code
        return table, _normal_velocity_fields(table, field_names, ndim)

    if isinstance(boundary_conditions, np.ndarray):
        raw = np.asarray(boundary_conditions, dtype=np.int32)
        if raw.shape != table.shape:
            raise ValueError(f"boundary_conditions must have shape {table.shape}, got {raw.shape}")
        if np.any((raw < 0) | (raw > BOUNDARY_MODE_CODES["noinflow"])):
            raise ValueError("boundary_conditions contains an unknown boundary condition code")
        return raw.copy(), _normal_velocity_fields(raw, field_names, ndim)

    if not isinstance(boundary_conditions, dict):
        raise ValueError("boundary_conditions must be None, a mode string, a table, or a field mapping")

    field_to_pos = {name: i for i, name in enumerate(field_names)}
    side_to_pos = {name: i for i, name in enumerate(side_names)}

    for field_name, value in boundary_conditions.items():
        if field_name not in field_to_pos:
            raise ValueError(f"Unknown boundary condition field: {field_name}")
        field_pos = field_to_pos[field_name]

        if isinstance(value, str):
            table[field_pos, :] = _mode_code(value)
            continue

        if not isinstance(value, dict):
            raise ValueError(f"Boundary condition for field {field_name} must be a mode string or side mapping")

        for side_name, mode in value.items():
            if side_name not in side_to_pos:
                raise ValueError(f"Invalid boundary side for {ndim}D data: {side_name}")
            table[field_pos, side_to_pos[side_name]] = _mode_code(mode)

    return table, _normal_velocity_fields(table, field_names, ndim)


def _mode_code(mode):
    key = mode.lower() if isinstance(mode, str) else mode
    if key not in BOUNDARY_MODE_CODES:
        raise ValueError(f"Unknown boundary condition mode: {mode}")
    return BOUNDARY_MODE_CODES[key]


def _normal_velocity_fields(table, field_names, ndim):
    normal_fields = np.full(ndim, -1, dtype=np.int32)
    normalized_names = {name.lower(): i for i, name in enumerate(field_names)}

    for idim in range(ndim):
        if not np.any(table[:, 2 * idim:2 * idim + 2] == BOUNDARY_MODE_CODES["noinflow"]):
            continue

        for candidate in _VELOCITY_NAMES[idim]:
            if candidate in normalized_names:
                normal_fields[idim] = normalized_names[candidate]
                break

        if normal_fields[idim] == -1:
            sides = SIDE_NAMES_BY_NDIM[ndim][2 * idim:2 * idim + 2]
            raise ValueError(
                "noinflow for sides "
                f"{sides} requires a loaded normal velocity or momentum field; "
                f"loaded fields are {field_names}"
            )

    return normal_fields
