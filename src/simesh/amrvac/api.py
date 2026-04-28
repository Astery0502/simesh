import numpy as np

from .amrvac_dataset import AMRVACDataSet
from .layouts import datau_to_udata


def open_dataset(path: str, *, ghost_width: int = 0) -> AMRVACDataSet:
    """
    Open an AMRVAC dataset for stateful workflows.

    Data is loaded lazily. Use ``read_blocks()`` or ``read_uniform()`` when you
    only need arrays and do not need to keep the dataset object.
    """
    return AMRVACDataSet(path, ghost_width=ghost_width)


def read_blocks(
    path: str,
    *,
    field_indices: list[int] | None = None,
    ghost_width: int = 0,
    include_ghosts: bool = False,
) -> np.ndarray:
    """
    Read AMRVAC block data in SFC layout.

    Returns interior data with shape ``(nleafs, nw, bx, by, bz)`` by default.
    With ``include_ghosts=True`` and ``ghost_width > 0``, returns
    ``(nleafs, nw, bx + 2g, by + 2g, bz + 2g)``.
    """
    ds = open_dataset(path, ghost_width=ghost_width)
    ds.load_data(field_indices=field_indices)
    return np.asarray(ds.blocks(include_ghosts=include_ghosts)).copy()


def read_uniform(
    path: str,
    *,
    resolution,
    bounds: tuple | None = None,
    field_indices: list[int] | None = None,
    ghost_width: int = 0,
    interpolation: str = "zero",
) -> np.ndarray:
    """
    Read AMRVAC data on a user-facing uniform grid.

    The returned array has shape ``(nx, ny, nz, nw)``. For 2D datasets,
    ``resolution`` may be ``(nx, ny)`` or ``(nx, ny, 1)`` and the returned
    z-length is one. ``bounds`` may be ``(xmin, xmax)``; by default the full
    physical domain is sampled.

    ``interpolation="zero"`` preserves the previous piecewise-constant
    behavior, is exact for native level-1 uniform data, and avoids allocating
    ghost-cell storage. ``interpolation="linear"`` uses trilinear interpolation
    from ghost-cell-padded mesh storage and therefore requires
    ``ghost_width > 0``. For level-1 data sampled on its native full-domain
    grid, use ``open_dataset(...).uniform_full()`` for exact block placement.
    """
    ds = open_dataset(path, ghost_width=ghost_width)
    ds.load_data(field_indices=field_indices)

    nx = np.asarray(resolution, dtype=np.uint32)
    if int(ds.ndim) == 2:
        if nx.shape == (2,):
            nx = np.array([nx[0], nx[1], 1], dtype=np.uint32)
        elif nx.shape != (3,) or int(nx[2]) != 1:
            raise ValueError(f"2D resolution must have shape (nx, ny) or (nx, ny, 1), got {resolution}")
    elif nx.shape != (3,):
        raise ValueError(f"resolution must have three entries, got {resolution}")

    if bounds is None:
        xmin = ds.physical_domain[0]
        xmax = ds.physical_domain[1]
    else:
        if len(bounds) != 2:
            raise ValueError("bounds must be a tuple of (xmin, xmax)")
        xmin, xmax = bounds

    datau = ds.uniform_grid(
        nx,
        xmin=xmin,
        xmax=xmax,
        field_indices=field_indices,
        interpolation=interpolation,
    )
    return datau_to_udata(datau)


def write_datfile(
    path: str,
    output_path: str,
    *,
    field_indices: list[int] | None = None,
    ghost_width: int = 0,
    overwrite: bool = False,
) -> dict:
    """
    Write an AMRVAC ``.dat`` file from an existing file.

    ``field_indices`` can be used to write a subset of fields. Ghost cells are
    used for in-memory exchange when ``ghost_width > 0`` but are not written to
    the output file.
    """
    ds = open_dataset(path, ghost_width=ghost_width)
    ds.load_data(field_indices=field_indices)
    return ds.write_datfile(output_path, overwrite=overwrite)
