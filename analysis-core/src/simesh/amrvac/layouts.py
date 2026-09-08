import numpy as np


"""
Canonical array layout helpers for AMRVAC data.

Conventions
-----------
- ``udata``: user-facing uniform data with shape ``(nx, ny, nz, nw)``
- ``datau``: compute-oriented uniform data with shape ``(nw, nx, ny, nz)``
- ``sfc_data``: AMR block data in Morton/SFC order with shape
  ``(nleafs, nw, bx, by, bz)``
"""


def udata_to_datau(udata: np.ndarray) -> np.ndarray:
    """
    Convert user-facing uniform data from ``(nx, ny, nz, nw)`` to
    compute-oriented ``(nw, nx, ny, nz)`` layout.
    """
    udata = np.asarray(udata)
    if udata.ndim != 4:
        raise ValueError(f"udata must have shape (nx, ny, nz, nw), got {udata.shape}")
    return np.ascontiguousarray(np.transpose(udata, (3, 0, 1, 2)))


def datau_to_udata(datau: np.ndarray) -> np.ndarray:
    """
    Convert compute-oriented uniform data from ``(nw, nx, ny, nz)`` to
    user-facing ``(nx, ny, nz, nw)`` layout.
    """
    datau = np.asarray(datau)
    if datau.ndim != 4:
        raise ValueError(f"datau must have shape (nw, nx, ny, nz), got {datau.shape}")
    return np.ascontiguousarray(np.transpose(datau, (1, 2, 3, 0)))
