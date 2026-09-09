"""Binary legacy VTK output for completed uniform cell-center volumes."""

import math
import os
from pathlib import Path
import tempfile

import numpy as np

from ..fields import FieldDefinition
from .._validation import admit, array_bytes


_BUFFER_BYTES = 256 * 1024
_VALID_NAME = "simesh_valid"


def _destination(path, overwrite):
    if type(overwrite) is not bool:
        raise TypeError("overwrite must be boolean")
    destination = Path(path)
    if not overwrite and os.path.lexists(destination):
        raise FileExistsError(destination)
    return destination


def _write_array(stream, array, dtype):
    dtype = np.dtype(dtype)
    with np.nditer(array, flags=["external_loop", "buffered"],
                   op_flags=["readonly", "contig"], op_dtypes=[dtype], order="F",
                   buffersize=max(1, _BUFFER_BYTES // dtype.itemsize)) as iterator:
        for chunk in iterator:
            stream.write(memoryview(chunk).cast("B"))
    stream.write(b"\n")


def write_uniform_vtk(path, grid, *, overwrite=False, memory_limit=None):
    """Save a uniform volume as binary legacy VTK structured points.

    Parameters
    ----------
    path : str or Path
        Destination .vtk file; its parent must exist.
    grid : UniformResult
        Completed uniform cell-center values and coverage, for example from
        applications.uniform_grid or export_uniform. Values must be float64 with
        shape (nx, ny, nz, component), and validity must be bool (nx, ny, nz).
        Strided arrays and memory maps are supported. Component names must be
        distinct printable ASCII tokens without whitespace; simesh_valid is reserved.
    overwrite : bool
        Replace an existing destination atomically; otherwise refuse it.
    memory_limit : int, optional
        Accounted-array budget including grid backing and a bounded byte-order
        conversion buffer; not a process RSS limit.

    Returns
    -------
    Path
        Published file. Failure preserves any existing destination.

    Notes
    -----
    - Values are scalar CELL_DATA on cells spanning lower to upper; point-grid
      dimensions are (nx+1, ny+1, nz+1), with x varying fastest in the file.
    - Values, including nonfinite values, are preserved. The unsigned-char scalar
      simesh_valid records coverage (1 or 0), independently of finiteness.
    - Units, interpretation and source identity are not serialized. Keep writable
      aliases unchanged during export; AMR hierarchy and line geometry are unsupported.
    """
    from ..applications import UniformResult

    destination = _destination(path, overwrite)
    if not isinstance(grid, UniformResult):
        raise TypeError("VTK export requires a UniformResult")
    values, valid = grid.values, grid.valid
    if (not isinstance(valid, np.ndarray) or valid.dtype != np.bool_
            or valid.ndim != 3 or any(n < 1 for n in valid.shape)
            or not isinstance(values, np.ndarray) or values.dtype != np.float64
            or values.shape != (*valid.shape, len(grid.definitions))
            or not grid.definitions
            or not all(isinstance(f, FieldDefinition) for f in grid.definitions)):
        raise ValueError("uniform values, validity and definitions must match")
    lower, upper = np.asarray(grid.lower, dtype=float), np.asarray(grid.upper, dtype=float)
    if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(upper <= lower)):
        raise ValueError("uniform bounds must be finite ordered three-dimensional vectors")
    spacing = (upper-lower)/valid.shape
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("uniform spacing must be finite and positive")
    names = [f.name for f in grid.definitions]
    if (len(set(names)) != len(names) or _VALID_NAME in names
            or any(not all(33 <= ord(c) <= 126 for c in name) for name in names)):
        raise ValueError("VTK names must be unique printable ASCII tokens; simesh_valid is reserved")
    count = math.prod(valid.shape)
    admit(array_bytes((values, valid, grid.lower, grid.upper, lower, upper, spacing))
          + min(_BUFFER_BYTES, 8*count),
          memory_limit, "uniform VTK export")
    dimensions = " ".join(str(n+1) for n in valid.shape)
    origin = " ".join(format(float(x), ".17g") for x in lower)
    step = " ".join(format(float(x), ".17g") for x in spacing)
    header = ("# vtk DataFile Version 3.0\nsimesh uniform cell-center fields\nBINARY\n"
              f"DATASET STRUCTURED_POINTS\nDIMENSIONS {dimensions}\n"
              f"ORIGIN {origin}\nSPACING {step}\nCELL_DATA {count}\n")
    fd, temporary = tempfile.mkstemp(prefix=".simesh-vtk-", suffix=".vtk", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(header.encode("ascii"))
            for column, name in enumerate(names):
                stream.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n".encode("ascii"))
                _write_array(stream, values[..., column], ">f8")
            stream.write(f"SCALARS {_VALID_NAME} unsigned_char 1\nLOOKUP_TABLE default\n".encode("ascii"))
            _write_array(stream, valid, "u1")
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
        return destination
    finally:
        Path(temporary).unlink(missing_ok=True)
