from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np


_VALID_BACKENDS = {"auto", "direct", "fft"}
_GREEN_NORMALIZATION = 1.0 / (2.0 * np.pi)


@dataclass(frozen=True)
class PotentialFieldGeometry:
    """Uniform Cartesian geometry for a potential-field extrapolation."""

    xmin: tuple[float, float, float]
    xmax: tuple[float, float, float]
    domain_nx: tuple[int, int, int]
    dx: float
    dy: float
    dz: float
    flux_balanced: bool
    removed_flux_mean: float

    @property
    def spacing(self) -> tuple[float, float, float]:
        return (self.dx, self.dy, self.dz)

    def cell_center_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nx, ny, nz = self.domain_nx
        x = self.xmin[0] + (np.arange(nx, dtype=np.float64) + 0.5) * self.dx
        y = self.xmin[1] + (np.arange(ny, dtype=np.float64) + 0.5) * self.dy
        z = self.xmin[2] + (np.arange(nz, dtype=np.float64) + 0.5) * self.dz
        return x, y, z


def potential_field_green(
    b3_bottom,
    xmin,
    xmax,
    nz,
    *,
    backend: str = "auto",
    balance_flux: bool = True,
):
    """Compute a Cartesian potential-field extrapolation with Green kernels.

    The FFT backend accelerates the same midpoint-source Green-kernel
    convolution used by the direct NumPy backend.
    """

    b3_bottom_array = np.asarray(b3_bottom, dtype=np.float64)
    if b3_bottom_array.ndim != 2:
        raise ValueError("b3_bottom must be a 2D array")

    nx, ny = b3_bottom_array.shape
    if nx <= 0 or ny <= 0:
        raise ValueError("b3_bottom dimensions must be positive")

    xmin_array = _as_length3_array(xmin, "xmin")
    xmax_array = _as_length3_array(xmax, "xmax")
    nz_value = _as_positive_int(nz, "nz")

    domain_lengths = xmax_array - xmin_array
    if np.any(domain_lengths <= 0.0):
        raise ValueError("domain lengths must be positive")

    if backend not in _VALID_BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(_VALID_BACKENDS)}")

    removed_flux_mean = float(np.mean(b3_bottom_array)) if balance_flux else 0.0
    geometry = PotentialFieldGeometry(
        xmin=tuple(float(value) for value in xmin_array),
        xmax=tuple(float(value) for value in xmax_array),
        domain_nx=(int(nx), int(ny), int(nz_value)),
        dx=float(domain_lengths[0] / nx),
        dy=float(domain_lengths[1] / ny),
        dz=float(domain_lengths[2] / nz_value),
        flux_balanced=bool(balance_flux),
        removed_flux_mean=removed_flux_mean,
    )

    prepared_bottom = b3_bottom_array - removed_flux_mean if balance_flux else b3_bottom_array
    bfield = _compute_potential_field(prepared_bottom, geometry, backend)
    return bfield, geometry


def _as_length3_array(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must be a length-3 array")
    return array


def _as_positive_int(value, name: str) -> int:
    try:
        int_value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc

    if int_value <= 0:
        raise ValueError(f"{name} must be positive")
    return int_value


def _compute_potential_field(b3_bottom: np.ndarray, geometry: PotentialFieldGeometry, backend: str) -> np.ndarray:
    if backend == "direct":
        return _convolved_potential_field(b3_bottom, geometry, _direct_convolve2d_same)

    if backend == "fft":
        fftconvolve = _require_fftconvolve()
        return _convolved_potential_field(
            b3_bottom,
            geometry,
            lambda source, kernel: fftconvolve(source, kernel, mode="same"),
        )

    try:
        fftconvolve = _require_fftconvolve()
    except ImportError:
        return _convolved_potential_field(b3_bottom, geometry, _direct_convolve2d_same)

    return _convolved_potential_field(
        b3_bottom,
        geometry,
        lambda source, kernel: fftconvolve(source, kernel, mode="same"),
    )


def _convolved_potential_field(b3_bottom: np.ndarray, geometry: PotentialFieldGeometry, convolve2d) -> np.ndarray:
    nx, ny, nz = geometry.domain_nx
    bfield = np.empty((3, nx, ny, nz), dtype=np.float64)

    for iz in range(nz):
        z_offset = (iz + 0.5) * geometry.dz
        kernels = _green_kernels(nx, ny, geometry.dx, geometry.dy, z_offset)
        for component in range(3):
            bfield[component, :, :, iz] = convolve2d(b3_bottom, kernels[component])

    return bfield


def _green_kernels(nx: int, ny: int, dx: float, dy: float, z_offset: float) -> np.ndarray:
    x_offsets = (np.arange(-(nx - 1), nx, dtype=np.float64) * dx)[:, None]
    y_offsets = (np.arange(-(ny - 1), ny, dtype=np.float64) * dy)[None, :]
    r_squared = x_offsets * x_offsets + y_offsets * y_offsets + z_offset * z_offset
    r_cubed = r_squared * np.sqrt(r_squared)
    scale = _GREEN_NORMALIZATION * dx * dy / r_cubed

    kernels = np.empty((3, 2 * nx - 1, 2 * ny - 1), dtype=np.float64)
    kernels[0] = scale * x_offsets
    kernels[1] = scale * y_offsets
    kernels[2] = scale * z_offset
    return kernels


def _direct_convolve2d_same(source: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    nx, ny = source.shape
    output = np.zeros((nx, ny), dtype=np.float64)
    x_origin = nx - 1
    y_origin = ny - 1

    for ix in range(nx):
        for iy in range(ny):
            total = 0.0
            for sx in range(nx):
                for sy in range(ny):
                    total += source[sx, sy] * kernel[ix - sx + x_origin, iy - sy + y_origin]
            output[ix, iy] = total

    return output


def _require_fftconvolve():
    try:
        from scipy.signal import fftconvolve
    except ImportError as exc:
        raise ImportError("backend='fft' requires scipy.signal.fftconvolve") from exc

    return fftconvolve
