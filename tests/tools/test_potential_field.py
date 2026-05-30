import os
import time

import numpy as np
import pytest

import simesh.tools.potential_field as potential_field_module
from simesh.tools import PotentialFieldGeometry, potential_field_green


def _explicit_direct_sum(b3_bottom, xmin, xmax, nz, *, balance_flux=True):
    b3_bottom = np.asarray(b3_bottom, dtype=np.float64)
    xmin = np.asarray(xmin, dtype=np.float64)
    xmax = np.asarray(xmax, dtype=np.float64)
    nx, ny = b3_bottom.shape
    dx, dy, dz = (xmax - xmin) / np.array([nx, ny, nz], dtype=np.float64)
    source = b3_bottom - np.mean(b3_bottom) if balance_flux else b3_bottom
    output = np.zeros((3, nx, ny, nz), dtype=np.float64)

    x = xmin[0] + (np.arange(nx, dtype=np.float64) + 0.5) * dx
    y = xmin[1] + (np.arange(ny, dtype=np.float64) + 0.5) * dy
    z = xmin[2] + (np.arange(nz, dtype=np.float64) + 0.5) * dz
    area_scale = dx * dy / (2.0 * np.pi)

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                for sx in range(nx):
                    for sy in range(ny):
                        rx = x[ix] - x[sx]
                        ry = y[iy] - y[sy]
                        rz = z[iz] - xmin[2]
                        r_squared = rx * rx + ry * ry + rz * rz
                        scale = source[sx, sy] * area_scale / (r_squared * np.sqrt(r_squared))
                        output[0, ix, iy, iz] += scale * rx
                        output[1, ix, iy, iz] += scale * ry
                        output[2, ix, iy, iz] += scale * rz

    return output


def _smooth_balanced_bottom(n):
    x = (np.arange(n, dtype=np.float64) + 0.5) / n
    y = (np.arange(n, dtype=np.float64) + 0.5) / n
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.sin(2.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy)


def _compact_balanced_bottom(n):
    b3_bottom = np.zeros((n, n), dtype=np.float64)
    lo = n // 4
    hi = n // 2
    b3_bottom[lo:hi, lo:hi] = 1.0
    b3_bottom[hi : hi + (hi - lo), lo:hi] = -1.0
    return b3_bottom


def _interior_divergence_curl_residuals(bfield, spacing, margin=2):
    dx, dy, dz = spacing
    bx, by, bz = bfield

    div = (
        bx[margin + 1 : -margin + 1, margin:-margin, margin:-margin]
        - bx[margin - 1 : -margin - 1, margin:-margin, margin:-margin]
    ) / (2.0 * dx)
    div += (
        by[margin:-margin, margin + 1 : -margin + 1, margin:-margin]
        - by[margin:-margin, margin - 1 : -margin - 1, margin:-margin]
    ) / (2.0 * dy)
    div += (
        bz[margin:-margin, margin:-margin, margin + 1 : -margin + 1]
        - bz[margin:-margin, margin:-margin, margin - 1 : -margin - 1]
    ) / (2.0 * dz)

    curl_x = (
        bz[margin:-margin, margin + 1 : -margin + 1, margin:-margin]
        - bz[margin:-margin, margin - 1 : -margin - 1, margin:-margin]
    ) / (2.0 * dy) - (
        by[margin:-margin, margin:-margin, margin + 1 : -margin + 1]
        - by[margin:-margin, margin:-margin, margin - 1 : -margin - 1]
    ) / (2.0 * dz)
    curl_y = (
        bx[margin:-margin, margin:-margin, margin + 1 : -margin + 1]
        - bx[margin:-margin, margin:-margin, margin - 1 : -margin - 1]
    ) / (2.0 * dz) - (
        bz[margin + 1 : -margin + 1, margin:-margin, margin:-margin]
        - bz[margin - 1 : -margin - 1, margin:-margin, margin:-margin]
    ) / (2.0 * dx)
    curl_z = (
        by[margin + 1 : -margin + 1, margin:-margin, margin:-margin]
        - by[margin - 1 : -margin - 1, margin:-margin, margin:-margin]
    ) / (2.0 * dx) - (
        bx[margin:-margin, margin + 1 : -margin + 1, margin:-margin]
        - bx[margin:-margin, margin - 1 : -margin - 1, margin:-margin]
    ) / (2.0 * dy)

    interior = bfield[:, margin:-margin, margin:-margin, margin:-margin]
    scale = np.max(np.abs(interior)) / min(spacing)
    curl_norm = max(np.max(np.abs(curl_x)), np.max(np.abs(curl_y)), np.max(np.abs(curl_z)))
    return np.max(np.abs(div)) / scale, curl_norm / scale


def test_public_import_and_geometry_shell():
    b3_bottom = np.ones((2, 3), dtype=np.float64)

    bfield, geometry = potential_field_green(
        b3_bottom,
        xmin=[0.0, 2.0, 10.0],
        xmax=[4.0, 8.0, 22.0],
        nz=6,
        backend="direct",
    )

    assert isinstance(geometry, PotentialFieldGeometry)
    assert bfield.shape == (3, 2, 3, 6)
    assert np.all(np.isfinite(bfield))
    assert geometry.xmin == (0.0, 2.0, 10.0)
    assert geometry.xmax == (4.0, 8.0, 22.0)
    assert geometry.domain_nx == (2, 3, 6)
    assert geometry.spacing == (2.0, 2.0, 2.0)


def test_cell_center_coordinates_are_reconstructable():
    _, geometry = potential_field_green(
        np.zeros((2, 2), dtype=np.float64),
        xmin=[0.0, 0.0, 1.0],
        xmax=[2.0, 4.0, 5.0],
        nz=2,
    )

    x, y, z = geometry.cell_center_coordinates()

    assert np.array_equal(x, np.array([0.5, 1.5]))
    assert np.array_equal(y, np.array([1.0, 3.0]))
    assert np.array_equal(z, np.array([2.0, 4.0]))


def test_flux_balance_metadata_records_removed_mean():
    b3_bottom = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    _, balanced = potential_field_green(b3_bottom, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 2)
    _, unbalanced = potential_field_green(
        b3_bottom,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        2,
        balance_flux=False,
    )

    assert balanced.flux_balanced is True
    assert balanced.removed_flux_mean == 2.5
    assert unbalanced.flux_balanced is False
    assert unbalanced.removed_flux_mean == 0.0


def test_balance_flux_changes_field_by_subtracting_bottom_mean():
    b3_bottom = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float64)
    expected = _explicit_direct_sum(
        b3_bottom,
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        2,
        balance_flux=True,
    )

    bfield, geometry = potential_field_green(
        b3_bottom,
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        2,
        backend="direct",
    )

    assert geometry.removed_flux_mean == np.mean(b3_bottom)
    assert np.allclose(bfield, expected)


def test_direct_backend_matches_explicit_small_grid_sum():
    b3_bottom = np.array(
        [
            [1.0, -0.5],
            [0.25, 2.0],
            [-1.5, 0.75],
        ],
        dtype=np.float64,
    )
    xmin = [-1.0, 0.5, 2.0]
    xmax = [2.0, 3.5, 5.0]
    nz = 2

    bfield, _ = potential_field_green(
        b3_bottom,
        xmin,
        xmax,
        nz,
        backend="direct",
        balance_flux=False,
    )
    expected = _explicit_direct_sum(b3_bottom, xmin, xmax, nz, balance_flux=False)

    assert np.allclose(bfield, expected, rtol=1.0e-14, atol=1.0e-14)


def test_balanced_uniform_bottom_field_returns_zero_field():
    bfield, geometry = potential_field_green(
        np.ones((3, 4), dtype=np.float64),
        [0.0, 0.0, 0.0],
        [3.0, 4.0, 2.0],
        2,
    )

    assert geometry.removed_flux_mean == 1.0
    assert np.array_equal(bfield, np.zeros((3, 3, 4, 2), dtype=np.float64))


def test_single_source_sign_orientation():
    bfield, _ = potential_field_green(
        np.array([[1.0]], dtype=np.float64),
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        1,
        backend="direct",
        balance_flux=False,
    )

    assert bfield[0, 0, 0, 0] == 0.0
    assert bfield[1, 0, 0, 0] == 0.0
    assert bfield[2, 0, 0, 0] == pytest.approx(2.0 / np.pi)


def test_invalid_b3_bottom_raises_value_error():
    with pytest.raises(ValueError, match="b3_bottom must be a 2D array"):
        potential_field_green(np.zeros((2, 2, 1)), [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 2)


def test_invalid_xmin_and_xmax_raise_value_error():
    with pytest.raises(ValueError, match="xmin must be a length-3 array"):
        potential_field_green(np.zeros((2, 2)), [0.0, 0.0], [1.0, 1.0, 1.0], 2)

    with pytest.raises(ValueError, match="xmax must be a length-3 array"):
        potential_field_green(np.zeros((2, 2)), [0.0, 0.0, 0.0], [1.0, 1.0], 2)


def test_invalid_nz_and_domain_lengths_raise_value_error():
    with pytest.raises(ValueError, match="nz must be positive"):
        potential_field_green(np.zeros((2, 2)), [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0)

    with pytest.raises(ValueError, match="domain lengths must be positive"):
        potential_field_green(np.zeros((2, 2)), [0.0, 0.0, 0.0], [1.0, 0.0, 1.0], 2)


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="unknown backend"):
        potential_field_green(
            np.zeros((2, 2)),
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            2,
            backend="missing",
        )


def test_fft_backend_matches_direct_when_scipy_is_available():
    pytest.importorskip("scipy.signal")
    b3_bottom = np.array(
        [
            [1.0, -2.0, 0.5],
            [0.25, 1.5, -0.75],
            [-1.25, 0.0, 2.0],
            [0.5, -0.5, 1.0],
        ],
        dtype=np.float64,
    )

    direct, _ = potential_field_green(
        b3_bottom,
        [-1.0, -2.0, 0.0],
        [3.0, 1.0, 2.0],
        3,
        backend="direct",
    )
    fft, _ = potential_field_green(
        b3_bottom,
        [-1.0, -2.0, 0.0],
        [3.0, 1.0, 2.0],
        3,
        backend="fft",
    )

    assert np.allclose(fft, direct, rtol=1.0e-13, atol=1.0e-13)


def test_auto_backend_returns_finite_output():
    bfield, _ = potential_field_green(
        np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float64),
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 1.0],
        2,
        backend="auto",
    )

    assert bfield.shape == (3, 2, 2, 2)
    assert np.all(np.isfinite(bfield))


def test_auto_backend_falls_back_to_direct_when_scipy_is_unavailable(monkeypatch):
    def raise_import_error():
        raise ImportError("no scipy")

    monkeypatch.setattr(potential_field_module, "_require_fftconvolve", raise_import_error)
    b3_bottom = np.array([[1.0, -1.0], [0.25, -0.25]], dtype=np.float64)

    auto, _ = potential_field_green(
        b3_bottom,
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 1.0],
        2,
        backend="auto",
    )
    direct, _ = potential_field_green(
        b3_bottom,
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 1.0],
        2,
        backend="direct",
    )

    assert np.array_equal(auto, direct)


def test_forced_fft_backend_requires_scipy(monkeypatch):
    def raise_import_error():
        raise ImportError("backend='fft' requires scipy.signal.fftconvolve")

    monkeypatch.setattr(potential_field_module, "_require_fftconvolve", raise_import_error)

    with pytest.raises(ImportError, match="requires scipy.signal.fftconvolve"):
        potential_field_green(
            np.zeros((2, 2)),
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            2,
            backend="fft",
        )


def test_smooth_balanced_source_has_small_interior_divergence_and_curl():
    bfield, geometry = potential_field_green(
        _smooth_balanced_bottom(16),
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        16,
        backend="direct",
    )

    divergence, curl = _interior_divergence_curl_residuals(bfield, geometry.spacing, margin=2)

    assert divergence < 0.05
    assert curl < 0.04


def test_smooth_balanced_residuals_decrease_under_refinement():
    coarse, coarse_geometry = potential_field_green(
        _smooth_balanced_bottom(12),
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        12,
        backend="direct",
    )
    fine, fine_geometry = potential_field_green(
        _smooth_balanced_bottom(24),
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        24,
        backend="auto",
    )

    coarse_divergence, coarse_curl = _interior_divergence_curl_residuals(coarse, coarse_geometry.spacing)
    fine_divergence, fine_curl = _interior_divergence_curl_residuals(fine, fine_geometry.spacing)

    assert fine_divergence < coarse_divergence
    assert fine_curl < coarse_curl


def test_balanced_dipole_preserves_symmetry_and_sign_orientation():
    b3_bottom = np.zeros((9, 9), dtype=np.float64)
    b3_bottom[2, 4] = -1.0
    b3_bottom[6, 4] = 1.0

    bfield, _ = potential_field_green(
        b3_bottom,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        5,
        backend="direct",
    )

    assert np.allclose(bfield[0], bfield[0][::-1, :, :])
    assert np.allclose(bfield[0], bfield[0][:, ::-1, :])
    assert np.allclose(bfield[1], -bfield[1][::-1, :, :])
    assert np.allclose(bfield[1], -bfield[1][:, ::-1, :])
    assert np.allclose(bfield[2], -bfield[2][::-1, :, :])
    assert np.allclose(bfield[2], bfield[2][:, ::-1, :])
    assert bfield[2, 6, 4, 0] > 0.0
    assert bfield[2, 2, 4, 0] < 0.0
    assert bfield[0, 4, 4, 0] < 0.0


def test_compact_balanced_source_is_finite_and_decays_with_height():
    b3_bottom = np.zeros((16, 16), dtype=np.float64)
    b3_bottom[6:10, 6:10] = 1.0
    b3_bottom[2:6, 2:6] = -1.0

    bfield, _ = potential_field_green(
        b3_bottom,
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        16,
        backend="auto",
    )

    near_b3 = np.max(np.abs(bfield[2, :, :, 0]))
    top_b3 = np.max(np.abs(bfield[2, :, :, -1]))
    near_magnitude = np.mean(np.linalg.norm(bfield[:, :, :, 0], axis=0))
    top_magnitude = np.mean(np.linalg.norm(bfield[:, :, :, -1], axis=0))

    assert np.all(np.isfinite(bfield))
    assert top_b3 < 0.05 * near_b3
    assert top_magnitude < 0.05 * near_magnitude


@pytest.mark.heavy
def test_heavy_64_cubed_auto_backend_performance():
    if os.environ.get("SIMESH_RUN_HEAVY_TESTS") != "1":
        pytest.skip("set SIMESH_RUN_HEAVY_TESTS=1 to run heavy potential-field performance tests")

    started = time.perf_counter()
    bfield, geometry = potential_field_green(
        _compact_balanced_bottom(64),
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        64,
        backend="auto",
    )
    elapsed = time.perf_counter() - started

    print(f"potential_field_green 64x64x64 backend=auto elapsed={elapsed:.6f}s")
    assert bfield.shape == (3, 64, 64, 64)
    assert geometry.domain_nx == (64, 64, 64)
    assert np.all(np.isfinite(bfield))
    assert elapsed < 30.0


@pytest.mark.heavy
def test_heavy_128_cubed_fft_backend_performance():
    if os.environ.get("SIMESH_RUN_HEAVY_TESTS") != "1":
        pytest.skip("set SIMESH_RUN_HEAVY_TESTS=1 to run heavy potential-field performance tests")
    pytest.importorskip("scipy.signal")

    started = time.perf_counter()
    bfield, geometry = potential_field_green(
        _compact_balanced_bottom(128),
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        128,
        backend="fft",
    )
    elapsed = time.perf_counter() - started

    print(f"potential_field_green 128x128x128 backend=fft elapsed={elapsed:.6f}s")
    assert bfield.shape == (3, 128, 128, 128)
    assert geometry.domain_nx == (128, 128, 128)
    assert np.all(np.isfinite(bfield))
    assert elapsed < 120.0
