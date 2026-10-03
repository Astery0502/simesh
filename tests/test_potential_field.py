"""Prepared FFT convolution retains the direct Green-field definition."""

import numpy as np
import pytest

from simesh.tools import potential_field_green
import simesh.tools.potential_field as implementation


@pytest.mark.parametrize("shape", [(1,1), (1,5), (4,3), (5,6)])
@pytest.mark.parametrize("balance", [False, True])
def test_reused_bottom_spectrum_matches_direct_and_independent_convolutions(shape, balance):
    signal = pytest.importorskip("scipy.signal")
    bottom = np.random.default_rng(71).normal(size=shape)
    args = (bottom, (-.7,.2,1.), (1.3,3.2,2.5), 3)
    actual, geometry = potential_field_green(*args, backend="fft", balance_flux=balance)
    direct, _ = potential_field_green(*args, backend="direct", balance_flux=balance)
    prepared = bottom-bottom.mean() if balance else bottom
    expected = np.empty_like(actual)
    for height in range(3):
        kernels = implementation._green_kernels(*shape, geometry.dx, geometry.dy,
                                                (height+.5)*geometry.dz)
        for component in range(3):
            expected[component,:,:,height] = signal.fftconvolve(prepared, kernels[component], mode="same")
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-14)
    np.testing.assert_allclose(actual, direct, rtol=2e-14, atol=1e-14)
    assert actual.flags.owndata and not np.shares_memory(actual, bottom)


def test_auto_potential_field_retains_optional_scipy_fallback(monkeypatch):
    def unavailable(source):
        raise ImportError("SciPy is unavailable")

    monkeypatch.setattr(implementation, "_fft_convolver", unavailable)
    args = (np.arange(12.).reshape(3,4), (0,0,0), (1,1,1), 2)
    actual, _ = potential_field_green(*args)
    direct, _ = potential_field_green(*args, backend="direct")
    np.testing.assert_array_equal(actual, direct)
    with pytest.raises(ImportError, match="SciPy"):
        potential_field_green(*args, backend="fft")
