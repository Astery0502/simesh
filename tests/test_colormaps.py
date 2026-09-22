"""Contract checks for optional, independently mutable AIA palettes."""
import numpy as np
import pytest

pytest.importorskip('matplotlib')
from simesh.colormaps import AIA_WAVELENGTHS, aia_colormap


def test_local_palettes_are_independent():
    for wave in AIA_WAVELENGTHS:
        cmap = aia_colormap(wave)
        assert cmap.name == f'sdoaia{wave}'
        rgba = cmap(np.linspace(0, 1, 256))
        assert rgba.shape == (256, 4)
        assert np.isfinite(rgba).all()
        assert ((rgba >= 0) & (rgba <= 1)).all()
        original = cmap.get_bad().copy()
        cmap.set_bad('red')
        np.testing.assert_array_equal(aia_colormap(wave).get_bad(), original)


@pytest.mark.parametrize('wave', [True, 13, 192, 171.0, '171', None])
def test_channel_selection_is_explicit(wave):
    with pytest.raises(ValueError):
        aia_colormap(wave)


@pytest.mark.parametrize('channel', ['FUV', 'NUV'])
def test_iris_spectral_maps_are_independent(channel):
    from simesh.colormaps import iris_colormap
    cmap = iris_colormap(channel)
    assert cmap.name == f'irissji{channel}'
    original = cmap.get_bad().copy()
    cmap.set_bad('red')
    np.testing.assert_array_equal(iris_colormap(channel).get_bad(), original)
    with pytest.raises(ValueError):
        iris_colormap(1354)


def test_eis_intensity_convention():
    from matplotlib import colormaps
    from simesh.colormaps import eis_intensity_colormap
    samples = np.linspace(0, 1, 256)
    cmap = eis_intensity_colormap()
    np.testing.assert_array_equal(cmap(samples), colormaps['Blues_r'](samples))
    original = cmap.get_bad().copy()
    cmap.set_bad('red')
    np.testing.assert_array_equal(eis_intensity_colormap().get_bad(), original)
