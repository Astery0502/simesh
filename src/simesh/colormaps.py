"""Offline solar display palettes; Matplotlib is loaded only when requested.

AIA RGB controls are derived from SunPy 7.0.0 (BSD-2-Clause), whose AIA tables
trace to SSWIDL aia_lct.pro. See ASSETS.md and LICENSES/SunPy-BSD-2-Clause.txt.
These palettes do not change instrument responses or intensity normalization.
"""

from functools import lru_cache
from importlib.resources import files
from operator import index

import numpy as np

__all__ = ['AIA_WAVELENGTHS', 'aia_colormap', 'iris_colormap', 'eis_intensity_colormap']

AIA_WAVELENGTHS = (94, 131, 171, 193, 211, 304, 335, 1600, 1700, 4500)


@lru_cache(maxsize=2)
def _tables(instrument="aia"):
    with files('simesh').joinpath('_data', f'{instrument}_colormaps.npz').open('rb') as stream:
        with np.load(stream, allow_pickle=False) as archive:
            tables = {(int(key) if instrument == "aia" else key): archive[key] for key in archive.files}
    for table in tables.values():
        table.flags.writeable = False
    return tables


def aia_colormap(wavelength):
    """Return the classic SDO/AIA palette for a wavelength in Angstroms.

    Parameters
    ----------
    wavelength : int
        Channel: 94, 131, 171, 193, 211, 304, 335, 1600, 1700 or 4500 Angstroms.
        Use the full integer channel, not an abbreviation or a unit-bearing object.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Independent 256-color map named sdoaia followed by the channel number.
        Pass directly as cmap to Matplotlib plotting and colorbar functions.

    Notes
    -----
    Requires the plot extra. Uses bundled SunPy-derived tables without network
    access or a SunPy installation. Does not register global Matplotlib names;
    changing bad/under/over colors on the result does not affect later calls.
    """
    if isinstance(wavelength, (bool, np.bool_)):
        raise ValueError('wavelength must be an integer AIA channel in Angstroms')
    try:
        wave = index(wavelength)
    except TypeError:
        raise ValueError('wavelength must be an integer AIA channel in Angstroms') from None
    if wave not in AIA_WAVELENGTHS:
        raise ValueError(f'unsupported AIA wavelength {wave}; choose from {AIA_WAVELENGTHS}')
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(f'sdoaia{wave}', _tables()[wave], N=256)


def iris_colormap(channel="FUV"):
    """Return a standard IRIS spectral FUV or NUV display palette.

    Parameters
    ----------
    channel : str
        FUV or NUV detector family. Use FUV for the 1354 Angstrom spectral line;
        this is a detector-family convention, not a dedicated 1354 line palette.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Independent 256-color map named irissjiFUV or irissjiNUV, matching SunPy.

    Notes
    -----
    Requires the plot extra. Bundled SunPy tables are available offline; their
    historical irissji names also cover these spectral detector palettes.
    """
    if not isinstance(channel, str) or channel not in ('FUV', 'NUV'):
        raise ValueError('IRIS spectral channel must be FUV or NUV')
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(f'irissji{channel}', _tables('iris')[channel], N=256)


def eis_intensity_colormap():
    """Return the Blues_r palette used for intensity maps by EISPAC.

    Returns
    -------
    matplotlib.colors.Colormap
        Independent Matplotlib Blues_r palette: dark blue at low intensity,
        nearly white at high intensity.

    Notes
    -----
    Requires the plot extra. This is EISPAC's intensity-display convention, not
    a wavelength-specific standard. Choose the intensity normalization separately.
    """
    from matplotlib import colormaps
    return colormaps['Blues_r'].copy()
