"""Array-only EUV response models with explicit emission-measure conventions."""

from dataclasses import dataclass
import numpy as np

from ._aia171_table import LOG_T, RESPONSE, UPSTREAM_COMMIT
from . import _euv_tables
from .composition import CoronalComposition
from .._validation import frozen_array


_RESPONSE_GRID = frozen_array(LOG_T, float)
_LOG_RESPONSE = frozen_array(np.log10(RESPONSE), float)
_RESPONSE_SLOPES = frozen_array(np.diff(_LOG_RESPONSE)/np.diff(_RESPONSE_GRID), float)
_EUV_TABLES = {}
for _wave, (_grid, _values, _logarithmic) in _euv_tables.RESPONSES.items():
    _ordinates = np.log10(np.maximum(_values, 1e-99)) if _logarithmic else np.asarray(_values)
    _EUV_TABLES[_wave] = (frozen_array(_grid, float), frozen_array(_ordinates, float),
                        frozen_array(np.diff(_ordinates)/np.diff(_grid), float),
                        0 if _logarithmic else 1)


@dataclass(frozen=True)
class AIA171:
    """Historical response, log10(T)-log10(R) interpolation, zero outside table.

    R has DN cm^5 s^-1 pixel^-1 under the selected emission-measure convention.
    electron is the explicit physical baseline; amrvac-hydrogen reproduces the
    upstream eq_state_units=True H/He density factor. Neither fixes the table's
    undocumented original calibration settings.
    """
    density_convention: str = "electron"
    composition: CoronalComposition = CoronalComposition()

    def __post_init__(self):
        if self.density_convention not in ("electron", "amrvac-hydrogen", "electron-hydrogen"):
            raise ValueError("unknown emission-measure density convention")
        if not isinstance(self.composition, CoronalComposition):
            raise TypeError("composition must be a CoronalComposition")

    @property
    def _table(self):
        return _RESPONSE_GRID, _LOG_RESPONSE, _RESPONSE_SLOPES, 0

    @property
    def emissivity_name(self):
        """Prepared emissivity field name."""
        return "aia171_emissivity"

    @property
    def identity(self):
        """Response, density-convention and composition identity used to check compatible thermal fields."""
        return f"amrvac-{UPSTREAM_COMMIT}-171-{self.density_convention}-He{self.composition.helium_abundance:g}"

    def response(self, temperature_k):
        """Interpolate the selected response table; EUV defines each instrument's interpolation axes."""
        t = np.asarray(temperature_k, dtype=float)
        if not np.isfinite(t).all() or np.any(t <= 0):
            raise ValueError("temperature must be finite and positive in kelvin")
        grid, ordinates, _, mode = self._table
        lookup = np.log10(t) if mode == 0 else t
        value = np.interp(lookup, grid, ordinates)
        value = np.where(value > -99., 10.**value, 0.) if mode == 0 else value
        return np.where((lookup >= grid[0]) & (lookup <= grid[-1]), value, 0.)

    def emissivity(self, mass_density_cgs, temperature_k):
        """Evaluate emission from mass density in g/cm³ and positive kelvin temperature."""
        n = self.composition.number_density(mass_density_cgs, convention=self.density_convention)
        return self.from_number_density(n, temperature_k)

    def from_number_density(self, number_density_cm3, temperature_k):
        """Evaluate n² R(T) for number density in cm^-3 under this model convention."""
        n = np.asarray(number_density_cm3, dtype=float)
        if not np.isfinite(n).all() or np.any(n < 0):
            raise ValueError("number density must be finite and nonnegative")
        with np.errstate(over="ignore", invalid="ignore"):
            value = n*n*self.response(temperature_k)
        if not np.isfinite(value).all():
            raise ValueError("unrepresentable thermal emissivity")
        return value


@dataclass(frozen=True)
class EUV(AIA171):
    """Upstream EUV response with explicit fully ionized emission measure.

    Parameters
    ----------
    density_convention : str
        electron-hydrogen uses n_e n_H R(T), as in AMRVAC 3.3. electron and
        amrvac-hydrogen select n_e² and n_H² explicitly.
    composition : CoronalComposition
        Fully ionized H/He composition for emission-measure conversion.
    wavelength : int
        AIA 94, 131, 171, 193, 211, 304, 335; IRIS 1354; or EIS 192, 255,
        263, 264, in Angstrom. Supply by keyword.

    Notes
    -----
    AIA/IRIS interpolate log10(T)-log10(R); EIS interpolates T-R linearly.
    Responses vanish outside their tables; original calibration settings are
    unspecified. This model does not recover a partially ionized simulation EOS.
    Response and emissivity methods are inherited from [AIA171][simesh.AIA171].
    """
    density_convention: str = "electron-hydrogen"
    wavelength: int = 171

    def __post_init__(self):
        super().__post_init__()
        if type(self.wavelength) is not int or self.wavelength not in _EUV_TABLES:
            raise ValueError("unsupported EUV wavelength")

    @property
    def _table(self):
        return _EUV_TABLES[self.wavelength]

    @property
    def identity(self):
        """Pinned response, wavelength, composition and emission-measure identity."""
        return (f"amrvac-{_euv_tables.UPSTREAM_COMMIT}-{self.wavelength}-"
                f"{self.density_convention}-He{self.composition.helium_abundance:g}")

    @property
    def emissivity_name(self):
        """Prepared emissivity field name."""
        return f"euv{self.wavelength}_emissivity"
