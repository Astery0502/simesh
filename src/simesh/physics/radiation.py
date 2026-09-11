"""EUV absorption and radio free-free coefficients on explicitly prepared fields.

These are radiation post-processing models, not thermodynamic EOS recovery.
The H/He absorber solves charge neutrality at fixed total hydrogen density and
temperature. Emission retains the explicitly selected fully ionized convention.
"""

from dataclasses import dataclass
import numpy as np

from .thermal import (AIA171, EUV, CoronalComposition, BOLTZMANN_ERG_K,
                      _check_thermal, _map_thermal)
from ._euv_tables import UPSTREAM_COMMIT
from ..fields import FieldDefinition


def _density_temperature(density, temperature):
    n, t = np.broadcast_arrays(np.asarray(density, dtype=float),
                               np.asarray(temperature, dtype=float))
    if not np.isfinite(n).all() or np.any(n < 0):
        raise ValueError("number density must be finite and nonnegative")
    if not np.isfinite(t).all() or np.any(t <= 0):
        raise ValueError("temperature must be finite and positive in kelvin")
    return n, t


def _saha_log_ratios(temperature):
    base = 1.5*np.log10(temperature)-np.log10(BOLTZMANN_ERG_K)-.48
    with np.errstate(over="ignore"):
        return (base-5040.*13.6/temperature,
                base+np.log10(4.)-5040.*24.587/temperature,
                base-5040.*54.416/temperature)


def _saha_populations(log_ratios, electron_density):
    # Form ratios in log space, retaining neutral populations directly rather
    # than losing hot-plasma absorption to 1 - ionized_fraction cancellation.
    logne = np.log10(electron_density)
    h, he1, he2 = (np.log(10.)*(ratio-logne) for ratio in log_ratios)
    hnorm = np.logaddexp(0., h)
    scale = np.maximum(0., np.maximum(he1, he1+he2))
    he0 = np.exp(-scale)
    he1_weight = np.exp(he1-scale)
    he2_weight = np.exp(he1+he2-scale)
    total = he0+he1_weight+he2_weight
    return np.exp(-hnorm), np.exp(h-hnorm), he0/total, he1_weight/total, he2_weight/total


@dataclass(frozen=True)
class HHeAbsorption:
    """AMRVAC 3.3 photoionization opacity with a radiation-only Saha closure.

    Parameters
    ----------
    helium_abundance : float
        Absorber n_He/n_H; this does not change the emitting gas composition.

    Notes
    -----
    Uses LTE Saha equilibrium at fixed n_H,T, as upstream's fully ionized EOS
    post-processing branch does. This is not a non-LTE prominence model or a
    reconstruction of simulation PI/LTE populations.
    """
    helium_abundance: float = .1

    def __post_init__(self):
        CoronalComposition(self.helium_abundance)

    @property
    def identity(self):
        """Pinned absorber prescription and independent helium abundance."""
        return f"amrvac-{UPSTREAM_COMMIT}-HHe-Saha-He{self.helium_abundance:g}"

    def opacity(self, hydrogen_density_cm3, temperature_k, *, wavelength):
        """Evaluate H I, He I and He II absorption in cm^-1.

        Parameters
        ----------
        hydrogen_density_cm3 : array-like
            Nonnegative total hydrogen nuclei density in cm^-3.
        temperature_k : array-like
            Positive kelvin temperatures, broadcastable with density.
        wavelength : float
            Positive wavelength in Angstrom; the edges are 912, 504 and 228 A.

        Returns
        -------
        ndarray
            Broadcast opacity, zero in vacuum and beyond the H I edge.
        """
        n, t = _density_temperature(hydrogen_density_cm3, temperature_k)
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("wavelength must be positive and finite in Angstrom")
        result = np.zeros(n.shape)
        active = n > 0
        if wavelength > 912 or not np.any(active):
            return result
        n, t = n[active], t[active]
        log_ratios = _saha_log_ratios(t)
        helium = self.helium_abundance
        lo = np.zeros_like(n)
        hi = np.full_like(n, 1+2*helium)
        electrons = hi.copy()
        for _ in range(64):
            h0, h1, he0, he1, he2 = _saha_populations(log_ratios, np.maximum(n*electrons, 1e-100))
            he_charge = he1+2*he2
            residual = electrons-h1-helium*he_charge
            converged = np.abs(residual) < 1e-10
            if np.all(converged):
                break
            hi = np.where(residual > 0, electrons, hi)
            lo = np.where(residual <= 0, electrons, lo)
            derivative = (1+(h1*h0+helium*(he1*(1-he_charge)
                          +2*he2*(2-he_charge)))/electrons)
            trial = electrons-residual/derivative
            trial = np.where((trial > lo) & (trial < hi), trial, .5*(lo+hi))
            electrons = np.where(converged, electrons, trial)
        else:
            raise ValueError("Saha charge-neutrality solve did not converge")
        ratio = wavelength/171.
        sigma_h = 5.16e-20*ratio**3
        sigma_he0 = 9.25e-19*ratio**2 if wavelength <= 504 else 0.
        sigma_he1 = 7.17e-19*ratio**2.75 if wavelength <= 228 else 0.
        result[active] = n*(h0*sigma_h+helium*(he0*sigma_he0+he1*sigma_he1))
        if not np.isfinite(result).all():
            raise ValueError("unrepresentable H/He opacity")
        return result


@dataclass(frozen=True)
class RadioFreeFree:
    """Thermal radio free-free brightness-temperature coefficients from AMRVAC.

    Parameters
    ----------
    frequency_hz : float
        Positive observing frequency in Hz.
    composition : CoronalComposition
        Fully ionized H/He conversion from mass density to electron density.

    Notes
    -----
    Uses the upstream piecewise logarithmic Gaunt factor, floored at one,
    kappa = 9.78e-3 n_e² gff / (nu² T^1.5), and j_T = T kappa. Output is
    Rayleigh-Jeans brightness temperature; gyro-emission and refraction are absent.
    """
    frequency_hz: float = 17e9
    composition: CoronalComposition = CoronalComposition()

    def __post_init__(self):
        if not np.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("radio frequency must be finite and positive in Hz")
        if not isinstance(self.composition, CoronalComposition):
            raise TypeError("composition must be a CoronalComposition")

    @property
    def density_convention(self):
        """Electron number density is the radio emission-measure input."""
        return "electron"

    @property
    def identity(self):
        """Pinned prescription, frequency and emitting composition."""
        return (f"amrvac-{UPSTREAM_COMMIT}-radio-ff-{self.frequency_hz!r}Hz-"
                f"He{self.composition.helium_abundance:g}")

    def response(self, temperature_k):
        """Return j_T/n_e² in K cm^5 for positive kelvin temperatures."""
        _, t = _density_temperature(0., temperature_k)
        logt = np.log(t)
        lognu = np.log(self.frequency_hz)
        gaunt = np.maximum(1., np.where(t < 2e5, 18.2+1.5*logt-lognu, 24.5+logt-lognu))
        with np.errstate(over="ignore"):
            response = np.exp(np.log(9.78e-3*gaunt)-2*lognu-.5*logt)
        if not np.isfinite(response).all():
            raise ValueError("unrepresentable radio response")
        return response

    def from_number_density(self, number_density_cm3, temperature_k):
        """Return j_T in K/cm from electron density in cm^-3 and temperature in K."""
        n, t = _density_temperature(number_density_cm3, temperature_k)
        with np.errstate(over="ignore", invalid="ignore"):
            result = n*n*self.response(t)
        if not np.isfinite(result).all():
            raise ValueError("unrepresentable radio emissivity")
        return result

    def emissivity(self, mass_density_cgs, temperature_k):
        """Return j_T in K/cm from mass density in g/cm³ and temperature in K."""
        return self.from_number_density(self.composition.number_density(mass_density_cgs), temperature_k)


def radiation_fields(thermodynamics, *, model=EUV(), absorption=None, memory_limit=None):
    """Detach emissivity and opacity nodes for ordered radiative transfer.

    Parameters
    ----------
    thermodynamics : Fields
        Number-density and kelvin fields from thermal_fields with matching model.
    model : EUV, AIA171 or RadioFreeFree
        Explicit emitting model; radio coefficients produce brightness temperature.
    absorption : HHeAbsorption, optional
        EUV absorber. None gives zero EUV opacity; radio always uses its own
        free-free opacity and requires None here.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Independent j and kappa (cm^-1) nodes preserving common valid support.
        EUV j is DN s^-1 pixel^-1 cm^-1; radio j is K cm^-1.

    Notes
    -----
    Coefficients are evaluated before interpolation. Use radiative_los with
    increasing subdivisions to check convergence; consumers do not read Sources
    or restore missing coverage. Input mass density and temperature remain intact.
    """
    _check_thermal(thermodynamics, model, halo=0)
    radio = type(model) is RadioFreeFree
    if type(model) not in (AIA171, EUV, RadioFreeFree):
        raise TypeError("radiation coefficients require AIA171, EUV or RadioFreeFree")
    if absorption is not None and (radio or not isinstance(absorption, HHeAbsorption)):
        raise TypeError("only EUV models accept HHeAbsorption")
    units = "K" if radio else "DN s^-1 pixel^-1"
    definitions = (FieldDefinition("emissivity", units+" cm^-1", "prepared-node"),
                   FieldDefinition("opacity", "cm^-1", "prepared-node"))
    density_factor = model.composition._number_density_factor(model.density_convention)

    def coefficients(data):
        n, t = data[..., 0], data[..., 1]
        j = model.from_number_density(n, t)
        if radio:
            opacity = j/t
        elif absorption is None:
            opacity = np.zeros_like(j)
        else:
            # Recover n_H using exactly the emitter's declared density factor;
            # the absorber abundance enters only its own opacity/charge closure.
            opacity = absorption.opacity(n/density_factor, t, wavelength=getattr(model, "wavelength", 171))
        return np.stack((j, opacity), axis=-1)

    return _map_thermal(thermodynamics, definitions, coefficients,
        "radiation-coefficients-before-interpolation", memory_limit,
        stats={"radiation_units": units, "absorption": absorption.identity if absorption else
               ("radio-free-free" if radio else "none")},
        scratch_per_node=1024 if absorption else 256)
