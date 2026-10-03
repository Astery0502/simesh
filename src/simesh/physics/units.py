"""Explicit magnetic and MHD unit conversions, independent of field consumers."""

from dataclasses import dataclass
import math
import numpy as np

from .composition import CoronalComposition, PROTON_MASS_G, BOLTZMANN_ERG_K


@dataclass(frozen=True)
class MagneticUnits:
    """SI factors per stored field/coordinate unit, with scalar permeability.

    The default is the conventional vacuum approximation 4*pi*1e-7 H/m.
    Supply permeability_h_m explicitly when a different or more precise value
    is required. Metadata labels alone never establish these conversion factors.
    """
    field_tesla: float
    length_m: float
    permeability_h_m: float = 4*np.pi*1e-7

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in
               (self.field_tesla,self.length_m,self.permeability_h_m)):
            raise ValueError("magnetic field, length and permeability factors must be finite and positive")


def _positive_scale(value, label):
    if (isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0 or
            not np.isfinite(value) or value <= 0):
        raise ValueError(f"{label} must be finite positive scalars")
    return float(value)


@dataclass(frozen=True, kw_only=True)
class MHDUnits:
    """Gaussian CGS multipliers per stored MHD quantity and coordinate.

    Parameters
    ----------
    density_g_cm3 : float
        Grams per cubic centimeter per stored density.
    momentum_g_cm2_s : float
        Grams per square centimeter per second per stored momentum density.
    energy_erg_cm3 : float
        Ergs per cubic centimeter per stored total or internal energy density.
    field_gauss : float
        Gauss per stored magnetic field; magnetic pressure is B**2/(8*pi).
    length_cm : float
        Centimeters per coordinate unit, retained for downstream consumers.

    Notes
    -----
    Input factors are independent and never inferred from field metadata.
    Recovery outputs use CGS, kelvin and dimensionless ratios/status.
    """

    density_g_cm3: float
    momentum_g_cm2_s: float
    energy_erg_cm3: float
    field_gauss: float
    length_cm: float

    def __post_init__(self):
        for name in ("density_g_cm3", "momentum_g_cm2_s", "energy_erg_cm3",
                     "field_gauss", "length_cm"):
            object.__setattr__(self, name, _positive_scale(getattr(self, name), "MHD unit multipliers"))

    @classmethod
    def solar(cls, *, length_cm=1.e9, number_density_cm3=1.e9,
              temperature_k=1.e6, composition=CoronalComposition()):
        """Construct the common AMRVAC solar-coronal CGS normalization.

        Parameters
        ----------
        length_cm : float, optional
            Coordinate scale in cm; the default is 10 Mm.
        number_density_cm3 : float, optional
            Hydrogen nucleus density scale in cm^-3, not electron density.
        temperature_k : float, optional
            Temperature scale in kelvin.
        composition : CoronalComposition, optional
            Fully ionized H/He abundance; use the same composition in IdealMHD.

        Returns
        -------
        MHDUnits
            rho0=(1+4a)*mp*nH0, p0=(2+3a)*nH0*kB*T0,
            v0=sqrt(p0/rho0), momentum0=rho0*v0, energy0=p0 and
            B0=sqrt(4*pi*p0). The velocity scale is not the sound speed.

        Notes
        -----
        This preset follows AMRVAC with si_unit=False, eq_state_units=True and
        fully ionized H/He. It is a common coronal choice, not a universal solar
        standard. Match the simulation's scales explicitly when they differ.
        Constants are shared with CoronalComposition (mp=1.67262192369e-24 g,
        kB=1.380649e-16 erg/K); older AMRVAC constants differ slightly.
        """
        if not isinstance(composition, CoronalComposition):
            raise TypeError("composition must be an explicit CoronalComposition")
        length, number_density, temperature = (
            _positive_scale(value, "solar scales")
            for value in (length_cm, number_density_cm3, temperature_k))
        a = composition.helium_abundance
        rho = ((1 + 4*a)*PROTON_MASS_G)*number_density
        pressure = ((2 + 3*a)*BOLTZMANN_ERG_K)*number_density*temperature
        if not (math.isfinite(rho) and rho > 0 and math.isfinite(pressure) and pressure > 0):
            raise ValueError("solar density and pressure scales must be finite and positive")
        velocity = math.sqrt(pressure)/math.sqrt(rho)
        return cls(density_g_cm3=rho, momentum_g_cm2_s=rho*velocity,
                   energy_erg_cm3=pressure, field_gauss=math.sqrt(4*math.pi)*math.sqrt(pressure),
                   length_cm=length)

    @property
    def velocity_cm_s(self):
        """Centimeters per second per stored velocity, from momentum/density."""
        return self.momentum_g_cm2_s / self.density_g_cm3

    @property
    def time_s(self):
        """Seconds per code time, from length/velocity."""
        return self.length_cm / self.velocity_cm_s

    @property
    def magnetic_si(self):
        """Equivalent SI normalization for the separate magnetic diagnostics."""
        return MagneticUnits(field_tesla=self.field_gauss*1.e-4, length_m=self.length_cm*.01)
