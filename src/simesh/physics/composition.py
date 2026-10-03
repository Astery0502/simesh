"""Fully ionized H/He composition shared by MHD and emission models."""

from dataclasses import dataclass
import numpy as np

PROTON_MASS_G = 1.67262192369e-24
BOLTZMANN_ERG_K = 1.380649e-16


@dataclass(frozen=True)
class CoronalComposition:
    """Fully ionized H/He, ignoring electron mass; abundance is n_He/n_H."""
    helium_abundance: float = .1

    def __post_init__(self):
        if not np.isfinite(self.helium_abundance) or self.helium_abundance < 0:
            raise ValueError("helium abundance must be finite and nonnegative")

    def number_density(self, mass_density_cgs, *, convention="electron"):
        """Convert g/cm³ to n_e, n_H, or sqrt(n_e n_H) in cm^-3.

        Conventions are electron, amrvac-hydrogen and electron-hydrogen,
        respectively, for the explicitly fully ionized composition.
        """
        rho = np.asarray(mass_density_cgs, dtype=float)
        if not np.isfinite(rho).all() or np.any(rho < 0):
            raise ValueError("mass density must be finite and nonnegative")
        h = self.helium_abundance
        nh = rho / ((1 + 4*h)*PROTON_MASS_G)
        return nh*self._number_density_factor(convention)

    def _number_density_factor(self, convention):
        factors = {"electron": 1+2*self.helium_abundance, "amrvac-hydrogen": 1.,
                   "electron-hydrogen": np.sqrt(1+2*self.helium_abundance)}
        try:
            return factors[convention]
        except KeyError:
            raise ValueError("unknown number-density convention") from None

    def temperature(self, mass_density_cgs, thermal_pressure_cgs):
        """p = (2+3a) n_H k_B T. Pressure is thermal, not total energy."""
        nh = self.number_density(mass_density_cgs, convention="amrvac-hydrogen")
        p = np.asarray(thermal_pressure_cgs, dtype=float)
        if not np.isfinite(p).all() or np.any(p <= 0) or np.any(nh <= 0):
            raise ValueError("EOS temperature requires positive density and thermal pressure")
        return p / ((2+3*self.helium_abundance)*nh*BOLTZMANN_ERG_K)
