"""Explicit classical ideal-gas MHD recovery on completed native Fields.

Energy is a density per volume, selected as total (internal + kinetic +
magnetic) or internal. No file name, field label or metadata selects the EOS,
energy definition, composition or normalization.
"""

from dataclasses import dataclass
from enum import IntFlag
import math

import numpy as np

from .._validation import admit, array_bytes
from ..diagnostics import MagneticUnits, _components
from ..fields import FieldDefinition, publish, require_fields
from .thermal import CoronalComposition, PROTON_MASS_G, BOLTZMANN_ERG_K


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


@dataclass(frozen=True, kw_only=True)
class IdealMHD:
    """Classical m=rho*v, constant-gamma ideal gas, fully ionized H/He.

    ``energy_kind`` is exactly ``"total"`` or ``"internal"``. Total includes
    kinetic and magnetic energy but no rest mass, gravity, radiation, electric
    field, cleaning-variable energy or background-field splitting terms.
    ``gamma``, ``composition`` and all units must be supplied explicitly.
    """

    gamma: float
    energy_kind: str
    composition: CoronalComposition
    units: MHDUnits

    def __post_init__(self):
        if (isinstance(self.gamma, (bool, np.bool_)) or np.ndim(self.gamma) != 0 or
                not np.isfinite(self.gamma) or self.gamma <= 1):
            raise ValueError("gamma must be a finite scalar greater than one")
        if self.energy_kind not in ("total", "internal"):
            raise ValueError("energy_kind must be total or internal energy density")
        if not isinstance(self.composition, CoronalComposition):
            raise TypeError("composition must be an explicit CoronalComposition")
        if not isinstance(self.units, MHDUnits):
            raise TypeError("units must be an MHDUnits configuration")
        object.__setattr__(self, "gamma", float(self.gamma))


class MHDStatus(IntFlag):
    """Node flags; inspect native windows, never interpolate these bit masks."""

    OK = 0
    NONFINITE_INPUT = 1
    NONPOSITIVE_DENSITY = 2
    NONPOSITIVE_INTERNAL_ENERGY = 4
    UNREPRESENTABLE_STATE = 8
    ZERO_MAGNETIC_FIELD = 16
    UNREPRESENTABLE_DIAGNOSTIC = 32


_INVALID_STATE = int(MHDStatus.NONFINITE_INPUT | MHDStatus.NONPOSITIVE_DENSITY |
                     MHDStatus.NONPOSITIVE_INTERNAL_ENERGY | MHDStatus.UNREPRESENTABLE_STATE)


class MHDStateError(ValueError):
    """First invalid node in selection order, including common valid support."""

    def __init__(self, leaf_id, cell_index, status):
        self.leaf_id = int(leaf_id)
        self.cell_index = tuple(map(int, cell_index))
        self.status = MHDStatus(int(status))
        super().__init__(f"invalid MHD state at leaf {self.leaf_id}, cell {self.cell_index}: "
                         f"{self.status.name}")


_OUTPUTS = {
    "density": (("density", "g cm^-3"),),
    "velocity": tuple(("v"+axis, "cm s^-1") for axis in "xyz"),
    "speed": (("speed", "cm s^-1"),),
    "internal_energy": (("internal_energy", "erg cm^-3"),),
    "pressure": (("pressure", "dyn cm^-2"),),
    "temperature": (("temperature", "K"),),
    "beta": (("beta", "1"),),
    "sound_speed": (("sound_speed", "cm s^-1"),),
    "alfven_speed": (("alfven_speed", "cm s^-1"),),
    "sonic_mach": (("sonic_mach", "1"),),
    "alfven_mach": (("alfven_mach", "1"),),
    "status": (("mhd_status", "1"),),
}
_DEFAULT_OUTPUTS = ("density", "velocity", "pressure", "temperature", "beta",
                    "sound_speed", "alfven_speed", "sonic_mach", "alfven_mach", "status")


_DIAGNOSTICS = ("sound_speed", "alfven_speed", "beta", "sonic_mach", "alfven_mach")


def _diagnostic_outputs(outputs):
    """Resolve the small fixed dependency set; status requests a complete check."""
    selected = set(_DIAGNOSTICS if "status" in outputs else outputs)
    if "sonic_mach" in selected:
        selected.add("sound_speed")
    if "alfven_mach" in selected:
        selected.add("alfven_speed")
    return tuple(name for name in _DIAGNOSTICS if name in selected)


def _select(fields, components, count):
    if components is None:
        raise ValueError("MHD components must be selected explicitly")
    selected,_ = _components(fields,components,count=count)
    return tuple(map(int, selected))


def _recover(rho_raw, momentum_raw, energy_raw, magnetic_raw, model, outputs, diagnostics):
    """One supported block; temporaries do not grow with the number of leaves."""
    status = np.zeros(rho_raw.shape, dtype=np.uint8)
    for value in (rho_raw, *momentum_raw, energy_raw, *magnetic_raw):
        status[~np.isfinite(value)] |= int(MHDStatus.NONFINITE_INPUT)
    units = model.units
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        rho = rho_raw * units.density_g_cm3
        momentum = tuple(value * units.momentum_g_cm2_s for value in momentum_raw)
        energy = energy_raw * units.energy_erg_cm3
        b = tuple(value * units.field_gauss for value in magnetic_raw)
        velocity = tuple(value / rho for value in momentum)
        speed = np.hypot(np.hypot(*velocity[:2]), velocity[2])
        bnorm = np.hypot(np.hypot(*b[:2]), b[2])
        b_over_sqrt_mu = bnorm / np.sqrt(4*np.pi)
        magnetic_pressure = (.5 * b_over_sqrt_mu) * b_over_sqrt_mu
        internal = energy
        if model.energy_kind == "total":
            kinetic = sum((.5 * m) * v for m, v in zip(momentum, velocity))
            internal = (energy - kinetic) - magnetic_pressure
            del kinetic
        pressure = (model.gamma - 1) * internal
        status[rho <= 0] |= int(MHDStatus.NONPOSITIVE_DENSITY)
        status[internal <= 0] |= int(MHDStatus.NONPOSITIVE_INTERNAL_ENERGY)
        finite = np.isfinite(rho) & np.isfinite(internal) & np.isfinite(pressure)
        for value in (*momentum, energy, *b, *velocity, speed, magnetic_pressure):
            finite &= np.isfinite(value)
        # Use the shared CGS H/He EOS; invalid nodes are never clipped or repaired.
        eligible = finite & (rho > 0) & (pressure > 0) & (status == 0)
        hydrogen_density = model.composition.number_density(
            rho[eligible], convention="amrvac-hydrogen")
        eligible[eligible] = np.isfinite(hydrogen_density) & (hydrogen_density > 0)
        temperature = np.full(rho.shape, np.nan)
        temperature[eligible] = model.composition.temperature(rho[eligible], pressure[eligible])
        representable = finite & np.isfinite(temperature) & (temperature > 0)
        # Do not obscure primary input/positivity errors with consequent NaNs.
        status[(status == 0) & ~representable] |= int(MHDStatus.UNREPRESENTABLE_STATE)
        valid = (status & _INVALID_STATE) == 0
        zero_b = valid & (bnorm == 0)
        status[zero_b] |= int(MHDStatus.ZERO_MAGNETIC_FIELD)
        # Unit conversions and state validation are shared by every request.
        # Release their scratch before allocating any optional diagnostics.
        del momentum, b, energy, hydrogen_density
        del finite, eligible, representable
        values = {"density": (rho,), "velocity": velocity, "speed": (speed,),
                  "internal_energy": (internal,), "pressure": (pressure,),
                  "temperature": (temperature,)}
        computed = {}
        if "sound_speed" in diagnostics or "alfven_speed" in diagnostics:
            sqrt_rho = np.sqrt(rho)
            if "sound_speed" in diagnostics:
                computed["sound_speed"] = np.sqrt(model.gamma) * (np.sqrt(pressure) / sqrt_rho)
            if "alfven_speed" in diagnostics:
                computed["alfven_speed"] = b_over_sqrt_mu / sqrt_rho
        if "beta" in diagnostics:
            beta = np.full(rho.shape, np.nan)
            np.divide(pressure, magnetic_pressure, out=beta, where=valid & ~zero_b)
            computed["beta"] = beta
        if "alfven_mach" in diagnostics:
            alfven_mach = np.full(rho.shape, np.nan)
            np.divide(speed, computed["alfven_speed"], out=alfven_mach, where=valid & ~zero_b)
            computed["alfven_mach"] = alfven_mach
        if "sonic_mach" in diagnostics:
            computed["sonic_mach"] = speed / computed["sound_speed"]
        # Evaluate dependencies before replacing nonfinite intermediates with
        # NaN, preserving the original formula ordering for extreme states.
        for name, diagnostic in computed.items():
            nonfinite = ~np.isfinite(diagnostic)
            bad = valid & nonfinite
            if name in ("beta", "alfven_mach"):
                bad &= ~zero_b
            status[bad] |= int(MHDStatus.UNREPRESENTABLE_DIAGNOSTIC)
            diagnostic[nonfinite] = np.nan
            values[name] = (diagnostic,)
    values = {key: values[key] for key in outputs if key != "status"}
    invalid = ~valid
    for components in values.values():
        for value in components:
            value[invalid] = np.nan
    values["status"] = (status,)
    return values, status


def mhd_fields(conserved, *, model, magnetic=None, density="rho",
               momentum=("m1", "m2", "m3"), energy="e",
               magnetic_components=("b1", "b2", "b3"), outputs=_DEFAULT_OUTPUTS,
               invalid="raise", memory_limit=None):
    """Recover selected MHD quantities as independently owned Fields.

    Parameters
    ----------
    conserved : Fields
        Conserved variables on complete leaves; prepare first if outputs will be
        interpolated.
    model : IdealMHD
        Explicit gamma, total/internal energy convention, fully ionized H/He composition and
        Gaussian CGS scales.
    magnetic : Fields, optional
        Separate magnetic group sharing Mesh identity and leaf coverage; slot order may
        differ. None selects B from conserved.
    density : str or int
        Density component.
    momentum : sequence
        Three classical momentum rho*v components.
    energy : str or int
        Energy component matching model.energy_kind.
    magnetic_components : sequence
        Three ordered magnetic components.
    outputs : str or sequence of str
        Groups: density, velocity (vx/vy/vz), speed, internal_energy, pressure,
        temperature, beta, sound_speed, alfven_speed, sonic_mach, alfven_mach, status.
        Diagnostics and dependencies are evaluated on demand; status requests all
        diagnostics.
    invalid : str
        raise reports the first invalid physical state as MHDStateError; nan marks
        invalid physical outputs. Neither applies positivity floors.
    memory_limit : int, optional
        Accounted-array budget in bytes for this call, not a process RSS limit.

    Returns
    -------
    Fields
        Independent selected CGS quantities and optional categorical status, in conserved
        leaf order with common valid halo. preparation_stats records
        evaluated_diagnostics and interior/all-node flag counts (not volumes); unchecked
        diagnostic counts are None, not zero.

    Notes
    -----
    - Every call checks the complete physical state, regardless of requested outputs.
    - Zero B is valid but leaves beta/Alfven Mach undefined; these and unrepresentable
      diagnostics are NaN with nonfatal flags.
    - Recovery is pointwise on supplied nodes, before any sampling. Select continuous
      outputs when interpolating; status columns are categorical.
    """
    if not isinstance(model, IdealMHD):
        raise TypeError("model must be an explicit IdealMHD configuration")
    if invalid not in ("raise", "nan"):
        raise ValueError("invalid must be raise or nan")
    conserved = require_fields(conserved)
    magnetic = conserved if magnetic is None else require_fields(magnetic)
    if (magnetic.mesh is not conserved.mesh or len(magnetic.leaf_ids) != len(conserved.leaf_ids) or
            np.any(magnetic.slot_of_leaf[conserved.leaf_ids] < 0)):
        raise ValueError("MHD inputs must share the same Mesh and leaf coverage")
    rho_id = _select(conserved, density, 1)[0]
    momentum_ids = _select(conserved, momentum, 3)
    energy_id = _select(conserved, energy, 1)[0]
    b_ids = _select(magnetic, magnetic_components, 3)
    c_ids = (rho_id, *momentum_ids, energy_id)
    if (len(set(c_ids)) != len(c_ids) or
            (magnetic is conserved and set(c_ids).intersection(b_ids))):
        raise ValueError("density, momentum, energy and magnetic selectors must be distinct")
    outputs = (outputs,) if isinstance(outputs, str) else tuple(outputs)
    if (not outputs or not all(isinstance(key, str) and key in _OUTPUTS for key in outputs) or
            len(set(outputs)) != len(outputs)):
        raise ValueError(f"outputs must be nonempty unique keys from {tuple(_OUTPUTS)}")
    diagnostics = _diagnostic_outputs(outputs)
    definitions = tuple(FieldDefinition(name, unit, "categorical-node" if key == "status" else
                                         "pointwise-MHD-recovery")
                        for key in outputs for name, unit in _OUTPUTS[key])
    halo = min(conserved.valid_halo, magnetic.valid_halo)
    block_shape = tuple(n + 2*halo for n in conserved.mesh.block_shape)
    shape = (len(conserved.leaf_ids), *block_shape, len(definitions))
    input_bytes = array_bytes(array for group in (conserved, magnetic)
                              for array in (group.values, group.leaf_ids, group.slot_of_leaf))
    required = (conserved.mesh.nbytes + input_bytes + 8*math.prod(shape) +
                (384+16*len(diagnostics))*math.prod(block_shape) + 8*conserved.mesh.leaf_count)
    admit(required, memory_limit, "MHD recovery")
    values = np.empty(shape, dtype=np.float64)
    c_box, b_box = (tuple(slice(group.storage_halo-halo, group.storage_halo+n+halo)
                          for n in group.mesh.block_shape) for group in (conserved, magnetic))
    interior = tuple(slice(halo, halo+n) for n in conserved.mesh.block_shape)
    counts = {region: {flag.name: 0 for flag in MHDStatus} for region in ("interior", "evaluated")}
    invalid_counts = {region: 0 for region in counts}
    c_values, b_values = conserved.values, magnetic.values
    for row, leaf in enumerate(conserved.leaf_ids):
        c_node = (conserved.slot_of_leaf[leaf], *c_box)
        b_node = (magnetic.slot_of_leaf[leaf], *b_box)
        recovered, status = _recover(c_values[(*c_node, rho_id)],
            tuple(c_values[(*c_node, i)] for i in momentum_ids), c_values[(*c_node, energy_id)],
            tuple(b_values[(*b_node, i)] for i in b_ids), model, outputs, diagnostics)
        bad = (status & _INVALID_STATE) != 0
        if invalid == "raise" and bad.any():
            index = np.unravel_index(np.argmax(bad), bad.shape)
            raise MHDStateError(leaf, tuple(i-halo for i in index), status[index])
        for region, flags in (("interior", status[interior]), ("evaluated", status)):
            invalid_counts[region] += int(np.count_nonzero(flags & _INVALID_STATE))
            for flag in MHDStatus:
                counts[region][flag.name] += int(np.count_nonzero(flags & int(flag)))
        column = 0
        for key in outputs:
            for component in recovered[key]:
                values[row, ..., column] = component
                column += 1
        del recovered, status, bad
    if not diagnostics:
        for region in counts:
            counts[region][MHDStatus.UNREPRESENTABLE_DIAGNOSTIC.name] = None
    stats = {"model": model, "invalid_policy": invalid, "outputs": outputs,
             "evaluated_diagnostics": diagnostics,
             "status_counts": counts, "invalid_state_counts": invalid_counts,
             "controlled_upper_bytes": required}
    return publish(conserved.mesh, values, conserved.selection, definitions, halo, halo,
                   f"mhd-{model.energy_kind}({conserved.scheme},{magnetic.scheme})",
                   (conserved.source, magnetic.source, model), stats)
