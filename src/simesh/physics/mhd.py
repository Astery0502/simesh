"""Explicit classical ideal-gas MHD recovery on completed native Fields.

Energy is a density per volume, selected as total (internal + kinetic +
magnetic) or internal. No file name, field label or metadata selects the EOS,
energy definition, composition or normalization. See docs/application-guide.md.
"""

from dataclasses import dataclass
from enum import IntFlag
import math

import numpy as np

from .._validation import admit, array_bytes
from ..diagnostics import MagneticUnits, _components
from ..fields import FieldDefinition, publish, require_fields
from .thermal import CoronalComposition


@dataclass(frozen=True, kw_only=True)
class MHDUnits:
    """SI multipliers per stored density, momentum density and energy density.

    ``magnetic`` supplies tesla per stored B, permeability and coordinate length.
    The length is retained for consumers; local recovery does not use it.
    Momentum normalization is independent: rho_unit * velocity_unit is a common
    choice, but is never inferred. All output quantities use SI, except kelvin
    temperature and dimensionless ratios/status.
    """

    density_kg_m3: float
    momentum_kg_m2_s: float
    energy_j_m3: float
    magnetic: MagneticUnits

    def __post_init__(self):
        if not isinstance(self.magnetic, MagneticUnits):
            raise TypeError("magnetic must be a MagneticUnits configuration")
        for name in ("density_kg_m3", "momentum_kg_m2_s", "energy_j_m3"):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0 or
                    not np.isfinite(value) or value <= 0):
                raise ValueError("MHD unit multipliers must be finite positive scalars")
            object.__setattr__(self, name, float(value))


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
    "density": (("density", "kg m^-3"),),
    "velocity": tuple(("v"+axis, "m s^-1") for axis in "xyz"),
    "speed": (("speed", "m s^-1"),),
    "internal_energy": (("internal_energy", "J m^-3"),),
    "pressure": (("pressure", "Pa"),),
    "temperature": (("temperature", "K"),),
    "beta": (("beta", "1"),),
    "sound_speed": (("sound_speed", "m s^-1"),),
    "alfven_speed": (("alfven_speed", "m s^-1"),),
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
        rho = rho_raw * units.density_kg_m3
        momentum = tuple(value * units.momentum_kg_m2_s for value in momentum_raw)
        energy = energy_raw * units.energy_j_m3
        b = tuple(value * units.magnetic.field_tesla for value in magnetic_raw)
        velocity = tuple(value / rho for value in momentum)
        speed = np.hypot(np.hypot(*velocity[:2]), velocity[2])
        bnorm = np.hypot(np.hypot(*b[:2]), b[2])
        b_over_sqrt_mu = bnorm / np.sqrt(units.magnetic.permeability_h_m)
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
        # Reuse the established H/He EOS, converting kg/m^3 -> g/cm^3 and Pa
        # -> erg/cm^3. Invalid nodes are excluded rather than clipped or repaired.
        rho_cgs, pressure_cgs = rho * 1.e-3, pressure * 10.
        eligible = (finite & (rho_cgs > 0) & np.isfinite(rho_cgs) &
                    (pressure_cgs > 0) & np.isfinite(pressure_cgs) & (status == 0))
        hydrogen_density = model.composition.number_density(
            rho_cgs[eligible], convention="amrvac-hydrogen")
        eligible[eligible] = np.isfinite(hydrogen_density) & (hydrogen_density > 0)
        temperature = np.full(rho.shape, np.nan)
        temperature[eligible] = model.composition.temperature(rho_cgs[eligible], pressure_cgs[eligible])
        representable = finite & np.isfinite(temperature) & (temperature > 0)
        # Do not obscure primary input/positivity errors with consequent NaNs.
        status[(status == 0) & ~representable] |= int(MHDStatus.UNREPRESENTABLE_STATE)
        valid = (status & _INVALID_STATE) == 0
        zero_b = valid & (bnorm == 0)
        status[zero_b] |= int(MHDStatus.ZERO_MAGNETIC_FIELD)
        # Unit conversions and state validation are shared by every request.
        # Release their scratch before allocating any optional diagnostics.
        del momentum, b, energy, rho_cgs, pressure_cgs, hydrogen_density
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

    Input selectors accept names or component indices. ``magnetic=None`` selects
    B from ``conserved``; a separate group must have the same Mesh and leaf set.
    Slot order may differ. Output follows conserved selection order, retaining
    the minimum valid halo and evaluating only that support. It neither fills
    halos nor changes coverage. Prepared-node recovery precedes any sampling.

    ``outputs="velocity"`` gives exactly vx, vy, vz for trace(). Other output
    keys are density, speed, internal_energy, pressure, temperature, beta,
    sound_speed, alfven_speed, sonic_mach, alfven_mach, status. Every call checks
    the complete physical state, even when requesting only velocity or status.

    ``invalid="raise"`` fails on the first invalid state; ``"nan"`` marks all
    its physical outputs NaN. No positivity floors are applied. Zero B is a
    valid state with beta/Alfven Mach undefined (NaN); unrepresentable diagnostic
    ratios/speeds are also NaN. Both conditions have nonfatal status bits.
    Include status to compute and locate every diagnostic flag. Otherwise only
    requested diagnostics and their speed dependencies are evaluated. Statistics
    record evaluated_diagnostics; an unchecked diagnostic counter is None, not
    zero. State flags are always counted over interiors and all evaluated nodes
    separately. These are node counts, not volumes. Status fields are categorical
    and must not be interpolated.

    The memory admission includes input arrays, output, mapping, mesh and a
    conservative fixed scratch allowance per supported block; no input arrays
    or batch lease are retained by the result.
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
