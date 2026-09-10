"""Quantitative reductions of native leaf interiors, without halo preparation.

All integrals use a piecewise-constant representation of the supplied interior
values. Each operation documents its coverage, weight and surface-side semantics.
"""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from ._validation import frozen_array
from .fields import FieldDefinition, require_fields, _field_index
from .mesh import Selection, _axis_candidates, _cell_edges, _cell_index


REPRESENTATION = "piecewise-constant leaf interiors"


@dataclass(frozen=True)
class LengthUnits:
    """Explicit isotropic length conversion: output lengths = coordinates * scale."""

    scale: float
    unit: str

    def __post_init__(self):
        if (isinstance(self.scale, (bool, np.bool_)) or not isinstance(self.scale, Real)
                or not math.isfinite(self.scale) or self.scale <= 0):
            raise ValueError("length scale must be finite and positive")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("length unit must be a nonempty label")


@dataclass(frozen=True)
class AxisAlignedSurface:
    """Rectangle at coordinate on axis, with bounds on the other axes in XYZ order.

    normal is +1 or -1. side chooses the positive/negative coordinate-side
    cell at internal interfaces; domain faces always use the interior cell.
    Neither the side nor the selected component changes when normal is reversed.
    """

    axis: int | str
    coordinate: float
    bounds: np.ndarray
    normal: int = 1
    side: str = "positive"

    def __post_init__(self):
        axis = "xyz".index(self.axis) if isinstance(self.axis, str) and self.axis in ("x", "y", "z") else self.axis
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)) or axis not in (0, 1, 2):
            raise ValueError("axis must be x, y, z or 0, 1, 2")
        if not isinstance(self.coordinate, Real) or not math.isfinite(self.coordinate):
            raise ValueError("surface coordinate must be finite")
        if isinstance(self.normal, (bool, np.bool_)) or self.normal not in (-1, 1):
            raise ValueError("normal must be +1 or -1")
        if self.side not in ("positive", "negative"):
            raise ValueError("side must be positive or negative")
        object.__setattr__(self, "axis", int(axis))
        object.__setattr__(self, "bounds", frozen_array(_box(self.bounds, 2), float))


@dataclass(frozen=True)
class Coverage:
    """Geometric coverage, independent of statistical weight magnitude.

    Attributes
    ----------
    requested_measure, domain_measure, available_measure, valid_measure : float
        Requested, in-domain, supplied and finite-contributing volume or area.
    units : str
        Unit of the reported geometric measures.
    cell_count, valid_cell_count : int
        Intersecting supplied cells and finite contributing cells.
    complete : bool
        Whether the whole requested measure contributed valid values.
    missing, nonfinite : str
        Rejection or omission policies used for this result.
    """

    requested_measure: float
    domain_measure: float
    available_measure: float
    valid_measure: float
    units: str
    cell_count: int
    valid_cell_count: int
    complete: bool
    missing: str
    nonfinite: str

    @property
    def outside_measure(self):
        """Requested measure outside the original physical domain."""
        return max(0., self.requested_measure - self.domain_measure)

    @property
    def missing_measure(self):
        """In-domain requested measure absent from supplied field coverage."""
        return max(0., self.domain_measure - self.available_measure)

    @property
    def invalid_measure(self):
        """Available requested measure excluded by nonfinite input values."""
        return max(0., self.available_measure - self.valid_measure)

    @property
    def fraction(self):
        """Valid measure divided by requested measure; zero for a zero requested measure."""
        return self.valid_measure / self.requested_measure if self.requested_measure else 0.


@dataclass(frozen=True)
class ScalarResult:
    """Native-cell integral or mean with physical coverage.

    Attributes
    ----------
    value, units : object
        Scalar value and resulting unit label.
    coverage : Coverage
        Geometric measures, omission policies and completion.
    field : FieldDefinition
        Analyzed component.
    weight_sum, weight_units, weight_field, weight_mode : object
        Weight normalization and interpretation, when applicable.
    surface : AxisAlignedSurface, optional
        Oriented rectangle for a surface flux; coordinates retain the original mesh scale.
    representation : str
        Piecewise-constant leaf-interior interpretation. save_result preserves this
        contract together with coverage, field definitions and weight metadata.
    """
    value: float
    units: str
    coverage: Coverage
    field: FieldDefinition
    weight_sum: float | None = None
    weight_units: str | None = None
    weight_field: FieldDefinition | None = None
    weight_mode: str | None = None
    surface: AxisAlignedSurface | None = None
    representation: str = REPRESENTATION


@dataclass(frozen=True)
class Extremum:
    """Position is the clipped cell's centroid, in original mesh coordinates."""

    value: float
    position: tuple[float, float, float]
    leaf_id: int
    cell_index: tuple[int, int, int]


@dataclass(frozen=True)
class ExtremaResult:
    """Native-cell extrema with coverage and component interpretation.

    Attributes
    ----------
    minimum, maximum : Extremum
        Value/location records. The producer rejects a region without finite covered cells.
    coverage : Coverage
        Geometric completion and omission policies.
    field : FieldDefinition
        Analyzed component.
    units, position_units : str
        Value units and original coordinate-length units for extrema locations.
    representation : str
        Piecewise-constant leaf-interior interpretation; retained by save_result.
    """
    minimum: Extremum
    maximum: Extremum
    units: str
    coverage: Coverage
    field: FieldDefinition
    position_units: str = "coordinate-length"
    representation: str = REPRESENTATION


@dataclass(frozen=True)
class HistogramResult:
    """Weighted native-cell histogram retaining tails and coverage.

    Attributes
    ----------
    edges, bin_weights : ndarray
        Bin boundaries and accumulated bin weights.
    underflow, overflow, total_weight : float
        Below-range, above-range and all included weights.
    value_units, weight_units : str
        Units of sample values and accumulated weights.
    coverage : Coverage
        Geometric completion and omission policies.
    field, weight_field, weight_mode : object
        Component and weight interpretation.
    representation : str
        Piecewise-constant leaf-interior interpretation; save_result retains the
        complete histogram, including underflow and overflow contributions.
    """
    edges: np.ndarray
    bin_weights: np.ndarray
    underflow: float
    overflow: float
    total_weight: float
    value_units: str
    weight_units: str
    coverage: Coverage
    field: FieldDefinition
    weight_field: FieldDefinition | None
    weight_mode: str
    representation: str = REPRESENTATION

    def __post_init__(self):
        object.__setattr__(self, "edges", frozen_array(self.edges, float))
        object.__setattr__(self, "bin_weights", frozen_array(self.bin_weights, float))


def _box(bounds, dimension=3):
    box = np.asarray(bounds, dtype=float)
    if box.shape != (2, dimension) or not np.isfinite(box).all() or np.any(box[0] >= box[1]):
        raise ValueError(f"bounds must be finite strictly ordered (2, {dimension}) vectors")
    return box


def _component(fields, component):
    require_fields(fields)
    if isinstance(component, str):
        return _field_index(fields.fields, component)
    if (isinstance(component, (bool, np.bool_)) or not isinstance(component, (int, np.integer))
            or not 0 <= component < len(fields.fields)):
        raise ValueError("component must be a unique field name or valid integer index")
    return int(component)


def _measure_units(units, dimension):
    if units is None:
        return 1., f"coordinate-length^{dimension}"
    if not isinstance(units, LengthUnits):
        raise TypeError("units must be LengthUnits or None for coordinate measures")
    try:
        factor = float(units.scale) ** dimension
    except OverflowError:
        factor = math.inf
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("length scale gives an unrepresentable measure conversion")
    return factor, f"{units.unit}^{dimension}"


def _product(a, b):
    return f"({a}) * ({b})"


def _sum(values):
    with np.errstate(over="raise", invalid="raise"):
        result = float(np.sum(values, dtype=np.float64))
    if not math.isfinite(result):
        raise FloatingPointError("reduction sum is not representable as float64")
    return result


def _weighted_sum(values, measure):
    with np.errstate(over="raise", invalid="raise"):
        return _sum(values * measure)


def _total(values):
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise FloatingPointError("reduction total is not representable as float64") from exc
    if not math.isfinite(result):
        raise FloatingPointError("reduction total is not representable as float64")
    return result


class _Domain:
    """One requested rectangle/leaf union and its per-leaf cell intersections."""

    def __init__(self, fields, region, surface):
        self.mesh, self.surface = fields.mesh, surface
        mesh = self.mesh
        self.allowed = None
        if surface is None:
            selection = fields.selection if region is None else region
            if isinstance(selection, Selection):
                if selection.mesh is not mesh:
                    raise ValueError("selection belongs to a different mesh")
                self.allowed = np.zeros(mesh.leaf_count, dtype=bool)
                self.allowed[selection.leaf_ids] = True
                box = selection.requested_bounds
            else:
                box = _box(selection)
            if box is None:
                self.box = np.array([mesh.lower, mesh.upper])
                self.ids = np.sort(selection.leaf_ids)
                self.requested = self.domain = _sum(np.prod(
                    mesh.bounds[self.ids, 1] - mesh.bounds[self.ids, 0], axis=1))
                self.outside = False
                return
            axes = np.arange(3)
            candidates = np.ones(mesh.leaf_count, dtype=bool)
        else:
            if not isinstance(surface, AxisAlignedSurface):
                raise TypeError("surface must be an AxisAlignedSurface")
            axis, coordinate = surface.axis, surface.coordinate
            axes = np.array([i for i in range(3) if i != axis])
            box = surface.bounds
            candidates, self.side = _axis_candidates(mesh, axis, coordinate, surface.side)
        self.box = box
        lower, upper = mesh.lower[axes], mesh.upper[axes]
        width = np.maximum(0., np.minimum(box[1], upper) - np.maximum(box[0], lower))
        self.requested = float(np.prod(box[1] - box[0]))
        self.domain = float(np.prod(width))
        self.outside = bool(np.any(box[0] < lower) or np.any(box[1] > upper))
        if surface is not None and not mesh.lower[surface.axis] <= surface.coordinate <= mesh.upper[surface.axis]:
            self.domain, self.outside = 0., True
        overlap = np.all((mesh.bounds[:, 1, axes] > box[0]) & (mesh.bounds[:, 0, axes] < box[1]), axis=1)
        self.ids = np.flatnonzero(candidates & overlap)
        if not math.isfinite(self.requested) or self.requested <= 0 or not math.isfinite(self.domain):
            raise ValueError("requested geometric measure is not representable")

    def piece(self, leaf):
        mesh = self.mesh
        slices, lengths, centers = [], [], []
        transverse = 0
        for axis in range(3):
            edges = _cell_edges(mesh, leaf, axis)
            if self.surface is not None and axis == self.surface.axis:
                index = _cell_index(edges, self.surface.coordinate, self.side)
                slices.append(slice(index, index + 1))
                lengths.append(np.ones(1))
                centers.append(np.array([self.surface.coordinate]))
            else:
                lower = np.maximum(edges[:-1], self.box[0, transverse])
                upper = np.minimum(edges[1:], self.box[1, transverse])
                active = np.flatnonzero(upper > lower)
                if not len(active):
                    return None
                window = slice(int(active[0]), int(active[-1]) + 1)
                slices.append(window)
                lengths.append((upper - lower)[window])
                centers.append(lower[window] + .5 * (upper - lower)[window])
                transverse += 1
        measure = lengths[0][:, None, None] * lengths[1][None, :, None] * lengths[2][None, None, :]
        if not np.isfinite(measure).all() or np.any(measure <= 0):
            raise ValueError("cell intersection measure is not representable")
        return tuple(slices), measure, centers


class _Reduction:
    def __init__(self, fields, component, region, units, missing, nonfinite,
                 weights=None, weight_component=0, weight_mode="density", surface=None):
        self.component = _component(fields, component)
        self.fields, self.definition = fields, fields.fields[self.component]
        if missing not in ("raise", "omit") or nonfinite not in ("raise", "omit"):
            raise ValueError("missing and nonfinite must each be raise or omit")
        self.missing, self.nonfinite = missing, nonfinite
        self.factor, self.measure_units = _measure_units(units, 2 if surface is not None else 3)
        self.domain = _Domain(fields, region, surface)
        if missing == "raise" and self.domain.outside:
            raise ValueError("requested geometry extends outside the mesh domain")
        self.weights, self.weight_definition = weights, None
        if weight_mode not in ("density", "cell-total"):
            raise ValueError("weight_mode must be density or cell-total")
        if weights is None:
            if weight_mode != "density" or weight_component != 0:
                raise ValueError("weight options require explicit weights Fields")
            self.weight_mode, self.weight_units = "volume", self.measure_units
        else:
            self.weight_component = _component(weights, weight_component)
            if weights.mesh is not fields.mesh:
                raise ValueError("weights must share the same Mesh object")
            self.weight_definition = weights.fields[self.weight_component]
            self.weight_mode = weight_mode
            self.weight_units = (_product(self.weight_definition.units, self.measure_units)
                                 if weight_mode == "density" else self.weight_definition.units)
        self.available, self.valid = [], []
        self.count = self.valid_count = 0
        self.incomplete = self.domain.outside

    @staticmethod
    def _values(fields, component, leaf, window):
        h = fields.storage_halo
        box = tuple(slice(s.start + h, s.stop + h) for s in window)
        return fields.values[(fields.slot_of_leaf[leaf], *box, component)]

    def pieces(self):
        for leaf in self.domain.ids:
            geometry = self.domain.piece(leaf)
            if geometry is None:
                continue
            window, native_measure, centers = geometry
            supplied = (self.fields.slot_of_leaf[leaf] >= 0 and
                        (self.domain.allowed is None or self.domain.allowed[leaf]) and
                        (self.weights is None or self.weights.slot_of_leaf[leaf] >= 0))
            if not supplied:
                self.incomplete = True
                if self.missing == "raise":
                    raise ValueError(f"missing field coverage on leaf {leaf}")
                continue
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                measure = native_measure * self.factor
                if np.any(measure <= 0):
                    raise ValueError("converted cell measure is not representable")
                values = self._values(self.fields, self.component, leaf, window)
                finite = np.isfinite(values)
                weight = None
                if self.weights is not None:
                    weight = self._values(self.weights, self.weight_component, leaf, window)
                    weight_finite = np.isfinite(weight)
                    if np.any(weight_finite & (weight < 0)):
                        raise ValueError("explicit weights must be nonnegative")
                    finite = finite & weight_finite
                self.count += values.size
                self.valid_count += int(finite.sum())
                self.available.append(_sum(measure))
                effective = measure[finite]
                self.valid.append(_sum(effective))
                if not finite.all():
                    self.incomplete = True
                    if self.nonfinite == "raise":
                        raise ValueError(f"nonfinite field or weight on leaf {leaf}")
                if weight is not None:
                    if self.weight_mode == "cell-total":
                        full_measure = float(np.prod(self.fields.mesh.spacing[leaf]))
                        if not math.isfinite(full_measure) or full_measure <= 0:
                            raise ValueError("full cell measure is not representable")
                        effective = native_measure[finite] / full_measure
                    effective = effective * weight[finite]
            yield int(leaf), window, centers, finite, values[finite], effective

    def coverage(self):
        requested = self.domain.requested * self.factor
        domain = self.domain.domain * self.factor
        if not math.isfinite(requested) or not math.isfinite(domain):
            raise FloatingPointError("converted coverage measure is not representable")
        available = min(domain, _total(self.available))
        valid = min(available, _total(self.valid))
        return Coverage(requested, domain, available, valid, self.measure_units,
                        self.count, self.valid_count, not self.incomplete,
                        self.missing, self.nonfinite)


def volume_integral(fields, component=0, *, region=None, units=None,
                    missing="raise", nonfinite="raise"):
    """Sum value times the exact cell/box intersection volume.

    Parameters
    ----------
    fields : Fields
        Native cell interiors; halos are not used. Values have a piecewise-constant
        representation.
    component : str or int
        Name or local index of the scalar component.
    region : Selection or array-like, optional
        Requested physical (2, 3) box or Selection; contributions use cell-overlap measures.
    units : LengthUnits, optional
        Isotropic physical length conversion; omission retains coordinate units.
    missing : str
        raise rejects missing coverage; omit retains only covered contributions.
    nonfinite : str
        raise rejects nonfinite values; omit excludes them from the result.

    Returns
    -------
    ScalarResult
        Values and units with explicit geometric coverage and omission policies.
    """
    reduction = _Reduction(fields, component, region, units, missing, nonfinite)
    totals = [_weighted_sum(values, measure) for *_, values, measure in reduction.pieces()]
    return ScalarResult(_total(totals), _product(reduction.definition.units, reduction.measure_units),
                        reduction.coverage(), reduction.definition)


def weighted_mean(fields, component=0, *, weights=None, weight_component=0,
                  weight_mode="density", region=None, units=None,
                  missing="raise", nonfinite="raise"):
    """Volume mean, or mean with explicit nonnegative weights on the same Mesh.

    Parameters
    ----------
    fields : Fields
        Native cell interiors; halos are not used. Values have a piecewise-constant
        representation.
    component : str or int
        Name or local index of the scalar component.
    weights : Fields, optional
        Finite nonnegative weights on the same Mesh and coverage; omission uses volume
        weights.
    weight_component : str or int
        Weight component name or local index.
    weight_mode : str
        density multiplies weights by volume; cell-total apportions cell weights by overlap.
    region : Selection or array-like, optional
        Requested physical (2, 3) box or Selection; contributions use cell-overlap measures.
    units : LengthUnits, optional
        Isotropic physical length conversion; omission retains coordinate units.
    missing : str
        raise rejects missing coverage; omit retains only covered contributions.
    nonfinite : str
        raise rejects nonfinite values; omit excludes them from the result.

    Returns
    -------
    ScalarResult
        Values and units with explicit geometric coverage and omission policies.
    """
    reduction = _Reduction(fields, component, region, units, missing, nonfinite,
                           weights, weight_component, weight_mode)
    numerators, denominators = [], []
    for *_, values, weight in reduction.pieces():
        numerators.append(_weighted_sum(values, weight))
        denominators.append(_sum(weight))
    total = _total(denominators)
    if total <= 0:
        raise ValueError("weighted mean requires positive total weight")
    value = _total(numerators) / total
    if not math.isfinite(value):
        raise FloatingPointError("weighted mean is not representable as float64")
    return ScalarResult(value, reduction.definition.units, reduction.coverage(), reduction.definition,
                        total, reduction.weight_units, reduction.weight_definition, reduction.weight_mode)


def extrema(fields, component=0, *, region=None, units=None,
            missing="raise", nonfinite="raise"):
    """Finite extrema and clipped-cell positions; ties use leaf ID then XYZ index.

    Parameters
    ----------
    fields : Fields
        Native cell interiors; halos are not used. Values have a piecewise-constant
        representation.
    component : str or int
        Name or local index of the scalar component.
    region : Selection or array-like, optional
        Requested physical (2, 3) box or Selection; contributions use cell-overlap measures.
    units : LengthUnits, optional
        Isotropic physical length conversion; omission retains coordinate units.
    missing : str
        raise rejects missing coverage; omit retains only covered contributions.
    nonfinite : str
        raise rejects nonfinite values; omit excludes them from the result.

    Returns
    -------
    ExtremaResult
        Values and units with explicit geometric coverage and omission policies.

    Raises
    ------
    ValueError
        No finite covered cell contributes to the requested region.
    """
    reduction = _Reduction(fields, component, region, units, missing, nonfinite)
    minimum = maximum = None
    for leaf, window, centers, finite, values, _ in reduction.pieces():
        if not len(values):
            continue
        flat_indices = np.flatnonzero(finite)
        for low in (True, False):
            offset = int(np.argmin(values) if low else np.argmax(values))
            value = float(values[offset])
            previous = minimum if low else maximum
            if previous is not None and not (value < previous.value if low else value > previous.value):
                continue
            index = np.unravel_index(flat_indices[offset], finite.shape)
            item = Extremum(value, tuple(float(centers[a][i]) for a, i in enumerate(index)), leaf,
                            tuple(int(i + window[a].start) for a, i in enumerate(index)))
            if low:
                minimum = item
            else:
                maximum = item
    if minimum is None:
        raise ValueError("extrema require at least one finite covered cell")
    return ExtremaResult(minimum, maximum, reduction.definition.units,
                         reduction.coverage(), reduction.definition)


def histogram(fields, edges, component=0, *, weights=None, weight_component=0,
              weight_mode="density", region=None, units=None,
              missing="raise", nonfinite="raise"):
    """Weighted histogram with explicit finite edges; the last bin includes its right edge.

    Parameters
    ----------
    fields : Fields
        Native cell interiors; halos are not used. Values have a piecewise-constant
        representation.
    edges : array-like
        Strictly increasing finite histogram edges.
    component : str or int
        Name or local index of the scalar component.
    weights : Fields, optional
        Weights on the same Mesh and leaf coverage; omission uses volume weights.
    weight_component : str or int
        Weight component name or local index.
    weight_mode : str
        density multiplies weights by volume; cell-total apportions cell weights by overlap.
    region : Selection or array-like, optional
        Requested physical (2, 3) box or Selection; contributions use cell-overlap measures.
    units : LengthUnits, optional
        Isotropic physical length conversion; omission retains coordinate units.
    missing : str
        raise rejects missing coverage; omit retains only covered contributions.
    nonfinite : str
        raise rejects nonfinite values; omit excludes them from the result.

    Returns
    -------
    HistogramResult
        Values and units with explicit geometric coverage and omission policies.
    """
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or np.any(edges[1:] <= edges[:-1])):
        raise ValueError("edges must be a finite strictly increasing vector")
    reduction = _Reduction(fields, component, region, units, missing, nonfinite,
                           weights, weight_component, weight_mode)
    bins = np.zeros(len(edges) - 1)
    below, above, totals = [], [], []
    for *_, values, weight in reduction.pieces():
        low, high = values < edges[0], values > edges[-1]
        inside = ~(low | high)
        indices = np.minimum(np.searchsorted(edges, values[inside], side="right") - 1, len(bins) - 1)
        # Accumulate directly into bins; subtracting a large cumulative sum can
        # erase small weights in later bins.
        with np.errstate(over="raise", invalid="raise"):
            bins += np.bincount(indices, weights=weight[inside], minlength=len(bins))
        below.append(_sum(weight[low]))
        above.append(_sum(weight[high]))
        totals.append(_sum(weight))
    if not np.isfinite(bins).all():
        raise FloatingPointError("histogram weights are not representable as float64")
    return HistogramResult(edges, bins, _total(below), _total(above), _total(totals),
                           reduction.definition.units, reduction.weight_units, reduction.coverage(),
                           reduction.definition, reduction.weight_definition, reduction.weight_mode)


def surface_flux(fields, surface, component=0, *, units=None,
                 missing="raise", nonfinite="raise"):
    """Integrate the explicitly selected coordinate-normal component over a rectangle.

    Parameters
    ----------
    fields : Fields
        Native cell interiors; halos are not used. Values have a piecewise-constant
        representation.
    surface : AxisAlignedSurface
        Oriented rectangle with its explicit adjacent-cell choice; component must be the
        field's normal component.
    component : str or int
        Name or local index of the scalar component.
    units : LengthUnits, optional
        Isotropic physical length conversion; omission retains coordinate units.
    missing : str
        raise rejects missing coverage; omit retains only covered contributions.
    nonfinite : str
        raise rejects nonfinite values; omit excludes them from the result.

    Returns
    -------
    ScalarResult
        Values and units with explicit geometric coverage and omission policies.

    Notes
    -----
    Cell-centered one-sided flux is not CT face flux and does not guarantee coarse/fine flux continuity.
    """
    if not isinstance(surface, AxisAlignedSurface):
        raise TypeError("surface must be an AxisAlignedSurface")
    reduction = _Reduction(fields, component, None, units, missing, nonfinite, surface=surface)
    totals = [_weighted_sum(values, area) for *_, values, area in reduction.pieces()]
    return ScalarResult(surface.normal * _total(totals),
                        _product(reduction.definition.units, reduction.measure_units),
                        reduction.coverage(), reduction.definition, surface=surface)
