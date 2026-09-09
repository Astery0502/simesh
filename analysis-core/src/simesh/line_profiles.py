"""Sample completed fields on stored curves without tracing or resampling geometry."""

from dataclasses import dataclass

import numpy as np

from ._execution import worker_context
from ._validation import admit, array_bytes, workers_count
from .fields import _component_indices, require_fields
from .geometry import LineSet
from .operators.sampling import _sample
from .reductions import LengthUnits

__all__ = ["LineProfiles", "LineProfile", "sample_line_profiles"]


def _readonly(*arrays):
    for array in arrays:
        array.flags.writeable = False


@dataclass(frozen=True, eq=False)
class LineProfile:
    """One branch or an along-oriented display, with original packed point indices.

    Branch distances start at zero and increase away from the seed. A joined
    display uses negative/positive distances for the against/along branches.
    Both copies of the seed are retained. Termination always has both statuses.
    Arrays are read-only; branch arrays may share the parent result's backing.
    """

    seed_id: int
    positions: np.ndarray
    arclength: np.ndarray
    values: np.ndarray
    owners: np.ndarray
    valid: np.ndarray
    finite: np.ndarray
    boundary_adjusted: np.ndarray
    point_indices: np.ndarray
    directions: np.ndarray
    termination: np.ndarray
    definitions: tuple
    length_units: LengthUnits
    source_identity: object
    line_source_identity: object
    scheme: str
    boundary: str

    @property
    def usable(self):
        """Per-component coverage AND finiteness, without a physics validity claim."""
        return self.valid[:, None] & self.finite


@dataclass(frozen=True, eq=False)
class LineProfiles:
    """Owned profile arrays aligned exactly with a shared read-only LineSet.

    source_identity identifies sampled values; line_source_identity identifies
    the field used to trace the geometry. They need not be equal. Neither token
    retains Fields or a Source. Unit conversion affects arclength only.
    """

    lines: LineSet
    arclength: np.ndarray
    values: np.ndarray
    owners: np.ndarray
    valid: np.ndarray
    finite: np.ndarray
    boundary_adjusted: np.ndarray
    definitions: tuple
    component_indices: tuple
    source_identity: object
    scheme: str
    length_units: LengthUnits
    boundary: str

    @property
    def line_source_identity(self):
        return self.lines.source_identity

    @property
    def offsets(self):
        return self.lines.offsets

    @property
    def seed_ids(self):
        return self.lines.seeds.ids

    @property
    def termination(self):
        return self.lines.termination

    @property
    def usable(self):
        """Per-component coverage AND finiteness, independent of trace termination."""
        return self.valid[:, None] & self.finite

    @property
    def nbytes(self):
        return self.lines.nbytes + array_bytes((self.arclength, self.values,
            self.owners, self.valid, self.finite, self.boundary_adjusted))

    def _row(self, seed_id):
        # Avoid allocating a full seed mask before a display's memory admission.
        if isinstance(seed_id, (int, np.integer)) and not isinstance(seed_id, (bool, np.bool_)):
            for row, stored_id in enumerate(self.seed_ids):
                if stored_id == seed_id:
                    return row
        raise ValueError("select an existing seed ID")

    def branch(self, seed_id, direction, *, memory_limit=None):
        """Read-only seed-to-endpoint branch; even empty branches retain status."""
        row = self._row(seed_id)
        if isinstance(direction, (bool, np.bool_)) or direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        index = 2*row + int(direction == 1)
        start, stop = map(int, self.offsets[index:index+2])
        admit(self.nbytes + (stop-start)*9 + 16, memory_limit, "profile branch view")
        point_indices = np.arange(start, stop, dtype=np.int64)
        directions = np.full(stop-start, direction, dtype=np.int8)
        _readonly(point_indices, directions)
        return self._view(row, slice(start, stop), self.arclength[start:stop],
                          point_indices, directions)

    def line(self, seed_id, *, memory_limit=None):
        """Owned along-oriented display with signed distances and BOTH seed copies.

        No points are dropped. point_indices maps every displayed row back to
        packed storage; directions distinguishes the coincident seed samples.
        """
        row = self._row(seed_id)
        start, middle, stop = map(int, self.offsets[2*row:2*row+3])
        count = stop-start
        # Output copies, indices, and transient indexing/sign arrays coexist.
        admit(self.nbytes + count*(80+9*len(self.definitions)) + 16, memory_limit,
              "joined line profile")
        point_indices = np.empty(count, dtype=np.int64)
        negative = middle-start
        point_indices[:negative] = np.arange(middle-1, start-1, -1, dtype=np.int64)
        point_indices[negative:] = np.arange(middle, stop, dtype=np.int64)
        directions = np.ones(count, dtype=np.int8)
        directions[:negative] = -1
        distance = self.arclength[point_indices]
        distance[:negative] *= -1
        _readonly(point_indices, directions, distance)
        return self._view(row, point_indices, distance, point_indices, directions)

    def _view(self, row, index, distance, point_indices, directions):
        arrays = tuple(a[index] for a in (self.lines.positions, self.values,
            self.owners, self.valid, self.finite, self.boundary_adjusted))
        _readonly(*arrays)
        positions, values, owners, valid, finite, adjusted = arrays
        termination = self.termination[row].copy()
        _readonly(termination)
        return LineProfile(int(self.seed_ids[row]), positions, distance, values,
            owners, valid, finite, adjusted, point_indices, directions,
            termination, self.definitions, self.length_units,
            self.source_identity, self.line_source_identity, self.scheme, self.boundary)


def sample_line_profiles(fields, lines, components=None, *, point_batch=4096,
                         workers=1, length_units=None, boundary="interior",
                         memory_limit=None):
    """Sample selected components at every stored LineSet point, in bounded batches.

    Names and indices can be mixed; ordering and individual unit labels survive.
    Each branch arclength starts at zero and is the cumulative geometric length
    of its stored polyline, NOT the tracer's ODE length or a time coordinate.
    Optional LengthUnits explicitly scales these distances, not positions.

    boundary='interior' samples exact upper physical faces at the nearest
    representable interior coordinate, only for points in the closed domain.
    boundary='native' retains the sampler's half-open domain. Original positions
    are never changed; boundary_adjusted records each adjusted point.

    Geometry can originate from another field or caller-supplied curves encoded
    as LineSet. The caller establishes compatible spatial coordinates and times;
    value identities are recorded separately, never required to match.
    """
    fields = require_fields(fields, halo=1)
    if not isinstance(lines, LineSet):
        raise TypeError("lines must be a LineSet")
    workers_count(workers)
    if type(point_batch) is not int or point_batch < 1:
        raise ValueError("point_batch must be a positive integer")
    if boundary not in ("interior", "native"):
        raise ValueError("boundary must be interior or native")
    if length_units is None:
        length_units = LengthUnits(1., "coordinate-length")
    if not isinstance(length_units, LengthUnits):
        raise TypeError("length_units must be LengthUnits or None")
    selected = _component_indices(fields, components)
    count, width = len(lines.positions), len(selected)
    # Retained: distances, selected values, owners, coverage, component finite
    # masks, adjustment mask. Scratch includes ALL native sampled components,
    # bounded coordinate copies, geometry differences and mask/index temporaries.
    retained = count*(18+9*width)
    scratch = min(point_batch, count)*(128+16*len(fields.fields)+16*width)
    admit(fields.mesh.nbytes + fields.nbytes + lines.nbytes + retained + scratch,
          memory_limit, "line profiles")
    distance = np.empty(count, dtype=float)
    values = np.empty((count, width), dtype=float)
    owners = np.empty(count, dtype=np.int64)
    valid = np.empty(count, dtype=bool)
    finite = np.empty((count, width), dtype=bool)
    adjusted = np.zeros(count, dtype=bool)

    for branch in range(len(lines.offsets)-1):
        start, stop = map(int, lines.offsets[branch:branch+2])
        if start == stop:
            continue
        if not np.array_equal(lines.positions[start], lines.seeds.positions[branch//2]):
            raise ValueError("each nonempty branch must start at its seed position")
        distance[start] = 0.
        with np.errstate(over="ignore", invalid="ignore"):
            for first in range(start+1, stop, point_batch):
                last = min(first+point_batch, stop)
                delta = lines.positions[first:last] - lines.positions[first-1:last-1]
                distance[first:last] = np.hypot(np.hypot(delta[:, 0], delta[:, 1]), delta[:, 2])
            # One cumulative pass per branch keeps arithmetic independent of batch size.
            np.cumsum(distance[start:stop], out=distance[start:stop])
            distance[start:stop] *= length_units.scale
        for first in range(start, stop, point_batch):
            if not np.isfinite(distance[first:min(first+point_batch, stop)]).all():
                raise ValueError("polyline arclength is not representable in the requested units")

    lower, upper = fields.mesh.lower, fields.mesh.upper
    with worker_context(workers) as executor:
        for first in range(0, count, point_batch):
            last = min(first+point_batch, count)
            points = lines.positions[first:last].copy()
            if boundary == "interior":
                inside = ((points >= lower) & (points <= upper)).all(axis=1)
                for axis in range(3):
                    face = inside & (points[:, axis] == upper[axis])
                    points[face, axis] = np.nextafter(upper[axis], lower[axis])
                    adjusted[first:last] |= face
            sampled, sample_owners, sample_valid = _sample(fields, points, workers, executor)
            for output, component in enumerate(selected):
                values[first:last, output] = sampled[:, component]
            owners[first:last] = sample_owners
            valid[first:last] = sample_valid
            np.isfinite(values[first:last], out=finite[first:last])
            # Release the previous native batch before allocating the next one.
            del sampled, sample_owners, sample_valid, points

    _readonly(distance, values, owners, valid, finite, adjusted)
    return LineProfiles(lines, distance, values, owners, valid, finite, adjusted,
        tuple(fields.fields[i] for i in selected), selected, fields.value_identity,
        fields.scheme, length_units, boundary)
