# Native AMR integrals and statistics

`simesh.reductions` consumes native `Fields`, including the zero-halo result of
`read_fields`. It provides volume integrals, volume or explicitly weighted means,
extrema with positions, weighted histograms, and flux through an axis-aligned
rectangle. Functions and the `LengthUnits`/`AxisAlignedSurface` inputs are also
available from top-level `simesh`; result types remain in this module.

Reduction result dataclasses are not registered with `save_result`. Save their
values, units and coverage explicitly as JSON/NumPy data or as caller metadata
on a supported application result. The [output guide](capabilities-and-outputs.md)
and [combined example](quantitative-workflow.md) describe this delivery path.

Only complete leaf interiors contribute. Ghost storage and refined parent nodes
are excluded. Each leaf uses its own cell geometry, and values are accessed through
`slot_of_leaf`; reversed selections, reordered physical slots and unused storage
slots do not change the result. Halo preparation is unnecessary. Prepared fields
are also accepted, using only their interiors.

## A complete workflow

This synthetic mesh has eight fine leaves covering the left half and one coarse
leaf covering the right half. The input density is already in kg/m³, temperature
in K and the z component of the magnetic field in T. The example explicitly sets
one mesh coordinate unit to 2 m.

```python
import numpy as np
import simesh as sm
from simesh.reductions import (
    AxisAlignedSurface, LengthUnits, extrema, histogram,
    surface_flux, volume_integral, weighted_mean,
)

mesh = sm.mesh_from_forest(
    (2, 1, 1), np.array([False] + [True] * 9),
    lower=(0., 0., 0.), upper=(2., 1., 1.), block_shape=(4, 4, 4),
)
values = np.empty((mesh.leaf_count, 3, *mesh.block_shape))
for leaf in range(mesh.leaf_count):
    xyz = (mesh.bounds[leaf, 0, :, None, None, None]
           + (np.indices(mesh.block_shape) + .5)
           * mesh.spacing[leaf, :, None, None, None])
    x = xyz[0]
    values[leaf] = [np.where(x < 1., 2., 4.), 100. + 10.*x,
                    np.full_like(x, 3.)]
definitions = [sm.FieldDefinition("rho", "kg m^-3"),
               sm.FieldDefinition("temperature", "K"),
               sm.FieldDefinition("bz", "T")]
with sm.source_from_arrays(mesh, values, definitions) as source:
    fields = sm.read_fields(source, leaf_ids=np.arange(mesh.leaf_count)[::-1])

units = LengthUnits(2., "m")
mass = volume_integral(fields, "rho", units=units)
temperature = weighted_mean(fields, "temperature", units=units)
mass_weighted = weighted_mean(
    fields, "temperature", weights=fields, weight_component="rho",
    units=units,
)
distribution = histogram(
    fields, [100., 110., 120.], "temperature",
    weights=fields, weight_component="rho", units=units,
)
limits = extrema(fields, "temperature")
rectangle = AxisAlignedSurface("z", .5, [[0., 0.], [2., 1.]])
flux = surface_flux(fields, rectangle, "bz", units=units)

assert fields.valid_halo == 0
assert mass.coverage.complete and mass.value == 48.  # kg
assert temperature.value == 110.                    # K
assert np.isclose(mass_weighted.value, 100. + 10.*7/6)
np.testing.assert_allclose(distribution.bin_weights, [16., 32.])  # kg
assert limits.minimum.value == 100.625
assert flux.value == 24.                            # T m^2
```

For a file, replace the source construction with:

```python
with sm.open_amrvac("snapshot.dat") as source:
    fields = sm.read_fields(
        source, ("rho", "b3"),
        region=[[.13, .11, .17], [1.7, .89, .92]],
    )
regional_integral = volume_integral(fields, "rho")
```

The second example returns stored-density times coordinate-volume. It does not
establish a physical density or length conversion for that snapshot. The source
may be closed before performing these reductions on detached fields.

## Functions and results

All component selectors accept a unique field name or integer component index.
Missing or ambiguous names raise. Every function accepts `units`, `missing` and
`nonfinite`; volume functions also accept `region`.

| Function | Result |
| --- | --- |
| `volume_integral(fields, component=0, ...)` | `ScalarResult.value = Σ fᵢ ΔVᵢ` |
| `weighted_mean(fields, component=0, ...)` | `ScalarResult.value = Σ fᵢ Wᵢ / Σ Wᵢ`, with `weight_sum` and `weight_units` |
| `extrema(fields, component=0, ...)` | `ExtremaResult.minimum` and `.maximum`, each with value, position, leaf ID and interior cell index |
| `histogram(fields, edges, component=0, ...)` | `HistogramResult.bin_weights`, edges, underflow, overflow and total weight |
| `surface_flux(fields, surface, component=0, ...)` | `ScalarResult.value = normal × Σ fᵢ ΔAᵢ`, retaining the surface specification |

Results own their metadata and arrays and retain no input field storage. They can
outlive a borrowed batch, provided the reduction completes while the borrow is
active. Expired `Fields` are rejected. Histogram arrays are read-only.

## Representation and rectangular regions

These operations integrate the **piecewise-constant representation of supplied
cell interiors**. For primary `cell-average` fields, the full-cell contribution is
the stored cell average times its cell volume. A rectangular region weights each
cell by its actual intersection volume, including partial edge and corner cells.
This is an integral of the discrete AMR representation without uniform resampling.

For a continuous linear field whose exact cell averages were stored, integration
over whole cells reproduces its analytic integral up to floating-point error.
A region cutting a cell uses that cell's constant average on the retained part;
it generally differs from integrating the continuous linear function over that
part. No subcell variation is inferred. Ghost preparation does not switch the
reduction to an interpolated or limited reconstruction.

Pointwise-derived and derivative fields are permitted. Their definition and
interpretation are returned as `result.field`. Integrating squared cell averages,
for example, is not the same as integrating the square of a reconstructed field.
The `representation` attribute identifies this choice. These functions make no
claim that a derived quantity is a conserved simulation variable.

Region rules:

- `region=None` uses `fields.selection`. With no `requested_bounds`, the requested
  geometry is the union of its listed complete leaves. This can be a proper subset
  of the mesh; `coverage.complete` then refers to that subset.
- When requested bounds exist, the target is the entire requested box, clipped
  by individual cells. Thus `read_fields(source, region=box)` followed by a
  reduction automatically respects the original box, rather than all cells of
  the intersecting leaves.
- An explicit `(2, 3)` box targets that entire box, regardless of the fields'
  original requested bounds. A `Selection` must use the same `Mesh` object.
  For a `Selection` with bounds, unlisted leaves inside its box count as missing
  coverage. Without bounds, only the listed leaves form the target.
- To require a full-domain integral of a partial field group, pass
  `region=[fields.mesh.lower, fields.mesh.upper]` explicitly.
- Bounds must be finite and strictly increasing on all axes. Domain-exterior
  regions are rejected by default; `missing="omit"` explicitly permits clipping.
  Empty leaf selections are permitted for sums and histograms.

Extrema inspect cells with positive intersection volume. The position is the
centroid of the **clipped cell portion**, in original mesh coordinates, so it lies
inside the region even when the original cell center does not. Positions are
representatives of constant cell values, not inferred continuous extrema. Ties
select the smallest global leaf ID, then the first XYZ interior cell index in
NumPy C order, independent of selection and storage order. `cell_index` excludes
halos. `position_units` remains `coordinate-length`.

## Weights and histograms

With no explicit `weights`, means and histograms use `Wᵢ = ΔVᵢ` (volume weights).
For explicit weights, supply another `Fields` group and a `weight_component`.
It may be the same group as the values. Both must share the identical `Mesh`
object, but may have different selected leaves, slots, halo allocation and order.
Missing coverage is evaluated jointly. The caller ensures compatible time,
physical interpretation and provenance; mesh identity does not prove these.

| `weight_mode` | Effective weight `Wᵢ` | Example |
| --- | --- | --- |
| `"density"` (default with explicit weights) | `wᵢ ΔVᵢ` | Density-weighted temperature or a mass histogram |
| `"cell-total"` | `wᵢ ΔVᵢ / Vᵢ` | A stored total mass or count per cell, apportioned uniformly inside that cell |

Here `Vᵢ` is the complete native cell volume, and `ΔVᵢ` is its retained volume.
For cell totals their ratio is dimensionless and independent of the length
conversion. The default density mode treats the stored weight as piecewise
constant per unit volume. Neither mode infers semantics from a field name or
unit label. With no `weights`, leave weight options at their defaults.

Explicit weights must be nonnegative. A finite negative weight raises even with
`nonfinite="omit"`. Zero weights are allowed, but a mean with zero total weight
raises. A finite value with zero weight still contributes to valid geometric
coverage; a nonfinite value with zero weight follows the nonfinite policy.
Means always retain the value field's units.

Histogram `edges` must be a finite, strictly increasing vector with at least two
entries, expressed in the stored value's units. Bins are left-closed/right-open
except for the last bin, which includes its rightmost edge. Finite values below
or above the edges contribute to `underflow` or `overflow`. `total_weight`
includes these tails, so it equals the sum of bins and tails up to rounding.
Histogram coverage includes finite values outside the bins. No density
normalization or automatic bin selection is performed.

## Rectangular surface flux

`AxisAlignedSurface(axis, coordinate, bounds, normal=1, side="positive")` defines
a plane-aligned rectangle. Its `(2, 2)` bounds are on the other two axes, in
XYZ order: YZ for X-normal surfaces, XZ for Y-normal surfaces, XY for Z-normal
surfaces. Coordinates and bounds use the original mesh coordinate system.

Select the appropriate **coordinate-normal component** explicitly. For a
Z-normal rectangle and magnetic field, pass `"b3"` or `"bz"` as appropriate.
`normal=-1` reverses the sign of the result. The other vector components are not
consumed, and no component naming convention or unit conversion is inferred.

The integral uses the area of every intersected cell patch, including partially
covered patches. Each patch has exactly one owner. At internal cell or block
faces, `side="positive"` chooses the cell on the higher-coordinate side and
`side="negative"` the lower side. This rule also applies to coarse/fine
interfaces; their two sides can legitimately yield different fluxes. Exact
floating-point coordinates determine ownership; no tolerance-based snapping is
performed. At either physical domain boundary, both settings use the interior
cell. Changing `normal` does not change `side`.

The surface request defines its own geometry. It does not inherit the volume
clip from `fields.selection.requested_bounds`; the complete supplied leaves are
available. Missing required patches are handled by the coverage policy. Flux
from a cell-average magnetic field is a one-sided trace of its constant-cell
representation. It does **not** recover staggered CT face fluxes, enforce a
discrete divergence theorem, or guarantee continuity across AMR interfaces.
Arbitrary curved surfaces and reconstructed surface quadrature are outside this
API.

## Units, coverage and failure policy

Without `units`, measures are labeled `coordinate-length^3` or
`coordinate-length^2`. To obtain physical measures, pass the explicit isotropic
conversion `LengthUnits(scale, unit)`, where a coordinate length of 1 equals
`scale` in the named unit. Volume and area scale by its cube and square.
This does not convert field values, weights, input bounds or returned extrema
positions. Field and measure unit labels are combined symbolically; no unit
algebra, string parsing or dimensional inference is performed. For example,
`(kg m^-3) * (m^3)` has a mass interpretation only when the supplied values and
explicit conversion actually establish those units.

Each result has `coverage`, with all measures expressed in `coverage.units`:

| Attribute | Meaning |
| --- | --- |
| `requested_measure` | Requested box, rectangle or explicit leaf-union measure |
| `domain_measure` | Requested geometry inside the mesh domain |
| `available_measure` | Geometry with all required field/weight storage supplied |
| `valid_measure` | Available geometry with finite values and, if used, finite weights |
| `outside_measure` | Requested measure outside the mesh domain |
| `missing_measure` | In-domain measure lacking required supplied coverage |
| `invalid_measure` | Available measure excluded for nonfinite inputs |
| `fraction` | Valid / requested measure, or 0 for an empty request |
| `cell_count`, `valid_cell_count` | Available contributing cells/patches and their finite subset |
| `complete` | No domain-exterior, missing or nonfinite contributions in the requested geometry |

The chosen `missing` and `nonfinite` policies are also recorded in coverage.
Geometric completeness does not mean all histogram values lie within the bins,
or that a nonzero explicit weight covers the whole region.

Both policies default to `"raise"`:

- `missing="raise"` rejects any requested domain-exterior geometry or missing
  leaf in the value/weight coverage. `missing="omit"` integrates the available
  portion and marks coverage incomplete.
- `nonfinite="raise"` rejects NaN and either infinity in contributing interiors
  or weights. `nonfinite="omit"` removes those cells from numerator, denominator,
  histogram and effective coverage together. Ghosts, other field components and
  cells outside the requested geometry are not inspected.
- Omission does not impute values or extrapolate to uncovered regions. A mean
  normalizes only the retained weights. Empty integrals, surface integrals and
  histograms return zero with their coverage; an empty mean or extrema request
  raises rather than returning a fabricated statistic.

Calculations use float64 block operations and compensated summation of scalar
block totals via `math.fsum`, in ascending global leaf order. Histogram bins use
direct float64 accumulation. This does not promise exact summation or bitwise
invariance under mesh refinement. Nonrepresentable measure conversions are
rejected; overflowing arithmetic raises `FloatingPointError`, even when the
final mathematical result could be finite after cancellation. Omission policies
apply to input validity, not arithmetic overflow. Ordinary float64 rounding and
underflow limits still apply.

The implementation visits native blocks with temporary arrays bounded by one
block, plus per-leaf scalar totals and histogram bins. It does not materialize a
uniform volume, retain sample clouds, or add a preparation/execution framework.
It currently runs serially and expects the required `Fields` to have already
been read. No streaming accumulator, distributed reduction or performance claim
is provided.
