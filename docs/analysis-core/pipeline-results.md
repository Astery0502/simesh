# Consumer Spec: Geometry, Access And Scientific Results

Status: confirmed result requirements with open numerical/interface choices.
The shared input is [prepared AMR field data](prepared-fields.md), including
valid ghost neighborhoods. This page defines how each application consumes it
and what its result means. It does not select a physical layout or executor.

## Confirmed Priorities And Scale

[Intent](intent.md#product-intent) owns priorities and the
[operating envelope](intent.md#operating-envelope) owns input, seed and output
scale. Read this page through consumption geometry: irregular RK paths, regular
regional operations, and full-domain observation rays. Different physical field
dependencies remain necessary details within that common model.

## F: Magnetic Lines And Along-Line Diagnostics

### Geometry And Prepared Access

Each seed generates a line sequentially through RK stages and accepted steps.
Its later coordinates depend on sampled values; the resulting lines need not
form a regular geometric shape. Nearby seeds can share neighborhoods initially
and then diverge. Each line needs fast owner lookup, local indices/weights and
repeated access to valid ghost-prepared values, including trial points.

Ordinary tracing samples B with trilinear interpolation and therefore uses the
shared prepared-data boundary. "B only" describes physical field dependencies;
the selected B data still supplies valid ghost neighborhoods. Diagnostics can
add derived fields and coupled state without making a whole-domain D calculation
a prerequisite.
Independent seeds are the parallel unit; stages and steps within one line retain
their dependencies. Endpoint-map Q can add a later neighbor-result dependency.

### Minimum Result Contract

| Item | Required observable result |
| --- | --- |
| Inputs | Supported B interpretation, explicit seed positions/grid, stable seed IDs and direction |
| Termination | Terminal positions, termination information and validity; distinguish last accepted point from a localized endpoint |
| Diagnostics | Requested named accumulators and any coupled auxiliary state; an ordinary trace does not prepare unrequested derivatives |
| Delivery | Diagnostics/summary-only or optional trajectories with stable seed association |
| Later selection | Filter diagnostics/seed IDs; retrieve retained lines or explicitly retrace selected seeds |

| Consumer | Additional field or state demand |
| --- | --- |
| Ordinary line | B values at integration positions, with valid interpolation support |
| Twist | B and curl(B) at required positions, plus the defined integral/quadrature |
| Scott-style Q candidate | B and gradient of b = B/norm(B); auxiliary vectors evolved with position, endpoint projection/normalization |
| Endpoint-map Q candidate | Neighboring seed endpoints, seed spacing and mapping differences; endpoint error can be amplified |

[Shared validity](prepared-fields.md#input-reach-and-remaining-validity) governs
support through these compositions. The Q method is open. Saved trajectory
points can serve later attribute sampling but do not automatically supply the
stages, derivatives or quadrature needed by a coupled diagnostic.

Name the tangent, parameterization and orientation. With arclength tangent
`t = dx/ds`, a directional derivative can be `grad(q) dot t`; an integral must
name its integrand and whether it uses arclength or an oriented line element.
Differentiating sampled path values and sampling a spatial derivative are
different discrete operations. Do not evaluate every diagnostic at every RK
stage unless its coupling or quadrature requires it.

Earlier investigation consulted FastQSL2's
[private integrands](https://github.com/el2718/FastQSL2/blob/main/privates.f90),
[tracing source](https://github.com/el2718/FastQSL2/blob/main/fastqsl.f90) and
[paper](https://arxiv.org/html/2604.16195v1) on 2026-09-07. It recorded the twist
integrand `curl(B) dot B / (4*pi*|B|^2)`, segment accumulation, `dbdc_grid`
derivatives of normalized B, and auxiliary evolution in `interpolate_cartesian`,
with precomputed/on-demand derivative branches. These are scientific and
organizational references on structured grids, not native-AMR storage or simesh
performance evidence. This editorial pass preserves that provenance; no external
source was executed or remeasured.

### Retention And Acceptance

Diagnostics-only and user-selected retracing are valid modes. When trajectories
are requested, they are scientific outputs under
[product ownership](prepared-fields.md#explicit-retention-versus-internal-caching),
independent of field caches. If selection depends on a completed diagnostic,
declare bounded per-line staging or a sink that can finalize retention; do not
assume the choice is known at launch. Retracing discarded/unretained paths is
a new user-requested action, not an automatic replay mechanism.

Acceptance covers boundary/section launches, weak/null fields, coarse/fine
transitions, short/long paths, coherent/divergent seeds, termination and endpoint
error. Separate trajectory, derivative, diagnostic and quadrature errors.
Validate auxiliary dynamics or endpoint differencing for the selected Q method.
Exercise diagnostics-only output, post-diagnostic selection and explicit
retracing; test joint retention only when that delivery mode is selected.

Count rejected trial work, owner transitions, misses, bytes per accepted point,
active RK/auxiliary state, output staging and total worker/field memory. A changed
coloring field can reuse a saved curve; a changed tracing field or accuracy can
invalidate its geometry. Velocity streamlines are an adjacent consumer and are
instantaneous curves, not time-dependent particle paths.

Existing SLE/TRM fixed RK4, oriented B-dot-dx accumulation, whole-step rejection
and accepted-prefix rules remain scoped references. `DOMAIN_EXIT` does not prove
an exact boundary intersection. Eight-seed/eight-step WENO results do not validate
long-path accuracy or million-seed scaling. Open choices are [F1](baseline.md#f1).

## D: Whole-Domain Current/Gradient Then Slices

### Geometry And Prepared Access

D traverses known full-domain or selected regular regions over locally regular
leaf grids. Boxes, slabs and planes can be selected before field arithmetic;
AMR levels and boundary support still make their block/storage coverage nonuniform.
This favors geometry-based selection, batched block stencils and reuse of prepared
neighborhoods. It must also support local derivative consumers under their own
requested coverage.

The confirmed D mainline calculates the requested derivative throughout the
physical leaf domain, then supplies slices for structure analysis. That request
cannot silently become a calculation on the displayed plane alone. Streaming or
other bounded delivery may preserve global coverage without global RAM residency.
A stand-alone slice can have narrower coverage; its geometry is specified in
[U04](workflows.md#u04-axis-aligned-slices-oblique-slices-and-line-profiles).

| Item | Required observable result |
| --- | --- |
| Definition | Inputs, component/axis operations, centering, units and normalization; distinguish physical current from curl(B) |
| Coverage | Requested valid derivative outputs on physical leaf cells, without duplicate parent or ghost contributions |
| Validity | Interior and remaining support available to later sampling, following the shared reach rules |
| Slice | Declared positions/geometry, components or magnitude, values and validity; a specified sampled image or native intersection |
| Subsequent access | State how repeated slices access previously calculated data under its promised lifetime/delivery |

Isosurfaces are a secondary consumer of the derived result and need their own
reconstruction/seam contract. Acceptance covers interior/interface/physical-boundary
error, global coverage, derived validity, global-plus-slice cost and live
input/derived/output memory. Compare resident and bounded organizations for the
same requested work. Open choices are [D1](baseline.md#d1).

Canonical resident derivatives and rewrite local kernels are references. LFE
allocates full spatial blocks for selected primaries, writes selected ROI windows
and sums one unweighted component. It does not deliver a global derived-field
lifecycle, packed ROI output or a general interpolatable physical current product.

## L: Full-Domain LOS Integration Of A Local Response

### Geometry And Prepared Access

The confirmed result is

```text
I(u,v) = integral along the declared LOS of epsilon(rho,T) dl
```

L needs consistent processing of the global AMR domain. Observation direction,
pixel position and physical bounds determine each ray's entry, exit and integration
depth. Different rays can traverse different sequences of leaves, cell spacings
and refinement levels. Those crossings determine where field neighborhoods are
needed and how contributions reach image pixels. A small image can require
access across the entire declared volume.

Use the same prepared-field meanings for response reconstruction and interpolation.
Ray-oriented traversal and block/cell-oriented contribution accumulation are open
ways to supply this geometry; neither is selected. Their ownership and reuse
patterns differ from the evolving RK paths and regular regional stencils.
Global treatment requires correct coverage and weights, not a finest-resolution
uniform intermediate or a simultaneously resident snapshot.

| Item | Required observable result |
| --- | --- |
| Response | Explicit epsilon, fields/EOS dependencies, units and normalization, including how temperature is obtained |
| Observation | Image plane/basis, LOS direction, bounds, pixels and physical depth for each ray |
| AMR coverage | Physical path lengths through valid leaves, without counting covered parents or ghost values as extra material |
| Output | Image coordinates, units, validity/completion and declared empty-ray behavior |
| Parallel accumulation | Disjoint image ownership or a defined race-free merge, with total worker/image resources |

For a declared piecewise-constant reference ray, the integral is the sum of cell
response times ray-cell intersection length. This is useful for testing AMR
coverage/weights; production reconstruction and its support remain open under
the shared preparation contract. Pixel-area averages require
additional geometric weights. Cell counts and volume sums do not substitute for
path length. A weighted projection also differs from an unnormalized integral.
Choose response/reconstruction ordering under the shared field semantics.

Earlier investigation consulted AMRVAC's
[thermal-emission module](https://amrvac.org/mod__thermal__emission_8t.html) and
[source](https://amrvac.org/mod__thermal__emission_8t_source.html) on 2026-09-07.
It identified `get_image_datresol` and LOS routines using cell spacing and length
units as references for response/image integration. The solver's runtime and MPI
organization are not adopted. Absorption, full radiative transfer, spectral-line
physics and instrument effects remain unselected extensions.

Acceptance needs analytic constant-response/path-length cases, differing view
angles and per-pixel depths, mixed AMR levels/interfaces, full coverage and no
double counting. Independently check physical response and units. Validate the
declared serial/parallel accumulation and whole-domain throughput with complete
input/worker/image memory. Magnetic tracing or local curl evidence does not
validate L. Open choices are [L1](baseline.md#l1).

## Shared Foundations And Separate Execution

The [shared spec](prepared-fields.md#consumption-geometry-and-access-patterns)
owns common geometry, data, halo validity and ownership. The distinguishing
execution state is F's ordered RK/auxiliary state, D's regional stencil work and
L's ray-depth/AMR traversal with image accumulation. None of these requires one
universal traversal or forces the others' scientific computation.

## What Is Ready, And The Next Stage

Use [baseline](baseline.md#first-outcome-and-spec-readiness) for the next shared
boundary decisions and the candidate first result. This page supplies consumer
constraints, not an activated implementation group. Numerical choices and
quantitative acceptance remain open until recorded for the selected consumer.
