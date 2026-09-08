# Application Catalog And Supporting References

Status: application reference. The main reading path is [intent](intent.md),
[shared data spec](prepared-fields.md) and [F/D/L results](pipeline-results.md).
This catalog preserves U01--U12 lookup and the detailed geometry/acceptance of
additional products. It does not define a second intent, architecture or work queue.

<a id="authority-and-priorities"></a>
<a id="scope-evidence-and-requirement-status"></a>

Priorities, scale, supported input and exclusions are defined in [intent](intent.md).
U01--U07 and uniform output are requested families; U08--U10 and spectra are
candidates; U12 records existing uses and continuity obligations. Desired output
is not a delivered API. Native surface connectivity does not expand the supported
input mesh family, and GUI/rendering remains the consumer's responsibility.

<a id="what-analysis-needs-from-a-snapshot"></a>

Physical dependencies, units, centering and EOS interpretation have one definition
in [physical fields](prepared-fields.md#physical-field-definitions). The mainlines
share prepared AMR data; their [geometric consumption patterns](prepared-fields.md#consumption-geometry-and-access-patterns)
drive access and organization.

## Concrete Workflow Catalog

### U01: Open, Inspect, And Read Native Data

Inspect time, fields, units, dimensions, levels and bounds without loading field
payload. Return selected native fields or load the requested set for dense use.
[Selection](prepared-fields.md#request-and-physical-selection) includes a non-leaf
level-one parent and its descendant leaves.

Choose geometry before payload reads and reuse compatible offsets/metadata.
Actual read granularity follows the file layout; record amplification rather
than promising individual-cell I/O. Acceptance requires exact values/order and
centering, useful first-result latency, explicit calls/bytes and complete memory.
Bulk reads need competitive throughput; selective reads cannot require an
unadvertised whole-file conversion. The staggered-record adapter question is
[X1](baseline.md#x1).

### U02: Local And Global Ghost Completion

Prepare valid neighborhoods for a local request or complete selected fields
for dense regional/global analysis. The complete contract is
[preparation and consumption](prepared-fields.md#preparation-and-consumption-requirements).
Compare local and global cases separately with matching fields, reach, boundary
meaning and input residency. Count useful halo values, planning/checking,
transfer/read/copy work and repeated fills. A locally small request and a bulk
refresh can favor different organizations of the same prepared-data meaning.

### U03: Magnetic Lines, Velocity Streamlines, And Along-Line Diagnostics

See [F](pipeline-results.md#f-magnetic-lines-and-along-line-diagnostics) for
irregular RK-path access, dependencies, output/retention and acceptance.
[Sequence B](lifetime-sketches.md#sequence-b-field-lines-more-seeds-and-diagnostics)
examines additional seeds and later attributes. Velocity streamlines are an
adjacent use; null/reconnection diagnostics need separate physical definitions.

### U04: Axis-Aligned Slices, Oblique Slices, And Line Profiles

Return density on `z=z0`, current magnitude on an oblique plane, or a pressure
profile on a segment. Choose native intersected cells/polygons or a specified
2D image/1D sample array, including coordinates and validity masks. Pixel-center
sampling, pixel averaging and cell intersection are distinct products.

The plane/segment is geometrically regular and known before evaluation, even
when its AMR intersection is nonuniform. Select intersected blocks and required
sampling/stencil support. Nearby planes may share prepared values; a different
field on the same plane can reuse geometry/location. Oblique cuts can still
have substantial support cost. D's [global-then-slice request](pipeline-results.md#d-whole-domain-currentgradient-then-slices)
retains its larger computation coverage.

Acceptance covers thin non-block-aligned cuts, oblique planes, coarse/fine seams,
physical boundaries and nearby-plane sequences. Cost should follow requested
geometry, real support and pixels, without a prerequisite 3D uniform cube.
State resolution and unresolved-feature behavior. Surface integrals/fluxes
need geometric weights; smooth interpolation alone does not establish accuracy.

### U05: Isosurfaces And Field Values On Surfaces

**Request/result:** extract a surface `rho=rho0`, `T=T0`, or `|curl(B)|=j0`, then
return geometry and optionally sample a different field on it, or reduce its
area/flux without retaining the full geometry.

**Access/reuse:** the defining scalar and its dependencies determine candidate
cells. Without a conservative field-range index, a first full-domain search
must scan that scalar throughout the domain even if the result is small. Reuse
scalar blocks/ranges when changing the level; reuse surface geometry when only
the coloring field changes. Read coloring dependencies at the surface/support
after candidate selection when that reduces total work.

**Acceptance:** choose a cell-to-point/reconstruction rule and AMR transition
handling before selecting an extraction algorithm. Test interface cracks,
duplicate faces/vertices, orientation, ambiguity/equality cases, and geometric
or integral error. Cell-center extrema alone are not a conservative bound for
every reconstruction or derived field. Record scan/pruning cost, transition
support, output triangles and peak geometry memory. Chunked output must preserve
seams; a local uniform intermediate is permissible only with explicit resolution,
memory and scientific error, not an automatic whole-volume finest-level grid.

### U06: Threshold Regions And Isovolumes

**Request/result:** find `T1 <= T <= T2`, `|J| > J0`, or dense/cool material;
return a native cell selection, a volume clipped within cells, its boundary,
or only volume/mass/energy. These are distinct products.

**Access/reuse:** evaluate predicate fields first, then requested quantity fields
in the selected region and required support. Whole-cell thresholding needs no
interpolation halo for a stored scalar. An interpolated isovolume needs a field
reconstruction and partial-cell geometry. Reuse raw scalar ranges and masks as
appropriate; changing threshold invalidates the mask, not the unchanged raw
field. Arbitrary new predicates may require another full scalar scan.

**Acceptance:** whole-cell membership must not be described as partial-cell
isovolume accuracy. Volume/mass reductions use the chosen occupied volume,
without counting ghost or covered coarse cells. A reduction-only request must
not require allocating a volumetric mesh. Measure predicate scan, secondary
field reads, selected fraction, geometry size and threshold-sweep reuse.
Connected components require cross-block/coarse-fine reconciliation and may
need retained global labels/frontiers beyond a single chunk.

### U07: Full-Domain LOS Integration And Synthetic Images

See [L](pipeline-results.md#l-full-domain-los-integration-of-a-local-response)
for global coverage, observation direction, per-pixel depth, AMR crossings,
response and image acceptance. A small image can require full-volume processing.
Persistent ray/intersection maps are candidates in [T12](technique-candidates.md#t12-reuse-geometry-separately-from-sampled-attributes);
weighted projections, Faraday and fuller transfer models need their own scope.

### U08: Statistics, Budgets, Profiles, And Histograms (Candidate)

**Request/result:** regional/global mass and energy, extrema with locations,
volume/mass-weighted distributions, density-temperature phase histograms,
radial profiles, or quality summaries such as magnetic-divergence norms.

**Access/reuse:** scan only integrand, predicate, bin and weight fields. Fuse
compatible reductions over a chunk and discard payload when it has no likely
reuse. Geometry provides cell volumes. A derivative-based statistic adds real
halo dependencies; pointwise mass does not. Reuse bin maps for unchanged
geometry when beneficial. Automatic bin limits may need a first pass.

**Acceptance:** distinguish cell counts, volume weights, mass weights and fluxes;
count each physical leaf cell once. Return a table/accumulator, not an obligatory
field volume. Specify empty selections, nonfinite values, normalization and
reduction order. Exact quantiles are not assumed to be constant-memory streaming
reductions; an approximate sketch requires declared error and a separate choice.

### U09: Detect, Rank, And Follow Structures (Candidate)

**Request/result:** identify dense clumps, shocks/compressions, vortical regions,
current sheets or magnetic structures; return masks, component properties,
selected geometries, or tracked identities over time.

**Access/reuse:** predicate and derivative neighborhoods, then connectivity
across selected cells; later time matching may use a spatial overlap or motion
model. Share underlying current/vorticity fields with slices and surfaces.

**Acceptance:** each physical detector needs its own criterion and scale;
thresholding current is not automatically identifying a reconnection site.
Define adjacency, coarse/fine joining, minimum feature size, boundary treatment,
and identity under splits/merges. Budget labels/frontiers and global reconciliation;
do not infer bounded memory from chunked field loading alone.

### U10: Snapshot Series, Comparisons, And Time-Dependent Paths (Candidate)

**Request/result:** repeat a slice/statistic/line query across snapshots; compare
two states on matching physical locations; optionally integrate actual particle
paths through time-dependent velocity fields.

**Access/reuse:** read the required fields per snapshot and release old payloads.
Geometry reuse requires an established compatible topology/coordinate lifecycle;
payload, halos, derived fields and field-range summaries change with values.
Different meshes require a specified common sampling/integration representation.
Time-dependent paths need bracketing snapshots and temporal interpolation.

**Acceptance:** bound resident snapshots/in-flight outputs, preserve times/units,
and identify interpolation in space and time. Same leaf count is not proof of
same topology; same topology is not proof of unchanged values. Measure throughput
and peak memory across the series, including topology changes, not just frame 1.

### U11: Uniform Sampling, Spectra, And External Consumers (Mixed)

Uniform output is confirmed; spectral diagnostics remain candidates. Use the
[intent scale](intent.md#operating-envelope) and shared delivery/resource rules.

**Request/result:** export a requested regular subvolume for NumPy/VTK or obtain
a regular patch for FFT spectra/correlations. Canonical regular sampling/export
is existing-use evidence; spectral diagnostics are candidates.

**Access/reuse:** select fields/ROI/resolution explicitly and reuse the sample
map. An FFT genuinely couples a whole chosen patch; it is not a local stencil.
Materialize only the requested representation and account for transform buffers.

**Acceptance:** distinguish exact level-1 placement from refined resampling;
name interpolation, resolution, windowing, normalization and boundary assumptions
for spectra. Do not claim native-AMR conservation or spectral fidelity solely
because a uniform array was produced. Export selected attributes and geometry
needed by the consumer; current level-1 VTK support does not establish native
refined surface/volume export.

### U12: Reference Fields, Model Comparison, And Dataset Editing (Existing Use)

**Request/result:** extrapolate a potential field from bottom normal B, construct
a dipole/bipolar/flux-rope model, compare with a simulation in a region or on
lines, or edit/subset/write a dataset for another tool or experiment.

**Access/reuse:** potential extrapolation uses a boundary map and generally has
nonlocal coupling to that map; model evaluation need not read a DAT at all.
Reuse fixed geometry/kernels when justified. Compare models at matching physical
coordinates. Editing is an explicit lifecycle that invalidates affected reuse.

**Acceptance:** preserve the helper's boundary, flux-balancing, centering and
normalization meaning; direct/FFT choices need their own comparison. Model-field
and potential-field support do not imply a nonlinear force-free solver or a
gauge-consistent helicity calculation. Write-back/roundtrip and uniform
construction remain supported migration obligations, but the core is not a
replacement HD/MHD evolution solver.

## Shared Specifications And Acceptance

Earlier sections at the anchors below were consolidated into their owning
specification. These links retain access from historical rewrite navigation.

<a id="workflow-acceptance"></a>
<a id="turn-fast-into-a-reviewable-result"></a>
<a id="scientific-accuracy"></a>

Scientific acceptance and measurable request profiles: [performance](performance.md#scientific-accuracy)
and [reviewable cost](performance.md#turn-fast-into-a-reviewable-result).
F/D/L specifics stay in [consumer results](pipeline-results.md); other product
criteria remain beside U05--U12 above.

<a id="memory-and-execution-intent"></a>
<a id="large-data-and-compute-backends"></a>

Memory, residency and device evidence: [shared resources](prepared-fields.md#resources-and-parallel-ownership)
and [complete accounting](performance.md#complete-workflow-resources).

<a id="reuse-depends-on-what-the-user-changes"></a>
<a id="repeated-snapshot-analysis"></a>

Reuse and invalidation: [shared rules](prepared-fields.md#reuse-and-invalidation)
and [usage sequences](lifetime-sketches.md).

<a id="checks-and-numerical-trust"></a>
<a id="local-derived-fields"></a>

Owned checks and derived composition: [preparation](prepared-fields.md#preparation-and-consumption-requirements)
and [input reach/remaining validity](prepared-fields.md#input-reach-and-remaining-validity).

<a id="along-field-analysis"></a>

Along-field parameterization, diagnostics and acceptance: [F](pipeline-results.md#f-magnetic-lines-and-along-line-diagnostics).

<a id="what-the-latest-evidence-changes"></a>

Measured organization costs and their qualifications: [baseline evidence](baseline.md#decision-relevant-evidence).

## Historical Evidence

### Where These Intentions Come From

The application families combine user requirements with observed canonical and
legacy uses. Scope follows the current [intent](intent.md), not the breadth of
an external example or an old helper name.

| Source | Observable use and its implication for the catalog |
| --- | --- |
| [Project README](../../../../README.md), [public API map](../../../python-api-map.md), [user guide](../../../user-guide.md) | Native block access, regular sampling, derived fields, selected fields, mutation/write-back and format interoperability: U01/U04/U11/U12 |
| [Canonical derived fields](../../../../src/simesh/amrvac/derived_fields.py), [AMR kernels](../../../../src/simesh/utils/lib/amr/mesh.pyx) | Shared derivative computation and remaining valid halo layers: U02 and derived consumers |
| [Legacy analysis helpers](../../../../src/simesh/legacy/meshes/amr_mesh.py) | `calculate_current`, `calculate_divv`, `export_uniform_current`: current/compression and displayable fields; legacy names/formulas are evidence to verify, not new scientific definitions |
| [Potential-field tools](../../../potential-field-tools.md), [magnetic configurations](../../../../src/simesh/utils/configurations.py) | Boundary-driven reference fields, dipole/bipolar/fan/flux-rope models and comparison workflows: U12 |
| [Capability ledger](../../../../rewrite/CAPABILITIES.md), [migration ledger](../../../../rewrite/SOURCE_MIGRATION.md) | Existing internal refined sampling/curl/tracing versus unfinished public composition and compatibility obligations |
| [WENO full comparison](../../../../rewrite/evidence/M1-WENO-REFERENCE-COMPARISON.md), [thin-region assessment](../../../../rewrite/evidence/M1-WENO-ASSESSMENT.md) | Measured resident/bounded and support-amplification limitations; grounds for changed acceptance, not a blanket rejection of the rewrite |

The following external primary documentation was recorded as consulted on
2026-09-07. It supports terminology and product distinctions, not workload
frequency, architecture selection or a claim of simesh delivery. These are
preserved investigation references, not new measurements:

- ParaView documents slices, contours, thresholding and stream tracing as
  different data transformations, with different output dimensions and memory
  consequences. This supports distinguishing U03--U06 products.
  [ParaView filtering guide](https://docs.paraview.org/en/latest/UsersGuide/filteringData.html).
- VisIt's isovolume operation retains cells and parts of cells within a scalar
  interval; its isosurface operation can use a defining field different from
  the plotted field. These distinctions motivate separate selection, geometry
  and attribute requests in U05/U06.
  [VisIt isovolume](https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.5.0/using_visit/Operators/OperatorTypes/Isovolume_operator.html),
  [VisIt isosurface](https://visit-sphinx-github-user-manual.readthedocs.io/en/v3.3.2/using_visit/Operators/OperatorTypes/Isosurface_operator.html).
- yt exposes geometric/field selections and quantities over selected data;
  its streamline objects support querying fields along a path, and its profiles
  distinguish weighted values from sums. These motivate U03/U08 composition.
  [yt data objects](https://yt-project.org/doc/analyzing/objects.html),
  [yt streamlines](https://yt-project.org/doc/visualizing/streamlines.html),
  [yt profiles](https://yt-project.org/doc/reference/api/yt.data_objects.profiles.html).
- MPI-AMRVAC's Python documentation distinguishes metadata inspection from
  payload loading and describes H-alpha and Faraday synthetic views, including
  the importance of units. These inform U01 and the now-confirmed U07. Its documented
  regridding workflow is an example of an existing approach, not the preferred
  simesh native-AMR strategy.
  [MPI-AMRVAC DAT analysis](https://amrvac.org/md_doc_2python__datfiles.html).

### Retained Rewrite Records

[M1 strategy records](../../../../rewrite/evidence/M1-ANALYSIS-DECISIONS.md) retain
original choices and reopen triggers. The
[M1 horizon](../../../../rewrite/evidence/M1-ARCHITECTURE-HORIZON.md) records its qualified
fixture pass and support/cache trade-offs; the
[earlier reorientation audit](../../../../rewrite/evidence/ANALYSIS-PRIORITY-REAUDIT.md)
is historical. They are evidence, not recurring audit commands or stage order.

The explicit [WENO profile](../../../../rewrite/WENO-REFERENCE.md) is a complex real-data
assessment, not a routine test or a substitute for independent manufactured
accuracy cases. Apply [performance policy](performance.md#weno-reference-comparisons)
when a selected claim needs it.
