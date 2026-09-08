# Native Analysis Core: Selected Development Design

Selected 2026-09-08 for P0--P4. Requirements remain in [prepared-fields](prepared-fields.md)
and [pipeline-results](pipeline-results.md). This is the implementation decision
record. Selection is not a completion claim.

The runtime and thermal/geometry branches are now integrated in one checkout.
[Current status](current.md) owns recovery and remaining acceptance;
[runtime execution](runtime-execution.md) owns the retained cache/backend choices.
Later sections preserve their original experimental scope. Optional geometric
plans coexist with source value validation and raw-value caching; nonlinear
thermal rays retain their resident dispatch and share the native inline sampler.
No production rebricking or automatic persistent plan policy is selected.

## P0 Decisions

S1: retain zero-based original leaf IDs and immutable validated flat forest
geometry. Each field group has a separate leaf-to-slot directory: source identity
is not a storage address or permanent compute ID. Ownership is half-open; upper
physical boundary handling belongs to the consumer. Regions use exact cell-center
windows; parent selection resolves descendant leaves. Ghosts/parents add no
physical coverage.

S2: initial strategy `rewrite-ratio2-minmod-exactphase-cont-v1` uses ordinary
cell-centered float64 cell averages, RST-001 ordered eight-cell averages,
LIM-001 slopes, PRL-001 exact +/-0.25 phases, all-26 support closure and continuous
physical boundaries. Prepared inputs have two valid layers. Wider unsupported
reach is rejected. Reuse RHE/RHC without changing their checks, failure rules or
arithmetic rather than introduce another untested transfer during assembly.

S3: independent component-adjacent `(slot,x,y,z,component)` float64 groups,
complete original-leaf patches and explicit rectangular views. Each group owns
field order and remaining halo. Derived groups do not rebuild primary storage.
The bootstrap uses RHE's bounded field-major scratch and one publication copy;
padded support and planning are measured limitations, not claims of compact
support/fused preparation. Compare complete consumers before substitution.

S4: resident owned products and a fixed-capacity bounded pool share this layout.
Idle pool entries may be evicted; scoped borrowing pins backing until return.
Detached products own backing and outlive pool/source closure. Publish only
successful complete preparations; failure never marks partial storage ready.
Sources remain immutable during their lifecycle. Clear cannot invalidate active
borrows. Explicit retained products are not revocable cache keys.

S5: coordinator prepares/publishes. Compiled consumers read completed arrays,
keep private RK/ray state and write disjoint outputs. Parallel misses return to
the coordinator at defined continuation boundaries. Do not share serial CHS.
Resident work bypasses cache maintenance.

S6: limits and fixture are in [current](current.md). Initial acceptance uses exact
selection/slot conformance plus independent affine/constant/mixed-interface
examples. WENO finite comparisons retain `rtol=atol=1e-10` from the established
comparison profile; report nonfinite classifications separately. This is an
implementation-comparison tolerance, not a physical accuracy SLA. Freeze new
trajectory/derivative criteria with the selected reference before implementation.
Default >10% investigation and reproduced >20% regression rules apply.

## Providers And Packaging

Development adapters explicitly import rewrite metadata, forest, selected block
I/O and RHC. Numerical kernels import no file/cache/rewrite provider. Array/memmap
sinks deliver scientific arrays; complete `.dat` serialization remains canonical
`write_datfile_from_sfc` with matching ordinary-field metadata. Resolving the
checkout-local rewrite dependency is a P4 installed-workflow gate; do not hide it
in package imports. Retain canonical public workflows and rewrite comparators.

Resident WENO bootstrap uses canonical ordinary B reads. Bounded native reads
initially use the recorded ordinary bridge and count its startup. A direct
ordinary-field reader for staggered-tail records is a distinct extension if
measured startup justifies it, without enabling CT computation.

## Exploration Disposition

| Route | Decision and reason | Reopen condition |
| --- | --- | --- |
| E1 | Defer default merged bricks after measured WENO geometry and F/D/LOS prototype; retain source/compute/slot separation. | Direct brick-boundary preparation and native mapped consumers justify full costs; see [E1 evidence](evidence/rebricking.md). |
| E2 | Defer basis reconstruction, which changes interpolated/derived meaning and is unnecessary for the confirmed baseline. | A named consumer needs continuity/analytic reconstruction derivatives with applicable acceptance. |
| E3 | Defer gridlets/dual mesh. Sampled slices and rays can use native prepared groups. | Native intersections/isosurface seams justify explicit interface topology. |
| E4 | Defer RT hardware: no selected RTX/OptiX platform on this CPU host. | Suitable resources and measured location bottleneck. |
| E5 | Adopt optional selected retained fill plans after exact conformance and repeated F/D/LOS cost comparisons; keep ordinary preparation. | Automatic retention or fused execution requires full resource/runtime evidence; see [E5 evidence](evidence/geometry-plans.md). |

These decisions use the preserved [route analysis](algorithmic-directions.md)
and existing WENO evidence; no external result is a new simesh measurement.

## First Checks And Costs

A mixed-level affine fixture with permuted fields/leaves establishes RHC packing,
two-layer validity and direct sampling. A bounded pool case checks eviction,
pinning, source failure and detached lifetime. A derivative composition checks
validity shrink. Measure preparation+packing+first sample and ready repeats,
resident outputs and bounded support/scratch. Reuse unchanged provider evidence.

Compare direct ready sampling against identical values and an independent scalar
trilinear reference; distinguish general points from canonical regular-grid
sampling. Include setup. The bootstrap may retain a measured memory trade-off;
it is not a preparation speed improvement over canonical bulk refresh.

## First Derivative Consumption

Centered first differences consume extended primary values, leaving one valid
layer from two. Compute requested linear combinations in declared term order,
with `(right-left)/(2*dx)` then coefficient multiplication and ordered addition.
This new consumer strategy is numerical conformance within finite tolerances,
not a promise of canonical expression-tree bits. Curl is explicitly curl(B) in
field-units/coordinate-length, not a normalized physical current. Independently
allocated output can be sampled later; further derivatives propagate reach and
sampling rejects zero remaining halo. P3-D global delivery and slice acceptance
remain distinct from this P1 boundary check.

## P0/P1 Checkpoint

The assembled design and initial native fields passed the focused composed
checks and WENO selected width-two comparisons in [P1 evidence](evidence/p1-native-fields.md).
One active scoped pool borrow is shared by workers; nested leases are rejected
rather than allocating unbounded per-lease directories. Nonpacked pool slots
provide explicit per-leaf windows; detached products provide packed interiors.
The RHC publication copy is about 1% of the mixed first-preparation composition,
so retain component-adjacent storage and target provider planning if optimizing
preparation. Continue P2 without claiming the RHC bootstrap is a faster dense
fill or that a 1 GB fixture establishes target scale.

## P2 Numerical And Execution Contract

F1 initial strategy is `rk4-arclength-accepted-prefix-v1`: ordered classical
fixed-step RK4 of `dx/ds = direction*B/|B|`, float64, caller step/max_steps/
max_length/null threshold. Last step is capped to remaining length. Accumulate
accepted arclength parameters, not chord lengths or B-dot-dx. Source coordinates
set length units; field components must share declared units. Fixed step is an
explicit accuracy input, independent of seed spacing; no global fourth-order
claim is made for AMR-interpolated fields.

Seeds have stable caller IDs and finite (n,3) coordinates. Half-open domain
ownership accepts lower-boundary launches; upper faces are outside under this
initial profile. A domain-exiting RK trial or final proposal rejects the entire
step. Return the last accepted position, accepted length/steps, terminal reason
and `localized_endpoint=False`; no exact footpoint or boundary-length claim.
Null/nonfinite values have distinct termination. Ordinary F prepares no curl.
Exact endpoint localization and upper-face inward launch need a later explicit
profile before endpoint-map Q or full boundary footpoint delivery.

Each seed keeps its RK stage, four tangents and accepted state across preparation
misses. Missing owners pause without classifying a physical termination. The
coordinator batches missing leaves, freezes the pool, and resumes disjoint seed
arrays in nogil compiled kernels. A Python thread pool supplies at most four
workers; no OpenMP runtime is required for this initial independent-seed path.
The same kernel with one worker is the matching serial comparator. Bound active
seed chunks independently of total output, and support summary-only chunk output.
Retained trajectories/retracing are later P3-F output work, not implicit replay.

Reference acceptance before implementation: uniform affine/constant vector traces
with analytic paths, zero/null/nonfinite and exiting seeds; a smooth interior
helical field with step refinement must converge to its analytic trajectory,
with finite same-step serial/parallel and resident/bounded agreement. Use the
same arithmetic ordered scalar RK4 reference for implementation tolerance
rtol=atol=2e-12 on finite results; record observed analytic errors and step-refinement
ratios rather than invent a user physical SLA. WENO starts with the recorded
eight-seed/eight-step request and a named longer/divergent request; both report
termination, field preparation and complete resources. Endpoint validity remains
explicitly limited to accepted prefixes.

P2 comparison profile (frozen before measurement): WENO short = eight SFC-spread
mixed-ROI centers/eight steps; long = 32 such centers/128 steps, step one quarter
of the minimum selected spacing. Compare one/four workers at 16 or 64 slots,
respectively, application cleared and warm, one warmup/three repetitions. Add
four-slot short control to reveal undersized-cache cost. Rewrite SLE supplies
short position/accepted-count comparison only: its one-layer CHS, retained
trajectories and B-dot-dx accumulator are different work and not an equal
end-to-end speed comparator. New one/four-worker results must match exactly.
A separate 2048-seed/600-step manufactured helix supplies enough ready-data work
to measure parallel throughput. Include source/geometry setup and pool/workers/
summary memory; no million-seed claim follows from this case.

The new F tangent computes nested `hypot` and divides components by that norm;
this is a named finite-tolerance strategy, not FLN-001's scaled expression-tree
bits. Nonfinite input and unrepresentable norm are separate terminal statuses.
The provider and historical SLE/FLN/RKS entrypoints retain their original rules.

P2 bounded cache probe: the 32-seed/128-step WENO case needed 163 prepared-owner
loads from a cold 64-slot pool and about 126 on warm repeats. Hypothesis: 256
slots cover the actual visited working set and remove warm preparation without
changing any trajectory. Compare 64 versus 256 slots, three repeats, one worker;
allow at most 15 minutes and <32 MiB additional controlled arrays. Retain 256 for
this repeated request only if saved reads/fills and whole-trace time justify its
extra memory. This is a cache-sizing decision, not a universal capacity default
or an E5 fused-plan implementation.

## P3-D Global Derivative And Slice Contract

D1 selects full-domain curl(B), in stored B units per coordinate-length, with
centered differences of extended input fields. Publish independently allocated
three-component native results with one valid halo. A bounded pass computes every
physical leaf exactly once; no slice limits the requested global computation.
Caller-supplied float64 array/memmap output is supported with explicit full-output
admission. On provider failure, no product is returned; an explicit sink may
contain completed earlier batches and is not certified complete.

A retained global result supplies repeated axis-aligned or oblique pixel-center
slices by trilinear interpolation. A plane is `origin + u*t + v*s`, with t/s the
uniform pixel-center fractions in [0,1]; u/v are independent full-width vectors,
not unit directions. Outside or missing points are NaN with false validity.
No area average, native intersection, current normalization or implicit EOS is
claimed. Geometry is separate and reusable; slice calls do not recompute curl.

Compare bounded RHE exact-phase preparation with a named
`canonical-coordinatephase-cont-v1` resident preparation adapter. The canonical
provider retains its actual C-owned padded/coarse memory and supplies the same
component-adjacent consumer boundary; no hidden repack is needed. These transfer
strategies have the documented small phase-rounding difference, not bitwise
identity. Expose all retained owner storage in budgets/lifetimes. Keep the bounded
path even if the affordable resident path wins dense throughput.

Acceptance: global coverage plus independent centered differences and boundary
extension on manufactured data, mixed AMR derivative samples, exact repeated
retention/field-order behavior, and WENO all-cell canonical comparison at the
predeclared finite 1e-10 tolerance. Measure global preparation+derivative+first
slice, repeated slices, resident versus bounded resources and canonical operator
cost; preserve accepted input/group semantics if attribution prompts redesign.

P3-D WENO profile: all 22,614 leaves, B fields 4/5/6, two primary halos and
one derived halo, continuous boundaries; retain all three curl components and
sample a 128x128 axis plane plus a 96x80 oblique plane. Record first bounded
pass and one repeat (descriptive, not a tight timing gate), and three resident
operator/slice repetitions. Bounded output uses an explicit ~543 MB memmap;
close its mapping before the resident comparison and read comparison tiles only.
Keep output-file bytes, accessed mapped pages and resident allocations distinct.
Compare every retained derived value and canonical derivative components in
bounded comparison tiles at 1e-10; sum per-component canonical operator times
only as an operator reference, not a complete retained Dataset workflow. Count
canonical C-owned coarse storage and anchor it in every exposed NumPy view.

P3-D feedback before closure: the new generic derivative loop measured 0.516780 s
for all three components, while canonical's three separate component kernels
summed to 0.256313 s (not identical output-retention work). Inspection identifies
term metadata, axis selection and spacing reads inside every cell. Bound one
E5-related execution probe to existing cell-major versus term-major loops,
15 minutes, three same-runner repetitions at mixed-region and full-domain sizes.
Move invariant metadata/denominator outside cell loops while preserving division,
coefficient multiplication and each output's addition order exactly. No reciprocal
substitution or relaxed tolerance is selected. Retain only if core conformance
and actual derivative/consumer costs improve without a material regression.

## Ready P3-F Choice

Select twist for P3-F, not endpoint-map Q. The named diagnostic is
`Tw = integral [curl(B) dot B / (4*pi*|B|^2)] ds` over the accepted traced segment,
with positive arclength ds; reversing trace direction changes the path but does
not multiply this integrand by the direction sign. Use the extended-input
centered derivative already validated in D, prepared locally for pool misses
rather than requiring a global D pass. Independently cache the one-halo curl
group beside the two-halo B group and publish both before readonly worker use.
Retained resident B/curl groups can likewise be reused without concatenation.

Evaluate the integrand at the four RK stages and accumulate its RK4 quadrature
only on an accepted step. This is a composed stage diagnostic, not a derivative
estimated from saved points. Validate a helical affine B field against its known
constant twist density and independent quadrature, plus serial/parallel/bounded
agreement. Record derivative, trajectory and quadrature errors separately.
Optional trajectories store accepted points only, with stable seed association
and explicit output admission/streaming; diagnostics-only remains the default.
Selected-seed retracing is an explicit function call with supplied integration
parameters, never automatic replay. Q remains deferred until a chosen method
and actual endpoint/auxiliary-dynamics acceptance are supplied; accepted-prefix
Tw does not certify full-boundary line twist or exact footpoints.

P3-F implementation evaluates twist density as ordered sums of
`(curl_component/norm)*(B_component/norm)`, divided by `4*pi`, avoiding an
unnecessary norm square. A nonfinite diagnostic has its own terminal status;
no current trial contributes to accepted twist or trajectories. The helical
fixed profile admits <2e-10 twist error at h=.01/300 steps and requires >10x
improvement on halving h; the independent augmented RK4 conformance tolerance
is 2e-12. Owned seed selectors are frozen in pools; close drops borrowed source
references without closing caller-owned file descriptors. Companion views share
primary borrowing expiry. Retracing uses bounded numeric selection arrays and
admits the still-live prior result instead of creating an uncounted Python ID map.

P3-F WENO profile: the P2 long 32-seed/128-step request, 256-slot B plus separate
curl companion, one/four workers, summaries versus optional accepted trajectories,
first-use and one warmup/three repetitions. Independently compare the first eight
seeds against scalar augmented RK4 on the same declared prepared fields. Select
three diagnostic IDs and explicitly retrace with points, checking identity and
results. No old code provides a matching twist consumer; ordinary F is a measured
incremental-cost control, not an equal-work speed comparator. Record companion
fills, retained/copy/trajectory memory and source startup. Also check the affected
ordinary 2048x600 ready-data control after extending the hot kernel.

## P3-L Initial Scalar Geometry Scope (Historical)

At the initial scalar checkpoint, L1 physical response/EOS/instrument units
were pending the user question. The new explicitly delegated thermal selection
and evidence below supersede that question; this section records the old scope.
Implement the independent geometric consumer for an explicitly supplied scalar
emissivity/field; do not invent thermal physics or certify P3-L's response gate.
Its numerical strategy is trilinear interpolation of the supplied prepared scalar
followed by midpoint quadrature within exact ray/leaf intervals. Precomputing a
nonlinear response then interpolating is distinct from evaluating a response on
interpolated rho/T; the latter is not silently supplied by this scalar boundary.

Observation uses a Plane of ray origins, a normalized common direction and
per-pixel near/far arclength bounds. Clip each ray to the physical domain, traverse
leaf intervals without covered parents or ghost path length, and subdivide each
interval to at most `step_fraction * min(local spacing)` (default .5). Empty rays
complete with zero integral/depth; nonfinite fields, missing coverage, geometry
failure or a sample limit give explicit invalid pixels, not shortened valid images.
The interval owner is the leaf immediately after entry in the ray direction;
parallel rays on an upper half-open face are empty. Face-crossing probes do not
add integration length. A pixel owns its ordered sum, private continuation and
output, so one/four workers retain the same per-ray arithmetic.

Use 2D image tiles for locality and bounded pool misses. Scalar input preparation
retains the shared two-layer minimum; a valid one-halo derived scalar can also
be consumed. Full image output and input/worker/tile resources are admitted.
Analytic constant and affine supplied scalars verify view/depth/weights, mixed-AMR
coverage and convergence; compare serial/parallel and resident/bounded results.
A real WENO scalar-column case may assess actual geometry/data costs, but does
not close the still-unspecified thermal/response physics requirement.

P3-L quadrature probe: trilinear interpolation restricted to a ray is cubic
between native cell-center planes. Splitting at those planes and applying
2-point Gauss quadrature therefore integrates the declared scalar reconstruction
exactly up to roundoff, including half-cell intervals at leaf faces. Compare
that candidate with retained midpoint quadrature (fractions .5 and .125) on
analytic fields and selected WENO scalar rays/images; allowance is two strategies,
15 minutes per probe and the existing memory envelope. Keep a brute leaf/knots
reference outside the kernel. Compare both accuracy and total consumer costs;
these are distinct quadrature strategies, not bitwise interchangeable results.
No new response/EOS meaning or global unstructured mesh is introduced.

Large/canceling ray coordinates also require checked local stencil addressing.
Shared compiled interpolation now checks finite local coordinates against actual
buffer extents before integer conversion/access. Unrepresentable samples have
explicit F/LOS status and invalid point results; they are not preparation misses.
Retain these dynamic guards even when metadata is already validated.

P3-L geometry correction: an oblique manufactured view exposed seven pixels
whose final positive interval was only 1--2 ulps long. Reconstructing and nudging
world coordinates crossed an outer face early, producing GEOMETRY_FAILURE.
Replace probe-based interval ownership with root/tree comparisons against
ray/face intersection times. Directional ties choose the interval immediately
after entry; no interval is dropped or tolerance widened. Both quadrature
strategies now pass the independent full leaf-intersection reference. This is
a geometry/addressing correction; F's point-ownership convention is unchanged.

Frozen WENO scalar probe: the file contains rho/m1/m2/m3/b1/b2/b3 but no energy
or temperature field. Use stored rho in code units solely for a column diagnostic
and LOS consumer assessment, not a selected thermal response. Compare 32x32
full-box orthographic axis and oblique views, resident midpoint (.5/.125) and
Gauss2, one/four workers, plus matching 512-slot bounded Gauss2. Use independent
leaf/knot scalar references on eight spread nonempty pixels per view. Resident
strategies have one warmup/three repetitions; bounded first/second passes are
descriptive complete costs. Record reconstruction quadrature error and all field,
worker/image resources. The missing temperature/response gate remains open.

## P4 Source And Packaging Selection

Select a distinct v5 ordinary-field reader for 3D records with staggered tails.
Keep DAT-003's factory rejection unchanged. Reuse its selector/coalescing/endian/
copy/failure implementation; the new profile validates an ordinary prefix plus
three float64 tails of shape `(stored_shape + 1)`, matching the inspected 3D
reader convention and WENO record sizes. Never expose or compute CT/staggered
values. Record/header/address preflight remains before payload mutation, and
borrowed-fd identity/position semantics are preserved. Short/malformed tails
must fail instead of being ignored. New tests target this added record meaning;
reuse existing DAT-003 checks for unchanged transfers.

For installed workflows, bundle the existing `simesh_rewrite` provider package
explicitly in the simesh distribution rather than copy its kernels into a new
namespace or hide a checkout dependency. Its original build uses only the
language-level directive: preserve that configuration separately from canonical
build directives (especially cdivision), and leave its numerical APIs unchanged.
The analysis core remains provider-independent; file convenience functions lazily
load the bundled provider. Canonical Dataset/I/O/helpers remain the default and
are not retired. P4 is additive integration, not the old ledger's final cutover.

Move the thin provider assembly into `simesh.analysis.providers`, with a small
compatibility import for development drivers. Add explicit file-source and
resident-prepared entrypoints. Selected file fields are frozen as logical source
columns with recorded original field IDs; public names/indices are unambiguous.
Support interior-only native products without ghost work, alongside the confirmed
zero-or-two primary halo policy. Count reader raw/endian/record scratch and all
resident input/output/metadata before large allocation. Preserve source context
ownership and detached result lifetime; no hidden FD close or source mutation.

P4 source comparison found the intended sparse gain: direct original-file short
tracing took 0.129 s versus 5.118 s for the same result after eager input, reading
about 3 MB and using an 18.3 MB pool bound instead of a 294.5 MB resident-source
pool. Full checked reading took 6.837 s before bulk preparation, so investigate
its avoidable zero-saved-ghost overhead rather than weakening record validation.
Bound one two-path probe: retain general records/copying as a selectable internal
reference, and reuse immutable zero-ghost shape/byte facts plus contiguous field
copy plans. Header/tail end, selectors, FD identity, endian and failure checks
remain binding. Compare complete selected-field reads and source-to-result costs
on B and rho requests; allowance 15 minutes, at most two paths, existing budget.

The zero-ghost/packed-field probe preserved all bits and reduced complete-reader
CPU time about 15--19% on B/rho, but wall time improved only about 1--2%:
I/O wait dominates. Retain the small invariant/copy improvement without claiming
a material wall speedup. Close that probe. A final bounded I/O-order probe compares
128-record preflight batches with one-record batches for private dense interior
loading. Both retain per-callback DAT preflight; the new source's documented
partial-prefix behavior does not promise a fixed batch size. The question is
whether sequential header/payload access removes the observed wait. Two batch
sizes, B/rho requests, <=15 minutes, same byte/error semantics and memory budget.

## P4 Feasible Scale Profile

The available Cartesian real input is still ~1 GB. An additional 3.6 MB spherical
reference was found under reference/; it cannot certify Cartesian thermal or 2D
analysis. No 10--20 GB input fixture is available in the repository.

Run actual one-million-seed (1000x1000 section) summary tracing on WENO, 32 fixed
steps, one/four workers, batch 16,384, with complete array equivalence and actual
termination counts. Launch at half the finest z-cell spacing above the lower
face; step is one quarter of the finest spacing. This tests that declared scale,
not long complete field lines. Admit both comparison results explicitly under
2 GiB and save only the selected four-worker summary (~96 MB).

Also execute a 1000^3 uniform sampling stream from the same prepared fields,
consuming each z slab into a descriptive checksum. It delivers 24 GB of float64
values cumulatively without allocating or saving that full cube. Record every
requested point's validity, first slab/total costs and peak active slab storage.
This certifies larger-than-RAM output delivery, not larger-than-RAM input I/O.
The iterator reserves its current and one normally retained prior slab; additional
caller retention is explicit. These are selected large acceptance runs, not
ordinary tests or discretionary implementation shootouts.

Retain one-record order for private full-domain loading: B median wall time
fell 7.495 -> 5.012 s; rho 4.240 -> 4.042 s. CPU cost rises (about .63 -> .92 s
for B and .36 -> .73 s for rho), an explicit I/O-locality trade-off. This does
not change the checked reader's per-call guarantees; bounded halo reads retain
their admitted batch behavior. Keep the raw callback's batch-size option and
reopen on CPU/cache-dominated evidence rather than hardcode WENO field IDs.

The first clean-source validation copied stale egg-info containing absolute
paths from the checkout. Exclude generated egg-info along with generated C and
extensions, and explicitly use each source tree's Cython include path. This is
a build-harness/source-discovery correction, not a numerical change. Inspection
also found the existing clean helper recursed through the entire repository,
including .venv. It now removes only generated files corresponding to package
.pyx sources and known build metadata; make clean cleans without rebuilding.
A focused filesystem test protects the virtualenv and unrelated reference/results.

Clean wheel validation now passes from a source copy with no generated extensions
or egg-info. The wheel is ~9.35 MB; an isolated `python -S` process loaded both
simesh and its provider from the extracted wheel and completed file write/read,
bounded tracing and detached resident sampling without editable path hooks.
Retained 2D roundtrip/bilinear and periodic-metadata/VTK value workflows pass.
Canonical forest source explicitly initializes periodic connectivity off; neither
this work nor header preservation establishes periodic ghost computation. New
native sources reject that flag rather than silently using nonperiodic halos.

## 2026-09-08 Thermal / E1 / E5 Round

The active user instruction delegates the first physical model and routine
choices. Retain the existing architecture/process. AIA171 selects the pinned
MPI-AMRVAC numerical response with log-log interpolation, explicit cgs density
and path-length scales, fully ionized H/He (n_He/n_H=0.1), and n_e squared.
Expose the upstream eq_state_units=True hydrogen-density convention separately;
its brightness differs by (1+2a)^2. The historical table's calibration generation
settings are not identified in its source. No current-instrument/observational
accuracy claim follows. Missing T requires explicitly labelled external prepared
K values or an isothermal scalar. Physical snapshot validation remains open.

Compare pointwise response on prepared thermodynamic nodes before interpolation
against interpolation of n,T before response. This keeps nodal support identical;
it does not silently replace thermodynamic halo transfer with emissivity transfer.
The nonlinear path uses knot-split composite Gauss2 with refinement; scalar Gauss2
exactness certifies only the node-emissivity interpolant. Core acceptance includes
independent tabular power-law normalization and manufactured continuum quadrature;
record observed errors without inventing an observational SLA. WENO profile: full
domain 8x8 axis/oblique images, actual rho, declared demonstration density/length
scales, manufactured 0.45--1.65 MK z sinusoid; subdivisions 1,4,16 (then 64 to
quantify reference residual) and node-response
order. One run per expensive variant is descriptive; no speedup inferred from
unmatched quadrature or Python versus compiled backends.

E1 bounded probe: native leaves, aligned 2x1x1 and 2x2x2 filled groups with native
fallback. Scan every WENO leaf for exact coverage. Quantify merged fraction,
retained halos, source-to-brick mapping and sparse expansion. Measure selected
preparation/packing and complete sparse F, dense D and full-depth LOS request
with identical per-consumer arithmetic. Prototype packing may reuse native fills;
report this amplification explicitly, never call theoretical compact halos saved
preparation. Keep native direct consumer as a separately identified comparator.

E5 bounded probe: persist checked source-leaf support, same/fine/coarse transfer
boxes and physical widening records for the selected immutable mesh and two-halo
continuous strategy. Local support ordinals are not runtime cache slots. Bind new
field IDs and private arrays at each execution; restriction retains ordered 1/8
averages and prolongation +/-1/4 geometry, while minmod slopes remain numerical
execution. Probe at most unretained versus retained actions, plan budget 128 MiB,
selected sparse and dense/ray geometry, one warmup/three repetitions. Include
construction, Python/array plan storage, fresh reads, scratch, output and complete
consumption. Rebuild on geometry, source-cell order, width, boundary/transfer rule,
selection or support partition changes; value/field changes do not themselves
invalidate geometric facts. Runtime ghost caches and parallel scheduling belong
to the main session. No automatic plan retention or global cache is introduced.

### Thermal And E1 Outcomes

Thermal: adopt the pinned AIA171 API, explicit electron/H-proxy distinction and
thermodynamics-first LOS with a default four subdivisions. Manufactured continuum
and actual WENO rho/geometry with labelled manufactured T verify the chain;
[thermal evidence](evidence/p3-l-thermal.md) records remaining calibration/physical
input gaps. A 64-subdivision follow-up quantified the /16 reference's residual,
and separate retained-emissivity repeats established its request-specific view
amortization. No scalar integral exactness was transferred to nonlinear response.

E1: defer default rebricking after the bounded probe. WENO supports 96.67% paired
and 93.15% octet leaves; octets reduce all-domain padding by 39.24%, but sparse
primary selection expands 4.38x and compact storage for that request grows 2.31x.
Dense compact storage falls to 63.38% of native requested storage; current packing
still prepares expanded native halos. Full mapped F and dense D/LOS results
conform, with no mapped-consumer throughput advantage in the prototype. Keep the
exact geometry mapping outside production identity and reopen only with direct
brick-boundary preparation/native consumers and complete-cost evidence.
[Rebricking evidence](evidence/rebricking.md) owns details and reproduction.

E5: adopt explicit optional selected geometric plans. Source reads and every
prepared/consumer value match ordinary preparation; repeated composed F and D/LOS
costs improve 2.56x and 6.67x in the declared WENO profile. Sparse/dense plans retain
0.64/52.18 MB and construction amortizes at the second execution. Keep unretained
one-shot preparation, and defer automatic retention/full-domain/fused execution.
[Geometry-plan evidence](evidence/geometry-plans.md) owns lifetime, invalidation,
complete costs and the runtime handoff boundary. The 128 MiB retained-plan build
allowance additionally needs one bounded prospective chunk/accounting temporary
at peak; it is not a hard RSS limit.


## Thermal LOS Traversal / Parallel Follow-up

Authorized after `a61a0ac`: research ray-tracing libraries/algorithms, identify
thermal hotspots and implement a bounded measured improvement. Preserve the
historical response, H/He normalization, thermodynamics-first nodal meaning,
knot-split composite Gauss2 and per-pixel complete/failure semantics. No opacity
termination, color transfer function, stochastic scattering or unconstrained
reconstruction change is an optimization of this optically thin integral.

Candidate: replace Python all-leaf intersections and point-array allocation with
existing native AMR tree interval ownership, incremental cell-center knot
traversal and compiled interpolation/response/accumulation. Reuse native geometry
through a small Cython declaration boundary; do not fork a general locator or
modify main-session numerical caches. Precompute only response-table logarithms
and interpolation slopes; no approximation of R(T) or reduced quadrature.
Then dispatch independent complete rays to 1/2/4 GIL-free calls with readonly
shared state and disjoint image writes. Default workers remains one.

Acceptance before implementation: manufactured constant/affine thermal cases,
AMR faces/grazing/negative directions, variable near/far, incomplete coverage,
invalid values and limits; optimized/reference finite agreement rtol=1e-10 and
atol=1e-10 DN/s/pixel, identical completion classifications except an evidenced
geometric bug must be fixed, and exact serial/parallel equality. Knot degeneracy
can change roundoff-size intervals/sample counts; preserve full path length and
declare any different sample-limit arithmetic. Existing continuum convergence
remains the physical reconstruction check. Profile old code separately from
wall-time comparisons. WENO uses the same explicit manufactured temperature and
unit scales, axis, prior 20-degree oblique and a more diagonal view. Matched
64x64 reference/native comparisons use one warmup/three repetitions; measure
larger native 500x500 images only after conformance and affordable throughput are
established. Do not run the old 500x500 path just to spend the earlier extrapolated
time. Include complete setup, retained memory, thread startup, outputs and actual
sample counts. This is an implementation comparison, not a new physical model.

A bounded follow-up to the first compiled candidate is justified by generated-C
inspection: each point calls the exported interpolate C-API pointer with three
memoryview descriptors passed by value. The 64x64 axis preliminary comparison
passed, but the candidate's per-sample metadata cost is avoidable. Preserve its
partial measurements, interrupt its larger run, and relocate the existing
interpolation body unchanged into a shared inline native.pxd definition. Keep
ray_owner exported only at the per-leaf boundary. This is one implementation
refinement within the same numerical strategy, not another reconstruction.
Rebuild and rerun affected checks, then complete the frozen comparison. Record
wall/CPU and major faults because initial timings show host-state sensitivity;
no claim of an external-library comparison or hard RSS/disk-cache isolation.


Thermal follow-up outcome: adopt compiled tree traversal and optional GIL-free
ray workers, retaining the Python reference and default one worker. Three 64x64
views conform (max image difference 2.73e-12 DN/s/pixel); three actual 500x500
views complete with exact 1/2/4-worker equality. Large-image medians span 17.9--29.4 s
serial, 8.4--15.6 s with two workers and 6.4--18.3 s with four. Desktop memory
pressure causes large dispersion (axis serial up to 81 s); do not promote four
workers or a fixed scaling factor as universal. Input guards protect the compiled
two-component sampler and invalid geometry remains NaN. Full source/library
selection, ranges, setup, faults, memory and labelled images are in
[thermal ray evidence](evidence/thermal-rays.md). No external dependency, hardware,
main-session runtime cache or default OpenMP policy was added.
