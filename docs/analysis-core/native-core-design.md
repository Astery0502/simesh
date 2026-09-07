# Native Analysis Core: Selected Development Design

Selected 2026-09-08 for P0--P4. Requirements remain in [prepared-fields](prepared-fields.md)
and [pipeline-results](pipeline-results.md). This is the implementation decision
record. Selection is not a completion claim.

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
| E1 | Defer merged bricks. Keep source/slot identity separate. Bigger bricks reduce padding but amplify sparse requests and need coordinate/interface proof; current evidence does not quantify WENO merging. | Full F/D costs identify padding as limiting; a bounded geometry probe finds useful filled same-level bricks. |
| E2 | Defer basis reconstruction, which changes interpolated/derived meaning and is unnecessary for the confirmed baseline. | A named consumer needs continuity/analytic reconstruction derivatives with applicable acceptance. |
| E3 | Defer gridlets/dual mesh. Sampled slices and rays can use native prepared groups. | Native intersections/isosurface seams justify explicit interface topology. |
| E4 | Defer RT hardware: no selected RTX/OptiX platform on this CPU host. | Suitable resources and measured location bottleneck. |
| E5 | Adopt geometry/value/slot lifetime separation and direct compiled consumption; bootstrap with RHE, then assess retained fill facts/fused batches if preparation dominates. | Equivalent support/arithmetic/failures, bounded plan bytes and improved complete consumer costs. |

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
