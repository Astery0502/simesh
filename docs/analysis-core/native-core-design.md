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
