# Application-Led Preparation And Storage Review

Status: design review, 2026-09-08. The latest user request is to reconsider the
first three storage proposals against actual applications before implementing
them: separate final storage from preparation workspace; reuse preparation facts;
and improve bounded batch placement. Prioritize preparing once for repeated use,
while preserving processing when selected fields do not fit in memory. Rebricking
(E1, the fourth proposal) is outside this follow-up.

This record narrows the [data organization proposal](data-organization.md); it
does not replace [shared validity/lifetime requirements](prepared-fields.md) or
[F/D/L result definitions](pipeline-results.md). Findings below come from source
inspection and existing evidence. No new performance experiment was run, and no
new numerical strategy, allocator or preparation backend is selected here.
Independent runtime work and its accepted/removed candidates remain owned by
[runtime execution](runtime-execution.md). The optional persistent geometry-plan implementation is now integrated; see
[plan evidence](evidence/geometry-plans.md). Reuse those implementations and
their retained outcomes when considering the application sequences below.

## The Product To Preserve

An application should be able to retain an affordable prepared field group and
use it in several consumers without reading or preparing that group again.
Applications retain the groups they need, not a mandatory session-wide bundle.
Changing seeds, planes or observation direction changes consumer work; it does
not invalidate compatible prepared values. New physical fields, changed values,
boundary rules or wider reach have their own preparation obligations.

This is already partly delivered through `PreparedFields`, `with_curl`,
`PreparedPool` and `CurlPool`. Optimize their actual use before adding another
session object or automatic dependency/cache framework. Prepared values, computed
products, geometry plans and reusable empty scratch are different reusable things.

## Concrete Application Sequences

| Sequence and delivered application | What remains useful | What still must be computed | Relevant improvement |
| --- | --- | --- | --- |
| A: open B, trace seeds, add seeds, retain curl(B), inspect several slices and run accepted-segment twist | The same two-halo B; one separately retained one-halo curl group; geometry | New paths, new plane samples and twist RK-stage quadrature | Final-product ownership and explicit derivative retention; no need for a fill-plan cache after B is ready |
| B: bounded whole-domain curl, close the input source, inspect repeated axis/oblique slices | The complete derived product, including its one valid halo, in admitted array/memmap backing | Each new plane's samples | B/support scratch can end after the global pass; preparation planning affects the pass, not subsequent slices |
| C: bounded B tracing, then further seeds in a fitting or expanding working set | Completed B blocks and an explicit curl companion when requested | New trajectories and preparations for genuinely missing/evicted owners | Batch support sharing and planning may matter; retained final blocks should be compared with raw caches under the same budget |
| D: several LOS views of the same supplied scalar, either known together or requested interactively | Prepared scalar coverage and completed output images | Each view's physical intervals and ordered pixel integrals | Resident scalar reuse when affordable; bounded tile ordering only for views already known together |
| E: a single full-domain pass or a single small query | Only requested output after completion | Initial reads, support and arithmetic | Control against speculative retention: no assumed benefit from persistent caches/plans without actual reuse |

These are compositions of current native applications, not new promises of Q,
thermal response or complete boundary footpoints. In A, adding twist after an
ordinary trace requires an explicit diagnostic trace; saved accepted points do
not supply the selected RK-stage quadrature. In B, every physical leaf is computed;
a few requested slices do not justify replacing the global calculation.

For D, future interactive views cannot be interleaved before the user supplies
them. Retaining a pool may help their overlap, but the known-multi-view scheduling
result is not evidence for arbitrary interactive latency. Spatially distant views
and divergent seeds are useful controls against assumed locality.

### An Existing Reuse Opportunity At The Call Boundary

[Tracing](../../src/simesh/analysis/field_lines.py) creates a temporary curl
companion when `trace(B, ..., twist=True)` receives plain prepared B or a plain
primary pool. Repeating that convenience call retains B but can recompute curl.
For repeated diagnostics, retain `with_curl(B)` or one `CurlPool(primary)` and
pass that composition to successive calls. The resident companion's `.curl` also
supplies subsequent slices. Calling `curl(B)` or `with_curl(B)` anew remains an
explicit new computation; there is no automatic expression-result cache.

This opportunity needs clear usage, not a new numerical kernel. Keep one-shot
calls convenient and avoid silently retaining all derived fields. Retained
trajectory/image outputs still count toward the whole sequence's memory budget.

## Proposal 1: Final Storage And Preparation Workspace

**Judgment: an application-level ownership boundary, with a concrete resident
adapter limitation. Prioritize A and resident D.**

[The resident adapter](../../src/simesh/amrvac/analysis.py) exposes the canonical
padded C buffer without repacking and anchors the whole `AMRMesh` owner in its
NumPy backing. That owner retains the coarse workspace along with values.
[AMRMesh](../../src/simesh/utils/lib/amr/mesh.pyx) frees both allocations on
destruction. This is safe ownership but retains preparation-only storage during
later readonly consumption. The independent raw input can already be released
after `open_prepared`; do not count it as a mandatory permanent duplicate.

WENO's three B components use 937,847,808 padded bytes and 277,880,832 coarse
workspace bytes. A one-halo curl adds 542,736,000 bytes. Separating the coarse
lifetime could remove about 278 decimal MB from retained storage in this profile.
That is allocation arithmetic, not a measured RSS reduction or speedup.

The desired boundary is: preparation completes; final values and required
geometry remain owned; temporary coarse/support backing can end independently.
Prefer a provider-owned completion/ownership-transfer boundary that preserves
the final allocation and canonical arithmetic. Exact C ownership mechanics need
a focused design before coding. Do not free buffers behind escaped views, expose
an unsafe manual pointer-release API, change mutable Dataset refresh semantics,
or copy the entire padded result merely to claim that workspace was released.

There are three separate memory questions:

1. Storage retained between queries: this proposal directly addresses it.
2. Peak preparation storage: releasing workspace only after completion does not
   reduce the peak while raw input, padded output and coarse data coexist.
3. Peak of the full application sequence: this may improve when later B/curl,
   worker and output allocations dominate, but must be measured across phases.

If the final group fits but initial preparation does not, bounded construction
into admitted final backing is a separate candidate. Existing `prepare` can
publish a selected product through bounded provider batches, but its provider
strategy/cost differs from canonical bulk preparation. No fast hybrid or
automatic promotion is implied. If the final values themselves do not fit, an
ownership fix alone cannot provide an all-resident solution.

In bounded C, keep a bounded preparation workspace while further misses are
expected if reuse justifies it; freeing and reallocating scratch after every
miss is not the objective. Scratch capacity does not certify valid source data.

## Proposal 2: Reusable Preparation Facts And Compact Support

**Judgment: conditional on further preparation, especially B/C/D. It does not
accelerate a ready resident group's repeated consumers.**

Keep long-lived geometric contacts and transfer templates separate from each
batch's support closure, value identity and slot binding. Plans must name all
required support before reads, including coarse slopes and physical bases.
Actual limiter/value arithmetic remains dynamic. Equal topology does not certify
equal values, source offsets or boundary/field-role compatibility.

For a bounded dense pass, stable geometric facts and compatible support can be
shared across adjacent targets even when every target is visited only once.
A persistent full per-target action cache, however, may have no repeat reuse in
that pass. Its construction/storage needs a benefit over direct compiled batch
planning. For recurring misses or compatible new field groups, reuse may avoid
more work. Count only the part of historical planning attribution actually
removed; total nested planning time is not a predicted speedup.

Keep support-only data compact and batch-scoped where possible, while completed
targets remain directly consumable. A raw interior cache trades bytes against
more completed targets and still leaves ghost arithmetic/planning. The existing
optional cache is a comparator, not a universal layer to enable. Feeding support
from already prepared interiors is also a provider-boundary candidate, not a
delivered promise of zero duplicate reads in the current RHE adapter.

Use the integrated `build_fill_plan` / `FillPlan.prepare` implementation. This review defines
its application requirement: improve actual missing preparations under bounded
storage without burdening the resident ready path or changing preflight/failure
and numerical contracts.

## Proposal 3: Batch Placement Independent Of Cache Slots

**Judgment: the durable requirement is efficient batch delivery into stable owned
destinations. Generic staging/paging is not itself the application objective.**

The provider currently receives a contiguous destination. Scattered cache slots
can split one missing-owner request into multiple provider calls, losing shared
support. A future native preparation boundary can name explicit destination
slots while sharing source closure. Small staging is an implementation alternative
only when its copies/storage beat direct placement or a larger prepared pool.
Do not compact or move borrowed values; only the coordinator changes placement
after consumers return. Page growth is not justified solely by fragmentation in
a fixed-capacity workload.

The [selected runtime outcome](runtime-execution.md#selected-cache-outcome)
does not promote generic actual-hit feedback or padded miss staging. Recorded
warm F work rose from 659 to 836--838 prepared owners. Staging reduced fragmented
provider calls to 54 but did not remove the excess owner work. The selected path
retains application-specific scheduling instead. Do not restart this experiment
or label a more precise recency policy a structural improvement without a new
decision-changing boundary or workload.

For A and resident D this mechanism should be bypassed. For B's ordered full pass,
contiguous output ranges already give a natural batching order. Its main potential
is bounded C and known-view D, where actual missing destinations are scattered.

## Next Delivery And Proportional Acceptance

Recommend the first implementation target as a **retained resident B/curl
sequence with independent preparation-workspace lifetime**, while retaining the
current canonical bulk adapter as the comparator. This is a proposed next target,
not an implementation claim or permission to resume every research route.

Before coding, settle final C-buffer ownership and whether workspace can end
without a full result copy. Then use one composed sequence from A: open B;
run two seed requests; create one retained B/curl composition; run two twist
requests and repeated axis/oblique curl slices. Separate newly integrated paths
from reused fields and record:

- Initial file/preparation time; bytes live during preparation, between requests,
  and after adding curl; full-sequence high water and separately observed RSS.
- Any full-field copy; B fill and curl evaluation counts; numerical results and
  validity against the unchanged strategy and existing references.
- Escaped view validity after producer/source release, correct eventual freeing,
  and unchanged canonical mutable workflows at the affected ownership boundary.

Use source inspection to establish straightforward unchanged formulas and
allocation shapes. Add focused ownership checks only for the implemented C
lifetime change. A one-shot control prevents extra copies/setup being hidden by
many repeats. Existing evidence is enough to defer proposals 2/3 during this
resident improvement; no broad benchmark matrix is required.

For subsequent bounded work, pick B or C/D according to the measured unresolved
bottleneck, compare equal upper budgets and identical complete products, and
include a case whose working set does not fit. Preserve full support, explicit
missing/failure states and the two-primary/one-derived halo rules. Bounded reuse
cannot promise each block is prepared only once after eviction, and larger-than-
RAM input acceptance still requires the missing real fixture. Disk-persistent
prepared caches, rebricking, alternative reconstruction and new parallel backends
are outside this review.
