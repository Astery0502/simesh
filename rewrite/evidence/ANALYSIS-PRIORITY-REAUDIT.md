# Analysis Priority Reorientation Audit

## Executive Conclusion

The completed rewrite has the right numerical and interoperability core for
analysis.  Its flat forest, selected geometry and relation records, canonical
block workspace, explicit access reach, functional block adapters, and
separated halo value rules should be retained.  Existing evidence already
shows bounded payload flow, selected-field transfers, caller-owned artifacts,
and allocation-free hot kernels; no production code or stable numerical
contract needs changing in this audit.

Two boundaries must be reopened before their next consumers:

1. `STO-004` admits only a dense ascending primary prefix.  A selected physical
   region or sparse leaf set must instead drive support union and reads without
   processing intervening leaves.  Preserve the dense operation for M0/full
   traversal and add an explicit selected-primary refined planner.
2. `SAM-002`/`SAM-003` combine uniform-output ownership traversal with sample
   arithmetic.  Preserve their wrappers and exact zero/trilinear rules, but do
   not extend that execution shape into refined sampling.  Refined sampling
   must expose point ownership/location and reusable point interpolation so a
   streamline does not traverse leaves or materialize a dense grid per sample.

`INT-001` is valid M0 evidence, not the template for analysis execution.  Its
whole-domain two-pass traversal, five eager full outputs, and per-call global
preflight are intentionally deprioritized.  The next analysis executor should
consume explicit selected primaries and an already validated metadata
lifecycle, share reads/support/halos across a compatible operator batch, and
write only requested regional outputs or reduction state.

No historical benchmark needs rerunning.  The recorded results already expose
the decision-changing costs: `INT-001` halo amplification ranges from 10.250 at
capacity 27 to 1.000 only at full capacity; `STO-004` dense WENO traversal ranges
from 10.126 to 1.886; a full REL table costs 20,578,740 bytes; and the current
canonical `.dat` reader reads every leaf and copies every selected field into a
full resident array.  Those facts are sufficient to set the next priority.

## Workload Assumptions

### Selected-region local fields

- Inputs are a physical ROI or explicit leaf selection plus a small field set.
- Directional/mixed derivatives, gradient/divergence/curl/current/vorticity,
  pointwise transforms, magnitude, and reductions are composed in batches.
- A batch shares fields, exact access reach, support closure, geometry, halo
  provision, and workspace only where numerical and boundary semantics agree.
- Uniform grids and derived block arrays are optional sinks.  A streaming
  regional reduction or selected block output must not require either.
- Metadata may live for an open immutable snapshot; payload and plans have an
  explicit query/session lifetime and remain budgeted separately.

### Streamlines and field lines

- Execution is data dependent and point oriented: locator -> last-leaf test ->
  exact neighbor transition when available -> hierarchy fallback -> bounded
  block cache/reader -> sampler -> stepper -> along-line operation ->
  termination -> reducer/integrator.
- The cache holds canonical block interiors for explicit leaf/field identities;
  cache capacity, prefetch, and eviction are execution policy, not sampling
  semantics.  Halo or stencil artifacts are shared only when a measured
  consumer justifies their separate lifetime.
- Local-field and streamline execution share forest/topology/geometry,
  selected reads, ownership rules, and interpolation arithmetic.  They do not
  share traversal, scheduling, output materialization, or cache policy.

## Completed-Boundary Classification

Each row has exactly one audit classification.  “Reopen” means preserve the
validated numerical behavior and change or add the consumer-facing boundary;
it is not permission to edit a stable contract without independent review.

| Completed boundary | Classification | Analysis finding |
| --- | --- | --- |
| `FND-001`, `FND-002` layout, ownership, valid regions, access reach | retain unchanged | Slot/field/x/y/z payloads, explicit IDs, half-open regions, and tight asymmetric reach are the needed interchange and planning vocabulary. |
| `MOR-001`, `FST-001`, `FST-002` Morton and flat forest lifecycle | retain unchanged | Flat parent/child/root/leaf arrays support resident metadata, hierarchy descent, and one validation per unchanged artifact without pointer-owned forest state. |
| `TOP-001`, `TOP-002`, `BAL-001` contact and admissibility semantics | retain unchanged | On-demand contacts avoid a mandatory global face cache; balance is correctly global metadata validation, not per-query payload work. |
| `GEO-001`, `GEO-002` selected geometry | retain unchanged | Geometry is already selected-slot aligned and retains neither full leaf tables nor centers (`72*S` bytes for refined outputs). |
| `REL-001` selected directional relations | retain unchanged | It accepts selected leaves and direction subsets and keeps support, cache, value, and execution policy outside the relation representation. |
| `WSP-001` capacity accounting | retain unchanged | Exact workspace bytes are independent of source mapping, page cache, outputs, and backend state, which permits honest analysis memory accounting. |
| `STO-001`, `STO-003` transfers and functional adapters | retain unchanged | Explicit block/field/region selectors and coarse callbacks are the correct native `.dat`, resident, mapped, and cached substitution seam. |
| `PRI-001` dense primary prefixes | deprioritize because it primarily serves migration parity or a simulation-style workload | Keep exact M0 traversal/reduction ordering, but do not make dense global IDs the primary-selection API for ROI analysis. |
| `FCL-001`, `HCL-001` level-1 closure policies | retain semantics and defer optimization until a real analysis consumer | Their face/full closure meanings remain useful references; a selected consumer, not synthetic planner repetition, should decide indexing or cache changes. |
| `STO-002` compatibility composition | deprioritize because it primarily serves migration parity or a simulation-style workload | Preserve wrappers; do not add refined or selected modes to the compatibility dispatcher. |
| `STO-004` refined dense-primary/support planner | reopen before its next consumer because it materially obstructs local-field or streamline analysis | Dense candidate prefixes can read/plan gaps between sparse ROI leaves.  Add an explicit sparse ascending selected-primary stream while retaining support union/order/capacity semantics. |
| `HAL-001`, `PBC-001` physical rules | retain unchanged | Explicit per-axis coordinate/value rules and deterministic multi-axis order are directly reusable by selected refined execution. |
| `HPL-001`, `HAX-001`, `HAL-002` level-1 plan/apply and fused path | retain semantics and defer optimization until a real analysis consumer | Keep HPL/HAX as the substitutable reference and HAL-002 as measured fused M0 compatibility; refined execution must use separate relation actions. |
| `RST-001`, `LIM-001`, `PRL-001` restriction/prolongation numerics | retain unchanged | Numerical meaning, reach, operation order, and caller-owned mutation are independent of selection, storage, and execution policy. |
| `RSL-001`, `TGT-001`, `RPH-001`, `SLB-001` refined slot/box/phase artifacts | retain unchanged | These small explicit artifacts compose with arbitrary selected slots and do not force global topology or payload materialization. |
| `SAM-001` exact full-grid placement | deprioritize because it primarily serves migration parity or a simulation-style workload | Retain as an exact requested-output sink; it is not an analysis intermediate. |
| `SAM-002`, `SAM-003` dense uniform samplers | reopen before its next consumer because it materially obstructs local-field or streamline analysis | Preserve exact ownership/tie and blend semantics plus existing wrappers; extract point ownership/interpolation before refined and repeated-point sampling. |
| `OPR-001`, `OPR-002` pointwise and derivative primitives | retain semantics and defer optimization until a real analysis consumer | They are correct allocation-free primitives.  A curl/current batch must first show whether multi-output fusion beats shared-workspace composition. |
| `RED-001` fixed-order streaming sum | retain semantics and defer optimization until a real analysis consumer | It already avoids output materialization and is capacity invariant.  Accuracy-oriented or parallel reductions are separate explicit strategies. |
| `INT-001` whole-domain M0 executor | deprioritize because it primarily serves migration parity or a simulation-style workload | Keep as integration/conformance evidence; do not extend its two global passes, five eager outputs, or repeated full preflight into M1 analysis. |

## Cross-Cutting Findings

### Layout, I/O, and transfer

The canonical `(slot, field, x, y, z)` workspace is suitable for block-local
operators and selected fields.  The native format has per-leaf offsets and
field-contiguous records, so a native adapter can seek only selected
leaf/field records and convert directly into a caller workspace.  The current
canonical reader is evidence for byte order and ghost stripping only: it walks
all block offsets, builds a full `(nleaf, nfield, ...)` result, transposes each
record, and copies each interior.  Reusing it would defeat selective I/O.

Selected-primary planning must precede payload reads.  Support is the union of
only the directions required by the compatible operator batch, not
unconditionally all 26 directions or a complete `3x3x3` neighborhood.  Read,
support, and output bytes must be recorded separately.  Canonical conversion is
allowed at the adapter boundary; a second payload-sized staging array is not.

### Cache and lifetime

Forest, conformance, balance, and immutable geometry inputs may be reused for
the lifetime of one unchanged snapshot.  Query plans and completed halos may be
reused only with explicit selection, field set, reach, boundary policy, and
artifact-lifetime keys.  A streamline block cache is separately byte-bounded
and keyed by leaf plus field identity.  No evidence supports a global payload,
REL, face, halo, or center cache today.

### Halo, operators, and validation

The halo decomposition is sound.  Local operator batching should union exact
`FND-002` reaches, build one selected support closure, provision one compatible
halo workspace, and execute several separately contracted kernels before the
workspace expires.  This avoids the `INT-001` pattern of a full-halo pass for a
single derivative while retaining standalone checked wrappers.

Global forest/balance/conformance validation belongs once at snapshot open or
unchanged-artifact admission.  Query preflight validates selectors, plan
compatibility, capacities, aliases, and output regions once, then calls
explicit unchecked kernels inside the validated batch.  It must not rescan the
whole forest or rebuild all relations per chunk/operator/sample.

### Execution families and future milestones

The local executor is region/batch oriented and the streamline executor is
trajectory/cache oriented.  A future 2D generalization should parameterize
active dimensions at these semantic boundaries rather than duplicate either
executor.  Periodicity enters topology transitions, support, and boundary
semantics explicitly; it is not a cache trick.  Public Dataset APIs should own
convenience/session lifetime while passing explicit artifacts to the core.
Parallel work should start across independent seeds, regions, fields, or
snapshots after serial read/transfer amplification is measured; a dependent
trajectory and fixed-order reduction are not default inner parallel targets.

## Highest-Value Reopen Recommendations

1. Add `SPR-001`, an explicit ascending selected-primary refined support planner.
   It preserves `STO-004` relation-source union, promotion, deterministic order,
   capacity, and progress semantics but accepts a caller selection rather than
   inventing a dense ascending range.  Physical ROI-to-leaf selection remains a
   separate producer.
2. At refined sampling, extract exact point ownership/location and zero/
   trilinear point interpolation behind the retained `SAM-002`/`SAM-003`
   wrappers.  Add last-leaf and exact neighbor-transition fast paths before a
   bounded reader/cache fallback.  Do not route a streamline through a dense
   uniform-grid sampler.
3. Build a new selected analysis executor rather than extending `INT-001`.
   Its validation lifetime, optional sinks, shared batch workspace, and reader
   accounting are explicit execution decisions; lower numerical contracts stay
   unchanged.

## Deferred Alternatives And Reopen Triggers

| Alternative | Why deferred | Concrete reopen trigger |
| --- | --- | --- |
| `TOP-003` six-face cache | One REL pass cannot amortize build; real faces are only 24.8% of reduced relation records and the cache retains 1,221,156 bytes on WENO. | A repeated unchanged-snapshot consumer shows lower total contact time including build and memory, with no worse selected time-to-first-result. |
| Whole-tree REL materialization | The measured table is 20,578,740 bytes; bounded sliding production already emits each row once. | Repeated compatible queries amortize construction and a profile shows relation generation dominates while the retained bytes fit the declared metadata budget. |
| Spatial hash/BVH over leaves | Flat root selection plus hierarchy descent is exact, and one ROI metadata scan has not been shown material. | Repeated ROI or locator profiles show hierarchy/scan time dominates and record enough query density to compare build, bytes, and hit latency. |
| Persistent payload or halo cache for local batches | A batch can share one bounded workspace without retained state. | Warm repeated queries over the same leaf/field/reach keys show avoided reader bytes exceed cache construction/retention cost. |
| Streamline prefetch and block cache policy | No streamline consumer yet supplies transition locality, cache-hit, or bytes/sample evidence. | The first streamline slice reports block loads, transition distribution, cache hits, eviction, bytes/sample, and a capacity knee. |
| Multi-output/fused pointwise or derivative kernels | Existing kernels are bandwidth-scale; no real curl/current batch has isolated arithmetic from read/halo cost. | A selected diagnostic profile shows kernel traffic or repeated passes are material and a fused variant improves composed runtime by at least the policy threshold without payload-sized temporaries. |
| Alternate payload layout or direct framework array backend | Canonical layout already gives selected-field block transfers and a stable adapter seam. | Native-read conversion or streamline bytes/sample dominates after selective I/O, or a framework backend validates the complete affected contracts and removes a measured transfer. |
| Compensated, pairwise, or parallel reductions | They change floating order/accuracy and are not replacements for `RED-001`. | Accuracy requirements, reduction-dominated profiles, or independent-region parallel execution justify a separately contracted result strategy. |
| Simulation-style full-domain fusion/parallelism | It ranks below avoiding analysis reads and transfer amplification. | Migration parity requires it or a shared measured bottleneck remains after selected execution is complete. |

## Recommended Next Capability Group

Use a three-member **Selected Refined Transfer Planning** group:

- `SPR-001`: explicit selected-primary refined support planning;
- `FRP-001`: Cartesian FINER restriction placement;
- `CWP-001`: Cartesian COARSER explicit-workspace placement and reach.

Composed outcome: an arbitrary sparse ascending selected-leaf batch is closed over the
minimum requested REL directions and mapped to exact SAME/FINER/COARSER
source/workspace boxes without scanning unselected primary leaves or reading
payload.  `SLB-001` supplies SAME geometry; physical widening remains in later
PBC-aware application.

SPR-001 is `composition-only`; FRP-001 and CWP-001 are `cold/control` exact
`O(R)` integer geometry with caller-owned outputs and no size-dependent
internal allocation.  Focused correctness freezes each independent reference;
the group gate measures the selected plan-to-gather composition, including
requested/primary/support counts, direction count, reader calls, useful/read
bytes, support amplification, plan/workspace bytes, and small/medium/full
selections.  It also requires one clean rewrite build, accumulated rewrite
regression, safe current comparisons, exact preservation/order checks, and
proof that neither a whole-tree REL table nor an unselected-primary payload
read is required.  No standalone FRP/CWP timing is warranted before their
value-transfer consumer.

## Minimum Vertical Slice

After selected refined halo application is available, validate one local-field
slice before broad 2D/periodic/public work:

```text
small physical ROI + Bx/By/Bz
  -> explicit selected leaves
  -> SPR selected support for the union of curl reaches
  -> native selected `.dat` reads
  -> one shared refined halo workspace
  -> three curl/current outputs and an in-ROI streaming sum
  -> selected block sink only
```

Run resident-array and native-bounded strategies against the same independent
reference.  Record metadata/open time, time to first result, requested/read/
support/output bytes, support amplification, reader calls, halo provisions,
workspace and peak RSS, cold/warm runtime, and numerical error.  Small and
medium ROIs must demonstrate that payload work scales with selection plus
support, not total leaves.  The existing staggered WENO file remains metadata
evidence; the payload slice requires a supported non-staggered refined fixture.

The first streamline slice follows as a separate executor: one and several
seeds, exact locator, zero/trilinear vector sampling through a byte-bounded
block cache, a fixed stepper and termination policy, and one line integral.
It must report last-leaf/neighbor/fallback counts, reader calls, cache behavior,
bytes per accepted point, latency, trajectory equivalence, and independent-seed
scaling.

## Explicit Non-Work

- Do not reimplement or change FND, forest, topology, balance, geometry,
  relation, transfer-numeric, physical-rule, slot, target, phase, or same-level
  contracts.
- Do not replace the canonical workspace layout, the STO functional adapter
  seam, or validated level-1 wrappers without consumer evidence.
- Do not port the current Dataset object, eager reader, global neighbor tables,
  derived-field materialization, or full-domain uniform workflow into the next
  group.
- Do not build `TOP-003`, a whole-tree REL/payload/halo cache, a generic
  execution framework, or a simulation updater now.
- Do not rerun historical kernel benchmarks or the full historical benchmark
  matrix.  The next measurements are only the selected planning/gather group
  gate and the later local-field/streamline vertical slices.
- Do not change stable numerical contracts during this reorientation.  Any
  future material ownership, representation, arithmetic, or failure change
  requires the independent contract-review workflow.
