# Design Status, Evidence And Open Decisions

Requirement/evidence baseline, updated 2026-09-07. [Intent](intent.md) owns
confirmed goals; [prepared fields](prepared-fields.md) owns shared semantics;
[consumer results](pipeline-results.md) owns F/D/L obligations. This page records
what is still undecided and what existing work can support. Confirmation does
not mean implementation. [Current](current.md) owns active scope/state and
[development](development.md) owns progression. The latest user instruction
governs authorization; this baseline is not a competing execution checkpoint.

## Requirement Status

| State | Authoritative location |
| --- | --- |
| Confirmed goals, priorities, scale and exclusions | [Intent](intent.md) |
| Shared data and geometric consumption requirements | [Shared spec](prepared-fields.md) |
| Required F/D/L results and scientific acceptance | [Consumer spec](pipeline-results.md) |
| Open numerical/representation/resource choices | [Decision table below](#open-decisions) |
| Candidate mechanisms and compositions | [Techniques](technique-candidates.md), [algorithmic routes](algorithmic-directions.md), [usage sequences](lifetime-sketches.md) |
| Delivered historical behavior and migration closure | [Capabilities](../../rewrite/CAPABILITIES.md), [contracts](../../rewrite/contracts/README.md), [migration](../../rewrite/SOURCE_MIGRATION.md) |

Supported-feature integration, geometry products, full diagnostic compositions
and actual parallel execution remain delivery work. They are not all prerequisites
for a first isolated consumer. Actual implementation/evidence state is recorded
in current.md; a requirement or design proposal does not establish delivery.

## Existing Work: Contract And Implementation Disposition

Retain means usable within existing scope; adapt means introduce an explicit new
boundary; reconsider means justify a choice again; reference means comparison
evidence. These dispositions do not authorize removal or mark migration complete.

| Existing asset | Preserved contract/evidence | Role and limit for the new design |
| --- | --- | --- |
| FND, Morton, FST/TOP/GEO, LOC/HLO | Exact IDs, geometry/ownership, valid regions and supported dimensions | Retain primitives; adapt representations when a consumer justifies it; geometry need not be rebuilt for every query |
| RST/LIM/PRL and local numerical kernels | Named arithmetic strategies and independent references | Reuse within scope; consumer-level interface/physical accuracy still needs evidence |
| DAT-001/002/003 and STO | Format/selection, copy, borrowed-source and failure rules | Retain selective access; adapt outer lifetime/budget and read granularity; direct staggered input is not delivered |
| RHE/RHC and padded/all-26 support | Transfer reach, synchronous borrowing, all-request preflight and failure guarantees | Retain bounded/reference execution; reconsider universal padding, repeated preparation and dense scheduling. RHC views expire at callback return |
| HPR/CQP | Invocation-local proof reuse and scoped access planning | Retain measured improvements; neither establishes persistent proof reuse or zero planning cost |
| CHS/RPS | Ordered sampling; CHS has three fields, one layer, serial/non-reentrant LRU state and borrowed source | Useful sampling/cache evidence, requiring adaptation to the shared prepared-field contract. Clear invalidates keys while retaining arrays |
| SLE/TRM | Fixed RK4, stage batches, declared integral, accepted prefixes and caller outputs | Reference stepping/delivery; not complete endpoint localization, Q/twist or parallel tracing |
| LFE | Output allocation `float64[P,3,Bx,By,Bz]` for selected primaries, writes only ROI windows, unweighted one-component sum | Reference local curl; not packed ROI output, reduction-only delivery or general derived lifecycle |
| Canonical resident mesh and derived batching | Fast bulk preparation, current public behavior and scientific comparisons | Preserve an alternative/comparator; layout/arithmetic adaptation is explicit. Adding fields can copy/rebuild storage; ghost-dependent recipes use original loaded fields |
| Canonical I/O, Dataset, writers and helpers | Buffered/mmap paths, lifecycle, roundtrip/build/helper obligations | Retain compatibility; mmap copies into a full result and is not an out-of-core proof. Helpers can be retained/adapted |
| Legacy/archive and old strategy records | Historical behavior and useful independent oracles | Reference, not automatic truth or file-by-file port targets |

## Decision-Relevant Evidence

These are historical findings, not freshly executed measurements. Detailed runs,
provenance and limitations stay in their source reports; source-level mechanism
records stay in [T01--T18](technique-candidates.md#candidate-technique-records).

| Evidence | Established finding | Design implication and limit |
| --- | --- | --- |
| [WENO full-domain control](../../rewrite/evidence/M1-WENO-REFERENCE-COMPARISON.md#full-domain-same-width-control) | All 22,614 leaves, three fields, two layers: resident canonical refresh median 0.309 s; bounded RHC 38.202 s including reads/copies; separate action preflight 23.345 s | Preserve the resident comparison and investigate preparation overhead. Different I/O/output envelopes and a first-sample outlier prevent an isolated kernel-speedup claim; nested times are not additive |
| [Thin query](../../rewrite/evidence/M1-WENO-ASSESSMENT.md) | 2.996 MB selected values required 116.048 MB reads at capacity 57; HPR gives little benefit in that all-SAME case | Separate mathematical support from block/read amplification; a small result does not imply small work |
| [Sampling/cache controls](../../rewrite/evidence/M1-WENO-REFERENCE-COMPARISON.md#sampling-and-cache-feasibility) | Fitting completed-neighborhood caches avoid warm reads/fills; smaller capacities thrash; resident warm sampling remains cheaper in its own prepared state | Retained artifact and working-set fit matter. Include initial preparation and cache copies, not hit rate alone |
| [Short trajectories](../../rewrite/evidence/M1-WENO-REFERENCE-COMPARISON.md#trajectory-costs) | Eight seeds/eight steps show a strong cache-capacity effect; warm dynamic checks/sampling boundaries outweigh RK arithmetic | Useful control for organization costs, not long-path, million-seed or parallel proof |
| [HPR/CQP round](../../rewrite/evidence/M1-OPTIMIZATION-ROUND.md) | Owned preflight and query planning improve selected compositions while preserving existing numerical rules | Reuse proven facts within their owned lifetime; gains are workload-dependent |
| [Native selective reads](../../rewrite/evidence/NATIVE-SELECTIVE-AMRVAC-READ.md) | Supported exact selectors and native source access | Useful adapter evidence; outer regional-product and non-leaf selection contracts still need composition |

The WENO comparison uses a bridge of ordinary fields from records with staggered
tails and reports its construction cost. Application-cleared/warm state is not
controlled cold disk. Cross-implementation comparisons record roundoff differences;
they do not establish bitwise interchangeability. The profile is not an actual
larger-than-RAM, physical-accuracy or new operating-scale proof. Existing test
counts in these reports remain historical.

## Common Contract Baseline

The definitions previously repeated here now live in the shared spec:
[selection and field meaning](prepared-fields.md#request-and-physical-selection),
[reach/validity](prepared-fields.md#input-reach-and-remaining-validity),
[ownership/delivery](prepared-fields.md#explicit-retention-versus-internal-caching),
[reuse](prepared-fields.md#reuse-and-invalidation), and
[resources/parallel ownership](prepared-fields.md#resources-and-parallel-ownership).
This section preserves navigation without a competing requirement list.

## Requirements, Responsible Capabilities And Gaps

R labels are retained traceability rows, not new capabilities or an implementation
queue. Existing capability IDs identify evidence families, not required modules.
Each selected design names which affected rows it delivers, protects or defers.

| Row | Requirement definition | Existing evidence/families | Open decision |
| --- | --- | --- | --- |
| R1 | [Native regional selection](prepared-fields.md#request-and-physical-selection), U01 | DAT/STO, FST/GEO, native-read evidence | S1; parent-to-leaf and outer source/product boundary |
| R2 | [Local/global preparation](prepared-fields.md#preparation-and-consumption-requirements), U02 | RHE/RHC, HPR, resident mesh, WENO | S2/S3; efficient preparation under actual geometric demand |
| R3 | [Task/retained regional fields](prepared-fields.md#user-model-region-fields-and-two-lifetimes) | CHS, canonical lifecycle; narrower than the new product | S4; minimum retention/delivery protocol |
| R4 | [F results](pipeline-results.md#f-magnetic-lines-and-along-line-diagnostics), U03 | LOC/SAM/CHS, SLE/TRM | F1 and S5; diagnostic state and independent-seed execution |
| R5 | [D results](pipeline-results.md#d-whole-domain-currentgradient-then-slices), U02/U04--U06 | OPR/LFE, resident derivative kernels | D1; full-domain derived lifecycle and slice meaning |
| R6 | [Complete resources](performance.md#complete-workflow-resources) | WSP, current executors/outputs | S4/S6; caller outputs, worker peaks and large delivery |
| R7 | [Replaceable compute](intent.md#compute-portability) and parallel ownership | FND/kernels, canonical OpenMP | S5/S6; current CHS remains serial |
| R8 | [Supported-feature continuity](intent.md#scope-and-migration), U11/U12 | Canonical APIs, DIM-001, migration ledger | Existing migration gates; no closure inferred from preserving code |
| R9 | [L results](pipeline-results.md#l-full-domain-los-integration-of-a-local-response), U07 | Scalar LOS plus [historical AIA171 delivery](evidence/p3-l-thermal.md); actual snapshot thermodynamics unverified | L1/S5; view/depth/AMR coverage and accumulation |
| R10 | [Additional breadth](workflows.md#concrete-workflow-catalog), U08--U12 | Workflow/source references | X1; candidates and retained uses retain distinct scope |
| R11 | [Shared data organization](prepared-fields.md#consumption-geometry-and-access-patterns), F/D/L | Prepared-data requirements and T03--T12 | S1--S5; geometric access, ghost data, local addressing and usable views together |

## Open Decisions

The rows below are the decision inventory. Active resolutions and remaining
gaps are in [native-core-design.md](native-core-design.md) and [current.md](current.md);
the initial structural S1--S5 selections and first F profile now have executable
evidence. Listing a later technique does not promote it. Application rows own
their numerical questions; S rows own the shared boundary.

[Data organization](data-organization.md) provides a concrete recommended
resolution of the structural parts of S1--S5, based on the existing two designs
and WENO evidence. It recommends the initial layout and ownership arrangement;
these rows still track formal adoption, numerical details and implementation
contracts. A proposal does not change historical completion or prove new performance.
The retained [algorithmic routes](algorithmic-directions.md) also constrain the
decision: resolve E1's source-leaf versus compute-brick identity before S1/S3 are
frozen; use E5 for preparation design. E2/E3 reconstruction and E4 hardware remain
separately selected strategies, not automatic baseline substitutions.

| ID | Decision to resolve | Evidence or comparison that can resolve it | Affected work |
| --- | --- | --- | --- |
| <a id="s1"></a>S1 | Concrete regional request and geometry/access boundary: leaf/field identity, parent selection, local addressing and query coverage representation | Trace one evolving RK sample request, one regular region and one view-defined ray through the shared semantic model | R1/R11; selecting any prepared view |
| <a id="s2"></a>S2 | Transfer strategy and support representation on the target grid; precise closure versus full-block preparation under declared centering/boundaries | Existing transfer contracts plus backward/forward reach for the selected stencil and sampler; preserve slope/base dependencies | R2/R5/R11; preparing scientifically valid data |
| <a id="s3"></a>S3 | Physical buffer/view layout and geometry/neighbor indexing: padded blocks, separate halos or prepared regions; component grouping and slot mapping | Compare complete preparation, lookup, gather/copy and immediate consumption using the three geometries; WENO constrains claims | R2/R11; memory organization and consumer cost |
| <a id="s4"></a>S4 | Minimum backing, retention/release and delivery protocol; affordable resident/bounded use and useful retained artifact | One concrete usage sequence, active-buffer ownership and complete transient/output accounting; [sketches](lifetime-sketches.md) provide alternatives | R3/R6; retained products and large output |
| <a id="s5"></a>S5 | Single-node work ownership, preparation misses and read publication: worker-local preparation versus shared completed data; output merge/order | Serial reference and explicit seed/region/ray ownership, shared reads, private scratch and total worker budget | R4/R7/R9/R11; valid parallel support |
| <a id="s6"></a>S6 | Selected-machine budget, finite accuracy/latency targets, fixtures and serial/thread/scale evidence | Freeze an honest profile before optimization; disclose unavailable target-scale evidence | R6/R7 and acceptance of every selected result |
| <a id="f1"></a>F1 | Launch/seed-spacing/units and null rules, endpoint localization including tangential boundary encounters, maximum steps/length, tangent/integrand/quadrature, diagnostic state/error and any chosen trajectory delivery; select a Q method only for its delivery | Boundary/interface trace and independent trajectory/diagnostic references; long/divergent seed cases when needed | R4; useful F result and future diagnostic compatibility |
| <a id="d1"></a>D1 | Derivative stencil/current normalization; differentiate extended inputs versus transfer derived interiors; slice sample/average/intersection and subsequent access to global results | Interior/interface/boundary accuracy and a complete global-then-slice request | R5; derived validity and scientific slice meaning |
| <a id="l1"></a>L1 | Response/EOS/units/invalid or out-of-range behavior; view orientations, per-pixel depth and ray versus area average; traversal/weights, reconstruction, accumulation precision/order and empty rays | First historical AIA171/EOS/units and both reconstruction orders selected and tested in [thermal evidence](evidence/p3-l-thermal.md); current calibration and real snapshot T remain open | R9; correct global LOS image |
| <a id="x1"></a>X1 | Additional geometry/statistics/temporal/spectral/transfer features, direct ordinary-field reads in staggered records, GPU or cross-snapshot reuse when selected | A real consumer and the relevant existing scope/compatibility evidence | R8/R10; no immediate prerequisite framework |

## First Outcome And Spec Readiness

The immediate design focus is S1--S5 as needed for shared prepared data under
F/D/L consumption geometries. Specify preparation and consumption together,
including the numerical reach, usable buffers, local addressing and ownership.
Lifetimes support this concrete design; they are not a separate research gate.
The sequence and research entry points are defined in
[development.md](development.md#delivery-stages); active selection is in current.md.

After that boundary is explicit, a leading first-result candidate remains
single-node independent-seed tracing with terminal information and a simple
along-line accumulator such as length, including parallel execution. Define
length's measure/evaluation. Diagnostics-only is useful initially; optional
trajectory delivery has its own contract. Accommodate diagnostic derivative
requests and coupled state without requiring full Q/twist delivery first.
This is a recommendation, not an activated implementation group.

Before implementation, resolve the applicable F1/S6 details, serial reference,
parallel preparation/ownership and evidence/resource plan through the
[decision workflow](README.md#how-a-decision-advances). D/L boundary needs must
be protected; their full implementations need not precede an isolated F result.
Use practical initial seed sizes and suitable resources for later scale claims.

Propose an experiment only if existing evidence leaves a decision-changing gap,
such as one interface trace, one mixed-level ray or one allocation sequence.
Use the [exploration rule](development.md#how-exploration-enters-development) to
bound it and carry its conclusion forward. Earlier repeated-sampling phase advice
and mandatory joint-line retention were superseded; [reorganization](reorganization.md)
records the scope of that editorial resolution.
