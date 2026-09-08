# Investigation Reference: Candidate Techniques

Status: conditional mechanisms and preserved source findings. Read this reference
for a specific open decision in [baseline](baseline.md#open-decisions), after
[consumption geometry](prepared-fields.md#consumption-geometry-and-access-patterns)
and the affected result contract. T labels are local lookup keys, not modules,
priorities or required implementation stages.

The subsequently investigated [E1--E5 algorithmic routes](algorithmic-directions.md)
cover rebricking, reconstructed fields, dual meshes/gridlets, RT-core location
and retained fill execution. They supplement these mechanism records and enter
development through the [stage and promotion rules](development.md).

## How To Interpret This Inventory

- **Measured:** a historical report demonstrates the stated result in its named
  workload. Measurements have not been rerun for this reorganization.
- **Present:** inspected source supplies a mechanism, which does not by itself
  prove speed or suitability for the new contract.
- **Candidate:** an alternative or extension, including a new use of a present
  mechanism. No architecture or public API is selected by its inclusion.

A desired property such as avoiding repeated preparation can be served by
several mechanisms. Existing entrypoint contracts still govern their own
arithmetic, ownership and failures. The [shared spec](prepared-fields.md) owns
new prepared-data requirements; historical CHS needs adaptation to that scope.

## Conditions That Change The Answer

Describe the requested geometric coverage and access order, known versus
value-dependent locations, operator reach, output shape, locality/repetition,
and whether the complete working set fits memory. A small result can require
a global scan; input size alone does not choose residency. Use the
[operating envelope](intent.md#operating-envelope) without inventing latency or
budget targets. Evidence limits are collected in [baseline](baseline.md#decision-relevant-evidence).

## Scenario To Property Map

The rows group independent possibilities for shared prepared data. They are not
recipes requiring every listed technique. A suspected bottleneck remains a
hypothesis until the linked evidence supports it for the selected case.

### Requested Core Workflows

| Geometric/access condition | Property or cost to examine | Records |
| --- | --- | --- |
| Inspect/select before reading | Metadata cost, exact field/region access, read granularity and conversion | T01, T02, T17 |
| Irregular RK paths, coherent or diverging seeds | Owner/local-coordinate cost, repeated neighborhoods, misses/thrashing, stage/output state | T06, T07, T08, T09, T11, T15, T16 |
| Regular boxes/slabs/planes, known before computation | Selection/support amplification, batch preparation, derivative validity, reusable geometry | T03, T04, T05, T09, T10, T12 |
| Full-domain LOS with different view directions and pixel depths | Ray/AMR crossing geometry, global coverage, response work, image contribution/delivery | T04, T05, T10, T11, T12, T16 |
| Dense repeated use with affordable prepared fields | Bulk throughput, padding/coarse storage and setup amortization | T02, T03, T09, T16 |
| Thresholds and extracted geometry | First-scan cost, conservative pruning, partial cells/seams, global output state | T11, T13, T14 |

### Adjacent Candidates And Retained Uses

Statistics and snapshot series can share chunk reductions and compatible geometry
(T01, T04, T08, T10--T12); connected structures add global reconciliation (T14).
Uniform patches and spectral consumers need explicit conversion/transform costs
(T17). Model/potential-field comparisons may exploit mathematical structure
(T18). Mutation requires the [shared invalidation boundary](prepared-fields.md#reuse-and-invalidation).
Detailed product distinctions remain in [U08--U12](workflows.md#concrete-workflow-catalog).

## Candidate Technique Records

### T01: Metadata Before Payload, With Explicit Reuse Lifetimes

**Evidence:** Present in canonical `get_metadata` and Dataset's separate
metadata/data loading, and in rewrite's DAT index/forest binding.
[Canonical I/O](../../src/simesh/amrvac/datio.py),
[Dataset](../../src/simesh/amrvac/amrvac_dataset.py),
[rewrite index](../../rewrite/src/simesh_rewrite/amrvac_dat.py).

**Conditional judgment:** metadata-first access serves inspection and selective
requests without obligating field loading. Retaining a compatible index can
avoid reparsing during an immutable session. This is different from retaining
field values or materializing every neighbor relation.

**Limits/open question:** metadata is not constant-size and opening/binding can
include whole-forest work. Canonical field loading calls the reader, which reads
metadata again; separate public operations do not by themselves prove reuse.
Across snapshots, offsets and values may change even when geometry matches.
Which metadata costs actually recur, and which lifetime can safely reuse them,
remain workload-specific questions; no persistent index service is selected.

### T02: Selective Reads, Coalesced Reads, And Mapped Access

**Evidence:** Present canonical buffered sequential and mmap functions both
allocate a result for all leaves and selected fields; Dataset currently calls
the buffered version. Present rewrite DAT-003 uses position-independent reads,
deduplicated selectors, consecutive field runs for eligible full interiors,
and per-field byte envelopes for other boxes.
[Canonical readers](../../src/simesh/amrvac/datio.py),
[rewrite reader](../../rewrite/src/simesh_rewrite/amrvac_dat_reader.py),
[scoped read evidence](../../rewrite/evidence/NATIVE-SELECTIVE-AMRVAC-READ.md).

**Conditional judgment:** exact selective reads can save unrelated bytes;
larger/coalesced reads can save calls at the expense of extra bytes. Mapped
access is a different transfer mechanism worth retaining as a candidate, not
an automatic memory-budget solution or the canonical Dataset default.

**Limits/open question:** the current mmap function copies into a full result,
and source presence is not evidence of superior speed or complete lifecycle
correctness. Rewrite scratch/headers/layout conversion and cold storage matter.
Direct ordinary-field reads inside staggered records remain an unselected
extension; WENO's bridge adds startup work. The useful unresolved comparison is
which read granularity/mechanism serves sparse, sequential and repeated access
on the actual storage, not whether `mmap` or `pread` is universally better.

### T03: Resident Padded Fields And Materialized Connectivity

**Evidence:** Present canonical forest construction builds connectivity, and
AMRMesh retains field/geometry/coarse-work storage for bulk ghost and sampling
operations. Measured WENO resident refresh is much faster than bounded RHC for
the recorded full-domain request, with different I/O/output work included.
[Forest](../../src/simesh/utils/lib/amr/forest.pyx),
[mesh](../../src/simesh/utils/lib/amr/mesh.pyx),
[comparison](../../rewrite/evidence/M1-WENO-REFERENCE-COMPARISON.md#full-domain-same-width-control).

**Conditional judgment:** when the selected fields and support fit and many
dense operations follow, retaining prepared data and relations can amortize
construction, transfer and halo work. It supplies an important baseline.

**Limits/open question:** payload, padded halos, coarse storage, derived outputs
and transient copies must all fit. This does not establish fast initial loading,
and eager setup can be wasteful for sparse one-shot requests. The evidence
measures a composition; it does not isolate connectivity as the cause of speed.
Retaining just the needed region/fields is a Candidate variation, not a selected
new resident executor. Canonical and rewrite arithmetic are not silently
interchangeable under strict existing contracts.

### T04: Capacity-Bounded Workspaces And Chunk Consumers

**Evidence:** Present rewrite WSP/STO/RHE/RHC mechanisms account for slots, close
support, reuse workspace and synchronously consume completed primaries. Measured
local/native workflows demonstrate bounded workspace and capacity trade-offs.
[Workspace](../../rewrite/src/simesh_rewrite/workspace.py),
[refined execution](../../rewrite/src/simesh_rewrite/refined_halo.py),
[local evidence](../../rewrite/evidence/M1-WENO-ASSESSMENT.md).

**Conditional judgment:** when the active field payload exceeds affordable
residency, process a useful primary group with its support and consume results
before reusing buffers. Capacity is a throughput/first-result trade-off, not a
goal to minimize independently.

**Limits/open question:** overlap can reread support across chunks; planning
and checks can dominate. Bounded field workspace does not bound metadata,
caller outputs, geometry or labels. The present RHC borrowed callback is
synchronous, not an asynchronous streaming API. Which chunk size, traversal
and support retention reduce repeated work enough remains open. No automatic
choice of streaming over a feasible resident path follows from this mechanism.

### T05: Select Output Windows And Propagate Real Support Needs

**Evidence:** Present rewrite ROI-001 produces cell-center windows; LFE computes
selected curl windows in full selected-block allocations. Access/valid-region contracts and refined support
closure already exist. Measured thin-region read amplification shows remaining
cost. General fine-grained requested-direction/support-only storage improvements
are Candidate, not delivered by the existence of ROI windows.
[ROI](../../rewrite/src/simesh_rewrite/region_selection.py),
[curl consumer](../../rewrite/src/simesh_rewrite/local_field.py),
[validity boundaries](../../rewrite/FUNCTIONAL_COMPOSITION.md#field-validity-and-halo-composition),
[thin-query evidence](../../rewrite/evidence/M1-WENO-ASSESSMENT.md).

**Conditional judgment:** a thin cut or local derived result needs only its
actual inputs and dependency support; smaller output should avoid unnecessary
full-block work where the numerical operation permits it.

**Limits/open question:** a cell-center ROI is not plane intersection or a
partial-cell clip. Output selection does not prove equally selective I/O/halos.
Direct target directions omit neither coarse slope support nor physical-base
dependencies. Composition follows the [shared reach rules](prepared-fields.md#input-reach-and-remaining-validity).
The open question is which amplification comes from required mathematics,
file/block granularity, or avoidable execution, before choosing finer storage.

### T06: Location Hints And Grouping Samples By Owner

**Evidence:** Present rewrite exact last-owner hints fall back to hierarchy
descent; point execution groups samples by owner. Cached/bounded sampling has
scoped measured evidence. Exact adjacency walking and optional spatial indexes
remain Candidate alternatives.
[Hints](../../rewrite/src/simesh_rewrite/hinted_location.py),
[point location](../../rewrite/src/simesh_rewrite/point_location.py),
[batch sampling](../../rewrite/src/simesh_rewrite/repeated_sampling.py),
[strategy evidence](../../rewrite/evidence/M1-ANALYSIS-DECISIONS.md).

**Conditional judgment:** coherent paths can avoid repeated tree descent;
many points sharing owners can share preparation. An oblique plane also offers
batch location even though it is not a trajectory.

**Limits/open question:** scattered points may offer few hits, and grouping
can cost more than it saves for tiny batches. New traversal/index construction
uses memory and may have little benefit while halo misses dominate. Grouping
must restore caller point order and preserve exact ownership at interfaces.
Reuse for slice sampling is plausible but not a delivered slice API. The
remaining question is the fraction of time actually spent locating/grouping
after field preparation has been accounted for.

### T07: Retain Completed Halo Neighborhoods

**Evidence:** Measured CHS retains three-field completed owner halos under a
byte-budgeted LRU and uses hinted sampling; CQP improves access planning for
some larger warm batches. WENO records both useful warm reuse and thrashing.
[Session](../../rewrite/src/simesh_rewrite/completed_halo_sampling.py),
[cache evidence](../../rewrite/evidence/CACHED-REFINED-VECTOR-SAMPLING.md),
[WENO cache/trajectory comparisons](../../rewrite/evidence/M1-WENO-REFERENCE-COMPARISON.md).

**Conditional judgment:** if the snapshot/field/boundary meaning is unchanged
and reused owners fit, a hit can avoid reading and completing those halos.
This targets expensive preparation, not just the raw payload transfer.

**Limits/open question:** a miss still pays support/halo work; first-use cache
population can be slower than a one-shot batch. Capacity below the working set
can repeatedly evict useful entries. CHS is not a general arbitrary-field or
derived-field cache, and lookup/checking remains work on a hit. Longer traces
and different seed orders need their own evidence. Cache usefulness depends on
avoided work and retained bytes, not hit rate alone.

### T08: Retain Raw Data, Derived Fields, Or Shared Support

**Evidence:** Present canonical Dataset retains loaded/materialized fields,
tracks original/derived columns and drops/reloads derived results. CHS is the
more specialized retained halo example in T07. A general bounded raw-support
or derived-result cache for rewrite analysis is Candidate.
[Field lifecycle](../../src/simesh/amrvac/derived_fields.py),
[columns and reload](../../src/simesh/amrvac/amrvac_dataset.py),
[reuse boundary](../../rewrite/FUNCTIONAL_COMPOSITION.md#explicit-reuse-and-sessions).

**Conditional judgment:** raw support retention could help neighboring owners
whose completed outputs differ but whose inputs overlap. Derived retention
could help repeated expensive diagnostics; cheap pointwise transforms may be
better recomputed. These are different artifacts with different savings.

**Limits/open question:** raw hits do not remove halo arithmetic/checks; derived
hits need compatible physical definitions and valid coverage. Retaining all
layers can duplicate payload and displace more useful state. Existing mutable
Dataset machinery does not establish a complete cache invalidation protocol
for future execution. The open choice is which artifact avoids the most actual
work per retained byte for the specific query sequence.

### T09: Amortize Owned Checks And Reduce Scheduling Overhead

**Evidence:** Measured HPR reuses some invocation-owned relation/alias facts
while preserving per-action geometry checks; CQP adds indexed planning for
selected cache query shapes. WENO still exposes substantial action-preflight
and warm dynamic-check costs.
[Owned preflight](../../rewrite/src/simesh_rewrite/refined_halo.py),
[query planning](../../rewrite/src/simesh_rewrite/completed_halo_sampling.py),
[optimization evidence](../../rewrite/evidence/M1-OPTIMIZATION-ROUND.md),
[remaining costs](../../rewrite/evidence/M1-WENO-REFERENCE-COMPARISON.md).

**Conditional judgment:** many small actions on privately owned or immutable
state can share established facts rather than repeatedly crossing full checked
boundaries. Compiled/batched preparation and longer-lived prepared plans are
Candidate ways to investigate remaining cost.

**Limits/open question:** present HPR proofs live within one invocation; they
do not establish persistent plan validity across mutation or callbacks. Some
checks depend on each new sample or source slot. Existing failure-before-I/O,
alias and numerical contracts remain binding. Identify what is invariant and
which lifetime owns it before choosing a technique; "unchecked everywhere"
does not express the required behavior.

### T10: Named Dependencies, Batched Derivatives And Fused Arithmetic

**Evidence:** Present canonical derived registration maps fields and derivative
terms into a Cython batch, with retained valid-layer metadata. It still loops
over derivative terms and appends/rebuilds field storage; it is not universal
common-subexpression elimination. Rewrite has a concrete fused curl consumer
and primitive operations.
[Derived implementation](../../src/simesh/amrvac/derived_fields.py),
[compiled derivatives](../../src/simesh/utils/lib/amr/mesh.pyx),
[selected curl](../../rewrite/src/simesh_rewrite/local_field.py).

**Conditional judgment:** compatible operators can share source/halo preparation;
a concrete fused operation can avoid intermediate arrays and dispatch. Known
field dependencies can prevent loading unrelated inputs. These benefits can be
examined separately without designing a generic expression engine.

**Limits/open question:** registering names does not implement arbitrary
dependency planning; canonical ghost-dependent recipes/derivatives require
original loaded fields. Appending columns copies arrays and can rebuild the
mesh. Different stencils, physical definitions or validity needs limit sharing.
Fusion must preserve contracted arithmetic or use a separately reviewed
strategy. Determine whether reads, preparation, temporaries or arithmetic are
the repeated cost before choosing which form of sharing helps.

### T11: Consume Results Incrementally Or Keep Only Reductions

**Evidence:** Present rewrite has a fixed-order scalar accumulator and
synchronous completed-primary consumers. LFE currently returns compact curl
values plus a sum; its required allocation has full spatial blocks for selected
primaries, with only selected ROI windows written, rather than packed ROI cells.
The sum is one unweighted component.
SLE currently requires
caller trajectory/integral arrays. General accumulator-only LFE, segmented
curves, image tiles and streamed surface output are Candidate extensions.
[Reductions](../../rewrite/src/simesh_rewrite/reductions.py),
[local outputs](../../rewrite/src/simesh_rewrite/local_field.py),
[trajectory outputs](../../rewrite/src/simesh_rewrite/field_lines.py).

**Conditional judgment:** a requested statistic can discard intermediate field
values, and a large result can be delivered in pieces when the consumer accepts
them. This controls a cost that input chunking alone cannot bound.

**Limits/open question:** preserve required reduction ordering/weighting; a
generic field sum is not a volume integral. Surface stitching and component
labels may require global retained state. Data-dependent result sizes do not
justify a mandatory full precount scan. Which result must be retained, which
can be consumed, and what completion/partial-output semantics are acceptable
remain questions for each product, not one universal streaming interface.

### T12: Reuse Geometry Separately From Sampled Attributes

**Evidence:** Present rewrite separates location from sampling and returns
trajectory coordinates; canonical uniform sampling accepts requested geometry.
Persistent slice/ray intersection maps and reusable extracted surface geometry
are Candidate uses, not established public features.
[Location](../../rewrite/src/simesh_rewrite/point_location.py),
[sampling](../../rewrite/src/simesh_rewrite/refined_sampling.py),
[field lines](../../rewrite/src/simesh_rewrite/field_lines.py).

**Conditional judgment:** when positions or extracted geometry stay fixed,
changing the attribute can reuse geometry/location and fetch just the new
field dependencies. Moving a slice preserves less state than recoloring it.

**Limits/open question:** changing a trajectory's vector field, an isovalue or
its defining field changes geometry. Sample maps also depend on topology,
coordinates and interpolation conventions. Storing interpolation weights may
cost more than recomputing them, and derivative fields need valid support.
Decide what users actually vary, then identify the reusable portion; no generic
pipeline/result-cache object is selected.

### T13: Predicate-First Work And Conservative Field Summaries

**Evidence:** Candidate for native scalar thresholds/isosurface queries. Existing
ROI selection is geometric, not a field-value index. The [workflow catalog](workflows.md) cites
external threshold/contour uses; no implemented native range-summary mechanism
is established by the inspected canonical/rewrite paths.
[Desired products](workflows.md#u06-threshold-regions-and-isovolumes),
[existing geometric selector](../../rewrite/src/simesh_rewrite/region_selection.py).

**Conditional judgment:** evaluate the defining/predicate field before unrelated
attributes; repeated level sweeps might reuse conservative block/subtree ranges
to exclude impossible regions. Predicate masks can support reduction without
surface/volume mesh construction.

**Limits/open question:** the first value-based query may require a full scan.
An index costs construction/storage and becomes stale when fields change.
Cell-center min/max is not a conservative bound for every reconstructed or
derived field. A second pass can increase I/O if the selection is dense.
The unresolved questions are whether repetition pays for summaries and what
field definition makes exclusion valid; no index structure is selected.

### T14: AMR Geometry Extraction And Connectivity Reconciliation

**Evidence:** Candidate in simesh for contour surfaces, clipped isovolumes and
cross-block feature labels. Existing AMR contacts/halos provide ingredients,
not those products. As an external example, ParaView's `vtkAMRDualContour`
exposes merging points across blocks and ghost-copy controls, illustrating
that interfaces have explicit work beyond processing cell interiors.
[ParaView class reference](https://www.paraview.org/paraview-docs/latest/cxx/classvtkAMRDualContour.html)
(consulted 2026-09-07).

**Conditional judgment:** native extraction with explicit transition handling,
or extraction on a deliberately chosen resampled patch, are alternatives to
investigate. Whole-cell selection, partial-cell clipping, surface construction
and component labeling satisfy different products; they are not interchangeable.

**Limits/open question:** require a reconstruction/centering definition, AMR
seam ownership and appropriate accuracy before choosing an algorithm. Labels,
vertex deduplication and frontiers can grow globally even with bounded field
chunks. No external class compatibility, crack-free simesh extraction or
conservative clipped-volume implementation is claimed. What exact geometry or
statistic is needed is the next decision-changing question.

### T15: Step Control, Events And Diagnostic Evaluation Along Paths

**Evidence:** Present and scoped Measured rewrite SLE uses normalized magnetic
tangents, fixed RK4, stage batches, accepted prefixes and whole-step rejection,
with a specific oriented B-dot-dx integral. Adaptive steps, accurate event
localization, general diagnostics and endpoint-only output are Candidate.
The [F contract](pipeline-results.md#f-magnetic-lines-and-along-line-diagnostics)
owns Q/twist dependencies and coupled-state requirements; existing position/integral
state does not complete those consumers.
[Executor](../../rewrite/src/simesh_rewrite/field_lines.py),
[termination](../../rewrite/src/simesh_rewrite/field_line_termination.py),
[field-line evidence](../../rewrite/evidence/NATIVE-REFINED-FIELD-LINES.md).

**Conditional judgment:** cheap fixed stepping may serve a known sampling scale;
adaptive stepping might reduce work for a given trajectory error in a varying
field. Endpoint requirements can favor event localization. Diagnostic sampling
frequency should follow the requested quantity and quadrature/coupling needs.

**Limits/open question:** RK order alone does not guarantee composed accuracy
through AMR interpolation. Rejected trials also consume reads/preparation, and
boundary rejection is not a boundary intersection. yt documents a different
streamline approach using AMR tiling bricks; this is an external alternative,
not evidence that its representation would help simesh.
[yt streamline method](https://yt-project.org/doc/visualizing/streamlines.html)
(consulted 2026-09-07). First decide visual-path, endpoint or integral accuracy;
do not choose one stepper to satisfy every goal by name alone.

### T16: Compiled Loops, Batching And Parallel Work

**Evidence:** Present canonical uniform sampling and derivative kernels use
Cython `prange`; actual OpenMP availability depends on the build. Rewrite has
compiled numerical kernels and batched point/stage execution, while independent
seed parallelism is now a confirmed delivery target, not yet provided by the
current serial rewrite session. Its scheduling/cache mechanisms remain open;
a GPU analysis backend remains Candidate.
[Canonical kernels](../../src/simesh/utils/lib/amr/mesh.pyx),
[runtime](../../src/simesh/utils/runtime.py),
[build documentation](../cython-build.md),
[rewrite stages](../../rewrite/src/simesh_rewrite/field_lines.py).

**Conditional judgment:** compilation can reduce hot-loop/dispatch cost;
independent blocks, seeds or snapshots may provide parallel work. Batching is
useful even without threading when it shares transfers/preparation.

**Limits/open question:** small jobs, serial checks, storage bandwidth and
shared cache contention can prevent gains. Worker scratch and in-flight outputs
must fit the total budget. Reordering reductions can change numerical meaning.
GPU transfers/residency need separate evidence; a fast device kernel would not
prove fast file-to-result analysis. Which portion is independently parallel and
large enough after preparation costs is still a workload question.

### T17: Explicit Representation Conversion And Exact Placement

**Evidence:** Present canonical APIs provide native blocks, exact level-1
placement, refined uniform sampling, uniform construction, layouts and DAT/VTK
output. Rewrite provides storage adapters and native/array reference paths.
[API map](../python-api-map.md),
[layouts](../../src/simesh/amrvac/layouts.py),
[uniform operations](../../src/simesh/amrvac/amrvac_uniform.py),
[rewrite storage](../../rewrite/src/simesh_rewrite/storage.py).

**Conditional judgment:** deliver native data when the consumer accepts it;
explicitly resample a bounded patch/resolution when its next operation requires
regular arrays. Keep exact placement separate from interpolation. A conversion
may pay off across repeated downstream operations if its costs are counted.

**Limits/open question:** finer regular grids can amplify memory and alter
scientific meaning. A transpose view may still require copying for the next
kernel's layout; borrowed storage has lifetime constraints. Current level-1
VTK export is not native AMR surface export. Determine the consumer's actual
layout/precision/resolution requirement and conversion frequency before
choosing persistent transformed storage.

### T18: Exploit A Scientific Operator's Mathematical Structure

**Evidence:** Present canonical potential-field tools implement direct and FFT
convolution for the same discrete Green-kernel model; tests compare the paths.
Existing magnetic configuration helpers evaluate dipole/bipolar/flux-rope
models and uniform curl. No fresh speed or accuracy run is claimed here.
[Potential field](../../src/simesh/tools/potential_field.py),
[tests](../../tests/tools/test_potential_field.py),
[model definitions](../potential-field-tools.md),
[configurations](../../src/simesh/utils/configurations.py).

**Conditional judgment:** a specific operator can have a useful algorithmic
alternative beyond storage/caching choices. Shared geometry or bottom-field
transforms might help repeated extrapolation; tiled model evaluation might
reduce large intermediate arrays. These reuse/tiling changes are Candidate.

**Limits/open question:** present FFT calls are made per component/height; they
do not establish persistent transform reuse. Potential extrapolation remains
nonlocal in the boundary input and returns a full field box. Flux balancing,
boundary model, centering and normalization are part of the scientific meaning.
The flux-rope helper forms point-by-axis intermediates, so NumPy vectorization
alone is not a bounded-memory proof. Which mathematical structure and repeated
inputs are actually shared must be settled per operator.

## Gaps And Decision-Changing Questions

Use [S1--S6 and F1/D1/L1/X1](baseline.md#open-decisions) as the single decision
list. These records supply mechanism-specific uncertainties; they do not create
a second readiness gate. [Usage sequences](lifetime-sketches.md) illustrate when
combining selected mechanisms could help.

## Where This Step Stops

This inventory preserves evidence and conditional alternatives. Narrow the
consumer and the uncertainty that can change a choice before applying the
[decision workflow](README.md#how-a-decision-advances) and
[measurement policy](performance.md). No universal cache stack, dispatcher or
new completion claim follows from these records.
