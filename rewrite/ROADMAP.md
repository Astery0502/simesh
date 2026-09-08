# Rewrite Roadmap

This is the retained rewrite milestone/dependency map. Current project planning
is in [docs/analysis-core](../docs/analysis-core/README.md); the paired
preparation/consumption spec precedes selecting a new pipeline implementation.

This document owns stage outcomes and sequencing. CAPABILITIES owns exact member
status; CURRENT owns the active group. Each stage must have a useful composed
result and the scientific/resource acceptance in ANALYSIS_WORKLOADS.

Current selection authority is [baseline.md](../docs/analysis-core/archive/core-development-2026-09-08/baseline.md): spec convergence only,
with [three pipeline result contracts](../docs/analysis-core/archive/core-development-2026-09-08/pipeline-results.md) to close
before a proposed independent-seed parallel tracing outcome, not an active group. The old
repeated-sampling phase suggestion is superseded. F tracing/Q/twist and D global
current/gradient-to-slice have similar priority, F slightly preferred; L full-domain
LOS synthesis is confirmed. F does not require D's full-domain preparation.
The requirement/evidence map and entry conditions supersede
earlier operational priority suggestions below. M0/M1 completion and later
milestone families remain history and a dependency/compatibility horizon, not an
automatic execution queue. No implementation starts under the current task.

The 2026-09-07 intent clarification precedes selecting another implementation.
Use the [workflow catalog](../docs/analysis-core/archive/core-development-2026-09-08/workflows.md) to name the result and matching
baseline first. Fast dense/resident analysis, slices, isosurfaces and isovolumes
join the explicit product outcomes; adjacent candidates remain unscheduled.
This is an intent checkpoint, not a new execution plan or a change to completed
M0/M1 contracts.

## Established Foundation And M0

Status: complete under the original contracts.

Foundation established layout/index/ownership/valid-region conventions, access
requirements, functional storage, explicit workspaces, migration disposition,
performance protocol, and an isolated build/test path.

M0 supplies Cartesian 3D level-1, non-periodic, non-staggered topology/geometry,
physical and same-level halos, exact placement, zero/trilinear sampling,
pointwise/stencil/reduction primitives, and a bounded composition compared with
the current path. These remain references and compatibility strategies.

## M1: Cartesian 3D Refined AMR

Status: complete for its original scope with the qualified real-fixture
limitation in [M1 horizon](evidence/M1-ARCHITECTURE-HORIZON.md).

M1 includes refined reconstruction/conformance, contacts/balance/geometry,
selected support, SAME/FINER/COARSER/physical halo application,
restriction/prolongation, refined ownership/sampling, native selective v5 input,
selected curl/reduction, completed-owner caching, and fixed-step field lines.
Its original local-field and streamline priority gates are complete.

There is no direct real fixture simultaneously refined, non-staggered, native,
and Cartesian 3D. Existing evidence combines real tdm, WENO metadata, a streamed
regular-field bridge, and synthetic refined native files. Preserve that
qualification; new documentation is not fresh numerical validation.

## Post-M1 Native Derived Analysis

Status: retained planning horizon; BASELINE owns current outcome selection.
This added product scope preserves M1 completion. Earlier planning advanced a
small part of M4/M6 before broad M2/M3 generalization; that order does not select
the next group. A complete Dataset or generic derived-field framework is not a
prerequisite for a concrete consumer.

Select the next outcome from actual missing contracts and measured bottlenecks,
splitting implementation into groups of at most four under WORKFLOW. Retained
directions are:

1. **Reusable bounded analysis.** Exercise raw and derived queries repeatedly
   under an explicit complete resource budget. Use another concrete operator to
   test shared reads/halos. Choose retained artifacts and bounded output from
   measured reuse, not a mandatory cache stack.
2. **Efficient halo support and storage.** Assess requested-direction support,
   support-interior versus padded storage, shared halo provision, and plan reuse
   without losing required slope/base closure or active-dimension compatibility.
3. **Along-field diagnostics.** Share location/field preparation while tracing,
   evaluating derived quantities and directional derivatives, and integrating a
   named quantity. Provide bounded/accumulator output and separate trajectory,
   diagnostic, endpoint, and integral acceptance as appropriate.

The completed optimization round's first objective was to reduce selected
native-analysis halo preparation cost. Its
[execution plan](designs/M1-ANALYSIS-OPTIMIZATION.md#execution-order)
records the order: fixed comparison set, halo support/planning, cache scaling,
storage/output amplification, conditional compute, and round closure. These
are assessment priorities, not mandatory algorithms or predeclared capability
groups. Dependencies or blocking resource constraints may change the order with
an explicit reason. No field-specific demonstration is a
prerequisite. Operator access, produced valid regions, and downstream sampling
are general requirements applied when selecting and composing real consumers.
The former DVA-001 proposal is withdrawn. Freeze only a demonstrated missing
behavior or selected strategy; do not reopen all M1 capabilities. Existing
full-halo and strict numerical paths remain references.

Directional support, raw caches, halo storage changes, adaptive steps, and
parallel execution are candidates, not prerequisites for every outcome.
Historical reopen conditions remain in
[M1 strategy records](evidence/M1-ANALYSIS-DECISIONS.md). New query/validity/reuse
assumptions can justify reconsideration with a recorded cost model or probe.

Preserve M2's active-dimension requirements. Before changing a shared direction,
child-phase, halo, or session representation, assess its 2D implications and
activate the minimum DIM-001 prerequisite if needed. Do not entrench a new
3D-only shared representation or require all 2D implementation before a valid
3D composition.

The optimization round uses its own
[exit criteria](designs/M1-ANALYSIS-OPTIMIZATION.md#group-and-round-completion):
validated retained gains or a justified unchanged baseline, resolved direction
dispositions, no unmet hard requirements/regressions, and final composed
evidence. It does not claim all post-M1 functionality complete. The round is
closed; CURRENT now records an intent checkpoint with no implementation selected.
Future work must select a justified user outcome and schedule its dependencies;
there is no automatic transition to DIM-001/M2 during intent clarification. Product
acceptance, not optimization completion alone, closes this broader stage.

## M2: Cartesian 2D

Status: queued; DIM-001 remains proposed. Its former active group is deferred
while post-M1 analysis planning is active.

DIM-001 must precede all 2D forest, DAT, geometry, halo, and sampling consumers.
It makes `active_ndim` explicit, keeps `ndir` separate, and preserves canonical
five-dimensional singleton-z arrays. A legitimate 3D singleton axis is not 2D.

Then generalize quadtree/Morton, variable-width v5 input, active-axis geometry,
refined/physical halos, bilinear sampling, and a representative 2D operator.
Requested target directions remain distinct from extra COARSER slope/physical
base support. Preserve exact 3D wrappers and generalize proven shared concepts
rather than creating unrelated 2D copies.

Close a representative real 2D singleton-z workflow. No such fixture is currently
recorded; acquire or generate one with provenance before closure.

## M3: Periodic Cartesian Meshes

Add explicit periodic topology, support/halo transfer, refined sampling, and
stencil behavior, followed by periodic integration. Distinguish parity with
current supported behavior from new periodic capability; validate both honestly.

## M4: Scientific Breadth And Derived Lifecycle

Extend the post-M1 concrete analysis contracts to pointwise transforms,
derivative batching, AMR operators, geometry-aware sampling, reductions,
current/divergence diagnostics, and broader field-line integration.

Complete derived dependency registration, materialization, field selectors,
dropping, and ghost-valid-layer lifecycle. Audit independent potential-field
and configuration helpers, retaining suitable functional implementations.
Local-stencil execution and trajectory execution remain distinct.

## M5: Complete I/O, Construction, And Export

Complete supported header/forest/tree/offset/block reading, mapped/native
sources, DAT writing and roundtrip, uniform-to-SFC construction, singleton-z
conversion, and VTK export. Record unsupported format behavior and cold/warm
I/O, bytes, faults, memory, and throughput. Native reading from M1 remains a
foundation, not a claim that writing/export is already migrated.

## M6: Dataset And Public Integration

Compose metadata and loaded-field lifecycle, boundary/name normalization,
derived public workflows, `open_dataset`, block/uniform readers, construction,
and write APIs. Specify compatibility for signatures, defaults, layouts,
mutation, and failures before replacing a public workflow. Reuse the internal
analysis session rather than adding scientific semantics to Dataset methods.

Small stateful convenience facades are allowed over the functional core.
Validate public workflows and staged backend selection/fallback through the
canonical package; SOURCE_MIGRATION remains the closure authority.

## M7: Packaging, Parallelism, And Cutover

Integrate clean/editable extension builds and cleanup, supported-platform
evidence, OpenMP or other useful parallel strategies with one-thread baselines,
public documentation, feature disposition, fallback/rollback, and default
backend switching. Retire/archive superseded paths only after the final
SOURCE_MIGRATION and public performance gates pass.

GPU portability is a durable design constraint, not an implemented backend
claim or an automatic M7 blocker. Activate a GPU implementation only with a
concrete useful workload, explicit buffer/numerical contracts, available
validation, and end-to-end host/device resource evidence. An earlier measured
CPU/OpenMP or device opportunity may be scheduled through WORKFLOW without
silently relaxing existing contracts or skipping final cutover obligations.
