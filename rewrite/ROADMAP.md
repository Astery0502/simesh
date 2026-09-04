# Rewrite Roadmap

The roadmap is ordered by semantic dependency. It may evolve through the
contract-change workflow, but each milestone must remain independently usable
and verifiable.

## Foundation

Establish the common vocabulary and the smallest executable contracts:

- scalar, index, and layout conventions;
- explicit array ownership and valid regions;
- operator access patterns;
- block sources, block sinks, workspaces, and memory budgets;
- a minimal build and test path isolated from the current package.

## Migration And Performance Baseline

Status: complete as a protocol; individual feature rows and baselines close as
their capabilities become active.

- maintain the source-wide supported-feature ledger in `SOURCE_MIGRATION.md`;
- distinguish rewrite, adapter, retain, replace, retire, and unsupported
  dispositions;
- use the benchmark levels, recording format, and regression policy in
  `PERFORMANCE.md`;
- require a real-data vertical slice and milestone performance summary in
  addition to kernel evidence.

## M0: Cartesian 3D Level-1

Status: complete.

Target: a complete non-periodic, non-staggered 3D path without refinement.

Required capabilities:

- layout and block-index primitives;
- level-1 Morton ordering;
- validated level-1 topology and Cartesian block geometry;
- in-memory and bounded block/chunk field sources;
- physical-boundary and same-level halo provision;
- exact level-1 placement, zero-order sampling, and trilinear sampling;
- one pointwise operator, one stencil operator, and one streaming reduction;
- numerical comparison with the current implementation;
- a bounded-memory path whose complete field payload need not reside in memory.

The existing AMRVAC reader may initially feed the new core through an adapter.
A native block source should be added when it is needed to demonstrate genuine
out-of-core execution.

## M1: Cartesian 3D Refined AMR

Status: in progress.  Refined forest reconstruction/conformance, raw contact
lookup, all-touch balance, leaf geometry, and balanced directional relation
records are complete.  The triggered STO-002 workspace/traversal/closure
decomposition precedes refined support planning: workspace accounting, shared
primary-prefix planning, direct-face closure, complete one-block halo closure,
and refined support union/bounded planning are complete.  Ratio-two
cell-average restriction, the independent current three-point limiter, and
ratio-two limited prolongation and refined source-slot resolution are also
complete.  REL already supplies action classification; refined target-region
and child-phase planning are also complete.  Source/workspace geometry and
value application were decomposed; combined source/workspace geometry split into
now-complete same-level translation, FINER placement, and COARSER workspace
geometry.  The analysis re-audit places an explicit selected-primary refined
support planner before the next executor consumer.  It groups that planner with
FINER placement and COARSER workspace geometry so the composed planning result
does not inherit STO-004's dense-primary traversal.  That selected transfer-
planning group is complete; PBC-aware support completion and value application
are also complete through CSP-001/PWA-001/CWA-001 and the bounded selected
RHE-001 consumer. Exact refined point ownership, reusable zero/trilinear point
kernels, and bounded repeated point sampling are complete while the fused
SAM-002/SAM-003 uniform strategies remain available. The native selective
`.dat` adapter and selected local-field slice are complete; streamline is next.
Six-face TOP-003 caching remains optional and requires repeated-consumer
performance justification.

The native-read group now completes safe v5 byte decoding, canonical forest
binding, and non-staggered selected transfer through RHE/RPS. The local-field
slice is complete and the streamline analysis slice is next; v3/v4 format
breadth and owned dataset/file lifecycle remain assigned to M5/M6.

The completed local-field slice adds exact refined cell-center ROI windows, fixed
Cartesian curl, and a stable completed-primary consumer before composing native
selected output and reduction. The separate streamline location/cache/stepping
policy is the next M1 vertical slice.

The completed streamline preparation group isolates exact last-owner hints
and a byte-bounded completed-owner halo sampling session. It retains full RHE
miss semantics; raw source caching and direction-projected support remain
separate measured reopen choices before field-line stepping is frozen.

Before refined halo work, complete the functional-composition checkpoint:

- expose block readers and writers as explicit coarse-grained function
  adapters over canonical buffers;
- retain array/memmap behavior as one backend rather than a semantic
  dependency;
- keep resident and bounded traversal as interchangeable execution strategies;
- separate halo requirements, support closure, relation planning, and value
  transfer before adding coarse/fine behavior.

Add:

- validated parent/child reconstruction;
- coarse, sibling, and fine neighbor relations;
- explicit selected-primary support planning for sparse/ROI leaf streams;
- restriction and prolongation;
- refined ghost provision;
- refined zero-order and trilinear sampling;
- a native selective AMRVAC block-reader adapter;
- refined real-data correctness, runtime, memory, and I/O comparison.

M1 storage and halo implementations must compose through the same canonical
contracts. Native AMRVAC, mapped, resident, cached, or other adapters may alter
I/O and scheduling but not refined numerical semantics.

## Analysis Priority Gate Before M2

Before broad dimensional and periodic generalization, validate that the
completed Cartesian 3D refined core supports the primary analysis direction in
`ANALYSIS_WORKLOADS.md`. Activate the minimum selected M4-style capabilities
needed for two vertical slices:

- a native selective read feeding a bounded physical-region local diagnostic,
  with shared input/halo work and a regional output or reduction;
- an exact spatial locator and field sampler feeding one representative
  streamline composition with explicit stepping, termination, and cache
  behavior.

This gate does not require the complete derived-field public lifecycle or every
scientific operator. Its purpose is to let real analysis consumers test the
storage, layout, ownership, caching, and execution boundaries before M2/M3 make
them dimension- and periodic-aware. Record time to first result, useful/read
bytes, support amplification, peak memory, query latency, cache behavior, and
the appropriate numerical comparisons.

## M2: Cartesian 2D

Generalize established concepts to active x/y dimensions:

- quadtree traversal;
- singleton-z external arrays;
- 2D halo provision and refined interfaces;
- exact placement, bilinear sampling, and operator behavior.
- a representative real Cartesian 2D `.dat` vertical slice.

Do not implement 2D as unrelated duplicate logic. Generalize only after the 3D
contracts make the shared and dimension-specific parts visible.

## M3: Periodic Cartesian Meshes

Add periodicity as an explicit topology and halo input, then validate periodic
connectivity, refined periodic interfaces, sampling, and stencil behavior.

Separate parity from new capability: periodic metadata or behavior already
supported by the current canonical path is compared directly; newly completed
periodic execution is labeled and validated as an extension rather than
reported as migrated parity.

## M4: Scientific Operators And Derived Fields

Complete and generalize the scientific and field-lifecycle families. Selected
Cartesian 3D local-analysis and streamline primitives may already exist from
the analysis priority gate; preserve their contracts and extend them rather
than implementing unrelated replacements:

- pointwise field transforms;
- derivative batching and local stencils;
- AMR block operators;
- associative reductions;
- geometry-aware sampling;
- field-line traversal and integration.
- derived-field dependency registration, materialization, selectors, dropping,
  and ghost-valid-layer behavior;
- current/divergence diagnostics represented by concrete operator contracts;
- an audit of `simesh.tools` and configuration helpers, retaining already good
  independent functional implementations where appropriate.

Field-line work should compose a spatial locator, field sampler, stepper,
termination policy, and reducer. It should not be forced into the local stencil
kernel interface.

## M5: AMRVAC I/O, Construction, And Export

Complete format-facing behavior over functional adapters:

- header, forest, tree, offset, and selective block parsing;
- resident, mapped, and native bounded block sources;
- `.dat` writing and read/write roundtrip;
- uniform-to-SFC construction and Cartesian 2D singleton-z behavior;
- VTK export and explicit unsupported-format rejection;
- cold/warm I/O, bytes transferred, page-fault, memory, and throughput evidence.

## M6: Dataset And Public API Integration

Build the user-facing functional composition:

- dataset metadata and loaded-field lifecycle;
- boundary-condition normalization and field-name adapters;
- derived-field public workflows;
- `open_dataset`, `read_blocks`, `read_uniform`, uniform construction, and
  write APIs;
- compatibility tests and real workflow benchmarks through canonical
  `simesh.amrvac` entrypoints;
- a staged backend-selection and fallback path.

The user-facing facade may use small stateful convenience objects, but domain
semantics and heavy work remain explicit functional transformations.

## M7: Packaging, Parallelism, And Cutover

- integrate rewrite extensions into clean/editable package builds and clean
  flows;
- establish OpenMP or other parallel implementations through the same
  contracts, including one-thread baselines, speedup, and efficiency;
- verify supported-platform build/test/benchmark profiles;
- complete the source migration ledger and public documentation;
- exercise fallback and rollback, switch canonical defaults, then retire or
  archive superseded current/legacy computational paths.

M7 completes the project only when the final cutover gate in
`SOURCE_MIGRATION.md` and public-workflow performance gates in `PERFORMANCE.md`
are satisfied.
