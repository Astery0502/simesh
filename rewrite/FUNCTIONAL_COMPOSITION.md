# Functional Composition Boundaries

## Three Substitution Boundaries

This document records boundaries for the existing rewrite implementation.
The new project-level shared boundary is in
[prepared fields](../docs/analysis-core/prepared-fields.md); old representations
remain scoped reference contracts. Product goals are in
[intent.md](../docs/analysis-core/intent.md); concrete signatures and guarantees remain in contracts.

| Boundary | Responsibility | Examples |
| --- | --- | --- |
| Storage adapter | Transfer selected data between external storage and an explicit buffer protocol | Native DAT, resident arrays, memmap, future formats |
| Execution strategy | Select chunks, cache entries, schedules, resource lifetimes, and output delivery | Resident, bounded, cached, independent-seed parallelism |
| Compute implementation | Apply contracted numerical transformations to buffers and explicit plans | Cython CPU, OpenMP loops, future GPU kernels |

Storage does not discover topology or select halo arithmetic. Numerical kernels
do not infer storage, caching, scheduling, or backend from a dataset object or
global state. Execution composes existing meanings without introducing hidden
boundary, ownership, or numerical policy. Local-field and field-line execution
remain distinct families over shared geometry, access, and sampling contracts.

The confirmed mainlines now also include full-domain local-response LOS
synthesis. Observation-defined ray/image accumulation need not use the magnetic
trajectory executor. Q/twist are magnetic-line consumers: request derivative
fields and coupled auxiliary state as their method needs, without making the
independent whole-domain current pipeline a prerequisite. Detailed result needs
are in [THREE-PIPELINE-RESULTS](../docs/analysis-core/pipeline-results.md).

## Semantics, Buffer Protocols, And Strategies

Distinguish these concerns when a capability becomes active:

- Domain semantics: block/cell identity, coordinates, field meaning, valid
  regions, stencil reach, halo rules, results, and numerical comparison.
- Buffer protocol: dtype, layout, strides, memory location, ownership, aliases,
  and producer/consumer lifetime. FND-001 is the current CPU interchange.
- Strategy: loop tiling, fusion, vectorization, traversal, caching, concurrency,
  transfer scheduling, and backend choice.

Existing contracts can bind all three; they remain unchanged until explicitly
reviewed. This separation guides future extensions, not reinterpretation of
already frozen behavior. A different buffer protocol needs a validated adapter
or an independently implemented compute contract. GPU buffers are not accepted
by current NumPy wrappers merely because they have a similar shape.

The canonical host payload is C-contiguous native float64 in
`(slot,field,x,y,z)` order with signed int64 IDs. Another backend may choose a
different device layout and keep intermediate fields resident across kernels.
It must preserve field/cell mapping and declared numerical guarantees, expose
conversions explicitly, and count transfers. No mandatory host roundtrip is
introduced between device kernels. Build device abstractions only for an active
backend with a concrete workflow.

## Storage And Failure

STO-003 readers/writers are immutable descriptors with explicit state, canonical
shape, a coarse-grained callable, and arrays participating in alias validation.
Calls occur per transfer, never inside cell/stencil/block hot loops. The array
adapter is the reference; native DAT-003 is another implementation. Selection,
duplicate ordering, exact copies, and preservation remain contract-defined.

Storage failures and ordinary contract errors have distinct guarantees. Preserve
checked preflight and overlap rules; document whether I/O can leave a completed
or partial output prefix. An executor is not an external transaction. Read
callbacks and synchronous consumers may not retain borrowed workspace views.

## Field Validity And Halo Composition

Halo composition has independently owned layers:

1. Each operator input declares the access needed for a requested output region.
2. Topology/geometry determines physical, same-level, coarse/fine, or periodic
   relations.
3. A planner closes support, including extra prolongation slope and physical-base
   dependencies beyond the direct output stencil.
4. Physical transforms, copies, restriction, and prolongation define values.
5. An executor schedules transfers and applies those rules to explicit outputs.

Propagate requested regions backward through operator dependencies and valid
regions forward through results. Allocated halo width is not evidence of valid
values. Required input reach follows the actual operator chain and downstream
consumer. Asymmetric inputs, composed stencils, inactive axes, and coarse/fine
support need their own proofs within the relevant contracts, not a separate
field-specific exploration program. Existing RHE extent/reach
limits continue to apply; wider requests require another supported strategy or
explicit rejection, not clipping.

Before freezing a derived-field contract, choose between differentiating an
extended input field and exchanging already-derived interiors. These generally
do not commute at coarse/fine interfaces. Resolve the choice when a concrete
consumer needs it and establish its scientific error before promotion.
Conservation, interface continuity, and derivative order are not inferred from
primitive equality.

Full padded blocks remain the current compute reference. Interior-only support
storage, retained completed halos, sparse requested directions, and separate
halo storage are candidate strategies. Never omit required slope/base support
to obtain a smaller direct-neighbor count. Reuse shared input and halo work
across compatible operators without merging their numerical definitions.

## Explicit Reuse And Sessions

An immutable snapshot may support a reusable analysis session owning workspace,
plans, caches, and resource state. Prefer free functions consuming explicit
descriptors; do not make kernel results depend on undocumented session history.

A retained artifact identifies its source lifecycle, leaf/field identity or
operator and dependencies, relevant geometry/boundary/numerical configuration,
buffer representation, and actual valid region. Invariants fixed by a session
need not be repeated in every entry key. A cache hit must cover the requested
valid region and matching semantics, not merely an allocated halo width.

Raw interiors, completed fields, derived fields, and metadata/plans are distinct
reuse opportunities, not four mandatory caches. Select the smallest useful
combination from a real consumer and declare admission, eviction, failure,
release, and accounting. A source or dependency change creates a new lifecycle
or explicitly invalidates dependent artifacts before reuse. No content hashes
or speculative cache framework are required.

Keep location and sampling reusable across trajectory and diagnostic fields.
The stepper advances geometry; independent diagnostic evaluators and reducers
consume needed fields. For example, a contracted arclength derivative may use
`grad(q) dot tangent`; an integral must name its integrand, orientation, and
measure. Existing fixed-RK and termination contracts remain their own baseline.

## Execution, Ownership, And Resources

Resident, bounded, and cached strategies select traversal and resource policy.
Checked wrappers validate full contracts; equivalent executor preflight permits
unchecked inner kernels. Retain simple resident/reference compositions when
optimizing or fusing operations. Capacity/history/thread changes obey the
chosen numerical contract, including fixed reduction order where required.

Bounded workflows account for all controlled simultaneously live resources as
specified in PERFORMANCE. Provide bounded output delivery or accumulator-only
execution when output size can exceed memory. Explicitly supplied full outputs
remain a supported strategy, with their cost visible.

The current RHC callback is synchronous and borrowed views expire on return.
A future asynchronous/device boundary must separately define completion,
buffer reuse, ownership until completion, dependencies, synchronization,
cancellation/failure, and concurrent access. Do not reinterpret a synchronous
contract as asynchronous or release a buffer while device work still uses it.
Device pools and explicit session state are compatible with a functional core.

Current parallel delivery targets are single-node/OpenMP, not MPI/cross-node.
Scientific trajectories, diagnostic maps, images and large uniform outputs have
caller/sink lifetimes separate from disposable field/ghost caches. Joint
diagnostic/trajectory retention and bounded uniform delivery must fit the total
budget, including per-worker state; no automatic trajectory replay is required.

## Conformance And Numerical Strategies

Every backend/strategy validates exact selection and discrete semantics,
preservation, valid regions, ownership/alias/failure rules, and the applicable
numerical guarantees against the independent and resident references. Storage
conformance includes reordered/repeated fields, nontrivial block order, multiple
capacities, refined/physical halos, and non-ndarray backend state.

Existing strict operation trees, dtype, IEEE behavior, and serial reductions
remain frozen. FMA, reassociation, parallel reduction trees, or another precision
are not automatically equivalent. A separately named numerical strategy needs
explicit tolerances/invariants, determinism scope, scientific error evidence,
and WORKFLOW material review before any dependent consumer changes. Compatible
implementations of one strategy share a conformance suite; intentionally
different numerical strategies also need an explicit comparison and migration
decision. PERFORMANCE measures the full host/device workflow, not kernel speed
alone.
