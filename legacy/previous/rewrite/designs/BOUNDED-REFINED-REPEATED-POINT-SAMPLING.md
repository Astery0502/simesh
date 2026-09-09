# Bounded Refined Repeated Point Sampling Design

## Problem And Outcome

The completed refined core can construct a flat forest, canonical leaf
geometry, selected support, and complete one-cell primary halos, but its only
samplers still materialize block-owned uniform grids.  Local analysis and
streamlines instead need many values at explicit physical points, often with
many points in a few leaves and a snapshot too large to keep resident.

This group produces values in caller point order without a uniform-grid
intermediate.  It separates four independently variable decisions:

1. LOC-001 maps physical points to exact refined leaf owners.
2. SAM-004 copies the canonical owner-level cell for explicit point groups.
3. SAM-005 applies the fixed trilinear stencil to explicit point groups over
   completed primary workspaces.
4. RPS-001 groups owners and schedules bounded functional reads/halo work.

SAM-002 and SAM-003 remain fused block-centric uniform strategies.  Shared
inline face, cell-owner, stencil, and blend primitives may be extracted, but
their public operations and output-range loops remain unchanged.

## Alternatives And Exploration Coverage

The current mesh loops over every leaf and finds overlapping uniform output
windows.  That performs avoidable whole-domain work for sparse paths, makes
exact shared-face ownership an iteration-order effect, and offers no reusable
point locator.

A maximum-level dense owner grid makes lookup constant time but grows as the
deepest complete lattice.  A BVH, spatial hash, or sorted leaf-face index uses
`O(L)` retained state and duplicates the existing octree.  An `O(P*L)` scan is
simple and independent, so it is retained only as the locator oracle.  The
selected production locator chooses a root under canonical face comparisons
and descends the existing flat children in `O(P*depth)` after normal root
candidate verification.

For payload execution, a complete resident halo array is the simple semantic
comparator but defeats out-of-core analysis. Per-point reads minimize planning
but repeat owner and support bytes. RPS-001 therefore locates once and stable-
groups equal owners. Zero-order execution processes bounded unique-owner
batches and reads only those owners. Trilinear execution passes every unique
owner to RHE-001 and samples each fully completed primary prefix synchronously
before its workspace is reused. The hook is a private parameterization of
RHE's existing terminal writer boundary, not a BlockWriter: using a sampling
callback as STO-003 scatter or declaring a fixed compact writer with incomplete
logical coverage would violate the storage contract.

No persistent LRU, prefetch, last-leaf state, neighbor walk, or streamline
stepper belongs to this group.  Those choices need snapshot identity, cache-key
and lifetime rules, path state, and real native-reader evidence.  Direction-
projected refined support is likewise deferred until all-26 read amplification
is material in this consumer.

The decision-changing uncertainty is the crossover between owner grouping and
its `O(P log P)` plan versus repeated read/halo avoidance.  The standard group
benchmark therefore includes clustered `P >> U`, scattered `P ~= U`, coherent
crossings, and capacity scaling, and records planning separately from read,
halo, and sample time.

## Numerical And Ownership Choices

All faces reuse GEO's explicit binary64 operation order.  For level `ell`,

```text
base_h = (domain_upper - domain_lower) / float(domain_cell_counts)
h      = ldexp(base_h, -(ell - 1))
face(g)= domain_lower                         when g == 0
       = domain_upper                         when g == level_domain_cells
       = domain_lower + float(g) * h          otherwise
```

Multiplication and addition remain separate.  Root selection and owner-level
cell selection use the greatest canonical lower-face index whose face is no
greater than the point.  Internal root, child, leaf, edge, and corner equality
therefore chooses the positive/upper side independently on each axis, including
coarse/fine interfaces and coincident representable faces.

The physical domain is half-open `[domain_lower, domain_upper)`. The lower
endpoint is owned; exact upper endpoints and finite points outside any axis
receive the `-1` owner sentinel and do not mutate sample rows. Nonfinite query
coordinates are metadata errors rejected before output mutation or I/O. A
closed-upper proposal was rejected after independent review of a GEO-valid
large-origin case: at the upper endpoint SAM-003 arithmetic produced `j0=B`
and `j1=B+1`, beyond one-cell reach. Clamping would invent a new numerical rule
and obstruct later periodic wrap-before-location composition.

Zero order repeats the owner-level global canonical-face search and copies one
value exactly.  It does not use a naked local quotient.  Trilinear sampling
uses, separately in binary64,

```text
delta = point - leaf_lower
ratio = delta / leaf_spacing
u     = ratio - 0.5
j0    = floor(u)
j1    = j0 + 1
w     = u - float(j0)
```

and SAM-003's fixed z, y, x eight-load/seven-blend tree.  All eight values are
loaded even for zero weights.  Its exact lower and upper reach is `(1,1,1)`.

## Stable Intermediate Representation

Both numerical samplers consume explicit slot-to-point groups rather than
locating, sorting, or resolving cache entries:

```text
slot_leaf_ids[S]
point_indices[Q]
slot_point_offsets[S+1]
```

Slot `s` owns `point_indices[offsets[s]:offsets[s+1]]`. Slot leaf IDs may be in
arbitrary order and repeat; ascending slot/group iteration gives deterministic
last-occurrence overwrite. Points and result rows remain in caller order. The
mapping is validated against LOC-001 ownership before mutation. This
representation works for resident canonical leaf slots, a bounded owner batch,
or a future cache without making any one policy part of SAM-004/SAM-005.

RPS-001 privately creates a full-call owner/order/group plan. Its point-sized
bytes are explicit and separately accounted from block workspaces. Zero-order
owner capacity bounds directly read interiors. Trilinear RHE slot capacity
independently bounds primary/support halo work and needs no second completed-
owner buffer: the private synchronous consumer receives one complete primary
prefix, its exact ascending input-primary offset and IDs, and the full padded
valid box before the next read. It may not mutate or retain workspace views and
must return `None`. Public RHE writer semantics, empty callback, output bits,
stats, and failure behavior remain unchanged. A later point-plan streaming
strategy may trade repeated owner reads for a smaller plan, but is not selected
without a point-array-scale memory bottleneck.

## Decomposition Records

### LOC-001

- Responsibility: map each physical point to one canonical Cartesian refined
  leaf or the exterior sentinel.
- Owned decisions: domain inclusion, exact face equality, root selection,
  hierarchy descent, and leaf identity.
- Non-owned: cell selection, interpolation, payload, fields, reader/cache,
  locality hints, adjacency, stepping, and termination policy.
- Inputs/outputs: borrowed domain/count/MOR/FST metadata and `float64[P,3]`;
  caller-owned `int64[P]` leaf IDs.
- Mutation/ownership: output is fully overwritten after validation; nothing is
  retained.
- Access/reach: metadata traversal only; no payload or halo.
- Reference: independent all-leaf GEO bounds scan with identical partition
  inequalities, plus SAM-002 level-one ownership reduction.
- Producer/consumer: FST-002/GEO-002 lifecycle -> SAM-004/SAM-005/RPS-001.
- Performance: `hot-kernel`; single/batch latency, points/s, depth/root and
  coherent/scattered scaling, with fixed per-point scratch.
- Five questions: yes/yes/yes/yes/yes.

### SAM-004

- Responsibility: copy the exact owner-level canonical cell value for explicit
  slot-to-point groups.
- Owned decisions: highest-face cell ownership and exact value copy.
- Non-owned: leaf location, group planning, storage/cache, fields, halos,
  exterior fill, scheduling, and parallelism.
- Inputs/outputs: borrowed canonical payload/interior, slot leaf IDs, FST/GEO
  metadata, points and group map; caller-owned `float64[P,F]` rows.
- Mutation/ownership: only selected point rows are overwritten after complete
  validation; nothing is retained.
- Access/reach: one owner interior cell; zero halo reach.
- Reference: scalar global-face search and bitwise SAM-002 reduction.
- Producer/consumer: LOC/group routing -> resident and RPS-001 execution.
- Performance: `hot-kernel`; points/s and field-values/s plus composed read path.
- Five questions: yes/yes/yes/yes/yes.

### SAM-005

- Responsibility: evaluate one fixed trilinear reconstruction for explicit
  slot-to-point groups.
- Owned decisions: axis stencil/index/weight, exact one-cell reach, eight loads,
  and z/y/x blend tree.
- Non-owned: leaf location, group planning, halo values, PBC/topology,
  storage/cache, scheduling, and parallelism.
- Inputs/outputs: borrowed completed canonical payload, slot geometry, points
  and group map; caller-owned `float64[P,F]` rows.
- Mutation/ownership: only selected rows are overwritten after complete
  validation; nothing is retained.
- Access/reach: `LOCAL_STENCIL`, lower/upper `(1,1,1)`.
- Reference: scalar fixed-tree interpolation, affine/convergence checks, and
  bitwise SAM-003 level-one reduction.
- Producer/consumer: LOC/RHE-001 -> resident and RPS-001 execution.
- Performance: `hot-kernel`; points/s, field-values/s, and completed-halo
  composition.
- Five questions: yes/yes/yes/yes/yes.

### RPS-001

- Responsibility: schedule one bounded repeated-point request over a functional
  block reader while preserving LOC/SAM semantics.
- Owned decisions: one-time location, stable owner grouping, zero-order owner
  batch order/capacity, trilinear RHE support capacity, and the two explicit
  compositions.
- Non-owned: ownership/ties, cell or interpolation arithmetic, halo values,
  reader backend, persistent cache/prefetch, neighbor transitions, stepping,
  termination, and parallelism.
- Inputs/outputs: explicit reader, geometry/forest, points, field selection,
  capacities and (for trilinear) PBC metadata; caller-owned `float64[P,F]`.
- Mutation/ownership: call-scoped plans/workspaces only; the private RHE
  consumer is synchronous and retains no views. External failures may leave
  completed owner-group rows, while ordinary contract errors precede I/O and
  mutation.
- Access/reach: owner interiors for zero order; RHE-001 all-26 support for one
  trilinear layer.
- Reference: resident full-owner composition and capacity/backend invariance.
- Producer/consumer: STO-003/RHE-001 -> native `.dat` and later streamline
  execution.
- Performance: `composition-only`; first useful result, read/support/output
  bytes, owner reuse, calls, amplification, runtime, and peak managed memory.
- Five questions: yes/yes/yes/yes/yes.  Two numerical modes are separate entry
  points over one scheduling policy and retain their independent SAM contracts.

## Completion Gate And Reopen Triggers

Post-completion promotion note: LFE-001 supplied the second completed-primary
consumer, so RHC-001 now owns the stable synchronous descriptor/executor and
RPS consumes it directly. The private-hook language below records the original
design sequence rather than the current architecture.

The group closes only after independent locator/reference agreement at all
root, child, coarse/fine, edge, corner, and physical-domain boundaries;
bitwise zero-order comparison; trilinear finite-error and IEEE-category tests;
RHE SAME/FINER/COARSER/pure/mixed PBC composition; resident/bounded/non-array
reader and capacity invariance; and level-one bitwise reduction to the retained
fused wrappers.  If shared inline extraction regresses either fused standard
median by more than 10%, investigate; two stable cases above 20% block
completion under the existing policy.

Reopen a retained spatial index only when hierarchy descent is a measured
bottleneck.  Reopen neighbor/last-leaf execution at the streamline slice,
persistent caching at a repeated native-reader workload, point-plan streaming
when plan bytes are material relative to points/results, and direction-projected
support when all-26 amplification dominates trilinear execution.

## Independent Contract Review

Independent review approved the four-way semantic/execution split and the
private synchronous RHE consumer, subject to preservation of public RHE writer
behavior. The initial closed-upper proposal was rejected after a separate
current/history audit supplied the reproduced large-origin stencil
counterexample. Review therefore froze half-open global ownership, atomic
nonfinite rejection, zero/trilinear separation, full pre-I/O group/stencil
preflight, and distinct zero-owner versus trilinear-support capacity meanings.
