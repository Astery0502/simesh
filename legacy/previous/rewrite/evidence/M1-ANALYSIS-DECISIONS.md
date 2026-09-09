# M1 Analysis Strategy Decisions

These records were moved unchanged from `../ANALYSIS_WORKLOADS.md` during the
2026-09-05 documentation realignment. They describe the original M1 workloads
and decisions, not new measurements or permanent restrictions on later work.
Use the current workload requirements to decide whether their assumptions still
apply. Detailed raw-run summaries remain in the linked group evidence and
[M1 horizon](M1-ARCHITECTURE-HORIZON.md).

### Completed Record: Bounded Refined Repeated Point Sampling

- Primary analysis workflow and query shape: selected magnetic/vector fields at
  ordered points, initially explicit clustered and scattered batches and later
  data-dependent streamline points; point count may greatly exceed unique leaf
  count.
- Expected access density and locality: sparse relative to the whole snapshot,
  with both long same-leaf runs and cross-leaf paths; benchmark `P >> U` and
  `P ~= U` instead of assuming either.
- Reusable state and its lifetime: immutable FST/GEO metadata for one snapshot
  lifecycle; the implementation retains no payload or cache across calls and uses
  only call-scoped owner/group plans and bounded canonical workspaces.
- Dominant expected cost: selective reader bytes and repeated refined support/
  halo completion for trilinear sampling, followed by point location and group
  planning; arithmetic is expected to dominate only after those costs are
  amortized.
- Current and structurally different candidate strategies: choose canonical
  root selection plus flat hierarchy descent, stable owner grouping, direct
  owner reads for zero order, and bounded RHE-001 completion for trilinear.
  Defer a retained leaf BVH/hash, maximum-level ownership grid, locality-aware
  neighbor walk, persistent payload LRU, and complete uniform-grid materialization.
- Workload-specific metrics and representative consumer: single/batch point
  latency, point and field-value throughput, depth/locality scaling, unique
  owners, owner reuse, time to first result, requested/read/support/output
  bytes, reader calls, support amplification, capacity scaling, managed bytes,
  peak RSS, and exact/numerical equivalence for repeated zero/trilinear sampling.
- Deferred alternatives and concrete reopen triggers: reconsider a last-leaf/
  neighbor locator when hierarchy fallbacks materially dominate coherent paths;
  a persistent cache when the native-reader warm streamline slice repeats
  material reads; direction-projected halo support when all-26 amplification is
  material; and another spatial index only when descent is a stable bottleneck
  after I/O and support costs are controlled.

### Completed Record: Native Selective AMRVAC Read

- Primary analysis workflow and query shape: open one v5 snapshot, then issue
  repeated sparse block/field reads for selected-region operators and point
  paths; blocks are read as complete interiors by current RHE/RPS consumers,
  while STO substitution still requires arbitrary interior boxes.
- Expected access density and locality: small/medium selections are sparse and
  may repeat support leaves; full traversal remains a comparator, not the
  default architecture.
- Reusable state and its lifetime: decoded header/tree arrays and a canonical
  FST binding live for one immutable file-descriptor lifecycle. Record headers,
  transfer groups, and byte buffers are call-scoped; no payload is retained.
- Dominant expected cost: cold file/page I/O and bytes read, then syscall and
  x-fast-to-canonical transfer cost; parsing and arithmetic are secondary.
- Current and structurally different candidate strategies: choose borrowed-fd
  position-independent reads, selected record-header preflight, exact unique
  field runs for full interiors, and one contiguous per-field envelope for
  other boxes. Defer eager all-record scans, mmap, path-open callbacks, and a
  persistent payload cache.
- Workload-specific metrics and representative consumer: metadata/open and
  time to first result; requested/useful/read bytes; pread/header/payload calls;
  duplicate suppression; support amplification; cold/warm RHE/RPS runtime;
  scratch/managed bytes, faults, RSS, and bitwise array/current equality.
- Deferred alternatives and concrete reopen triggers: reconsider eager record
  indexing only if warm repeated header reads dominate and cold sparse TTFW is
  protected; mmap or an owned persistent source if pread/syscall overhead is
  material; a payload cache only after the streamline slice supplies snapshot,
  field, reach, and PBC cache-key/lifetime evidence.

### Completed Record: Native Selected Refined Curl

- Primary analysis workflow and query shape: one contained physical ROI over a
  v5 snapshot, three magnetic input fields, exact cell-centered Cartesian curl
  outputs, and one unweighted regional component sum; small, medium, and full
  regions include mixed refinement and non-block-aligned boundaries.
- Expected access density and locality: ROI cell count and leaf support range
  from sparse to full; only cells whose canonical centers lie in the half-open
  ROI are output/reduced, while complete selected block interiors and one-layer
  support are current transfer units.
- Reusable state and its lifetime: immutable DAT/FST/GEO metadata and borrowed
  fd for one snapshot; ROI windows, spacing, RHE workspace, compact curl output,
  and reduction state are call-scoped. No payload/cache persists.
- Dominant expected cost: selected/support file bytes and refined halo planning,
  then curl memory traffic; ROI scan and serial sum are measured rather than
  assumed negligible.
- Current and structurally different candidate strategies: choose a two-pass
  full-leaf center-window scan, fixed fused curl, stable synchronous RHE
  consumer, equal-window run grouping, and persistent serial reduction. Defer
  hierarchy/BVH ROI lookup, block-cover/post-crop, six derivative temporaries,
  generic expressions, direction-projected support, and payload caching.
- Workload-specific metrics and representative consumer: selected cells/leaves,
  block-cover amplification, requested/read/support/output bytes, header/payload
  calls, chunks/halo/operator/reduction calls, first result, cold/warm runtime,
  managed/output bytes, RSS, exact reduction capacity invariance, and numerical
  native/array/current curl comparison.
- Deferred alternatives and concrete reopen triggers: hierarchy pruning remains
  closed because ROI selection is below 0.5% of small first-result time; the
  measured 27x small-tdm support load activates direction projection for the
  streamline design; batched generic recipes still require a second operator,
  and cache/header indexing awaits the streamline reuse trace.

### Completed Record: Cached Refined Vector Sampling

- Primary analysis workflow and query shape: ordered trilinear magnetic-vector
  stages for one or many data-dependent field lines, including long same-owner
  runs, owner transitions, and interleaved divergent seeds.
- Expected access density and locality: points are sparse relative to the
  snapshot; coherent runs show about 98% last/completed-owner reuse, while
  divergent orders expose a small working-set capacity knee.
- Reusable state and its lifetime: one immutable reader/FST/GEO/ordered-field/
  PBC lifecycle, one persistent RHE miss workspace, and a byte-bounded completed
  owner-halo LRU. Dynamic point/group plans remain call-scoped.
- Dominant expected cost: cache misses still pay all-26 support planning, reads,
  and halo application; on hits, unchecked exact location plus SAM-005 dominate.
- Current and structurally different candidate strategies: select exact hinted
  owner fallback plus completed-halo LRU. Reject per-point checked RPS as the
  session path; defer stage-only batching, raw-interior LRU, uniform grids, and
  direction-projected support.
- Workload-specific metrics and representative consumer: point latency and
  throughput, hint/fallback and owner reuse, cache hits/misses/evictions, halo
  fills, selected/support loads, logical/native bytes, capacity/working set,
  session/call/peak memory, RPS equality, and the following field-line executor.
- Deferred alternatives and concrete reopen triggers: neighbor transitions only
  if hierarchy fallbacks remain material after hints; raw caching only if native
  support rereads dominate miss-heavy traces; direction projection only through
  a separate proven COARSER/PBC planner when all-26 miss cost remains material.

### Completed Record: Native Refined Field Lines

- Primary analysis workflow and query shape: forward/backward magnetic field
  lines from one or many finite seeds, fixed physical-arclength steps, accepted-
  prefix trajectories, and cumulative oriented `integral(B dot dx)`.
- Expected access density and locality: four data-dependent stages per attempted
  step, long coherent owner runs for individual seeds, and divergent working
  sets across stage-major seed batches.
- Reusable state and its lifetime: one caller-owned CHS snapshot/field/PBC/cache
  session across calls; SLE stage arrays and active-seed plans are call-scoped;
  result arrays are caller-owned.
- Dominant expected cost: cold CHS owner misses/all-26 support and then warm
  per-stage cache/location/sampling calls; FLN/RK arithmetic is expected to be
  secondary but is measured as a hot path.
- Current and structurally different candidate strategies: choose scale-safe
  unit-tangent RHS, fixed classical RK4, whole-step nonperiodic rejection, and
  serial stage-major batches. Defer unnormalized time ODEs, Euler/midpoint,
  adaptive RK, boundary clipping, periodic wrap, and per-seed parallel loops.
- Workload-specific metrics and representative consumer: trajectory/integral
  error and convergence, termination codes/stages, point/step latency, hints/
  fallbacks/transitions, cache and halo fills, logical/native bytes per accepted
  point, capacity/cold/warm behavior, scratch/session/output/RSS memory, and
  single/multiple-seed scaling on analytic fields plus real tdm.
- Deferred alternatives and concrete reopen triggers: direction projection or
  raw caching only from actual miss-heavy trajectory time/bytes; neighbor lookup
  only if post-hint LOC remains material; adaptive/parallel execution only after
  fixed-step error and independent-seed scaling establish a useful target.

### M1 Horizon Disposition

- Both priority workflows are complete and M1 may close; see
  `M1-ARCHITECTURE-HORIZON.md`.
- Retain completed-halo working-set caching, sparse DAT headers, exact hinted
  location, bounded selected execution, and all-26 3D reference semantics.
- Carry requested-direction support into the dimension-aware M2 halo design;
  require separate target versus COARSER-slope/physical-base support and a
  measured 20% composed or 2x byte/support improvement before promotion.
- Defer raw-interior caching, eager headers/mmap, neighbor materialization,
  reusable all-26 plans, adaptive RK, and seed parallelism under the recorded
  triggers; none blocks M2 foundation work.


