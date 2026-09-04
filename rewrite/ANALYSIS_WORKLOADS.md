# Analysis Workload Priorities

## Purpose

The rewrite serves local scientific analysis of mostly immutable AMR snapshots.
Its primary workloads are selected-region field analysis and field-line or
streamline analysis. Simulation-style full-domain state updates remain useful
comparators but do not define the default architecture or optimization target.

This policy guides algorithm selection, data structures, execution strategies,
benchmarks, and the order in which broader support is activated. It does not
weaken numerical semantics, reproducibility, valid-region rules, or supported
format compatibility.

## Workload Model

The common analysis session:

- opens one or more large read-mostly snapshots;
- selects a small set of fields and often a physical region, slice, or seeds;
- performs several related queries against reusable forest, topology, and
  geometry metadata;
- computes local derived quantities such as gradient, divergence, curl,
  current density, vorticity, magnitude, and regional reductions;
- samples fields along data-dependent trajectories;
- may process data larger than memory on one local machine.

This differs from a simulation loop that repeatedly updates every cell, fills
global halos, writes new state, and optimizes aggregate cell updates or MPI
exchange. The rewrite may support those operations where required for parity,
but they do not outrank the analysis workloads above.

## Product Priorities

Use this order when trade-offs cannot satisfy every workload equally:

1. **Correct spatial and numerical semantics.** Preserve deterministic leaf and
   cell ownership, coarse/fine relations, valid regions, interpolation rules,
   operator meaning, and reproducible results across execution strategies.
2. **Selective I/O and computation.** Read and compute only requested fields,
   primary blocks, regions, and required support whenever the contract permits.
3. **Time to first useful result and bounded memory.** Prefer useful partial or
   regional results without requiring the complete payload to be resident.
4. **Reuse with explicit lifetime.** Reuse immutable forest/topology/geometry
   metadata and measured query plans or payload cache entries without making
   hidden state part of numerical semantics.
5. **Composed throughput.** Reduce bytes moved, support amplification, repeated
   halo work, and repeated traversal before micro-optimizing arithmetic kernels.
6. **Parallel independent analysis.** Prefer parallelism across seeds, regions,
   fields, and snapshots. Parallelize one dependent streamline only when
   evidence establishes a useful strategy.
7. **Full-domain update and write throughput.** Optimize these after supported
   analysis workflows unless migration parity or a measured shared bottleneck
   requires earlier work.

## Local-Field Execution Family

The intended composition is:

```text
physical region or selected leaves
        -> primary selection
        -> operator access requirements
        -> support closure and selected block reads
        -> shared halo provision
        -> batched compatible operators
        -> selected outputs and/or streaming reductions
```

Group local operators when they share input fields, access reach, boundary
semantics, and workspace validity. A batch should amortize reads, support
planning, and halo provision without merging independently variable numerical
definitions. Uniform-grid materialization is an output strategy, not the
default intermediate representation.

Optimize this family primarily by useful bytes read, support amplification,
number of halo provisions, workspace peak, time to first result, and composed
throughput. A faster isolated stencil does not justify extra whole-domain reads
or payload-sized temporaries.

## Streamline Execution Family

Streamline work composes a spatial locator, field sampler, stepper, termination
policy, and reducer. It does not share the local-stencil executor merely because
both consume fields.

Prefer this lookup order when evidence supports it:

1. test whether the next point remains in the last leaf;
2. use an exact adjacent relation when the path crosses a known face;
3. fall back to root selection and hierarchy descent;
4. obtain required fields from a bounded block cache or reader;
5. interpolate, advance, terminate, and reduce through explicit contracts.

Forest and geometry metadata may remain resident when small enough. Field
payload caching is budgeted separately and keyed by explicit leaf and field
identity. Cache policy, prefetch, step size, interpolation, and termination are
replaceable decisions. The result must remain within its numerical contract
across cache sizes and traversal strategies.

Optimize this family by point/step latency, samples per second, last-leaf and
neighbor-transition hit rates, hierarchy fallbacks, block loads, cache hit
rate, bytes per sample, peak cache bytes, and trajectory equivalence. Global
cells per second alone is not a useful streamline metric.

## Algorithm Selection Gate

Before selecting or optimizing an algorithm or retained data structure, record:

```text
Primary analysis workflow and query shape:
Expected access density and locality:
Reusable state and its lifetime:
Dominant expected cost (I/O, transfer, planning, arithmetic, allocation):
Current and structurally different candidate strategies:
Workload-specific metrics and representative consumer:
Deferred alternatives and concrete reopen triggers:
```

Prefer optimizations in this order unless evidence shows another bottleneck:

1. avoid unnecessary file reads and payload materialization;
2. reduce selected-support and transfer amplification;
3. reuse metadata, plans, halos, and cache entries with explicit ownership;
4. batch or fuse compatible operators at an already validated boundary;
5. improve hot-loop locality, vectorization, and arithmetic reuse;
6. parallelize independent queries and then measured inner kernels.

Do not optimize for a hypothetical simulation loop, global cache, or complete
uniform grid when the representative analysis workflow does not consume it.

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

## Required Performance Profiles

### Selected Local Field

Use a representative refined snapshot, a small field set such as magnetic
components, and physical regions covering small, medium, and full-domain cases.
Compute at least one multi-term diagnostic such as curl/current density and one
regional reduction. Record:

- metadata/open and time to first result;
- requested, read, support, and output bytes;
- primary/support amplification and halo count;
- cold and warm runtime;
- managed workspace and peak RSS;
- cells or output values per second;
- exact or numerical comparison with the reference/current behavior.

### Streamline

Use single and multiple seeds, short and long paths, and both spatially coherent
and divergent seed sets. Record:

- point and integration-step throughput plus latency distribution;
- last-leaf, neighbor-transition, hierarchy-fallback, and cache hit counts;
- reader calls and bytes per accepted point;
- cache capacity and peak memory;
- cold and warm behavior;
- termination classification and trajectory comparison;
- scaling across independent seeds.

### Snapshot Series

When repeated topology occurs across files, measure metadata/plan reuse and
per-snapshot payload work separately. Do not assume topology reuse until the
format adapter proves identity and lifetime safely.

## One-Time Reorientation Audit

Before resuming new M1 implementation under this policy, audit completed M0/M1
capabilities once. This is not a rewrite or a new benchmark campaign. Classify
each relevant boundary as:

- retain unchanged;
- retain semantics and defer optimization until an analysis consumer;
- reopen before the next consumer because it forces avoidable global work,
  payload movement, retained state, or an incompatible query boundary;
- deprioritize because it serves migration parity rather than the primary
  analysis path.

Reopen code only when the finding can materially affect the selected local-field
or streamline path, asymptotic behavior, transfer volume, peak memory, or a
cross-layer representation. Use existing contracts, tests, and evidence first.
Add a focused experiment only for a remaining decision-changing uncertainty;
do not rebuild, rebenchmark, or refactor every completed capability.

Record one concise audit summary with findings, owners, and reopen triggers.
After that summary is accepted, resume the next capability group. Later
milestone horizon reviews apply the same workload priorities without repeating
this whole historical audit.
