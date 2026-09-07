# Runtime Execution Round

Status: active, 2026-09-08. The latest user instruction reopens execution backends,
parallel scheduling and finite numerical reuse under the existing native layout.
Read [current](current.md), [development](development.md),
[prepared fields](prepared-fields.md) and [performance](performance.md).

## Scope And Evidence Separation

Previous one/four-worker F and LOS measurements use the **same Python thread
pool calling GIL-free kernels**. They establish scaling for those workloads, not
backend selection. The OpenMP build and 23 AMR checks establish build/behavior
compatibility, not matched backend performance. The previous 64/256-slot trace
comparison measures avoided preparation, not parallel acceleration. Dense bulk
versus bounded preparation comparisons have different residency and are not
same-budget backend comparisons.

This round measures sparse F, dense whole-domain curl plus slices, and repeated
scalar LOS with unchanged numerical requirements and output. Another session
owns AIA response, rebricking and persistent geometry plans. This session may
measure transient planning and describe a plan interface; it does not implement
plan persistence or a thermal response.

## Selected Initial Questions

1. Attribute complete source-to-result wall/CPU time to reading, transient
   support planning/checks, ghost application, packing, consumer kernels and
   coordination/waiting. Separate instrumented attribution from timing runs.
2. At equal upper memory budgets, compare current complete-block retention with
   finite reuse of raw support values, spatial grouping and preparation batches.
   Measure cold numerical caches and compatible repeated queries. OS file cache
   is uncontrolled; do not call these physical disk-cold results.
3. Select at most two execution alternatives from those costs. Compare complete
   cold/hot flows and worker counts 1/2/4. Preserve current thread-pool path.
   Kernel-only speedups are diagnostic evidence, never sufficient acceptance.

The first attribution run is observational. Each following probe will bind its
question, comparator and acceptance before implementation. Initial probes use
two variants and a 15-minute experiment envelope; expand only for a concrete
unresolved result. Existing lightweight core-behavior checks remain binding.

## Lifetime And Resource Rules

Numerical values belong to an immutable source lifecycle, selected physical
fields/order, reconstruction/boundary strategy and halo validity. Mesh equality
or persistent geometry never certifies unchanged values. Replacing/updating
values starts a new source lifecycle and new numerical caches; active leases
must end first. Failure cannot publish partially filled slots. Idle entries may
be evicted; no active worker can lose its backing. Retained products keep their
existing ownership semantics. No value hashing or editable Dataset integration
is implied.

Use the actual ~1.045 GB WENO fixture and existing small manufactured cases.
Limits: four compute workers, 2 GiB controlled live arrays, 2 GiB task-created
scratch and at least 2 GiB free disk. Each comparison states tighter allocations,
includes cache and scratch overhead, keeps requested outputs equal and reports
RSS separately. This machine has eight logical CPUs and 8 GiB RAM. Other hardware
and cross-machine runs remain unverified when unavailable.

## Recovery Point

Initial audit complete; no runtime candidate adopted yet. Next: instrument the
existing file-backed F/D/LOS paths, then select the first finite-reuse experiment.
No source fixtures or previous evidence products should be deleted for this work.

## Probe R1: Finite Numerical Reuse (Selected)

Initial instrumented file-backed F (256 shuffled central seeds, 128 steps,
256 prepared slots / 128 support slots) spends ~0.81 s warm: ~0.48 s support
planning, ~0.14 s reads, ~0.14 s ghost application and ~0.02 s tracing. A warm
run prepares 659 owners and loads 13,740 support blocks. Repeated oblique 32²
LOS with 512 prepared / 256 support slots spends ~14.26 s: ~6.87 s planning,
~5.38 s reader callbacks, ~1.88 s ghost application, ~0.016 s rays. Packing is
~0.025 s. These nested timings are attribution, not additive independent costs.

Question: can a bounded interior-value cache remove repeated support reads, and
how does allocating those bytes to complete prepared blocks compare? Implement
one optional cache behind the existing provider, default disabled. It binds one
ordered field selection at a time and invalidates all entries on selection
change. New values require a new source lifecycle. Preserve checked-reader
liveness/identity checks even on cache hits, and never publish failed reads.
Compare equal upper budgets, full numerical outputs, cold/warm time, support
loads, prepared owners and resources. Spatial seed order and LOS tile sizes are
existing scheduling parameters to probe before adding scheduling APIs.

Persistent geometry-plan handoff: measured support planning is an upper bound
on removable plan work, not a promised speedup. A reusable plan must separate
source-value identity from topology/strategy/field-role identity, expose support
IDs before value reads, allow current slot binding, and preserve all-request
preflight/failure semantics. No persistent plan implementation is owned here.

The initial dense profile covers all 22,614 leaves, complete one-halo curl output
(542,736,000 bytes) and two 128² slices: 32.74 s including checksum delivery;
16.56 s planning, 10.69 s reader, 4.58 s ghost application, 0.204 s derivative
kernels, 0.168 s packing and 0.056 s final output copy. Peak controlled bound
583,954,233 bytes. Instrumentation leaves numerical hashes unchanged across
repeats; timings include transient checks/plans inside ghost application.

R1 implementation under test: optional `value_cache_capacity` on `open_source`
and `make_source`, default zero. File sources now validate context liveness and
(dev, ino, size, mtime_ns, ctime_ns) before prepared-cache hits as well as reads.
Modified files require a new source; detached products remain valid snapshots.
Array-backed sources retain their explicit immutable-owner contract. Focused
checks cover selection changes, failed miss publication, eviction, file changes
with identical geometry, closed sources and detached values. Cache storage,
miss staging and selectors are included in source/pool admission.

## Probe R2: Task Size And Native Parallel Dispatch (Selected)

R1 also shows that a 600-slot complete-block pool holds this F working set:
590 initial preparations, then zero fills and ~0.02 s complete repeated traces.
That opens a meaningful compute-bound case. Compare the existing thread-pool
backend with an optional native OpenMP dispatch over independent seeds/rays,
using identical row kernels and per-row arithmetic. Also measure the same
backends on bounded preparation-heavy flows to expose any complete-flow limit.
One/four-thread scaling alone will not be relabeled as a backend comparison.

Keep OpenMP opt-in at build and invocation; default builds and thread-pool calls
remain usable. Static row partitioning is the initial native candidate; dynamic
row scheduling is considered only for demonstrated path-length imbalance. The
first implementation must factor private scalar/stack state per row (no shared
C arrays), preserve accepted-prefix/twist/trajectory semantics and permit no
parallel cache mutation. Compare 1/2/4 workers and both cold and warm execution.
Process/device backends are not selected without a bottleneck they can remove.
