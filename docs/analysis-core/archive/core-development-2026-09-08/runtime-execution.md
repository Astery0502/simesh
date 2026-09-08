# Runtime Execution Round

Status: locally complete, 2026-09-08. The user instruction reopened execution backends,
parallel scheduling and finite numerical reuse under the existing native layout.
Read [current](current-before-freeze.md), [development](development.md),
[prepared fields](prepared-fields.md) and [performance](performance.md).

## Scope And Evidence Separation

Previous one/four-worker F and LOS measurements use the **same Python thread
pool calling GIL-free kernels**. They establish scaling for those workloads, not
backend selection. The OpenMP build and 23 AMR checks establish build/behavior
compatibility, not matched backend performance. The previous 64/256-slot trace
comparison measures avoided preparation, not parallel acceleration. Dense bulk
versus bounded preparation comparisons have different residency and are not
same-budget backend comparisons.

This round measured sparse F, dense whole-domain curl plus slices, and repeated
scalar LOS with unchanged numerical requirements and output. AIA response,
rebricking and persistent geometry plans were developed independently and are
now integrated in the same checkout; see [current](current-before-freeze.md). Their evidence
remains separate from these runtime measurements. Planned preparation can use
the retained raw-value cache and the same source-lifetime checks; it does not
automatically replace pool miss scheduling or thermal ray dispatch.

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

Validated selected source: `6172bbb`. Default non-OpenMP is the restored build.
Checkpoint `ce97fed` preserves all feedback/staging experiments before their
selective rollback; `775a0b9` preserves the initial cache/OpenMP candidates.
The selected source keeps process preparation, optional raw-value caching and
multi-view tile retention. Actual-hit tracking and padded miss staging were
removed after comparison. Final range dispatch, backend controls, default-build
checks and isolated-wheel execution passed. Default non-OpenMP is restored.
See [runtime evidence](evidence/runtime-execution.md) for compared cases and
limits. No fixtures or previous evidence products should be deleted.

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

## Probe R3: Parallel Preparation For Dense Delivery (Selected)

The dense profile spends ~16.6 s in Python-involved planning versus ~0.2 s in
GIL-free derivatives. Native row dispatch cannot address this bottleneck.
Compare serial preparation with two independent thread/provider tasks and two
spawned process/provider tasks. Each task opens and validates the same immutable
file, prepares a bounded contiguous leaf range, computes the unchanged curl and
returns that owned range. The coordinator admits at most `workers` tasks, writes
disjoint result ranges and performs both slices only after full output completion.

Use bounded returned arrays initially; include serialization, source startup,
coordination and final copies in complete time. Do not add shared-memory lifetime
machinery unless transfer is measured to dominate. Workers share the job budget,
not a budget each. Compare task sizes only for a concrete startup/load-balance
question. Existing `global_curl(source)` remains the reference and usable path.
This is runtime task scheduling within the current leaf layout, not rebricking or
persistent geometry-plan reuse.

## Probe R4: Nearby LOS Views With Tile-Local Value Reuse (Selected)

The single-view tile-4 probe increases preparations (10,488 versus 9,876 cold)
without improving latency; do not adopt smaller tiles for a single image. Each
small tile, however, visits far fewer owners than a complete view. Test three
nearby full-domain views, keeping all three requested images, first sequentially
and then interleaving matching tiles across views within the same 512-slot pool.
Use identical per-view pixel-coordinate arithmetic, ray integration and output
order; only task order changes. Compare tile-4 and original tile-16 view-major
execution. This reuses completed numerical ghosts, not geometry plans or output
memoization. Admit all images and one active tile together; use one executor for
the entire group, with no simultaneous cache mutation.

R4's first tile-interleaved attempt did **not** reduce numerical preparations:
30,411 cold / 29,964 warm owners for three views. Inspection found an execution
recency problem: each F/LOS advance borrows every resident leaf to expose coverage,
and that lease marks every slot equally recent. At capacity, deterministic ties
keep evicting low slots while old unrelated slots remain. Coverage publication
is not evidence that every available block was sampled.

Selected correction: preserve normal request-touching borrows, add a non-touching
coverage lease for F/LOS, and timestamp successful missing preparations. This
uses insertion age between explicit accesses when native consumers cannot report
all hits; do not call it exact sample-level LRU. No numerical kernel, output or
memory capacity changes. Re-run representative sparse F, single LOS and nearby
views against the original recency behavior before promoting the correction.

R4 recency correction passes focused borrowing/F/twist/LOS/file-task checks.
The three nearby-view images retain the exact same complete SHA-256 signature
before and after the change (`2b9f734b...e7b5`). At the same 512 prepared slots,
tile-4 interleaving falls from 37.44/36.35 s to 19.05/17.53 s (cold/warm), and
30,411/29,964 preparations to 13,768/13,768. These are avoided numerical work,
not parallel speedup. A corrected-recency view-major control is also required.

Corrected-recency view-major tile-4 still costs 40.29/40.42 s and prepares 28,707
owners. Thus the 19.05/17.53 s nearby-view result comes from tile-local reuse once
coverage leases stop destroying recency. It does not establish a universal
single-view improvement. Eviction can now leave noncontiguous destination slots;
current preparation invokes the provider separately for each contiguous run.

R1 follow-up: compare optional bounded miss staging, which prepares sorted
requests together then scatters complete values to cache slots, against spending
the same storage on more complete prepared slots. This can deduplicate support
across fragmented destinations. Default staging remains disabled; dense full-slot
passes should not pay an unnecessary copy. Admit staging before allocation, and
publish no new keys if any stage of a miss request fails.

## Actual Access Feedback Follow-Up

Insertion age alone regressed shuffled short-batch F: ~1.07 s warm versus the
~0.83 s original median, with 845--847 versus 659 preparations. Spatial F still
prepared only 590 owners. Before promoting the recency change, add bounded
private per-task/per-OpenMP-lane slot-use bytes to the native consumers. The
coordinator merges those hit sets only after every worker returns and updates
ages before the next miss. This preserves actual hot blocks without pretending
that every published block was accessed. Feedback storage is admitted explicitly;
no worker mutates cache keys, values or recency. This is recency per advance, not
per individual scalar sample. The OpenMP build completed; behavior and workload
rechecks are next after the interrupted turn.

Some late timing rows experienced a changing host load (one-worker LOS increased
from ~0.41 s to ~0.65 s, despite identical samples and output hashes). Do not use
that cross-run change to judge a backend. Use matched/interleaved controls for
small differences; retain all raw rows and distinguish clear algorithmic gains
from uncertain latency differences.

Actual-hit F feedback preserved all numerical outputs but fragmented destination
slots severely: ~750 provider calls per warm trace, versus ~42 originally.
Preparing 838 owners still exceeds the original 659. The final bounded follow-up
pairs actual feedback with 32-slot staging on F and nearby LOS to distinguish
fragmented preparation from eviction quality. Do not promote precise-access
feedback merely because it sounds more accurate; preserve the cheaper existing
F heuristic if complete-work evidence favors it. Candidate source remains
recoverable before rollback.

## Selected Cache Outcome

Checkpoint `ce97fed` preserves actual-hit feedback and staging candidates, with
24 composed checks passing. They are **not promoted**. Precise per-advance hit
feedback still prepares 836--838 warm F owners (original 659) and fragments cache
slots into ~750 fill calls; staging reduces calls to 54 but not the excess owner
work. Nearby LOS gains only ~2% fewer preparations than the simpler insertion-age
variant (13,527--13,542 versus 13,768), with extra tracking/staging. Late wall times
are load-affected, so the decision also uses unchanged work counts and added
storage/complexity. Both candidates were removed from the selected source.

Preserve the original request-priority heuristic for F and single-view LOS.
Only multi-view tile-interleaving uses non-touching coverage leases, retaining
recent completed blocks between nearby views. Explicit direct borrows still
update recency. There is no universal claim that one eviction policy is best.
No actual-hit bitmap or extra padded staging is allocated in the selected path.

## Final Local Disposition

- Adopt explicit raw-interior reuse for I/O-constrained cases, default disabled;
  compare its bytes with retaining complete prepared blocks. File value/context
  changes reject even cached preparations. Geometry is never a value epoch.
- Retain the existing sparse-F/single-view behavior. Use affordable complete-block
  retention and existing seed-ID/spatial grouping/batch controls where measured.
- Adopt known multi-view tile scheduling with insertion-age coverage leases;
  it does not predict arbitrary future interactive views or cache whole images.
- Adopt independent-process dense preparation as an explicit file entrypoint;
  two workers/tasks of 512 are conservative defaults, with measured four-worker
  scaling. Keep serial and independent-thread comparators.
- Keep thread-pool consumers as default. Dynamic OpenMP is optional for retained
  repeated LOS; static OpenMP is not a tracing replacement. The final range
  dispatcher preserves one serial chunk loop and private parallel ranges.
- Roll back per-advance hit maps and generic padded staging. Do not add automatic
  shared-memory storage, device engines or preparation/compute overlap without
  a new bottleneck. Persistent plan ownership stays with the other session.

All selected source changes have behavior and local efficiency evidence. Final
checks cover both build modes, the pre-round default query, complete file/process
products and a fresh installed wheel. There is no outstanding local runtime task
in this round. Overall thermal/large-input acceptance, other hardware, and the
separately scoped storage-ownership follow-up are not marked complete here.
