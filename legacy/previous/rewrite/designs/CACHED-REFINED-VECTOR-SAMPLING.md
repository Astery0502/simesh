# Cached Refined Vector Sampling Design

## Problem And Outcome

RPS-001 is the correct bounded batch sampler, but a data-dependent field-line
integrator cannot know all future points. Calling its checked locate/group/RHE
composition once per RK stage repeats metadata validation, all-26 support
planning, halo construction, and selected reads even while the path remains in
one leaf. The first streamline slice therefore needs an explicit reusable
sampling session before it can define stepping or termination.

This group produces:

```text
ordered points + last-owner hints
  -> HLO-001 exact hint-or-hierarchy ownership
  -> CHS-001 completed-owner halo LRU
  -> unchanged SAM-005 trilinear values
```

HLO-001 owns only the exact hinted locator strategy. CHS-001 owns one serial,
bounded completed-halo caching and sampling strategy over an immutable reader/
forest/field/PBC lifecycle. Neither capability defines a field-line ODE, RK
method, termination, trajectory output, raw-block cache, direction-projected
support, prefetch, concurrency, or dataset ownership.

## Disposable Decision Probe

A `/tmp`-only probe used a balanced Cartesian 3D forest with 71 leaves, `B=8`,
three fields, RHE capacity 57, and 192 ordered points. Every candidate result
was bitwise equal to public per-point RPS.

- A coherent path visited five owners in runs with median/max length 48.
  Per-point RPS took 583.1 ms, selected 4,032 loads, and moved 49.55 MB
  logically. A one-entry completed-halo cache took 20.18 ms, hit 187/192
  points, completed five owners, selected 99 loads, and moved 1.217 MB.
- Four owners visited round-robin made a one-entry cache ineffective: 515.1 ms
  versus 536.2 ms for RPS. Four completed entries took 11.80 ms, hit 188/192
  points, completed four owners, selected 102 loads, and moved 1.253 MB.
- One completed `B=8` three-field padded owner is 24,000 bytes versus 12,288
  bytes for its interior. Four entries retain 96,000 payload bytes. A miss still
  needs about 1.50 MB of RHE workspace and dominates miss-heavy execution.
- Checked singleton LOC costs about 32 us/point and unchecked LOC about
  2.2 us/point. Once halo work is cached, repeated boundary validation is
  material. The coherent last-owner hit rate is 97.9%; it is zero for the
  round-robin order.

The probe activates a retained completed-owner artifact and session-level
prevalidation. It does not justify a raw-interior cache: that lower cache cannot
remove the measured RHE planning/application, and its native reuse/capacity knee
has not been measured after completed-halo hits.

## Alternatives And Selected Boundaries

Calling RPS for every point is the simple correctness reference but repeats the
dominant work. Stage-major RPS batches amortize across many active seeds but do
not help a single coherent path and still rebuild the same owner across stages.
A full uniform grid changes interpolation and memory semantics. A raw block LRU
removes only backend transfers. A completed-owner halo is the highest reusable
artifact that preserves LOC-001/SAM-005/RHE-001 meanings and eliminates both
support reads and halo application on a hit.

The cache is a separate capability from streamline scheduling. Its provenance,
lifetime, admission/eviction, byte budget, invalidation, failure atomicity, and
warm-history behavior vary independently of RK order and seed scheduling. The
session fixes reader, exact ordered magnetic field IDs, forest/geometry,
one-cell reach, boundary modes, normal slots, and RHE capacity. Within that
immutable lifecycle an entry key is only canonical leaf ID.

HLO-001 is separate because last-owner testing and hierarchy descent are a
locator strategy, not cache policy. It computes the hinted leaf's canonical
faces directly, accepts the same lower and rejects the same upper faces as
LOC-001, and falls back to the unchanged hierarchy descent. No neighbor table is
materialized. CHS may call its prevalidated unchecked boundary after all dynamic
points and hints pass one call-level validation.

CHS groups interior points by ascending owner as RPS does. It looks up/touches
one owner group at a time. A hit samples the retained primary with SAM-005. A
miss uses one persistent RHE workspace to preflight, read, and complete exactly
that owner and samples synchronously. Admission invalidates a valid victim key
immediately before copying the completed payload, then publishes the new key
only after the full copy succeeds. An invalid slot is chosen before the
least-recent slot; recency ties use the lowest slot. Capacity zero is the
uncached numerical reference and never admits an entry.

The cache budget bounds entry-dependent arrays, not RHE workspace:

```text
entry bytes = 3 * product(B + 2) * 8 + 16
capacity    = min(L, floor(cache_byte_budget / entry_bytes))
```

The additional 16 bytes are one `int64` leaf key and one `int64` recency value.
Fixed session arrays, reusable RHE workspace, call plans, outputs, borrowed
metadata, backend page cache, and process RSS are reported separately.

## Direction Projection And Other Deferred Strategies

The probe's optimistic direct-relation lower bound is compelling but not yet an
implementation contract. Along the coherent path, 87.5% of stencils are wholly
inside their owner and the others cross one axis; a direct relation union would
average 1.19 loads versus the measured all-26 21. The divergent corner case
needs seven nonempty direction subsets and has a 9.5-load direct lower bound
versus 25.5. These are not valid end-to-end load claims: COARSER PRL needs CSP
slope sources beyond direct contacts, and mixed physical widening needs complete
base subsets.

Direction-projected support therefore remains a distinct planner/executor
capability. It reopens only if native miss-heavy or cache-thrashing streamline
profiles show material support time/bytes after CHS hits. A raw-block LRU
likewise reopens only if repeated native support transfer remains material; its
key is source identity/leaf/source fields, whereas a completed halo key also
depends on reach/PBC/numerical halo semantics. Neighbor transitions, prefetch,
adaptive steps, and concurrent cache access remain later choices.

## Decomposition Records

### HLO-001

- Responsibility: resolve exact refined point owners by testing one explicit
  prior-owner hint before LOC-001 hierarchy descent.
- Owned decisions: valid hint representation, canonical hinted-face test,
  fallback condition, and exact hit/fallback statistics.
- Non-owned: point ownership/ties, adjacency, payload, cache, sampling,
  stepping, termination, and scheduling.
- Inputs/outputs: borrowed LOC/GEO/FST metadata, finite points, `-1`/valid leaf
  hints, caller-owned owner IDs, and exact counts.
- Mutation/ownership: complete preflight before owner writes; no retention.
- Access/reach: metadata only, one hinted leaf or one root/descent path.
- Reference: LOC-001 owner IDs exactly for every point/hint.
- Producer/consumer: FST/GEO/LOC -> CHS and later streamline execution.
- Performance: `hot-kernel`; singleton/batch coherent/divergent latency and CHS.
- Five questions: yes/yes/yes/yes/yes.

### CHS-001

- Responsibility: serve ordered refined trilinear vector batches through one
  explicit serial completed-owner halo LRU session.
- Owned decisions: immutable session provenance, one persistent RHE miss
  workspace, owner grouping, LRU lookup/admission/clear, cache budget, synchronous
  SAM application, statistics, and external-failure prefix behavior.
- Non-owned: LOC/SAM/RHE numerical meaning, raw cache, direction projection,
  field-line/RK/termination behavior, prefetch, concurrency, fd ownership, and
  dataset lifecycle.
- Inputs/outputs: one borrowed reader and FST/GEO/PBC lifecycle; copied field and
  boundary provenance; owned cache/RHE arrays; dynamic point/hint inputs and
  caller-owned owner/value outputs.
- Mutation/ownership: static and dynamic preflight precede cache/output mutation
  and nonempty I/O; successful earlier owner groups/admissions survive an
  external later miss failure.
- Access/reach: one-cell all-26 RHE completion on misses, retained primary-only
  padded payload on hits, and SAM-005's exact eight-value stencil.
- Reference: public RPS-001 owner/value results with cache capacity zero, plus
  capacity/history/backend numerical invariance; execution statistics differ
  because CHS completes one owner per miss.
- Producer/consumer: DAT/STO/RHC/HLO/SAM -> native streamline stage sampler.
- Performance: `composition-only`; cached sampling profile and next SLE workflow.
- Five questions: yes/yes/yes/yes/yes.

## Group Completion Gate

Build both compiled locator paths, then cover exact hint face ties, exterior and
wrong hints, mixed depth, HLO fallback parity, empty/interior mixes, cache
capacity zero/one/working-set/resident, deterministic eviction/clear, all
refined relation/PBC kinds, repeated/reordered field IDs, native/array readers,
ordinary atomicity, first/later reader failure, cache admission failure boundary,
and exact owner/value/stats/memory equivalence. Run HLO/CHS plus LOC/SAM/RPS/RHC/
RHE/DAT regression, the accumulated rewrite suite, and one standard coherent/
divergent native cache profile. Commit one group evidence summary and checkpoint.
