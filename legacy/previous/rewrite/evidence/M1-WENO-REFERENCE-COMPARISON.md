# M1 WENO Reference Comparison

Date: 2026-09-07. Status: reference norm and selected M1 assessment complete,
including canonical, full-domain halo, sampling, cache and trajectories.

The principal new finding is an efficiency gap in full-domain execution:
canonical resident halo refresh is about 0.31 s, versus 38.20 s for bounded RHC
including input reads and output copies. Numerical comparison passes. This
does not erase M1's selected-query gains or original numerical completion, but
bounded RHC is not a competitive resident full-domain refresh strategy here.

## Development Rule Adopted

The user selected WENO as the classic complex data example for feature-completion
assessment. [WENO-REFERENCE](../WENO-REFERENCE.md) freezes the dataset, requests,
comparison classifications, budgets and explicit commands. ANALYSIS_WORKLOADS
owns its workload role; WORKFLOW owns applicability and completion timing;
PERFORMANCE owns comparable costs and evidence. AGENTS/README link the profile.
It is not registered in pytest, Make, scripts or CI. Ordinary edits/commits do
not trigger it. Synthetic accuracy/error tests remain necessary.

This changes development evidence selection, not numerical contracts or product
scope. No capability was added, no computational implementation changed, and
M2 is not activated. Existing documentation drafts were preserved.

## Assessment Scope

Original simesh means the canonical `src/simesh` core at unchanged source
revision `7b0c1b1`; it is not an early rewrite or a legacy reimplementation.
Original M1 is `397aaffc61be68d1141c8f30c0427dd37a4c8bc8`; HPR and retained CQP
variants restore the appropriate private preflight/cache planners for paired
comparisons. The executable engine/build remains unchanged.

Reuse [the previous local-field assessment](M1-WENO-ASSESSMENT.md): four fixed
ROIs, capacities 57/256, 778,512 output values, independent interior reference,
reader equivalence and component attribution. New runs fill the canonical
comparison, repeated queries, cache working sets and short-trajectory gaps.
Existing kernel/manufactured-field evidence is reused; no full historical audit.

WENO is the same 22,614-leaf, levels 3--6, B=8 Cartesian AMR fixture. Original
regular fields (4,5,6) are copied into a non-staggered bridge. Each process records
its bridge creation separately (6.25--6.94 s, 278.42 MB read, 279.07 MB written).
This remains a format qualification: direct staggered input is unsupported.
OS page cache is uncontrolled, so cleared/warm refers to application state.

Canonical measurements deliberately compose the actual AMRForest/AMRMesh core
with three fields. They do not measure Dataset registry materialization, which
has additional allocation/lifecycle costs. Full maximum-resolution output,
legacy duplication, periodic/2D features, new diagnostics and writer/export are
outside this assessment; the fixture cannot certify all those product scopes.

## Numerical Comparisons

| Compared outputs | Values | Maximum absolute difference | Classification |
| --- | ---: | ---: | --- |
| Mixed ROI common width-one halo, M1 versus canonical width-two buffer | 1,209,000 | 1.11e-15 | restricted-common-domain; within 1e-12 |
| Small ROI curl | 12,000 | 0 | bitwise equal |
| Mixed ROI curl | 391,032 | 7.46e-14 | within frozen 5e-13 absolute/relative tolerance |
| Thin ROI curl | 374,544 | 0 | bitwise equal |
| Physical ROI curl | 936 | 0 | bitwise equal; no interior-only oracle in this ROI |
| 512-point common grid, zero order | 1,536 | 0 | bitwise equal |
| 512-point common grid, linear | 1,536 | 1.39e-15 | within frozen 1e-10 interpolation tolerance |

All 305,628 safe-interior curl values match canonical bits. The common halo has
47,545 differing floating bit patterns and mixed curl 14,081; these are recorded
roundoff-sized cross-implementation differences, not silently declared bitwise
parity. Linear samples differ in 822 bit patterns, including safe interiors,
consistent with different coordinate/arithmetic evaluation. All finite/nonfinite
classifications agree. Tolerances were fixed before measurement.

CHS owner/value arrays match RPS and all three M1 variants exactly, across cache
capacities and histories. Eight-seed/eight-step trajectories match all output
bits, integrals, terminations and stats: nine points per seed, MAX_STEPS for all.
Original simesh has no corresponding native cache/tracer contract, so no legacy
trajectory parity is claimed. Real-field agreement is not an absolute physical
accuracy oracle; previous manufactured-field evidence retains that role.

## Component Costs

Five measured repetitions after warmup; medians below. Counts/work volumes are
part of each comparison. These independently measured components do not sum to
a measured cold Dataset workflow.

| Component | Canonical simesh | M1 counterpart and interpretation |
| --- | ---: | --- |
| Metadata read | 4.01 ms | bridge index/bind/balance about 31.9--35.0 ms combined |
| Forest plus eager connectivity | 18.80 ms | different work: rewrite validates balance without retaining the same neighbor tables |
| Three-field eager read | 4.999 s | selective bridge reads counted by request; original-input bridge cost is additional |
| Mesh allocation/geometry | 182.23 ms | canonical padded full-domain storage; selected GEO costs are in reused LFE evidence |
| Interior copy | 103.10 ms | native reader copies directly into chunk interiors |
| Full-domain width-two ghost refresh | 650.79 ms | mixed selected width-one RHC: 491.02 ms; different work, no direct speedup ratio |
| Full-domain derivative components | 113.57 / 118.63 / 113.26 ms | original kernel, one component at a time including padded-output initialization; M1 selected curl arithmetic about 1.7--1.8 ms in mixed/thin ROIs |

The prior LFE evidence remains unchanged: COARSER-bearing ROIs improve about
28--30% from HPR, while the same-level thin query gains essentially nothing.
Canonical bulk kernels remain an important efficiency baseline; bounded M1's
planning/checking work can be large compared with the useful arithmetic.

## Full-Domain Same-Width Control

Both paths complete all 22,614 leaves, three fields and two halo layers, using
continuous physical modes. Canonical input is resident; M1 reads the regular-
field bridge at capacity 256 and copies completed rows into the supplied full
output. The output allocation is shared between sequential variants to fit the
budget, not shared with the immutable reader. One numerical pass precedes three
timed M1 runs, with a separate attribution pass.

- All 117,230,976 output values pass the frozen 1e-12 comparison; maximum
  absolute difference is 7.33e-15. There are 6,508,408 roundoff-sized differing
  bit patterns. No tolerance or numerical strategy was changed.
- Canonical wall samples: 0.745/0.309/0.301 s, median 0.309 s; process CPU median
  0.304 s. The first sample is an outlier, so no tight speedup ratio is claimed.
- M1 wall samples: 34.940/40.246/38.202 s, median 38.202 s; CPU median 27.382 s.
  Shared-runner scheduling adds material wall time, but cannot explain the large
  CPU-cost gap. First completed output occurs after 19.70--25.08 s.
- M1 performs 304 chunks and 77,040 loads for 22,614 distinct requested leaves:
  946.668 MB payload plus 1.849 MB headers, with 154,080 pread calls. Output copy
  alone takes 0.55--1.23 s in timed runs.
- Separate instrumented times: action preflight 23.345 s; complete preparation
  23.992 s (includes preflight); reading 5.055 s; application 5.136 s. These are
  not all additive, and application includes Python scheduling as well as value
  kernels. Repeated planning/check work is a demonstrated primary bottleneck.
- RHC managed arrays are 11,120,754 bytes; the caller's full padded output is
  937,847,808 bytes. Canonical coarse storage is 277,880,832 bytes. The explicit
  controlled upper bound is 1,627,827,200 bytes, under 2 GiB; RSS peaks at
  1,278,459,904 bytes.

This is a full requested-value comparison between resident and bounded storage
strategies, not an isolated arithmetic-kernel ratio. It establishes that the
bounded executor should not silently replace the efficient resident path for
full-domain work. Omitting the bridge or output from cost accounting would not
resolve the measured preflight gap. No new software optimization is attempted
as part of this assessment.

## Sampling And Cache Feasibility

The common mixed-box (16,8,4) grid uses exactly the same 512 target centers:

| Strategy/state | Query latency | Preparation and retained storage |
| --- | ---: | --- |
| Canonical resident zero order | 0.075 ms | full fields already loaded |
| Canonical resident linear | 0.084 ms | full width-two halos already completed |
| M1 uncached RPS linear | 202.32 ms | each call plans/reads; 172 owners, 50 chunks, 2,379 loads at capacity 57 |
| M1 completed-owner cache, first query | 350.91 ms | completes 172 owner halos individually; bridge excluded |
| M1 completed-owner cache, warm | 0.655 ms | 172 hits, zero reads/fills; 5,633,826 session bytes |

These distinguish query state rather than presenting the resident/cold ratio
as kernel speedup. RPS can be preferable for a one-shot batch; CHS amortizes
preparation for repetition with much less retained field storage than canonical
whole-domain residency. The uncached RPS timing has substantial wall dispersion
(about 23%); retain raw wall/CPU samples and avoid tight ratio claims for it.

The coherent 192-point line has eight owners. At eight cache slots, retained
CHS is approximately 23.9 ms cleared and 4.33 ms warm, with zero warm reads.
At four slots it reloads eight owners per traversal; zero capacity reloads
54 owner groups. The 144-point divergent batch has 36 owners: 18 slots thrash,
while 36 slots reach zero warm reads and approximately 0.399 ms warm latency.
HPR-only warm timing there is 0.427 ms; CQP's incremental benefit is modest in
this real workload, not the larger synthetic-case gain applied universally.

## Trajectory Costs

The CPU-recorded paired follow-up uses the same eight seeds, eight steps and
one-quarter-minimum-cell step length. It supersedes noisy wall-only timing for
these rows; numerical results remain identical.

| Cache slots | Retained cleared | Retained warm | Cleared / warm payload read |
| --- | ---: | ---: | ---: |
| 4 | 700.47 ms | 694.62 ms | 83.362 / 83.362 MB |
| 8 | 25.14 ms | 3.757 ms | 2.605 / 0 MB |
| 16 | 25.89 ms | 3.762 ms | 2.605 / 0 MB |

At eight slots original M1 is 34.94 ms cleared; HPR is 25.29 ms and retained is
25.14 ms. This is chiefly HPR plus cache working-set fit, not a new RK method.
An extra four entries require only 96,064 cache bytes yet avoid severe thrashing
in this short case. Longer trajectories can have a different working set.

Separate warm attribution (4.56 ms instrumented) assigns about 29.7% to dynamic
array validation, 3.2% to validation/location, 6.0% to point grouping, 5.0% to
cache access planning, and 11.5% to owner sampling boundaries. RHS plus RK4 is
about 1.2%. Parent sampler time includes child stages and is not additive.
Cold attribution is dominated by owner completion and sampling preparation;
it is instrumented, not a headline query-time sample.

## Complete Resources

Canonical padded payload is 937,847,808 bytes and coarse workspace 277,880,832;
one derivative output adds 312,615,936 bytes. The stage admits a conservative
1,873,334,272-byte live-buffer bound including metadata/scratch headroom; actual
RSS high water was 1,682,964,480 bytes on the 8 GiB runner. C-allocated buffers
are counted; NumPy-only tracing would miss them. Public registry expansion and
whole-domain 512x512x256 output are not included or claimed feasible here.

Bounded sampling's inventoried simultaneous arrays peak at 14.59 MB, including
the three comparison sessions; trajectories peak at 10.56 MB. The cache-grid
and CPU-recorded trajectory controls admit conservative complete bounds of
about 49 MB, including metadata, query/output arrays and a 32 MiB allowance for
transient backend/Python storage. They fit 512 MiB. Bridge disk fits 512 MiB.
RSS, allocator retention and OS file cache remain separate quantities; no
larger-than-RAM/out-of-core claim is made.

## Next Decisions

1. Preserve an explicit resident strategy where full fields fit and many dense
   queries or full halo refreshes follow. The full-domain control makes this
   essential to future feasibility/cutover review. The original canonical
   strategy has different roundoff behavior; do not silently substitute it into
   a strict M1 arithmetic contract without the required strategy review.
2. Choose query batching and cache capacity from observed owners. Avoid calling
   zero-cache CHS repeatedly when an RPS batch is available; avoid capacity below
   the trajectory working set. Do not equate a cleared cache with cold disk.
3. Reopen ROI-aware halo targets and invocation-local geometry/check reuse from
   the measured thin-query and full-halo costs. Evaluate compiled or amortized
   preflight and application orchestration, retaining independent references.
   Keep source-slot guards, slope closure, physical-base dependencies and
   ordinary-failure guarantees. Kernel arithmetic alone does not address 23 s
   of action checks, nor does a raw cache eliminate this cost.
4. For warm traces, investigate equivalent reuse of executor-owned array/alias
   proofs and batched owner sampling before arithmetic parallelism. Standalone
   checked wrappers and numerical step acceptance must remain intact.
5. Treat ordinary-field reading inside staggered records as a separate future
   storage contract if WENO is a routine input: the 6--7 s bridge dominates
   initial use. Do not remove the rejection without validating record tails.

These are assessment recommendations, not selected implementations or new
completion claims. They do not start M2 or reopen every historical capability.

## Evidence And Checks

The profile's explicit commands reproduce the selected sections. Raw records
under ignored benchmark-results: `weno-canonical.json`, `weno-sampling.json`,
`weno-trajectory-cpu.json`, `weno-cache-grid-final.json`, `weno-attribution.json`,
`weno-halo-full.json` (three repeats, explicit expensive control);
earlier wall-only/partial records remain available. Environment is macOS arm64,
Python 3.11.14, NumPy 2.4.4, Cython 3.2.4, clang -O2, no OpenMP. No extension
source changed, so the existing forced build is reused.

Fresh focused checks: 92 pass across DAT integration, CHS, RPS and SLE. The
previous local-field assessment's 31 checks and the optimization round's full
1,229-check gate remain historical evidence, not freshly rerun claims.
No expensive test hook or numerical implementation was added.
