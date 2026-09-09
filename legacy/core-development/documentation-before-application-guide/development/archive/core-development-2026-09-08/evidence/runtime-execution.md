# Runtime Execution Evidence

Status: this runtime round is locally complete, 2026-09-08. Scope and decisions are in
[runtime execution](../runtime-execution.md); the live checkpoint is
[current](../current-before-freeze.md). Source checkpoints: `775a0b9` (initial cache/backend
candidates), `ce97fed` (recoverable scheduling/feedback/staging experiments).
Validated selected implementation: `6172bbb`.

## Conditions And Comparability

The original `data/weno509_sub_0000.dat` is 1,045,232,320 bytes, 22,614 native
8³ leaves, seven ordinary fields plus staggered tails. This round reads ordinary
B or rho without decoding the tails. Rho LOS is a supplied-scalar diagnostic,
not thermal AIA emission. All comparisons retain two primary halo layers, the
same reconstruction, accepted-prefix RK4, centered extended curl and Gauss2 LOS.
Numerical arrays, statuses, accepted paths or complete output hashes are checked;
cache-miss counts may change because they describe execution rather than physics.

Host: macOS arm64, eight logical CPUs, 8 GiB RAM, Python 3.11 / NumPy 2.4 / Cython
3.2. Limits are four compute workers, 2 GiB controlled live arrays and 2 GiB
scratch, with at least 2 GiB free disk. Dense preparation comparisons additionally
fit a common **1 GiB** admitted envelope. Cache comparisons keep source/consumer
requirements fixed and compare closely matched active allocations; those bounds
include source/mesh, cache, provider scratch and output reserves. RSS is reported
separately; it is not the managed-array bound. A source-open transient and the
active consumption footprint are distinct lifetime phases.

"Cold" means new numerical caches, not a flushed OS page cache. Uncontrolled OS
caching, a heterogeneous CPU and desktop activity affect absolute latency. Later
rows showed substantial drift even for an unchanged one-worker control. Small
cross-run differences are not adopted as speedups. An interrupted backend run
that overlapped a build is retained as `runtime-paired-backends-interrupted.json`
and excluded from performance conclusions. Builds, checks and accepted timing
runs are otherwise sequential.

## Where The Original Bounded Work Goes

These instrumented timings are nested wall measurements, not independent values
to add. "Ghost application" still includes invocation-local coarse-support
planning. Reader time includes checked transfer/conversion. Future wait time
includes worker computation and synchronization, not just wasted CPU.

| Complete consumer case | Wall | Support planning | Reader callbacks | Ghost application | Consumer kernels | Packing / final copy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F, 256 shuffled central seeds × 128 steps; warm pool 256, support 128 | 0.81 s | 0.48 s | 0.14 s | 0.14 s | 0.020 s | 0.0066 s packing |
| Oblique 32² LOS; repeated view, pool 512, support 256 | 14.26 s | 6.87 s | 5.38 s | 1.88 s | 0.016 s | 0.025 s packing |
| Full curl (542,736,000 bytes) + two 128² slices | 32.74 s | 16.56 s | 10.69 s | 4.58 s | 0.204 s curl; 0.013 s slices | 0.168 s packing; 0.056 s sink copy |

Raw files: `runtime-f-profile.json`, `runtime-l-profile.json`,
`runtime-d-profile.json` in `benchmark-results/analysis-core/`.

These costs reject preparation/consumer double buffering as the first local
optimization: perfect overlap removes at most the very small consumer portion
of these bounded cases. They justify preparation-process and numerical-reuse
experiments. Persistent geometry plans are a separate session's responsibility;
planning time is only an upper bound on their possible saving.

## Finite Values, Request Order And Reuse

Two reversed-order rounds of the original sparse F profile used the same seeds,
output order and 32,768 accepted steps. Warm medians:

| Choice | Wall | Pool/source/scratch bound | Meaning |
| --- | ---: | ---: | --- |
| 256 complete slots, input order, batches 32 | 0.825 s | 28,280,569 B | Reference |
| Same storage, spatial seed order | 0.649 s | same | Less repeated work; restore original order via seed IDs |
| Same storage, batches 256 | 0.674 s | same | Larger simultaneous request groups |
| 256 complete slots + 1,024 raw-interior slots | 0.735 s | 42,711,465 B | Fewer support reads, still repeated ghost work |
| 600 complete slots | 0.0202 s | 42,571,705 B | All 590 required owners fit; zero warm preparation |

The 600-slot cold consumer median was ~0.867 s. This is not a parallel speedup.
A larger support workspace paid for by shrinking the complete-block pool was
slower on this sparse case. Raw files: `runtime-f-*-r0.json` and `*-r1.json`.

The optional interior-value cache is retained for explicit I/O reduction, default
off. In repeated 32² LOS it reduces requested file-transfer bytes from about
695 MB to 117 MB, while wall time changes only slightly (~14.3 to ~13.9 s).
A dense curl pass drops about 978 MB to 355 MB with 2,048 raw slots, again without
an equally large time saving. Raw/support and complete-ghost retention must not
be treated as interchangeable. `read_value_bytes` now counts loaded underlying
values; `requested_value_bytes` counts all support requests. Earlier R1 records
used the old logical-request interpretation of `read_value_bytes`; their explicit
file-reader byte counters are the comparison source.

### Lifetime Checks

Cache compatibility binds source lifecycle, ordered fields, strategy and halo
validity. Raw cache field-selection changes clear its keys. File contexts check
(dev, inode, size, mtime_ns, ctime_ns) and liveness on preparation and prepared
cache hits. New values require a new source/pool, even with identical geometry.
Array sources retain the immutable-owner contract, optionally enforced by a
raising `validate_values` hook. Detached products remain independent snapshots.
Tests cover changed values with identical geometry, closed contexts, selection
changes, bounded eviction and failed-read publication.

## Nearby LOS Views

Three distinct nearby oblique 32² views deliver 731,542 samples and all three
images. Same 512 complete slots, no raw value cache, no parallel execution:

| Organization | Cold / repeated wall | Prepared owners, cold / repeated |
| --- | ---: | ---: |
| Original view-major, tile 16 | 39.22 / 37.07 s | 28,855 / 28,472 |
| Tile 4 interleaving, original request-touch policy | 37.44 / 36.35 s | 30,411 / 29,964 |
| Tile 4 interleaving, non-touching coverage lease | 19.05 / 17.53 s | 13,768 / 13,768 |
| View-major tile 4 with non-touching leases (control) | 40.29 / 40.42 s | 28,707 / 28,707 |

All complete numerical outputs have the same signature
`2b9f734b8570470640fafb7dcfe7e39932c296744a58051214deb5d5c784e7b5`.
The selected policy is scoped to **multi-view tile interleaving**. F and
single-view LOS retain their original request-priority behavior. The active
three-image bound is about 24.05 MB; no source field or ghost validity is reduced.
A larger 640-slot pool used about 25.52 MB, prepared 10,651 owners and reduced CPU
work further, but its ~18.2 s wall results did not establish an additional latency
win under the changing host load.

Actual-access feedback was tested and rejected for this round. It still prepared
836--838 warm sparse-F owners versus the original 659 and created roughly 750
fragmented provider calls. Staging restored batched calls but not better reuse.
On nearby LOS it reduced owner work only ~2% versus the simpler selected policy.
The extra hit buffers and padded staging were removed; `ce97fed` preserves the
candidate. Single-image tile shrinking alone was not adopted.

## Independent Dense Preparation

Every result covers all 22,614 leaves and both slices, with identical full output
signature `ef359f8fde41c5b864fe9255d8f5371d3500133cbe6a494535b42851315faef2`.
Timing includes file/metadata opens, preparation, curl, process startup/IPC where
applicable, output assembly, slices and checksum delivery. Each call starts new
numerical state; repeat calls do not retain process/provider caches.

| Execution | Complete wall observations | Controlled upper bound |
| --- | ---: | ---: |
| Serial existing path | 31.30 s initially; 34.71 / 32.98 s later | 583,954,233 B |
| Independent threads, 2 workers | 30.65 / 30.09 s | 734,745,065 B |
| Processes, 1 worker, tasks 512 | 41.07 / 39.00 s | 643,651,184 B |
| Processes, 2 workers, tasks 512 | 17.63 / 17.53 s | 734,745,065 B |
| Processes, 4 workers, tasks 512 | 12.97 / 13.14 s | 916,932,827 B |
| Processes, 2 workers, tasks 2,048 | 21.50 / 20.87 s | 1,029,657,065 B |

Adopt explicit process preparation, with two workers as the conservative default
and 512-leaf tasks. Four workers improve wall time at higher memory/CPU cost.
Do not recommend independent-thread preparation or the tested larger task size.
The process path submits at most `workers` tasks, owns every provider independently
and releases completed futures before admitting more. IPC staging is included;
there is no full-domain shared-memory object or hidden unbounded result queue.

For two processes, parent CPU was ~1.2 s and child CPU ~25.3 s, versus ~23--24 s
serial CPU. Thus lower latency costs more CPU. Parent waiting (~17 s) overlaps
worker work; it is not 17 s of avoidable synchronization. Result sink copying was
~0.05--0.06 s. Parent RSS peaked near 626 MB, child process peaks ~151--174 MB each
in those runs; summing per-process high-water marks is only a conservative RSS
indicator, not a simultaneous unique-memory measurement.

The existing resident bulk path remains preferable for dense reuse when its
larger full-primary residency fits. The process result is an additive bounded
preparation option, not a replacement for all resident workflows.

## Actual Consumer Backend Comparisons

The optional native engine runs the same per-seed/per-ray functions through
OpenMP; private row state and ordered pixel sums are unchanged. This is a real
engine comparison, unlike the earlier OpenMP build-only check. The default
thread pool remains available. OpenMP was tested with `OMP_WAIT_POLICY=PASSIVE`
and `OMP_DYNAMIC=FALSE`; requested worker counts were 1/2/4.

On ready WENO B, 4,096 seeds with at most 512 steps and batches of 256 took
1.014 / 0.543 / 0.291 s through the thread pool and 1.020 / 0.542 / 0.291 s through
static OpenMP. Larger batches of 4,096 brought four-worker times to 0.275 / 0.273 s.
There is no useful evidence for replacing the tracing backend with static OpenMP.

On a ready 256² oblique rho image, tile 64, the corresponding thread-pool times
were 0.408 / 0.242 / 0.133 s. Static OpenMP was essentially equal; dynamic OpenMP
measured 0.407 / 0.210 / 0.114 s. Thus the useful difference is load distribution,
not merely compiling with OpenMP. Thread-pool dynamic overpartitioning was also
implemented and compared; it can cost more for small tasks. It is explicit,
never an automatic default.

A final **64 distinct view** comparison rotated all three backend orders for
each view under changing desktop load, checked every complete image, and streamed
all image arrays through the same checksum. It used one shared file-to-ready
preparation, then measured whole image calls plus delivery for each engine:

| Four-worker engine, tile 64 | Measured image work + delivery | Shared preparation + work estimate |
| --- | ---: | ---: |
| Thread pool, static | 9.738 s | 14.479 s |
| Thread pool, dynamic | 9.135 s | 13.876 s |
| OpenMP, dynamic | 7.880 s | 12.621 s |

These totals include all public image work, not just a hot interpolation kernel.
The right column adds the common measured 4.741 s preparation; it is **not** three
independent disk-cold trials. Identical output signature:
`c5a7223bb8baf50111b1bd348c9f2b4a84a4fe44c8e6fc2cae1933977e5ed866`.
The final range dispatcher therefore improves this complete-cost estimate by
about 13% over static threads. Its active managed upper bound is 456,581,984 bytes;
process RSS peaked at 564,920,320 bytes. Keep OpenMP opt-in; no general backend
replacement or promised speedup for preparation-bound queries is adopted. The
earlier `runtime-paired-backends-final.json` records the same signatures under
heavier desktop load (58.25 / 55.09 / 49.46 s complete-cost estimates); its absolute
latencies are not machine-wide baselines.

Raw files: `runtime-ready-f.json`, `runtime-ready-l.json`,
`runtime-ready-l-thread-dynamic.json`, `runtime-paired-range-final.json`.
The separate late dynamic-thread run's one-worker drift is not used to infer
speedup. A pre-round Python/Cython whole-query control is checked separately.

## Interface Handoff And Remaining Limits

Persistent geometry plans should expose support IDs before value reads, accept
fresh slot bindings and keep topology/strategy/field-role identity separate from
source-value lifetime. They must preserve checked preflight and failure behavior.
Measured planning time bounds possible savings; it does not predict the other
session's implementation. AIA response, thermodynamic definitions and rebricking
are deliberately outside this round's implementation.

Other CPUs, GPU/device engines, cross-machine execution, 10--20 GB input, and
larger-than-RAM real-file performance remain unverified. No synthetic enlargement
is offered as that evidence. This round does not revalidate the previous million-
seed or billion-sample scale profiles, and it does not claim thermal emission.

## Regression And Installation Checks

The final dispatcher passes **contiguous ranges** to one native computation loop.
The serial/thread-pool path calls that loop once per chunk; static OpenMP supplies
one range per worker and dynamic OpenMP supplies eight-row ranges. Scalar/stack
state remains private. This avoids a large array-argument call for every ray or
seed and keeps the default chunk computation close to the original implementation.

A separate module was built from the pre-round `bdfda30` Cython source, with its
original Python F/LOS consumers loaded under separate names. Paired complete
queries use identical ready WENO inputs and alternate old/new order. All result
hashes agree. Under matching default non-OpenMP builds, final median timings:

| Query | Before | Selected | Median paired selected/before |
| --- | ---: | ---: | ---: |
| F, 2,048 seeds, at most 256 steps | 0.338 s | 0.298 s | ~0.97 |
| LOS, one 256² oblique image | 0.357 s | 0.370 s | ~1.028 |

These support no material default-path regression on the checked cases, not a
new tracing speedup claim. Earlier highly variable and cross-build comparisons
are retained as `runtime-regression-row-dispatch-noisy.json` and
`runtime-regression-range-openmp.json`; use `runtime-regression.json` for the
matching default-build check. The final bounded F check restores exactly 886 cold
and 659 warm preparations, with the original result signature; warm wall was
0.830 s. The final nearby-view check again prepares 13,768 owners and matches
all three original images (wall 23.14 s in that run, illustrating host variation).

Validation completed on the selected source:

- 23 composed native checks with OpenMP enabled, and 23 with the default build.
  Coverage includes mixed AMR geometry, accepted trajectories/twist, LOS task
  scheduling, cache lifetimes and failure publication, complete threaded/process
  curl delivery, and budget rejection before touching a caller sink.
- Full `make test PYTHON=.venv/bin/python` with default compilation.
- 92 canonical public/AMR/helper checks passed; two opt-in heavy tests deselected.
- Fresh-source default wheel: 9,367,791 bytes, built in 54.48 s. Isolated `python -S`
  execution imports both packages from the extracted wheel and performs original-
  file tracing with a value cache, detached sampling, spawned-process global curl
  and multiple LOS views, without editable checkout hooks. Raw stage:
  `benchmark-results/analysis-core/source-build-id9v95_8`.
- The final range-backend comparison completed, the default non-OpenMP extension
  was restored, and its explicit unavailable-backend/dynamic-thread check passed.
  Final native build info is `enabled=False`, `openmp_version=0`.

The reused provider implementation itself was not changed in this round; its
1,232-test result is prior integration evidence, not falsely reported as a new
run. No million-seed or billion-point rerun was needed for these scoped changes.

## Reproduction

Run timing campaigns sequentially, separately from builds/tests:

```sh
PYTHONPATH=scripts .venv/bin/python scripts/analysis_core/compare_runtime.py f
PYTHONPATH=scripts .venv/bin/python scripts/analysis_core/probe_parallel_preparation.py \
  --backend process --workers 2 --repeats 2 \
  --output benchmark-results/analysis-core/recheck-process.json
PYTHONPATH=scripts .venv/bin/python scripts/analysis_core/probe_los_views.py \
  --order tile --tile 4 --repeats 2 \
  --output benchmark-results/analysis-core/recheck-views.json
.venv/bin/python build.py --inplace --group analysis --openmp
OMP_WAIT_POLICY=PASSIVE OMP_DYNAMIC=FALSE PYTHONPATH=scripts \
  .venv/bin/python scripts/analysis_core/probe_paired_backends.py --frames 64
SIMESH_OPENMP=0 .venv/bin/python build.py --inplace --group analysis
```

For pre-round regression, retrieve `native.pyx`, `field_lines.py` and `los.py`
from `bdfda30` into `benchmark-results/analysis-core/runtime-native-reference`,
name them `native_before.pyx`, `field_lines_before.py`, `los_before.py`, and build
`native_before` as a Cython extension with `-O3` and the source's original
directives. Then run `PYTHONPATH=scripts .venv/bin/python
scripts/analysis_core/probe_runtime_regression.py`. Historical feedback/staging
commands require checkpoint `ce97fed`; those options are deliberately absent
from the final API. Preserve existing dirty documentation when making any
isolated historical comparison.
