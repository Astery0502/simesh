# Native Refined Field-Line Evidence

## Outcome And Contract Review

FLN-001, RKS-001, TRM-001, and SLE-001 close the M1 native field-line vertical
slice. CHS-001 supplies exact cached refined vector samples. FLN maps them to a
unit-arclength tangent and oriented `integral(B dot dx)` RHS, RKS supplies fixed
classical RK4 arithmetic, TRM classifies nonperiodic results, and SLE schedules
ascending active seeds by RK stage into caller-owned accepted prefixes.

No repository-owned tracer exists in the current, legacy, archive, tests, docs,
or reachable history. An old external `yt.LinePlot` is a fixed straight segment,
not field following. These are explicit new contracts; parity claims remain
limited to LOC/SAM/RHE/RPS sampling semantics.

Independent design review approved the four-way split after two corrections.
First, every augmented RK stage checks all four `(x,y,z,I)` components for
nonfinite state before coordinate-domain exit. Second, SAM-005 may reject a
finite owned point whose stencil arithmetic is unrepresentable. CHS now raises
an indexed `UnrepresentableRefinedSampleError` during its atomic dynamic
preflight; SLE removes that seed, records `UNREPRESENTABLE_SAMPLE` at the exact
K1--K4 marker, and retries the untouched compact batch. The exception carries
the exact rejected-plan raw-array bytes so SLE's sampler peak includes failed
attempts. Other data-dependent CHS/backend/resource errors propagate with prior-
prefix semantics.

A read-only implementation audit and adversarial probes approved stage
compaction, hint alignment, retry indexing, all-four classification, candidate
commit, busy restoration, failure prefixes, counters, histogram identities, and
the exact `314*N` scratch formula.

## Numerical And Failure Evidence

The clean extension build, 301 focused FLN/RKS/TRM/SLE/CHS/RPS/native-DAT
checks, and all 1,210 rewrite tests pass. Coverage includes:

- FLN scalar/checked/unchecked bit equality over 800 random exponent-scaled
  vectors; both signs/axes; scale invariance; smallest subnormal, minimum normal,
  maximum finite and overflowing norms; all signed-zero triples; NaN/infinity in
  every component; mixed statuses; failed-row preservation; tangent/dot
  invariants; complete alias/layout/atomicity;
- exact RKS half/full/final binary64 trees over empty/single/batch, cancellation,
  signed zero, subnormal/large/nonfinite values, read-only inputs, atomic errors,
  constant RHS exactness, and isolated rotational fourth-order convergence;
- exhaustive TRM exact faces/nextafter values, all augmented nonfinite positions
  and precedence collisions, all 64 signed-zero coordinate pattern pairs,
  integral-only no progress, RHS maps/invalid bytes, and checked/unchecked truth
  tables;
- SLE empty/exterior/max-zero, exact forward/back constant paths and oriented
  integrals, rotation convergence away from an AMR interpolation error floor,
  K2 domain compaction, every numerical status/stage, multiple indexed sample-
  preflight removals, cache/backend/capacity/refined/PBC invariance, tail and
  alias preservation, reentrancy, and exact stats/memory; and
- first/later K1--K4 reader failures: previous accepted positions/integrals and
  terminal seeds persist, the current candidate/tails do not change, successful
  cache admissions remain, active controls identify the incomplete call, and
  the session returns idle.

Commands:

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_fln_001.py rewrite/tests/test_rks_001.py rewrite/tests/test_trm_001.py rewrite/tests/test_sle_001.py rewrite/tests/test_chs_001.py rewrite/tests/test_rps_001.py rewrite/tests/test_dat_003_integration.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
```

## Kernel And Analytic Standard Profile

The standard Apple arm64/Python 3.11.14 run uses seven kernel repetitions and
8,192-row batches. FLN's mixed status batch contains equal OK/zero/nonfinite/
unrepresentable quarters; checked and unchecked values/statuses are exact.

| Kernel | Rows | Checked median | Unchecked median | Unchecked throughput |
| --- | ---: | ---: | ---: | ---: |
| FLN RHS | 8,192 | 0.586 ms | 0.082 ms | 100.3 Mrow/s |
| RK half stage | 8,192 | 0.036 ms | 0.034 ms | 244.2 Mrow/s |
| RK full stage | 8,192 | 0.036 ms | 0.036 ms | 230.2 Mrow/s |
| RK finish | 8,192 | 0.146 ms | 0.143 ms | 57.2 Mrow/s |

Singleton unchecked medians are 0.625 us for FLN, about 0.458 us for an RK
stage, and 0.792 us for RK finish. Checked calls retain zero traced bytes with
sub-kilobyte peaks. The kernels are not the composed bottleneck.

For constant `B=(2,0,0)`, six `h=0.03125` steps produce exact forward/backward
positions and oriented integrals `+0.375/-0.375`. For the unit-tangent rotational
field wholly inside one coarse leaf, `h=0.05/0.025` position errors are
`1.681e-6/1.129e-7`, observed order 3.896; integral errors are
`7.326e-7/4.877e-8`.

## Synthetic Stage-Major Workflows

The native synthetic v5 fixture has nine mixed level-one/two leaves and `B=4`.
Three measured repetitions compare native/array and every tested cache capacity
bitwise/exactly. Capacity zero is limited to short paths because it intentionally
rebuilds one completed halo per stage.

| Workflow/cache | Accepted steps / samples | Hits / misses | Loads | Native bytes | Cleared median | Warm median | Session / SLE scratch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one seed, 2 steps, cap 0 | 2 / 8 | 0 / 8 | 64 | 99,840 B | 17.558 ms | 17.257 ms | 78,242 / 314 B |
| one seed, 2 steps, cap 1 | 2 / 8 | 7 / 1 | 8 | 12,480 B | 3.072 ms | 1.036 ms, 0 B | 83,442 / 314 B |
| 8 co-located, 32 steps, cap 1 | 256 / 1,024 | 127 / 1 | 8 | 12,480 B | 14.468 ms | 12.664 ms, 0 B | 83,442 / 2,512 B |
| 32 divergent, 2 steps, cap 1 | 64 / 256 | 0 / 32 | 240 | 374,400 B | 90.700 ms | 90.368 ms | 83,442 / 10,048 B |
| 32 divergent, 2 steps, cap 4 | 64 / 256 | 28 / 4 | 30 | 46,800 B | 13.170 ms | 1.880 ms, 0 B | 99,042 / 10,048 B |
| 32 divergent, 2 steps, cap 9 | 64 / 256 | 28 / 4 | 30 | 46,800 B | 13.179 ms | 1.906 ms, 0 B | 125,042 / 10,048 B |

All one-seed stage hints after the first are hits. For co-located seeds, the
eight initial hintless samples are the only hierarchy fallbacks; 1,016 later
hints hit. For divergent seeds, 32 initial fallbacks and 224 hits remain exact.
The sampled owner-transition count on a successful sampled run is therefore
`hierarchy_fallback_count - interior_seed_count`; max-steps-zero performs no
location and has both counts zero.

Stage-major batching shares each CHS call across active seeds, while cache
capacity must cover the concurrent owner working set. Capacity 1 is sufficient
for one/co-located owners but thrashes across four divergent owners. Capacity 4
reduces divergent cleared loads and native bytes 8x and warm runtime 48x;
resident capacity 9 gives no material benefit.

## Real TDM Native Field Lines

The real non-staggered Cartesian v5 tdm run traces three seeds starting in
owners `[0,13,26]` for eight `h=1/15` steps. Every capacity returns the same 24
accepted steps, 96 stage samples, final trajectory/integral bits, and three
`MAX_STEPS` results. Hints hit 93/96 samples; the three fallbacks are initial
locations, not transitions.

| Cache | Hits / misses | Loads | Header / payload | Cleared median | Warm median | Session / output / scratch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 / 96 | 896 | 21,504 / 21,504,000 B | 188.513 ms | 188.445 ms | 1,255,134 / 894 / 942 B |
| 3 | 93 / 3 | 28 | 672 / 672,000 B | 8.814 ms | 2.957 ms, 0 B | 1,338,110 / 894 / 942 B |
| 27 | 93 / 3 | 28 | 672 / 672,000 B | 8.805 ms | 2.945 ms, 0 B | 2,333,822 / 894 / 942 B |

The exact concurrent-owner cache knee is three; resident capacity retains about
996 KB more without runtime or I/O benefit. Capacity 3 cuts cold native bytes
32x and warm bytes to zero. It reads 24,914 physical bytes per accepted point on
the cold call; warm execution is pure cached sampling/RK scheduling.

Benchmark command and ignored raw record:

```text
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/sle_001.py --profile standard --output rewrite/benchmark-results/sle-001-standard.json
```

## Items Left For The M1 Horizon Review

The completed trajectory evidence activates, but does not implement, the
direction-projected support question. With the correct three-entry tdm cache,
three cold owner fills still select 28 loads and account for about 5.86 ms of the
8.81 ms cold-versus-warm difference. Any projection must separately prove
COARSER slope closure and mixed physical widening; it cannot weaken all-26 RHE
or be hidden in CHS/SAM.

A raw-interior cache can at most collapse those 28 tdm loads to the 20-row union
observed by one-batch RPS, saving 192,000 payload bytes while retaining every RHE
plan/application. Evaluate that smaller benefit against direction projection at
the horizon rather than adding a second cache now. Exact neighbor transition is
also deferred: after HLO hints, the measured trajectory fallbacks are initial
locations only. Adaptive RK and independent-seed parallelism now have stable
fixed-step/error and stage-major scaling baselines but remain later strategies.

The repository still lacks a genuinely refined, non-staggered, native Cartesian
3D real fixture. Synthetic native refined trajectories and real level-one tdm
provide the supported evidence; the earlier WENO bridge's staggered origin must
remain explicit in the milestone review.
