# Bounded Refined Repeated Point Sampling Evidence

## Outcome And Review

LOC-001, SAM-004, SAM-005, and RPS-001 provide the first complete refined
point consumer without uniform-grid materialization. Finite points are located
once, equal leaf owners are stable-grouped, zero order reads only owner
interiors, and trilinear execution samples each RHE-001-completed primary prefix
synchronously before its bounded workspace is reused. Numerical kernels see
only canonical payload plus explicit slot-to-point groups; reader, grouping,
capacity, and future cache policy remain outside them. SAM-002/SAM-003 remain
the fused block-centric uniform strategies.

Independent contract review approved the four-way decomposition and the private
RHE terminal-consumer implementation. It rejected the initial closed-upper
domain proposal after a separate history/current audit supplied a concrete
GEO-valid large-origin case in which exact `domain_upper` produces
`j0=B,j1=B+1` under the frozen SAM-003 arithmetic. LOC-001 therefore uses
`[domain_lower,domain_upper)`, rejects nonfinite queries atomically, and assigns
every internal exact face/edge/corner to the positive side. No clamp or new
endpoint interpolation rule was introduced.

Independent implementation reviews found no locator, sampler, RPS, or public
RHE defect. The reviews requested stronger standalone repeated-slot, payload-
bit, alias, uniform-reduction, RPS failure, PBC, and relation coverage; every
requested case was added before the closing gate. A 40-forest randomized LOC
challenge plus a level-63 all-high-child chain also matched the independent
all-leaf scan.

## Correctness, Atomicity, And Composition

The final build, 98 group-focused plus SAM-002/SAM-003/RHE regression tests,
and all 765 accumulated rewrite tests pass. Evidence includes:

- independent canonical all-leaf scan versus flat hierarchy descent at every
  root/child/coarse-fine face and `nextafter` neighbor;
- the large-origin upper-endpoint counterexample, exterior sentinel,
  nonfinite rejection, level 63, overflow, subnormal spacing, alias, and output
  atomicity;
- scalar owner-level global-face cell search, arbitrary/repeated slot and point
  maps, deterministic last write, exact signed-zero/NaN payload copying, and
  unselected-row preservation;
- fixed z/y/x eight-load trilinear arithmetic, including zero-weight NaN
  behavior, affine error bounds, invalid stencils, and exact one-cell reach;
- bitwise level-one reductions to SAM-002/SAM-003 for native full-domain,
  upsampled, downsampled, and subdomain output-center sequences;
- bounded zero reads containing unique owners only, trilinear capacity
  invariance across nonzero chunk offsets, exact stats/managed bytes, array and
  non-array readers, all-exterior zero-capacity calls, and first/later reader
  failures;
- one RPS/RHE composition whose relation set is exactly PHYSICAL, COARSER,
  SAME, and FINER, includes mixed physical masks, exercises boundary mode codes
  0/1/2/3, freezes reader IDs, and matches public-RHE-plus-SAM resident bits;
- read-only private consumer views, exception/non-`None` behavior, complete
  pre-I/O preflight, and unchanged public RHE empty-writer call, values, order,
  stats, and failure behavior.

The safe current comparison uses a dyadic refined `2^3` root mesh, one refined
root, continuous boundaries, `4^3` interiors, and 960 non-tie output centers.
Zero order is bitwise identical. Trilinear differs in 264 of 1,920 field values
because the rewrite retains explicitly rounded SAM-003 arithmetic; worst error
is `4.656612873077393e-10`, relative `3.637633042492621e-16`, and two ULP.
Current overlapping exact-face windows and rounded `rnode` faces remain
documented differences and were not used as a tie oracle.

Commands:

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_loc_001.py rewrite/tests/test_sam_004.py rewrite/tests/test_sam_005.py rewrite/tests/test_rps_001.py rewrite/tests/test_sam_002.py rewrite/tests/test_sam_003.py rewrite/tests/test_rhe_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
```

## Standard Repeated-Point Performance

The standard run uses Apple arm64, Python 3.11.14, NumPy 2.4.4, Cython 3.2.4,
a `4x4x4` root forest with one refined root, 71 leaves, `4^3` interiors, and
selected field positions `[2,0,2]`. Seven raw repetitions are retained in the
ignored `benchmark-results/rps-001-standard.json`. The clustered query has
4,096 points in four leaves; the scattered query has one point in every leaf.
Every bounded result is bitwise equal to its resident point-kernel comparator.

| Query | Stage | Capacity | Median | Throughput | Reads/loads | Amplification | Managed bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clustered `P=4096,U=4` | LOC checked batch | -- | 0.088 ms | 46.33 Mpoint/s | -- | -- | output only |
| clustered | stable group plan | -- | 0.110 ms | -- | -- | -- | included below |
| clustered | SAM-004 resident kernel | -- | 0.157 ms | 78.25 Mvalue/s | -- | -- | caller payload |
| clustered | SAM-005 resident kernel | -- | 0.299 ms | 41.15 Mvalue/s | -- | -- | caller payload |
| clustered | RPS zero | 8 | 0.404 ms | 30.40 Mvalue/s | 1 / 4 | 1.00 | 71,800 |
| clustered | RPS zero | 71 | 0.386 ms | 31.82 Mvalue/s | 1 / 4 | 1.00 | 71,800 |
| clustered | RPS trilinear | 57 | 4.201 ms | 2.93 Mvalue/s | 1 / 26 | 6.50 | 481,624 |
| clustered | RPS trilinear | 71 | 4.155 ms | 2.96 Mvalue/s | 1 / 26 | 6.50 | 580,156 |
| scattered `P=U=71` | LOC checked batch | -- | 0.023 ms | 3.14 Mpoint/s | -- | -- | output only |
| scattered | stable group plan | -- | 0.006 ms | -- | -- | -- | included below |
| scattered | SAM-004 resident kernel | -- | 0.050 ms | 4.23 Mvalue/s | -- | -- | caller payload |
| scattered | SAM-005 resident kernel | -- | 0.061 ms | 3.52 Mvalue/s | -- | -- | caller payload |
| scattered | RPS zero | 8 | 0.321 ms | 0.66 Mvalue/s | 9 / 71 | 1.00 | 14,616 |
| scattered | RPS zero | 71 | 0.129 ms | 1.65 Mvalue/s | 1 / 71 | 1.00 | 111,384 |
| scattered | RPS trilinear | 57 | 79.875 ms | 0.003 Mvalue/s | 2 / 106 | 1.49 | 418,296 |
| scattered | RPS trilinear | 71 | 82.439 ms | 0.003 Mvalue/s | 1 / 71 | 1.00 | 516,828 |

Checked single-point locator latency is 20.7 us; future per-step execution will
use an already validated inner boundary rather than repeat the Python contract
for every coordinate. Clustered time to first sample is 0.331 ms for zero order
and 3.964 ms for trilinear at the smaller capacities. The scattered zero path
shows the explicit trade-off: capacity eight uses about 7.6x less managed
memory but nine reader calls and 2.48x the full-capacity runtime. The bounded
trilinear path saves about 19% managed memory; full capacity removes 35 support
reloads, while runtime is dominated by checked refined-halo planning rather
than transfer bytes at this small synthetic payload.

The representative traced trilinear call retains 1,464 bytes and peaks at
498,880 traced bytes versus 481,624 contracted managed-array bytes. Process
peak RSS is 47,218,688 bytes. Caller points/results, immutable metadata, and
backend page cache are excluded from contracted managed bytes as specified.

Benchmark command:

```text
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/rps_001.py --profile standard --output rewrite/benchmark-results/rps-001-standard.json
```

## Retained Fused And Public-RHE Regression Checks

SAM-002 and SAM-003 were rebuilt at detached checkpoint `0c5c426` and at the
group state with the same interpreter/compiler, then run sequentially on the
existing standard matrices. Raw JSON is retained under ignored names
`sam-002-shared-{baseline,current}.json` and
`sam-003-shared-{baseline,current}.json`. SAM-002 current/baseline ratios range
from `0.481` to `1.010` for standalone cases and are `1.003/1.035` for the two
bounded cases. SAM-003 ratios range from `0.956` to `1.006` standalone and are
`1.034/0.999` bounded. No representative median regresses by 10%; exact and
numerical results remain unchanged.

The affected public RHE path was also run once on the full 22,614-leaf WENO
metadata after its private-core extraction. Small/medium/full times are
0.078/0.835/30.550 seconds with exactly the prior calls, loads, and 1,824,978
managed bytes. The small one-shot result is noisy relative to the earlier
0.061-second median; medium/full improve from 1.020/33.474 seconds. The
instrumented medium run is 0.716 seconds, with action preflight 0.582 seconds
(81.3%), matching the previously recorded dominant stage rather than revealing
a new wrapper regression.

Commands:

```text
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/sam_002.py --repeats 21
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/sam_003.py --repeats 15
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/rhe_001.py --dat data/weno509_sub_0000.dat --capacity 128 --small 32 --medium 512 --fields 3 --block 4 --halo 2 --warmups 0 --repeats 1 --pwa-repeats 5 --output rewrite/benchmark-results/rhe-001-post-rps-smoke.json
```

## Retained Strategy And Reopen Triggers

Flat hierarchy descent is retained because it reuses FST state, has fixed
per-point scratch, and reaches 46 Mpoint/s once validation is amortized over a
coherent batch. Owner grouping is retained because 4,096 clustered points need
only four zero-order reads or 26 complete-halo support loads. The bounded RHE
consumer avoids a second completed-block copy and preserves public RHE.

No persistent payload cache, prefetch, direction projection, last-leaf hint,
neighbor transition, or streamline stepper is introduced. The native-reader
and streamline slices are the concrete reopen consumers. In particular, the
observed 6.5x all-26 support amplification and scattered checked-halo planning
cost require those later profiles to test direction-specific support and
last-leaf/neighbor/cache reuse. A retained BVH/hash is reopened only if flat
hierarchy fallback remains material after those higher-priority reads and halo
costs are controlled. Point-plan streaming is reopened only if `O(P+U)` plan
bytes become material relative to caller points/results.
