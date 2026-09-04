# Native Selected Refined Curl Evidence

## Outcome And Boundary Review

ROI-001, OPR-003, RHC-001, and LFE-001 close the M1 selected local-field
vertical slice. A contained half-open physical box becomes ascending refined
leaf/local-cell windows. One bounded DAT-003/RHE traversal completes the
selected magnetic primaries, a fixed per-slot-spacing Cartesian curl writes
only those windows, and RED-001 adds one chosen component in exact
leaf/x/y/z order. No uniform grid, full-leaf geometry table, derivative
temporary field, retained payload, or dataset lifecycle is introduced.

Independent review approved the four-way decomposition after one correction:
the OPR-003 reference must invoke OPR-002 one slot at a time, or in groups with
bitwise-identical spacing, before the three OPR-001 differences. The final
review approved finite/signed-zero bit comparison, nonfinite classification and
sign comparison, mixed spacing, and all four signed-zero derivative pairs. A
second read-only LFE audit found no implementation defect and added 30 random
mixed-depth comparisons with arbitrary nonempty windows, reordered/repeated
fields, and minimum/resident capacities; every curl and persistent sum matched
the independent full-halo/per-leaf reference bit for bit.

## Correctness, Failure, And Integration

The complete extension build, 168 focused/dependency checks, and all 905
rewrite tests pass. The focused evidence includes:

- exact ROI center arithmetic and lower-inclusive/upper-exclusive ties,
  empty/thin/partial/full windows, mixed depths, ascending omission, overflow,
  aliasing, and pre-mutation validation; the two-pass compiled scan retains only
  its exact `56*S` result arrays;
- OPR-003 mixed per-slot spacing, affine fields, smooth second-order convergence,
  six one-slot OPR-002 plus three OPR-001 reference composition, all twelve
  neighbor-NaN cases, all signed-zero derivative pairs, translated output,
  empty slots/boxes, reach, alias, and atomicity;
- RHC descriptor normalization/freezing, every declared mutable output alias,
  six read-only callback arrays, exact offsets/IDs, one synchronous callback per
  completed prefix, bounded multi-chunk order, empty reader conformance without
  an empty callback, full later-chunk preflight before I/O, and reader/callback/
  non-`None` failure propagation;
- unchanged public RHE values, writer behavior, statistics, and failure
  boundaries, plus bitwise/statistically unchanged RPS trilinear sampling after
  its private hook was replaced by RHC;
- LFE empty/single/partial/multiple chunks, maximal equal-window runs, unequal
  windows, mixed levels, every SAME/FINER/COARSER/PHYSICAL relation, mixed
  physical masks, boundary modes 0--3, repeated/reordered file fields, compact
  sentinel preservation, exact serial reduction, and later-reader completed-
  prefix semantics; and
- exact counts for cells/values/chunks/readers/consumers/operators/reductions,
  primary/support loads, logical bytes, maximum slots, and raw managed arrays.
  LFE adds exactly `72*P + 112` bytes to the unchanged RHC workspace.

Commands:

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_roi_001.py rewrite/tests/test_opr_003.py rewrite/tests/test_rhc_001.py rewrite/tests/test_lfe_001.py rewrite/tests/test_opr_001.py rewrite/tests/test_opr_002.py rewrite/tests/test_red_001.py rewrite/tests/test_rhe_001.py rewrite/tests/test_rps_001.py rewrite/tests/test_dat_003.py rewrite/tests/test_dat_003_integration.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
```

## Standard Synthetic Refined Profile

The standard Apple arm64/Python 3.11.14 profile uses three measured repetitions
after one warmup. Its deterministic affine fixture has 216 leaves at levels one
and two, `4^3` cells per block, three fields, and a native non-staggered v5 file.
The native file is decoded/bound in 0.599 ms and its eager current-reader copy is
331,776 bytes. Native/array outputs, capacity variants, and manual serial sums
are bitwise identical in every case. Native payload bytes equal LFE logical
reader bytes exactly; record headers add 24 bytes per selected load.

| ROI | Primary cells/leaves | Capacity | Loads (support) | Native payload | Native median | Instrumented first result | Managed arrays |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 56 / 8 | 57 | 112 (104) | 172,032 B | 28.611 ms | 25.785 ms | 416,656 B |
| small | 56 / 8 | 216 | 84 (76) | 129,024 B | 28.185 ms | 27.604 ms | 1,535,698 B |
| medium | 2,096 / 56 | 57 | 415 (359) | 637,440 B | 270.156 ms | 238.739 ms | 420,112 B |
| medium | 2,096 / 56 | 216 | 180 (124) | 276,480 B | 266.842 ms | 271.893 ms | 1,539,154 B |
| full | 13,824 / 216 | 57 | 603 (387) | 926,208 B | 888.103 ms | 804.858 ms | 431,632 B |
| full | 13,824 / 216 | 216 | 216 (0) | 331,776 B | 879.090 ms | 881.415 ms | 1,550,674 B |

ROI scan medians are 0.118/0.110/0.096 ms for small/medium/full and are below
0.5% of small first-result latency, so the hierarchy-pruned ROI reopen gate is
not activated. The fused curl and reduction themselves take only tens to
hundreds of microseconds; the synchronous halo-consumer stage accounts for
nearly all wall time. Larger capacity removes repeated support reads but raises
managed memory. The retained bounded strategy therefore exposes the intended
memory/I/O trade-off instead of selecting one capacity as universally best.

At capacity 57, native medians are 6.5%, 2.7%, and 0.9% above the resident-array
path for small, medium, and full. This is accepted because native execution
materializes no 331,776-byte backing field array and reads exactly the selected
payload. Full capacity reduces selected loads substantially but costs about
1.54--1.55 MB of managed arrays versus about 0.42 MB at capacity 57.

## Real TDM Local Field

The real `data/tdm.dat` run is non-staggered Cartesian v5 with 27 level-one
leaves, `10^3` cells per block, and magnetic file fields `[4,5,6]`. Index and
binding take 0.853 ms. Current eager input materializes 648,000 bytes in
4.243 ms; the current dataset derived-current path opens, loads/exchanges two
ghost layers, and materializes three components in 9.532 ms and retains a
1,296,000-byte six-field interior array.

| ROI | Cells/leaves | Loads (support) | Native median | Array median | Output allocation | LFE managed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 64 / 1 | 27 (26) | 2.458 ms | 1.823 ms | 24,008 B | 1,213,732 B |
| medium | 4,096 / 27 | 27 (0) | 34.736 ms | 34.207 ms | 648,008 B | 1,215,604 B |
| full | 27,000 / 27 | 27 (0) | 35.124 ms | 35.052 ms | 648,008 B | 1,215,604 B |

Native and eager-array LFE values/sums are bitwise equal. On cells not requiring
any halo (`1 <= q < B-1`), all 192 small, 5,184 medium, and 41,472 full curl
values are also bitwise equal to the current dataset's batched derived-current
result despite the recorded operation-tree difference. Every native query reads
648,000 payload plus 648 record-header bytes because the present all-26 closure
selects the whole `3^3` level-one domain. Thus the small ROI has 27x support-load
and 15.625x block-cover amplification even though its compact output is only
1,536 logical bytes.

Benchmark command and ignored raw record:

```text
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/lfe_001.py --profile standard --dat data/tdm.dat --output rewrite/benchmark-results/lfe-001-standard.json
```

## Retained Strategy And Reopen Triggers

Keep the exact two-pass ROI selector, fixed fused curl, explicit synchronous RHC
boundary, equal-window run grouping, compact output, and persistent serial sum.
The ROI hierarchy alternative remains closed. Generic recipes still require a
second scientific operator, and parallel reduction cannot weaken the exact
serial-order contract.

The measured local-field consumer activates the direction-projected support
question: its one-cell curl reach does not require every edge/corner relation,
yet the current all-26 closure reads the whole tdm domain for one leaf. Evaluate
that together with repeated-header/payload caching in the separate streamline
workload, whose direction sequence and temporal reuse can choose the boundary
cleanly. Do not silently alter RHE/RPS support semantics.

No repository fixture is both genuinely refined, non-staggered, Cartesian 3D,
and native. The synthetic file proves native refined integration, real tdm proves
the supported current comparison, and the earlier WENO regular-field bridge
retains real refined field bits without relabeling its staggered origin. This
limitation remains explicit for the M1 horizon review.
