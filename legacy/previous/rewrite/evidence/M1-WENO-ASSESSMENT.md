# M1 On Real WENO Refined Fields

Date: 2026-09-07. User-requested focused assessment after the optimization round.
No numerical implementation, contract, capability status or M2 work changed.

## Fixture And Scope

The optional canonical tests use `data/weno509_sub_0000.dat`, including
`tests/amrvac/test_amrvac_current_derived_optional.py`. It is block-structured
Cartesian AMR, not an arbitrary unstructured mesh. The file is 1,045,232,320
bytes, with 22,614 leaves, blocks of 8^3 cells, and root shape (2,2,1).

| Leaf level | Leaves |
| --- | ---: |
| 3 | 88 |
| 4 | 646 |
| 5 | 3,256 |
| 6 | 18,624 |

The source is staggered and direct DAT-003 construction correctly rejects it.
Reuse the established DAT-003 benchmark bridge: copy ordinary `b1,b2,b3` fields
(source positions 4,5,6) without floating arithmetic into a temporary
non-staggered file. Preserve and verify the complete forest arrays, and verify
all-touch 2:1 balance. This exercises the real refined grid and real regular
field values, but does not implement staggered/CT computation or a direct
staggered-file reader. The source file is untouched; the bridge is removed
automatically after measurement.

Bridge streaming took 6.327 s, read 278,423,568 source bytes, and wrote a
279,069,936-byte file. Source index/binding took 23.63 ms. These are setup costs,
excluded from query timing. A fresh one-shot workflow must include the bridge;
this report does not claim sparse cold access directly to the original file.
The canonical eager reader took 4.809 s to load the same three original regular
fields into 277,880,832 bytes, used as independent storage evidence. It is a
full-field read, not an equivalent local-curl performance comparator.

## Fixed Assessment

Consumer: existing LFE-001 selected native curl plus serial component sum.
Original M1 preflight: `397aaffc61be68d1141c8f30c0427dd37a4c8bc8`.
Retained implementation: `7b0c1b1`; CQP is not exercised by LFE.
One warmup followed by five alternating baseline/retained repetitions at both
57 and 256 slots. No new performance-promotion threshold or algorithm search.
Fields, halo width one, physical modes, reduction order and outputs are identical.
OS page cache is uncontrolled and already affected by the bridge. Wall and
process CPU samples are both retained; attribution/tracing runs are separate.

Regions are fractional physical-domain boxes, converted to ROI-001 cell-center
windows, with no uniform-grid materialization:

| Region | Fractional lower -> upper | Primaries | Primary levels | COARSER / SAME / FINER directional rows |
| --- | --- | ---: | --- | --- |
| small | (.48,.48,.48) -> (.52,.52,.52) | 32 | 6 | 144 / 688 / 0 |
| mixed | (.42,.42,.42) -> (.58,.58,.58) | 403 | 4:6, 5:65, 6:332 | 1,808 / 8,087 / 583 |
| thin | (.495,.3,.3) -> (.505,.7,.7) | 728 | 6 | 0 / 18,928 / 0 |
| physical | (0,.3,.3) -> (.02,.7,.7) | 8 | 3 | 0 / 168 / 32 |

The physical case has 72 physically masked direction rows and eight pure
PHYSICAL relations. Primary level alone does not establish interface absence:
the small all-level-six selection contacts coarser neighbors.

## Correctness Result

All four cases pass at both capacities:

- All 778,512 selected curl values are finite. Full output buffers, including
  sentinel regions, and accumulator bits match original M1 and across capacities.
- Native bridge curl outputs match execution on canonical eager reads from the
  original staggered file. For every distinct support block actually read,
  direct native payload bits also match that canonical array.
- Original/retained execution stats match at each capacity; reader payload
  bytes match logical transfer accounting. Forest arrays are unchanged.
- The independent scalar curl reference checks 305,628 interior-stencil values
  bitwise, with maximum difference zero. There are no eligible interior cells
  in the physical ROI, so that oracle does not apply to that case.
- The focused DAT integration/LFE/HPR suite passes: 31 tests.

These establish storage/implementation conformance on real refined fields.
They do not establish an absolute physical error against a real-data ground
truth. Coarse/fine and physical output equality here uses original M1 semantics;
the existing manufactured-field tests remain the scientific-accuracy evidence.
The earlier full optional uniform-grid diagnostic was not rerun: it has a
different output and substantial full-grid memory cost.

## Query Timing And Transfers

Medians in milliseconds; byte counts below use decimal MB.

| Region | Slots | Original M1 | Retained | Retained / original | Payload read | Distinct support-read blocks / total loads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 57 | 48.158 | 34.026 | 0.707 | 2.654 MB | 112 / 216 |
| small | 256 | 47.046 | 33.360 | 0.709 | 1.376 MB | 112 / 112 |
| mixed | 57 | 633.071 | 458.935 | 0.725 | 59.916 MB | 896 / 4,876 |
| mixed | 256 | 613.682 | 431.607 | 0.703 | 23.126 MB | 896 / 1,882 |
| thin | 57 | 406.503 | 405.389 | 0.997 | 116.048 MB | 1,792 / 9,444 |
| thin | 256 | 348.530 | 348.053 | 0.999 | 48.808 MB | 1,792 / 3,972 |
| physical | 57 | 5.870 | 5.628 | 0.959 | 0.946 MB | 61 / 77 |
| physical | 256 | 5.617 | 6.166 | 1.098 | 0.750 MB | 61 / 61 |

The two COARSER-bearing cases improve 27.5--29.7%, confirming HPR's benefit
on real refined fields. Wall and CPU ratios agree closely. The all-SAME thin
case gains essentially nothing from HPR, as expected. The physical case has no
COARSER actions; its -4.1%/+9.8% wall differences are not promoted as useful gains.
Retained repetition dispersion is 0.5--3.1% except physical/256 at 4.9%.

At 57 slots, retained preflight is 44.3--66.9% of instrumented wall; reading is
7.8% small, 11.9% mixed, 26.3% thin and 15.3% physical. Disjoint value-kernel
boundaries occupy only 2.6--5.4%, including dispatch. Enclosing halo application
also includes planning and Python scheduling, so it is not pure arithmetic.
The mixed/256 preflight share is 69.6%; thin/256 reading falls to 13.2% while
preflight remains 53.6%. Curl itself is about 1.7--1.8 ms in mixed/thin cases.

## Resources And Actionable Recommendations

Increasing capacity from 57 to 256 costs 5,144,946 extra managed bytes. Managed
workspace ranges from approximately 1.50--1.56 MB to 6.65--6.70 MB. Caller output
is 0.098--8.946 MB. The largest inventoried live arrays plus traced call peak
are 311.55 MB, including the 277.88 MB canonical reference backing. That backing
is a comparison cost, not required by the native consumer. RSS high water was
228.36 MB; RSS and allocated bytes differ and are not interchangeable budgets.
No out-of-core claim or hard-RSS guarantee follows from these measurements.

1. **Use capacity as the first available control.** At 256 slots, mixed reads
   fall 2.59x and total time falls 6.0%; thin reads fall 2.38x and time falls
   14.1%. This needs no new implementation. It is a throughput trade-off, not a
   universal default: instrumented first result is slightly later (mixed
   309 -> 325 ms; thin 199 -> 210 ms), and tiny physical queries do not benefit.
2. **Prioritize ROI-aware halo targets in the next halo design.** The thin case
   produces 2.996 MB of selected values but reads 116.048 MB at 57 slots. Its
   all-SAME relations expose costs that HPR's COARSER-specific proof reuse does
   not address. Propagate actual output-window and per-field stencil access;
   do not simply replace all-26 closure with six faces, since the mixed case
   still needs extra slope and physical-base support. Measure gains on both.
3. **Assess repeated geometry proof next, rather than arithmetic threading.**
   Preflight remains large and the thin ROI has 18,928 SAME rows. A bounded,
   invocation-local checked box plan may reuse direction/geometry proofs while
   retaining per-action source guards and failure guarantees. Do not remove
   RHC's all-request preflight merely to improve first-result timing.
4. **Condition raw support reuse on the remaining budget.** Mixed/thin total
   loads remain 2.10x/2.22x their unique unions even at 256 slots. These are
   upper bounds on deduplication benefit, not measured cache speedups. First
   choose capacity, then assess whether a bounded raw-interior cache or better
   chunk grouping saves material residual I/O. Full union payloads are about
   11.01/22.02 MB and must be counted alongside workspace and output.
5. **Treat the bridge as a separate input-product limitation.** If WENO becomes
   a routine input, a separately specified adapter for ordinary fields within
   staggered records could avoid the 6.327 s full bridge. It must validate the
   complete record layout; simply removing DAT-003's rejection is not safe.
   This is future format work, not delivered staggered support or a new M1 claim.

The real thin-query read fraction exceeds the previous round's synthetic
0.6--9.4% range. That is concrete evidence to reopen selective support planning;
it does not invalidate the completed optimization's original workload scope.
Support-only storage/output compression and GPU work were not measured here,
so this report does not claim their gains or optimality. No M2 work is started.

## Reproduction

```text
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/m1_weno_assessment.py --output rewrite/benchmark-results/m1-weno-assessment.json
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_dat_003_integration.py rewrite/tests/test_lfe_001.py rewrite/tests/test_hpr_001.py
```

The existing unchanged extension build is reused. Runner: macOS arm64,
Python 3.11.14, NumPy 2.4.4, Cython 3.2.4, clang -O2, no OpenMP. Full environment,
raw paired wall/CPU samples, stage attribution, useful/read/output counts,
trace/resource scopes, fixture provenance and finite-value summaries are in
the ignored JSON. Existing user documentation edits are left untouched.
