# Selected Refined Halo Completion Evidence

## Result

CSP-001, PWA-001, CWA-001, and RHE-001 form the first complete selected
Cartesian 3D refined-halo consumer.  For positive even block extents `B>=4`
and per-side reach at most `B/2`, a strictly increasing sparse primary stream
is closed over canonical all-26 REL sources, read once per bounded chunk, filled
through SAME/FINER/COARSER and pure/mixed physical rules, and emitted once in
primary order.  Support-only interiors are read, but support slots are never
sent to the writer and their halos are never mutated.

The 210-test focused group/producer/consumer gate and all 715 accumulated
rewrite tests pass.  Evidence includes independent per-primary composition,
capacity-57 versus resident-71 bit equality, exact sparse reader union with an
intervening gap omitted, reordered/repeated and empty fields, asymmetric and
empty selections, custom non-array adapter state, output/managed-byte
accounting, external failure prefix semantics, and a safe dyadic current
`AMRMesh` refined-halo comparison.  Every exact comparison is bitwise.

## Consumer-Found Contract Correction

The first RHE all-action run exposed a real hole in CWP-001's initial checked
domain.  RPH-001 correctly permits any primary phase for a COARSER direction,
and current `gc_prolong` processes every such row, but CWP had inherited the
opposite-direction fine-to-coarse send gate.  In the minimal counterexample,
phase `001` with `(+x,+y)`, `(+x,+z)`, and `(+x,+y,+z)` left 40 target cells per
field without an owner when those rows were skipped.

An independent material review approved removing only that phase/sign check
from CWP and CSP.  Broad randomized balanced-forest scans, a complete WENO
COARSER metadata scan, and a non-singleton dyadic phase-`001`, `(+x,+y)` current
comparison found no FINER slope owner, missing selected source, record overflow,
or numerical difference.  The CWP formula, CSP 18-row/4,500-byte representation,
prior accepted outputs, and SPR 57-slot progress bound are unchanged.  Committed
focused tests cover all 208 direction/phase pairs and the decisive current case.

## Support, Application, And Execution

- CSP partitions the complete CWP rectangular valid box on the primary-relative
  half-block lattice.  Raw face/edge/corner maxima are 18/12/8; merging CWP's
  direct box yields observed plan counts no larger than 15/11/8.  Nonphysical
  records use direct coarse copy or exact RST; physical records reference an
  already completed base and add no leaf ID.
- PWA keeps the true logical interior and storage offset separate from each
  completed base.  It maps all incident physical axes, loads once, applies
  x/y/z transforms, and writes disjoint targets.  Signed zero, infinities, and
  NaN payload bits match the independent PBC reference.
- CWA fills every transfer base before one PWA suffix and overwrites the complete
  CWP required rectangle, so the scratch may be reused without clearing and is
  a valid checked-PRL input.
- RHE validates caller-controlled state once, guards generated slots before
  unchecked RPH, completes each chunk's actual CSP/PWA/PRL proof before its
  reader, applies all chunk bases before any primary PWA, and calls reader and
  writer exactly once per nonempty chunk.  Empty adapter-conformance calls are
  excluded from returned call counts.

The initial simple RHE repeated complete standalone CWA and zero-field
copy/RST validation for every internally generated action.  The first standard
profile attributed 92.2% of medium wall time to that preflight.  Two independent
reviews approved a narrower equivalent production proof while retaining the
complete checked preflight as a private reference: checked SLB/FRP, checked
CWP/CSP, CSP-physical PWA, zero-field PRL, primary PWA, and explicit
kind/count/slot guards.  Focused fast-versus-checked tests keep every output bit,
stat, and callback unchanged.  The retained initial/final standard raw records
show 1.82x, 1.90x, and 2.29x small/medium/full wall improvements.

## Standard Selected-Halo Composition

The standard run uses the real 22,614-leaf WENO forest as metadata and a
canonical synthetic `float64[leaf,3,4,4,4]` payload because the real fixture is
staggered.  Capacity is 128, halo reach is two, one warmup precedes five raw
repetitions, and the sink performs a real compact full-padded-payload copy.

| Selection | Chunks | Loads | Amplification | Requested / read / output bytes | TTFW median | Final wall median (pstdev) | Initial wall | Final / initial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ROI 32 | 1 | 112 | 3.500 | 49,152 / 172,032 / 393,216 | 60.659 ms | 60.702 ms (2.641 ms) | 110.716 ms | 0.548 |
| ROI 512 | 26 | 3,129 | 6.111 | 786,432 / 4,806,144 / 6,291,456 | 41.735 ms | 1.020 s (0.269 s) | 1.934 s | 0.527 |
| full 22,614 | 923 | 114,487 | 5.063 | 34,735,104 / 175,852,032 / 277,880,832 | 73.129 ms | 33.474 s (0.662 s) | 76.656 s | 0.437 |

Every selection reports exactly 1,824,978 managed raw-array bytes.  The full
compact sink is 278,061,744 bytes including IDs; the final process high-water
RSS is 327,761,920 bytes.  This is an absolute process measure, not a claim of
zero RSS growth.  Executor workspace, external output, synthetic backing,
adapter state/page cache, and process RSS remain separate quantities.

The medium set contains one 1.655-second outlier; the other four final samples
are 0.947--1.034 seconds.  Its 26.4% population deviation is recorded as runner
variability, while the full profile deviation is 1.98%.  Median improvement,
exact output/stats, and the paired iteration results agree.  No performance
regression gate failed.

An instrumented, non-headline medium pass records 0.655 seconds of action
preflight and 0.134 seconds of actual application in 0.803 seconds wall,
versus 1.806/0.140 in the initial 1.958-second pass.  Preflight is 63.7% faster
but remains 81.6% of the instrumented wall, so it is the next optimization
candidate only if refined sampling or the native/local-field consumer still
shows it material.  Reader and writer callbacks are 0.00329 and 0.00153 seconds;
the profile is nested and its stages must not be summed.

## PWA Hot-Kernel Gate

Both four-field standard cases are bit-exact with checked/unchecked/reference
and the corresponding compiled comparator.  Candidate and written field-cell
counts are equal and semantic traffic is 16 bytes per field-cell.

| Case | Written field-cells | PWA unchecked | Comparator | Ratio | Throughput | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| pure all-26 | 6,912 | 38.500 us | HAL 63.417 us | 0.607 | 179.53 M cells/s | pass |
| mixed edge/corner | 224 | 2.500 us | HAX 20.583 us | 0.121 | 89.60 M cells/s | pass |

The checked PWA call retains zero traced bytes and peaks at 2,184 traced bytes
in both cases.  No size-dependent allocation or >10% regression was observed.
These ratios apply to the recorded equivalent targets, not every halo workload.

## Reproduction And Raw Records

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_csp_001.py rewrite/tests/test_pwa_001.py rewrite/tests/test_cwa_001.py rewrite/tests/test_rhe_001.py rewrite/tests/test_cwp_001.py rewrite/tests/test_pbc_001.py rewrite/tests/test_spr_001.py rewrite/tests/test_rsl_001.py rewrite/tests/test_rph_001.py rewrite/tests/test_slb_001.py rewrite/tests/test_frp_001.py rewrite/tests/test_rst_001.py rewrite/tests/test_prl_001.py rewrite/tests/test_sto_003.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/rhe_001.py --dat data/weno509_sub_0000.dat --capacity 128 --small 32 --medium 512 --fields 3 --block 4 --halo 2 --warmups 1 --repeats 5 --pwa-repeats 9 --output rewrite/benchmark-results/rhe-001-standard-final.json
```

The runner was macOS 26.5.2 arm64, Python 3.11.14, NumPy 2.4.4, Cython 3.2.4,
clang `-O2`, without OpenMP.  Complete repetitions, environment, stage counts,
sink checks, and dispersion are ignored raw files
`rewrite/benchmark-results/rhe-001-standard.json` (initial checked preflight)
and `rewrite/benchmark-results/rhe-001-standard-final.json` (retained result).

## Deferred Alternatives And Reopen Triggers

- `B=2`, reach above `B/2`, a wider stencil, or weaker all-touch balance may
  require secondary support closure or a separately contracted cross-valid PRL
  path; do not hide it in CSP/RHE.
- Direction-subset source projection remains deferred until the selected
  local-field consumer shows all-26 read amplification is material.  Additional
  sources must be planned before I/O.
- Merging repeated CWP tiles, retaining plans, or fusing CWA/RHE is deferred.
  Reopen only if the next real consumer attributes material time after including
  construction and memory; medium retained plans would already be about 9.8 MB.
- The complete checked preflight remains the substitution reference.  Restore
  the relevant validation if plans/workspaces become external, or direction,
  balance, reach, or lifecycle assumptions are relaxed.
- Payload/halo caching remains deferred until repeated selected queries show a
  measured byte/time benefit under an explicit lifetime and budget.
