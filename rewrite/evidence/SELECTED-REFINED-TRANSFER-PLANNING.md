# Selected Refined Transfer Planning Evidence

## Result

SPR-001, FRP-001, and CWP-001 complete the selected refined transfer-planning
group without changing STO-004 or any stable numerical contract.

- SPR consumes an explicit strictly increasing sparse primary selection and emits the same
  primary-prefix/support representation as STO-004.  Sparse gaps are never
  invented, dense-equivalent inputs produce identical plans, and RSL/STO
  consumers need no representation change.
- FRP partitions each FINER TGT box by active child phase and emits a contained
  fine-source box whose extent is exactly twice the target extent.  The emitted
  boxes compose directly with RST-001.
- CWP emits a normalized nonnegative PRL workspace required region/origin plus
  the COARSER relation source's exact source/destination intersection.  Missing
  one-cell slope reach remains explicit for later support/PBC application.

During CWP's first composition check, normalizing only by logical required
lower produced a negative PRL origin for an upper-side case.  Before CWP became
integrated, the representation was corrected to normalize by
`min(required_lower, logical_origin)` and expose both required bounds.  All
required regions and origins are now nonnegative and the existing PRL contract
is unchanged.

Independent review challenged SPR's initial arbitrary-order input because its
allocation-free duplicate validation was quadratic.  A focused zero-direction
probe took 0.000404/0.004154/0.067249 seconds at 1,000/5,000/20,000 primaries;
at real WENO size the duplicate scan alone exceeded the composed unchecked path.
Before group closure, SPR was narrowed to strictly increasing sparse leaf IDs
and now uses one allocation-free linear Cython range/order scan.  The amended
probe takes 0.000024/0.000015/0.000048/0.000229 seconds at
1,000/5,000/20,000/100,000 primaries.  Public request
order, if later required, belongs to a selection/result adapter with an inverse
permutation.  RSL/STO output representation remains unchanged.

Focused group/producer/consumer checks pass (`166 passed`), including sparse
order, promotion, first-fit/maximality, dense STO equivalence, selected gather,
all compatible FINER/COARSER directions and phases, exact target partition,
source/workspace containment, RST/PRL compositions, read-only inputs, empty
rows, overlap, and atomic errors.  The accumulated rewrite suite passes
(`632 passed`).  Existing RST, PRL, RPH, TGT, and SLB tests retain the safe
current-path comparisons.

## Selected Composition

The standard group run used the real 22,614-leaf WENO forest as validated
metadata.  Its staggered payload remains unsupported, so the transfer used a
canonical synthetic `float64[leaf,3,4,4,4]` backing and a fixed 128-slot
workspace.  Small and medium selections are leaves nearest the normalized
domain center, sorted into deterministic leaf order.  Five repetitions select
the median composed traversal.  Relation generation is recorded separately.

| Selection | Primaries | Chunks/calls | Support amplification | Requested bytes | Read bytes | Dense-envelope bytes | Read/envelope | Selected traversal wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coherent ROI | 32 | 1 | 3.500 | 49,152 | 172,032 | 26,068,992 | 0.66% | 0.055 ms |
| coherent ROI | 512 | 26 | 6.111 | 786,432 | 4,806,144 | 26,337,792 | 18.25% | 1.156 ms |
| full domain | 22,614 | 923 | 5.063 | 34,735,104 | 175,852,032 | 34,735,104 | 506.27% | 49.239 ms |

The reusable selected ID plus payload workspace is 197,632 bytes.  Relation
artifacts are 29,120 bytes, 465,920 bytes, and 20,578,740 bytes for the three
selection sizes and are not counted as payload workspace.  Maximum one-primary
progress requirements are 27, 47, and 53 slots, all below capacity.

The full-domain selected strategy is 1.073x the preserved dense STO-004
composition on the same run (49.239 versus 45.878 ms) and has identical chunk
and selected-load counts.  This is the expected policy boundary: selected
planning avoids sparse gaps; an actual dense full-domain request continues to
use unchanged STO-004.  No universal auto-dispatch was added.

Traversal wall time includes plan, gather, loop overhead, and the same exact
primary checksum consumer on both selected and dense paths.  For the full-domain
selected run, median planning and gather components were 18.972 and 22.039 ms
(41.011 ms combined); the remaining 8.228 ms includes checksum and loop work.

FRP and CWP are cold/control metadata geometry.  Their exact output
budgets are `96*active_finer_sources` and `168*coarser_records`; the standard
run records 0/887/37,168 FRP rows and 144/2,181/86,186 CWP rows for the three
selection sizes.  Both implementations are `O(R)`, allocate no internal
size-dependent storage, and compose exactly with RST/PRL without inventing an
isolated repetition timing.  Every exact plan/traversal comparison reported
zero discrepancy.

The closing run used one warmup and five recorded repetitions.  Selected wall
time population deviations were 0.004, 0.022, and 0.931 ms for the three
selection sizes; the full-domain dense deviation was 0.278 ms.  The runner was
macOS 26.5.2 arm64 with Python 3.11.14, NumPy 2.4.4, Cython 3.2.4, and clang
`-O2`.  The complete raw repetitions, dispersion, compiler flags, Git commit,
and dirty state are retained outside Git at
`rewrite/benchmark-results/srt-001-standard-20260904.json`.

Command:

```text
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/srt_001.py --dat data/weno509_sub_0000.dat --capacity 128 --small 32 --medium 512 --fields 3 --block 4 --warmups 1 --repeats 5 --output rewrite/benchmark-results/srt-001-standard-20260904.json
```

Group checks:

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_spr_001.py rewrite/tests/test_frp_001.py rewrite/tests/test_cwp_001.py rewrite/tests/test_sto_004.py rewrite/tests/test_rst_001.py rewrite/tests/test_prl_001.py rewrite/tests/test_rsl_001.py rewrite/tests/test_tgt_001.py rewrite/tests/test_rph_001.py rewrite/tests/test_slb_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
```

## Deferred Alternatives

- Dense-run decomposition remains available through STO-004 for truly dense
  work.  Reopen run coalescing only if a real ROI selection contains a small
  measured number of long dense runs and reduces planning without reading gaps.
- Hash or dense-marker membership is deferred.  Reopen when a selected
  plan-to-gather profile shows SPR membership work is material at representative
  capacity after I/O and transfer are included.
- Arbitrary primary order is deferred.  Reopen only when a real public consumer
  cannot restore request order at the result boundary with an explicit inverse
  permutation.
- Fused relation/box/value application is deferred until refined halo execution
  measures retained plan bytes or repeated passes as a material bottleneck.
- CWP does not silently expand SPR support for PRL slope reach.  The next refined
  support/application capability must consume its uncovered required region;
  reopen the selected support representation only if that consumer cannot add
  the needed leaves without global traversal.

Consumer follow-up: the complete RHE composition found that CWP's original
phase/sign validation described the current fine-to-coarse send gate, not the
accepted COARSER prolongation domain already frozen by RPH-001.  Independent
review approved accepting either phase on nonzero CWP axes.  Every previously
accepted output is unchanged; the active halo-group evidence owns the expanded
phase matrix, complete-target current comparison, and final regression counts.
