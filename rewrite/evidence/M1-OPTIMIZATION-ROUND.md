# M1 Optimization Round

## Step 0: Frozen Comparison And Protection

Baseline: `397aaffc61be68d1141c8f30c0427dd37a4c8bc8` on this runner.
The starting dirty tree contains only the user-provided rewrite documentation
realignment and its new design/evidence files; executable sources match HEAD.
Build both states with `.venv/bin/python rewrite/build_ext.py` (forced rebuild,
clang, default flags, no OpenMP). Raw environment and repetitions accompany JSON.

Frozen standard cases reuse `lfe_001` and `chs_001` fixtures and acceptance:

- LFE synthetic native alternating-refinement 216 leaves, B=4, three fields,
  small [0.42,0.58], medium [0.22,0.78], full domain; add a thin physical-boundary
  slab [0,0.06] x [0.22,0.78] x [0.22,0.78] in fractional domain coordinates.
  Capacities 57 and 216; real tdm small/medium/full at existing bounded capacity.
- CHS synthetic native B=8: 192 points in batches of four, coherent five owners
  and divergent four owners; existing capacities below/at/above working set,
  cleared and warm histories, plus RPS uncached reference. Real tdm 96 points.
- SLE existing standard native synthetic/tdm trajectories protect the adjacent
  consumer. Cache scaling uses all available synthetic owners, not invented IDs.

One warmup and five timed repetitions for LFE/CHS; original SLE standard profile.
Headline timing excludes stage attribution and allocation tracing. Report first
result, total, preflight/planning, reads, application, curl/reduction, useful/read
bytes, exact stats, cache work and output. Nested times are never summed.
OS page cache is uncontrolled: cleared refers to application cache, not disk.

Hard resource limits: no increase to existing RHE/CHS managed-array formulas or
read/output counts for the first candidate; additional transient Python state
must remain below 1 MiB per invocation. These cases must stay below 256 MiB
controlled simultaneous arrays (metadata, backing, workspace, cache, scratch,
outputs) and 1 GiB process peak RSS. These are bounded-workspace measurements,
not an out-of-core claim. Exact historical stats keep their original scope.

Promotion: first planning/check candidate must improve composed LFE median at
least 20% on two refined selections. Protect all small/dense and cold/warm cases:
investigate >10%; repeated >20% regressions on two stable cases block promotion.
Cache candidate requires >=20% composed improvement in two plausible cache-heavy
cases or >=2x access-plan scaling improvement with <=10% composed regression.
Storage/output candidate requires >=2x complete controlled-memory reduction at
<=10% latency cost; compute requires >=20% composed improvement on two cases.
Direction projection retains >=20% composed or >=2x read-byte gain on two refined
cases. Thresholds are fixed before candidate measurement.

Preserve strict arithmetic, selected/valid regions, stats, mutation, read/failure
prefixes, and existing scientific tests. Full standalone checked kernels and
all-26 reference execution remain available; revert a candidate when its gate
fails. No shared dimension representation change, B-to-J exploration, new
scientific operator, M2 activation, or production-package edit is authorized.

## Decision Probe

One cProfile pass of the existing array-backed LFE medium/capacity-57 case:
0.417 s instrumented total, 0.384 s preparation, 0.382 s action preflight,
0.270 s checked CSP (624 calls), 0.027 s application. CSP repeatedly validates
the same relation row and checks allocator-owned array aliases. This supports
testing executor-owned preflight reuse first. Native read attribution is added
to the same standard run before implementation. These profile times are not
headline timings or independent additive stages.

## First Group: HPR-001 Complete

Group: **Owned Halo Preflight**, singleton HPR-001, composition-only.
Dependencies: completed RHE/RHC/CSP/CWP/RPH/REL/CHS/LFE.
Responsibility: establish CSP action safety inside the existing owned executor.
Owned choices: where invariant validation is amortized; no transfer/geometry,
cache, numerical, selection, or failure-policy choice.

The executor allocates all plan arrays privately, validates the external forest,
reach, modes and shape, generates canonical relation/slot/phase rows, and checks
CWP boxes. Validate each CSP relation row once per primary, retain the exact
existing `_validate_plan_geometry` for every COARSER action, then use the same
unchecked CSP generator and existing output/PWA/PRL checks. No persistent plan,
new shared direction representation, or allocation proportional to action count.
The full checked path stays as reference. Lifetime is one preflight invocation;
no proof survives external callbacks or mutation.

Alternative: requested-direction support projection changes closure/validity and
does not remove repeated CSP checks; first quantify reads before adopting it.
The five decomposition questions pass: singular execution proof, independently
checked reference, interchangeable preflight, measured immediate consumers, and
unchanged semantics. DIM-001 is unaffected because no artifact changes.

Independent review approved the proof and final implementation; test fast/reference intermediate
plans, output bits, stats, failure-before-I/O and existing error injection tests.
Focused command: `PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_hpr_001.py rewrite/tests/test_rhe_001.py rewrite/tests/test_rhc_001.py rewrite/tests/test_csp_001.py rewrite/tests/test_lfe_001.py rewrite/tests/test_chs_001.py`.
Group gate: forced rewrite build, focused plus full rewrite tests, standard
LFE/CHS/SLE comparisons, evidence, staged-diff inspection and executable commit.

### HPR Results And Checkpoint

Forced rebuild passed; 115 focused checks and all 1,222 rewrite checks passed.
The final strengthened 12 HPR checks also pass. No arithmetic changed. Intermediate
CSP arrays/counts, complete output bits and I/O argument traces match the complete
checked reference, including physical support and asymmetric reach. The 264 LFE,
29 CHS and 90 SLE stats/digest/resource records compared across standard runs agree.

Native synthetic medians (ms), original M1 -> HPR:

| Selection | Capacity 57 | Capacity 216 | Reduction range |
| --- | --- | --- | --- |
| thin physical slab | 19.536 -> 14.863 | 19.342 -> 14.938 | 22.8--23.9% |
| small | 17.860 -> 12.870 | 17.552 -> 12.425 | 27.9--29.2% |
| mixed medium | 169.130 -> 107.760 | 166.396 -> 104.900 | 36.3--37.0% |
| full | 557.979 -> 395.600 | 554.560 -> 390.850 | 29.1--29.5% |

Original native medium attribution: 143.97 ms action checks, 1.25 ms remaining
preparation, 4.08 ms reader boundary, 18.82 ms halo application, 0.075 ms curl,
0.034 ms reduction. Halo preparation and preflight are nested. Native reads are
2.4% of instrumented wall; eliminating all I/O alone cannot meet promotion.
All-26 support is retained; projection is deferred until read cost reaches 20%
or a consumer requires a smaller valid target, with dimension-aware closure and
explicit slope/physical-base support as prerequisites. No projection gain claimed.

CHS native coherent cold cases improve 29.2--39.4%; divergent cold cases improve
22.6--31.3%. Fully warm sessions remain within 2.2% of baseline. Real tdm has no
COARSER work: LFE small/medium/full 1.620/22.218/22.715 ->
1.664/22.087/22.519 ms; no material regression. Tdm CHS changes <=2.2%, and native
SLE controls have no >10% regressions. Four array SLE cases initially drifted
10--17%; interleaved seven-repetition reruns yield retained/baseline ratios
0.966, 1.020, 1.000, 0.978 with exact outputs/stats. The drift did not reproduce.

Existing canonical tdm materialization takes 9.148 ms for the full three-field
output; it remains faster than full-domain bounded LFE (22.519 ms). It loads
and materializes the whole dataset, whereas selected LFE small takes 1.664 ms.
Safe-cell comparisons retain 5e-13 tolerances because canonical arithmetic and
interface rules differ; this is not full-interface bitwise parity or a tracer
comparison. Synthetic native and independent numerical tests supply that scope.

Workspace/read/output stats are unchanged. LFE managed arrays range from 0.40
to 1.55 MB for synthetic cases; tdm about 1.22 MB, with full output 648,008 bytes.
CHS 71-entry case retains 3,208,210 managed bytes including 1,704,000 payload
cache bytes, 1,136 cache metadata bytes and 1,502,984 workspace/fixed bytes.
The full standard process peaks are 53.8 MB (CHS) and 55.4 MB (SLE), below 1 GiB.
These retain existing raw-array accounting scope; complete footprint is resolved
in step 3. No persistent arrays or plans were added by HPR.

Raw same-runner records: `m1-opt-{lfe,chs,sle}-{before,hpr}.json`,
`m1-opt-sle-controls.json`, and forced build logs under ignored benchmark-results.
Use `m1_optimization.py --family lfe --profile standard --repeats 5 --dat data/tdm.dat
--output ...`; CHS/SLE use `--profile standard --output ...`; add `--baseline`
to load the original preflight from the fixed Git revision. The original LFE
run predates edits; later baseline runs load the exact original source at runtime.
The helper changes only the preflight under study. Run the interleaved check with
`--family sle-controls --output ...`.

Executable checkpoint: HPR-001 complete; next assess cache scaling using the
same retained halo path. No active HPR experiment remains. User documentation
realignment remains in the working tree; the executable commit stages only
optimization-owned source, contract, benchmark, tests and this evidence checkpoint.
