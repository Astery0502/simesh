# M1 Optimization Round

## Round Result

Closed 2026-09-07: HPR-001 and CQP-001 implemented and validated. Native refined
local analysis is 22.8--37.0% faster; indexed 36/71-owner warm batches improve
18.5/36.1% beyond HPR, with 4.4/8.5x faster access planning. Small warmed batches
and trajectories retain their prior strategy. No read-byte reduction is claimed.
Full rewrite regression: 1,229 passing checks. Resource bounds fit step 0.

Costs: invocation-local Python index (<=6,240 conservative bytes for the frozen
set), unchanged `16*C` metadata copying, and retained per-action geometry proof.
No extra persistent arrays, precision, shared halo representation or numerical
strategy. Original M1 scope, scan and checked halo references remain available.
All directions have dispositions below; no experiment or selected group remains
open. This closes the optimization round, not broader post-M1 work or M2.

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

## Step 2: Cache Query Planning Freeze

HPR executable checkpoint: `da861f1`. Cache probe uses the 71-leaf native CHS
fixture, spread queries over 16/36/71 actual leaf centers, capacities matching
the working set, batches of four and a whole working set. No synthetic cache
keys or extreme repetition counts. Raw record: `m1-opt-cache-hpr.json`.

Warm whole-set composed/access-plan medians: 16 owners 0.134/0.0137 ms;
36 owners 0.223/0.0515 ms; 71 owners 0.435/0.1738 ms. At 71 owners the access
plan is 40% of composed time, versus 0.137 ms actual grouped interpolation.
Four-owner batches spend more on repeated checked call boundaries; no new
unchecked public sampler is selected.

Group: **Cache Query Planning**, singleton CQP-001, composition-only; dependency
CHS-001 complete, retaining HPR-001. The single responsibility is simulating the
existing serial cache access/eviction plan before I/O. Build an invocation-local
owner-to-slot dictionary only for capacity>=16 and owner_count>=8. Smaller calls
retain the original scan. This amortizes O(C) index construction over at least
eight lookups; hits become O(C+O) rather than O(C*O). Miss admission and lowest-
recency/lowest-slot eviction remain unchanged and update the temporary index.
No dictionary persists or participates in cache identity; array copies still
isolate simulation from state. No numerical or shared dimension artifact changes.

Alternative: persistent dictionary/LRU links would remove copies and repeated
construction but extend mutation/rollback state. The measured 36/71-owner cases
first test the smaller transient alternative; no new raw/plan/derived cache.
Bounded experiment: one indexed variant plus retained scan; keep it only under
step 0's >=2x access-plan gain on both 36/71 whole-set cases, <=10% composed
regression, unchanged exact array budget/stats and <=1 MiB incremental traced
transient state for the frozen set. Large cold/miss-heavy query controls must
remain exact; full standard CHS/SLE protects existing cold/warm small batches.

Five-question gate passes: singular cache execution policy, independent scan
reference, privately substitutable plan, immediate CHS cost, and no geometry or
storage policy change. Ordinary review is sufficient: no stable public contract,
representation, ownership or failure guarantee changes. Focused command:
`PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_cqp_001.py rewrite/tests/test_chs_001.py rewrite/tests/test_sle_001.py`.
Group gate: forced build, focused/full rewrite suite, standard cache scaling plus
CHS/SLE, exact baseline comparison, resource evidence and executable commit.

### CQP Results And Gate

The forced build, 80 focused checks and all 1,229 rewrite tests pass. Five
capacities including zero, cold/warm and mixed eviction histories, repeated
owners, tied victims, I/O traces, five plan arrays and failure before I/O match
the scan. Final standard CHS/SLE stats, digests and raw-array budgets match HPR
and original M1. LFE does not consume cache planning, so its final composition
is the already measured HPR state; no duplicate LFE run is needed.

Nine paired warm repetitions alternate scan and index within one native session:

| Capacity / batch | Scan wall | Indexed wall | Final / scan |
| --- | ---: | ---: | ---: |
| 16 / 4 | 0.347 ms | 0.349 ms | 1.004 |
| 16 / 16 | 0.133 ms | 0.125 ms | 0.944 |
| 36 / 4 | 0.832 ms | 0.839 ms | 1.008 |
| 36 / 36 | 0.219 ms | 0.179 ms | 0.815 |
| 71 / 4 | 1.700 ms | 1.701 ms | 1.001 |
| 71 / 71 | 0.425 ms | 0.272 ms | 0.639 |

Access-plan medians at 36/71 owners fall from 51.46/173.83 us to 11.75/20.42 us;
the >=2x gate passes for both. Warm traced allocation peaks do not grow: the
index expires before the larger later stage. Its separately conservative
Python-object bounds are 1,528/3,184/6,240 bytes for 16/36/71 slots.

Separate standard runs and the first broad interleaved controls encountered
substantial unrelated system load (one renderer measured at 234% CPU). Preserve
those records, including a 54% zero-cache wall outlier; they are not stable
speedup claims. Targeted drift-only reruns give final/HPR wall ratios
0.998/0.955 for coherent capacity-five cold/warm, CPU ratios 0.992/0.953.
Divergent zero-cache cold remains 1.221 wall but 1.005 process CPU, consistent
with scheduling delay on a path where indexing is disabled. No >20% regression
reproduced in two stable cases. The primary indexed cases have stable paired
gains; shared-runner jitter remains an explicit timing limitation, not a relaxed
correctness/resource gate. Final paired SLE control ratios to original M1 are
1.025/1.001/1.004/0.918, with exact outputs/stats. No tracer speedup is claimed.

Raw records: `m1-opt-cache-{hpr,indexed,paired}.json`,
`m1-opt-{chs,sle}-final.json`, `m1-opt-query-controls.json`,
`m1-opt-drift-controls.json`, `m1-opt-sle-controls-final.json`.
Use `m1_cache_scaling.py --paired --output ...`; `--baseline` selects the original
scan. `m1_optimization.py --baseline` now restores both original preflight and
cache simulation from the fixed Git revision. `m1_query_controls.py --output ...`
interleaves original M1, HPR and final; `--drift-only` adds CPU-time attribution
for the three noisy cases. Commands use `PYTHONPATH=rewrite/src:src .venv/bin/python`.

### Value-Kernel Attribution

One additional medium/capacity-57 native instrumented pass separates value
kernels from the planning and Python scheduling inside halo application.
Original/retained preflight is 149.42/82.70 ms, reader boundary 4.63/4.99 ms,
and disjoint copy/restriction/prolongation/PWA kernel boundaries total
7.50/6.34 ms. The enclosing application is 23.13/21.63 ms, so it must not be
presented as pure arithmetic time. These are attribution samples with dispatch
overhead, not fresh headline speedup claims; no value-kernel code changed.
Selected output and accumulator bits agree. Reproduce with
`m1_halo_attribution.py --output rewrite/benchmark-results/m1-opt-halo-attribution.json`.
This confirms the choice of preflight reuse over arithmetic optimization.

## Steps 3 And 4: Dispositions

Resource command: `m1_resources.py --lfe rewrite/benchmark-results/m1-opt-lfe-hpr.json
--cache rewrite/benchmark-results/m1-opt-cache-paired.json --output
rewrite/benchmark-results/m1-opt-resources.json`. It inventories metadata and
caller outputs and composes managed stats with measured/bounded transients.
LFE traced call peak already includes workspace, native scratch and temporary
Python allocations, so these are not summed twice. Native local peak is at most
2,086,842 bytes, plus optional resident comparison backing (331,776 synthetic
or 648,000 selected tdm bytes) and benchmark reference copies. Cache inventory
includes fixture backing and live result copies: <=4,155,414 bytes including
the native conversion-scratch bound and dictionary. Standard peak RSS stays
below 61 MB. Allocator retention and OS page cache remain separate. No large,
out-of-core or hard-RSS-budget claim is made.

| Direction | Disposition and concrete reopen condition |
| --- | --- |
| Support-interior storage | Defer: even deleting all padding, including required primary padding, yields <=1.905x complete local-memory reduction, below 2x. Reopen when a real request cannot fit its complete budget or larger halos make padding dominate. |
| Alternate completed-halo storage | Retain valid padded entries: even deleting all workspace/cache padding yields <=1.565x complete measured-memory reduction. Reopen at a demonstrated miss knee, counting reconstruction latency and validity. |
| Bounded/compact output | Retain caller arrays: eliminating all LFE output yields <=1.592x; largest SLE output is 8,528 bytes. Reopen when actual output exceeds the complete budget, with chunk delivery or reduction-only semantics. |
| Requested-direction projection | Defer: retained refined read fractions are 0.6--9.4%. Small unrefined tdm is an exception (35.6%, 27 loads for one primary), retained as the next sparse-read target. Promotion still needs two refined cases and dimension-aware target/slope/base closure. |
| Raw/plan/derived retention | Defer: warm completed-field hits already remove reads/halos; HPR saves checks without retaining actions. Reopen raw retention at >=20% repeated-transfer wall under the same complete budget; plans need measured repeated geometry cost with bounded identity/lifetime. No second derived operator is active here. |
| Metadata copying/persistent index | Retain copies (<=1,136 bytes here). Large-batch hits now avoid quadratic scanning. Reopen when O(C) copies/index construction dominate realistic small queries at larger capacity; miss victim scans remain linear. |
| Batching/scheduling | Retain existing whole-query batching: identical 71-owner warm results take 1.701 ms in batches of four versus 0.272 ms in one batch. This is an existing option, not a new implementation speedup. |
| Fusion/OpenMP/GPU | Defer: local arithmetic is <0.1%, halo application <=18.1%; arithmetic-only changes cannot meet the 20% composed gate. Warm group timing includes Python setup; a 0.272 ms query does not justify a new backend. Reopen for larger real fields/seeds and a profiled remaining kernel or independent-query bottleneck. |

## Focused Horizon And Closure

Storage/execution/compute substitution is preserved. HPR's reviewed proof is
invocation-local and allocator-owned; standalone checks remain complete. CQP
changes private simulation, preserves failure atomicity and exact LRU, and adds
no session representation that DIM-001 must migrate. Float64/int64 layout,
strict arithmetic, valid regions, failure prefixes and all-26 3D transfers keep
the original M1 scope. Residual validation/copy costs are explicit limitations.

Both fixed workflow families have final same-runner evidence. All directions
have dispositions; adopted groups pass their predeclared gain/resource gates;
no selected experiment remains. Direct real refined non-staggered data is still
missing: synthetic refined native and real unrefined tdm qualify this round as
they qualified M1. Derived lifecycles, additional diagnostics/integrals, I/O,
public API and dimensional work remain pending. Do not automatically enter M2
or start another feature stage.
