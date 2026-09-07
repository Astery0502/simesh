# E5 Retained Geometry Plans: 2026-09-08

## Disposition

**Adopt explicit optional selected fill plans for repeated fixed-geometry requests.**
Keep ordinary unretained preparation for one-shot and changing selections. Plans
show a material repeated whole-consumer benefit on sparse and dense WENO requests,
with unchanged reads and exact numerical output. **Defer** automatic retention,
full-domain persistent plans, numerical ghost cache integration and a fused native
plan executor; those are not prerequisites for this useful optional boundary.
The main session owns backend/scheduling and numerical cache policy.

Implementation: `src/simesh/analysis/geometry_plans.py`; the existing provider adds
one `FieldSource.plan_builder` hook. `build_fill_plan(source, ids, capacity=...)`
constructs checked metadata; `plan.prepare(source, field_ids)` makes a detached
prepared product. Tests: `tests/analysis/test_geometry_plans.py`.
Raw evidence: ignored `benchmark-results/analysis-core/plans.json`, `plans.log`.

## Retained Facts And Lifetime

Each chunk stores ordered original source-leaf IDs, its primary prefix and
same-level copies, fine-to-coarse restriction boxes, coarse workspace/slope-support
records, prolongation placement and physical-widening records. Source references
inside actions are **chunk-local support ordinals**, not runtime cache slots or
raw pointers. Geometry construction resolves adjacency, indexing and transfer
ranges once; execution creates fresh buffers and binds the selected fields.
Plans sort the selected leaf IDs; the resulting product publishes that explicit
leaf order and directory.

The numerical strategy remains `rewrite-ratio2-minmod-exactphase-cont-v1`:
ordered eight-cell restriction averages carry fixed 1/8 weights; prolongation
uses the original +/-1/4 geometric phase. Those applicable constant weights are
encoded by the retained operation kind/phase and unchanged kernel, not duplicated
as a sparse coefficient matrix. Minmod slopes depend on current values and are
always recomputed. Continuous physical widening remains the selected boundary.
No-inflow clipping, alternative boundary modes and other reconstruction strategies
are not admitted by this first plan, nor silently treated as fixed linear maps.

| Object | Lives across | Invalidated / rebound by |
| --- | --- | --- |
| Mesh and source-cell identity | Prepared products and geometry plans | Geometry, topology, source-cell ordering or mesh object change |
| Fill plan | Repeated fields and new immutable value lifecycles on the same mesh | Halo width, transfer/boundary strategy, selected leaf set or support partition change; different mesh/strategy rejected |
| Read values / coarse scratch | One execution only | Every execution allocates and reads fresh values; limiter results are never retained |
| Private support ordinal -> buffer binding | One chunk execution | Each fresh execution binds its own numeric storage |
| Output product | Independently owned until released | Never invalidated by another execution or plan release |
| Runtime numerical cache slots | Owned by the main session's runtime | Its eviction/generation rules; no such slots appear in a plan |

Plan construction owns no source data or open fd. A plan can outlive source close,
but another execution requires a live compatible source. Reopening a file normally
creates a new MeshIndex and is rejected unless a future explicit geometry-sharing
boundary establishes that identity. No structural-equality guess or cross-snapshot
cache reuse is implicit. Source buffers remain immutable within each lifecycle.
New field count/order and values do not invalidate the geometric facts. A read or
transfer failure publishes no PreparedFields; private partial arrays are discarded.
There is no mutable execution workspace in a plan to poison later calls.

Retained metadata accounting includes NumPy backing **and Python action objects**,
with aliases counted once. The default build allowance is 128 MiB; construction
checks accumulated retained chunks plus its provider planning workspace before
publication. A candidate chunk is constructed before that check, so peak build
memory additionally includes that bounded prospective chunk and temporary Python
accounting containers. This is a retained-plan allowance, not a strict RSS cap.
Executions admit source/mesh/plan storage, output, selected raw/padded support,
coarse scratch and reader scratch under the normal 2 GiB case budget. Runtime
cache residency is neither credited nor assumed free.

## Core And Real-Data Evidence

The mixed-level check compares every prepared value to ordinary preparation with
permuted 3-, 1- and 2-field selections. It then changes the backing to oscillatory
values in a fresh declared value lifecycle: limiter decisions change, but the
same geometry plan still matches ordinary preparation **exactly**. Direct sampling
and curl agree; different mesh, insufficient output budget and reader failure
are rejected. A successful execution after failure confirms no retained numerical
state was poisoned. Existing source/core tests remain applicable and the complete
analysis suite passed **22 tests** after integration.

Real profile is shared with [E1](rebricking.md): 37 sparse primary leaves supporting
eight spread seeds/128 steps; dense union 1,695 primaries for 768-leaf curl and an
8x8 coherent full-depth LOS. Fields are actual WENO B1/B2/B3, two valid halo layers,
continuous boundaries, capacity 256. The plan and ordinary provider use identical
chunk support and read the same original-file values: **435 / 6,074 leaf loads**,
**5,345,280 / 74,637,312 bytes**. Every prepared value, resulting trajectory,
derivative and LOS sample agrees exactly in this comparison. No temperature or
physical response assumption enters the geometry-plan timing.

| Quantity | Sparse | Dense |
| --- | ---: | ---: |
| Plan construction s | 0.024252 | 1.640444 |
| Retained plan bytes | 643,462 | 52,183,060 |
| First fresh execution s | 0.007589 | 0.227791 |
| Repeated planned preparation median s | 0.007353 | 0.209583 |
| Repeated unretained preparation median s | 0.027059 | 1.523037 |
| Same consumer median s | 0.005261 | 0.022251 |
| Planned repeat prepare + consumer s | 0.012614 | 0.231833 |
| Unretained repeat prepare + consumer s | 0.032320 | 1.545288 |
| Full repeat composition ratio | 2.56x | 6.67x |
| Planned execution controlled upper bytes | 31,299,853 | 151,613,291 |

Timings have one warmup/three repetitions with application values freshly prepared
and uncontrolled OS cache. All repeat wall/CPU samples and dispersion are in JSON.
The complete-consumer totals above add disjoint preparation and unchanged consumer
stages; they are composed costs, not a separately instrumented sum of nested stages.
The common source-open cost is 0.04315 s; ray-plan construction is 0.05476 s and
request arrays occupy 1,163,808 B. Add the common applicable setup once to either
first-use workflow. Outputs/source/retained baseline controls were kept live for
comparison; measured peak process RSS was **527,450,112 B**.

Initial plan build + first execution + consumption measured **0.03710 / 1.89049 s**,
versus the initial ordinary calls **0.03089 / 2.42261 s**. The first ordinary dense
call is visibly colder than its repeat median; do not infer a one-shot speedup
from that ordering. Using the stable repeated preparation controls, construction
amortizes after **two executions** in both selected cases. A one-shot caller can
therefore avoid plan construction, with the ordinary path remaining available.

The first execution attributes reads/copies, numerical transfers and packing to
**0.005285 / 0.001553 / 0.000389 s** sparse and
**0.081699 / 0.130766 / 0.014768 s** dense. Repeated planning/dispatch avoidance,
rather than reduced reads or relaxed arithmetic, accounts for the benefit.
No full-domain plan or automatic million-seed plan cache was measured. Dense
metadata is substantial (~52 MB for this selection); extrapolation alone does
not authorize retaining every leaf's plan.

## Reproduction And Next Interface

```bash
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.benchmark_organization --stage plans
PYTHONPATH=src:rewrite/src:scripts:tests/analysis:rewrite/benchmarks .venv/bin/python -m unittest discover -s tests/analysis -p test_geometry_plans.py -v
```

The main session can select a geometry-plan key, hold a plan reference, and bind
field/slot storage at a future runtime boundary. It must retain dynamic alias,
publication, numerical validity and slot-generation guards. The present API
intentionally performs fresh reads and private numerical execution, making the
benefit and lifecycle independent of any ghost value cache. A later compact/fused
executor should compare total build/storage/repeat costs against this usable
boundary, while retaining exact-phase/minmod arithmetic or explicitly selecting
a different numerical strategy.
