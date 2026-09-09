# Scientific Acceptance And Performance Evidence

## Meaning Of High Performance

Correctness comes first. Among correct strategies, assess useful-result latency,
complete throughput, memory, I/O, allocation/copies, support amplification and
scaling together. The [intent](intent.md) owns scope and target scale; the
[shared spec](prepared-fields.md) owns data validity and ownership. No strategy
must dominate every workload, and fewer bytes alone do not establish a better
result when a faster affordable path exists.

Compare preparation and immediate consumption together under the actual
[geometric access pattern](prepared-fields.md#consumption-geometry-and-access-patterns).
Match the required input reach and remaining derived validity; historical
one-layer sampling is only a scoped control for the new prepared-field work.
Task-scoped comparisons include internal sharing and every requested product;
exploration includes initial preparation, later queries, changes and release.
Native regional-field delivery counts as a result when requested.

## Choose The Smallest Useful Check

Start by reading the changed code, its contract and relevant existing evidence.
If that settles the issue, direct inspection is sufficient. Otherwise run an
existing focused check or the immediate useful consumer. Add a regression test
only for a concrete core failure or behavior that is not adequately protected.
A performance claim needs an affected-path measurement, not a test-count target.

| Change or question | Default verification |
| --- | --- |
| Source-data descriptions, copied metadata facts, names, declared units or simple configuration wiring | Read the source/header/definition and the diff; no dedicated test by default |
| Internal helper order, mock call sequences or another incidental implementation detail | Inspect directly; do not lock the implementation's call order into tests |
| Straightforward physical-quantity labels, an unchanged formula or a simple explicit conversion | Check the definition and wiring directly; avoid one physics test or redundant physical invariant per quantity |
| Ghost completion, coarse/fine transfer, consumed valid regions, point ownership or field/slot selection | Reuse a representative core/consumer check; add a focused case only for an uncovered failure |
| Slot reuse, live buffers, parallel writes, resource bounds or observable result ordering | Check the behavior affected by the change; use outputs/resource observations rather than internal call traces |
| A changed reconstruction, derivative or integral whose numerical behavior is not established by inspection | Use the smallest independent numerical example that answers that uncertainty; no automatic per-quantity or full-physics matrix |
| A hot path or a claimed memory/I/O/speed improvement | Compare the relevant complete operation and costs; isolate a kernel only when that would guide a decision |

Descriptive metadata is different from parser/indexing code that can select the
wrong physical values. Incidental call order is different from seed-to-result
identity or an explicitly contracted arithmetic order. Exercise these distinctions
only where the change creates a real question; do not build a suite for every
metadata field, ordering variant, quantity, helper or branch.

One meaningful composed case may protect several helpers and quantities. Reuse
it instead of duplicating assertions that mirror the implementation. After the
relevant checks pass, continue development; broaden or repeat only for a new
change, failure, cross-consumer impact or unresolved concern. Existing scientific
and observable behavior remains the target, without a mandatory new test for
every sentence of its specification.

## Scientific Accuracy

Implementation conformance and scientific acceptance are separate questions.
Use the following only for numerical behavior actually changed or claimed; they
are not a checklist of tests to add for every physical quantity:

- Freeze exact ownership, selected cells, valid regions, and capacity/backend
  behavior under the chosen contract.
- Use independent analytic/manufactured fields as well as operation-tree
  references. Distinguish sample values from cell averages.
- Report interior, coarse/fine-interface, and physical-boundary errors
  separately, with the refinement sequence and norms used.
- Do not infer global second-order derivatives from a uniform centered stencil
  or fourth-order trajectories from the RK tableau. Halo reconstruction and
  owner-dependent interpolation affect composed error and continuity.
- Distinguish cell sums, volume-weighted integrals, and flux integrals. Require
  conservation or continuity only where scientifically contracted, with a
  suitable test.
- Record worst discrepancies, cancellation/IEEE behavior where relevant, and
  separately declared trajectory, endpoint, derivative, and integral errors.

A primitive's bitwise oracle alone does not establish a new scientific claim.
An unmet target needs investigation of the actual numerical issue; do not widen
a tolerance to hide it. Directly readable definitions and unchanged formulas
do not require a separate scientific test or independent review.

## Turn "Fast" Into A Reviewable Result

The single [design record](historical-index.md#how-a-decision-advances) states:

```text
Fixture, AMR pattern, storage and machine:
Fields/physical definitions; region, seeds, view/depth or output resolution:
Scientific result, accuracy, boundary meaning and completeness:
Geometric access, locality/repetition and task/retention sequence:
Complete memory limit, minimum working set and output delivery:
Matching comparator, numerical strategy and cold/warm state:
First-use, first-result, warm-query and total-time targets:
Permitted amplification/overhead and material regression rule:
```

Freeze case-specific numbers before optimization; unspecified budgets/latencies
are open decisions, not invented SLAs. Name what "first result" means and include
all prerequisites, including existing whole-request preflight when applicable.

| Measure | What it exposes |
| --- | --- |
| Open, first-use, first-result and complete time | Startup and actual useful delivery, not just kernel execution |
| Warm queries and whole-sequence time | Preparation/retention amortization and costs when geometry or attributes change |
| Useful/read bytes, calls/seeks and repeated loads | Read amplification; coalescing can justify reading extra bytes |
| Selected versus support blocks/cells and valid halos | Required reach versus avoidable block-cover/storage amplification |
| Planning/checks, location/index work, I/O/copies, arithmetic and output | The cost the user actually waits for; nested stage times are not additive |
| Live memory, retained products/cache state, output and RSS | Complete feasibility and release behavior |
| Product error and completion | Interior/interface/boundary, endpoint, geometry, response and integration meaning |

For comparable repeated queries, `T_setup + N*T_warm` versus `N*T_one_shot`
estimates break-even; use the actual sequence for evolving requests. Cache
assessment includes fitting, slightly undersized and strongly oversized working
sets, supported zero-cache execution, changed fields/coverage and actual avoided
reads/fills/derivatives, not hit rate alone.

Select affected coverage axes: sparse paths, thin/oblique cuts, moderate regions,
full-domain D/L; one scalar/vector/derived chain; first file use, cleared/warm
application state and nearby changes; resident-fit, cache-fit/thrashing and actual
larger-than-RAM evidence when claimed; uniform/mixed levels and supported
boundaries/dimensions. These are not a Cartesian-product suite. New geometry or
radiation products need their own fixtures. Reuse valid previous evidence.

## Performance Classes

These labels describe evidence when useful; assigning a label to every helper
is not a development prerequisite. Choose only the evidence needed for the change:

- `cold/control`: correctness, asymptotic complexity, and allocation behavior
  are sufficient; no isolated timing is required;
- `composition-only`: measure the capability in its first real immediate
  consumer because a standalone timing would not guide a decision;
- `hot-kernel`: measure the affected consumer, adding or refreshing isolated
  kernel timing only when it can explain the cost or guide a decision;
- `milestone-workflow`: record an end-to-end real-data or public workflow with
  the relevant runtime, memory, I/O, and scaling metrics.

Promote a capability only when a profiler, cost model, scaling hazard, or real
consumer shows that the more expensive evidence can change a decision. Do not
invent extreme repetition counts or artificial consumers solely to demonstrate
that materializing a small helper is faster than recomputing it.

## Benchmark Levels

<a id="1-kernel"></a>
<a id="2-composition"></a>
<a id="3-real-user-workflow"></a>
<a id="4-scaling-and-resources"></a>

| Level | Evidence |
| --- | --- |
| Kernel | Cells/blocks or effective bytes per second; call/check cost; allocation; shape/fields/reach/relations/dimensions; independent versus optimized path |
| Composition | Read/support/halo/operator/sample/delivery stages, primary/support counts, chunks, amplification, workspace capacity and resident/bounded/cache alternatives |
| User workflow | Selected native reads, regional/global diagnostics, tracing, slices/LOS, uniform sampling, ghost/derived fields, roundtrip/construction/export and public lifecycle as applicable |
| Scaling/resources | Leaves/cells/fields/refinement distribution, capacity/threads, live bytes/RSS/faults, I/O, cache state, speedup and efficiency |

## Comparators

- The independent Python/NumPy reference establishes semantics, not speed.
- The current canonical Cython path is the main migration performance baseline.
- Legacy is a performance comparator only where it still represents an
  independent supported workflow.
- An appropriate simple resident implementation is a composition reference for bounded/cached
  variants.
- Parallel comparisons always include the matching one-thread build/run.

Use the same data, compiler/optimization flags, Python/NumPy/Cython versions,
thread settings, warmup policy, and machine for relative gates.

## Profiles

- `smoke`: small, fast, one/few repetitions; detects broken benchmark paths,
  not small timing regressions.
- `standard`: representative sizes, warmups, and enough repetitions for a
  stable median and dispersion; used for capability and milestone evidence.
- `large`: opt-in memory/I/O/scaling workloads; required when claiming
  out-of-core or large-scale behavior.

Reuse the repository benchmark reporting conventions where useful: raw JSON,
CSV for tabular analysis, a concise Markdown report, machine/build metadata,
and optional figures. Avoid building a benchmark database or framework until
the existing scripts can no longer express a concrete need.

## Recording

Raw runs live outside Git under the ignored `benchmark-results/` tree. A raw
record contains:

- Git commit and dirty/clean state;
- OS, CPU/architecture, Python, NumPy, Cython, compiler, optimization and
  OpenMP information;
- fixture/profile parameters and cold/warm policy;
- every raw repetition plus median and dispersion;
- all runtime, memory, I/O, allocation, amplification, and scaling metrics
  relevant to the claim.

The selected compact result belongs with the selected project design/contract
or its evidence record, with the exact reproduction
command, environment summary, baseline ratio, worst correctness discrepancy,
memory trade-off, and reason the retained variant was chosen. Milestones add an
integrated evidence summary instead of relying only on microbenchmarks.

## WENO Reference Comparisons

Use [WENO-REFERENCE.md](../../../../../previous/rewrite/WENO-REFERENCE.md) for the explicit feature-completion
real-data profile. Compare original canonical simesh, fixed original M1, the
previous retained version and a candidate where each has matching operations.
Label each comparison as equivalent, restricted-common-domain, different-work,
unsupported or resource-deferred before interpreting a ratio. A rewrite
reference is not the original simesh implementation. Do not label selective
queries versus full-domain materialization as like-for-like speedup.

Separate original-file open/index, tree/connectivity/geometry, bridge/setup,
support/action planning, checks, backend reads/copies, transfer kernels, operator,
locator/grouping/cache/stepper and output when those boundaries are affected.
Give both component and composed measurements. Unchecked-kernel time alone is
not feature latency, and nested instrumented stages must not be summed. Report
cold construction, cleared application caches, warm queries, first result and
total time separately; OS page cache is uncontrolled unless explicitly managed.

Freeze query/fields/validity/boundaries/precision/capacity, original revisions
and build configuration before comparative runs. Raw samples and environment
belong in ignored benchmark-results; commit a compact comparison and feasibility
record under evidence. Include all simultaneously live storage, including
canonical C-allocated buffers, metadata, conversions, caches and full outputs;
report RSS/faults separately. Persistent bridge reuse must show its one-time
construction cost. Use existing regression policy and explicit numerical
contracts; do not loosen tolerances or change acceptance after observing results.

## Regression Policy

Before optimizing a hot path, its design/contract states:

1. workload and profile;
2. current/simple comparator;
3. concrete performance hypothesis;
4. metrics that decide the trade-off;
5. what constitutes a material regression.

Correctness, exact memory-budget formulas, and output equivalence are hard
gates. Performance gates run only on a controlled same-runner comparison;
cross-machine absolute milliseconds are descriptive.

Unless a capability declares a better domain-specific rule:

- a median regression above 10% on a controlled standard case is investigated
  and recorded;
- a regression above 20% reproduced in at least two stable representative
  cases blocks `complete` by default;
- the regression may be accepted only for a declared, material memory, I/O,
  determinism, or complexity improvement with retained alternatives and
  evidence.

Noisy smoke CI should not enforce tight timing thresholds. It verifies that the
benchmark and metrics run. Stable scheduled/dedicated runners own performance
gates.

## Complete Workflow Resources

For disjoint simultaneously live allocations, use:

```text
M_peak = max over time(metadata + resident inputs + retained products/caches/plans
                      + execution scratch + transfer/backend scratch + live outputs)
```

For new bounded workflows, state the budget and count controlled resources at
their simultaneous live peak: workspace, retained caches/plans, temporary query
and sorting buffers, backend transfer/conversion scratch, and output buffers.
For a device backend, separate host, pinned-host, and device pools, including
in-flight buffers. Do not sum mutually exclusive stage peaks or double-count
aliases. Account for opaque/transient allocations with measured bounds and
report allocator overhead separately when an exact byte formula is unavailable.

Caller-owned or borrowed does not mean free: report resident metadata, inputs,
and full outputs in the workflow footprint even when outside an executor's
budget. State what admission can control, the minimum working set, and behavior
when the request cannot fit. Reserve known minimum support/fixed buffers before
optional retention; variable curves/surfaces/frontiers need declared bounded
delivery or admission, not a mandatory complete precount scan. Workflows with
unbounded result size need bounded delivery or accumulator-only modes; a supplied
full-output array is another explicitly costed strategy. Output sinks' staging
buffers count while live.

Exact historical WSP/RHE/CHS managed-array formulas retain their original scope.
New accounting composes them; it does not redefine their stats or reinterpret
old benchmarks as total-memory proofs. Report mapped pages, OS page cache,
Python/allocator overhead, faults, and RSS separately. A managed budget is not a
hard promise that process RSS or OS cache stays below the same number.

## CPU And Device Comparisons

For OpenMP or other CPU parallel work, use the matching one-thread baseline and
report thread/seed scaling, scheduling overhead, memory growth, and efficiency.
Separate per-kernel threading from parallel independent analysis. Check
determinism and numerical results under the exact strategy contract; a different
reduction tree is not automatically equivalent to serial RED.

For an active GPU/other device backend, record device/runtime/compiler identity,
precision and math flags, compilation/context/allocation startup, transfers,
layout conversions, synchronization, kernel execution, and output delivery.
Synchronize timed completion so queue submission is not reported as completed
work. Report device-only timing and end-to-end timing separately, with cold
one-shot and warm resident/repeated queries. Include host and device peak memory
and the amortization needed for retained data. Compare the same scientific
workload and error target against the useful CPU baseline.

No GPU benchmark is required when no GPU implementation is being claimed.
Portability readiness can be assessed from explicit semantics and boundaries;
backend support and speedup require an implemented conformance path and actual
measurements. Precision or arithmetic changes need an explicit numerical meaning
and the focused evidence relevant to that change; independent review is optional
under the [development rule](development.md#autonomous-decisions-and-user-judgments).

## Optimization Stop And Completion

Use the [feature feedback loop](development.md#feature-completion-and-feedback)
to turn a consumer comparison into a code fix, design change or justified spec
revision. This applies to costs in reused providers as well as the new core.

Stop the active optimization cycle when the implementation is correct,
integrated, meets its declared scientific/resource acceptance, has a useful
measured trade-off, and has no unresolved material regression. Bound experiments
by the question/variant limit recorded in the design. An unimplemented idea is
not itself a blocker: record its reason for deferral and concrete reopen
evidence. Do not keep trying variants solely to prove that none can improve.

Use proportional validation under the [decision workflow](historical-index.md#how-a-decision-advances).
Use the affected consumer's measurements and any useful kernel attribution;
composition-only work shares its consumer's evidence. User-facing completion needs
applicable real-data and resource/scaling evidence with fixture limitations stated. Public
replacement still needs matching canonical workflow and compatibility checks.
