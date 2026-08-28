# Performance And Benchmark Policy

## Meaning Of High Performance

Correctness is the first gate. Among correct implementations, performance is a
multi-objective result over runtime, throughput, peak and managed memory, I/O,
allocation, transfer amplification, cache behavior, and parallel scaling. The
rewrite does not require one strategy to dominate every workload.

Resident, bounded, cached, and framework-specific strategies share semantic
contracts but receive separate measurements. A bounded path may be slower than
resident execution while still being the preferred result when it makes a
larger-than-memory workload possible. Such a trade-off must be quantified.

## Benchmark Levels

### 1. Kernel

Measure the smallest established transformation:

- cells/s, blocks/s, or payload/effective GB/s;
- fixed call/validation cost;
- temporary and retained allocation;
- block shape, field count, halo width, relation type, and 2D/3D effects;
- simple versus optimized implementation.

### 2. Composition

Measure the immediate useful pipeline:

- read/gather, support planning, halo, compute, sample, and write stages;
- primary/support counts, chunk count, and transfer amplification;
- workspace reuse and capacity scaling;
- resident, bounded, and cached strategies;
- adapter, I/O, and compute time separately.

### 3. Real User Workflow

Measure representative canonical work:

- metadata/open and selective block reads;
- zero-order and linear uniform sampling;
- refined and 2D ghost exchange;
- derived-field/derivative materialization;
- `.dat` read/write roundtrip and uniform construction/export;
- public dataset/API workflows.

### 4. Scaling And Resources

Vary cells, leaves, fields, refinement depth/distribution, workspace capacity,
and thread count. Record managed bytes, peak RSS, major/minor page faults,
bytes read/written, cold/warm cache behavior, speedup, and parallel efficiency.

## Comparators

- The independent Python/NumPy reference establishes semantics, not speed.
- The current canonical Cython path is the main migration performance baseline.
- Legacy is a performance comparator only where it still represents an
  independent supported workflow.
- The simple resident rewrite is the composition reference for bounded/cached
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

The selected compact result is committed under `rewrite/evidence/<ID>.md` with
the exact reproduction command, environment summary, baseline ratio, worst
correctness discrepancy, memory trade-off, and reason the retained variant was
chosen. Milestones add an integrated evidence summary instead of relying only
on microbenchmarks.

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

## Capability And Milestone Completion

A hot-path capability requires both its kernel benchmark and the immediate
composed benchmark. A milestone requires at least one real-data workflow plus
scaling/resource evidence. Public cutover requires same-runner comparisons of
the canonical user workflows, not a collection of isolated fast kernels.
