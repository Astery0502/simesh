# PERF-001 Contract: Performance And Benchmark Evidence

## Responsibility

PERF-001 defines what high performance means, which benchmark levels establish
it, how results are compared and recorded, and when a regression blocks
capability or milestone completion. It does not prescribe one optimization,
backend, parallel runtime, or universal absolute timing.

## Stable Benchmark Levels

1. kernel: throughput, call cost, allocation, and parameter scaling;
2. composition: stages, chunk/support amplification, strategy, transfer, and
   workspace behavior;
3. real workflow: representative `.dat`, dataset, derived, construction,
   export, and public API work;
4. scaling/resources: cells/leaves/fields/refinement/capacity/threads plus
   memory, I/O, page faults, speedup, and efficiency.

Correctness references establish semantics. The current canonical Cython path
is the primary migration performance baseline, the simple resident rewrite is
the bounded/cached composition baseline, and every parallel result includes its
matching one-thread run.

## Recording And Gates

Raw JSON/CSV/report artifacts and repetitions are stored under ignored
benchmark results with machine/build/profile metadata. Selected compact
evidence, exact commands, median/dispersion, baseline ratio, correctness,
memory, and trade-off conclusions are committed under `rewrite/evidence/`.

Correctness and exact budget/output rules are hard gates. Same-runner standard
comparisons own performance gates. By default a controlled median regression
above 10% is investigated; above 20% in at least two stable representative
cases blocks completion unless explicit material memory, I/O, determinism, or
complexity evidence justifies it. Cross-machine absolute timing and noisy smoke
CI do not enforce tight thresholds.

A hot-path capability needs kernel and immediate composition evidence. A
milestone needs a real workflow and scaling/resource summary. Public cutover
needs canonical user-workflow comparisons.
