# PERF-001 Design: Performance Evidence Protocol

## Problem

Capability evidence already records useful runtime and memory measurements, but
separate microbenchmarks cannot establish that the final rewrite is high
performance. Results need consistent levels, current/simple baselines,
environment records, real workflows, resource scaling, and material-regression
decisions without turning noisy cross-machine timings into correctness gates.

## Selected Direction And Review

Keep small capability scripts and reuse the repository's existing JSON/CSV/
Markdown reporting conventions. Define kernel, composition, real workflow, and
scaling/resource levels. Raw runs remain ignored; compact selected evidence is
committed. Same-runner relative comparisons drive gates, while absolute
timings remain descriptive.

Independent review agreed that correctness and memory formulas remain hard
gates; controlled median regressions above 10% require investigation and above
20% across stable representative cases block completion by default unless a
material memory, I/O, determinism, or complexity improvement is demonstrated.
No benchmark database, global framework, or tight timing gate is added to noisy
smoke CI.
