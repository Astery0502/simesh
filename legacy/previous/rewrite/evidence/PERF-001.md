# PERF-001 Evidence

## Existing Evidence Audit

Every completed numerical capability already has a focused benchmark and
committed evidence summary, including runtime, throughput, allocation, memory,
chunk/capacity behavior, numerical comparison, and composed trade-offs where
relevant. The repository also has canonical scaling, 2D validity, ghost/
interpolation, and OpenMP thread benchmark/report tooling that writes machine
metadata and JSON/CSV/Markdown artifacts.

The missing layer was a shared definition connecting those measurements to
milestone and final-product claims. Existing results are mainly M0 synthetic
and single-machine; they do not yet establish refined, 2D, real `.dat`, public
workflow, or parallel rewrite performance.

## Protocol Result

`PERFORMANCE.md` now defines four levels: kernel, composition, real workflow,
and scaling/resources. It separates correctness references, canonical Cython
baselines, resident rewrite baselines, bounded/cached strategies, I/O/adapters,
and parallel one-thread comparisons. It also defines smoke/standard/large
profiles, ignored raw artifacts, committed compact evidence, environment and
repetition records, and controlled same-runner regression decisions.

Independent review agreed on investigation above a default controlled 10%
median regression and a default completion block above 20% across two stable
representative cases, subject to explicit domain-specific rules and measured
memory/I/O/determinism/complexity trade-offs. No cross-machine absolute timing
gate or noisy smoke-CI performance gate was introduced.
