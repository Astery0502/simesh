# Current Rewrite Checkpoint

The active analysis-core run is governed by
[its current checkpoint](../docs/analysis-core/current.md), which supersedes the
historical authorization notes below. The user activated P0--P4 on 2026-09-08;
native field/F/D/scalar-LOS and installed-provider work has now been delivered
within its recorded scope. The provider adds a separate ordinary-prefix reader
for 3D staggered-tail records; DAT-003's original factory and rejection remain
unchanged. The old M0/M1/M2 history below is not the new core's execution state.

## Active State

- No rewrite implementation is active. Project-level spec work has moved to
  [docs/analysis-core](../docs/analysis-core/README.md). Its current shared
  decision is ghost preparation plus prepared-data consumption, with at least
  two valid input layers for one-layer differentiation then trilinear sampling.
  This file retains rewrite history; it does not select the new architecture.
- Baseline updated from the corrected side handoff: independent F tracing/Q/twist,
  D whole-domain derivatives/slices and L full-domain LOS synthesis. The
  [result-contract draft](../docs/analysis-core/archive/core-development-2026-09-08/pipeline-results.md) records known scale,
  R1--R10 responsibility/evidence gaps and revised first-outcome entry conditions. No
  numerical/performance suite or experiment was run; code and contracts unchanged.
  Independent review included the latest clarification: prioritize seed-parallel
  tracing; retention is optional and selected seeds may be retraced explicitly.
- Delivered groups: **Owned Halo Preflight (HPR-001)** and **Cache Query Planning
  (CQP-001)**. HPR checkpoint: `da861f1`; CQP and round closure have their own
  executable checkpoint. See [round evidence](evidence/M1-OPTIMIZATION-ROUND.md).
- Recorded optimization outcome: native refined local analysis 22.8--37.0%
  faster; 36/71-owner indexed
  warm batches 18.5/36.1% faster beyond HPR. Numerical results, I/O counts and
  managed-array formulas are unchanged. Full rewrite suite: 1,229 passing tests.
- Timing limitation: shared-runner scheduling jitter affects some wall-clock
  controls; paired/CPU-time evidence and raw outliers are recorded in the summary.
- Follow-up complete: adopted the opt-in [WENO reference](WENO-REFERENCE.md) norm
  and completed the [M1 reference comparison](evidence/M1-WENO-REFERENCE-COMPARISON.md).
  Local-field evidence was reused; canonical, full-halo, sampling/cache and short
  trajectory gaps were measured. Recorded focused checks at closure: 92 passed.
- Principal finding: full-domain two-layer halo values agree within 1e-12, but
  canonical resident refresh is about 0.31 s versus bounded RHC 38.20 s including
  reads/output copies; action preflight alone is about 23.3 s. Preserve explicit
  resident/bounded trade-offs and prioritize planning/check orchestration review.
- Intent clarification: documented and independently reviewed, 2026-09-07,
  user-requested. The expanded
  [analysis catalog](../docs/analysis-core/archive/core-development-2026-09-08/workflows.md) defines requested results, minimum
  data needs, reuse, resident/bounded trade-offs and acceptance. It adds no
  numerical implementation or completion claim. Documentation checks cover
  requirement preservation, local links/anchors, status consistency and diff;
  no numerical or performance suite was rerun for this prose-only change.
- Workflow-to-technique mapping: initial open inventory documented, 2026-09-07.
  [WORKFLOW_TECHNIQUES](../docs/analysis-core/archive/core-development-2026-09-08/technique-candidates.md) relates twelve workflow families
  and their usage variants to eighteen candidate technique records, with
  source/evidence status, limits and unresolved questions. This is a
  documentation checkpoint; it selects no combined architecture or implementation.
  Source cross-checks, workflow coverage, local links and diff checks passed;
  no numerical/performance suite was run for this inventory.
- Lifetime/execution exploration: draft documented and independently reviewed,
  2026-09-07. The
  [shared design](../docs/analysis-core/archive/core-development-2026-09-08/lifetime-sketches.md) describes ownership, invalidation
  and release, three continuous usage sequences, alternative execution sketches
  and costs of optional mechanisms. No final architecture or implementation is
  selected; numerical contracts and capability statuses are unchanged.
  Documentation links, scenario coverage and diff checks passed; no numerical
  or performance suite was run for the draft.
- For new spec work, follow the project checkpoint and
  [prepared-field requirements](../docs/analysis-core/archive/core-development-2026-09-08/prepared-fields.md).
  No new pipeline implementation, code deletion or historical milestone
  progression is authorized. Existing WENO commands below remain opt-in evidence.
- Original M0/M1 completion remains unchanged, including Native Refined Field
  Lines (FLN-001, RKS-001, TRM-001, SLE-001).
- M2 Active-Dimension Foundation is queued; DIM-001 remains proposed and is
  required before any 2D consumer or shared generalization that needs it.
- Documentation realignment: 2026-09-05, user-requested. See
  [change record](evidence/DOCUMENTATION-REALIGNMENT.md).
- Exact capability status: [CAPABILITIES.md](CAPABILITIES.md).
  Group readiness and gates: [WORKFLOW.md](WORKFLOW.md).

The prior optimization round delivered only HPR/CQP; it did not deliver the
broader derived-field/diagnostic lifecycle. Current selection follows BASELINE.
Operator access and output validity remain general consumer requirements, not
a separate field-specific gate. Historical claims retain their original scope.

## Current Constraints

- Retain signed int64 IDs and canonical float64 host layout under existing
  contracts. FUNCTIONAL_COMPOSITION defines future compute substitution without
  weakening strict arithmetic, ownership, or failure behavior.
- RHE currently supports balanced nonperiodic 3D, even block extents at least
  four, and each side reach at most half a block. Prove each consumer's reach fits;
  do not silently reduce reach or infer validity from allocation.
- Specify extended-input differentiation versus derived-interior exchange,
  cell-average/point interpretation, and interface/boundary error before
  promoting an interpolatable derived result.
- Completed-halo caching, all-26 reference execution, and exact hinted location
  are retained. Reopen support/storage/cache strategies only from changed
  workload assumptions or measured costs; account for active dimensions.
- No direct real refined non-staggered native Cartesian 3D fixture is recorded.
  Preserve the [M1 qualification](evidence/M1-ARCHITECTURE-HORIZON.md).
  No real Cartesian 2D fixture is recorded either.
- Staggered payloads and non-Cartesian computation remain outside scope.
  Full migration, 2D, periodicity, scientific breadth, I/O/write/export,
  public API, packaging, and cutover remain in ROADMAP/SOURCE_MIGRATION.

## Resume

1. For future-core spec work, use docs/analysis-core/README.md. For an explicitly
   selected existing rewrite task, use this checkpoint, CAPABILITIES and its
   scoped contract; do not import historical priority into the new project.
2. Steps 0--5 are resolved in round evidence. Reopen a deferred direction only
   when its trigger is met; do not restart the completed audit or all benchmarks.
3. Freeze the consumer, contract and resource acceptance before new authorized
   work. DVA-001 remains withdrawn; no field-specific gate is pending.
4. Broader diagnostics and M2/DIM-001 remain queued, not automatically activated.

## Optimization Reproduction

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_hpr_001.py rewrite/tests/test_cqp_001.py rewrite/tests/test_rhe_001.py rewrite/tests/test_chs_001.py rewrite/tests/test_lfe_001.py rewrite/tests/test_sle_001.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/m1_optimization.py --family lfe --profile standard --repeats 5 --dat data/tdm.dat --output rewrite/benchmark-results/m1-opt-lfe-reproduced.json
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/m1_optimization.py --family chs --profile standard --output rewrite/benchmark-results/m1-opt-chs-reproduced.json
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/m1_optimization.py --family sle --profile standard --output rewrite/benchmark-results/m1-opt-sle-reproduced.json
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/m1_cache_scaling.py --paired --output rewrite/benchmark-results/m1-opt-cache-reproduced.json
```

Add `--baseline` to restore original M1 preflight and cache planning from
`397aaffc61be68d1141c8f30c0427dd37a4c8bc8`. Attribution, resource and drift commands
are in the evidence. User documentation realignment is preserved separately
from optimization-owned executable commits.

## Existing Reproduction Commands

These reproduce the previous executable state; they were not rerun for the
documentation change. The last field-line evidence records 301 focused/
dependency checks and 1,210 rewrite tests passing.

```text
.venv/bin/python rewrite/build_ext.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests/test_fln_001.py rewrite/tests/test_rks_001.py rewrite/tests/test_trm_001.py rewrite/tests/test_sle_001.py rewrite/tests/test_chs_001.py rewrite/tests/test_rps_001.py rewrite/tests/test_dat_003_integration.py
PYTHONPATH=rewrite/src:src .venv/bin/python -m pytest -q -p no:cacheprovider rewrite/tests
PYTHONPATH=rewrite/src:src .venv/bin/python rewrite/benchmarks/sle_001.py --profile standard --output rewrite/benchmark-results/sle-001-standard.json
```

Evidence: [last group](evidence/NATIVE-REFINED-FIELD-LINES.md),
[M1 horizon](evidence/M1-ARCHITECTURE-HORIZON.md),
[historical analysis decisions](evidence/M1-ANALYSIS-DECISIONS.md).
