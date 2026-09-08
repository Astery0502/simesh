# Rewrite Agent Instructions

## Scope

These instructions apply under `rewrite/`. Write project content in English.
The current `src/simesh/`, tests, data, and reports are read-only evidence unless
the task explicitly authorizes changes outside the rewrite. Keep development
isolated until the production integration stage.

## Resume And Read By Task

The independent N4 core is fixed; new application work starts at
[docs/analysis-core](../docs/analysis-core/README.md). This file governs explicitly
requested work on existing rewrite implementations only. Archived project
procedures below do not impose a workflow on applications using the N4 core.

Start each work cycle with [CURRENT.md](CURRENT.md). For spec convergence or
selection of a new outcome, read [baseline.md](../docs/analysis-core/archive/core-development-2026-09-08/baseline.md) before consulting
[CAPABILITIES.md](CAPABILITIES.md) and the selected design/contract. Resume the
authorized work; historical completion does not select the next implementation.

| When | Read | Authority |
| --- | --- | --- |
| Converging requirements, assessing existing assets, or entering a new implementation phase | [baseline.md](../docs/analysis-core/archive/core-development-2026-09-08/baseline.md) | Requirement status, asset roles, responsibility/evidence gaps, first-outcome readiness |
| First rewrite cycle, changed intent, or milestone boundary | [intent.md](../docs/analysis-core/archive/core-development-2026-09-08/intent.md) | Product goals, durable engineering constraints, scope |
| Selecting a consumer, algorithm, cache, or scientific workflow | [workflows.md](../docs/analysis-core/archive/core-development-2026-09-08/workflows.md) | Workload acceptance and priorities |
| Changing a storage, execution, field, halo, or compute boundary | [FUNCTIONAL_COMPOSITION.md](FUNCTIONAL_COMPOSITION.md) | Architecture and substitution |
| Resolving unclear responsibility or coupling | [DECOMPOSITION.md](DECOMPOSITION.md) | Optional design questions and focused refinement |
| Starting work, changing a contract, or closing a checkpoint | [WORKFLOW.md](WORKFLOW.md) | Exploration, review, validation, and commits |
| Measuring or claiming performance/resource behavior | [performance.md](../docs/analysis-core/archive/core-development-2026-09-08/performance.md) | Evidence classes, budgets, comparisons, stopping rule |
| Planning a feature-completion real-data assessment | [WENO-REFERENCE.md](WENO-REFERENCE.md) | Explicit opt-in reference cases, comparison scope and reproduction |
| Choosing or revising a stage | [ROADMAP.md](ROADMAP.md) | Stage outcomes and sequencing |
| Migrating behavior or preparing integration/cutover | [SOURCE_MIGRATION.md](SOURCE_MIGRATION.md) | Feature disposition and public compatibility |

Read the relevant authority once per active context; follow links for details.
Do not reread all process documents during routine member iteration. Keep each
rule in its owning document. The checkpoint selects work; it does not override
a numerical contract. When intent conflicts with a stable contract, record the
conflict and resolve it with source/contract evidence and focused core checks.
Use WORKFLOW's optional review only when an important question remains unresolved.

## Operational Constraints

- Prefer free functions with explicit dependencies, ownership, state, and valid
  regions. Keep storage, scheduling, and compute decisions at their declared
  boundaries. Explicit reusable sessions and mutable workspaces are allowed.
- Use Cython typed memoryviews, structs, pointers, and contiguous buffers where
  appropriate. Keep Python callbacks out of cell, stencil, and block hot loops;
  avoid hot-loop allocation unless measurement justifies it.
- Checked standalone wrappers validate their complete contracts. An executor
  with equivalent preflight may call explicit unchecked kernels internally.
- Preserve simple semantic references and existing strict contracts when
  introducing fusion, parallelism, another layout, or another compute backend.
- Follow WORKFLOW for concrete outcomes, short checkpoints and incremental
  refinement. Default to direct inspection and one agent; use independent
  sub-agent review only for a specific unresolved core question where it helps.
- Select tests for core behavior and efficiency. Read descriptive metadata,
  incidental call order and simple unchanged physical-quantity wiring directly;
  do not create per-helper suites or routinely rerun complete regressions.
- Keep documentation proportional. Do not add speculative frameworks, global
  backend dispatch, hashes, signatures, or integrity manifests as governance.
  Validate demonstrated invariants at the boundary that owns them.
- Use `.venv/bin/python` for local Python commands. The rewrite build helper is
  `rewrite/build_ext.py`; current reproduction commands belong in CURRENT.
- Prefer completion-triggered waits. Otherwise poll long-running work no more
  often than every 120 seconds unless a shorter interval is necessary.

The supported long-running development loop and optional review rules
have one definition in WORKFLOW. Documentation-only work uses its documentation
checks and never reports numerical capabilities as newly complete.
