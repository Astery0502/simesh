# Supported Feature Migration

The 2026-09-08 analysis-core integration is additive. Its actual source-family
dispositions and compatibility evidence are recorded in
[P4 integration](../docs/analysis-core/evidence/p4-integration.md). Canonical
defaults remain available; that delivery does not claim this ledger's final
cutover or retroactively complete historical M2--M7 milestones.

## Final Goal

Every supported, user-observable capability currently owned under
`src/simesh/` must have an explicit disposition, contract, validation path, and
production integration decision. The new functional implementation must carry
the declared supported behavior before the current computational path is
retired.

[baseline.md](../docs/analysis-core/baseline.md) separately assesses what to retain, adapt, reconsider
or use as reference on the new path. Those roles are not migration closure:
all rows and public compatibility obligations here keep their existing status.
Milestone labels below identify dependency/closure families, not current priority.

This is not a requirement to copy every source file, class, method, or line.
Canonical behavior is the migration unit. Internal algorithms and object models
may be replaced. Independent helpers that already satisfy the functional
boundary may be retained and composed. Legacy modules provide evidence and are
retired or archived rather than automatically reimplemented.

## Status And Disposition

Each feature-family entry progresses through:

```text
inventoried -> contracted -> implemented -> integrated -> replaced
```

One of these explicit dispositions closes an entry:

- `rewrite`: new functional implementation owns the behavior;
- `adapter`: current/external representation feeds a rewrite contract;
- `retain`: an already suitable independent function remains canonical;
- `replace`: a different implementation provides contract-equivalent behavior;
- `retire`: behavior is obsolete and removed with compatibility evidence;
- `unsupported`: behavior is recognized and rejected intentionally.

No canonical feature may remain merely `inventoried` at final cutover.

## Current Source Feature Map

| Feature family | Current authority | Rewrite state | Planned closure |
| --- | --- | --- | --- |
| Array/index/layout conventions | `amrvac/layouts.py`, canonical array APIs | M1 3D conventions complete; DIM-001 active-dimension/singleton-z provenance remains proposed and queued | M2 prerequisite, earlier if shared post-M1 representations require it; then M6 |
| Morton, forest, leaf traversal, connectivity | `utils/lib/amr/morton.pyx`, `forest.pyx` | Level-1/refined 3D reconstruction, conformance, contacts, all-touch balance, selected direction/source records, and bounded support planning complete; optional face cache deferred | M2 quadtree and M3 periodic |
| Mesh geometry and coordinate bookkeeping | `utils/lib/amr/mesh.pyx` | Level-1/refined Cartesian 3D bounds, spacing, ROI, and hinted/exact ownership complete | M2 active-axis geometry, then M3 |
| Physical, sibling, coarse/fine and periodic halos | `amrvac/boundary.py`, `mesh.pyx` | Physical/level-1 sibling boundaries plus selected Cartesian 3D refined support, SAME/FINER/COARSER application, PBC-aware slope completion, and bounded complete-halo reader/writer execution are complete for `B>=4`, side reach `<=B/2`; 2D/periodic and wider-reach variants pending | M2 active-dimension halos, then M3 periodic |
| Exact/uniform sampling and uniform-to-SFC placement | `mesh.pyx`, `amrvac_uniform.py` | M1 refined repeated/cached sampling and field-line composition complete | M2 bilinear/singleton-z, then M3/M5 |
| Pointwise, stencil, reduction and diagnostics | `derived_fields.py`, `mesh.pyx`, legacy diagnostic evidence | Original M1 selected curl/reduction and fixed field-line contracts complete; general access/validity requirements apply to future consumers | Post-M1 analysis, M2 representative 2D operator, then M4 breadth |
| Derived-field registration/materialization and field selectors | `amrvac_dataset.py`, `derived_fields.py` | Public lifecycle not migrated; internal validity/reuse composition planned | Concrete post-M1 consumers, then complete M4/M6 lifecycle |
| AMRVAC header/forest/tree/block reading | `amrvac/datio.py` | Native spatially-3D v5 metadata/FST binding and selective non-staggered reader complete | M2 variable-width 2D v5; full M5 |
| AMRVAC writing and roundtrip | `amrvac/datio.py`, `amrvac_uniform.py` | Not migrated | M5 |
| Uniform construction, singleton-z conversion and VTK export | `amrvac_uniform.py`, `layouts.py` | Core placement only | M2 and M5 |
| Dataset lifecycle and public `simesh.amrvac` API | `amrvac_dataset.py`, `api.py`, `dataset_base.py` | Not integrated | Bounded internal analysis/output composition after M1; complete public compatibility and lifecycle in M6 |
| Independent potential-field and configuration helpers | `tools/potential_field.py`, `utils/configurations.py` | Inventoried; some already function oriented | M4 retain/rewrite audit |
| Runtime/OpenMP/build/package exports | `utils/runtime.py`, build files, package `__init__` modules | Rewrite-local build only | M7 |
| `src/simesh/legacy/` duplicate/reference behavior | legacy tree | Evidence source, not default target | Classify then retire/archive in M7 |

When a family becomes active, split it into dependency-ordered capabilities in
`CAPABILITIES.md` and link exact parity tests and benchmarks. This table remains
the compact source-wide ledger rather than expanding into a method inventory.

## Public Compatibility And New Analysis Scope

Before replacing a user workflow, record preservation or intentional change of
signatures, defaults, field selectors/names, layouts, mutation/lifecycle, and
supported failures. Internal freedom does not imply unreviewed public breakage.
Use WORKFLOW's material review and retain fallback/rollback before retirement.

Post-M1 derived validity, reuse, and along-field analysis advances concrete
scientific/session work before broad M2. It does not close a public feature row
merely because an internal primitive exists. New field-line behavior has an
explicit new-feature contract instead of an invented current-path parity claim.
Future device support likewise needs actual conformance/build evidence; the
portability requirement alone closes no backend support entry.

## Vertical Slice Rule

Core milestones are not complete from synthetic kernels alone:

- M1 reads at least one representative refined Cartesian 3D `.dat` through a
  block adapter and runs topology, halos, sampling, and an operator end to end.
- Its historical closure has the explicit fixture qualification in
  [M1 horizon](evidence/M1-ARCHITECTURE-HORIZON.md); this refactor adds no new
  direct real refined non-staggered validation claim.
- Post-M1 analysis closes the new local-derived, repeated-analysis, and
  along-field outcomes in ANALYSIS_WORKLOADS with scientific/resource evidence.
- M2 repeats a real Cartesian 2D singleton-z workflow.
- M3 distinguishes parity with current periodic behavior from genuinely new
  periodic capability and records both honestly.
- M4 materializes representative derived/scientific fields through the same
  access and execution contracts.
- M5 proves read/write roundtrip, array construction, and export workflows.
- M6 proves the canonical public API and dataset lifecycle.
- M7 proves clean install/build, parallel/runtime behavior, backend switching,
  fallback/rollback, and production cutover.

## Final Cutover Gate

Cutover is complete only when:

- every feature-family entry has a closing disposition;
- supported public workflows run through the new implementation by default;
- parity and intentional differences are documented and tested;
- representative real-data correctness, performance, memory, and I/O reports
  pass the milestone criteria;
- editable and clean source builds work on supported environments;
- fallback and rollback are exercised before removing the old canonical path;
- no production import depends on legacy/reference computational state.
