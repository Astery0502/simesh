# Supported Feature Migration

## Final Goal

Every supported, user-observable capability currently owned under
`src/simesh/` must have an explicit disposition, contract, validation path, and
production integration decision. The new functional implementation must carry
the declared supported behavior before the current computational path is
retired.

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
| Array/index/layout conventions | `amrvac/layouts.py`, canonical array APIs | Core 3D conventions, workspace accounting, dense primary traversal, level-1 closures, and refined source-union planning complete; singleton-z pending | M1 transfer integration, then M2/M6 |
| Morton, forest, leaf traversal, connectivity | `utils/lib/amr/morton.pyx`, `forest.pyx` | Level-1/refined 3D reconstruction, conformance, contacts, all-touch balance, selected direction/source records, and bounded support planning complete; optional face cache deferred | M1 transfer integration, then M2 quadtree and M3 periodic |
| Mesh geometry and coordinate bookkeeping | `utils/lib/amr/mesh.pyx` | Level-1 and refined Cartesian 3D bounds/spacing complete; refined centers remain sampling-owned | M1 sampling integration, then M2/M3 |
| Physical, sibling, coarse/fine and periodic halos | `amrvac/boundary.py`, `mesh.pyx` | Physical/level-1 sibling boundaries plus refined restriction, limiter, prolongation, bounded source-slot mapping, and directed target boxes complete; child phases, source/workspace plans, and refined execution pending | M1 refined, then M2 and M3 |
| Exact/uniform sampling and uniform-to-SFC placement | `mesh.pyx`, `amrvac_uniform.py` | Level-1 3D placement/zero/trilinear complete | M1/M2/M3/M5 |
| Pointwise, stencil, reduction and diagnostics | `derived_fields.py`, `mesh.pyx`, legacy diagnostic evidence | Representative contracts complete | M4 |
| Derived-field registration/materialization and field selectors | `amrvac_dataset.py`, `derived_fields.py` | Not migrated | M4 and M6 |
| AMRVAC header/forest/tree/block reading | `amrvac/datio.py` | Array/memmap adapter only | M1 native refined read slice; full M5 |
| AMRVAC writing and roundtrip | `amrvac/datio.py`, `amrvac_uniform.py` | Not migrated | M5 |
| Uniform construction, singleton-z conversion and VTK export | `amrvac_uniform.py`, `layouts.py` | Core placement only | M2 and M5 |
| Dataset lifecycle and public `simesh.amrvac` API | `amrvac_dataset.py`, `api.py`, `dataset_base.py` | Not integrated | M6 |
| Independent potential-field and configuration helpers | `tools/potential_field.py`, `utils/configurations.py` | Inventoried; some already function oriented | M4 retain/rewrite audit |
| Runtime/OpenMP/build/package exports | `utils/runtime.py`, build files, package `__init__` modules | Rewrite-local build only | M7 |
| `src/simesh/legacy/` duplicate/reference behavior | legacy tree | Evidence source, not default target | Classify then retire/archive in M7 |

When a family becomes active, split it into dependency-ordered capabilities in
`CAPABILITIES.md` and link exact parity tests and benchmarks. This table remains
the compact source-wide ledger rather than expanding into a method inventory.

## Vertical Slice Rule

Core milestones are not complete from synthetic kernels alone:

- M1 reads at least one representative refined Cartesian 3D `.dat` through a
  block adapter and runs topology, halos, sampling, and an operator end to end.
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
