# P4 Installed Workflow And Source Integration

Updated 2026-09-08. Executable source checkpoint: `a5f1f88` and subsequent
acceptance-record changes. Selected requirements/decisions remain in
[native-core-design](../native-core-design.md).

## Installed Source Workflows

`open_source` owns an immutable v5 file context and exposes selected ordinary
fields as explicit logical columns. `original_field_ids` preserves the file
mapping. It reads original WENO records directly, validates the three-component
staggered tail size, and never exposes CT fields. Original DAT-003 still rejects
staggered indices. Full source/field/record/endian checks and callback preflight
remain intact; no invalid tail is silently accepted for speed.

The new factory reuses existing read/coalescing/copy logic. Little/big-endian
saved-ghost records, repeated/permuted selectors and complete-output preservation
on a truncated tail passed alongside the original reader suite (21 tests).
A real-format synthetic v5 file verified named source mapping, interior-only
products, prepared data, closed-fd failure and detached resident lifetime.

`open_prepared` uses an admitted full interior read plus canonical bulk halos,
returning owner-anchored readonly native backing. `prepare(..., halo=0)` exposes
native interiors without halo work. `sample_plane` and `iter_uniform` accept
bounded pools; uniform output is delivered as owned z slabs, never a full
coordinate cube or implicit concatenation.

## Same-Result WENO Source Comparison

The fixed eight-seed/eight-step B request used the immutable original WENO file:

| Source path | File to short trace | Pool controlled bound |
| --- | ---: | ---: |
| Direct ordinary-prefix file source | 0.129153 s | 18,310,009 bytes |
| Canonical eager B read plus array source | 5.118067 s | 294,462,570 bytes |

Trace arrays were equal and selected interior bits matched exactly. Direct input
used 646,432 metadata bytes in nine calls plus 2,351,592 reader bytes in 382 calls,
about 3 MB total; no bridge or eager B array was created. Measured library import
startup was another 0.280854 s. This is a same-result source workflow comparison,
not a warm-kernel ratio. Raw record: `native-source.json` under ignored results.

The initial checked full-domain read took 6.837 s before resident preparation.
Inspection/probes kept all checks while improving two different costs:

- Cached zero-saved-ghost shape/byte facts and contiguous field-copy plans reduced
  complete reader CPU time about 15--19% on B/rho. Wall time improved only about
  1--2%; I/O wait dominated. All bits remained equal, and the general path remains
  an internal reference.
- Private dense loading now uses sequential one-record header/payload order.
  B wall median fell 7.495 -> 5.012 s; rho 4.240 -> 4.042 s. CPU cost rose
  (~.63 -> .92 s for B; .36 -> .73 s for rho), an explicit locality trade-off.
  Per-callback preflight remains complete. The raw callback keeps an explicit
  batch-size option; bounded halo reads retain their batch behavior.

These bounded probes are closed. Do not claim universal speed across storage/
cache conditions, or attribute I/O wait to numerical kernels. Raw records are
`reader-records.json` and `read-order.json`. The selected settings use workload
structure, not hardcoded WENO field IDs.

## Packaging, Build And Compatibility

The distribution explicitly bundles `simesh_rewrite` as its retained provider.
Numerical kernels remain independent of file/cache providers. Source conveniences
load that provider lazily; installed workflows need no manual checkout PYTHONPATH.
Canonical and provider Cython directives are separate, preserving the provider's
original language-level-only configuration rather than inheriting cdivision.

Editable install passed. A fresh source copy excluding generated C/extensions
and egg-info built a 9,354,669-byte wheel in 49.345 s. An isolated `python -S`
process, without editable path hooks, imported both packages from the extracted
wheel and completed file write/read, bounded tracing and detached sampling.
The first harness copy accidentally included stale egg-info with absolute paths;
excluding that generated metadata fixed the source-discovery failure.
Raw build/import record: `install.json`; reproduction:

```sh
.venv/bin/python scripts/analysis_core/validate_install.py
```

Inspection found the old clean helper recursively targeted all repository .so/.c
and egg-info, including .venv. Cleanup now targets only generated siblings of
known package .pyx files and known build directories/metadata. `make clean` is
clean-only; the helper's `--clean` remains clean-before-build. The focused cleanup
test protected virtualenv/reference/result files, and actual scoped clean left
NumPy usable before a complete rebuild.

The supported `make test PYTHON=.venv/bin/python` workflow passed after scoped
clean. Broader integration checks passed once after the bundled build:

```text
provider tests: 1232 passed
canonical public/AMR/helper tests: 92 passed, 2 heavy cases deselected
new analysis composed checks: 17 passed
```

The retained tests cover Dataset/derived lifecycle, selectors, actual 2D
singleton-z/bilinear/coarse-fine behavior, write/read roundtrips, VTK values and
potential-field helper behavior. The opt-in heavy scientific cases remain under
their separate profiles; WENO acceptance was measured explicitly in P1--P3.
OpenMP and declared scale results are recorded in the completed-gates section below.

## Source Feature Disposition For This Additive Integration

| Feature family | Disposition and actual scope |
| --- | --- |
| Cartesian 3D topology/geometry/halos | Reused adapters: exact-phase bounded RHE and canonical coordinate-phase resident preparation, with measured validity and costs |
| Native fields, F/D/scalar LOS | New analysis implementations with independent groups, owned/scoped lifetimes and explicitly limited scientific definitions |
| Cartesian 2D | Retain canonical APIs; actual generated 2D read/write/bilinear/coarse-fine checks pass. New native sources reject 2D instead of inventing extrusion/field-line semantics |
| Periodic metadata/I/O | Retain and verify flags/data through write/read and VTK workflows; new native source rejects periodic flags |
| Periodic ghost computation | Not delivered: canonical forest currently initializes periodic connectivity off. Metadata preservation is not a wrapping-halo or periodic-trace claim |
| Mutable Dataset and derived registry | Retain existing public behavior; no redirection/removal |
| AMRVAC writing, construction and VTK | Retain canonical writers/helpers; new native products support array/memmap/stream sinks without claiming arbitrary ghost serialization |
| Independent scientific helpers | Retain; affected existing scientific helper checks pass |
| Build/runtime | Explicit provider packaging, clean-source wheel and scoped cleanup; CPU seed/ray parallelism has measured evidence |
| Legacy/archive | Retained reference paths, not the new numerical core's runtime dependency; no retirement/cutover |
| Non-Cartesian or CT computation | Explicitly unsupported by the selected new core; ordinary-field extraction is a distinct file-format capability |

This table scopes analysis-core integration; it does not relabel the old rewrite
ledger's M2--M7 milestones or claim its final default-path cutover. Canonical
public workflows remain available as the supported default and rollback path.

## Remaining Acceptance

P3-L physical response/EOS/units are pending user input. WENO/tdm lack energy or
T; the small spherical reference is outside the selected geometry. Repository
inventory found ~1 GB WENO, ~1.5 MB tdm and ~3.6 MB spherical bw, not a 10--20 GB
Cartesian input or a real 2D fixture. Million-seed and 1000^3 streamed-output
acceptance are feasible and run separately; neither substitutes for unavailable
large-input or thermal-image acceptance. Exact boundary footpoints, Q, GPU and
new periodic/2D native consumer semantics are not claimed.

## Completed Runtime And Feasible Scale Gates

OpenMP was enabled successfully (`openmp_version=202011`); 23 AMR checks passed
with four OpenMP threads. A forced default AMR rebuild restored non-OpenMP status
before scale timings. Independent analysis parallelism continues to use its
validated nogil CPU worker path.

The [actual scale record](p4-scale.md) closes the declared million-seed and 1000^3
output cases: one/four-worker million-seed results agree, with 8.176/2.274 s trace
times for the bounded 32-step profile; all billion uniform positions were valid,
delivering 24 GB of values cumulatively in 81.283 s with 33 MB maximum slabs.
High-water RSS was 1.438 GB. It does not replace the missing large-input fixture.
