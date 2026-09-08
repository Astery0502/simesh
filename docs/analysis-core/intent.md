# Analysis Core Intent

## Product Intent

Build a local scientific analysis toolkit for large, mostly immutable AMRVAC
snapshots in their native block-structured AMR representation. The central
problem is to organize physical fields and their support in memory, prepare
ghost data efficiently, and make the resulting data efficient to consume.
The mainlines share ghost-prepared data, including ordinary magnetic tracing's
trilinear samples. Their different consumption geometries and access sequences
are the main design pressure: evolving irregular paths, selectable regular
regions, and full-domain observation rays with different depths and AMR crossings.
Scientific meaning and complete application cost guide the design.

Users work with physical fields over selected regions. They may request an
end-to-end calculation or retain fields for successive exploration; either
style may share work and produce several results. Native interior, prepared
and derived fields are useful products in their own right. Their data and
lifetime contracts belong to [prepared-fields.md](prepared-fields.md).

| Priority | Confirmed outcome | Result definition |
| --- | --- | --- |
| Mainline F | Magnetic tracing, along-line integrals and requested Q/twist; optional trajectories and explicit selected-seed retracing | [F contract](pipeline-results.md#f-magnetic-lines-and-along-line-diagnostics) |
| Mainline D | Whole-domain current/gradient calculation followed by slices for structure analysis | [D contract](pipeline-results.md#d-whole-domain-currentgradient-then-slices) |
| Mainline L | Full-domain integration of a local response/emissivity along observation-defined lines of sight into images | [L contract](pipeline-results.md#l-full-domain-los-integration-of-a-local-response) |
| Supporting outcomes | Metadata/selective native access, local and global ghost preparation, slices/profiles, uniform output, isosurfaces and threshold/isovolume products | [Workflow catalog](workflows.md#concrete-workflow-catalog) |

F and D have similar priority, with F slightly preferred. L is also confirmed.
The mainlines are independent: a diagnostic's local derivative demand does not
require running D first. Isosurfaces are secondary within D. Independent-seed
parallel execution takes precedence over mandatory retention of every path.

Success means correct useful results with acceptable latency or batch throughput
inside the complete accuracy/memory envelope. Support selective bounded access
to payloads exceeding memory and competitive resident preparation when the
selected fields fit. Reuse unchanged reads, geometry, support and computed fields
when it saves actual work. Uniform materialization is an optional output.
Selectivity, chunking, caching and parallelism are strategies to evaluate.

## Operating Envelope

- Typical snapshots are 10--20 GB on personal PCs with RAM in the teens to
  twenties of GB. The available analysis budget and quantitative latency/error
  targets still need to be fixed for each selected case.
- Single-node CPU/OpenMP is in scope. MPI/cross-node work is excluded; GPU is an
  unselected extension, not a prerequisite for the current design.
- Bottom-boundary or section seed grids aim at finest-AMR spatial resolution:
  about 500--600, 1000, or 1700--1800 points per direction (roughly 0.25--3.24
  million seeds on square grids). Seed spacing differs from trace step size.
  Million-seed work is a scaling assessment on suitable resources; users choose
  job size and output retention. Practical smaller cases establish initial
  serial/parallel correctness.
- Full domain specifies physical coverage, not simultaneous RAM residency.
  Uniform outputs may approach 1000^3 and need bounded delivery without mandatory
  concatenation, alongside explicitly affordable full arrays.

Scale arithmetic, not measurements: 1000^3 float64 values occupy 8 decimal GB
per field, or 24 GB for three fields. A million paths with 1000 three-component
float64 points each also use 24 GB before IDs, diagnostics, fields and scratch.
These examples do not fix precision or actual path length.

## Durable Engineering Constraints

- Use a functional core with explicit dependencies, plain metadata, declared
  output/scratch ownership and allowed mutation. Prefer free functions;
  convenience objects, reusable buffers and device state have explicit lifetimes.
  Results are deterministic for the same inputs and declared strategy.
- Separate decisions about logical fields, geometry/topology, support planning,
  transfer arithmetic, storage and scheduling. Keep Python orchestration outside
  hot loops. Concrete fusion is allowed when it preserves contracted semantics.
- Storage adapters, execution strategies and compute implementations must be
  replaceable through explicit boundaries. Kernels have no hidden dependency on
  files, datasets, caches, backends or call history.
- Preserve exact discrete behavior and the arithmetic, mutation and failure
  guarantees of existing entrypoints. New arithmetic, precision, reconstruction
  or reduction order needs a declared reviewed strategy and conformance evidence.
- Keep independent references and scientific acceptance alongside implementation
  conformance. Avoid repeatedly proving immutable facts in hot execution while
  retaining necessary request, validity, ownership and dynamic checks.
- Expose complete memory, I/O, setup, useful-result and scaling costs. Introduce
  small interfaces justified by actual consumers; avoid hypothetical framework
  layers, hidden dispatch, payload hashing, signatures/authentication machinery
  and defensive workflow ceremony.

## Compute Portability

Scientific semantics must be expressible independently of Cython, NumPy object
identity or a device layout. Existing rewrite signed zero-based int64 IDs and
C-contiguous native float64 `(slot, field, x, y, z)` contracts remain binding
for those entrypoints. They do not freeze the future core's representation.

CPU and future device substitutions need explicit mapping and conformance.
Device layout, transfers, residency and synchronization require their own
contracts; Cython is not promised to compile unchanged to GPU. Include all
such costs under [performance.md](performance.md#cpu-and-device-comparisons).

## Scope And Migration

The computational core delivers scientific arrays, samples, curves, geometry
and reductions to Python and visualization consumers. GUI/rendering, HD/MHD
time evolution, arbitrary unstructured input meshes and a universal scientific
workflow engine are outside current goals. Irregular refinement still means
block-structured AMR. Absorption/full radiative transfer and additional
statistics, detection, temporal and spectral techniques remain candidates.

Supported-feature continuity includes Cartesian 3D, Cartesian 2D with
singleton-z interoperability, declared periodic behavior, mutable Dataset and
I/O/write/export workflows, independent helpers, build/runtime, fallback and
rollback. These obligations need not all precede the first isolated outcome.
Staggered computation and non-Cartesian geometry remain unsupported and must be
rejected clearly. Reading ordinary fields from records with staggered tails is
a separate, unselected adapter extension; the current bridge does not provide it.

Every supported observable source feature needs an explicit rewrite, adapter,
retain, replace, retire or unsupported disposition under the
[migration ledger](../../rewrite/SOURCE_MIGRATION.md). Existing code is evidence,
not a file-by-file port list. Before changing a public workflow, identify which
signatures, defaults, layouts, output bytes and failures are preserved or
intentionally changed, with compatibility evidence.

## Extension And Evolution

New CPU kernels may require rebuilding Cython. Runtime numerical callbacks,
a dynamic expression language, JIT and code generation remain out of scope until
several real kernels establish a shared need. Existing coarse-grained storage
and consumer callbacks retain their own scope.

Use the [decision workflow](README.md#how-a-decision-advances) for incremental,
recoverable work and material review. Current spec work does not initiate an
automatic implementation loop or alter historical completion claims.
