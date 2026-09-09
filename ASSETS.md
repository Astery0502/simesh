# Retained implementation assets

The selected implementation now lives at the repository root. Historical
source paths in the provenance tables below identify the pinned revision, not
runtime imports. During root promotion, retained `utils/lib/amr/` extensions
and `utils/lib/{tree,math}.pxd` moved to `amrvac/_mesh/`, `utils/runtime.py` moved
to `amrvac/runtime.py`, and `utils/configurations.py` moved to
`tools/configurations.py`. Their algorithmic provenance is unchanged. There is
no active `simesh.utils` namespace; see [migration](MIGRATION.md).

The subsequent `connectivity.py` and `_kernels/connectivity.pyx` consumer is an
independent implementation of published magnetic-connectivity mathematics using
this package's existing AMR primitives. No FastQSL2 code is bundled or translated.
Scientific references and external comparison provenance are recorded in
[archived method and reference notes](legacy/core-development/documentation-before-application-guide/connectivity.md). FastQSL2's CC BY-NC-SA 4.0 code
is used only as a separately obtained reference; it is not a runtime dependency.

Source revision: `b91bbc015d882ffd7dfcd7e10590cdce559f8532`.

These are selected low-level AMR, file-format and numerical assets. The new
source, field, preparation and consumer orchestration is implemented here.
No runtime import resolves to the donor checkout. Retained code remains under
the repository GPL-3.0 license. Primitive compiler semantics stay separate.

N2 adds `preparation/coordinate.py` and `_kernels/coordinate.pyx`. The latter
retains the grid-index, boundary, coarsening, copy and coordinate-prolongation
arithmetic from `src/simesh/utils/lib/amr/mesh.pyx`, while replacing tree-pointer
access with shared integer geometry and borrowing independent NumPy buffers.
There is no retained AMRMesh/AMRForest or C allocation owner in a published field.

The v2 coordinate scheme explicitly corrects the legacy prolong support tables:
`idphyb=2` requires a fourth boundary row, but the old three-row tables indexed
outside their declared extent. N2 supplies that row and combines the low/high
physical extensions. Constant-field tests cover this changed case; unaffected
cases and the WENO full-domain arrays retain bitwise comparisons with v1.
The local numerical phase order and serial cross-target transfers remain fixed.
N2 also validates coordinate-derived coarse stencil indices before execution.
Large origins or under-resolved coordinates that would index outside the coarse
buffer are rejected rather than entering the unchecked interpolation kernel.

N4 retains the public stateful AMRVAC and independent array-tool paths alongside
the new core. `amrvac/{api,amrvac_dataset,amrvac_uniform,boundary,datio,dataset_base,
derived_fields,layouts}.py` and its `__init__.py`, `tools/{__init__,potential_field}.py`,
`utils/{__init__,runtime,configurations}.py`, the three `utils/lib/amr/{forest,mesh,morton}`
extensions and their headers, and `utils/lib/{math,tree}.pxd` come from the same
pinned revision. Namespace paths are retained and trailing whitespace normalized.
Empty package initializers complete this self-contained dependency subset.
These describe the N4 extraction layout; current retained paths follow the
root-promotion mapping above.

These compatibility modules are not imported by native Source or scientific
consumers. Their stateful AMRMesh and compiler arithmetic remain independent of
the new data model. The compatibility build stays serial because the historical
uniform-grid OpenMP boundary writes have unresolved conformance evidence.
N4 applies the N2 fourth-row prolong support correction to its retained mesh,
rejects periodic ghost exchange that the old Dataset did not implement, and
rejects ordinary serialization of a header still marked as staggered. It retains
the existing VTK endpoint-coordinate convention; see `MIGRATION.md`.

`io/products.py` is new boundary code: Dataset interiors are copied into a
detached native Source; complete Fields are explicitly repacked into SFC order
for the retained serializer and atomically published as an ordinary data file.
Neither adapter carries old mutable ownership into a native Fields product.

The orchestration was adapted separately from these retained low-level files:

| New implementation | Source mechanisms at the pinned revision | Changed boundary |
| --- | --- | --- |
| `mesh.py` | `analysis/mesh.py`, `_prepare/regional.py` geometry assembly | Preserve coordinate arithmetic while retaining integer forest identity in shared geometry. |
| `io/amrvac.py`, `io/source.py` | `amrvac/analysis_io.py`, block reader descriptors | Sources own reading and field identity, with no attached halo factory, scheme or preparation workspace. |
| `preparation/exact.py` | `analysis/_prepare/regional.py` direct batch loop | Separate geometry binding, I/O and numerical execution; retain coarse-support arithmetic and batched SAME/FINER kernels. |
| `operators/derivatives.py`, `_kernels/native.pyx` | `analysis/derivatives.py`, compiled `differentiate` | Preserve floating-point expressions; pass an explicit integer input offset for storage/validity separation. |
| `operators/sampling.py`, `slices.py` | Sampling and plane worker ownership | Consume completed Fields only, keeping the accepted interpolation and pixel coordinates. |
| `tracing.py` | Detached branch of `analysis/field_lines.py` and compiled RK state machine | Remove pool dispatch from the direct API; collect results into admitted output without a full concatenation copy. |

N3 extends the native subset with scalar LOS and its ray-owner declaration,
and adds the thermal ray kernel and historical response table. The unused
cellwise derivative reference remains outside this extraction.

| N3 implementation | Source mechanisms at the pinned revision | Adaptation |
| --- | --- | --- |
| `tracing.py`, `bounded.py` | `analysis/field_lines.py`, `analysis/diagnostics.py` | Share concrete RK state between direct Fields and explicit pool recovery; preserve accepted-segment twist and retrace IDs. |
| `_pool.py` | `analysis/pool.py` | Coordinator-owned exact-phase workspaces and scoped leases; derived slots own storage independently. |
| `projection.py`, `_kernels/native.pyx` | `analysis/los.py`, scalar LOS tail in `utils/lib/analysis/native.pyx` | Keep scalar ray state and quadrature; direct Fields and bounded runners share image ordering without implicit dispatch. |
| `_kernels/thermal_rays.pyx`, `physics/_aia171_table.py` | `utils/lib/analysis/thermal_rays.pyx`, `analysis/_aia171_table.py` | Retain tree ray ownership, interpolation, response table and arithmetic; adapt imports and remove the historical four-worker API cap. |
| `physics/thermal.py` | `analysis/thermal.py` | Preserve explicit physical model and both reconstruction orders; bind storage offsets and valid windows once before per-leaf construction. |
| `preparation/plans.py` | `analysis/geometry_plans.py` action freezing, sizing and execution | Plans retain Mesh geometry only; new values and limiters are bound on every preparation; requested leaf order is preserved. |
| `io/cache.py` | `analysis/value_cache.py` | Preserve interior-value policy; expose an opt-in Source adapter and release readers/backing on close. |
| `io/jobs.py` | `analysis/global_fields.py` independent file-task mechanism | Each task opens its own new Source and uses bounded preparation; include task, result and IPC storage in admission. |
| `slices.py`, `bounded.py` | `analysis/slices.py` plane and slab traversal | Owned slab results and one concrete tile executor, with explicit bounded recovery. |

Curl compatibility uses immutable logical source/field/scheme provenance, not a
reference to the primary array. Plans do not retain sources or values. Cache
adapters borrow their parent source, and source validation applies to cache hits.
No imported asset requires the donor checkout after installation. Large worker
counts are not a numerical restriction; this host's validation remains capped
at four compute workers.

| Source path at the pinned revision | Destination under src/simesh | Treatment |
| --- | --- | --- |
| `src/simesh/_amr/__init__.py` | `_amr/__init__.py` | Namespace-only adaptation |
| `src/simesh/_amr/balance.py` | `_amr/balance.py` | Namespace-only adaptation |
| `src/simesh/_amr/blockio.py` | `_amr/blockio.py` | Namespace-only adaptation |
| `src/simesh/_amr/coarser_support.py` | `_amr/coarser_support.py` | Namespace-only adaptation |
| `src/simesh/_amr/coarser_workspace.py` | `_amr/coarser_workspace.py` | Namespace-only adaptation |
| `src/simesh/_amr/coarser_workspace_application.py` | `_amr/coarser_workspace_application.py` | Namespace-only adaptation |
| `src/simesh/_amr/contacts.py` | `_amr/contacts.py` | Namespace-only adaptation |
| `src/simesh/_amr/finer_boxes.py` | `_amr/finer_boxes.py` | Namespace-only adaptation |
| `src/simesh/_amr/forest.py` | `_amr/forest.py` | Namespace-only adaptation |
| `src/simesh/_amr/forest_conformance.py` | `_amr/forest_conformance.py` | Namespace-only adaptation |
| `src/simesh/_amr/foundation.py` | `_amr/foundation.py` | Namespace-only adaptation |
| `src/simesh/_amr/geometry.py` | `_amr/geometry.py` | Namespace-only adaptation |
| `src/simesh/_amr/halo.py` | `_amr/halo.py` | Namespace-only adaptation |
| `src/simesh/_amr/halos.py` | `_amr/halos.py` | Namespace-only adaptation |
| `src/simesh/_amr/morton.py` | `_amr/morton.py` | Namespace-only adaptation |
| `src/simesh/_amr/physical_widening.py` | `_amr/physical_widening.py` | Namespace-only adaptation |
| `src/simesh/_amr/prolongation.py` | `_amr/prolongation.py` | Namespace-only adaptation |
| `src/simesh/_amr/refined_geometry.py` | `_amr/refined_geometry.py` | Namespace-only adaptation |
| `src/simesh/_amr/relations.py` | `_amr/relations.py` | Namespace-only adaptation |
| `src/simesh/_amr/same_level_boxes.py` | `_amr/same_level_boxes.py` | Namespace-only adaptation |
| `src/simesh/_amr/storage.py` | `_amr/storage.py` | Namespace-only adaptation |
| `src/simesh/_amr/target_boxes.py` | `_amr/target_boxes.py` | Namespace-only adaptation |
| `src/simesh/_amr/topology.py` | `_amr/topology.py` | Namespace-only adaptation |
| `src/simesh/_amr/workspace.py` | `_amr/workspace.py` | Namespace-only adaptation |
| `src/simesh/amrvac/_v5/__init__.py` | `io/_v5/__init__.py` | Namespace-only adaptation |
| `src/simesh/amrvac/_v5/index.py` | `io/_v5/index.py` | Namespace-only adaptation |
| `src/simesh/amrvac/_v5/reader.py` | `io/_v5/reader.py` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/__init__.py` | `_kernels/primitives/__init__.py` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_balance.pyx` | `_kernels/primitives/_balance.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_boundary_rules.pxd` | `_kernels/primitives/_boundary_rules.pxd` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_boundary_rules.pyx` | `_kernels/primitives/_boundary_rules.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_coarser_support.pyx` | `_kernels/primitives/_coarser_support.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_coarser_workspace.pyx` | `_kernels/primitives/_coarser_workspace.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_contacts.pxd` | `_kernels/primitives/_contacts.pxd` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_contacts.pyx` | `_kernels/primitives/_contacts.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_finer_boxes.pyx` | `_kernels/primitives/_finer_boxes.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_forest.pyx` | `_kernels/primitives/_forest.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_forest_conformance.pyx` | `_kernels/primitives/_forest_conformance.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_foundation.pyx` | `_kernels/primitives/_foundation.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_geometry.pyx` | `_kernels/primitives/_geometry.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_halos.pyx` | `_kernels/primitives/_halos.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_morton.pyx` | `_kernels/primitives/_morton.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_physical_widening.pyx` | `_kernels/primitives/_physical_widening.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_prolongation.pyx` | `_kernels/primitives/_prolongation.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_refined_support.pyx` | `_kernels/primitives/_refined_support.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_relation_phases.pyx` | `_kernels/primitives/_relation_phases.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_relation_slots.pyx` | `_kernels/primitives/_relation_slots.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_relations.pyx` | `_kernels/primitives/_relations.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_restriction.pyx` | `_kernels/primitives/_restriction.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_same_level_boxes.pyx` | `_kernels/primitives/_same_level_boxes.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_storage.pyx` | `_kernels/primitives/_storage.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_target_boxes.pyx` | `_kernels/primitives/_target_boxes.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/_topology.pyx` | `_kernels/primitives/_topology.pyx` | Namespace-only adaptation |
| `src/simesh/utils/lib/primitives/limiter_core.pxd` | `_kernels/primitives/limiter_core.pxd` | Namespace-only adaptation |
| `src/simesh/utils/lib/analysis/native.pyx` | `_kernels/native.pyx` | Selected N1 kernels; derivative binding may be adapted |
| `src/simesh/utils/lib/analysis/native.pxd` | `_kernels/native.pxd` | Selected N1 kernels; derivative binding may be adapted |
| `src/simesh/utils/lib/analysis/preparation.pyx` | `_kernels/preparation.pyx` | Selected N1 kernels; derivative binding may be adapted |
| `src/simesh/amrvac/analysis_read.py` | `io/_bulk.py` | Reader namespace adaptation |
