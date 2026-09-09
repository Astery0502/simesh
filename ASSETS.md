# Retained implementation assets

The selected implementation now lives at the repository root. Historical
source paths in the provenance tables below identify the pinned revision, not
runtime imports. Array configurations moved to `tools/configurations.py` during
root promotion. Mutable Dataset and its `amrvac/_mesh` extensions have since
been removed from the active build; the preceding implementation remains under
`legacy/previous/`. There is no active `simesh.amrvac` or `simesh.utils` namespace.

The subsequent `connectivity.py` and `_kernels/connectivity.pyx` consumer is an
independent implementation of published magnetic-connectivity mathematics using
this package's existing AMR primitives. No FastQSL2 code is bundled or translated.
Definitions follow [Chen et al. (2026), FastQSL 2](https://arxiv.org/abs/2604.16195),
including Scott et al. (2017), Pariat and Démoulin (2012), Titov (2007), and Berger
and Prior (2006). See also [Zhang et al. (2022), FastQSL](https://arxiv.org/abs/2208.12569).
The [FastQSL2 repository](https://github.com/el2718/FastQSL2) declares CC BY-NC-SA 4.0;
it remains an external comparison reference, not a runtime dependency. Comparison
used revision `314bbf01ab72e43f82cb6b2e1c2a4d22d93aacdd`.

Source revision: `b91bbc015d882ffd7dfcd7e10590cdce559f8532`.

The native interior-only uniform exporter follows the block placement and
containing-cell sampling algorithms from the former `amrvac/_mesh/mesh.pyx`
(`uniform_full_level1` and `uniform_grid_zero_order`). `_uniform.py` and
`_kernels/native.pyx` use immutable geometry and component-last storage, with
half-open ownership, explicit missing coverage and bounded source orchestration
in `io/uniform.py`; they do not instantiate the mutable AMRMesh.

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

The N4 extraction included stateful AMRVAC interfaces from the same pinned
revision. Those interfaces and their mutable mesh are now excluded from the
active package; `legacy/previous/` retains the preceding implementation. The
independent array tools remain under `tools/`.

`io/_v5/writer.py` adapts the ordinary-field SFC serializer from
`amrvac/datio.py`, retaining the binary header, tree and block layout. It accepts
only the current nonperiodic Cartesian 3D v5 profile, calculates offsets from
the encoded header and writes NumPy buffers without per-value Python packing.
`io/products.py` repacks complete Fields interiors into SFC order and owns file
creation and atomic publication. The serializer excludes old readers, Dataset
constructors and uniform/VTK wrappers; it retains its GPL-3.0 provenance and
rejection of staggered-face serialization.

`io/vtk.py` independently implements uniform-volume output using the
[VTK legacy file specification](https://docs.vtk.org/en/latest/vtk_file_formats/vtk_legacy_file_format.html).
It writes binary structured-points geometry and cell data at the native sampling
bounds. It does not restore the former Dataset or endpoint-based VTK writer.

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
