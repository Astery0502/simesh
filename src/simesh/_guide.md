# Choosing a simesh workflow

Use the decision points relevant to the user's question. These are common
choices to clarify, not a prescribed application or an exhaustive catalog.
Consult the installed version's signatures and docstrings for exact controls.
The scientific objective, physical model, and acceptable approximations remain
with the user.

1. **Start from the requested result.** Does the user need cell quantities,
   a section, sampled points, trajectories, diagnostic values, a projection,
   or a volume? Explore `simesh.reductions`, `simesh.slices`, and
   `simesh.applications`. Native sampling can use interpolation; a uniform
   volume is a separate output choice, not a prerequisite for all analysis.

2. **Trace the input dependencies.** Work backward from the consumer to the
   required fields and derived quantities. Select those inputs rather than
   reading every field by default. Inspect `simesh.open_amrvac`,
   `simesh.read_fields`, and the consumer's requirements. Confirm the physical
   scales and field meanings needed by the chosen model; names alone do not
   establish units or the meaning of an energy variable.

3. **Choose the preparation point.** Does the consumer use cell interiors or
   require neighboring support? Inspect `simesh.read_fields`, `simesh.prepare`,
   and `simesh.select_region`. Choose preparation and coverage for that
   consumer. Allocated padding is not valid halo, and a selected region's edge
   is not a physical boundary of the original simulation.
   Specify physical boundary parity through Source `boundary` when opening
   arrays or a snapshot; vector component parity is never inferred.
   Periodic inputs require `exact-phase` for halo preparation. Coordinates and
   regions do not wrap; periodic trajectories/connectivity are not supported.
   AMRVAC/uniform exports and saved AMR slices reject periodic meshes.

4. **Order and reuse the work.** Share preparation, derived fields, and geometry
   where appropriate. Explore `simesh.derive_many`, `simesh.sample_line_profiles`,
   and `simesh.integrate_los_views`. Derivation, interpolation, and integration
   do not generally commute; preserve the intended mathematical order.

5. **Choose storage and delivery.** Are trajectories themselves needed, or only
   values calculated along them? Should the output be resident or delivered in
   batches? Inspect the consumer's iterator and ownership contracts, plus
   `simesh.save_result` and `simesh.save_result_shards`. Batched output does not
   imply bounded input-field memory; consult `simesh.bounded` when that matters.
   Choose resource limits for the task and available environment, not a fixed
   budget supplied by this guide.

6. **Check what the result actually covers.** Inspect the result's validity,
   coverage, and termination information before plotting or reducing it.
   A returned path may be an accepted prefix rather than a complete field line.
   Report incomplete results and the controls used; successful execution alone
   does not establish numerical convergence or a physical interpretation.

Discover other paths through `help(simesh)`, `help(simesh.applications)`, and
individual interface docstrings. The sketches below are examples of these
choices, not requirements to perform additional calculations.

## Decision sketches

- **Locate QSL/twist structures on a native AMR section.** Can native cell
  centers define the seed distribution? Explore `simesh.AxisSlice.cell_edges`
  and `simesh.PointSet`, then `simesh.applications.surface_diagnostics`.
  Retain the seed-to-cell association for a nonuniform plot. The section
  defines seeds; diagnostics consume the prepared 3D field and return values
  and status information without retaining full trajectories. If the user also
  wants field lines, inspect `simesh.applications.ConnectivityMap.threshold`
  and trace the selected seeds with `simesh.applications.trace`.

- **Trace paths or calculate along-line quantities.** When the user requests
  magnetic field-line geometry, `simesh.applications.trace` returns a
  `simesh.LineSet` for plotting and reuse. When only a scalar integral is
  required, inspect the `integrands` and `trajectories` controls of
  `simesh.trace`: it returns a `simesh.TraceResult` and can integrate without
  retaining the path. Retain trajectories when they are part of the requested
  output. For existing paths, explore `simesh.sample_line_profiles`,
  `simesh.line_integral`, and `simesh.line_derivative` before retracing.
  Integrating sampled profiles and integrating at RK stages are distinct
  numerical choices; select the one appropriate to the calculation.

- **Compare derived-field sections in a region.** Which inputs, preparation,
  and geometry can be shared? Distinguish native cell sections
  (`simesh.slice_axis`) from interpolated plane samples (`simesh.sample_plane`).
  Explore `simesh.derive_many` and the relevant derivative operators before
  sampling. A regularly sampled plane does not require a uniform 3D copy.

- **Deliver selected data or analysis products.** Choose output fields
  independently of the inputs needed to compute them. A section can go through
  `simesh.save_result` without exporting a volume. For direct file-to-uniform
  delivery, explore `simesh.export_uniform` and `simesh.export_uniform_vtk`.
  Existing fields can feed `simesh.applications.uniform_grid`; an existing
  uniform result can go to `simesh.write_uniform_vtk` or `simesh.save_result`.
  For native AMR delivery, inspect `simesh.crop_amrvac`, or
  `simesh.select_fields` and `simesh.write_amrvac` for existing fields, including
  their metadata and complete-root-subtree requirements. Reuse computed
  products rather than rereading and repeating analysis only to export them.
