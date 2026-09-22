# Choosing a simesh workflow

Use these decision points to organize a calculation. The interfaces below are
starting points, not an exhaustive workflow catalog. Inspect the installed
version's public interfaces and docstrings for other suitable paths.

1. **Start from the requested output.** Decide whether the task needs cell
   quantities, a section, sampled points, paths, a projection, or a volume.
   Explore `simesh.reductions`, `simesh.slices`, and `simesh.applications`.
   Native AMR sampling can itself use interpolation; a uniform volume is a
   separate output choice, not a required intermediate for every analysis.

2. **Trace the input dependencies.** Work backward from the consumer to the
   required fields and derived quantities. Select those inputs rather than
   reading every field by default. Start with `open_amrvac`, `read_fields`,
   and the chosen consumer's field requirements.

3. **Choose the preparation point.** Determine whether the consumer uses cell
   interiors or needs neighboring support. Inspect `read_fields`, `prepare`,
   and `select_region`. Choose the scheme and coverage for that consumer;
   regional selection is not a new physical boundary.
   Specify physical boundary parity through Source `boundary` when opening
   arrays or a snapshot; vector component parity is never inferred.
   Periodic inputs require `exact-phase` for halo preparation. Coordinates and
   regions do not wrap; periodic trajectories/connectivity are not supported.
   AMRVAC/uniform exports and saved AMR slices reject periodic meshes.

4. **Order and reuse the work.** Identify shared preparation, derived fields,
   and geometry before repeating operations. Explore `derive_many`,
   `sample_line_profiles`, and `integrate_los_views`. Derivation, interpolation,
   and integration do not generally commute; retain the intended mathematical
   order when arranging reuse.

5. **Choose how to deliver results.** Decide whether subsequent work consumes
   one result or successive pieces. Inspect the relevant consumer's iterator
   interfaces and `save_result` / `save_result_shards`; their input and lifetime
   contracts determine how they can be composed.

Follow only the decision points relevant to the request. Discover additional
branches through `help(simesh)`, `help(simesh.applications)`, and individual
interface docstrings. This guide does not choose resource budgets or assess
result reliability; those decisions remain with the user.

## Decision sketches

These sketches suggest questions and entry points, not fixed call sequences.

- **Locate QSL/twist structures on a native AMR section, then inspect field
  lines.** Can the section's native cell spacing define the seed distribution?
  Explore `AxisSlice.cell_edges` to construct cell-center positions on the
  section and pass them as a `PointSet` to `applications.surface_diagnostics`.
  Retain the seed-to-cell association for plotting the nonuniform result.
  The section defines seeds; diagnostics still consume the prepared 3D magnetic
  field. Inspect `ConnectivityMap.threshold` to select seeds, then
  `applications.trace` to retain only the requested paths. Could those paths
  also serve several physical quantities through `sample_line_profiles`?

- **Compare several derived-field sections in one region.** Which input fields,
  preparation, and geometry can be shared? Distinguish native cell sections
  (`slice_axis`) from interpolated plane samples (`sample_plane`), and inspect
  `derive_many` and the relevant derivative operators to arrange shared work
  before sampling. A regularly sampled plane does not require a uniform 3D
  copy of the simulation.

- **Deliver selected data or analysis products.** Which original or derived
  fields are needed, and is the deliverable a section, an AMR region, or a
  uniform volume? Choose output fields independently of the inputs needed to
  compute them. For a section, inspect `slice_axis` or `sample_plane` component
  selection and `save_result`; a section does not require exporting a volume.
  For direct file-to-uniform delivery, explore `export_uniform` and
  `export_uniform_vtk`, selecting fields, bounds, and reconstruction. After
  analysis, could existing Fields feed `applications.uniform_grid`, or an
  existing uniform result go straight to `write_uniform_vtk` or `save_result`?
  If the recipient needs native AMR instead, inspect `crop_amrvac` for direct
  file cropping, or `select_fields` and `write_amrvac` for existing Fields,
  including their metadata and complete-root-subtree requirements. Preserve
  already computed products where suitable rather than rereading and repeating
  the analysis just to export them.
