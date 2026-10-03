# Architecture

Modules group functions or classes by numerical responsibility. Reusable services
supply geometry, data organization, numerical methods and physical models;
features combine those services into scientific operations. Workflows control
when input is read, prepared, consumed and released. These are distinct concerns:
a lifecycle diagram alone does not establish interchangeable numerical backends.

## Responsibilities and composition

| Responsibility | Current implementation | Boundary |
| --- | --- | --- |
| Physical query geometry | `spatial.py`: XYZ points, planes, rays, paths and length scales | No field storage or mesh topology; box-face convenience constructors consume explicit bounds |
| Mesh organization and cell geometry | `mesh.py`, `_amr`, `geometry.py`: native forest, leaf identity, location, sections and cell measures | No physical field values; currently Cartesian block AMR |
| Field storage and support | `fields.py`, `field_ops.py` | Explicit component definitions, leaf-to-slot mapping, valid halo and ownership |
| Numerical methods | `operators`, `_kernels`: reconstruction, derivatives, line/ray quadrature and RK stages | Consume the support required by the selected method; no implicit input reads |
| Physical definitions and field transforms | `physics`: composition, units, MHD recovery, emission and absorption | Explicit units, model and reconstruction order |
| Scientific features | Tracing, connectivity, projection, diagnostics and current proxy | Compose methods and physical quantities; preserve coverage and termination status |
| Application workflows | `applications.py`, `bounded.py`, examples and result I/O | Bind queries/results and orchestrate resources at explicit entry points |

Coordinate geometry and mesh organization meet at cell geometry: a mesh supplies
cells and adjacency, while a coordinate map determines their physical meaning.
Reconstruction and differentiation additionally need a numerical scheme and field
association. Neither a coordinate conversion nor a locator alone defines a
complete sampling or differentiation backend.

`geometry.py` currently provides native Cartesian sections, cell edges and
bottom-face quadrature. It is not a general coordinate-chart interface.
`spatial.py` owns the mesh-independent query objects; former public paths remain
aliases. `slices.py` combines queries with Fields. `_uniform.py` owns uniform
placement and `UniformResult`; slab assembly belongs to `slices.py`.
`applications.UniformResult` remains a public alias.

`_field_data.PointwiseLayout` binds completed native inputs for component
selection, field algebra, MHD recovery and thermal/radiation transforms. It
resolves original leaf IDs through each input's storage directory, aligns the
common valid halo and publishes owned outputs. Array formulas consume these
bound chunks without managing leaf order, directories or input resources.
Requested coverage must be aligned explicitly when combining fields; temperature
inputs may cover a larger native selection. The binding has no physical model or
numerical stencil and currently accepts native Fields only.

The thermal dependency direction illustrates composition:

```text
thermal LOS / application ray integration
       |                     |
thermodynamics         operators.rays / projection / compiled kernels
       |
 emission models <--- radiation coefficients
       |
 composition
```

`physics/emission.py` evaluates tabulated EUV responses on arrays.
`physics/thermodynamics.py` publishes thermal nodes and emissivity from completed
Fields; `physics/radiation.py` constructs absorption and radio coefficients.
Neither field transform depends on LOS integration. `physics/thermal.py` composes
thermal LOS and retains historical API aliases. The reference consumers share
`operators/rays.py`; compiled consumers share `_kernels/thermal_rays.pyx`.
MHD recovery uses `physics/composition.py` and `physics/units.py` independently of
thermal integration. Radiation post-processing does not modify the simulation EOS.

The field-line dependency direction is:

```text
applications / connectivity / current_proxy
          |                   |
      tracing           operators.line / line_profiles
          |
_kernels.streamlines / _kernels.connectivity
          |
       rk4.pxd
```

`rk4.pxd` owns the classical RK4 stage loop, with a compiled stage function and
caller-owned buffers of arbitrary state width. It imports no mesh operations.
Prepared-field adapters own interpolation, local-cell step admission and the
right-hand side; drivers own event admission and commit. QSL rescaling and
endpoint projection remain in connectivity. Current proxy combines stored
`LineSet` geometry, sampled curl and array-only line calculus; it owns closed-line
acceptance, the arc-length average and display-cell deposition. Source reads and
checkpoint selection remain in the caller.

`applications.trace` delegates compact path collection to `tracing`. One execution
context advances unfinished branches and copies accepted prefixes into contiguous
segments. Original state rows remain stable, delivery capacity respects the
remaining array budget, and only accepted private prefixes are exposed. Public
dense trajectories retain their NaN-tail contract. Inputs and state views bind
before dispatch; each worker keeps private RK scratch.

Variational QSL retains normalization-before-differentiation. Each worker reuses a
block of normalized nodes and immediately applies the shared centered-difference
kernel. The nine-component gradient stays resident; a second complete normalized
vector field is unnecessary. This is an allocation change, not a new Q method.

## Numerical operation bindings

Applications associate queries and results with IDs, units and provenance.
Scalar ray traversal belongs to `operators.rays`; ordered emission/absorption
belongs to `operators.transfer`; thermal traversal belongs to `physics.thermal`.
Those operations own their prepared-field bindings, support-dependent memory
admission and native/reference computation. Consumers do not read Sources or
prepare missing coverage. Shared ray batches reuse one executor over disjoint
ranges and finish every worker before reusing buffers.

`cartesian.pxd` owns half-open box containment without a tree or field dependency.
`interpolation.pxd` owns the established trilinear arithmetic without a storage
dependency. Native adapters supply location, stencil samples and local
coordinates. Additional layouts can provide their own concrete bindings while
reusing these arithmetic definitions; support and numerical meaning remain
explicit for every operation.

## Data lifecycle

```text
file / arrays -> Source + Mesh -> explicit read / prepare -> Fields
                                                            |
                        query geometry + model + method -> feature -> result I/O
```

| Boundary | Contract |
| --- | --- |
| Source / Fields | Owned published fields survive Source closure; consumers do not read missing input |
| Coverage / region | Selections retain complete original leaves; region edges are not physical boundaries |
| Physical halo rules | Source owns immutable per-field face parity; preparation binds it to selected components |
| Halo topology / coordinates | Periodic flags connect real leaves for exact-phase preparation; query coordinates never wrap |
| Storage / validity | Allocated padding is not valid halo; a first derivative consumes one valid layer |
| Owned / borrowed | Iterator and pool views expire with their lease; caller output retains alias constraints |
| Execution / buffers | Bind inputs once per range, allocate thread-private scratch and finish workers before buffer reuse |
| Input / output batching | Bounded output does not imply bounded input or topology memory |

`bounded.py` and `_pool.py` explicitly coordinate preparation and consumption;
they are orchestration, not field or numerical foundations. `_execution` owns
thread-pool lifetimes and disjoint ranges. Missing coverage and full path buffers
preserve integration state. Resident integrands in bounded tracing must cover
the entire Mesh; only primary/curl coverage is supplied by the pool.

`io/products.py` assembles complete root subtrees in the output's Morton order.
Regional Fields retain their original Mesh; crops build export indices instead
of a second analysis Mesh. `io/_v5/writer.py` shares bounded serialization between
file and Fields exports, while global Mesh/index storage stays resident.
Historical mutable Dataset implementations are excluded from the package.
`_kernels/primitives` retains its original compiler semantics.

## Extension boundaries

Current Mesh and compiled field adapters bind Cartesian block-AMR arrays.
Morton ordering belongs to forest construction and file layout; the hot locator
follows root/child directories. Storage rows can already differ from leaf order.
Fields still requires five-dimensional block storage, derivatives use Cartesian
stencils, and preparation uses balanced AMR relations. Uniform output is not an
independent uniform-grid backend.
Applications delegate numerical bindings to the relevant operations. These
bindings currently consume native AMR Fields; a new layout still needs a real
locator, support rules and reconstruction or traversal implementation. Independent
uniform input arrays and VTK reading are deferred. Existing AMR uniform output
and VTK export retain their supported workflows.

A second backend should provide the locator, reconstruction and valid support
needed by a concrete operation, bound before entering its compiled batch loop.
Sampling support does not automatically provide gradients, conservative measures
or cell traversal. Applications should compose these operations without selecting
Morton/tree details inside their scientific loops; backend adapters retain those
details. Do not impose one universal interface on all numerical methods.

Spherical or cylindrical grids additionally require coordinate maps, vector bases,
physical measures and consistent discrete operators. Query objects use physical
XYZ; angular spacing is not a physical step length. Mapping coordinates alone
cannot supply curvilinear derivatives or ray/cell intersections. Input formats,
coordinate systems and mesh organizations are separate choices, and only
implemented combinations may be advertised. Current interfaces do not accept
arbitrary backends or non-Cartesian grids.

The root `__init__.py` is an eager public facade; implementations import defining
modules. `make architecture-check` checks declared Python/Cython imports for
cycles and selected dependency boundaries. It cannot detect every structural
assumption or prove numerical substitutability. Inspect the operation and validate
representative numerical compositions as described in the [build guide](cython-build.md).
Exact API contracts stay in source docstrings and the [generated API](api.md).
