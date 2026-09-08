# Shared Spec: Regional Fields And Prepared Block Data

Status: shared semantic requirements. This page owns what consumers must be
able to know and rely on; it does not select classes, signatures, buffer shapes,
a cache hierarchy or an executor. [Intent](intent.md) sets scope and scale;
[application contracts](pipeline-results.md) set scientific results. Remaining
representation and numerical choices are in [baseline](baseline.md#open-decisions).

<a id="how-f-d-and-l-constrain-this-foundation"></a>

## Consumption Geometry And Access Patterns

The shared foundation is usable AMR block data with valid ghost neighborhoods.
Ordinary field-line tracing needs this prepared data for trilinear interpolation,
just as derivatives and many other consumers need it for their stencils. The
organizing distinction between applications is how their geometry and execution
sequence demand that data, rather than a partition into halo and non-halo users.

| Consumer geometry | Access sequence | Pressure on shared data organization |
| --- | --- | --- |
| F: many irregular field lines | Each line's RK stages generate subsequent positions from sampled values; paths can revisit, cross levels or diverge | Fast owner/local-coordinate lookup, inexpensive repeated neighborhood access, reuse across stages/seeds and private integration state |
| D and slices: selected regular regions | Traverse a full domain, box, plane or slab; target geometry is known before field computation | Enumerate selected leaves/windows, batch stencil work and supply boundary support efficiently; reuse geometry across requested quantities |
| L: full-domain observation rays | View and pixel position determine each integration interval; rays cross different cells, blocks and refinement levels | Consistent global geometry, physical depth/path-length mapping, shared field preparation and explicit ray/block-to-image contribution ownership |

A regular physical selection is not necessarily a contiguous memory interval
or a uniform-resolution AMR region. An oblique plane is geometrically simple
but can intersect many blocks. F's irregular output paths do not change the
block-structured input model. For L, full-domain unified treatment means complete
and consistent physical coverage; a resident global array is not implied.

These patterns constrain region/block grouping, geometry lookup, buffer access,
retained neighborhoods, batching and parallel scheduling together. Field sets
and operator reach are additional dependencies within each pattern. They do
not replace this geometric distinction. The exact F/D/L result rules live in
[pipeline-results.md](pipeline-results.md).

## Request And Physical Selection

The common request identifies a source lifecycle, physical region, defined field
set, required data state, requested products, support/accuracy and delivery or
retention intent. A product request may let its consumer infer dependencies;
users need not enumerate stencils. Exploration can instead request a particular
field state for inspection and later use.

A region can be the full domain, a spatial selection or a tree parent. A selected
level-one parent resolves to its descendant leaves even when it is not a leaf;
physical coverage is their union, without duplicate covered parents. A spatial
selection needs the promised cell/sample/intersection convention. Existing
cell-center ROI windows do not already specify a geometric slice or partial cell.

Select one, several or all fields in an explicitly defined set. "All" does not
materialize every possible derived definition. Distinguish the selected result
region from support-only blocks and values outside it. Ghost padding never
expands the physical domain. Access beyond supplied coverage needs a declared
additional-preparation or failure/termination rule.

## Physical Field Definitions

Resolve centering, cell-average versus point-value meaning, units, normalization,
active dimensions and equation-of-state dependencies at the analysis boundary.
Names alone are insufficient. A 2D mesh can still carry three vector components.
Missing physical inputs are not silently replaced with zero.

| Quantity | Dependencies to resolve from the file model |
| --- | --- |
| Stored scalar/interior value | Selected scalar and needed geometry/units; no implicit halo work |
| Velocity component or speed | Stored velocity, or momentum and density; all active vector components for speed |
| Pressure, temperature, sound speed, Mach number or plasma beta | Actual EOS and normalization; recovering pressure from total energy may need density, momenta, B and model parameters |
| Magnetic geometry/strength | B and its centering/normalization; no implied thermodynamic fields |
| Curl, divergence, vorticity, compression or a gradient | Required vector/scalar components and stencil support; density when differentiating recovered velocity |
| Mass/energy distributions or magnetic flux | Integrand/weights and physical cell volumes or oriented surface geometry |

Do not apply a conserved-to-primitive conversion to an already primitive field
or another energy model. `curl(B)` is not automatically physical current; its
normalization belongs to the requested diagnostic.

## Data Products And Consumer Paths

| State | Usable content | Possible consumers |
| --- | --- | --- |
| Interior regional fields | Physical values on selected native leaves with declared interior coverage | Direct cell access, pointwise operations, reductions with declared interior-only reach, preparation or delivery |
| Ghost-prepared input fields | Target-scale values with the support specified below | Sampling, stencils, tracing, direct AMR operators or delivery |
| Derived regional fields | Named computed quantities with their own physical definition and remaining valid coverage | Further operators, samples/slices, integrands, reduction or delivery |

States and remaining validity can differ by field in one logical result. An
allocated buffer or a single global "ghosts valid" flag cannot express this.
Native data can be returned as a collection or stream of valid block views with
the geometry needed for promised location/traversal operations. It need not be
one all-resident array. Trilinear sampling is one consumer; it is not a mandatory
interface for every native AMR operation. Direct cell access also does not
choose a nearest-neighbor or other low-order reconstruction strategy.

## What A Prepared Block Must Describe

The following information must be available, possibly through shared immutable
metadata. These are semantic responsibilities, not mandatory struct members or
cache keys. Logical leaf identity must remain distinct from a temporary slot.

| Information | Consumer-visible meaning |
| --- | --- |
| Source and fields | Source lifecycle, target leaf ID, logical field/component order, physical definitions and numerical/boundary strategy |
| Geometry and addressing | Active dimensions, level, interior extents, origin/centering and spacing; mapping from physical coordinates to local indices |
| Storage | Backing buffer(s), shape/strides/component layout, interior/halo offsets and mapping from leaf/field to storage or slot |
| Validity | Usable fields and spatial regions, including remaining derived halo; completion must be established before publication |
| Ownership and lifetime | Owner of backing and shared metadata, borrowing scope, overwrite/release rules and invalidating changes |

A kernel may require contiguous input even if a strided view exists. Gathers,
packing, conversion and duplicate backing are explicit costs. Precomputed
origins, reciprocals or interpolation weights cannot silently change arithmetic
promised by an existing strategy. Field additions need not dictate a global
column rebuild: that is a storage choice to assess, not a semantic requirement.

## Numerical Meaning Before Storage Choice

Prepare neighbor data on the target's local grid/reconstruction convention.
Fine and coarse values cannot be joined as uniformly spaced samples. Centering,
field interpretation and boundary rules determine the transfer meaning.

| Relation | Preparation requirement |
| --- | --- |
| Same level | Map the correct source values, including required faces, edges and corners |
| Finer source to coarser target | Restrict with declared physical coverage and weighting; one fine value is not a coarse average |
| Coarser source to finer target | Prolong/reconstruct at target scale, including any additional slope support beyond direct neighbors |
| Physical boundary | Apply the declared field/side rule, including its valid base dependencies |

Preserve the distinction between direct stencil directions and the closure
needed to construct those values. All-26 3D execution is an existing reference;
requested-direction preparation must still close slope and physical-base
support. Account for active dimensions when changing representations.

### Input Reach And Remaining Validity

**Ghost-enabled prepared primary/input fields require at least two valid layers
on each active side.** This means completed usable values, not empty padding.
It applies to input preparation backing the analysis chain. Interior-only
consumers retain a direct path with no halo preparation.

Propagate the consumer's requested valid region backward through each operator's
per-input stencil, then propagate actual validity forward. For a first derivative
consuming one layer per required side, two valid input layers leave one valid
derived layer. That layer supplies the surrounding values for trilinear sampling
on the owner's grid, including edge/corner neighborhoods. The confirmed operation
is **trilinear interpolation**, not a third-order-accurate method.

Wider, repeated or asymmetric derivatives can need more support. Calculate reach
per input and active side; two is a minimum, not a maximum. A derivative view with
one remaining layer is a valid output, but differentiating it again needs a new
reach calculation. Neither raw-source reach nor temporary coarse-support storage
is limited to the final sampler's one layer. Reject unsupported reach explicitly.

Differentiating extended inputs and exchanging derived interiors need not agree
at AMR interfaces. Normalizing B before differentiation also defines a different
discrete operation from simply differentiating B. The selected strategy must
state these orders. For nonlinear LOS response, evaluating a response on interpolated
inputs need not equal interpolating precomputed response values. Consumer meaning
determines support, not the other way around.

Existing [SAM-005](../../../../rewrite/contracts/SAM-005.md),
[RST-001](../../../../rewrite/contracts/RST-001.md),
[PRL-001](../../../../rewrite/contracts/PRL-001.md) and
[canonical mesh notes](../../../amr-forest-mesh.md) supply scoped examples. In particular,
one-layer CHS remains unchanged evidence and needs adaptation to meet this
preparation baseline. These references do not select a universal transfer rule.

## Preparation And Consumption Requirements

| Responsibility | Required behavior |
| --- | --- |
| Request/dependencies | Resolve selected physical leaves, fields, result state, numerical meaning and required support |
| Geometry/support planning | Identify target owners and close real source dependencies; plans name the assumptions under which they can be reused |
| Storage access | Read or reuse needed values under adapter selection/copy/failure rules; the reader does not choose scientific dependencies |
| Value preparation | Complete target-scale neighborhoods and declare per-field validity; consumers never see incomplete or stale support |
| Consumption | Use established geometry and buffers without repeating file access or support discovery for a compatible ready neighborhood |
| Scheduling/resources | Choose admissible batches, residency and reuse; preserve live inputs and requested physical coverage through consumption/delivery |

These boundaries can be fused in a concrete implementation. Locate owners and
obtain local stencil indices/weights efficiently; hints, grouping and geometry
tables are candidates. F has data-dependent points, D can traverse known block
cells and L follows observation geometry. They can share field views while
keeping distinct control flow. Application-specific obligations are in
[pipeline-results.md](pipeline-results.md#shared-foundations-and-separate-execution).
No per-cell object or callback protocol is required. Group operators only when
their input fields, required validity and boundary semantics are compatible;
keep their definitions independent. A second concrete operator should establish
useful shared work before introducing generic recipes.

Preparation can be local, bulk resident or a full-domain chunk pass. A local
request must not silently complete the domain; dense work must not be forced
through repeated owner preparation. Fusion/elision must preserve every requested
product, so global D coverage cannot be replaced by slice-only computation.

Establish immutable source/layout/topology facts at their owning boundary and
reuse valid proofs where possible. New selectors, coordinates, slot bindings,
source lifetime and actual bounds/validity still need their appropriate checks.
Current HPR proofs are invocation-local, not persistent validation. Existing
all-request preflight, checked wrappers, accepted-prefix and I/O-failure rules
remain binding for their entrypoints; changed failure behavior needs a new
explicit contract. Measure planning/check cost separately from value work.

## User Model: Region Fields And Two Lifetimes

| Usage | Lifetime promise |
| --- | --- |
| Task-scoped end-to-end | Source to one or several requested products; temporary preparation may be shared until its last consumer, then released |
| Retained exploration | Keep explicitly requested compatible fields/preparation usable across successive requests, with an explicit end to retention |

These styles share numerical meanings and may use resident, bounded or suitable
backed storage. Neither requires a Dataset class or a separate executor. A
returned field product outlives task scratch according to its own contract;
retained exploration does not imply full RAM residency, disk persistence,
source mutation or transparent recomputation.

## Explicit Retention Versus Internal Caching

| State | Ownership and release rule |
| --- | --- |
| Source and snapshot interpretation | A borrower does not close the source; an owner keeps it and shared metadata alive for dependent work |
| Task scratch/active chunk or stage | Reuse or free only after its last declared consumer; active work cannot lose input to eviction or overwrite |
| Opportunistic cache or retained plan | Reuse only compatible identity, meaning and coverage; idle entries can be evicted and later misses prepared again |
| Explicitly retained field/line/image product | Keep promised backing and validity until its declared release/change; cache eviction cannot silently revoke it |
| Detached owned result | Remains usable after task/session closure; caller controls its lifetime |
| Borrowed result/view | Declare the backing and exact valid borrowing scope, including how overwrite is prevented |

Current RHC views expire on synchronous callback return; they are not retained
or asynchronous outputs. CHS clear invalidates keys but retains arrays.
Invalidation, allocation and storage release are different operations; releasing
storage does not promise immediate RSS reduction.

If a retention promise cannot fit, report it before promising the product or
use an explicitly chosen backing/delivery arrangement. A lazy reference cannot
pretend to be materialized. Automatic spill, replay, recomputation, cancellation
or asynchronous borrowing is not implied. Results specify order/selection,
written and untouched regions, and complete/partial/failure meaning; variable
output does not imply a mandatory full precount scan.

### Reuse And Invalidation

This table identifies dependencies, not a requirement to implement every cache.

| Change | Potentially reusable | Recompute or revalidate |
| --- | --- | --- |
| Repeat compatible query | Source, plans, fields and retained results | Missing coverage or changed output requirements |
| Move a slice | Topology and overlapping field/support values | Intersections, sample positions and uncovered values |
| Add seeds | Geometry and compatible neighborhoods | New trajectories; nearby seeds may diverge |
| Recolor fixed geometry | Curve/surface and compatible location map | New attribute dependencies; a color-map-only change needs no core work |
| Change threshold/isovalue | Unchanged defining values and valid conservative summaries | Masks, geometry, connectivity and selected quantities |
| Change stencil, interpolation, normalization or boundary rule | Unaffected interiors/geometry | Dependent support, derived fields and products |
| Change source/values | Explicitly proven compatible geometry; scratch capacity | Offsets as needed; changed values, halos, derivatives and value summaries |
| Change storage slots/layout | Logical geometry where valid | Bindings and layout-dependent plans/views |
| Change budget/policy | Compatible affordable state | Admission/eviction and any required conversion/rebuild |

Unchanged array identity, leaf count or topology does not prove unchanged values.
Mostly immutable analysis forbids unnoticed source mutation; existing editable
Dataset workflows retain their separate lifecycle. A new edit boundary must
invalidate dependents or start a new lifecycle. No payload hashing is required.

## Resources And Parallel Ownership

Apply [complete workflow accounting](performance.md#complete-workflow-resources)
to metadata, inputs, retained state, scratch, transfer and live outputs, including
caller/borrowed backing. Reserve known minimum active needs before optional
retention. Admit a full output explicitly; bounded computation must support
bounded delivery when requested, without forced reassembly. Memory pressure
cannot silently change the scientific request or numerical strategy.

Independent seeds, regional tasks or ray/image tasks may share immutable completed
neighborhoods but need private mutable stage/workspace state and disjoint output
regions or an explicit merge rule. Complete preparation before read publication; define preparation
misses, reuse and overwrite ownership under parallel execution. Threads do not
each receive the full job budget. Current CHS is serial/non-reentrant and cannot
become a shared parallel cache merely by threading its callers. Deterministic
partial-sum merging does not automatically preserve serial floating-point bits.

## Representation Candidates To Compare

The [concrete data-organization proposal](data-organization.md) recommends native
contiguous patches, with complete leaves and explicit rectangular windows,
independent field groups and compact support staging. It supplies a design choice
and rationale; the shared semantic requirements on this page remain its constraints.
The alternatives below retain their comparison role.

Compare the whole path: geometry/neighbor indexing, field and halo storage,
transfer/batching, local addressing and the actual consumer's buffer needs.
Ownership/validity are constraints on that comparison, not a framework to build.

| Candidate | Possible benefit | Cost or unresolved issue |
| --- | --- | --- |
| Complete padded target blocks, resident or in a bounded pool | Simple local stencils and repeated sampling | Padding, support-only waste, copies and retained memory |
| Raw interiors with separate halo/support storage | Avoid padding every support-only block/field | Gathers, mapping and temporary contiguous views |
| Prepared tiles/regions with explicit coverage | Match thin regions and operator reach | Plan/mapping cost, stitching, duplicate support and view lifetime |

Field-major, cell-major and component-grouped storage are further choices;
vector sampling and fieldwise stencils may differ. Avoid a transpose at every
consumption merely to impose a universal layout. Execution residency is also
independent of access density: a sparse output can need a dense scan, and a
large input can have a fitting working set. See
[techniques](technique-candidates.md#candidate-technique-records) for existing
mechanisms and [usage sequences](lifetime-sketches.md) for reuse trade-offs.

## Evidence And Next Spec Decision

Assess first preparation/result, warm sample/operator cost, location/index work,
support/read amplification, copies/packing and complete live memory together.
Historical [evidence](baseline.md#decision-relevant-evidence) explains why the
joint boundary matters; one-layer controls do not validate the new reach.
Scientific acceptance and comparable timing belong to [performance](performance.md).

The [data-organization proposal](data-organization.md) supplies initial structural
choices for S1--S5. Next specify its concrete preparation/view functions and the
applicable numerical, validity, lifetime and resource guarantees against F/D/L
needs. The remaining [decisions](baseline.md#open-decisions) retain their scope;
a full Q solver, full radiation model or separate lifecycle research stage is
not a prerequisite.
