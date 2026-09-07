# Algorithmic Exploration Routes

Status: preserved research directions, 2026-09-07. The user requested keeping
these possibilities available to the next development cycle. Inclusion records
interest and design relevance, not adoption, implementation or simesh performance.
[Development](development.md) owns when a route is considered and promoted;
[current](current.md) records any active selection. T01--T18 in
[technique candidates](technique-candidates.md) retain their existing scope.

The input remains native block-structured AMR. The routes below exploit its
regular regions and irregular interfaces. They change different things and
should not be bundled into one mandatory architecture or prototype campaign.
The source observations were collected in the preceding literature exploration;
the applications and entry conditions below are proposed simesh judgments.

## Route Map

| Route | Change being considered | Numerical relationship to the baseline | First relevant decision |
| --- | --- | --- | --- |
| E1: rebricking | Analysis compute regions can span several original same-level leaves | Intends to preserve cells and the selected discrete operations; coordinate/arithmetic conformance still needs proof | P0, before compute identity and ownership are frozen |
| E2: basis reconstruction and ABRs | Store/query a reconstructed field and its support regions | A distinct reconstruction, especially at refinement interfaces; analytic derivatives have a distinct meaning | A declared reconstruction strategy for F/D/L; never a silent baseline replacement |
| E3: dual mesh and gridlets | Regular gridlets plus explicit interface elements | A distinct interpolation/topology representation | Selected geometry or ray consumer with an interface-reconstruction requirement |
| E4: RT-core point location | Hardware BVH traversal for point/region lookup | Lookup acceleration can be separated from field reconstruction, but the cited implementation combines them | A justified device stage with suitable hardware and complete resource accounting |
| E5: retained fill plans and fused execution | Cache geometric copy/transfer facts and execute batches | Intends to preserve the declared numerical/failure contract | P0/P1 preparation design; closely related to T09/T16 |

P labels refer to [delivery stages](development.md#delivery-stages). A route's
placement is an opportunity to make a decision, not a demand to implement it.

## E1: Rebrick Same-Level Cells For Analysis

**Source observation.** ExaBricks reorganizes non-overlapping cells into compact
rectangular bricks of one refinement level, independently of original AMR block
boundaries. Its construction groups cells that fill their bounding box and limits
brick size for traversal granularity. [ExaBricks paper](https://www.sci.utah.edu/~wald/Publications/2020/exabrick/exabrick.pdf).

**Proposed use here.** Merge eligible same-level regions so former internal block
faces become ordinary interior accesses. Keep a mapping to original cells and
file records; do not resample, average away fine cells or change physical coverage.
This could reduce both retained halo values and repeated preparation across the
original 8^3 blocks.

For two halo layers, the following is shape arithmetic, not a measured saving:

| Interior cube | Padded cube | Padded/interior value count |
| --- | --- | ---: |
| 8^3 | 12^3 | 3.375 |
| 16^3 | 20^3 | 1.953125 |
| 32^3 | 36^3 | 1.423828125 |

The favorable case needs sufficiently large, completely filled same-level
regions. Fragmentation and refinement boundaries limit merging. Preprocessing,
mapping memory, partial reads and sparse-query overfetch can offset the saving.
The table says nothing about how much of WENO can actually be merged.

**Decision and acceptance.** Consider E1 during P0 because a permanent equation
of source leaf ID with compute-patch ID would make later adoption expensive.
Separate those meanings even if the initial chosen organization retains original
leaves. Establish exact source-cell/coverage mapping, unchanged field values,
boundary/stencil meaning and a complete preparation/consumption cost model.
Mapped numerical evaluation must conform to the selected arithmetic contract;
preserving raw values alone does not prove bitwise sampling equivalence.
Adopt only where the concrete consumer benefits; preserve a direct original-block
comparison and record reasons to defer unsuitable cases.

## E2: Reconstructed Fields And Active Brick Regions

**Source observation.** The basis method combines true cell values using
normalized local tent weights. In regular regions it recovers trilinear behavior;
at mixed-level interfaces it is continuous but not necessarily interpolating
through every original value. [CPU AMR reconstruction](https://www.sci.utah.edu/publications/Wal2017a/cvamr.pdf).

ExaBricks partitions overlapping supports into Active Brick Regions (ABRs), each
listing contributing bricks. Known brick grids then provide the required cells.
It also derives analytic gradients from the values used for reconstruction,
without the additional samples needed by central differences. The reconstructed
field is not continuously differentiable. [ExaBricks reconstruction and gradients](https://www.sci.utah.edu/~wald/Publications/2020/exabrick/exabrick.pdf).

**Proposed use here.** A shared field evaluator could serve B sampling, along-line
diagnostic derivatives and LOS response queries. Required neighboring data can
participate directly in reconstruction rather than always becoming materialized
ghost arrays. This still requires complete support and an explicit boundary rule.

**Decision and acceptance.** Distinguish two possible investigations: adopting
only a support-region index, or adopting the cited reconstruction too. An index
must describe the support of our actual operator; the paper's tent-function ABRs
do not automatically cover a different derivative/halo rule. Changing the
reconstructed field requires a named numerical strategy and application-specific
accuracy, conservation/divergence requirements where applicable, and interface
error analysis. Analytic curl of the reconstruction is not automatically the
existing finite-difference curl followed by interpolation.

Keep the [confirmed prepared-input validity](prepared-fields.md#input-reach-and-remaining-validity)
for the baseline. E2 cannot be used to relabel incomplete two-layer data as valid
or silently change D's result. Treat reconstruction choice as a scientific
decision when the requested meaning has not already been delegated.

## E3: Dual Mesh With Gridlets And Interface Elements

**Source observation.** A 2023 method splits the AMR dual mesh into regular voxels,
clustered as gridlets, and unstructured stitching elements at level interfaces.
A BVH covers both. Its comparison reports memory savings against other
unstructured dual-mesh representations, not against the original AMR payload.
[Paper and authors](https://wilsoncernwq.github.io/publications/eurovis2023-stitcher),
[owlExaStitcher source](https://github.com/owl-project/owlExaStitcher).

**Proposed use here.** Concentrate interface-specific reconstruction in explicit
elements while retaining cheap regular-grid access elsewhere. This is relevant
to native slices, isosurfaces and ray sampling, and is an alternative to E2's
basis-function representation rather than a prerequisite for it.

**Decision and acceptance.** Define the requested interpolant and geometry first.
Account for construction, connectivity, gridlet data and simultaneous original/
converted storage. Check interface coverage, cracks/duplicates, cell-to-vertex
meaning, scalar/vector precision and derivative interpretation. The published
renderer is not a validated current-density or Q implementation. Require the
selected user's geometry/integral result before promotion; do not construct a
global unstructured mesh merely because the method is available.

## E4: Ray-Tracing Hardware For AMR Point Location

**Source observation.** A 2022 study maps AMR point containment to RT-core BVH
traversal and combines it with ExaBricks reconstruction and RK particle tracing.
It includes magnetic-field exploration of astrophysical AMR data. The paper also
notes possible undersampling when a path enters a finer region; visualization
success is not an endpoint-error guarantee. [Paper](https://arxiv.org/html/2202.12020v1).

**Proposed use here.** Share a region locator between irregular RK sample positions
and view-defined rays. Consider hardware location separately from the numerical
sampler so a lookup change need not imply a new reconstruction strategy.

**Decision and acceptance.** Enter when a concrete workload is limited by location
and device resources are available within the agreed scope. The cited implementation
requires NVIDIA RTX/OptiX; its hardware benefit is not portable to an arbitrary
CPU/GPU. Include BVH construction, transfers, device/host residency and outputs.
Preserve exact ownership and the selected precision/arithmetic requirements.
Its trajectory allocation and stepping choices are not our million-seed output
or diagnostic contract. CPU/OpenMP delivery remains independently viable.

## E5: Cache Fill Metadata And Fuse Small Transfers

**Source observation.** AMReX retains FillBoundary communication/copy metadata;
its implementation has cached local/send/receive tags and conditional CUDA graph
storage. Its GPU work uses kernel fusion to reduce the launch cost of packing or
unpacking many small regions. [Metadata implementation](https://amrex-codes.github.io/amrex/doxygen/AMReX__FabArrayBase_8H_source.html),
[AMReX/pyAMReX paper](https://journals.sagepub.com/doi/10.1177/10943420241271017).
Across refinement levels, FillPatch uses explicit interpolation and boundary
rules, beyond ordinary same-level copying. [FillPatch documentation](https://amrex-codes.github.io/amrex/docs_html/AmrCore.html).

**Proposed use here.** Reuse invariant geometry/transfer templates for the mostly
immutable snapshot, bind actual buffers once per batch, and run compiled work.
This directly targets the existing WENO preparation costs and strengthens the
concrete [data-organization proposal](data-organization.md).

**Decision and acceptance.** Preserve dynamic slot, alias, validity and failure
obligations. A fixed dependency pattern is not necessarily a fixed linear matrix:
limiters and modes such as no-inflow can depend on values. Measure avoided
planning/dispatch together with retained plan bytes and actual consumption.
No distributed runtime, CUDA graph dependency or generic JIT is required merely
to apply the CPU-side principle.

## Relationship To The Initial Design

[Data organization](data-organization.md) is a concrete original-leaf baseline
proposal, not a declaration that E1--E4 have been rejected. E1 affects its compute
identity early; E5 fits its preparation boundary. E2 and E3 introduce alternative
reconstruction meanings. E4 changes the execution platform and location cost.

Use the [exploration rule](development.md#how-exploration-enters-development)
to select a route from a real decision, state what would count as success, and
carry the conclusion into the selected design. Existing-source evidence can
justify a design decision without a new experiment. A retained possibility must
have an entry condition and a scope; it is not an endless parallel research task.
