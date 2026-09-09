# Documentation Reorganization Record

Date: 2026-09-07. Scope: editorial consolidation of the nine existing analysis-core
documents, incorporating the user's clarification of consumption geometry.
No implementation, numerical strategy, public API or historical capability status
changed. This record explains content movement; current definitions live in the
documents linked below.

## Organization

The reading path is intent -> shared data spec -> consumer result spec -> remaining
decisions/evidence. Application breadth, candidate mechanisms, usage sequences
and measurement policy are references reached from a concrete question. Existing
filenames are retained to preserve repository navigation. Removed shared-topic
sections have short anchor redirects instead of duplicate normative text.

The central design question is how shared ghost-prepared AMR data serves different
geometric access patterns. F produces irregular paths sequentially through RK
stages; D and slices select regular physical regions; L processes the global
domain through view-dependent rays, different pixel depths and AMR crossings.
Physical field dependencies and exact operator reach remain part of each contract.

## Content Migration

| Previous content | Current primary location | Editorial action |
| --- | --- | --- |
| README organizing model, repeated checkpoint summaries and ownership list | [README](historical-index.md) | Replace discussion chronology with the reading path, shared geometric model and status vocabulary |
| Intent goals, baseline status prose and workflow priority/scale repetitions | [Intent](intent.md) | Consolidate priorities, operating envelope, engineering constraints and scope; retain compatibility/portability obligations |
| Prepared-fields model; baseline common contract; workflow field dependencies, checks, local derivation, memory/reuse rules; lifetime draft constraints | [Shared spec](prepared-fields.md) | Merge into request/geometry, field meaning, block views, reach, preparation, lifetime, reuse and parallel requirements |
| F/D/L result draft plus U03/U04/U07 and along-field numerical details | [Consumer spec](pipeline-results.md) | Organize by geometric access then scientific result/acceptance; keep stand-alone slice geometry in U04 |
| Baseline R1--R11, asset dispositions, scattered gaps and first-outcome advice | [Baseline](baseline.md) | Retain traceability and asset scope; consolidate unresolved choices into S1--S6/F1/D1/L1/X1; link requirements instead of restating them |
| WENO summaries repeated in workflows, baseline and design introductions | [Decision-relevant evidence](baseline.md#decision-relevant-evidence) | Keep compact findings and limitations together; detailed measurements remain in original rewrite evidence |
| Twelve workflow families, extended geometry/statistics/temporal/model uses and external product references | [Application catalog](workflows.md) | Preserve U01--U12 and product-specific distinctions; mainline/shared topics link to their owners |
| Large scenario/property map and T01--T18 investigation records | [Techniques](technique-candidates.md) | Replace repetitive scenario inventory with geometric/access lookup; preserve all 18 mechanism/source/limit records |
| Lifetime working hypothesis, duplicated constraints, sequences A/B/C and option costs | [Usage sequences](lifetime-sketches.md) | Move common guarantees to shared spec; retain concrete transitions, alternatives, invalidation examples and switching costs |
| Workflow scientific accuracy/"fast" sections and performance policy | [Performance](performance.md) | Consolidate scientific versus implementation acceptance, profiles, metrics and complete-memory accounting; preserve existing regression thresholds |

## Preserved Distinctions

These are pointers for checking information preservation, not new definitions.

| Detail that must survive consolidation | Definition or evidence |
| --- | --- |
| Ordinary B tracing uses ghost-prepared trilinear neighborhoods; geometric consumption distinguishes the mainlines | [Shared geometry](prepared-fields.md#consumption-geometry-and-access-patterns), [F access](pipeline-results.md#geometry-and-prepared-access) |
| At least two valid input layers, one-layer derivative consumption, one remaining layer for trilinear sampling; wider/asymmetric chains need their real reach | [Input reach and validity](prepared-fields.md#input-reach-and-remaining-validity) |
| Selected leaves versus support-only data; non-leaf parent selection; physical domain versus padding or RAM residency | [Selection](prepared-fields.md#request-and-physical-selection), [operating envelope](intent.md#operating-envelope) |
| Per-field physical meaning/validity; logical leaf versus storage slot; actual usable backing and geometry | [Field definitions](prepared-fields.md#physical-field-definitions), [block description](prepared-fields.md#what-a-prepared-block-must-describe) |
| Coarse/fine target-grid meaning, edge/corner coverage, slope/base closure and order of transfer/derivation/normalization | [Transfer and reach](prepared-fields.md#numerical-meaning-before-storage-choice) |
| Native field products and both usage lifetimes; retained product versus evictable cache; invalidation versus storage release | [Lifetime and ownership](prepared-fields.md#user-model-region-fields-and-two-lifetimes) |
| Explicit complete/partial/failure meaning, borrowing, output lifetime, no implicit spill/replay or mandatory precount | [Product delivery](prepared-fields.md#explicit-retention-versus-internal-caching) |
| Whole-domain D remains whole-domain even when the displayed slice is small; F diagnostics do not require running D | [D](pipeline-results.md#d-whole-domain-currentgradient-then-slices), [F](pipeline-results.md#f-magnetic-lines-and-along-line-diagnostics) |
| Q methods remain open; twist and Scott Q have distinct derivative/state needs; saved path points do not supply arbitrary coupled diagnostics | [F result](pipeline-results.md#minimum-result-contract) |
| Parallel independent seeds, optional paths and user-selected retracing; job/output size is chosen for suitable resources | [F retention](pipeline-results.md#retention-and-acceptance), [scale](intent.md#operating-envelope) |
| LOS view/pixel depth and AMR path lengths; response/EOS/units, nonlinear reconstruction and no duplicate parent/ghost material | [L](pipeline-results.md#l-full-domain-los-integration-of-a-local-response) |
| Slice samples versus intersections/averages; surface seams; whole-cell threshold versus clipped volume; global connectivity/output growth | [U04--U09](workflows.md#u04-axis-aligned-slices-oblique-slices-and-line-profiles) |
| CHS one-layer/serial scope, RHC callback borrowing, HPR invocation proofs, LFE full selected-block output and unweighted sum | [Asset roles](baseline.md#existing-work-contract-and-implementation-disposition) |
| Resident/bounded comparisons do different I/O work; bridge cost, uncontrolled OS cache and short WENO traces limit claims | [Evidence](baseline.md#decision-relevant-evidence) |
| Complete input/metadata/worker/output resources, alias accounting and RSS separation; no per-thread full-budget allocation | [Resources](performance.md#complete-workflow-resources), [parallel ownership](prepared-fields.md#resources-and-parallel-ownership) |

## Resolved Discussion Tensions

- Earlier prose overemphasized field dependencies or halo-free alternatives as
  the difference between F/D/L. The user's clarification establishes shared
  ghost-prepared data and different geometric consumption as the organizing
  principle. Interior-only operations remain valid within their declared scope;
  they do not characterize the three mainlines.
- The old repeated-sampling-first recommendation is historical. The current
  checkpoint specifies preparation and consumption together before choosing a
  first useful implementation. Sampling remains a reusable primitive.
- Earlier joint-trajectory-retention preference is superseded by optional
  retention, diagnostics-only results and explicit selected-seed retracing.
  Independent-seed parallel design remains a priority.
- The earlier lifetime study is a supporting composition reference, not a
  prerequisite framework. No session class, universal cache or executor follows
  from the two confirmed usage lifetimes.
- Historical one-layer CHS validity and the new input-halo requirement have
  different scopes. Keep the former as evidence; adapt it before claiming the
  latter. No historical completion state changes.
- Scientific quantities, layouts, exact budgets and algorithms that lacked a
  decision remain explicitly open. Reorganizing prose does not decide them or
  activate experiments. Historical review notes are not claimed as fresh review.

## Maintenance

Change a definition at its primary location, then update references and affected
decision/evidence rows. Preserve T/U/R identifiers and historical inbound anchors.
Validate local Markdown paths/anchors, requirement coverage, status and the diff
against the starting documents. This record is a one-time migration aid, not a
new recurring audit or reporting system.
