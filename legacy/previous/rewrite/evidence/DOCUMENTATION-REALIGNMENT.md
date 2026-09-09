# Development Documentation Realignment

Date: 2026-09-05. Scope: user-authorized development/design documentation only.
No numerical contract, implementation, test, benchmark code, or production
package was changed. Existing completion claims retain their original scope.

## Requirement Preservation

| Previous or clarified requirement | Current authority and treatment | Validation/next consumer |
| --- | --- | --- |
| Native AMR, large DAT, static analysis priority | CHARTER product goals; ANALYSIS_WORKLOADS requests | Large native and repeated-analysis profiles |
| Efficient halo exchange/storage | FUNCTIONAL_COMPOSITION support/value/execution separation | Local-derived and along-field consumers |
| Derivatives consume validity; interpolation needs valid halo | ANALYSIS_WORKLOADS and FUNCTIONAL_COMPOSITION general requirements | Enforced through actual consumers; no dedicated field-pair exploration |
| Along-line diagnostics, derivatives, integrals | ANALYSIS_WORKLOADS along-field acceptance | Post-M1 outcome after derived validity and useful reuse |
| Functional style, explicit state/ownership/mutation | CHARTER constraints; FUNCTIONAL_COMPOSITION protocols | Existing contracts and future conformance |
| Storage/execution/compute substitution | FUNCTIONAL_COMPOSITION three boundaries | CPU/OpenMP and any future active GPU workflow |
| Canonical float64/int64 host layout | CHARTER and existing FND-001 | Unchanged current contract; explicit alternate protocol required |
| Exact arithmetic, IEEE behavior, deterministic strategies | Existing contracts; FUNCTIONAL_COMPOSITION numerical strategies | No tolerance/order change in this refactor |
| No hidden dispatch, hot-loop callbacks, or speculative framework | AGENTS operational rules; CHARTER extension scope | Boundary review; Cython rebuild remains acceptable |
| Independent references and scientific evidence | WORKFLOW integration; ANALYSIS_WORKLOADS accuracy | Interface/trajectory/diagnostic error checks in new contracts |
| Bounded memory and multi-objective performance | PERFORMANCE composed resources and comparisons | Old byte formulas unchanged; new workflows include controlled outputs/scratch |
| Proportional evidence, 10%/20% regression rules | PERFORMANCE | Existing defaults retained |
| Five-question decomposition and justified exceptions | DECOMPOSITION; one WORKFLOW design record | No duplicate forms; same boundary gate |
| Bounded exploration, incremental refinement, useful stopping | WORKFLOW; PERFORMANCE stopping rule | Changed-workload impact assessment, not a full M1 redo |
| Groups, readiness, material review, autonomous checkpoints | WORKFLOW | 1-4 members; integrated/complete distinction; independent review retained |
| 3D -> 2D -> periodic support; exclusions | CHARTER and ROADMAP | DIM-001 queued, required before 2D; staggered/non-Cartesian still excluded |
| Feature migration, public compatibility, fallback/rollback | SOURCE_MIGRATION and ROADMAP | Full public/build/cutover obligations retained |
| M1 status, historical measurements, fixture qualifications | M1 horizon and archived decision records | No new numerical pass claimed |
| English content, local venv, careful waits and Git scope | AGENTS and WORKFLOW | Documentation diff/links and scope checks |

## Ownership And Scheduling Changes

AGENTS is the task-based entrypoint. WORKFLOW owns the sole exploration record,
group cycle, review, and checkpoint process. DECOMPOSITION owns the five-question
analysis, PERFORMANCE owns cost evidence and stopping, and ANALYSIS_WORKLOADS
owns product acceptance. CURRENT links these authorities instead of repeating
them. Historical completed decisions move to evidence; the completed one-time
audit is no longer a recurring command.

M0/M1 stay complete. A new post-M1 analysis stage advances concrete M4/M6 work
needed for the user's five goals before broad M2/M3 development. After user
clarification, DVA-001 was withdrawn before specification: validity is a general
composition requirement, not a field-specific exploration gate. The current
assessment selects an evidence-backed optimization or missing boundary from
the retained directions; no implementation group is preselected.
DIM-001 remains proposed and queued; a
shared representation change can activate that prerequisite earlier. Original
M1 horizon recommendations remain historical evidence; the changed user workload
is the explicit reason for the new scheduling decision.

GPU portability means explicit semantic/buffer/execution boundaries and future
conformance. It does not add immediate GPU implementation to the roadmap gate,
weaken CPU arithmetic, or require a generic framework. New full-resource
accounting does not relabel historical workspace-only metrics as total memory.

## Validation

This checkpoint passed WORKFLOW's documentation gate:

- Independent review agreed with the material charter/architecture changes and
  found no remaining actionable issues after restoring the explicit clean-build
  requirement for future implementation group checkpoints.
- An ad hoc `.venv/bin/python` structural check verified changed/new files are
  English ASCII Markdown, local links/anchors resolve, and code fences balance.
- Every pre-existing capability table row is unchanged, including its status;
  no new capability row remains after withdrawing DVA-001. No numerical contract
  or code file is in the diff.
- Archived M1 workload records and capability narratives were compared with
  their Git HEAD originals. Content is preserved, with the one relocated
  evidence path adjusted explicitly.
- `git diff --check` passed; working-tree scope contains only rewrite Markdown.

Existing M1 test counts remain historical evidence. No extension build,
numerical tests, or performance campaign was run for this prose-only change.

## Optimization Execution Plan Update: 2026-09-07

The user requested a concrete order and completion criteria for M1 optimization.
CURRENT now selects lowering native-analysis halo preparation cost as the first
operational objective. The existing optimization note owns the comparison-set,
halo, cache-scaling, storage/output, conditional-compute, and closing sequence.
CAPABILITIES and ROADMAP link that order without copying its acceptance rules.

The plan separates individual implementation-group completion, bounded
optimization-round closure, and the broader post-M1 product-stage completion.
Optional candidates may be retained/deferred/rejected with evidence; unmet hard
requirements remain blockers. Numerical contracts, performance policy, original
M1 completion, queued DIM requirements, and the withdrawn field-specific
exploration remain unchanged. No new implementation capability is declared.

Documentation checks passed: 18 changed/new Markdown files, 64 resolving local
links/anchors, balanced code fences, unchanged original capability rows/statuses,
and `git diff --check`. Only the execution note, CURRENT, CAPABILITIES, ROADMAP,
and this dated record were adjusted in this follow-up. No implementation,
numerical tests, builds, or new performance measurements were performed.
