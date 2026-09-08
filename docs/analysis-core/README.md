# Analysis core and application development

The independent N4 core under `analysis-core/` is the selected, fixed design.
The next development focus is applications that consume its existing data and
results. Core optimization and architectural exploration are not the default
workstream.

## Start here

| Need | Read |
| --- | --- |
| Current direction and selected baseline | [Current state](current.md) |
| Stable data, numerical and ownership boundaries | [Fixed core design](next-generation-design.md) |
| Develop an application using the core | [Application development](application-development.md) |
| Call the native Python interfaces | [Package README](../../analysis-core/README.md) |
| Use Dataset, file products or compatibility interfaces | [Migration guide](../../analysis-core/MIGRATION.md) |

For ordinary application work, start from the requested result, select existing
inputs and consumers, and verify the affected behavior with a small useful
example. A full core benchmark matrix, new worktrees, independent review and
research-stage paperwork are not routine prerequisites.

The core's scientific boundaries still apply: prepared coverage and valid halos,
explicit units and models, failure statuses, and ownership of retained outputs.
Investigate a core change only for a concrete correctness problem or an actual
application requirement that the existing interfaces cannot satisfy.

## Historical material

The earlier specifications, implementation stages, comparisons, experiments and
heavier development/performance procedures are preserved in the [archive](archive/README.md).
They retain historical context and evidence; they do not impose an active reading
sequence, task queue or acceptance process on new application work.
