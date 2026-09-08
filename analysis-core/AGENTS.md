# Independent simesh development

The N4 core design is fixed. Start with `../docs/analysis-core/current.md` and
`../docs/analysis-core/application-development.md` for new application work.
Use `README.md` and `MIGRATION.md` for actual data-consumption interfaces.
Read `../docs/analysis-core/next-generation-design.md` when a core boundary matters.

Earlier development procedures, comparisons and experiments live in
`../docs/analysis-core/archive/`. They are on-demand historical references,
not an active task queue or mandatory application-development process.

- Production code and build inputs live entirely in this directory.
- Use this directory's `.venv/bin/python`. Create that environment with the
  parent project's Python 3.11 when bootstrapping; never install into the parent
  environment or change the donor worktrees.
- Run `make build` here after changing Cython. For application changes, run a
  representative workflow and the relevant tests; a full build or benchmark
  matrix is unnecessary when the core is unchanged.
- Keep retained primitive compiler semantics separate from analysis kernels.
- Default to one task. Use additional review or targeted performance comparison
  when a concrete risk or measured problem warrants it, not at every checkpoint.
- Preserve the source/preparation and storage/validity boundaries. Core changes
  should address an actual defect or a required application capability.
- Read `ASSETS.md` only when retained code provenance is relevant. If a numerical
  comparison is needed, use separate interpreters for the two packages.
- Keep implementation and public documentation in English. Development evidence
  and collaboration records in `../docs/analysis-core/` are in Chinese.
- Read `MIGRATION.md` before changing compatibility or file-product boundaries.
  `amrvac/` and `utils/lib/amr/` retain stateful compatibility; native Source,
  Fields and scientific consumers must not depend on their mutable AMRMesh.
