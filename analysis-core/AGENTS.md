# Independent simesh development

For the active sampling reuse experiments, start with
`../docs/analysis-core/explorations/sampling-reuse.md` and the task-specific brief.
They define the authorized endpoint, shared baseline and serialized compute use.

Read `../docs/analysis-core/next-generation-design.md` and
`../docs/analysis-core/current.md` before
changing this implementation. Use current.md for the active delivery endpoint
and the distinction between delivered profiles and subsequent migration work.

- Production code and build inputs live entirely in this directory.
- Use this directory's `.venv/bin/python`. Create that environment with the
  parent project's Python 3.11 when bootstrapping; never install into the parent
  environment or change the donor worktrees.
- Run `make build` here after changing Cython, and `make test` for focused tests.
- Keep retained primitive compiler semantics separate from analysis kernels.
- Numerical reference runs must use separate interpreters for the two packages.
- Read `ASSETS.md` for retained code provenance; new orchestration must follow
  the source/preparation and storage/validity boundaries in the design.
- Keep implementation and public documentation in English. Development evidence
  and collaboration records in `../docs/analysis-core/` are in Chinese.
- Read `MIGRATION.md` before changing compatibility or file-product boundaries.
  `amrvac/` and `utils/lib/amr/` retain stateful compatibility; native Source,
  Fields and scientific consumers must not depend on their mutable AMRMesh.
