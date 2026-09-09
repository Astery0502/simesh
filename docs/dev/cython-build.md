# Cython build and installation

Run all commands from the repository root with Python 3.11 or newer.
The root package is the selected implementation; `legacy/` is not a build input.

## Documentation

```bash
.venv/bin/python -m pip install -e '.[docs]'
make docs
make docs-serve
```

`make docs` checks API coverage/parameters and links, then builds `site/` in strict
mode. `make docs-serve` previews the same pages locally. Generated HTML is ignored
by Git. `docs` dependencies are optional and are not runtime requirements.
Static parsing reads `src/` with module inspection disabled; building docs does
not import simesh or require compiled extensions. To install only documentation
tools without building simesh, install the `docs` extra's tool requirements
listed in `pyproject.toml` into a separate environment.

Register each interface once with `::: simesh.object` in the
[user](../user/api.md) or [developer](api.md) page. Keep executable examples in
`examples/` and follow the [content rules](#docstring-content) below.

`make docs-check` checks API resolution/coverage, structured parameter names,
selected members and links, including the LOS wrapper's forwarded keyword names.

### Docstring content

Keep each interface's description at its source definition. Generated Notes
sections come from that docstring; do not maintain a second copy in Markdown.

| Content | Location |
| --- | --- |
| Purpose | One-line docstring summary |
| Parameter meaning, shapes, units and accepted values | Parameters |
| Returned shape, attributes and status meaning | Returns / Attributes |
| Special constraints affecting correct calls | Notes |
| Contracts shared by several interfaces | Class/module docstring, linked from consumers |
| Internal algorithm steps and optimization rationale | Ordinary source comments |
| Cross-module architecture | Short architecture Markdown |

Aim for one to three necessary constraints in Notes, and omit the section when
none are needed. This is a brevity guideline, not a reason to remove a required
scientific or lifetime condition. Do not repeat Parameters/Returns, narrate the
implementation, or retain optimization history there. Explain a shared rule fully
once; keep only the operation-specific consequence at each call, such as a
derivative consuming one valid halo or a returned view expiring with its lease.

Signatures, defaults and available type annotations are extracted from source.
Numerical meaning, units and lifetime rules remain authored content and must be
reviewed with implementation changes; structural checks cannot prove them correct.

## Package

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
make build
make test
```

`make build` runs `setup.py build_ext --inplace --force` using
`.venv/bin/python`. Rebuild after editing `.pyx`/`.pxd` files or moving compiled
modules. `make test` runs pytest on `tests/`; it does not collect archived tests.
Old `build.py`, `scripts/build_ext.py` and `make build-amr` workflows belong to
the archived preceding package.

## Extension groups

| Sources | Role | Compiler configuration |
| --- | --- | --- |
| `src/simesh/_kernels/primitives/` | Retained AMR numerical primitives | Separate original Cython arithmetic/checking semantics |
| `src/simesh/_kernels/*.pyx` | Preparation, sampling, differentiation, tracing, connectivity and LOS | Optimized native configuration; explicit optional OpenMP |

Headers such as `tree.pxd`, `math.pxd` and `native.pxd` are Cython interfaces;
they are not separately compiled extensions. All build inputs live under `src/`.
Generated C files and shared objects are ignored by Git and must not be used as
a substitute for the source files in a release.

## Optional dependencies and OpenMP

The base runtime needs NumPy. `.[test]` adds pytest; `.[dev]` also installs
Cython/setuptools/wheel for direct local rebuilds. `.[fft]` adds SciPy for
optional FFT convolution, and `.[plot]` adds Matplotlib for example PNG output.

```bash
SIMESH_OPENMP=1 make build
```

This enables supported native kernels when a suitable compiler/OpenMP runtime
is available. Use ordinary serial builds when the optional runtime is unavailable.

## Isolated wheel verification

```bash
.venv/bin/python -m pip wheel . --no-deps --wheel-dir dist
.venv/bin/python scripts/verify_install.py \
    --wheel dist/<built-wheel>.whl --target /tmp/simesh-installed-check
```

Substitute the produced wheel filename and use a new target directory. The
checker loads the extracted wheel with isolated Python path handling, runs the
active tests against it, and checks that package modules come from that wheel.
Neither an editable checkout nor an archived provider may supply simesh code.
