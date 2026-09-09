# Cython build and installation

Run all commands from the repository root with Python 3.11 or newer.
The root package is the selected implementation; `legacy/` is not a build input.

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
| `src/simesh/amrvac/_mesh/*.pyx` | Stateful Dataset mesh, forest and Morton operations | Separate retained configuration; serial even with OpenMP enabled |

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
is available. It does not parallelize the stateful compatibility extensions.
`simesh.amrvac.openmp_build_info()` describes those compatibility extensions;
it is not the build status of the native kernels. Use ordinary serial builds
when the optional runtime is unavailable.

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
