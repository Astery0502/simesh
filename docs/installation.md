# Installation

## Install from source

The supported install path is `pip` and `setuptools`. Cython extensions are
compiled during installation, including editable installs.

```bash
git clone https://github.com/Astery0502/simesh.git
cd simesh
pip install .
```

For editable local development:

```bash
pip install -e .
```

Requirements:

- Python >= 3.11
- NumPy >= 1.23.5
- Cython >= 3.0 for source builds

## Development build and test

Package installation compiles all Cython modules found under
`src/simesh/utils/lib/`.

Useful development commands:

```bash
pip install -e .
make build
make build-amr
make test
make benchmark-smoke
```

Notes:

- `make build` compiles all `.pyx` files under `src/simesh/utils/lib/`
- `make build-amr` compiles only the `src/simesh/utils/lib/amr/` subtree
- after editing `.pyx` files, rerun `make build` to rebuild extensions in place
- `make clean` removes compiled extensions and generated packaging artifacts
- `make test` rebuilds extensions and runs the script-based tests under `tests/`

## Optional OpenMP acceleration

OpenMP is optional and is not enabled by default. The default installation path
builds the AMR Cython kernels without OpenMP compiler or linker flags.

To opt into OpenMP during installation:

```bash
SIMESH_OPENMP=1 pip install .
```

For editable development builds:

```bash
SIMESH_OPENMP=1 pip install -e .
make build-amr-openmp
```

On systems without an OpenMP implementation, use the default build commands and
do not set `SIMESH_OPENMP=1`. On macOS, OpenMP usually requires installing
`libomp` first, for example with Homebrew. On Linux, GCC usually provides
OpenMP through `-fopenmp`.

Check the compiled extension status from Python:

```python
from simesh.utils import openmp_build_info, openmp_enabled

print(openmp_enabled())
print(openmp_build_info())
```

When OpenMP is enabled, control runtime thread count with standard OpenMP
environment variables:

```bash
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE python your_script.py
```
