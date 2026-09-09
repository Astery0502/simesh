# simesh

`simesh` is a Python/Cython toolkit for native AMR scientific analysis and
numerical result delivery. The repository root contains the selected package,
including its AMRVAC file interfaces and array-based magnetic field tools.

Start with the [user interfaces](docs/user/index.md) and the
[source-generated API index](docs/user/api.md). The
[developer interfaces](docs/dev/index.md) cover core objects and advanced execution.
See the [documentation map](docs/index.md) for the full structure.

## Scientific workflows

- Native sampling, slices, uniform-grid products and custom derived fields
- Gradients, divergence, curl and explicitly normalized magnetic diagnostics
- Magnetic tracing, Q/Q-perpendicular, localized footpoints and twist
- CGS ideal-MHD recovery with `MHDUnits.solar()`, AMR integrals/statistics and rectangular surface flux
- Scalar and historical AIA171 thermal LOS
- Identified lines/profiles, result save/load and incremental output shards
- AMRVAC v5 ordinary-field input and complete-mesh or root-aligned `.dat` export
- Uniform-volume VTK export from sampled results or directly from AMRVAC files

Native analysis accepts balanced, nonperiodic Cartesian 3D AMRVAC v5 ordinary
fields and equivalent array sources. Public workflows use Source and Fields;
the archived mutable Dataset and 2D workflows are not shipped. VTK output is
limited to uniform volumes.
Models, unit scales and numerical schemes are explicit. CT face analysis,
spherical/periodic native analysis and GPU execution are outside the current profile. Inspect validity and termination
before treating an output as a complete physical result.

## Install and run

Python 3.11 or newer is required. From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python examples/user_quickstart.py --output example-output/user-quickstart
```

The quickstart creates a teaching snapshot and verifies a magnetic map, mass
and result save/load. Use a fresh output directory. More executable examples
are linked from the [user entry](docs/user/index.md).
Optional `.[plot]` enables PNG rendering; `.[fft]` adds FFT convolution.

## Build the interface reference

```bash
.venv/bin/python -m pip install -e '.[docs]'
make docs
make docs-serve
```

The build reads signatures and NumPy-style docstrings directly from `src/`
without importing compiled modules. `make docs` checks public API coverage,
parameter descriptions and links, then writes HTML to ignored `site/`.
GitHub Markdown shows the API directives; the built site renders their content.
For package development use `make build` and `make test`; see
[build instructions](docs/dev/cython-build.md).

## Project layout and history

Current implementation, tests and examples live in `src/`, `tests/` and
`examples/`. Historical implementations are under [legacy/](legacy/README.md)
and are excluded from the build. They are not prerequisites for using the
current package. [ASSETS.md](ASSETS.md) and `LICENSE` retain implementation
provenance and licensing information.
