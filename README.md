# simesh

`simesh` is a Python/Cython toolkit for native AMR scientific analysis and
numerical result delivery. The repository root contains the selected package,
including its AMRVAC file interfaces and array-based magnetic field tools.

Start with the [Chinese application guide](docs/application-guide.md) for
working examples, units, supported physics and output formats. The
[API reference](docs/api-reference.md) lists current signatures; the
[interface/resource review](docs/interface-review.md) records measured costs
and proposed improvements.

## Scientific workflows

- Native sampling, slices, uniform-grid products and custom derived fields
- Gradients, divergence, curl and explicitly normalized magnetic diagnostics
- Magnetic tracing, Q/Q-perpendicular, localized footpoints and twist
- Ideal-MHD recovery, AMR integrals/statistics and rectangular surface flux
- Scalar and historical AIA171 thermal LOS
- Identified lines/profiles, result save/load and incremental output shards
- Retained Dataset, ordinary `.dat`, Cartesian 2D and level-1 VTK interfaces

Native analysis accepts balanced, nonperiodic Cartesian 3D AMRVAC v5 ordinary
fields and equivalent array sources. Models, unit scales and numerical schemes
are explicit. CT face analysis, spherical/periodic native analysis and GPU
execution are outside the current profile. Inspect validity and termination
before treating an output as a complete physical result.

## Install and run

Python 3.11 or newer is required. From the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python examples/standard_applications.py --output /tmp/simesh-standard
.venv/bin/python examples/recovered_state_analysis.py --output /tmp/simesh-quantitative
```

The first example writes an ordinary NumPy bundle and JSON summary. The second
writes application result files readable with `simesh.load_result`. See the
application guide for the exact contents and the distinction between formats.
Optional `.[plot]` enables PNG rendering; `.[fft]` enables SciPy FFT convolution.

For development, `make build` rebuilds Cython and `make test` runs current tests.
See [build instructions](docs/cython-build.md) for OpenMP and isolated installation
checks, and [architecture](docs/architecture.md) for module boundaries.

## Project layout and history

Current implementation, tests and examples live in `src/`, `tests/` and
`examples/`. [Migration notes](MIGRATION.md) list moved imports. Historical
implementations and superseded documents are under [legacy/](legacy/README.md)
and are excluded from the build. They are not prerequisites for using the
current package. [ASSETS.md](ASSETS.md) and `LICENSE` retain implementation
provenance and licensing information.
