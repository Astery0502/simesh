# simesh

`simesh` is a Python/Cython toolkit for native AMR scientific analysis and
numerical result delivery, with AMRVAC file interfaces and array-based magnetic
field tools.

This page introduces the supported capabilities and the shortest path to a first
run. Continue with the [user guide](docs/user/index.md) for analysis workflows and
the [API reference](docs/user/api.md) for exact interface contracts.

## Scientific workflows

- Native sampling, slices, uniform-grid products and custom derived fields
- Gradients, divergence, curl and explicitly normalized magnetic diagnostics
- Magnetic tracing, Q/Q-perpendicular, localized footpoints and twist
- CGS ideal-MHD recovery with `MHDUnits.solar()`, AMR integrals/statistics and rectangular surface flux
- Scalar and historical AIA171 thermal LOS
- Identified lines/profiles, result save/load and incremental output shards
- Reloadable native AMR sections and quantitative integral/statistical results
- AMRVAC v5 input, root-block regional analysis retaining AMR refinement, and `.dat` export
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
.venv/bin/python -m pip install -e .
.venv/bin/python examples/user_quickstart.py --output example-output/user-quickstart
```

Use a fresh output directory. The quickstart creates a teaching snapshot and
verifies `Bz = 0.001 T`, `mass = 1e6 kg`, 64 usable map samples and result save/load.
These scales are teaching values, not simulation calibration. Continue with the
[workflow examples](docs/user/index.md#examples) for your analysis task.
Optional `.[plot]` enables PNG rendering; `.[fft]` adds FFT convolution.

## Documentation and development

The [documentation map](docs/index.md) links workflow guides and API references.
Guides explain how to combine interfaces; API details are generated from source
signatures and docstrings when the documentation site is built.

For extension builds, tests, development dependencies and local documentation
preview, follow the [developer build instructions](docs/dev/cython-build.md).

## Project layout and history

Current implementation, tests and examples live in `src/`, `tests/` and
`examples/`. Historical implementations are under [legacy/](legacy/README.md)
and are excluded from the build. They are not prerequisites for using the
current package. [ASSETS.md](ASSETS.md) and `LICENSE` retain implementation
provenance and licensing information.
