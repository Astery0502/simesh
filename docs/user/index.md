# User interfaces

```python
import simesh as sm
from simesh import applications as app
```

[API reference](api.md): choose a function and read its signature, parameters,
return type and required support. Native analysis supports balanced nonperiodic
Cartesian 3D AMRVAC v5 ordinary fields; units and physical models are explicit.

## Install and run

From the repository root, with Python 3.11 or newer:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python examples/user_quickstart.py --output example-output/user-quickstart
```

Use a fresh output directory. The example creates its input and verifies
`Bz = 0.001 T`, `mass = 1e6 kg`, 64 usable map samples and result save/load.
The scales are teaching values, not simulation calibration.

## Examples

| Example | Input and output |
| --- | --- |
| [Quickstart](../../examples/user_quickstart.py) | Generated teaching snapshot and reloadable magnetic map |
| [Magnetic applications](../../examples/standard_applications.py) | Synthetic AMR arcade, Q/twist, paths and LOS; custom NumPy archive |
| [MHD analysis](../../examples/recovered_state_analysis.py) | Analytic recovered state, reductions, profiles and result files |

Exact API text is generated when
building the documentation; GitHub Markdown displays the object directives.
