### Installation

You can install this package using either pip with setup.py or Poetry.

#### Using pip (setup.py)

1. Clone the repository:
   ```bash
   git clone https://github.com/Astery0502/simesh.git
   cd simesh
   ```

2. Install the package:
   ```bash
   pip install .
   ```

   To install with test dependencies:
   ```bash
   pip install .[test]
   ```

#### Using Poetry (Recommended)

1. First, install Poetry if you haven't already:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/Astery0502/simesh.git
   cd simesh
   ```

3. Install dependencies and the package:
   ```bash
   poetry install
   ```

   To install with test dependencies:
   ```bash
   poetry install --with test
   ```

#### Requirements

- Python ≥ 3.11
- NumPy ≥ 2.1.1

Optional test dependencies:
- pytest ≥ 8.3.4
- f90nml ≥ 1.4.4
- JupyterLab ≥ 4.3.4
- ipykernel ≥ 6.29.5

### Usage

The package provides interfaces to load AMR structured data from datfiles output by AMRVAC and manipulate over the AMR structured data in a AMRVAC-like manner. By the way, the package also provides interfaces to generate the AMR structured data independent of any file input.

#### Load AMRVAC data

```python
from simesh import amr_loader

ds = amr_loader(datfile)
```

or we can load the data from the uarray directly:

```python
from simesh import load_from_uarrays

ds = load_from_uarrays(uarrays)
```

#### Manipulation of the AMR structured data

See details in the [demo.ipynb](demo.ipynb) for now.

