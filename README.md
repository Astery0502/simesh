### Installation

You can install this package using either pip with setup.py or Poetry.

#### Using pip (setup.py)

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/simesh.git
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

### Requirements

- Python ≥ 3.11
- NumPy ≥ 2.2.1

Optional test dependencies:
- pytest ≥ 8.3.4
- yt ≥ 4.4.0
- f90nml ≥ 1.4.4
- JupyterLab ≥ 4.3.4
- ipykernel ≥ 6.29.5
