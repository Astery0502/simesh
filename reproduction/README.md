# Manuscript figure scripts

This directory contains the drawing scripts for Figures 1–7 of the simesh
manuscript. Download the plotting inputs separately and extract them into
`data/` beside these scripts. Git tracks only the scripts and documentation;
`data/` and `output/` are ignored. Original simulation snapshots are not required
for redrawing the figures. Download the
[2026-09-21 plotting data ZIP](https://drive.google.com/file/d/1XglvwIhu7mi8HKRKTBa28WoX4GM6d75G/view?usp=sharing)
and extract its top-level `data/` directory here. Google Drive may ask you to
confirm the download because the archive exceeds its virus-scan size limit.
The ZIP SHA-256 is
`8208bc79c354e551b9137df2cd9a31215aaef6e6223ac444715229daaadd8a1a`.

Figures 1 and 2 are diagrams generated entirely from code. Figures 3–7 redraw
retained scientific products supplied by the user. The drawing scripts do not rerun
the simulations or the full scientific analysis pipeline. Rendering requires
neither a compiled simesh installation nor network access.

## Run

Clone the repository and run the commands below from its root. Use Python 3.11
or newer; a dedicated environment keeps the rendering dependencies separate
from other projects:

```sh
git clone https://github.com/Astery0502/simesh.git
cd simesh
python -m venv .venv-figures
# macOS/Linux; on Windows use .venv-figures\Scripts\activate
source .venv-figures/bin/activate
python -m pip install -r reproduction/requirements.txt
```

Keep the repository commit ID (`git rev-parse HEAD`) with your outputs to identify
the exact drawing scripts used. The archive checksum above identifies the matching
plotting-data release.

Render the diagrams without any data:

```sh
python reproduction/render.py 1 2
```

For scientific figures, download the plotting inputs separately and extract
all their contents into `reproduction/data/`. The layout after extraction is:

```text
reproduction/
├── render.py
├── verify.py
├── requirements.txt
├── scripts/
├── data/                 # Download separately; ignored by Git
│   ├── figure3/
│   ├── figure4/
│   ├── figure5/
│   ├── figure6/
│   ├── figure7/
│   ├── DATA.md
│   └── manifest.json
└── output/               # Generated figures; ignored by Git
```

Check the inputs and render all figures:

```sh
python reproduction/verify.py
python reproduction/render.py
```

To render selected figures from a different data location, replace
`FIGURE_DATA_DIR` with that location:

```sh
python reproduction/render.py 4 5 --data FIGURE_DATA_DIR --output figure-output
```

The data root contains `figure3/` through `figure7/`, with the filenames listed
below. The separate data distribution also contains `DATA.md` (array meanings,
units, validity, and limitations), `manifest.json` (checksums and array schemas),
and data license notices. Only the selected figures' data directories are
needed for rendering; verification checks the complete data manifest.

`SIMESH_FIGURE_DATA` can be used instead of `--data`. Explicit command-line
arguments take precedence, followed by the environment variable, then the
`data/` directory beside `render.py`. The default works from any working
directory. Explicit relative paths are interpreted from the working directory.
The default output directory is `output/` beside
`render.py`, and is ignored by Git. Rendering does not modify input data.
Each figure produces PNG and PDF files; some also produce SVG files and
supporting panels. Figure 6 can take more time and memory because of its
three-dimensional polygons.

## Figure inputs

All filenames below are relative to the externally supplied data root.

| Figure | Script | External inputs |
| --- | --- | --- |
| 1 | `scripts/render_figure1.py` | None |
| 2 | `scripts/render_figure2.py` | None |
| 3 | `scripts/render_figure3.py` | `figure3/`: `slice.npz`, `native-analysis.npz`, `native-analysis.json` |
| 4 | `scripts/render_figure4.py` | `figure4/`: `profiles.npz`, `magnetic-profiles.npz`, `schematic-geometry.npz`, `summary.json`, `schematic-summary.json` |
| 5 | `scripts/render_figure5.py` | `figure5/`: `diagnostics.npz`, `sampling.npz`, `amr-slice.npz`, `lines.npz`, `selection.json` |
| 6 | `scripts/render_figure6.py` | `figure6/`: `lines.npz`, `bottom.npz`, `maps.npz`, `projection.npz` |
| 7 | `scripts/render_figure7.py` | `figure7/`: `diagnostics.npz`, `configuration.json`, `plot-settings.json`, `bands.npz`, `colormaps.npz` |

The scripts read NPZ archives without Python pickle. The dependency versions in
`requirements.txt` describe the checked rendering environment, not the original
scientific calculation environment. Pixel-identical rendering across different
Matplotlib versions, fonts, and platforms is not guaranteed.

## Recompute thermal products from a snapshot

`recompute_thermal.py` is a separate author-side calculation entry for the
thermal example. It needs an installed simesh package and the original eruption
snapshot, supplied separately. The figure-rendering requirements alone are not
sufficient. Install the package from the repository root with
`python -m pip install -e .`.

Use the physical configuration retained with Figure 7 explicitly:

```sh
python reproduction/recompute_thermal.py \
  --snapshot SNAPSHOT.dat \
  --config reproduction/data/figure7/configuration.json \
  --output data/analysis/thermal-new-run \
  --views y --diagnostics --verify
```

Replace `SNAPSHOT.dat` with the input location. The output must be a new or empty
directory outside the published data. Use `--views x y` to compute both views in
one run, or `--bands 94 1354 192` to select wavelengths. The default includes all
configured bands. `--verify` checks the 94 Angstrom result with independent
midpoint quadrature; `--diagnostics` adds the native slice, regional mass
histograms, column mass, and mass-weighted temperature.

Density and temperature are recovered once into memory, shared by this run's
bands, views, and diagnostics, and released when the process finishes. For the
original snapshot these two arrays occupy about 768 MiB; prepared integration
fields require additional memory. No persistent thermodynamic cache, sibling
experiment directory, or filesystem timestamp is used to resume work. Only
final results and scientific provenance are saved. A failed run leaves its
partial final results in that output directory; start the next run in a new or
explicitly cleared directory.

Results are saved as `view-x/` and/or `view-y/` result archives, optional
`diagnostics.npz`, and JSON provenance with the input and calculation-script
checksums. This is an independent recalculation, not an automatic replacement
of the plotting data. `render.py` continues to read the downloaded inputs; new
scientific products must be compared and deliberately packaged as a new data
release before being presented as updated paper figures.

## Scientific scope

The original calculations used development versions of simesh. Their exact
commits have not been established for every figure. Recomputing the scientific
products requires the original snapshots and analysis configurations, which
are not included here; no access to those materials is promised by this README.

Figure 7 retains the manuscript's coronal unit preset and optically thin model.
The preset still needs confirmation against the simulation setup, and response
normalization has not been established for absolute comparisons between
instruments. Reproducing the illustration does not resolve these calibration
questions.

## Data maintenance and license

Keep downloaded inputs in the ignored `data/` directory or another storage
location; never add them to Git. After intentional data changes, its maintainer
can refresh the manifest with `python reproduction/verify.py --write-manifest`
(or supply `--data FIGURE_DATA_DIR` for another location).
The machine-path check is not a general secrets scanner.

Scripts follow the repository's [GPL-3.0-only license](LICENSE). Data and
color-table license notices accompany the separate data distribution. Algorithm
and response-table provenance is recorded in the main repository's `ASSETS.md`.
