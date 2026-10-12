# Manuscript figure reproduction

This directory redraws **Figures 1–8 of the current simesh manuscript** from
code and separately supplied plotting inputs. It does not rerun simulations or
scientific analysis. Redrawing needs neither original simulation snapshots,
a compiled simesh installation, nor network access after downloading the inputs.
Figures 1 and 2 are generated entirely from code.

Use the **2026-10-12 eight-figure plotting-data bundle** with these scripts.
Download the [2026-10-12 plotting-data ZIP](https://drive.google.com/file/d/1WbO4nQGvkneY76LWbJ3rtrBpHijMbKAV/view?usp=sharing)
and check its [release identity and archive checksum](RELEASE.md). Google Drive
may ask for download confirmation because the archive exceeds its scan limit.
The 2026-09-21 seven-figure bundle has different numbering and older thermal
products; it must be used with its historical scripts, not this revision.
Data and generated output are ignored by Git.

## Run

Use Python 3.11 or newer. From the repository root:

```sh
git clone https://github.com/Astery0502/simesh.git
cd simesh
git checkout paper-figures-2026-10-12
python -m venv .venv-figures
# macOS/Linux; on Windows use .venv-figures\Scripts\activate
source .venv-figures/bin/activate
python -m pip install -r reproduction/requirements.txt
```

Extract the matching archive into `reproduction/`. Its top-level `data/`
directory contains `figure3/` through `figure8/`, `DATA.md`, `release.json`,
`manifest.json`, and license notices. Then verify the inputs and redraw all
current figures:

```sh
python reproduction/verify.py
python reproduction/render.py
```

Diagrams need no data:

```sh
python reproduction/render.py 1 2
```

You can select manuscript figure numbers and another data/output location:

```sh
python reproduction/verify.py --data FIGURE_DATA_DIR
python reproduction/render.py 4 8 --data FIGURE_DATA_DIR --output figure-output
```

`--data` takes precedence over `SIMESH_FIGURE_DATA`, then defaults to `data/`
beside `render.py`. Relative explicit paths are interpreted from the working
directory. The default output is `output/` beside `render.py`; output must be
outside the input directory. Rendering does not modify data. Keep
`git rev-parse HEAD`, the bundle checksum, and `release.json` with the outputs.

## Figure inputs and outputs

All input paths are relative to the separately supplied data root. Numeric
arguments, script names, input directories, and main output prefixes now follow
the **current manuscript numbering**.

| Figure | Inputs | Main output prefix |
| --- | --- | --- |
| 1 — framework | None | `figure1-framework` |
| 2 — architecture | None | `figure2-software-architecture` |
| 3 — AMR slice and ghost reconstruction | `figure3/slice.npz` | `figure3-native-amr-analysis` |
| 4 — recorded performance | `figure4/results/`: 15 endpoint-mode measurement JSON files | `figure4-performance` |
| 5 — representation profiles | `figure5/`: profiles, magnetic profiles, schematic geometry, and summary JSON files | `figure5-p16-field-current` |
| 6 — current-sheet diagnostics | `figure6/`: `diagnostics.npz`, `sampling.npz`, `amr-slice.npz`, `lines.npz`, `selection.json` | `figure6-current-sheet` |
| 7 — filament magnetic topology | `figure7/`: `lines.npz`, `bottom.npz`, `maps.npz`, `projection.npz`, `provenance.json` | `figure7-magnetic-topology` |
| 8 — thermal diagnostics and absorbed emission | `figure8/`: `diagnostics.npz`, `configuration.json`, `plot-settings.json`, `bands.npz`, `colormaps.npz` | `figure8-thermal-diagnostics` |

Every main figure produces PNG and PDF; some also produce SVG and supporting
panels. Figure 7 takes longer because it draws three-dimensional polygons.
See [the data guide](DATA.md) for complete file definitions, units, validity,
normalization, and limitations. NPZ inputs use no Python pickle.

The old seven-figure numbering maps as follows: old 1–3 remain 1–3, old 4–7
become current 5–8, and the new performance comparison is current Figure 4.
The updated Figure 3 contains only panels (a) and (b). Figure 7 retains the
already published variational Q data. Figure 8 adds continuum absorption,
emission-measure-weighted temperature, and a volume emission-measure temperature
distribution. Merely renaming the old thermal inputs cannot produce these panels.

The versions in `requirements.txt` identify the checked drawing environment,
not the original calculation environment. Exact pixels across different
Matplotlib versions, fonts, and platforms are not guaranteed.

## Recompute thermal products from a snapshot

`recompute_thermal.py` is a separate author-side calculation entry for the
thermal example. It needs an installed simesh package and the original eruption
snapshot, supplied separately. The figure-rendering requirements alone are not
sufficient. Install the package from the repository root with
`python -m pip install -e .`.

Use the physical configuration retained with Figure 8 explicitly:

```sh
python reproduction/recompute_thermal.py \
  --snapshot SNAPSHOT.dat \
  --config reproduction/data/figure8/configuration.json \
  --output data/analysis/thermal-new-run \
  --views y --absorption --subdivisions 8 --verify
```

Replace `SNAPSHOT.dat` with the input location. The output must be a new or empty
directory outside the published data. Use `--views x y` to compute both views in
one run, or `--bands 94 1354 192` to select wavelengths. The default includes all
configured bands. In an unabsorbed run, `--verify` checks the 94 Angstrom result
with independent midpoint quadrature. `--diagnostics` retains the older native
slice, regional mass histograms, column mass, and mass-weighted temperature;
it does not calculate the emission-measure products in current panels (e) and
(f). This entry alone therefore does not recompute the entire current Figure 8.

Add `--absorption` to include H I, He I, and He II continuum absorption using
LTE Saha populations. Emissivity and opacity are evaluated at native cell centers
before exact-phase support preparation and ordered radiative transfer. Select
the number of midpoint slabs with `--subdivisions` (default 4). In this mode,
`--verify` compares every band's transfer with refined quadrature and independent
reference traversal. The prescription gives zero opacity at IRIS 1354 Angstrom
and does not include non-LTE line formation or resonance scattering at 304
Angstrom. No external background is imposed.

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
checksums. Absorbed runs also save optical-depth and unabsorbed-intensity result
archives, absorption-fraction arrays, and per-band verification reports.
This is an independent recalculation, not an automatic replacement
of the plotting data. `render.py` continues to read the downloaded inputs; new
scientific products must be checked before use in revised figures. Distributing
updated plotting inputs requires a separate data release.

## Scientific scope

The exact scientific calculation commits are not established for every
historical product. A complete recalculation needs the original snapshots,
analysis configurations, and corresponding source versions, which are not
included in the plotting bundle. Figure 4 redraws measured timings and RSS;
it does not reproduce those timings on another machine.

Figure 8 uses the retained coronal unit preset, fully ionized H/He emission
and emission-measure diagnostics, and LTE Saha H/He continuum absorption.
The unit preset still needs confirmation against the original simulation setup;
complete response calibration metadata are unavailable. No cross-instrument
absolute intensity comparison is claimed. See `DATA.md` and the configuration
for the model and verification records.

## Assemble a new data release

The author-side assembler requires validated products supplied separately.
It copies only the files required by the current figures, retains license
notices, and generates a new checksum/schema manifest after checking for
machine-specific paths. Use a new or empty output directory and preserve the
historical bundle:

```sh
python reproduction/prepare_data.py \
  --previous-data PREVIOUS_DATA_DIR \
  --thermal-data VALIDATED_THERMAL_PLOT_DIR \
  --performance-results RECORDED_ENDPOINT_JSON_DIR \
  --output NEW_DATA_DIR
python reproduction/verify.py --data NEW_DATA_DIR
python reproduction/render.py --data NEW_DATA_DIR --output NEW_FIGURE_DIR
```

`PREVIOUS_DATA_DIR` is the verified 2026-09-21 bundle; the thermal directory
contains the five final plotting files for absorbed intensity and both
emission-measure diagnostics. The measurement directory supplies AMR, L5,
and L6 records for 1, 4, 8, 16, and 32 workers. The assembler does not calculate
new scientific results. Archive the new directory as top-level `data/` and
record the ZIP checksum and matching script revision in the release notes.
The machine-path check is not a general secrets scanner.

Scripts follow the repository's [GPL-3.0-only license](LICENSE). The separate
data bundle retains its original license and the SunPy color-table notice.
Algorithm and response-table provenance is recorded in the root `ASSETS.md`.
