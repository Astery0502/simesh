# Figure data guide

All NPZ archives use ordinary NumPy arrays and can be opened with
`numpy.load(path, allow_pickle=False)`. JSON files contain scientific parameters
and display settings. Exact shapes and dtypes are listed in `manifest.json`.
Indices and offsets are zero-based; offset pairs describe half-open slices.
Spatial axes are Cartesian. A code length unit is 10 Mm in these examples.

## Figure 3 — native AMR analysis

`slice.npz` stores the native slice at z = 0.5052083333333333 code units.
`values` contains the three components of curl(B), indexed by intersected leaf,
two cell coordinates, and component. Multiply its vector magnitude by
`current_scale` to obtain current density in A m^-2. `valid` marks leaves with
available values. `lower` and `upper` are the three-dimensional domain bounds;
`bounds`, `spacing`, and `cell_edges` describe the original two-dimensional
leaf sections in code lengths. `levels`, `leaf_ids`, `block_shape`, and
`mesh_block_shape` retain the AMR identities and cell counts. The slice was
exported using native exact-phase preparation and curl; no uniform resampling
was applied to these arrays.

## Figure 4 — recorded performance comparison

`results/<representation>-endpoints-w<workers>.json` stores the measurements
for Native AMR (`amr`), Uniform L5 (`l5`), and Uniform L6 (`l6`) with 1, 4, 8,
16, and 32 workers. Each record retains the snapshot SHA-256, tracing controls,
source-file SHA-256 identifiers, software versions, setup costs, and individual
calls. No field arrays or original snapshot are required to draw this figure.

Calls with `repeat = -1` are warm-ups and are excluded. The renderer uses the
median of measured `wall_seconds`, minimum/maximum error bars, and the maximum
`rss_peak_observed_call` divided by 2^30 for process RSS in GiB. Speedup divides
the one-worker median by the median at each worker count. The right panel shows
32-worker wall time and RSS on separate axes. Setup time is not added to the
plotted call time; process RSS includes data already resident before the call.
These are retained measurements, not performance predictions for the reader's
machine. Redrawing does not execute a benchmark.

## Figure 5 — representation comparison

`profiles.npz` and `magnetic-profiles.npz` share `s_Mm` and `points_code`:
physical distance along the probe in Mm and Cartesian query positions in code
lengths. Arrays ending in `_uA_m2` are full three-dimensional current-density
magnitudes in microampere m^-2. Arrays ending in `_B_G` contain `(Bx, By, Bz)` in
gauss. Prefixes identify Native AMR, VTK PointData, Uniform L5, and Uniform L6.
`native_levels` labels the sample owners; `native_block_cuts_Mm` locates crossed
block boundaries.

`summary.json` retains probe endpoints, refinement-interface crossings, unit
factors, validity checks, historical version strings, and a SHA-256 identifier
for the source snapshot. `magnetic-summary.json` records the magnetic-unit
conversion and checks. These version strings do not identify an exact simesh
commit for every figure in the paper.

`schematic-geometry.npz` stores a JSON structure in `structure_json`; objects of
the form `{"array": "array_000"}` reference other arrays in the archive. Geometry
keys ending in `_Mm` are in Mm, and current values ending in `_microA_m2` are in
microampere m^-2. Other members identify interface faces, block boundaries,
contours, and highlighted samples. `schematic-summary.json` records the view.

The rendering uses a continuous piecewise distance map giving the L4, L5, and
L6 intervals equal display widths. Axis ticks remain physical distances. The
Native AMR curve is a comparison reference, not an analytic ground truth.

## Figure 6 — current-sheet diagnostics

`sampling.npz:rectangles` lists adaptive sample rectangles in the y-z plane,
with columns `(y_min, y_max, z_min, z_max)` in code lengths. In
`diagnostics.npz`, `log10_q` is dimensionless log10(Q), with the `valid` mask;
`twist` is dimensionless Tw, with the `complete` mask. Invalid entries are not
silently filled. The plotting function rasterizes each rectangle with its
saved value for display; this is not a new field interpolation.

`amr-slice.npz` contains the domain `lower`/`upper`, native-cell rectangles in
the same column order, and `ratio`: |J|/|B| in A m^-2 T^-1.

`lines.npz` retains 20 selected seeds, their two traced branches, branch
`offsets`, sample `positions`, original seed `ids`, and per-seed `twist`, `q`,
and branch `termination`. Positions and seeds are in code lengths. All saved
termination codes are 3 (DOMAIN_EXIT). `selection.json` records the percentile,
spatial selection, step size, and finite-difference Q method. The later
variational Q map in Figure 7 is a separate calculation.

## Figure 7 — filament magnetic topology

`maps.npz` contains final two-dimensional `q` (log10 Q) and `twist` (Tw), plus
their explicit validity masks. Q uses variational transport, while the retained
twist map comes from its original calculation. Seed values are placed at their
original pixels without interpolation; invalid pixels remain NaN. Axis order
is `(x, y)`, transposed only for image display.

`bottom.npz` contains `bz`, the bottom-plane magnetic component in code field
units, with `lower`, `upper`, `spacing`, and `z` in code lengths.
`lines.npz` contains retained line `positions` in code lengths, branch offsets,
line identifiers, and `current`, the raw |curl(B)| in code units (not SI current
density). Tube radius, lighting, and polygon interpolation are display settings.

`projection.npz` contains the unnormalized emission-proxy projection and the
three-dimensional domain bounds `lower_Mm` and `upper_Mm`. The projection is
the sum of saved emissivity voxels over z = 6–80 Mm multiplied by the voxel
depth in code lengths. Full closed-line averages were used when constructing
the emissivity; the height cut applies only to this projection. The three-
dimensional emissivity volume is not bundled. The renderer divides the
projection by its positive-value 99.5th percentile. These are relative proxy
units, not calibrated instrument intensities. `provenance.json` records the
definition, reference, coefficient, sampling, and validity count.

## Figure 8 — thermal diagnostics and absorbed synthetic emission

`diagnostics.npz` contains:

- `slice_values`: native y = 0 sections with components density [g cm^-3],
  temperature [K], speed [cm s^-1], and velocity x/y/z [cm s^-1], in that order.
  `slice_edges` holds native x/z edges in code lengths.
- `density`: line-integrated column mass [g cm^-2], despite its historical name.
- `em_column`: the line integral of electron density times hydrogen-nucleus
  density [cm^-5]. `em_temperature_column` adds the temperature factor
  [K cm^-5]; `em_weighted_temperature` is their ratio [K]. The two integrands
  were formed at native cell centers before support preparation and integration.
- `temperature_edges` [K], `em_hist_0`/`em_hist_1` (volume emission measure per
  bin [cm^-3]), and `em_0`/`em_1` (region totals [cm^-3]). The renderer divides
  bin weights by log10 temperature widths to show dEM/dlog10(T). This is a
  volume emission-measure distribution, not a line-of-sight DEM per kelvin.

The two region boxes are defined in the drawing script in code lengths.
Column projections span the full Y domain; they are not restricted to those
boxes. These image arrays use `(z, x)` order over x = [-100, 100] Mm and
z = [0, 160] Mm. Neither the emission-measure temperature projection nor the
regional distribution includes instrument responses or foreground absorption.

`bands.npz` retains twelve channels in the order given by `configuration.json`:

- `intensity_<wavelength>`: intensity after ordered line-of-sight transfer with
  H I, He I, and He II continuum absorption using LTE Saha populations.
- `thin_intensity_<wavelength>`: corresponding intensity without absorption.
- `optical_depth_<wavelength>`: total continuum optical depth.

The observer is at the lower Y boundary and the rays cross the full Y domain.
Emissivity and opacity are formed at native cell centers before exact-phase
support preparation and integration with eight sublayers. Each image uses
`(z, x)` order over x = [-200, 200] Mm and z = [0, 400] Mm. The plotted viewport
crops this extent without recalculating the projection. The adopted continuum
model has zero opacity at IRIS 1354 Angstrom. No external background, resonance
scattering, non-LTE line transfer, instrument blur, or noise is included.

`colormaps.npz` contains the exact 256-entry RGBA display lookup table for each
channel (components between 0 and 1). `plot-settings.json` retains the display
color limits. AIA and IRIS tables derive from SunPy 7.0.0; the EIS table uses
Matplotlib's `Blues_r`. The SunPy notice is retained in `LICENSE-SunPy.txt`.

`configuration.json` records the thermodynamic preset, fully ionized H/He
composition for the emission/EM diagnostics, internal-energy interpretation,
viewing geometry, absorption prescription, snapshot checksum, calculation-script
checksums, and emission-measure verification records. Absorber populations use
the separately stated LTE prescription. The coronal unit preset still requires
confirmation against the original simulation settings, and full response-table
calibration metadata are unavailable. Intensities retain the source responses'
numerical normalization; cross-instrument absolute intensity comparisons are
not supported by these inputs.

## Release identity and verification

`release.json` identifies the eight-figure manuscript numbering and the prior
bundle manifest used to assemble the unchanged products. `manifest.json`
records checksums, array names, shapes, and dtypes for every bundled file.
Figures 1 and 2 have no external inputs. All original data follow `LICENSE`
(GPL-3.0-only); color-table derivatives retain the SunPy notice. Exact scientific
calculation commits are not established for every historical product. The
release supports redraws from saved products, not a complete rerun from raw
simulation snapshots.
