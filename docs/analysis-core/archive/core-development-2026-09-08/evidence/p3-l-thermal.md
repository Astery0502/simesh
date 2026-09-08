# Historical AIA 171 Thermal Response: 2026-09-08

After worktree integration, raw artifacts referenced below are preserved under
`benchmark-results/analysis-core/euv-geometry/` in the main checkout.
The migration manifest records file sizes and SHA-256 checksums; measurements
below retain their original revision and scope.

## Disposition And Scope

**Adopt** the explicit historical AIA171 model and resident thermodynamic-first
LOS as the first physical-response delivery. Retain prepared-node emissivity
as an explicitly different reconstruction for repeated views. **Defer** current
observing-date calibration and physical validation of WENO until actual thermal
inputs, simulation normalization and suitable reference observations/calculation
are available. Missing T does not block this implementation or manufactured
verification. No backend, scheduling or runtime numerical cache was replaced.

Code: `src/simesh/analysis/thermal.py`, `_aia171_table.py`.
Checks: `tests/analysis/test_thermal.py`.
Raw evidence: ignored `benchmark-results/analysis-core/thermal.json`, `thermal.log`.

## Source, EOS And Units

The numerical table is `t_aia` / `f_171`, 101 nodes at log10(T/K)=4..9 with 0.05
spacing, from MPI-AMRVAC commit
[`574fc6e3f76be669bcb066fd2ea10accfa06ab90`](https://github.com/amrvac/amrvac/blob/574fc6e3f76be669bcb066fd2ea10accfa06ab90/src/physics/mod_thermal_emission.t#L111).
Its source hash is embedded with the copied data. The table is GPL-3.0 material,
compatible with this repository's license. No network lookup occurs at runtime.
The pinned source, not a channel-shaped Gaussian, supplies every response value.

`get_EUV` selects the table, interpolates log10(R) in log10(T), squares its density
variable, and applies the response. Its image path multiplies by physical length
(and divides projected-volume contributions by image area) to produce DN/s.
The chosen response unit is therefore **DN cm^5 s^-1 pixel^-1**, local emissivity
**DN s^-1 pixel^-1 cm^-1**, and LOS brightness **DN s^-1 pixel^-1**. This is an
instrument-pixel-normalized ray intensity; changing our sample raster spacing
does not multiply brightness by pixel area. No extra 4*pi, inverse-square distance,
exposure-time factor, PSF, absorption or aperture averaging is applied.
The table's original CHIANTI version, abundance file, effective-area calibration,
SSW switches and observing date are **not identified in the pinned source**.
These units/model conventions are checked against upstream operations; they do
not independently recover that missing calibration provenance.

A material normalization distinction was found by reading
[`mhd_physical_units`](https://github.com/amrvac/amrvac/blob/574fc6e3f76be669bcb066fd2ea10accfa06ab90/src/mhd/mod_mhd_phys.t#L1232):
with `eq_state_units=True`, fully ionized H/He and a=n_He/n_H,
`unit_density = (1+4a) m_p unit_numberdensity`. Thus upstream's variable called
Ne in the thermal routine is actually the hydrogen-density proxy for these units.
The first explicit physical model selects

- n_H = rho_cgs / [(1+4a)m_p], n_e = (1+2a)n_H, a=0.1;
- p_thermal = (2+3a)n_H k_B T, with equal ion/electron temperatures;
- epsilon = n_e^2 R(T), using the historical table under the declared electron
  emission-measure convention.

`AIA171("amrvac-hydrogen")` supplies n_H^2 R(T) compatibility with that upstream
unit choice. At fixed mass density, the electron model is **1.44 times** the
hydrogen proxy. This is an explicit modeling distinction, not a claim that
upstream's undocumented table-generation normalization has been repaired.
The EOS helium abundance does not regenerate the response's metal abundances.
The selected fully ionized coronal model is not a partial-ionization model at
the cool edge of the tabulated temperature range.

Inputs require positive finite T in K, nonnegative finite mass density and
positive finite density/length conversion factors. Thermal pressure may supply
T through the named ideal gas EOS; total energy is not guessed to be pressure.
Values outside the positive table domain return zero response. Exactly 1e9 K
uses the final table node: upstream's inclusive upper range check combined with
exclusive interval tests leaves a `logGT=0` endpoint artifact; that implementation
artifact is deliberately not reproduced. Invalid/unrepresentable inputs fail
explicitly. Every thermal image carries its temperature label, model and scales.

## Reconstruction And Independent Checks

Both orders start with **the same two-halo thermodynamic nodes**. The baseline
interpolates n and T and then computes n^2 R(T). The alternative computes epsilon
at those nodes and then interpolates the scalar. Neither is a volume average of
subcell thermodynamics. In particular, evaluating response after primary halo
preparation differs from exchanging/restricting interior emissivity. No claim of
commutation across an AMR interface is made.

Nonlinear response along a ray is not generally a cubic polynomial. Gauss2 is
therefore applied with explicit interval refinement after splitting thermodynamic
cell-center knots, without an exactness claim. Table knots can still lie inside
those intervals. Per-pixel sample limits and invalid/missing coverage return NaN
and a distinct status. Grazing intervals only one ulp wide keep their complete
quadrature weight; rounded query points are bound to that leaf's half-open box.

Focused checks cover table nodes/geometric means/endpoints, the independent H/He
EOS and 1.44 normalization factor, full mixed-AMR constant images with axis/oblique
depths, external K products with permuted leaf order, missing T, missing coverage,
sample-limit failure, and nonlinear manufactured continuum integration. The
continuum reference uses explicit tabular power laws and 262,144 independent
midpoint samples of analytic affine rho/T, not the production sampler or marcher.
Its maximum relative errors at subdivisions 1,4,16 were **2.29347e-4, 4.06332e-5,
7.23797e-7**; node-response-first error was **1.39477e-3**. These observations
select a refinement-capable method, not an observational error SLA.

## WENO Comparison

Reproduce from the isolated worktree:

```bash
PYTHONPATH=src:rewrite/src:scripts .venv/bin/python -m analysis_core.benchmark_thermal
PYTHONPATH=src:rewrite/src:scripts:tests/analysis:rewrite/benchmarks .venv/bin/python -m unittest discover -s tests/analysis -p test_thermal.py -v
```

Actual fixture: `data/weno509_sub_0000.dat`, 22,614 leaves, levels 3--6, 8^3 blocks.
Actual stored rho is combined with **manufactured** T=1.05e6+6e5 sin(2*pi*z_norm) K.
Density multiplier = 1.4 m_p 1e9 g/cm^3 per code density; length multiplier = 1e8
cm per code coordinate. These are declared demonstration scales, not recovered
snapshot units. Images cover the full projected box at 8x8 centers, along z and
(0.3,0.2,1). All pixels complete, including empty rays.

| Order / refinement | Axis relative L2 vs thermo/64 | Oblique relative L2 vs thermo/64 | Axis / oblique first seconds |
| --- | ---: | ---: | ---: |
| Node emissivity then scalar Gauss2 | 2.10667e-3 | 3.26784e-3 | 1.4513 / 1.3962 |
| Thermodynamics then response / 1 | 9.93180e-5 | 1.42142e-4 | 0.05338 / 0.06912 |
| Thermodynamics then response / 4 | 3.19128e-5 | 1.78841e-5 | 0.05876 / 0.07276 |
| Thermodynamics then response / 16 | 1.07650e-6 | 6.00119e-7 | 0.07388 / 0.09306 |
| Thermodynamics then response / 64 | reference | reference | 0.14860 / 0.17906 |

The /64 reference is a refined reconstruction integral, **not physical WENO truth**.
The one-shot emissivity times include whole-domain response materialization.
Separately building emissivity once took **1.50894 s**, retaining **312,977,760 B**.
With one warmup/three repeats, scalar LOS on that retained field took medians
**0.000335 / 0.000397 s** versus thermo/4 **0.05374 / 0.07169 s**. For this very
small raster, retained emissivity amortizes its construction after roughly
**29 axis or 22 oblique views**, if its different reconstruction error is acceptable.
This crossover is request-specific; it is not a universal backend speedup.
The convenience `order="emissivity-first"` call creates a field per call; explicit
`emissivity_fields` plus repeated scalar LOS is the retention interface.

Thermodynamic nodes occupy **625,593,696 B**. The admission bound while density,
external prepared T and the new state coexist was **1,518,339,216 B**. Peak measured
RSS in the comparison was **997,916,672 B**; mapped pages/OS cache are not controlled
array bytes. Input open/read/preparation and manufactured T/state costs remain
separate in JSON. No observed-temperature, current-calibration, PSF, million-pixel,
parallel nonlinear LOS or larger-than-RAM input claim is made.
