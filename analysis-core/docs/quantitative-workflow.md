# Integrated quantitative analysis

The reduction, MHD recovery and application-file features share the existing
Fields and geometry/result contracts. Their public functions are available
through `import simesh as sm` or the focused `simesh.reductions`,
`simesh.physics.mhd` and `simesh.results_io` modules.

| Need | Entry points |
| --- | --- |
| Interior integrals and statistics | `volume_integral`, `weighted_mean`, `extrema`, `histogram` |
| Rectangular surface flux | `surface_flux`, `AxisAlignedSurface` |
| Explicit coordinate measure conversion | `LengthUnits` |
| Ideal-MHD physical recovery | `MHDUnits`, `IdealMHD`, `mhd_fields`, `MHDStatus` |
| Save and recover maps, rays and compact paths | `save_result`, `load_result` |

Use `read_fields` when only interior statistics are needed. Preparation is
necessary when subsequent consumers require interpolation or derivative support.
MHD recovery preserves the available coverage and halo; it does not construct
new support. Choose only physical outputs for interpolated maps or thermal LOS;
the optional MHD status field is categorical and must not be interpolated.

```python
recovered = sm.mhd_fields(interiors, model=model, outputs=("density", "temperature"))
lengths = sm.LengthUnits(model.units.magnetic.length_m, "m")
mass = sm.volume_integral(recovered, "density", units=lengths)
temperature = sm.weighted_mean(
    recovered, "temperature", weights=recovered,
    weight_component="density", units=lengths,
)

sm.save_result(
    "thermal.result.npz", thermal_image,
    metadata={"mass": mass.value, "mass_units": mass.units,
              "mean_temperature_K": temperature.value},
    source={"description": "explicit caller-provided dataset description"},
)
restored = sm.load_result("thermal.result.npz")
```

Reductions integrate the piecewise-constant supplied interiors. A derived
temperature or energy remains a pointwise recovery of those inputs; integrating
it does not reconstruct unresolved subcell structure. Coordinate volume/area
conversion is explicit and independent of the field's unit label.

The result-file format supports its documented application objects, not raw
Fields or arbitrary reduction dataclasses. Scalar summaries can be supplied as
JSON metadata; histogram arrays require explicit conversion if included there.
Native full Fields retain their existing AMRVAC export interface. Source identity
is not verified on file load; parameters not retained in the result object must
be supplied as metadata.

## Complete example

```bash
cd analysis-core
.venv/bin/python examples/recovered_state_analysis.py --output /tmp/simesh-integrated
```

This example is independent of plotting packages. It constructs a mixed-level
analytic state, recovers physical fields, computes mass/temperature/flux,
produces thermal LOS and instantaneous velocity streamlines, and verifies
saved/reloaded arrays and identifiers. Its analytic checks are mass
`2.4e-12 kg`, mass-weighted temperature `840000 K`, and outward bottom flux
`-2e-4 T*m^2` under its explicit SI configuration.

See the individual guides for partial-cell weights, one-sided surface values,
invalid-state policies, unit conversions, supported energy definitions and file
validation. Time-series processing and the previously deferred core/application
extensions are not part of this integration.
