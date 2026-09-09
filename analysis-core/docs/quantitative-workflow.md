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
| Select, rename and combine fields | `select_fields`, `merge_fields`, `derive_many` |
| Sample quantities on stored curves | `sample_line_profiles`, `LineProfiles`, `LineProfile` |

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

After preparing a state and tracing or loading a `LineSet`, compose the desired
quantities and sample them together. Mixed unit labels are retained per column:

```python
state = sm.mhd_fields(prepared, model=model, outputs=("density", "temperature"))
scaled = sm.derive_many(
    state, {"temperature_MK": "MK", "density_cgs": "g cm^-3"},
    lambda ctx: {"temperature_MK": ctx.field("temperature")*1e-6,
                 "density_cgs": ctx.field("density")*1e-3},
)
magnetic = sm.select_fields(prepared, ("b1", "b2", "b3"))
current = sm.current_density(magnetic, units=model.units.magnetic)
quantities = sm.merge_fields((scaled, current))
profiles = sm.sample_line_profiles(
    quantities, lines, ("temperature_MK", "density_cgs", "jz"),
    length_units=sm.LengthUnits(model.units.magnetic.length_m, "m"),
)
branch = profiles.branch(lines.seeds.ids[0], -1)
```

Branch distances measure the stored polyline. Sampling validity is independent
of trace completion; inspect `usable` per component and the branch termination
statuses. A profile can consume restored geometry, whose original source is
unverified. The caller establishes coordinate and time compatibility.
Profiles themselves are not supported by `save_result`.

Composition owns compact copies. Avoid retaining a merged full-domain group
when the next recipe can directly consume named input groups. Profile sampling
batches bound temporary work, while the complete output remains in memory.

```bash
cd analysis-core
.venv/bin/python examples/recovered_state_analysis.py --output /tmp/simesh-integrated
```

This example is independent of plotting packages. It constructs a mixed-level
analytic state, recovers physical fields, computes mass/temperature/flux,
produces thermal LOS and instantaneous velocity streamlines, samples composed
temperature/density/current fields along those curves, and verifies
saved/reloaded arrays and identifiers. Its analytic checks are mass
`2.4e-12 kg`, mass-weighted temperature `840000 K`, and outward bottom flux
`-2e-4 T*m^2` under its explicit SI configuration.

See the individual guides for partial-cell weights, one-sided surface values,
invalid-state policies, unit conversions, supported energy definitions and file
validation. Time-series processing and the previously deferred core/application
extensions are not part of this integration.
