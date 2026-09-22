"""Analytic MHD recovery, physical normalization, AMR support and consumers."""

from dataclasses import replace
import gc
import weakref

import numpy as np
import pytest

import simesh as sm
from simesh import applications as app
from simesh.physics.mhd import IdealMHD, MHDStateError, MHDStatus, MHDUnits, mhd_fields
from simesh.physics.thermal import BOLTZMANN_ERG_K, PROTON_MASS_G


NAMES = ("rho", "m1", "m2", "m3", "e", "b1", "b2", "b3")
CGS = MHDUnits(density_g_cm3=1., momentum_g_cm2_s=1., energy_erg_cm3=1.,
               field_gauss=1., length_cm=1.)


def model(energy_kind="total", units=CGS):
    return IdealMHD(gamma=2., energy_kind=energy_kind,
                    composition=sm.CoronalComposition(helium_abundance=.1), units=units)


def source_for(state=None, *, configuration=None, mixed=False):
    configuration = model() if configuration is None else configuration
    mesh = sm.mesh_from_forest((2, 1, 1) if mixed else (1, 1, 1),
        np.array([False]+[True]*9 if mixed else [True]),
        lower=(0, 0, 0), upper=(2 if mixed else 1, 1, 1), block_shape=(8, 8, 8))
    local = np.indices(mesh.block_shape)+.5
    raw = np.empty((mesh.leaf_count, 8, *mesh.block_shape))
    u = configuration.units
    for leaf in range(mesh.leaf_count):
        x, y, z = mesh.bounds[leaf, 0, :, None, None, None]+local*mesh.spacing[leaf, :, None, None, None]
        rho, velocity, pressure, b = ((2., (3., 4., 0.), 6., (3., 4., 0.))
                                     if state is None else state(x, y, z))
        energy = pressure/(configuration.gamma-1)
        if configuration.energy_kind == "total":
            energy = energy + .5*rho*sum(v*v for v in velocity)
            energy = energy + sum(component*component for component in b)/(2*(4*np.pi))
        raw[leaf, 0] = rho/u.density_g_cm3
        raw[leaf, 1:4] = np.array([np.broadcast_to(rho*v/u.momentum_g_cm2_s, x.shape) for v in velocity])
        raw[leaf, 4] = energy/u.energy_erg_cm3
        raw[leaf, 5:8] = np.array([np.broadcast_to(component/u.field_gauss, x.shape) for component in b])
    return sm.source_from_arrays(mesh, raw, NAMES), raw


def column(fields, name):
    return next(i for i, definition in enumerate(fields.fields) if definition.name == name)


@pytest.mark.parametrize("energy_kind", ["total", "internal"])
def test_analytic_state_and_consistent_magnetic_diagnostics(energy_kind):
    config = model(energy_kind)
    with source_for(configuration=config)[0] as source:
        ready = sm.prepare(source, scheme="exact-phase")
    state = mhd_fields(ready, model=config, outputs=("density", "velocity", "speed", "internal_energy",
        "pressure", "temperature", "beta", "sound_speed", "alfven_speed", "sonic_mach", "alfven_mach", "status"))
    temperature = 6./((2+3*.1)*(2./((1+4*.1)*PROTON_MASS_G))*BOLTZMANN_ERG_K)
    expected = [2., 3., 4., 0., 5., 6., 6., temperature, 48*np.pi/25, np.sqrt(6.),
                5/np.sqrt(8*np.pi), 5/np.sqrt(6.), np.sqrt(8*np.pi), 0.]
    np.testing.assert_allclose(state.values, np.broadcast_to(expected, state.values.shape), rtol=2e-14)
    assert state.storage_halo == state.valid_halo == 2
    assert tuple(f.units for f in state.fields)[0:6] == ("g cm^-3", "cm s^-1", "cm s^-1", "cm s^-1", "cm s^-1", "erg cm^-3")
    assert not state.values.flags.writeable and not np.shares_memory(state.values, ready.values)
    assert state.preparation_stats["invalid_state_counts"] == {"interior": 0, "evaluated": 0}
    magnetic_pressure = sm.magnetic_pressure(ready, units=config.units.magnetic_si, components=("b1", "b2", "b3"))
    np.testing.assert_allclose(state.values[..., column(state, "beta")]*magnetic_pressure.values[..., 0]*10, 6.)


def test_energy_definition_is_explicit_and_does_not_follow_field_name():
    with source_for()[0] as source:
        raw = sm.read_fields(source)
    total = mhd_fields(raw, model=model(), outputs="pressure")
    internal = mhd_fields(raw, model=model("internal"), outputs="pressure")
    np.testing.assert_allclose(total.values, 6.)
    np.testing.assert_allclose(internal.values, 31.+25/(8*np.pi))
    assert raw.fields[4].name == "e"
    with pytest.raises(TypeError):
        mhd_fields(raw)
    with pytest.raises(ValueError, match="energy_kind"):
        model("hydrodynamic")


@pytest.mark.parametrize("energy_kind", ["total", "internal"])
def test_independent_si_cgs_and_code_units(energy_kind):
    configurations = [model(energy_kind), model(energy_kind, MHDUnits(
        density_g_cm3=1e-3, momentum_g_cm2_s=.1, energy_erg_cm3=10.,
        field_gauss=1e4, length_cm=100.)),
        model(energy_kind, MHDUnits(density_g_cm3=.25, momentum_g_cm2_s=7., energy_erg_cm3=11.,
            field_gauss=.3, length_cm=9.))]
    results = []
    for config in configurations:
        with source_for(configuration=config)[0] as source:
            # Explicit factors control the conversion even with opaque labels.
            raw = sm.read_fields(source, fields=NAMES[::-1])
        results.append(mhd_fields(raw, model=config))
    for result in results[1:]:
        np.testing.assert_allclose(result.values, results[0].values, rtol=2e-14)


@pytest.mark.parametrize("energy_kind", ["total", "internal"])
@pytest.mark.parametrize("custom", [False, True])
def test_solar_units_recover_amrvac_dimensionless_conserved_state(energy_kind, custom):
    composition = sm.CoronalComposition(helium_abundance=.2 if custom else .1)
    temperature, number_density = (2e6, 3e8) if custom else (1e6, 1e9)
    overrides = dict(length_cm=5e8, number_density_cm3=number_density,
                     temperature_k=temperature) if custom else {}
    units = sm.MHDUnits.solar(composition=composition, **overrides)
    config = IdealMHD(gamma=5/3, energy_kind=energy_kind, composition=composition, units=units)
    # AMRVAC code units: rho=2, m=(3,4,0), p=6, B=(1,2,2).
    # E=p/(gamma-1)+|m|^2/(2*rho)+|B|^2/2, with no explicit 4*pi.
    energy = 6/(config.gamma-1)
    if energy_kind == "total":
        energy += 25/4 + 9/2
    mesh = sm.mesh_from_forest((1, 1, 1), np.array([True]), lower=(0, 0, 0),
                               upper=(1, 1, 1), block_shape=(8, 8, 8))
    raw = np.broadcast_to(np.array([2., 3., 4., 0., energy, 1., 2., 2.])[None, :, None, None, None],
                          (1, 8, 8, 8, 8)).copy()
    with sm.source_from_arrays(mesh, raw, NAMES) as source:
        conserved = sm.read_fields(source)
    state = sm.mhd_fields(conserved, model=config)
    np.testing.assert_allclose(state.values[..., column(state, "temperature")], 3*temperature)
    np.testing.assert_allclose(composition.number_density(
        state.values[..., column(state, "density")], convention="amrvac-hydrogen"), 2*number_density)
    np.testing.assert_allclose(state.values[..., column(state, "vx")], 1.5*units.velocity_cm_s)
    np.testing.assert_allclose(state.values[..., column(state, "pressure")], 6*units.energy_erg_cm3)
    np.testing.assert_allclose(state.values[..., column(state, "beta")], 4/3)
    np.testing.assert_allclose(state.values[..., column(state, "alfven_speed")],
                               3/np.sqrt(2)*units.velocity_cm_s)
    np.testing.assert_array_equal(state.values[..., column(state, "mhd_status")], 0.)
    magnetic_pressure = sm.magnetic_pressure(conserved, components=(5, 6, 7), units=units.magnetic_si)
    np.testing.assert_allclose(magnetic_pressure.values[..., 0]*10,
                               state.values[..., column(state, "pressure")]/(4/3))


def test_solar_units_reject_invalid_scales_and_unrepresentable_normalization():
    for name in ("length_cm", "number_density_cm3", "temperature_k"):
        for value in (0., -1., np.inf, np.nan, True, [1.]):
            with pytest.raises(ValueError, match="solar scales"):
                MHDUnits.solar(**{name: value})
    with pytest.raises(TypeError, match="composition"):
        MHDUnits.solar(composition=None)
    with pytest.raises(ValueError, match="solar density and pressure"):
        MHDUnits.solar(composition=sm.CoronalComposition(helium_abundance=1e308))


def test_mixed_amr_separate_groups_permutations_and_invalid_padding():
    state = lambda x, y, z: (2+x, (3., 4., 0.), 6+2*x, (3., 4., 0.))
    with source_for(state, mixed=True)[0] as source:
        order = np.arange(source.mesh.leaf_count)[::-1]
        conserved = sm.prepare(source, NAMES[:5], leaf_ids=order, scheme="coordinate-phase")
        magnetic = sm.prepare(source, NAMES[5:], leaf_ids=np.roll(order, 2), scheme="exact-phase")
        raw_b = sm.read_fields(source, NAMES[5:], leaf_ids=order)
    # Coordinate preparation retains physical SFC slots despite selection order.
    assert not np.array_equal(conserved.slot_of_leaf[order], np.arange(len(order)))
    storage = np.full((len(order), 14, 14, 14, 5), np.nan)
    storage[:, 1:-1, 1:-1, 1:-1] = conserved.values
    padded = replace(conserved, _values=storage, storage_halo=3)
    output = mhd_fields(padded, magnetic=magnetic, model=model())
    np.testing.assert_array_equal(output.leaf_ids, order)
    np.testing.assert_array_equal(output.slot_of_leaf[order], np.arange(len(order)))
    assert output.valid_halo == output.storage_halo == 2
    for leaf in order:
        window = output.window(leaf, (0, 0, 0), (8, 8, 8))
        x = source.mesh.bounds[leaf, 0, 0]+(np.indices((8, 8, 8))[0]+.5)*source.mesh.spacing[leaf, 0]
        np.testing.assert_allclose(window[..., column(output, "density")], 2+x, rtol=2e-14)
        np.testing.assert_allclose(window[..., column(output, "pressure")], 6+2*x, rtol=2e-14)
    interior = mhd_fields(padded, magnetic=raw_b, model=model())
    assert interior.valid_halo == interior.storage_halo == 0
    np.testing.assert_allclose(interior.values, output.interior(), rtol=2e-14)
    shortened = mhd_fields(padded, magnetic=replace(magnetic, valid_halo=1), model=model())
    assert shortened.valid_halo == shortened.storage_halo == 1
    np.testing.assert_allclose(shortened.values, output.values[:, 1:-1, 1:-1, 1:-1], rtol=2e-14)


def test_partial_coverage_requested_bounds_and_mesh_mismatch():
    with source_for(mixed=True)[0] as source:
        bounds = np.array([[.05, .05, .05], [.1, .1, .1]])
        partial = sm.prepare(source, region=bounds, scheme="exact-phase")
        all_leaves = sm.read_fields(source)
    result = mhd_fields(partial, model=model(), outputs="temperature")
    assert result.selection is partial.selection
    np.testing.assert_array_equal(result.selection.requested_bounds, bounds)
    assert len(result.leaf_ids) == 1
    _, owners, valid = sm.sample(result, np.array([[.1, .1, .1], [1.5, .5, .5]]))
    np.testing.assert_array_equal(valid, [True, False])
    assert owners[1] >= 0
    with pytest.raises(ValueError, match="coverage"):
        mhd_fields(partial, magnetic=all_leaves, model=model())
    with source_for(mixed=True)[0] as other:
        other_mesh = sm.read_fields(other)
    with pytest.raises(ValueError, match="Mesh"):
        mhd_fields(all_leaves, magnetic=other_mesh, model=model())


@pytest.mark.parametrize("component,value,flag", [(0, 0., MHDStatus.NONPOSITIVE_DENSITY),
    (0, -1., MHDStatus.NONPOSITIVE_DENSITY), (4, 25., MHDStatus.NONPOSITIVE_INTERNAL_ENERGY),
    (4, 24., MHDStatus.NONPOSITIVE_INTERNAL_ENERGY), (2, np.nan, MHDStatus.NONFINITE_INPUT),
    (6, np.inf, MHDStatus.NONFINITE_INPUT), (0, np.nan, MHDStatus.NONFINITE_INPUT),
    (4, np.inf, MHDStatus.NONFINITE_INPUT)])
def test_invalid_nodes_raise_or_remain_visible_as_nan(component, value, flag):
    with source_for()[0] as source:
        ready = sm.prepare(source, scheme="exact-phase")
    changed = ready.values.copy()
    changed[0, 3, 4, 5, component] = value
    ready = replace(ready, _values=changed)
    with pytest.raises(MHDStateError) as error:
        mhd_fields(ready, model=model())
    assert error.value.leaf_id == 0 and error.value.cell_index == (1, 2, 3)
    assert error.value.status & flag
    output = mhd_fields(ready, model=model(), invalid="nan")
    assert np.isnan(output.values[0, 3, 4, 5, :-1]).all()
    assert int(output.values[0, 3, 4, 5, -1]) & flag
    assert np.isfinite(output.values[0, 4, 4, 5]).all()
    assert output.preparation_stats["invalid_state_counts"] == {"interior": 1, "evaluated": 1}
    assert output.valid_halo == 2  # Spatial support is distinct from physical validity.
    with pytest.raises(ValueError, match="temperature|density"):
        sm.thermal_fields(ready, output, density_unit_g_cm3=1.,
            temperature_component=column(output, "temperature"), temperature_label="invalid-state test")


def test_invalid_halo_is_checked_and_counted_separately():
    with source_for()[0] as source:
        ready = sm.prepare(source, scheme="exact-phase")
    changed = ready.values.copy()
    changed[0, 0, 4, 5, 0] = 0.
    ready = replace(ready, _values=changed)
    with pytest.raises(MHDStateError) as error:
        mhd_fields(ready, model=model())
    assert error.value.cell_index == (-2, 2, 3)
    output = mhd_fields(ready, model=model(), invalid="nan", outputs="status")
    assert output.preparation_stats["invalid_state_counts"] == {"interior": 0, "evaluated": 1}
    # Invalid outer storage stops participating when it is not valid support.
    inner = mhd_fields(replace(ready, valid_halo=1), model=model())
    assert np.isfinite(inner.values).all()


def test_zero_magnetic_field_is_a_valid_state_with_undefined_ratios():
    with source_for(lambda x, y, z: (2., (0., 0., 0.), 6., (0., 0., 0.)))[0] as source:
        raw = sm.read_fields(source)
    output = mhd_fields(raw, model=model())
    for name in ("beta", "alfven_mach"):
        assert np.isnan(output.values[..., column(output, name)]).all()
    for name in ("alfven_speed", "sonic_mach"):
        np.testing.assert_array_equal(output.values[..., column(output, name)], 0.)
    np.testing.assert_array_equal(output.values[..., -1], int(MHDStatus.ZERO_MAGNETIC_FIELD))
    assert output.preparation_stats["invalid_state_counts"]["evaluated"] == 0


def test_nonfinite_normalization_and_diagnostic_overflow_are_visible():
    with source_for()[0] as source:
        raw = sm.read_fields(source)
    overflow = replace(CGS, energy_erg_cm3=1e308)
    output = mhd_fields(raw, model=model(units=overflow), invalid="nan")
    assert np.isnan(output.values[..., :-1]).all()
    np.testing.assert_array_equal(output.values[..., -1], int(MHDStatus.UNREPRESENTABLE_STATE))
    tiny_b = raw.values.copy()
    tiny_b[..., 5:8] = 1e-200
    output = mhd_fields(replace(raw, _values=tiny_b), model=model("internal"))
    assert np.isnan(output.values[..., column(output, "beta")]).all()
    assert np.isfinite(output.values[..., column(output, "temperature")]).all()
    np.testing.assert_array_equal(output.values[..., -1], int(MHDStatus.UNREPRESENTABLE_DIAGNOSTIC))


def test_finite_cgs_magnetic_energy_does_not_overflow_intermediate_square():
    with source_for(configuration=model("internal"))[0] as source:
        raw = sm.read_fields(source)
    changed = raw.values.copy()
    # B**2 overflows, while B**2/(8*pi) and the complete state are finite.
    b = 1.5e154
    magnetic_energy = (b/np.sqrt(8*np.pi))**2
    changed[..., 0] = 1e284
    changed[..., 1:4] = 0.
    changed[..., 4] = 1e307 + magnetic_energy
    changed[..., 5:8] = [b, 0., 0.]
    changed_fields = replace(raw, _values=changed)
    output = mhd_fields(changed_fields, model=model())
    np.testing.assert_allclose(output.values[..., column(output, "pressure")], 1e307)
    np.testing.assert_allclose(output.values[..., column(output, "beta")], 1e307/magnetic_energy)
    np.testing.assert_array_equal(output.values[..., -1], 0.)
    magnetic_pressure = sm.magnetic_pressure(changed_fields, components=(5,6,7), units=CGS.magnetic_si)
    np.testing.assert_allclose(magnetic_pressure.values[..., 0]*10, magnetic_energy)


def test_custom_field_names_composition_and_internal_energy_positivity():
    with source_for(configuration=model("internal"))[0] as source:
        raw = sm.read_fields(source)
    renamed = replace(raw, fields=tuple(sm.FieldDefinition(name, "opaque") for name in
        ("mass", "mx", "my", "mz", "internal", "Bx", "By", "Bz")))
    config = replace(model("internal"), composition=sm.CoronalComposition(helium_abundance=0.))
    output = mhd_fields(renamed, model=config, density="mass", momentum=("mx", "my", "mz"),
        energy="internal", magnetic_components=("Bx", "By", "Bz"), outputs="temperature")
    expected = 6./(2*(2./PROTON_MASS_G)*BOLTZMANN_ERG_K)
    np.testing.assert_allclose(output.values, expected, rtol=2e-14)
    for internal in (0., -1.):
        changed = raw.values.copy()
        changed[..., 4] = internal
        with pytest.raises(MHDStateError) as error:
            mhd_fields(replace(raw, _values=changed), model=model("internal"))
        assert error.value.status == MHDStatus.NONPOSITIVE_INTERNAL_ENERGY


def test_extreme_composition_conversion_and_empty_coverage():
    with source_for()[0] as source:
        raw = sm.read_fields(source)
        empty = sm.read_fields(source, leaf_ids=[])
    config = replace(model(), composition=sm.CoronalComposition(helium_abundance=1e308))
    result = mhd_fields(raw, model=config, invalid="nan")
    assert np.isnan(result.values[..., :-1]).all()
    np.testing.assert_array_equal(result.values[..., -1], int(MHDStatus.UNREPRESENTABLE_STATE))
    result = mhd_fields(empty, model=model())
    assert result.values.shape[0] == 0 and result.valid_halo == 0
    assert result.preparation_stats["invalid_state_counts"] == {"interior": 0, "evaluated": 0}


def test_selectors_validation_and_memory_admission():
    with source_for()[0] as source:
        raw = sm.read_fields(source)
    numbered = mhd_fields(raw, model=model(), density=0, momentum=(1, 2, 3), energy=4,
                          magnetic_components=(5, 6, 7))
    np.testing.assert_array_equal(numbered.values, mhd_fields(raw, model=model()).values)
    for kwargs in ({"outputs": ()}, {"outputs": ("pressure", "pressure")}, {"outputs": "e"},
                   {"density": "missing"}, {"momentum": (1, 2)}, {"momentum": (1, 1, 1)},
                   {"density": -1}, {"energy": "rho"}, {"magnetic_components": (1, 6, 7)},
                   {"invalid": "clip"}):
        with pytest.raises(ValueError):
            mhd_fields(raw, model=model(), **kwargs)
    with pytest.raises(TypeError, match="IdealMHD"):
        mhd_fields(raw, model="mhd")
    with pytest.raises(MemoryError):
        mhd_fields(raw, model=model(), memory_limit=1)
    needed = numbered.preparation_stats["controlled_upper_bytes"]
    mhd_fields(raw, model=model(), memory_limit=needed)
    ambiguous = replace(raw, fields=(*raw.fields[:-1], raw.fields[0]))
    with pytest.raises(ValueError, match="ambiguous"):
        mhd_fields(ambiguous, model=model())
    for gamma in (1., .5, np.inf, np.nan, True):
        with pytest.raises(ValueError, match="gamma"):
            replace(model(), gamma=gamma)
    for name in ("density_g_cm3", "momentum_g_cm2_s", "energy_erg_cm3", "field_gauss", "length_cm"):
        for factor in (0., -1., np.inf, np.nan, True):
            with pytest.raises(ValueError):
                replace(CGS, **{name: factor})
    with pytest.raises(TypeError):
        replace(model(), composition=None)
    with pytest.raises(TypeError):
        replace(model(), units="CGS")


def test_borrowed_inputs_detach_and_expired_inputs_fail():
    with source_for(mixed=True)[0] as source:
        batches = sm.iter_prepared(source, scheme="exact-phase", batch_size=1)
        borrowed = next(batches)
        result = mhd_fields(borrowed, model=model(), outputs="velocity")
        expected = result.values.copy()
        next(batches)
        with pytest.raises(RuntimeError, match="expired"):
            mhd_fields(borrowed, model=model())
        batches.close()
    np.testing.assert_array_equal(result.values, expected)
    with source_for()[0] as source:
        raw = sm.read_fields(source)
    reference = weakref.ref(raw.values)
    detached = mhd_fields(raw, model=model())
    del raw
    gc.collect()
    assert reference() is None and np.isfinite(detached.values).all()


def test_recovered_temperature_thermal_los_and_velocity_tracing():
    units = MHDUnits.solar()
    config = replace(model(units=units), gamma=5/3)
    rho, target_t = 2e-15, 8e5
    a = config.composition.helium_abundance
    pressure = (2+3*a)*(rho/((1+4*a)*PROTON_MASS_G))*BOLTZMANN_ERG_K*target_t
    state = lambda x, y, z: (rho, (0., 0., 2e6), pressure, (0., 0., 1.))
    with source_for(state, configuration=config, mixed=True)[0] as source:
        ready = sm.prepare(source, scheme="exact-phase")
    recovered = mhd_fields(ready, model=config, outputs=("density", "temperature"))
    np.testing.assert_allclose(recovered.values[..., 1], target_t, rtol=2e-14)
    response = sm.AIA171(composition=config.composition)
    thermo = sm.thermal_fields(recovered, recovered, density_unit_g_cm3=1.,
        temperature_component=1, temperature_label="explicit ideal-MHD total energy", model=response)
    plane = sm.orthographic_plane(ready.mesh.lower, ready.mesh.upper, [0, 0, 1], (6, 5))
    length_cm = units.length_cm
    expected_emissivity = response.emissivity(rho, target_t)
    for order in ("thermodynamics-first", "emissivity-first"):
        image = sm.integrate_thermal_los(thermo, plane, [0, 0, 1], length_unit_cm=length_cm,
                                        model=response, order=order, workers=2)
        assert image.complete
        np.testing.assert_allclose(image.values, expected_emissivity*image.depth*length_cm, rtol=2e-13)
    velocity = mhd_fields(ready, model=config, outputs="velocity")
    assert tuple(f.name for f in velocity.fields) == ("vx", "vy", "vz")
    lines = app.trace(velocity, sm.PointSet([[.25, .25, .5], [1.5, .5, .5]]), step=.025, step_fraction=None, max_steps=8)
    for seed_id, seed in zip(lines.seeds.ids, lines.seeds.positions):
        for direction in (-1, 1):
            branch = lines.branch(seed_id, direction)
            assert len(branch) == 9
            expected = seed + np.arange(9)[:, None]*[0., 0., direction*.025]
            np.testing.assert_allclose(branch, expected, atol=1e-14)
