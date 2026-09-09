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
SI = MHDUnits(density_kg_m3=1., momentum_kg_m2_s=1., energy_j_m3=1.,
              magnetic=sm.MagneticUnits(field_tesla=1., length_m=1., permeability_h_m=5.))


def model(energy_kind="total", units=SI):
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
            energy = energy + sum(component*component for component in b)/(2*u.magnetic.permeability_h_m)
        raw[leaf, 0] = rho/u.density_kg_m3
        raw[leaf, 1:4] = np.array([np.broadcast_to(rho*v/u.momentum_kg_m2_s, x.shape) for v in velocity])
        raw[leaf, 4] = energy/u.energy_j_m3
        raw[leaf, 5:8] = np.array([np.broadcast_to(component/u.magnetic.field_tesla, x.shape) for component in b])
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
    temperature = 60./((2+3*.1)*(.002/((1+4*.1)*PROTON_MASS_G))*BOLTZMANN_ERG_K)
    expected = [2., 3., 4., 0., 5., 6., 6., temperature, 2.4, np.sqrt(6.),
                np.sqrt(2.5), 5/np.sqrt(6.), np.sqrt(10.), 0.]
    np.testing.assert_allclose(state.values, np.broadcast_to(expected, state.values.shape), rtol=2e-14)
    assert state.storage_halo == state.valid_halo == 2
    assert tuple(f.units for f in state.fields)[0:6] == ("kg m^-3", "m s^-1", "m s^-1", "m s^-1", "m s^-1", "J m^-3")
    assert not state.values.flags.writeable and not np.shares_memory(state.values, ready.values)
    assert state.preparation_stats["invalid_state_counts"] == {"interior": 0, "evaluated": 0}
    magnetic_pressure = sm.magnetic_pressure(ready, units=config.units.magnetic, components=("b1", "b2", "b3"))
    np.testing.assert_allclose(state.values[..., column(state, "beta")]*magnetic_pressure.values[..., 0], 6.)


def test_energy_definition_is_explicit_and_does_not_follow_field_name():
    with source_for()[0] as source:
        raw = sm.read_fields(source)
    total = mhd_fields(raw, model=model(), outputs="pressure")
    internal = mhd_fields(raw, model=model("internal"), outputs="pressure")
    np.testing.assert_allclose(total.values, 6.)
    np.testing.assert_allclose(internal.values, 33.5)
    assert raw.fields[4].name == "e"
    with pytest.raises(TypeError):
        mhd_fields(raw)
    with pytest.raises(ValueError, match="energy_kind"):
        model("hydrodynamic")


@pytest.mark.parametrize("energy_kind", ["total", "internal"])
def test_independent_si_cgs_and_code_units(energy_kind):
    configurations = [model(energy_kind), model(energy_kind, MHDUnits(
        density_kg_m3=1e3, momentum_kg_m2_s=10., energy_j_m3=.1,
        magnetic=sm.MagneticUnits(field_tesla=1e-4, length_m=.01, permeability_h_m=5.))),
        model(energy_kind, MHDUnits(density_kg_m3=.25, momentum_kg_m2_s=7., energy_j_m3=11.,
            magnetic=sm.MagneticUnits(field_tesla=.3, length_m=9., permeability_h_m=5.)))]
    results = []
    for config in configurations:
        with source_for(configuration=config)[0] as source:
            # Explicit factors control the conversion even with opaque labels.
            raw = sm.read_fields(source, fields=NAMES[::-1])
        results.append(mhd_fields(raw, model=config))
    for result in results[1:]:
        np.testing.assert_allclose(result.values, results[0].values, rtol=2e-14)


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
    (0, -1., MHDStatus.NONPOSITIVE_DENSITY), (4, 27.5, MHDStatus.NONPOSITIVE_INTERNAL_ENERGY),
    (4, 26., MHDStatus.NONPOSITIVE_INTERNAL_ENERGY), (2, np.nan, MHDStatus.NONFINITE_INPUT),
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
        sm.thermal_fields(ready, output, density_unit_g_cm3=.001,
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
    overflow = replace(SI, energy_j_m3=1e308)
    output = mhd_fields(raw, model=model(units=overflow), invalid="nan")
    assert np.isnan(output.values[..., :-1]).all()
    np.testing.assert_array_equal(output.values[..., -1], int(MHDStatus.UNREPRESENTABLE_STATE))
    tiny_b = raw.values.copy()
    tiny_b[..., 5:8] = 1e-200
    output = mhd_fields(replace(raw, _values=tiny_b), model=model("internal"))
    assert np.isnan(output.values[..., column(output, "beta")]).all()
    assert np.isfinite(output.values[..., column(output, "temperature")]).all()
    np.testing.assert_array_equal(output.values[..., -1], int(MHDStatus.UNREPRESENTABLE_DIAGNOSTIC))


def test_finite_magnetic_energy_does_not_overflow_intermediate_permeability():
    units = replace(SI, magnetic=sm.MagneticUnits(field_tesla=1., length_m=1., permeability_h_m=1e308))
    config = model(units=units)
    # E = u + K + B^2/(2 mu) = 6 + 25 + .5. Squaring B or doubling
    # permeability first can overflow even though the recovered state is finite.
    with source_for(configuration=model("internal", units))[0] as source:
        raw = sm.read_fields(source)
    changed = raw.values.copy()
    changed[..., 4] = 31.5
    changed[..., 5:8] = [1e154, 0., 0.]
    changed_fields = replace(raw, _values=changed)
    output = mhd_fields(changed_fields, model=config)
    np.testing.assert_allclose(output.values[..., column(output, "pressure")], 6.)
    np.testing.assert_allclose(output.values[..., column(output, "beta")], 12.)
    np.testing.assert_array_equal(output.values[..., -1], 0.)
    magnetic_pressure = sm.magnetic_pressure(changed_fields,components=(5,6,7),units=units.magnetic)
    energy_density = sm.magnetic_energy_density(changed_fields,components=(5,6,7),units=units.magnetic)
    np.testing.assert_allclose(magnetic_pressure.values[...,0],.5)
    np.testing.assert_array_equal(magnetic_pressure.values,energy_density.values)
    np.testing.assert_allclose(output.values[...,column(output,"beta")]*magnetic_pressure.values[...,0],
                               output.values[...,column(output,"pressure")])


def test_custom_field_names_composition_and_internal_energy_positivity():
    with source_for(configuration=model("internal"))[0] as source:
        raw = sm.read_fields(source)
    renamed = replace(raw, fields=tuple(sm.FieldDefinition(name, "opaque") for name in
        ("mass", "mx", "my", "mz", "internal", "Bx", "By", "Bz")))
    config = replace(model("internal"), composition=sm.CoronalComposition(helium_abundance=0.))
    output = mhd_fields(renamed, model=config, density="mass", momentum=("mx", "my", "mz"),
        energy="internal", magnetic_components=("Bx", "By", "Bz"), outputs="temperature")
    expected = 60./(2*(.002/PROTON_MASS_G)*BOLTZMANN_ERG_K)
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
    for name in ("density_kg_m3", "momentum_kg_m2_s", "energy_j_m3"):
        for factor in (0., -1., np.inf, np.nan, True):
            with pytest.raises(ValueError):
                replace(SI, **{name: factor})
    with pytest.raises(TypeError):
        replace(model(), composition=None)
    with pytest.raises(TypeError):
        replace(model(), units="SI")


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
    units = MHDUnits(density_kg_m3=1e-12, momentum_kg_m2_s=1e-8, energy_j_m3=1e-4,
        magnetic=sm.MagneticUnits(field_tesla=1e-4, length_m=1e6))
    config = replace(model(units=units), gamma=5/3)
    rho, target_t = 2e-12, 8e5
    a = config.composition.helium_abundance
    pressure = (2+3*a)*(rho*.001/((1+4*a)*PROTON_MASS_G))*BOLTZMANN_ERG_K*target_t/10
    state = lambda x, y, z: (rho, (0., 0., 2e4), pressure, (0., 0., 1e-4))
    with source_for(state, configuration=config, mixed=True)[0] as source:
        ready = sm.prepare(source, scheme="exact-phase")
    recovered = mhd_fields(ready, model=config, outputs=("density", "temperature"))
    np.testing.assert_allclose(recovered.values[..., 1], target_t, rtol=2e-14)
    response = sm.AIA171(composition=config.composition)
    thermo = sm.thermal_fields(recovered, recovered, density_unit_g_cm3=.001,
        temperature_component=1, temperature_label="explicit ideal-MHD total energy", model=response)
    plane = sm.orthographic_plane(ready.mesh.lower, ready.mesh.upper, [0, 0, 1], (6, 5))
    length_cm = units.magnetic.length_m*100
    expected_emissivity = response.emissivity(rho*.001, target_t)
    for order in ("thermodynamics-first", "emissivity-first"):
        image = sm.integrate_thermal_los(thermo, plane, [0, 0, 1], length_unit_cm=length_cm,
                                        model=response, order=order, workers=2)
        assert image.complete
        np.testing.assert_allclose(image.values, expected_emissivity*image.depth*length_cm, rtol=2e-13)
    velocity = mhd_fields(ready, model=config, outputs="velocity")
    assert tuple(f.name for f in velocity.fields) == ("vx", "vy", "vz")
    lines = app.trace(velocity, sm.PointSet([[.25, .25, .5], [1.5, .5, .5]]), step=.025, max_steps=8)
    for seed_id, seed in zip(lines.seeds.ids, lines.seeds.positions):
        for direction in (-1, 1):
            branch = lines.branch(seed_id, direction)
            assert len(branch) == 9
            expected = seed + np.arange(9)[:, None]*[0., 0., direction*.025]
            np.testing.assert_allclose(branch, expected, atol=1e-14)
