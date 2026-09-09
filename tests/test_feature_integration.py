"""The recovered state feeds native reductions, application maps and result I/O."""

from dataclasses import asdict
import numpy as np
import simesh as sm
from simesh import applications as app
from simesh.physics.thermal import PROTON_MASS_G, BOLTZMANN_ERG_K
from test_mhd_thermodynamics import source_for


def test_recovery_reductions_los_and_persistence(tmp_path):
    composition = sm.CoronalComposition(helium_abundance=.1)
    units = sm.MHDUnits(density_kg_m3=1.,momentum_kg_m2_s=1.,energy_j_m3=1.,
                        magnetic=sm.MagneticUnits(field_tesla=1.,length_m=1.))
    model = sm.IdealMHD(gamma=5/3,energy_kind="total",composition=composition,units=units)
    density,temperature = 1e-12,8e5
    nh = density*1e-3/((1+4*composition.helium_abundance)*PROTON_MASS_G)
    pressure = (2+3*composition.helium_abundance)*nh*BOLTZMANN_ERG_K*temperature/10
    state = lambda x,y,z: (density,(0.,0.,2e4),pressure,(0.,0.,1e-4))
    with source_for(state,configuration=model,mixed=True)[0] as source:
        raw = sm.read_fields(source)
        prepared = sm.prepare(source,scheme="exact-phase")
    recovered_raw = sm.mhd_fields(raw,model=model,outputs=("density","temperature"))
    length = sm.LengthUnits(units.magnetic.length_m,"m")
    mass = sm.volume_integral(recovered_raw,"density",units=length)
    average = sm.weighted_mean(recovered_raw,"temperature",weights=recovered_raw,weight_component="density",units=length)
    np.testing.assert_allclose(mass.value,2e-12)
    np.testing.assert_allclose(average.value,temperature)
    assert mass.coverage.complete and average.coverage.complete
    state_fields = sm.mhd_fields(prepared,model=model,outputs=("density","temperature"))
    thermal = sm.thermal_fields(state_fields,state_fields,density_component=0,temperature_component=1,
        density_unit_g_cm3=1e-3,temperature_label="recovered ideal MHD temperature")
    plane = sm.Plane([0.,0.,.25],[2.,0.,0.],[0.,1.,0.],(3,2))
    rays = sm.RaySet.from_plane(plane,[0,0,1],ids=np.arange(10,16,dtype=np.int64))
    image = app.thermal_los(thermal,rays,length_unit_cm=100.,workers=2)
    assert image.complete and (image.values > 0).all()
    metadata = {"model":asdict(model),"total_mass":mass.value,"mass_units":mass.units,
                "mean_temperature_K":average.value}
    path = sm.save_result(tmp_path/"thermal.npz",image,metadata=metadata,source={"dataset":"analytic MHD"})
    loaded = sm.load_result(path)
    np.testing.assert_array_equal(loaded.result.image,image.image)
    np.testing.assert_array_equal(loaded.result.rays.origins.ids,rays.origins.ids)
    assert loaded.metadata == metadata and loaded.source_verification == "unverified"
    velocity = sm.mhd_fields(prepared,model=model,outputs="velocity")
    lines = app.trace(velocity,rays.origins,step=.05,max_steps=100,workers=2)
    restored = sm.load_result(sm.save_result(tmp_path/"streamlines.npz",lines,metadata={"vector":"velocity"})).result
    np.testing.assert_array_equal(restored.seeds.ids,rays.origins.ids)
    np.testing.assert_array_equal(restored.positions,lines.positions)
    assert np.all(restored.termination == sm.Termination.DOMAIN_EXIT)
    magnetic = sm.select_fields(prepared, ("b1", "b2", "b3"))
    current = sm.current_density(magnetic, units=units.magnetic)
    scaled = sm.derive_many(state_fields, {"temperature_MK": "MK", "density_cgs": "g cm^-3"},
        lambda ctx: {"density_cgs": ctx.field("density")*1e-3,
                     "temperature_MK": ctx.field("temperature")*1e-6})
    quantities = sm.merge_fields((scaled, current))
    profiles = sm.sample_line_profiles(quantities, restored, ("density_cgs", "temperature_MK", "jz"),
                                      point_batch=7, workers=2, length_units=length)
    assert profiles.usable.all() and profiles.line_source_identity is None
    assert profiles.source_identity is quantities.value_identity
    np.testing.assert_allclose(profiles.values, np.broadcast_to([density*1e-3, temperature*1e-6, 0.],
                                                               profiles.values.shape), atol=1e-14)
    np.testing.assert_array_equal(profiles.seed_ids, rays.origins.ids)
    for seed_id in profiles.seed_ids:
        for direction in (-1, 1):
            branch = profiles.branch(seed_id, direction)
            expected = np.r_[0., np.cumsum(np.linalg.norm(np.diff(branch.positions, axis=0), axis=1))]
            np.testing.assert_allclose(branch.arclength, expected)
