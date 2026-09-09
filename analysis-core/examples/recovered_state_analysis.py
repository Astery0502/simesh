"""Recover an analytic MHD state, reduce AMR interiors and persist application results."""

import argparse
from dataclasses import asdict
from pathlib import Path
import numpy as np
import simesh as sm
from simesh import applications as app
from simesh.physics.thermal import BOLTZMANN_ERG_K


def run(output):
    output.mkdir(parents=True,exist_ok=True)
    model = sm.IdealMHD(gamma=5/3,energy_kind="total",composition=sm.CoronalComposition(),
        units=sm.MHDUnits(density_kg_m3=1.,momentum_kg_m2_s=1.,energy_j_m3=1.,
                         magnetic=sm.MagneticUnits(field_tesla=1.,length_m=1.)))
    mesh = sm.mesh_from_forest((2,1,1),np.array([False]+[True]*9),
                              lower=(0,0,0),upper=(2,1,1),block_shape=(8,8,8))
    names = ("rho","m1","m2","m3","e","b1","b2","b3")
    values = np.empty((mesh.leaf_count,8,*mesh.block_shape))
    local = np.indices(mesh.block_shape)+.5
    for leaf in range(mesh.leaf_count):
        x,y,z = mesh.bounds[leaf,0,:,None,None,None]+local*mesh.spacing[leaf,:,None,None,None]
        rho,temperature = 1e-12*(1+.2*x),8e5*(1+.1*z)
        nh = model.composition.number_density(rho*1e-3,convention="amrvac-hydrogen")
        pressure = (2+3*model.composition.helium_abundance)*nh*BOLTZMANN_ERG_K*temperature/10
        values[leaf,0] = rho
        values[leaf,1:3] = 0.
        values[leaf,3] = rho*2e4
        values[leaf,4] = pressure/(model.gamma-1)+.5*rho*(2e4)**2+1e-8/(2*model.units.magnetic.permeability_h_m)
        values[leaf,5:7] = 0.
        values[leaf,7] = 1e-4
    labels = dict(zip(names,("kg m^-3",)+("kg m^-2 s^-1",)*3+("J m^-3",)+("T",)*3))
    with sm.source_from_arrays(mesh,values,names,units=labels) as source:
        raw = sm.read_fields(source)
        prepared = sm.prepare(source,scheme="exact-phase")
    recovered = sm.mhd_fields(raw,model=model,outputs=("density","temperature","internal_energy"))
    lengths = sm.LengthUnits(model.units.magnetic.length_m,"m")
    mass = sm.volume_integral(recovered,"density",units=lengths)
    mean_t = sm.weighted_mean(recovered,"temperature",weights=recovered,weight_component="density",units=lengths)
    distribution = sm.histogram(recovered,[7.9e5,8.4e5,8.9e5],"temperature",
                                weights=recovered,weight_component="density",units=lengths)
    bottom = sm.AxisAlignedSurface("z",0.,[[0.,0.],[2.,1.]],normal=-1)
    flux = sm.surface_flux(raw,bottom,component="b3",units=lengths)
    np.testing.assert_allclose(mass.value,2.4e-12)
    np.testing.assert_allclose(mean_t.value,8.4e5)
    np.testing.assert_allclose(flux.value,-2e-4)
    metadata = {"model":asdict(model),"mass":{"value":mass.value,"units":mass.units},
                "mean_temperature_K":mean_t.value,"bottom_flux":{"value":flux.value,"units":flux.units},
                "temperature_histogram":{"edges":distribution.edges.tolist(),
                    "weights":distribution.bin_weights.tolist(),"weight_units":distribution.weight_units}}
    state = sm.mhd_fields(prepared,model=model,outputs=("density","temperature"))
    thermal = sm.thermal_fields(state,state,density_component=0,temperature_component=1,
                                density_unit_g_cm3=1e-3,temperature_label="recovered analytic ideal-MHD state")
    rays = sm.RaySet.from_plane(sm.Plane([0,0,-.1],[2,0,0],[0,1,0],(8,6)),[0,0,1])
    image = app.thermal_los(thermal,rays,length_unit_cm=100.,workers=2)
    path = sm.save_result(output/"thermal.result.npz",image,metadata=metadata,
                          source={"description":"analytic mixed-level AMR state"},overwrite=True)
    restored = sm.load_result(path)
    np.testing.assert_array_equal(restored.result.image,image.image)
    velocity = sm.mhd_fields(prepared,model=model,outputs="velocity")
    seeds = sm.PointSet.from_plane(sm.Plane([0,0,.5],[2,0,0],[0,1,0],(3,2)))
    lines = app.trace(velocity,seeds,step=.025,max_steps=100,workers=2)
    line_path = sm.save_result(output/"streamlines.result.npz",lines,
                              metadata={"vector":"instantaneous velocity","step":.025},overwrite=True)
    np.testing.assert_array_equal(sm.load_result(line_path).result.positions,lines.positions)
    print(f"Mass: {mass.value:g} {mass.units}")
    print(f"Mass-weighted temperature: {mean_t.value:g} K")
    print(f"Outward bottom flux: {flux.value:g} {flux.units}")
    print(f"Thermal rays complete: {image.complete}; streamlines: {len(lines.seeds)}")
    print(f"Restored source verification: {restored.source_verification}")
    print(f"Saved: {path} and {line_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
