"""Compose RK line integrals and sampled-curve calculus on a synthetic AMR field."""

import json
import numpy as np
import simesh as sm
from simesh import applications as app
from simesh.bounded import PreparedPool, trace_bounded


def run():
    mesh = sm.mesh_from_forest((1,1,2),np.array([False]+[True]*9),
        lower=(-1,-1,0),upper=(1,1,1),block_shape=(8,8,8))
    local = np.indices(mesh.block_shape)+.5
    values = np.empty((mesh.leaf_count,7,*mesh.block_shape))
    for leaf in range(mesh.leaf_count):
        x,y,z = mesh.bounds[leaf,0,:,None,None,None]+local*mesh.spacing[leaf,:,None,None,None]
        values[leaf] = np.array([0*x,0*y,np.ones_like(z),np.ones_like(z),z,(2+x)*z,2+x])
    names = ('vx','vy','vz','one','height','weighted_height','weight')
    seeds = sm.PointSet([[.13,.17,.51],[.21,.27,.49]])
    with sm.source_from_arrays(mesh,values,names) as source:
        vector = sm.prepare(source,names[:3],scheme='exact-phase')
        rates = sm.prepare(source,names[3:],scheme='exact-phase')
        options = dict(direction=-1,max_length=.2,max_steps=1000,integrands=rates,workers=2)
        traced = sm.trace(vector,seeds.positions,**options)
        with PreparedPool(source,names[:3],scheme='exact-phase',capacity=1) as pool:
            resumed = trace_bounded(pool,seeds.positions,**options)
    np.testing.assert_array_equal(traced.integrals,resumed.integrals)
    means = traced.integrals[:,2]/traced.integrals[:,3]
    # Native halo preparation approximates mixed products at refinement faces.
    np.testing.assert_allclose(means,seeds.positions[:,2]-.1,rtol=1e-5)
    np.testing.assert_allclose(traced.integrals[:,0],traced.length,atol=2e-14)

    paths = app.trace(vector,seeds,direction='against',max_length=.2,max_steps=1000,workers=2)
    profiles = sm.sample_line_profiles(rates,paths,components='height',workers=2)
    sampled_integrals, derivatives = [], []
    for seed_id in seeds.ids:
        profile = profiles.branch(seed_id,-1)
        assert profile.usable.all()
        sampled_integrals.append(float(sm.line_integral(profile.values[:,0],profile.arclength)))
        slope = sm.line_derivative(profile.values[:,0],profile.arclength,edge_order=2)
        np.testing.assert_allclose(slope,-1.,atol=2e-12)
        derivatives.append(float(np.mean(slope)))
    np.testing.assert_allclose(sampled_integrals,traced.integrals[:,1],atol=2e-14)
    report = dict(seeds=len(seeds),weighted_mean_height=means.tolist(),
                  weighted_mean_max_error=float(np.max(np.abs(means-(seeds.positions[:,2]-.1)))),
                  rk_height_integrals=traced.integrals[:,1].tolist(),
                  sampled_height_integrals=sampled_integrals,
                  mean_along_branch_derivative=derivatives,
                  bounded_misses=int(resumed.misses.sum()))
    print(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    run()
