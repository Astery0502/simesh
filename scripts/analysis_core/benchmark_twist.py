"""P3-F WENO twist, trajectories and explicit selected-seed retracing."""

import json
import os
from pathlib import Path
import time
import resource
import numpy as np

from simesh.analysis import FieldDefinition, prepare, PreparedPool, CurlPool, trace, retrace, sample
from simesh.amrvac.datio import read_blocks_sequential
from simesh_rewrite.amrvac_dat import read_amrvac_v5_index, bind_amrvac_v5_forest
from simesh_rewrite.blockio import array_block_reader
from analysis_core.rewrite_provider import make_source
from analysis_core.benchmark_prepared import measure
from test_prepared import source_fixture
from test_field_lines import scalar_rk4


def main():
    def helix(x,y,z):
        return np.array([-(y-5.),x-3.,np.full_like(x,.4)])
    source,_ = source_fixture(helix)
    ready = prepare(source,np.arange(source.mesh.leaf_count),[0,1,2])
    angles = np.linspace(0,2*np.pi,2048,endpoint=False)
    seeds = np.ascontiguousarray(np.column_stack((3+np.cos(angles),5+np.sin(angles),np.zeros(2048))))
    result = {"ordinary_2048x600_one":measure(lambda:trace(ready,seeds,step=.005,max_steps=600,seed_batch=2048),3)}
    del ready,source
    path = 'data/weno509_sub_0000.dat'
    start = time.perf_counter()
    fd = os.open(path,os.O_RDONLY)
    try:
        index = read_amrvac_v5_index(fd)
        binding = bind_amrvac_v5_forest(index)
    finally:
        os.close(fd)
    backing = read_blocks_sequential(path,[4,5,6])
    source = make_source(binding.root_shape,binding.coord_to_rank,binding.forest,
        index.domain_lower,index.domain_upper,index.block_cell_counts,array_block_reader(backing),
        tuple(FieldDefinition(n,'code') for n in ('b1','b2','b3')))
    mesh = source.mesh
    result['source_setup_seconds'] = time.perf_counter()-start
    selected,_,_ = mesh.select_box(mesh.lower+.42*(mesh.upper-mesh.lower),
                                  mesh.lower+.58*(mesh.upper-mesh.lower))
    chosen = selected[np.linspace(0,len(selected)-1,32,dtype=np.int64)]
    seeds = np.ascontiguousarray(mesh.bounds[chosen].mean(axis=1))
    step = float(.25*np.min(mesh.spacing[chosen]))
    pool = PreparedPool(source,[0,1,2],256)
    coupled = CurlPool(pool)
    start = time.perf_counter()
    first = trace(coupled,seeds,step=step,max_steps=128,twist=True)
    result['first_twist_seconds'] = time.perf_counter()-start
    result['first_prepared_owners'] = pool.prepared_count
    result['first_derived_owners'] = coupled.derived_count
    result['controlled_pool_bytes'] = coupled.controlled_bytes
    result['warm'] = []
    for workers in (1,4):
        for trajectories in (False,True):
            def run():
                output = trace(coupled,seeds,step=step,max_steps=128,twist=True,
                               workers=workers,trajectories=trajectories)
                np.testing.assert_array_equal(output.positions,first.positions)
                np.testing.assert_array_equal(output.twist,first.twist)
                return output
            output = run()
            result['warm'].append({'workers':workers,'trajectories':trajectories,
                'timing':measure(run,3),'output_bytes':sum(a.nbytes for a in vars(output).values() if isinstance(a,np.ndarray))})
    result['warm_prepared_owners'] = pool.prepared_count
    result['warm_derived_owners'] = coupled.derived_count
    ordinary = trace(pool,seeds,step=step,max_steps=128)
    np.testing.assert_array_equal(ordinary.positions,first.positions)
    result['ordinary_warm'] = measure(lambda:trace(pool,seeds,step=step,max_steps=128),3)
    # All visited owners fit. One read lease supplies the independent scalar
    # stage composition; no I/O or shared mutable integrator is hidden here.
    with coupled.borrow(coupled.resident_leaf_ids) as bundle:
        def rhs(state):
            b,_,vb = sample(bundle.primary,state[None,:3])
            cb,_,vc = sample(bundle.curl,state[None,:3])
            if not (vb[0] and vc[0]):
                raise AssertionError('independent reference needs missing support')
            norm2 = np.dot(b[0],b[0])
            return np.r_[b[0]/np.sqrt(norm2),np.dot(cb[0],b[0])/(4*np.pi*norm2)]
        reference = np.array([scalar_rk4(np.r_[seed,0.],rhs,step,128) for seed in seeds[:8]])
    actual = np.column_stack((first.positions[:8],first.twist[:8]))
    np.testing.assert_allclose(actual,reference,rtol=2e-12,atol=2e-12)
    result['independent_stage_reference_max_abs'] = float(np.max(np.abs(actual-reference)))
    selected_ids = np.argsort(np.abs(first.twist))[-3:].astype(np.int64)
    retraced = retrace(coupled,first,selected_ids,step=step,max_steps=128,twist=True,workers=4)
    np.testing.assert_array_equal(retraced.positions,first.positions[selected_ids])
    np.testing.assert_array_equal(retraced.twist,first.twist[selected_ids])
    result['retrace'] = {'seed_ids':retraced.seed_ids.tolist(),
        'trajectory_bytes':retraced.trajectories.nbytes,'points':int(retraced.point_counts.sum())}
    result['twist_range'] = [float(first.twist.min()),float(first.twist.max())]
    result['accepted_steps'] = int(first.steps.sum())
    result['source'] = path
    result['source_bytes'] = Path(path).stat().st_size
    result['step'],result['max_steps'],result['seeds'] = step,128,32
    result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    coupled.close()
    pool.close()
    Path('benchmark-results/analysis-core/twist.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
