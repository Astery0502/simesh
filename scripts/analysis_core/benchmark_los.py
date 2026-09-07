"""WENO stored-rho columns: LOS geometry/consumer evidence, not thermal synthesis."""

import gc
import json
import os
from pathlib import Path
import resource
import time
import numpy as np

from simesh.analysis import FieldDefinition,PreparedPool,integrate_los,orthographic_plane
from simesh.amrvac.analysis import prepare_resident
from simesh.amrvac.datio import read_blocks_sequential
from simesh_rewrite.amrvac_dat import read_amrvac_v5_index,bind_amrvac_v5_forest
from simesh_rewrite.blockio import array_block_reader
from analysis_core.rewrite_provider import make_source
from analysis_core.benchmark_prepared import measure,compare
from analysis_core.los_reference import ray_integral


def main():
    path='data/weno509_sub_0000.dat'
    start=time.perf_counter()
    fd=os.open(path,os.O_RDONLY)
    try:
        index=read_amrvac_v5_index(fd)
        binding=bind_amrvac_v5_forest(index)
    finally:
        os.close(fd)
    if index.leaf_count!=22614 or index.field_names[0]!='rho' or index.field_count!=7:
        raise ValueError('fixture differs from the declared scalar-column profile')
    raw=read_blocks_sequential(path,[0])
    source=make_source(binding.root_shape,binding.coord_to_rank,binding.forest,
        index.domain_lower,index.domain_upper,index.block_cell_counts,array_block_reader(raw),
        (FieldDefinition('rho','stored-code-density'),),support_capacity=256)
    mesh=source.mesh
    result={'source':path,'source_bytes':Path(path).stat().st_size,
        'meaning':'integral of stored rho in code units; no thermal response/EOS supplied',
        'source_setup_seconds':time.perf_counter()-start,'views':[]}
    ready=prepare_resident(mesh,binding.root_shape,index.forest_flags,raw,source.fields)
    result['resident_preparation']=ready.preparation_stats
    images=[]
    for direction in ([0.,0.,1.],[.3,.2,1.]):
        plane=orthographic_plane(mesh.lower,mesh.upper,direction,(32,32))
        start=time.perf_counter()
        exact=integrate_los(ready,plane,direction,quadrature='gauss2')
        first=time.perf_counter()-start
        if not exact.complete:
            raise AssertionError(np.unique(exact.status,return_counts=True))
        variants=[]
        for method,fraction in [('midpoint',.5),('midpoint',.125),('gauss2',.5)]:
            values=integrate_los(ready,plane,direction,quadrature=method,step_fraction=fraction)
            if not values.complete:
                raise AssertionError(np.unique(values.status,return_counts=True))
            error=values.values-exact.values
            variants.append({'quadrature':method,'step_fraction':fraction,
                'timing':measure(lambda:integrate_los(ready,plane,direction,quadrature=method,step_fraction=fraction),3),
                'samples':int(values.samples.sum()),'max_abs_error':float(np.max(np.abs(error))),
                'rms_error':float(np.sqrt(np.mean(error**2)))})
        parallel=integrate_los(ready,plane,direction,quadrature='gauss2',workers=4)
        np.testing.assert_array_equal(exact.values,parallel.values)
        threading=measure(lambda:integrate_los(ready,plane,direction,quadrature='gauss2',workers=4),3)
        nonempty=np.flatnonzero(exact.depth.ravel()>0)
        chosen=nonempty[np.linspace(0,len(nonempty)-1,8,dtype=np.int64)]
        u=(chosen//plane.shape[1]+.5)/plane.shape[0]
        v=(chosen%plane.shape[1]+.5)/plane.shape[1]
        origins=plane.origin+u[:,None]*plane.u+v[:,None]*plane.v
        references=np.array([ray_integral(ready,p,exact.direction) for p in origins])
        check=compare(exact.values.ravel()[chosen],references[:,0])
        if not check['within_tolerance']:
            raise AssertionError(check)
        np.testing.assert_allclose(exact.depth.ravel()[chosen],references[:,1],atol=1e-12,rtol=1e-12)
        pool=PreparedPool(source,[0],512)
        bounded=[]
        for workers in (1,4):
            pool.clear()
            before=pool.prepared_count
            start=time.perf_counter()
            streamed=integrate_los(pool,plane,direction,quadrature='gauss2',workers=workers)
            elapsed=time.perf_counter()-start
            if not streamed.complete:
                raise AssertionError(np.unique(streamed.status,return_counts=True))
            parity=compare(streamed.values,exact.values)
            if not parity['within_tolerance']:
                raise AssertionError(parity)
            bounded.append({'workers':workers,'seconds':elapsed,'prepared_owners':pool.prepared_count-before,
                'misses':int(streamed.misses.sum()),'comparison':parity,'pool_controlled_bytes':pool.controlled_bytes})
        pool.close()
        result['views'].append({'direction':direction,'shape':plane.shape,'first_gauss_seconds':first,
            'resident_variants':variants,'gauss_four_workers':threading,'independent_reference':check,
            'bounded':bounded,'image_bytes':sum(a.nbytes for a in vars(exact).values() if isinstance(a,np.ndarray)),
            'nonempty_pixels':len(nonempty),'depth_range':[float(exact.depth.min()),float(exact.depth.max())]})
        images.append(exact.values.copy())
        print('LOS view complete',direction,flush=True)
    result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path('benchmark-results/analysis-core/los.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez('benchmark-results/analysis-core/los-columns.npz',axis=images[0],oblique=images[1])
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
