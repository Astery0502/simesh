"""Declared P4 million-seed and 1000^3 bounded-output acceptance on WENO."""

import argparse
import gc
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np

from simesh.analysis import open_prepared,trace,iter_uniform


def array_bytes(value):
    return sum(a.nbytes for a in vars(value).values() if isinstance(a,np.ndarray))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section',choices=['all','seeds','uniform'],default='all')
    args=parser.parse_args()
    budget=2*1024**3
    path='data/weno509_sub_0000.dat'
    start=time.perf_counter()
    fields=open_prepared(path,field_names=['b1','b2','b3'],budget_bytes=budget)
    mesh=fields.mesh
    result={'source':path,'source_bytes':Path(path).stat().st_size,
            'platform':platform.platform(),'source_setup_seconds':time.perf_counter()-start,
            'input_retained_bound':fields.nbytes+mesh.nbytes,'source_stats':fields.preparation_stats}
    print('scale input ready',flush=True)
    if args.section in ('all','seeds'):
        n=1000
        seeds=np.empty((n*n,3))
        seeds[:,0]=np.repeat(mesh.lower[0]+(np.arange(n)+.5)*(mesh.upper[0]-mesh.lower[0])/n,n)
        seeds[:,1]=np.tile(mesh.lower[1]+(np.arange(n)+.5)*(mesh.upper[1]-mesh.lower[1])/n,n)
        seeds[:,2]=mesh.lower[2]+.5*mesh.spacing[:,2].min()
        step=float(.25*mesh.spacing.min())
        kwargs=dict(step=step,max_steps=32,seed_batch=16384)
        start=time.perf_counter()
        one=trace(fields,seeds,workers=1,budget_bytes=budget,**kwargs)
        one_seconds=time.perf_counter()-start
        print('million seeds: one worker complete',flush=True)
        start=time.perf_counter()
        four=trace(fields,seeds,workers=4,budget_bytes=budget-array_bytes(one),**kwargs)
        four_seconds=time.perf_counter()-start
        for name in ('seed_ids','positions','length','steps','termination','samples'):
            np.testing.assert_array_equal(getattr(one,name),getattr(four,name))
        statuses,counts=np.unique(four.termination,return_counts=True)
        result['million_seeds']={'seeds':len(seeds),'max_steps':32,'step':step,
            'one_seconds':one_seconds,'four_seconds':four_seconds,'arrays_equal':True,
            'accepted_steps':int(four.steps.sum()),'samples':int(four.samples.sum()),
            'termination':dict(zip(map(str,statuses),map(int,counts))),
            'summary_bytes':array_bytes(four),'input_seed_bytes':seeds.nbytes,
            'comparison_controlled_upper':fields.nbytes+mesh.nbytes+seeds.nbytes+3*array_bytes(four)+16384*640}
        np.savez('benchmark-results/analysis-core/million-seeds.npz',
            **{key:value for key,value in vars(four).items() if isinstance(value,np.ndarray)})
        del one,four,seeds
        gc.collect()
        print('million seeds: comparison complete',flush=True)
    if args.section in ('all','uniform'):
        start=time.perf_counter()
        count=0
        checksum=np.zeros(3)
        max_slab_bytes=0
        first_slab=None
        for index,slab in iter_uniform(fields,(1000,1000,1000),budget_bytes=budget):
            if not slab.valid.all():
                raise AssertionError(f'invalid uniform points in slab {index}')
            count+=slab.valid.size
            checksum+=slab.values.sum(axis=(0,1))
            size=slab.values.nbytes+slab.valid.nbytes+slab.owners.nbytes
            max_slab_bytes=max(max_slab_bytes,size)
            if first_slab is None:
                first_slab=time.perf_counter()-start
            if (index+1)%100==0:
                print('uniform slabs complete',index+1,flush=True)
        total=time.perf_counter()-start
        if count!=1000**3:
            raise AssertionError('uniform stream did not deliver every requested point')
        result['uniform_stream']={'resolution':[1000,1000,1000],'points':count,
            'delivered_value_bytes':count*3*8,'first_slab_seconds':first_slab,'total_seconds':total,
            'maximum_slab_bytes':max_slab_bytes,'checksum':checksum.tolist(),
            'checksum_scope':'descriptive slab sum; no conservation or absolute physical accuracy claim',
            'controlled_upper':fields.nbytes+mesh.nbytes+2*max_slab_bytes+64*1000*152+2*1024**2}
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['peak_rss_bytes']=int(rss if sys.platform=='darwin' else rss*1024)
    Path(f'benchmark-results/analysis-core/scale-{args.section}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
