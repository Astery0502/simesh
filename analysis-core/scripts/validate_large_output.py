"""Actual million-seed summaries and billion-point streamed output, without a volume file."""
import argparse
import gc
import json
from pathlib import Path
import resource
import time

import numpy as np
import simesh as sm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    limit=2*1024**3
    mesh=sm.mesh_from_forest((1,1,1),np.array([True]),lower=(0,0,0),upper=(1,1,1),block_shape=(8,8,8))
    raw=np.zeros((1,3,8,8,8));raw[:,0]=1
    with sm.source_from_arrays(mesh,raw,('b1','b2','b3')) as source:
        ready=sm.prepare(source,scheme='coordinate-phase',memory_limit=limit)
    q=.1+.8*(np.arange(1000)+.5)/1000
    x,y=np.meshgrid(q,q,indexing='ij')
    seeds=np.ascontiguousarray(np.column_stack((x.ravel(),y.ravel(),np.full(x.size,.1))))
    del x,y
    start=time.perf_counter()
    lines=sm.trace(ready,seeds,step=1e-4,max_steps=32,workers=4,memory_limit=limit)
    elapsed=time.perf_counter()-start
    assert np.all(lines.steps==32) and np.all(lines.termination==sm.Termination.MAX_STEPS)
    np.testing.assert_array_equal(lines.seed_ids,np.arange(1_000_000,dtype=np.int64))
    np.testing.assert_allclose(lines.positions,seeds+[.0032,0,0],rtol=0,atol=2e-14)
    record={'profile':'constant native vector; output scaling only, not large-input acceptance',
            'million_seeds':{'seconds':elapsed,'seed_count':len(seeds),'accepted_steps':int(lines.steps.sum()),
                             'output_bytes':sum(v.nbytes for v in vars(lines).values() if isinstance(v,np.ndarray))}}
    a.output.write_text(json.dumps(record,indent=2)+'\n')
    print('million seeds complete',round(elapsed,3),flush=True)
    del lines,seeds
    gc.collect()
    start=time.perf_counter()
    points=total_bytes=largest=0
    expected=np.array([1.,0.,0.])
    for index,slab in sm.iter_uniform(ready,(1000,1000,1000),workers=4,memory_limit=limit):
        assert slab.valid.all()
        np.testing.assert_allclose(slab.values.min(axis=(0,1)),expected,rtol=0,atol=2e-14)
        np.testing.assert_allclose(slab.values.max(axis=(0,1)),expected,rtol=0,atol=2e-14)
        points+=slab.valid.size
        total_bytes+=slab.values.nbytes
        largest=max(largest,slab.values.nbytes+slab.valid.nbytes+slab.owners.nbytes)
        if (index+1)%100==0:
            print('slabs consumed',index+1,flush=True)
    assert points==1_000_000_000 and total_bytes==24_000_000_000
    record['uniform_stream']={'seconds_with_sink_checks':time.perf_counter()-start,'points':points,
                              'value_bytes_delivered':total_bytes,'largest_slab_bytes':largest}
    record['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    a.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
