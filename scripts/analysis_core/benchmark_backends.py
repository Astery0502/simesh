"""Actual WENO ready consumers: matched thread-pool/OpenMP execution backends.

Shared file-to-ready preparation is timed separately and added to each first
consumer cost, explicitly not independent cold disk trials per backend.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import resource
import statistics
import time
import numpy as np
from simesh.analysis import open_prepared, trace, integrate_los, orthographic_plane
from simesh.utils.lib.analysis.native import openmp_build_info
from analysis_core.probe_runtime import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workload',choices=('f','l'),required=True)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--thread-dynamic-only',action='store_true')
    args=p.parse_args()
    suffix='-thread-dynamic' if args.thread_dynamic_only else ''
    output=Path(f'benchmark-results/analysis-core/runtime-ready-{args.workload}{suffix}.json')
    result={'openmp':openmp_build_info(),'wait_policy':os.environ.get('OMP_WAIT_POLICY','runtime default'),
            'preparation':'one shared file-to-ready preparation; OS cache uncontrolled','cases':[]}
    start=time.perf_counter()
    ready=open_prepared('data/weno509_sub_0000.dat',
                        field_names=['rho'] if args.workload=='l' else ['b1','b2','b3'])
    prep=time.perf_counter()-start
    result['file_to_ready_seconds']=prep
    result['preparation_stats']=ready.preparation_stats
    mesh=ready.mesh
    if args.workload=='f':
        selected,_,_=mesh.select_box(mesh.lower+.3*(mesh.upper-mesh.lower),mesh.lower+.7*(mesh.upper-mesh.lower))
        ids=selected[np.linspace(0,len(selected)-1,4096,dtype=np.int64)]
        seeds=np.ascontiguousarray(mesh.bounds[ids].mean(axis=1))
        step=float(.25*np.min(mesh.spacing[ids]))
        result.update(seeds=len(seeds),step=step,max_steps=512)
        granularities=(256,4096)
    else:
        direction=[.3,.2,1.]
        plane=orthographic_plane(mesh.lower,mesh.upper,direction,(256,256))
        granularities=(16,64)
    reference=None
    for grain in granularities:
        for backend,schedule in ([('threadpool','dynamic')] if args.thread_dynamic_only else
                                 [('threadpool','static'),('openmp','static'),('openmp','dynamic')]):
            for workers in (1,2,4):
                row={'backend':backend,'schedule':schedule,'workers':workers,'grain':grain,'wall':[],'cpu':[]}
                for repeat in range(args.repeats+1):
                    start,cpu=time.perf_counter(),time.process_time()
                    if args.workload=='f':
                        out=trace(ready,seeds,step=step,max_steps=512,seed_batch=grain,
                                  workers=workers,backend=backend,schedule=schedule)
                        wall,used=time.perf_counter()-start,time.process_time()-cpu
                        signature=digest(out.positions,out.length,out.steps,out.termination,out.samples)
                        row['accepted_steps']=int(out.steps.sum())
                    else:
                        out=integrate_los(ready,plane,direction,tile_shape=(grain,grain),
                                          workers=workers,backend=backend,schedule=schedule)
                        wall,used=time.perf_counter()-start,time.process_time()-cpu
                        assert out.complete
                        signature=digest(out.values,out.entry,out.exit,out.status,out.samples)
                        row['samples']=int(out.samples.sum())
                    if reference is not None and reference!=signature:
                        raise AssertionError('backend/granularity changed numerical output')
                    reference=signature
                    if repeat==0:
                        row['first_consumer_seconds']=wall
                        row['shared_preparation_plus_first_seconds']=prep+wall
                    else:
                        row['wall'].append(wall);row['cpu'].append(used)
                    del out
                row['median']=statistics.median(row['wall'])
                row['signature']=signature
                result['cases'].append(row)
                result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                output.write_text(json.dumps(result,indent=2)+'\n')
                print(args.workload,grain,backend,schedule,workers,round(row['median'],6),flush=True)
    print('complete',output,flush=True)


if __name__=='__main__':
    main()
