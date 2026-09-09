"""Interleaved end-to-end ready LOS controls under a changing desktop load.

One file-to-ready preparation is shared. Report both measured per-backend work
and preparation-plus-work estimates; these are not separate disk-cold runs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
from simesh.analysis import open_prepared,integrate_los,orthographic_plane
from simesh.utils.lib.analysis.native import openmp_build_info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames',type=int,default=128)
    p.add_argument('--output',default='benchmark-results/analysis-core/runtime-paired-backends.json')
    args=p.parse_args()
    start=time.perf_counter()
    ready=open_prepared('data/weno509_sub_0000.dat',field_names=['rho'])
    prep=time.perf_counter()-start
    configs=[('threadpool','static'),('threadpool','dynamic'),('openmp','dynamic')]
    hashes=[hashlib.sha256() for _ in configs]
    rows=[dict(backend=b,schedule=s,wall=[],cpu=[],delivery_seconds=0.) for b,s in configs]
    result={'frames':args.frames,'openmp':openmp_build_info(),'preparation_seconds':prep,
            'preparation_stats':ready.preparation_stats,'cases':rows,
            'wait_policy':os.environ.get('OMP_WAIT_POLICY','runtime default'),
            'note':'rotating backend order for each view; shared preparation plus work is a complete-cost estimate'}
    for frame in range(args.frames):
        angle=2*np.pi*frame/args.frames
        direction=[.3+.08*np.cos(angle),.2+.08*np.sin(angle),1.]
        plane=orthographic_plane(ready.mesh.lower,ready.mesh.upper,direction,(256,256))
        reference=None
        for j in [(frame+k)%len(configs) for k in range(len(configs))]:
            backend,schedule=configs[j]
            stamp,cpu=time.perf_counter(),time.process_time()
            image=integrate_los(ready,plane,direction,workers=4,tile_shape=(64,64),
                                backend=backend,schedule=schedule)
            rows[j]['wall'].append(time.perf_counter()-stamp)
            rows[j]['cpu'].append(time.process_time()-cpu)
            if not image.complete:
                raise AssertionError('incomplete image')
            stamp=time.perf_counter()
            check=hashlib.sha256()
            for name in ('values','entry','exit','status','samples','misses'):
                a=getattr(image,name)
                payload=memoryview(a).cast('B')
                check.update(payload)
                hashes[j].update(payload)
            signature=check.hexdigest()
            if reference is not None and reference!=signature:
                raise AssertionError('backend changed an image')
            reference=signature
            rows[j]['delivery_seconds']+=time.perf_counter()-stamp
            del image
        if (frame+1)%16==0:
            print('paired views complete',frame+1,flush=True)
            Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    for j,row in enumerate(rows):
        row['signature']=hashes[j].hexdigest()
        row['query_and_delivery_seconds']=sum(row['wall'])+row['delivery_seconds']
        row['shared_preparation_plus_work_seconds']=prep+row['query_and_delivery_seconds']
    result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['runtime_controlled_upper_bytes']=ready.nbytes+ready.mesh.nbytes+256*256*96+64*64*384+256
    result['experiment_wall_seconds']=time.perf_counter()-start
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k not in ('wall','cpu')} for row in rows]),flush=True)


if __name__=='__main__':
    main()
