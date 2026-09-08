"""File-to-128 distinct scalar images; stream every complete image into a digest."""
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
    p.add_argument('--backend',choices=('threadpool','openmp'),default='threadpool')
    p.add_argument('--schedule',choices=('static','dynamic'),default='static')
    p.add_argument('--frames',type=int,default=128)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    started,cpu=time.perf_counter(),time.process_time()
    ready=open_prepared('data/weno509_sub_0000.dat',field_names=['rho'])
    prep=time.perf_counter()-started
    signature=hashlib.sha256()
    times=[]
    delivered=0
    samples=0
    image_started=time.perf_counter()
    for i in range(args.frames):
        angle=2*np.pi*i/args.frames
        direction=[.3+.08*np.cos(angle),.2+.08*np.sin(angle),1.]
        plane=orthographic_plane(ready.mesh.lower,ready.mesh.upper,direction,(256,256))
        stamp=time.perf_counter()
        image=integrate_los(ready,plane,direction,tile_shape=(64,64),workers=4,
                            backend=args.backend,schedule=args.schedule)
        times.append(time.perf_counter()-stamp)
        if not image.complete:
            raise AssertionError('incomplete frame')
        for name in ('values','entry','exit','status','samples','misses'):
            a=getattr(image,name)
            signature.update(memoryview(a).cast('B'))
            delivered+=a.nbytes
        samples+=int(image.samples.sum())
        del image
        if (i+1)%32==0:
            print('delivered',i+1,'frames',flush=True)
    result={'parameters':vars(args),'openmp':openmp_build_info(),
        'wait_policy':os.environ.get('OMP_WAIT_POLICY','runtime default'),
        'total_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,
        'file_to_prepared_seconds':prep,'views_and_delivery_seconds':time.perf_counter()-image_started,
        'consumer_wall_seconds':times,'all_outputs_signature':signature.hexdigest(),
        'delivered_bytes':delivered,'samples':samples,'preparation_stats':ready.preparation_stats,
        'runtime_controlled_upper_bytes':ready.nbytes+ready.mesh.nbytes+256*256*96+64*64*384+256,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'scope':'128 distinct images streamed to checksum; every complete image returned before release; OS cache uncontrolled'}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('consumer_wall_seconds','preparation_stats')}),flush=True)


if __name__=='__main__':
    main()
