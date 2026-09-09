"""Three nearby full-domain WENO views, same images and finite prepared budget."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
from simesh.analysis import open_source,PreparedPool,integrate_los_views,orthographic_plane
from analysis_core.probe_runtime import digest,Attribution
from analysis_core.benchmark_native_source import count_reads


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--order',choices=('view','tile'),default='tile')
    p.add_argument('--tile',type=int,default=4)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--pool',type=int,default=512)
    p.add_argument('--backend',choices=('threadpool','openmp'),default='threadpool')
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    result={'parameters':vars(args),'runs':[]}
    opening=start=time.perf_counter()
    with open_source('data/weno509_sub_0000.dat',field_names=['rho'],support_capacity=256) as source:
        result['open_seconds']=time.perf_counter()-start
        directions=[[.3,.2,1.],[.31,.2,1.],[.3,.21,1.]]
        planes=[orthographic_plane(source.mesh.lower,source.mesh.upper,d,(32,32)) for d in directions]
        pool=PreparedPool(source,[0],args.pool)
        result['pool_controlled_bytes']=pool.controlled_bytes
        result['consumer_controlled_upper_bytes']=pool.controlled_bytes+3*(32*32*96+256)+args.tile**2*384
        try:
            for repeat in range(args.repeats):
                before=pool.prepared_count
                start,cpu=time.perf_counter(),time.process_time()
                with count_reads() as io:
                    images=integrate_los_views(pool,planes,directions,view_order=args.order,
                        tile_shape=(args.tile,args.tile),workers=args.workers,backend=args.backend)
                wall,used=time.perf_counter()-start,time.process_time()-cpu
                if not all(image.complete for image in images):
                    raise AssertionError('incomplete image')
                signature=digest(*(getattr(im,name) for im in images for name in
                                   ('values','entry','exit','status','samples')))
                row={'wall_seconds':wall,'cpu_seconds':used,'signature':signature,
                     'prepared_owners':pool.prepared_count-before,'io':io,
                     'samples':sum(int(i.samples.sum()) for i in images)}
                if repeat==0:
                    row['source_to_first_result_seconds']=time.perf_counter()-opening
                result['runs'].append(row)
                result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(row),flush=True)
                del images
        finally:
            pool.close()


if __name__=='__main__':
    main()
