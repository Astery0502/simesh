"""Complete file-to-global-curl-and-slices comparison, bounded worker results."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
from simesh.analysis import open_source,global_curl,sample_plane,Plane
from simesh.amrvac.analysis_execution import global_curl_file
from analysis_core.probe_runtime import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--backend',choices=('serial','thread','process'),default='process')
    p.add_argument('--task',type=int,default=512)
    p.add_argument('--support',type=int,default=256)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--budget',type=int,default=1024**3)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    result={'parameters':vars(args),'runs':[],'cache':'fresh numerical state per complete call; OS cache uncontrolled'}
    for repeat in range(args.repeats):
        children=resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu=children.ru_utime+children.ru_stime
        start,cpu=time.perf_counter(),time.process_time()
        if args.backend=='serial':
            with open_source('data/weno509_sub_0000.dat',field_names=['b1','b2','b3'],support_capacity=args.support) as source:
                out=global_curl(source,batch_size=256,budget_bytes=args.budget)
        else:
            out=global_curl_file('data/weno509_sub_0000.dat',backend=args.backend,workers=args.workers,
                                 task_size=args.task,support_capacity=args.support,budget_bytes=args.budget)
        lower,upper=out.mesh.lower,out.mesh.upper
        extent=upper-lower
        slices=[sample_plane(out,Plane(lower+[0,0,.5*extent[2]],
                    [extent[0],0,0],[0,extent[1],0],(128,128))),
                sample_plane(out,Plane(lower+[0,0,.25*extent[2]],
                    [extent[0],0,.25*extent[2]],[0,extent[1],.25*extent[2]],(128,128)))]
        signature=digest(out.values,*(s.values for s in slices))
        row={'wall_seconds':time.perf_counter()-start,'parent_cpu_seconds':time.process_time()-cpu,
             'signature':signature,'preparation':out.preparation_stats,
             'parent_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'children_peak_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
        children=resource.getrusage(resource.RUSAGE_CHILDREN)
        row['child_cpu_seconds']=children.ru_utime+children.ru_stime-child_cpu
        result['runs'].append(row)
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(row),flush=True)
        del out,slices


if __name__=='__main__':
    main()
