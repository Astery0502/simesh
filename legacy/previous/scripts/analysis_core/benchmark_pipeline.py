"""Estimate and compare file reading, resident halo exchange and computation."""

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import numpy as np

from simesh.analysis import open_prepared,open_source,curl,Plane,sample_plane
from simesh.amrvac.analysis import prepare_resident
from simesh.amrvac.analysis_read import read_full_interiors
from simesh.amrvac.datio import read_blocks_sequential
from simesh_rewrite.amrvac_dat import read_amrvac_v5_index,bind_amrvac_v5_forest
from simesh_rewrite.amrvac_dat_reader import make_amrvac_v5_ordinary_block_reader


def timed(call):
    start=time.perf_counter()
    result=call()
    return result,time.perf_counter()-start


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file',type=Path,default=Path('data/weno509_sub_0000.dat'))
    parser.add_argument('--baseline-dir',type=Path,default=Path('benchmark-results/analysis-core/pipeline-baseline'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    spec=importlib.util.spec_from_file_location('simesh.amrvac._pipeline_before',args.baseline_dir/'analysis_io.py')
    before=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(before)
    result={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'dirty':True,'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
        'file':str(args.file),'file_bytes':args.file.stat().st_size,
        'cache':'OS page cache uncontrolled; sequential interleaved runs', 'readers':{},'pipelines':{}}
    fd=os.open(args.file,os.O_RDONLY)
    try:
        index=read_amrvac_v5_index(fd)
        binding=bind_amrvac_v5_forest(index)
        reader=make_amrvac_v5_ordinary_block_reader(fd,index,binding)
        fields=np.array([index.field_names.index(name) for name in ('b1','b2','b3')],dtype=np.int64)
        reference=read_blocks_sequential(args.file,fields.tolist())
        rows={name:[] for name in ('canonical','bulk16','bulk64')}
        for repeat in range(3):
            names=('canonical','bulk16','bulk64') if repeat%2==0 else ('bulk64','bulk16','canonical')
            for name in names:
                if name=='canonical':
                    data,seconds=timed(lambda:read_blocks_sequential(args.file,fields.tolist()))
                    stats={}
                else:
                    (data,stats),seconds=timed(lambda:read_full_interiors(reader,fields,
                                              chunk_bytes=(16 if name=='bulk16' else 64)*1024**2))
                np.testing.assert_array_equal(data.view(np.uint64),reference.view(np.uint64))
                rows[name].append({'seconds':seconds,**stats})
                del data
                gc.collect()
        result['readers']={name:{'runs':values,'median_seconds':float(np.median([v['seconds'] for v in values]))}
                           for name,values in rows.items()}
        del reference,reader,index,binding
    finally:
        os.close(fd)
    print('reader comparisons complete',flush=True)

    def canonical():
        start=time.perf_counter()
        with open_source(args.file,field_names=['b1','b2','b3']) as source:
            opened=time.perf_counter()
            raw=read_blocks_sequential(args.file,list(source.original_field_ids))
            read=time.perf_counter()
            ready=prepare_resident(source.mesh,source.mesh.roots.shape,source.mesh.node_leaves>=0,
                                   raw,source.fields,budget_bytes=2*1024**3-source.resident_bytes)
            ready.preparation_stats.update(open_seconds=opened-start,read_seconds=read-opened)
            return ready

    calls={'canonical_reader_and_mesh':canonical,
           'previous_analysis':lambda:before.open_prepared(args.file,field_names=['b1','b2','b3']),
           'bulk_analysis':lambda:open_prepared(args.file,field_names=['b1','b2','b3'])}
    references=None
    rows={name:[] for name in calls}
    for repeat in range(3):
        names=list(calls) if repeat%2==0 else list(reversed(calls))
        for name in names:
            start=time.perf_counter()
            ready=calls[name]()
            prepared=time.perf_counter()
            mesh=ready.mesh
            width=mesh.upper-mesh.lower
            derived=curl(ready)
            differentiated=time.perf_counter()
            a=sample_plane(derived,Plane(mesh.lower+width*[0,0,.5],width*[1,0,0],width*[0,1,0],(128,128)))
            b=sample_plane(derived,Plane(mesh.lower+width*[0,0,.2],width*[1,0,.3],width*[0,1,.3],(96,80)))
            finished=time.perf_counter()
            assert a.valid.all() and b.valid.all()
            # Full raw bits were compared above; unchanged canonical exchange
            # is additionally checked on spread blocks and both complete slices.
            ids=np.linspace(0,mesh.leaf_count-1,64,dtype=int)
            observed=(ready.values[ids],derived.values[ids],a.values,b.values)
            if references is None:
                references=tuple(value.copy() for value in observed)
            else:
                for value,reference in zip(observed,references):
                    np.testing.assert_array_equal(value,reference)
            row={'prepare_seconds':prepared-start,'curl_seconds':differentiated-prepared,
                 'slices_seconds':finished-differentiated,'total_seconds':finished-start,
                 'preparation':ready.preparation_stats,'prepared_bytes':ready.nbytes,
                 'derived_bytes':derived.nbytes,'geometry_bytes':mesh.nbytes}
            rows[name].append(row)
            print(name,repeat,round(row['total_seconds'],4),flush=True)
            del observed,ready,derived,a,b,mesh
            gc.collect()
    result['pipelines']={name:{'runs':values,'median_seconds':float(np.median([v['total_seconds'] for v in values]))}
                         for name,values in rows.items()}
    result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
