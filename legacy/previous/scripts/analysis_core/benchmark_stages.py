"""Stage attribution against canonical simesh, without changing runtime APIs.

Formal timings and instrumented attribution are separate. A is the canonical
reader, B keeps bulk I/O but decodes one record at a time, C is current bulk I/O
and decoding. All use the same canonical exchange and scientific consumers.
"""

import argparse
import builtins
from contextlib import contextmanager, ExitStack
import gc
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np

from simesh.analysis import open_source, open_prepared, curl, Plane, sample_plane, PreparedFields
from simesh.amrvac import datio, analysis_io, analysis_read, analysis
from simesh.analysis import providers
from simesh_rewrite import amrvac_dat, amrvac_dat_reader


def measure(call):
    wall,cpu=time.perf_counter(),time.process_time()
    value=call()
    return value,{'wall_seconds':time.perf_counter()-wall,'cpu_seconds':time.process_time()-cpu}


def median(rows):
    return {key:float(np.median([row[key] for row in rows])) for key in ('wall_seconds','cpu_seconds')}


def accumulate(rows,name,timing,**counts):
    row=rows.setdefault(name,{'wall_seconds':0.,'cpu_seconds':0.,'calls':0})
    for key,value in timing.items(): row[key]+=value
    row['calls']+=1
    for key,value in counts.items(): row[key]=row.get(key,0)+value


def residual(total,parts):
    return {key:total[key]-sum(p[key] for p in parts) for key in ('wall_seconds','cpu_seconds')}


@contextmanager
def recordwise_decode():
    original=analysis_read._copy_records
    def copy(raw,offset,count,stride,ghosts,state,fields,output):
        for row in range(count):
            original(raw,offset+row*stride,1,stride,ghosts,state,fields,output[row:row+1])
    with patch.object(analysis_read,'_copy_records',copy): yield


def profile_canonical(path,fields):
    """Count Python file operations; buffering means these are not syscall counts."""
    stages={}
    in_metadata=False
    original_metadata=datio.get_metadata
    class File:
        def __init__(self,handle): self.handle=handle
        def __getattr__(self,name): return getattr(self.handle,name)
        def __enter__(self):
            self.handle.__enter__()
            return self
        def __exit__(self,*args):
            value,t=measure(lambda:self.handle.__exit__(*args))
            accumulate(stages,'metadata_close' if in_metadata else 'data_close',t)
            return value
        def read(self,*args):
            value,t=measure(lambda:self.handle.read(*args))
            accumulate(stages,'metadata_read' if in_metadata else 'data_read',t,returned_bytes=len(value))
            return value
        def seek(self,*args):
            value,t=measure(lambda:self.handle.seek(*args))
            accumulate(stages,'metadata_seek' if in_metadata else 'data_seek',t)
            return value
    def opening(*args,**kwargs):
        value,t=measure(lambda:builtins.open(*args,**kwargs))
        accumulate(stages,'metadata_open' if in_metadata else 'data_open',t)
        return File(value)
    def metadata(*args,**kwargs):
        nonlocal in_metadata
        in_metadata=True
        try: value,t=measure(lambda:original_metadata(*args,**kwargs))
        finally: in_metadata=False
        accumulate(stages,'metadata_total',t)
        return value
    with patch.object(datio,'open',opening,create=True),patch.object(datio,'get_metadata',metadata):
        values,total=measure(lambda:datio.read_blocks_sequential(path,fields.tolist()))
    parts=[stages[name] for name in ('metadata_total','data_open','data_read','data_seek','data_close')]
    stages['decode_loop_allocation_and_probe_overhead']=residual(total,parts)
    return values,{'total':total,'stages':stages,'scope':'instrumented attribution, not speedup timing'}


def profile_bulk(reader,fields,recordwise):
    stages={}
    original_read,original_copy=analysis_read._pread_exact,analysis_read._copy_records
    def reading(fd,count,offset,**kwargs):
        value,t=measure(lambda:original_read(fd,count,offset,**kwargs))
        accumulate(stages,'data_read',t,returned_bytes=len(value))
        return value
    def copying(raw,offset,count,stride,ghosts,state,selected,output):
        def run():
            if recordwise:
                for row in range(count):
                    original_copy(raw,offset+row*stride,1,stride,ghosts,state,selected,output[row:row+1])
            else: original_copy(raw,offset,count,stride,ghosts,state,selected,output)
        _,t=measure(run)
        accumulate(stages,'decode_validate_copy',t,records=count,copy_groups=count if recordwise else 1)
    with patch.object(analysis_read,'_pread_exact',reading),patch.object(analysis_read,'_copy_records',copying):
        (values,stats),total=measure(lambda:analysis_read.read_full_interiors(reader,fields))
    stages['selection_headers_allocation_and_probe_overhead']=residual(total,stages.values())
    return values,{'total':total,'stages':stages,'reader':stats,'scope':'instrumented attribution, not speedup timing'}


def setup_profile(path):
    stages={}
    def wrap(name,original):
        def call(*args,**kwargs):
            value,t=measure(lambda:original(*args,**kwargs))
            accumulate(stages,name,t)
            return value
        return call
    targets=((os,'open','file_open'),(os,'close','file_close'),
             (amrvac_dat,'read_amrvac_v5_index','checked_index'),
             (amrvac_dat,'bind_amrvac_v5_forest','file_forest_binding'),
             (amrvac_dat_reader,'make_amrvac_v5_ordinary_block_reader','reader_factory'),
             (providers,'make_source','source_validation_and_consumer_geometry'))
    with ExitStack() as stack:
        for module,attribute,name in targets:
            stack.enter_context(patch.object(module,attribute,wrap(name,getattr(module,attribute))))
        def run():
            with open_source(path,field_names=['b1','b2','b3']) as source:
                return {'source_resident_bytes':source.resident_bytes,'consumer_geometry_bytes':source.mesh.nbytes}
        storage,total=measure(run)
    stages['request_context_and_probe_overhead']=residual(total,stages.values())
    metadata,t=measure(lambda:datio.get_metadata(path))
    del metadata
    return {'total':total,'stages':stages,'storage':storage,'canonical_metadata':t,
            'scope':'warm imports; canonical metadata and checked index have different validation scope'}


def planes(mesh):
    width=mesh.upper-mesh.lower
    return (Plane(mesh.lower+width*[0,0,.5],width*[1,0,0],width*[0,1,0],(128,128)),
            Plane(mesh.lower+width*[0,0,.2],width*[1,0,.3],width*[0,1,.3],(96,80)))


def prepared_core(source,raw,repeats):
    """Use the same raw array; repeat constructors, copy and exchange directly."""
    mesh=source.mesh
    block=np.asarray(mesh.block_shape,dtype=np.uint32)
    root=np.asarray(mesh.roots.shape,dtype=np.uint32)
    flags=(mesh.node_leaves>=0).astype(np.int32)
    ids=np.arange(mesh.leaf_count,dtype=np.int64)
    ids.flags.writeable=False
    probe_ids=np.linspace(0,mesh.leaf_count-1,64,dtype=int)
    reference=analysis.prepare_resident(mesh,root,flags,raw,source.fields)
    expected=reference.values[probe_ids].copy()
    extra=reference.owner_extra_bytes
    rows=[]
    del reference
    gc.collect()
    for repeat in range(repeats):
        forest,t0=measure(lambda:analysis.AMRForest(3,*map(np.uint32,root),flags))
        owner,t1=measure(lambda:analysis.AMRMesh(3,block,root*block,mesh.lower.copy(),mesh.upper.copy(),2,3,forest))
        _,t2=measure(lambda:owner.load_interior_data(raw))
        _,t3=measure(owner.apply_ghost_cells)
        backing=owner.padded_view().view(analysis._OwnerArray)
        backing._allocation_owner=owner
        values=backing.view(np.ndarray)
        values.flags.writeable=False
        ready=PreparedFields(mesh,values,ids,ids,source.fields,2,'canonical-coordinatephase-cont-v1',
                             object(),owner=owner,owner_extra_bytes=extra)
        np.testing.assert_array_equal(ready.values[probe_ids],expected)
        rows.append({'forest_connectivity':t0,'mesh_allocation':t1,'interior_copy':t2,'ghost_exchange':t3})
        if repeat+1<repeats:
            del ready,values,backing,owner,forest
            gc.collect()
    storage={'raw_bytes':raw.nbytes,'prepared_value_bytes':ready.values.nbytes,
        'prepared_retained_upper_bytes':ready.nbytes,'consumer_geometry_bytes':mesh.nbytes,
        'connectivity_array_bytes':sum(np.asarray(a).nbytes for a in
            (forest.neighbor_type,forest.neighbor_index,forest.neighbor_children)),
        'coarse_bytes':mesh.leaf_count*3*8*int(np.prod(block.astype(int)//2+4))}
    return ready,{'runs':rows,'medians':{key:median([r[key] for r in rows]) for key in rows[0]},'storage':storage}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file',type=Path,default=Path('data/weno509_sub_0000.dat'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    result={'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'dirty':True,'platform':platform.platform(),'python':sys.version,'numpy':np.__version__,
        'file':str(args.file),'file_bytes':args.file.stat().st_size,
        'cache':'OS page cache uncontrolled; cyclic order, three formal repetitions',
        'variants':{'A':'canonical reader','B':'bulk reads, recordwise decode','C':'bulk reads and batch decode'}}
    # Fresh processes measure import/open without pretending to flush OS caches.
    child="""import json,time,sys
t=time.perf_counter()
from simesh.analysis import open_source
i=time.perf_counter()
with open_source(sys.argv[1],field_names=['b1','b2','b3']) as source: pass
print(json.dumps({'import_seconds':i-t,'first_source_open_seconds':time.perf_counter()-i}))
"""
    result['fresh_processes']=[json.loads(subprocess.check_output([sys.executable,'-c',child,str(args.file)],text=True)) for _ in range(2)]
    result['setup']=[setup_profile(args.file) for _ in range(3)]
    with analysis_io._open_source(args.file,field_names=['b1','b2','b3']) as (source,index,reader):
        fields=np.array(source.original_field_ids,dtype=np.int64)
        reference,first=measure(lambda:datio.read_blocks_sequential(args.file,fields.tolist()))
        result['first_reference_read']=first
        def read(name):
            if name=='A': return datio.read_blocks_sequential(args.file,fields.tolist())
            if name=='B':
                with recordwise_decode(): return analysis_read.read_full_interiors(reader,fields)[0]
            return analysis_read.read_full_interiors(reader,fields)[0]
        rows={name:[] for name in 'ABC'}
        for repeat in range(3):
            order='ABC'[repeat:]+'ABC'[:repeat]
            for name in order:
                values,t=measure(lambda:read(name))
                np.testing.assert_array_equal(values.view(np.uint64),reference.view(np.uint64))
                rows[name].append(t)
                del values
                gc.collect()
        result['readers']={name:{'runs':values,'median':median(values)} for name,values in rows.items()}
        profiles={}
        for name,call in (('A',lambda:profile_canonical(args.file,fields)),
                          ('B',lambda:profile_bulk(reader,fields,True)),
                          ('C',lambda:profile_bulk(reader,fields,False))):
            values,profiles[name]=call()
            np.testing.assert_array_equal(values.view(np.uint64),reference.view(np.uint64))
            del values
            gc.collect()
        result['reader_attribution']=profiles
        print('A/B/C reading and attribution complete',flush=True)
        ready,result['same_raw_preparation']=prepared_core(source,reference,3)
        del reference
        gc.collect()
    geom=planes(ready.mesh)
    compute=[]
    reference=None
    for _ in range(3):
        derived,t0=measure(lambda:curl(ready))
        a,t1=measure(lambda:sample_plane(derived,geom[0]))
        b,t2=measure(lambda:sample_plane(derived,geom[1]))
        if reference is None: reference=(a.values.copy(),b.values.copy())
        else:
            np.testing.assert_array_equal(a.values,reference[0])
            np.testing.assert_array_equal(b.values,reference[1])
        compute.append({'curl':t0,'axis_slice':t1,'oblique_slice':t2,'derived_bytes':derived.nbytes})
        del derived,a,b
        gc.collect()
    result['same_prepared_compute']={'runs':compute,'medians':{name:median([r[name] for r in compute]) for name in ('curl','axis_slice','oblique_slice')}}
    del ready,source,index,reader
    gc.collect()
    print('same-input preparation and compute complete',flush=True)
    def canonical():
        with open_source(args.file,field_names=['b1','b2','b3']) as source:
            raw=datio.read_blocks_sequential(args.file,list(source.original_field_ids))
            return analysis.prepare_resident(source.mesh,source.mesh.roots.shape,source.mesh.node_leaves>=0,
                raw,source.fields,budget_bytes=2*1024**3-source.resident_bytes)
    def pipeline(name):
        if name=='A': fields=canonical()
        elif name=='B':
            with recordwise_decode(): fields=open_prepared(args.file,field_names=['b1','b2','b3'])
        else: fields=open_prepared(args.file,field_names=['b1','b2','b3'])
        derived=curl(fields)
        a,b=(sample_plane(derived,plane) for plane in planes(fields.mesh))
        return a.values,b.values
    rows={name:[] for name in 'ABC'}
    for repeat in range(3):
        order='ABC'[repeat:]+'ABC'[:repeat]
        for name in order:
            values,t=measure(lambda:pipeline(name))
            for value,expected in zip(values,reference): np.testing.assert_array_equal(value,expected)
            rows[name].append(t)
            print('pipeline',name,repeat,round(t['wall_seconds'],4),flush=True)
            del values
            gc.collect()
    result['pipelines']={name:{'runs':values,'median':median(values)} for name,values in rows.items()}
    result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
