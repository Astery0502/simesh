"""Compare selected delivery against 0bcbe2e in separate interpreter processes.

Set PYTHONPATH to the desired distribution's src directory. Run --make-fixture
once in a separate process so fixture construction cannot contaminate peaks.
Reported tracemalloc peaks span each entire operation; RSS is process high-water
storage, including setup and libraries. Neither is a large-input acceptance test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc

import numpy as np
import simesh as sm
from simesh import applications as app


def digest(arrays):
    result=hashlib.sha256()
    for array in arrays:
        # Hash after the measured interval; contiguous conversion is not part of
        # the scientific workflow's allocation measurement.
        result.update(np.ascontiguousarray(array).view('u1'))
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path',type=Path)
    parser.add_argument('--mode',choices=('read','coordinate','exact','plan','cache','sample','profile','uniform','lines'),default='read')
    parser.add_argument('--baseline',action='store_true')
    parser.add_argument('--make-fixture',action='store_true')
    args=parser.parse_args()
    if args.make_fixture:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
        from fixtures import write_dat
        mesh=sm.mesh_from_forest((8,4,4),np.ones(128,dtype=bool),
            lower=(0,0,0),upper=(2,1,1),block_shape=(24,24,24))
        raw=np.empty((128,8,24,24,24))
        local=np.indices(mesh.block_shape)+.5
        for leaf in range(128):
            xyz=mesh.bounds[leaf,0,:,None,None,None]+local*mesh.spacing[leaf,:,None,None,None]
            x,y,z=xyz
            for i in range(8):
                raw[leaf,i]=1+i+.2*x+.3*y+.4*z
        write_dat(args.path,mesh,raw)
        return
    source=sm.open_amrvac(args.path)
    names=('b8','b1')
    ready=plan=points=lines=None
    if args.mode in ('sample','profile','uniform','lines'):
        ready=sm.prepare(source,scheme='coordinate-phase')
    if args.mode == 'plan':
        plan=sm.plan_preparation(source.mesh,scheme='exact-phase',support_capacity=57)
    if args.mode == 'sample':
        points=np.random.default_rng(72).uniform([0,0,0],[2,1,1],size=(1000000,3))
    if args.mode == 'profile':
        n,length=512,512
        seed_xyz=np.random.default_rng(71).uniform([.1,.1,.3],[1.9,.9,.7],size=(n,3))
        seeds=sm.PointSet(seed_xyz,ids=np.arange(n,dtype=np.int64))
        paths=np.broadcast_to(seed_xyz[:,None,None,:],(n,2,length,3)).copy()
        paths[:,0,:,2]-=np.linspace(0,.1,length)
        paths[:,1,:,2]+=np.linspace(0,.1,length)
        lines=sm.LineSet(seeds,paths.reshape(-1,3),np.arange(2*n+1,dtype='i8')*length,
                         np.full((n,2),int(sm.Termination.MAX_STEPS),dtype='i8'),None)
    if args.mode == 'lines':
        ready=sm.select_fields(ready,(0,1,2))
        points=sm.PointSet(np.random.default_rng(72).uniform([.7,.3,.3],[1.3,.7,.7],size=(2000,3)))
    input_bytes=source.nbytes+source.mesh.nbytes
    if ready is not None:
        input_bytes+=ready.nbytes
    if plan is not None:
        input_bytes+=plan.nbytes
    if points is not None:
        input_bytes+=points.nbytes
    if lines is not None:
        input_bytes+=lines.nbytes+seed_xyz.nbytes
    tracemalloc.start()
    started=time.perf_counter()
    retained=0
    if args.mode == 'read':
        result=sm.read_fields(source,names)
        arrays=(result.values,)
        retained=result.nbytes
    elif args.mode in ('coordinate','exact','plan'):
        result=sm.prepare(source,('b1','b2','b3'),scheme='coordinate-phase' if args.mode=='coordinate' else 'exact-phase',
                          **({'plan':plan} if plan is not None else {'support_capacity':57}))
        arrays=(result.values,)
        retained=result.nbytes
    elif args.mode == 'cache':
        cache=sm.cache_source(source,capacity=64,**({} if args.baseline else {'fields':names}))
        result=sm.read_fields(cache,names,leaf_ids=np.arange(64))
        arrays=(result.values,)
        retained=result.nbytes+cache.nbytes-source.nbytes
    elif args.mode == 'sample':
        if args.baseline:
            sampled,owners,valid=sm.sample(ready,points)
            result=sampled[:,[7,0]]
            del sampled
        else:
            result,owners,valid=sm.sample(ready,points,components=names)
        arrays=(result,owners,valid)
        retained=sum(a.nbytes for a in arrays)
    elif args.mode == 'profile':
        result=sm.sample_line_profiles(ready,lines,names,point_batch=65536)
        arrays=(result.values,result.arclength,result.owners,result.valid,result.finite)
        retained=result.nbytes-lines.nbytes
    elif args.mode == 'uniform':
        if args.baseline:
            subset=sm.select_fields(ready,names)
            result=app.uniform_grid(subset,(128,128,128),tile_rows=32)
            del subset
        else:
            result=app.uniform_grid(ready,(128,128,128),components=names,tile_rows=32)
        arrays=(result.values,result.valid)
        retained=sum(a.nbytes for a in arrays)
    else:
        if args.baseline:
            result=app.trace(ready,points,step=.002,max_steps=200,seed_batch=64)
            arrays=(result.positions,result.offsets,result.termination)
            retained=result.nbytes
        else:
            # Consume shards by a bounded checksum; never concatenate payloads.
            checksum=hashlib.sha256()
            point_count=0
            for result in app.iter_lines(ready,points,step=.002,max_steps=200,seed_batch=64):
                checksum.update(result.positions.view('u1'))
                point_count+=len(result.positions)
            arrays=()
            retained=result.nbytes
    seconds=time.perf_counter()-started
    current,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform!='darwin':
        rss*=1024
    if args.mode=='lines':
        if args.baseline:
            value_digest=digest((result.positions,))
            point_count=len(result.positions)
        else:
            value_digest=checksum.hexdigest()
    else:
        value_digest=digest(arrays)
    print(json.dumps(dict(mode=args.mode,baseline=args.baseline,seconds=seconds,
        traced_peak_bytes=peak,traced_retained_bytes=current,result_bytes=retained,
        preexisting_input_bytes=input_bytes, controlled_live_peak_bytes=input_bytes+peak,
        process_peak_rss_bytes=rss,digest=value_digest,
        preparation_stats=getattr(result,'preparation_stats',None)),default=int))
    source.close()


if __name__=='__main__':
    main()
