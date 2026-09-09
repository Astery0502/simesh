"""Frozen N3 science workflows in a single explicitly selected package."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--dependencies',type=Path,required=True)
    p.add_argument('--flavor',choices=('donor','new'),required=True)
    p.add_argument('--profile',choices=('magnetic','thermal','bounded','file-curl'),required=True)
    p.add_argument('--file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--image-size',type=int,default=500)
    p.add_argument('--backend',choices=('threadpool','openmp'),default='threadpool')
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    if not sys.flags.isolated or not sys.flags.no_site: raise RuntimeError('use -I -S')
    sys.path[:0]=[str(a.source_root.resolve()),str(a.dependencies.resolve())]
    import numpy as np
    started=time.perf_counter()
    import simesh
    assert Path(simesh.__file__).resolve().is_relative_to(a.source_root.resolve())
    old=a.flavor=='donor'
    if old:
        from simesh import analysis as api
        from simesh.analysis.thermal import PROTON_MASS_G
    else:
        api=simesh
        from simesh.physics.thermal import PROTON_MASS_G
    imported=time.perf_counter()
    limit=2*1024**3
    budget={'budget_bytes' if old else 'memory_limit':limit}
    arrays={}
    stats={}
    def digest(value):
        value=np.asarray(value)
        assert value.flags.c_contiguous,(value.shape,value.dtype)
        result=hashlib.sha256(str((value.shape,value.dtype.str)).encode())
        result.update(memoryview(value).cast('B'))
        return result.hexdigest()
    def open_prepared(names):
        if old:
            return api.open_prepared(a.file,field_names=names,workers=a.workers,**budget)
        with api.open_amrvac(a.file,memory_limit=limit) as source:
            return api.prepare(source,names,scheme='coordinate-phase',workers=a.workers,memory_limit=limit)
    def add_trace(prefix,result):
        for name in ('seed_ids','seeds','positions','length','steps','termination','samples','misses','twist','trajectories'):
            value=getattr(result,name,None)
            if value is not None: arrays[prefix+name]=value
    if a.profile=='magnetic':
        ready=open_prepared(('b1','b2','b3'))
        prepared=time.perf_counter()
        mesh=ready.mesh
        q=.05+.9*(np.arange(256)+.5)/256
        x,y=np.meshgrid(q,q,indexing='ij')
        seeds=np.ascontiguousarray(mesh.lower+(mesh.upper-mesh.lower)*np.column_stack((x.ravel(),y.ravel(),np.full(x.size,.05))))
        step=.25*float(mesh.spacing.min())
        lines=api.trace(ready,seeds,step=step,max_steps=1000,workers=a.workers,seed_batch=256,
                        backend=a.backend,**budget)
        traced=time.perf_counter()
        if old:
            bundle=api.with_curl(ready,workers=a.workers,**budget)
            curve=bundle.curl
            trace_fields=bundle
            extra={}
        else:
            curve=api.curl(ready,workers=a.workers,**budget)
            trace_fields=ready
            extra={'curl_field':curve}
        differentiated=time.perf_counter()
        chosen=np.arange(0,len(seeds),1024,dtype=np.int64)
        selected=api.trace(trace_fields,np.ascontiguousarray(seeds[chosen]),seed_ids=chosen,step=step,
            max_steps=1000,twist=True,workers=a.workers,backend=a.backend,**extra,**budget)
        keep=selected.seed_ids[np.argsort(selected.twist)[-3:]]
        again=api.retrace(trace_fields,selected,keep,step=step,max_steps=1000,twist=True,
                          workers=a.workers,backend=a.backend,**extra,**budget)
        finished=time.perf_counter()
        add_trace('long_',lines);add_trace('twist_',selected);add_trace('retrace_',again)
        stats.update(prepare_seconds=prepared-imported,trace_seconds=traced-prepared,
                     curl_seconds=differentiated-traced,diagnostic_seconds=finished-differentiated,
                     seed_count=len(seeds),accepted_steps=int(lines.steps.sum()),
                     requested_max_steps=1000,prepared_bytes=ready.nbytes,derived_bytes=curve.nbytes)
    elif a.profile=='thermal':
        density=open_prepared(('rho',))
        mesh=density.mesh
        raw=np.empty((mesh.leaf_count,1,*mesh.block_shape))
        for leaf in range(mesh.leaf_count):
            z=mesh.bounds[leaf,0,2]+(np.arange(mesh.block_shape[2])+.5)*mesh.spacing[leaf,2]
            z=(z-mesh.lower[2])/(mesh.upper[2]-mesh.lower[2])
            raw[leaf,0]=1.05e6+6e5*np.sin(2*np.pi*z)
        if old:
            from simesh.amrvac.analysis import prepare_resident
            temperature=prepare_resident(mesh,mesh.roots.shape,mesh.node_leaves>=0,raw,
                (api.FieldDefinition('external_T','K'),),budget_bytes=limit-density.nbytes)
        else:
            with api.source_from_arrays(mesh,raw,['external_T'],units='K',copy=False) as source:
                temperature=api.prepare(source,scheme='coordinate-phase',memory_limit=limit-density.nbytes)
        del raw
        state=api.thermal_fields(density,temperature,density_unit_g_cm3=1.4*PROTON_MASS_G*1e9,
            temperature_label='manufactured 0.45--1.65 MK sinusoid along normalized z',**budget)
        del density,temperature
        gc.collect()
        prepared=time.perf_counter()
        for i,d in enumerate(([0.,0.,1.],[.3,.2,1.])):
            plane=api.orthographic_plane(mesh.lower,mesh.upper,d,(a.image_size,a.image_size))
            before=time.perf_counter()
            result=api.integrate_thermal_los(state,plane,d,length_unit_cm=1e8,workers=a.workers,
                                             backend=a.backend,subdivisions=4,**budget)
            stats[f'image_{i}_seconds']=time.perf_counter()-before
            assert result.complete
            for name in ('values','entry','exit','status','samples'):
                arrays[f'image_{i}_'+name]=getattr(result,name)
        small=api.orthographic_plane(mesh.lower,mesh.upper,[.3,.2,1.],(64,64))
        alternative=api.integrate_thermal_los(state,small,[.3,.2,1.],length_unit_cm=1e8,
            order='emissivity-first',workers=a.workers,backend=a.backend,**budget)
        assert alternative.complete
        arrays['emissivity_values']=alternative.values
        arrays['emissivity_status']=alternative.status
        finished=time.perf_counter()
        stats.update(prepare_seconds=prepared-imported,image_size=a.image_size,thermal_bytes=state.nbytes,
                     physical_validation=False,temperature='manufactured 0.45--1.65 MK',length_unit_cm=1e8)
    elif a.profile=='file-curl':
        names={'field_names' if old else 'fields':('b1','b2','b3')}
        result=api.global_curl_file(a.file,workers=a.workers,backend='process',task_size=512,
                                    batch_size=256,support_capacity=256,**names,**budget)
        completed=time.perf_counter()
        mesh=result.mesh
        width=mesh.upper-mesh.lower
        plane=api.Plane(mesh.lower+width*[.1,.1,.5],width*[.8,0,0],width*[0,.8,0],(128,128))
        sliced=api.sample_plane(result,plane,workers=a.workers,**budget)
        finished=time.perf_counter()
        arrays['file_curl']=result.values
        arrays['slice']=sliced.values
        arrays['slice_valid']=sliced.valid
        stats.update(file_curl_seconds=completed-imported,output_bytes=result.nbytes,
                     control=result.preparation_stats['controlled_upper_bytes'])
    else:
        if old:
            context=api.open_source(a.file,field_names=('b1','b2','b3'),support_capacity=128,value_cache_capacity=256)
        else:
            parent=api.open_amrvac(a.file,memory_limit=limit)
            context=api.cache_source(parent,capacity=256,memory_limit=limit)
        with context as source:
            mesh=source.mesh
            width=mesh.upper-mesh.lower
            region=np.array([mesh.lower+.46*width,mesh.lower+.54*width])
            ids=np.flatnonzero(np.all((mesh.bounds[:,1]>region[0])&(mesh.bounds[:,0]<region[1]),axis=1))
            plan=(api.build_fill_plan(source,ids,capacity=128,budget_bytes=128*1024**2) if old else
                  api.plan_preparation(mesh,leaf_ids=ids,scheme='exact-phase',support_capacity=128,memory_limit=128*1024**2))
            built=time.perf_counter()
            first=plan.prepare(source,[2,0],**budget) if old else plan.prepare(source,['b3','b1'],**budget)
            second=plan.prepare(source,[2,0],**budget) if old else plan.prepare(source,['b3','b1'],**budget)
            arrays['planned_first']=first.values;arrays['planned_repeat']=second.values
            prepared=time.perf_counter()
            if old:
                pool=api.PreparedPool(source,[0,1,2],capacity=64,budget_bytes=limit)
                coupled=api.CurlPool(pool,budget_bytes=limit)
                trace_call=api.trace
                los_call=api.integrate_los_views
                uniform_call=api.iter_uniform
            else:
                from simesh.bounded import PreparedPool,CurlPool,trace_bounded,integrate_los_views_bounded,iter_uniform_bounded
                pool=PreparedPool(source,('b1','b2','b3'),capacity=64,scheme='exact-phase',memory_limit=limit)
                coupled=CurlPool(pool,memory_limit=limit)
                trace_call=trace_bounded;los_call=integrate_los_views_bounded;uniform_call=iter_uniform_bounded
            coordinates=np.indices((4,4,2)).reshape(3,-1).T
            seeds=np.ascontiguousarray(mesh.lower+(.47+.06*(coordinates+.5)/[4,4,2])*width)
            result=trace_call(coupled,seeds,step=.125*float(mesh.spacing.min()),max_steps=128,twist=True,
                              trajectories=True,workers=a.workers,backend=a.backend,**budget)
            add_trace('bounded_',result)
            directions=([0.,0.,1.],[.3,.2,1.])
            planes=[api.orthographic_plane(mesh.lower,mesh.upper,d,(8,8)) for d in directions]
            images=los_call(pool,planes,directions,component=2,workers=a.workers,backend=a.backend,**budget)
            for i,image in enumerate(images):
                assert image.complete
                arrays[f'bounded_los_{i}']=image.values
                arrays[f'bounded_los_status_{i}']=image.status
            slabs=[]
            for i,slab in uniform_call(pool,(32,24,4),bounds=region,workers=a.workers,**budget):
                slabs.append(slab.values)
            arrays['bounded_uniform']=np.stack(slabs)
            stats.update(plan_seconds=built-imported,planned_prepare_seconds=prepared-built,plan_bytes=plan.nbytes,
                         prepared_count=pool.prepared_count,derived_count=coupled.derived_count)
            coupled.close();pool.close()
        if not old: parent.close()
        finished=time.perf_counter()
    record=dict(profile=a.profile,flavor=a.flavor,workers=a.workers,backend=a.backend,
                import_seconds=imported-started,file_to_result_seconds=finished-imported,
                startup_to_result_seconds=finished-started,hashes={k:digest(v) for k,v in arrays.items()},stats=stats,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                timing='NumPy ready; include package loading; no output hashing in measured interval')
    a.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('profile','flavor','startup_to_result_seconds','peak_rss_bytes')}),flush=True)


if __name__=='__main__':main()
