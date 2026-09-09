"""Bounded independent preparation tasks for file-backed dense curl delivery.

Spawned processes are optional. Call from an importable, __main__-guarded script.
No mutable provider/pool crosses workers; all returned ranges are owned arrays.
"""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
import multiprocessing
import os
import resource
import sys
import time
import numpy as np

from simesh.analysis.fields import PreparedFields, PreparedPool, _footprint
from simesh.analysis.derivatives import curl


def _identity(path):
    s=os.stat(path)
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


def _curl_job(path, names, units, expected, first, last, batch_size, support_capacity):
    from simesh.amrvac.analysis_io import open_source
    start=time.perf_counter()
    if _identity(path)!=expected:
        raise OSError('analysis source changed before parallel preparation')
    with open_source(path,field_names=names,field_units=units,support_capacity=support_capacity) as source:
        source.validate_values()
        if _identity(path)!=expected:
            raise OSError('analysis source changed while opening parallel preparation')
        opened=time.perf_counter()
        stats={'source_open_seconds':opened-start,'fill_seconds':0.,'packing_seconds':0.,
               'selected_load_count':0,'derivative_seconds':0.,'output_copy_seconds':0.}
        original=source.fill
        def fill(*args):
            result=original(*args)
            for name in ('packing_seconds','selected_load_count'):
                stats[name]+=result.get(name,0)
            stats['fill_seconds']+=result.get('total_seconds',0.)
            return result
        source=replace(source,fill=fill)
        pool=PreparedPool(source,[0,1,2],batch_size)
        target=np.empty((last-first,*(n+2 for n in source.mesh.block_shape),3))
        try:
            for lower in range(first,last,pool.capacity):
                upper=min(lower+pool.capacity,last)
                with pool.borrow(np.arange(lower,upper,dtype=np.int64)) as primary:
                    stamp=time.perf_counter()
                    derived=curl(primary)
                    stats['derivative_seconds']+=time.perf_counter()-stamp
                    stamp=time.perf_counter()
                    target[lower-first:upper-first]=derived.values
                    stats['output_copy_seconds']+=time.perf_counter()-stamp
                    definitions=derived.fields
                    del derived
            source.validate_values()
            if _identity(path)!=expected:
                raise OSError('analysis source changed during parallel preparation')
        finally:
            pool.close()
    stats['job_seconds']=time.perf_counter()-start
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    return first,target,definitions,stats,os.getpid(),rss


def global_curl_file(path,*,field_names=('b1','b2','b3'),field_units=None,
                     backend='process',workers=2,task_size=512,batch_size=256,
                     support_capacity=256,output=None,budget_bytes=2*1024**3):
    """Compute all native curl values through bounded independent file tasks.

    `backend` selects a process or thread pool with private providers. Full
    caller/owned output, in-flight results, IPC copies and per-worker preparation
    are admitted together. A failure returns no product; completed caller-sink
    ranges may remain. The source must be unchanged throughout the call.
    """
    from simesh.amrvac.analysis_io import open_source
    if backend not in ('process','thread') or type(workers) is not int or not 1<=workers<=4:
        raise ValueError('choose process/thread preparation and one to four workers')
    if any(type(n) is not int or n<1 for n in (task_size,batch_size,support_capacity)):
        raise ValueError('task, preparation and support capacities must be positive integers')
    if isinstance(field_names,str) or len(field_names)!=3:
        raise ValueError('parallel curl requires three ordered vector field names')
    field_names=tuple(field_names)
    path=os.path.abspath(os.fspath(path))
    started=time.perf_counter()
    with open_source(path,field_names=field_names,field_units=field_units,
                     support_capacity=support_capacity,budget_bytes=budget_bytes) as source:
        expected=_identity(path)
        source.validate_values()
        if len({f.units for f in source.fields})!=1:
            raise ValueError('curl vector components require matching units')
        # Jobs bind the resolved immutable request, not caller-owned mappings.
        units={definition.name:definition.units for definition in source.fields}
        mesh=source.mesh
        count=mesh.leaf_count
        task_size=min(task_size,count)
        batch_size=min(batch_size,task_size)
        shape=(count,*(n+2 for n in mesh.block_shape),3)
        bytes_per_leaf=8*int(np.prod(shape[1:]))
        output_bytes=count*bytes_per_leaf
        task_bytes=task_size*bytes_per_leaf
        _,worker_base=_footprint(source,batch_size,np.arange(3,dtype=np.int64),2)
        worker_base+=64*(batch_size+count)
        # Per worker: provider/pool, derived batch, task target, serialization,
        # transport/decoding and completed parent result, plus selector bounds.
        # Threads use the same conservative envelope for a matched comparison.
        required=(source.mesh.nbytes+source.resident_bytes+output_bytes+32*count+
                  workers*(worker_base+batch_size*bytes_per_leaf+4*task_bytes+64*count))
        if required>budget_bytes:
            raise MemoryError(f'parallel curl needs {required} controlled bytes, budget {budget_bytes}')
        if output is not None:
            if (not isinstance(output,np.ndarray) or output.dtype!=np.float64 or output.shape!=shape or
                    not output.flags.c_contiguous or not output.flags.writeable):
                raise ValueError(f'output must be writable contiguous float64 {shape}')
            if any(np.shares_memory(output,a) for a in source.memory_arrays):
                raise ValueError('output must not alias source backing')
        target=np.empty(shape) if output is None else output
        executor=(ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))
                  if backend=='process' else ThreadPoolExecutor(max_workers=workers))
        pending=set()
        cursor=0
        stats={'jobs':0,'result_copy_seconds':0.,'wait_seconds':0.,'worker_totals':{},
               'worker_peak_rss_bytes':{},'worker_job_seconds_by_pid':{}}
        with executor:
            try:
                while cursor<count or pending:
                    while cursor<count and len(pending)<workers:
                        last=min(cursor+task_size,count)
                        pending.add(executor.submit(_curl_job,path,field_names,units,
                            expected,cursor,last,batch_size,support_capacity))
                        cursor=last
                    stamp=time.perf_counter()
                    done,pending=wait(pending,return_when=FIRST_COMPLETED)
                    stats['wait_seconds']+=time.perf_counter()-stamp
                    for future in done:
                        first,values,definitions,worker_stats,pid,rss=future.result()
                        stamp=time.perf_counter()
                        target[first:first+len(values)]=values
                        stats['result_copy_seconds']+=time.perf_counter()-stamp
                        stats['jobs']+=1
                        for key,value in worker_stats.items():
                            stats['worker_totals'][key]=stats['worker_totals'].get(key,0)+value
                        stats['worker_peak_rss_bytes'][pid]=max(stats['worker_peak_rss_bytes'].get(pid,0),rss)
                        stats['worker_job_seconds_by_pid'][pid]=stats['worker_job_seconds_by_pid'].get(pid,0)+worker_stats['job_seconds']
                        del values
                    # Completed futures retain their result; release them before
                    # admitting more jobs so the in-flight bound remains real.
                    del future,done
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        source.validate_values()
        if _identity(path)!=expected:
            raise OSError('analysis source changed before complete parallel publication')
        ids=np.arange(count,dtype=np.int64)
        ids.flags.writeable=False
        values=target.view()
        values.flags.writeable=False
        stats.update(total_seconds=time.perf_counter()-started,controlled_upper_bytes=required,
                     output_bytes=output_bytes,backend=backend,workers=workers,task_size=task_size,
                     batch_size=batch_size,support_capacity=support_capacity)
        return PreparedFields(mesh,values,ids,ids,definitions,1,
            source.strategy+'/centered-extended',source.identity,stats,owner=target)
