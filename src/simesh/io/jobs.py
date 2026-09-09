"""Independent file tasks for exact-phase full-domain curl delivery."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
import os
import time
import numpy as np

from .amrvac import open_amrvac
from ..bounded import iter_prepared
from ..preparation import _request, exact
from ..operators.derivatives import curl
from ..fields import publish
from ..mesh import Selection
from .._validation import admit, workers_count


def _identity(path):
    s=os.stat(path)
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns


def _curl_job(path,names,units,expected,first,last,batch_size,support_capacity):
    if _identity(path)!=expected:
        raise OSError("source changed before independent file task")
    started=time.perf_counter()
    with open_amrvac(path,units=units) as source:
        if _identity(path)!=expected:
            raise OSError("source changed while opening file task")
        target=np.empty((last-first,*(n+2 for n in source.mesh.block_shape),3))
        loads=0
        for primary in iter_prepared(source,names,leaf_ids=np.arange(first,last,dtype=np.int64),
                                     scheme='exact-phase',batch_size=batch_size,support_capacity=support_capacity):
            offset=int(primary.leaf_ids[0])-first
            derived=curl(primary,output=target[offset:offset+len(primary.leaf_ids)])
            definitions=derived.fields
            loads+=primary.preparation_stats['selected_load_count']
            del derived
        source.validate()
        if _identity(path)!=expected:
            raise OSError("source changed during independent file task")
    return first,target,definitions,{'seconds':time.perf_counter()-started,'selected_load_count':loads}


def global_curl_file(path,*,fields=('b1','b2','b3'),units=None,backend='process',workers=2,
                     task_size=512,batch_size=256,support_capacity=256,output=None,memory_limit=None):
    """Compute full exact-phase curl with bounded independent file tasks.

    A process backend must be invoked from an importable, __main__-guarded script.
    Every task owns its source and scratch. Outputs are admitted together with
    in-flight task results and conservative serialization/copy reserves.
    """
    workers_count(workers)
    if backend not in ('process','thread'):
        raise ValueError("file task backend must be process or thread")
    if any(type(n) is not int or n<1 for n in (task_size,batch_size,support_capacity)):
        raise ValueError("task and batch capacities must be positive integers")
    path=os.path.abspath(os.fspath(path))
    started=time.perf_counter()
    with open_amrvac(path,units=units,memory_limit=memory_limit) as source:
        chosen=source.field_ids(fields)
        if len(chosen)!=3 or len({source.fields[i].units for i in chosen})!=1:
            raise ValueError("file curl needs three ordered vector components with common units")
        names=tuple(source.fields[i].name for i in chosen)
        # Resolved unique names prevent metadata ambiguity in independent jobs.
        if len(set(names))!=3 or any(sum(f.name==name for f in source.fields)!=1 for name in names):
            raise ValueError("file-task fields require unique names")
        units={source.fields[i].name:source.fields[i].units for i in chosen}
        expected=_identity(path)
        source.validate()
        mesh=source.mesh
        count=mesh.leaf_count
        task_size=min(task_size,count)
        batch_size=min(batch_size,task_size)
        shape=(count,*(n+2 for n in mesh.block_shape),3)
        per_leaf=8*int(np.prod(shape[1:]))
        output_bytes=count*per_leaf
        task_bytes=task_size*per_leaf
        worker_base=_request(source,chosen,None,None,'exact-phase',1,support_capacity,None,
                             output_count=batch_size)[-1]
        required=(mesh.nbytes+source.nbytes+output_bytes+32*count+
                  workers*(worker_base+batch_size*per_leaf+4*task_bytes+64*count))
        admit(required,memory_limit,"parallel file curl")
        if output is not None:
            if (not isinstance(output,np.ndarray) or output.dtype!=np.float64 or output.shape!=shape or
                    not output.flags.c_contiguous or not output.flags.writeable):
                raise ValueError("output must match the writable full native curl layout")
            if any(np.shares_memory(output,a) for a in source._memory_arrays):
                raise ValueError("output must not alias source storage")
        target=np.empty(shape) if output is None else output
        executor=(ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))
                  if backend=='process' else ThreadPoolExecutor(max_workers=workers))
        pending=set()
        cursor=jobs=0
        stats=[]
        with executor:
            try:
                while cursor<count or pending:
                    while cursor<count and len(pending)<workers:
                        last=min(cursor+task_size,count)
                        pending.add(executor.submit(_curl_job,path,names,units,expected,cursor,last,
                                                    batch_size,support_capacity))
                        cursor=last
                    done,pending=wait(pending,return_when=FIRST_COMPLETED)
                    for future in done:
                        first,values,definitions,worker_stats=future.result()
                        target[first:first+len(values)]=values
                        stats.append(worker_stats)
                        jobs+=1
                        del values
                    del future,done
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        source.validate()
        if _identity(path)!=expected:
            raise OSError("source changed before complete file-curl publication")
        ids=np.arange(count,dtype=np.int64)
        return publish(mesh,target.view(),Selection(mesh,ids),definitions,1,1,
            exact.SCHEME+'/centered-extended',source.identity,
            {'total_seconds':time.perf_counter()-started,'controlled_upper_bytes':required,
             'jobs':jobs,'workers':workers,'backend':backend,'worker_stats':stats},
            derivation=('curl',source.value_identity(chosen,exact.SCHEME),
                        tuple(source.fields[i] for i in chosen),(0,1,2),exact.SCHEME))
