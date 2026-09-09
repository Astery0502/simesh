"""Explicit bounded preparation; borrowed storage is outside direct consumers."""

import numpy as np

from .fields import _Lease, publish
from .mesh import Selection
from .preparation import _request, exact


def iter_prepared(source, fields=None, *, region=None, leaf_ids=None, scheme,
                  batch_size=128, support_capacity=128, memory_limit=None):
    """Yield scoped prepared batches in requested order using bounded storage.

    Each descriptor and its NumPy views expire on iterator advance or close.
    Consume synchronously or copy/write values before advancing. No eviction
    or automatic source access happens while a batch is being consumed.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    selection, field_ids, block, capacity, shape, required = _request(
        source, fields, region, leaf_ids, scheme, 1, support_capacity, memory_limit,
        output_count=batch_size)
    backing = np.empty(shape, dtype=float)
    workspace = exact.Workspace.allocate(source.mesh, len(field_ids), block, capacity)
    definitions = tuple(source.fields[i] for i in field_ids)
    for first in range(0, len(selection.leaf_ids), batch_size):
        ids = selection.leaf_ids[first:first+batch_size]
        selected = Selection(source.mesh, ids, selection.requested_bounds)
        output = backing[:len(ids)].view()
        stats = exact.fill(source, selected, field_ids, output, workspace)
        stats["controlled_upper_bytes"] = required
        lease = _Lease()
        product = publish(source.mesh, output, selected, definitions, 2, 2,
                          exact.SCHEME, source.identity, stats, lease,
                          value_identity=source.value_identity(field_ids,exact.SCHEME))
        try:
            yield product
        finally:
            lease.active = False


from ._pool import PreparedPool, CurlPool


def iter_traces_bounded(pool, seeds, *, seed_ids=None, step, max_steps=1000,
                        max_length=np.inf, null_threshold=0., direction=1, workers=1,
                        seed_batch=256, trajectories=False, twist=False,
                        memory_limit=None, backend="threadpool", schedule="static"):
    """Explicit missing-coverage coordination around the shared RK state machine."""
    from contextlib import nullcontext
    from .tracing import _validate_inputs, _new_state, _advance_state, _result, _trace_batch_bytes, Termination
    from ._validation import admit, remaining
    from ._execution import native_dispatch, worker_context
    coupled = isinstance(pool,CurlPool)
    primary = pool.primary if coupled else pool
    if not isinstance(primary,PreparedPool) or primary._closed:
        raise ValueError("bounded tracing requires an open PreparedPool or CurlPool")
    if len(primary.fields)!=3 or len({f.units for f in primary.fields})!=1:
        raise ValueError("tracing requires three ordered vector components with common units")
    seeds,seed_ids = _validate_inputs(seeds,seed_ids,step,max_steps,max_length,null_threshold,
                                    direction,workers,seed_batch,trajectories,twist)
    native,dispatch = native_dispatch(backend,schedule)
    seed_batch = min(seed_batch,primary.capacity)
    reserve = _trace_batch_bytes(seeds,seed_ids,min(seed_batch,len(seeds)),max_steps,trajectories)
    temporary = None
    if twist and not coupled:
        pool = temporary = CurlPool(primary,memory_limit=remaining(memory_limit,reserve))
    if not twist:
        pool = primary
    admit(pool.controlled_bytes+reserve,memory_limit,"bounded trace")
    context = nullcontext(None) if native else worker_context(workers)
    try:
        with context as executor:
            for first in range(0,len(seeds),seed_batch):
                last = min(first+seed_batch,len(seeds))
                state = _new_state(primary.mesh,seeds[first:last],seed_ids[first:last],
                                   max_steps,max_length,trajectories,twist)
                while np.any(state.status==Termination.RUNNING):
                    with pool.borrow(primary.resident_leaf_ids) as product:
                        field,companion = product if twist else (product,None)
                        _advance_state(field,companion,state,step=step,max_steps=max_steps,max_length=max_length,
                            null_threshold=null_threshold,direction=direction,workers=workers,executor=executor,
                            native=native,dispatch=dispatch)
                    missing = np.unique(state.requested[(state.status==Termination.RUNNING)&(state.requested>=0)])
                    if len(missing):
                        with pool.borrow(missing):
                            pass
                    elif np.any(state.status==Termination.RUNNING):
                        raise RuntimeError("tracer made no progress without a preparation request")
                yield _result(state,trajectories,twist)
                del state
    finally:
        if temporary is not None:
            temporary.close()


def trace_bounded(pool, seeds, *, max_steps=1000, trajectories=False, twist=False,
                  memory_limit=None, **kwargs):
    """Collect raw TraceResult batches from an open PreparedPool; controls follow iter_traces_bounded.
    """
    from .tracing import _collect, _trace_output_bytes
    from ._validation import remaining
    if type(max_steps) is not int or max_steps<0 or type(trajectories) is not bool or type(twist) is not bool:
        raise ValueError("invalid trace output settings")
    output = _trace_output_bytes(len(seeds),max_steps,trajectories,twist)
    batches = iter_traces_bounded(pool,seeds,max_steps=max_steps,trajectories=trajectories,
        twist=twist,memory_limit=remaining(memory_limit,output),**kwargs)
    return _collect(batches,len(seeds),max_steps,trajectories,twist)


def retrace_bounded(pool,result,selected_seed_ids,**kwargs):
    """Reintegrate selected original seed IDs; controls follow trace_bounded, without checkpoint resume.
    """
    from .tracing import _retrace_inputs
    seeds,ids,limit=_retrace_inputs(result,selected_seed_ids,kwargs.pop('memory_limit',None))
    kwargs.pop('trajectories',None)
    return trace_bounded(pool,seeds,seed_ids=ids,trajectories=True,memory_limit=limit,**kwargs)


def _tile_los(pool,origins,direction,near,far,component,step_fraction,quadrature,max_samples,
              workers,executor,native,dispatch,touch):
    from .projection import _new_rays,_advance_rays,_ray_arrays,LOSStatus
    state=_new_rays(pool.mesh,origins,direction,near,far)
    while np.any(state.status==LOSStatus.RUNNING):
        with pool.borrow(pool.resident_leaf_ids,touch=touch) as fields:
            _advance_rays(fields,state,component,step_fraction,quadrature,max_samples,
                          workers,executor,native,dispatch)
        missing=np.unique(state.requested[(state.status==LOSStatus.RUNNING)&(state.requested>=0)])
        if len(missing):
            with pool.borrow(missing):
                pass
        elif np.any(state.status==LOSStatus.RUNNING):
            raise RuntimeError("LOS made no progress without a preparation request")
    return _ray_arrays(state)


def integrate_los_views_bounded(pool,planes,directions,**kwargs):
    """Integrate equal-shaped scalar views through an open PreparedPool.

    Controls follow [integrate_los_views][simesh.integrate_los_views], except capacity
    is supplied by the pool; input preparation remains explicitly coordinated.
    """
    from .projection import _integrate_views
    if not isinstance(pool,PreparedPool) or pool._closed:
        raise ValueError("bounded LOS requires an open PreparedPool")
    return _integrate_views(planes,directions,pool.fields,pool.controlled_bytes,
        lambda *args:_tile_los(pool,*args),capacity=pool.capacity,**kwargs)


def integrate_los_bounded(pool,plane,direction,**kwargs):
    """Return one raw LOSResult through an open PreparedPool.

    Controls follow [integrate_los_views_bounded][simesh.bounded.integrate_los_views_bounded].
    """
    return integrate_los_views_bounded(pool,[plane],[direction],**kwargs)[0]


def _sample_pool(pool,points,workers,executor):
    from .operators.sampling import _sample
    from .fields import require_continuous
    owners=pool.mesh.locate(points)
    values=np.empty((len(points),len(pool.fields)))
    valid=np.empty(len(points),bool)
    pending=[(0,len(points))]
    while pending:
        first,last=pending.pop()
        ids=np.unique(owners[first:last])
        ids=ids[ids>=0]
        if len(ids)>pool.capacity:
            middle=(first+last)//2
            pending.extend(((middle,last),(first,middle)))
            continue
        with pool.borrow(ids) as fields:
            selected=np.asarray(require_continuous(fields),dtype=np.int64)
            _sample(fields,points[first:last],workers,executor,selected,
                    (values[first:last],owners[first:last],valid[first:last]))
    return values,owners,valid


def sample_plane_bounded(pool,plane,*,tile_rows=64,workers=1,memory_limit=None):
    """Sample all pool components on a pixel-center Plane, returning a raw SliceResult.
    """
    from .slices import _plane_result
    from ._execution import worker_context
    if not isinstance(pool,PreparedPool) or pool._closed:
        raise ValueError("bounded sampling requires an open PreparedPool")
    with worker_context(workers) as executor:
        return _plane_result(plane,len(pool.fields),pool.controlled_bytes,
            lambda points,w,e:_sample_pool(pool,points,w,e),tile_rows=tile_rows,
            workers=workers,executor=executor,memory_limit=memory_limit)


def iter_uniform_bounded(pool,resolution,*,bounds=None,tile_rows=64,workers=1,memory_limit=None):
    """Yield owned (z_index, SliceResult) outputs while bounding prepared input slots.
    """
    from .slices import _uniform_geometry,_plane_result,Plane
    from ._validation import remaining
    from ._execution import worker_context
    if not isinstance(pool,PreparedPool) or pool._closed:
        raise ValueError("bounded uniform output requires an open PreparedPool")
    (nx,ny,nz),lower,upper=_uniform_geometry(pool.mesh,resolution,bounds)
    width=upper-lower
    limit=remaining(memory_limit,nx*ny*(8*len(pool.fields)+9))
    with worker_context(workers) as executor:
        for iz in range(nz):
            origin=lower.copy()
            origin[2]=lower[2]+(iz+.5)*(width[2]/nz)
            plane=Plane(origin,[width[0],0.,0.],[0.,width[1],0.],(nx,ny))
            yield iz,_plane_result(plane,len(pool.fields),pool.controlled_bytes,
                lambda points,w,e:_sample_pool(pool,points,w,e),tile_rows=tile_rows,
                workers=workers,executor=executor,memory_limit=limit)


def global_curl(source,fields=('b1','b2','b3'),*,batch_size=128,support_capacity=128,output=None,memory_limit=None):
    """Deliver all exact-phase native curl values into owned or caller backing.

    A failed call returns no field product. A caller-supplied sink may contain
    earlier completed ranges; it is not a certified complete result on failure.
    """
    import time
    from .fields import publish
    from .mesh import Selection
    from .operators.derivatives import curl
    from ._validation import admit,remaining
    if type(batch_size) is not int or batch_size<1:
        raise ValueError("batch_size must be positive")
    chosen=source.field_ids(fields)
    if len(chosen)!=3 or len({source.fields[i].units for i in chosen})!=1:
        raise ValueError("global curl requires three ordered components with common units")
    mesh=source.mesh
    shape=(mesh.leaf_count,*(n+2 for n in mesh.block_shape),3)
    output_bytes=8*int(np.prod(shape))
    reserve=output_bytes+mesh.leaf_count*32
    request=_request(source,chosen,None,None,'exact-phase',1,support_capacity,None,output_count=batch_size)
    required=request[-1]+reserve
    admit(required,memory_limit,"global curl")
    if output is not None:
        if (not isinstance(output,np.ndarray) or output.shape!=shape or output.dtype!=np.float64 or
                not output.flags.c_contiguous or not output.flags.writeable):
            raise ValueError("output must be a matching writable contiguous float64 array")
        if any(np.shares_memory(output,a) for a in source._memory_arrays):
            raise ValueError("output must not alias immutable source storage")
    target=np.empty(shape) if output is None else output
    start=time.perf_counter()
    definitions=None
    for primary in iter_prepared(source,chosen,scheme='exact-phase',batch_size=batch_size,
                                 support_capacity=support_capacity,memory_limit=remaining(memory_limit,reserve)):
        first=int(primary.leaf_ids[0])
        derived=curl(primary,output=target[first:first+len(primary.leaf_ids)],memory_limit=memory_limit)
        definitions=derived.fields
        del derived
    source.validate()
    ids=np.arange(mesh.leaf_count,dtype=np.int64)
    view=target.view()
    return publish(mesh,view,Selection(mesh,ids),definitions,1,1,
        exact.SCHEME+'/centered-extended',source.identity,
        {'total_seconds':time.perf_counter()-start,'controlled_upper_bytes':required,'output_bytes':output_bytes},
        derivation=('curl',source.value_identity(chosen,exact.SCHEME),
                    tuple(source.fields[i] for i in chosen),(0,1,2),exact.SCHEME))
