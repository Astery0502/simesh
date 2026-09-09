"""Prepare physical coverage with an explicit, immutable numerical scheme."""

import numpy as np

from .._validation import admit, workers_count
from ..mesh import resolve_selection
from ..fields import publish
from . import exact


def _request(source, fields, region, leaf_ids, scheme, workers, support_capacity, memory_limit,
             output_count=None):
    source.validate()
    workers_count(workers)
    if scheme != "exact-phase":
        if scheme == "coordinate-phase":
            raise ValueError("bounded preparation uses exact-phase; use prepare for coordinate-phase")
        raise ValueError("unknown preparation scheme")
    if workers != 1:
        raise ValueError("exact-phase preparation currently executes with one worker")
    selection = resolve_selection(source.mesh, region, leaf_ids)
    field_ids = source.field_ids(fields)
    block, capacity, scratch = exact.parameters(source.mesh, len(field_ids), support_capacity)
    count = len(selection.leaf_ids) if output_count is None else min(output_count, len(selection.leaf_ids))
    shape = (count, *(block+4), len(field_ids))
    required = (source.mesh.nbytes + source.nbytes + source.read_footprint(capacity,len(field_ids)) +
                scratch + int(np.prod(shape))*8 + 8*(source.mesh.leaf_count+len(selection.leaf_ids)))
    admit(required, memory_limit, "preparation")
    return selection, field_ids, block, capacity, shape, required


def prepare(source, fields=None, *, region=None, leaf_ids=None, scheme,
            workers=1, memory_limit=None, support_capacity=128, backend="threadpool", plan=None):
    """Prepare detached two-halo fields over complete selected leaves.

    Region support is read from the original mesh; its edge is not a physical
    boundary. Field/leaf order is preserved. No product escapes a failed fill.
    Coordinate-phase requires complete coverage and keeps physical storage in
    SFC order; the leaf directory addresses other requested target orders.
    """
    if plan is not None:
        from .plans import FillPlan
        if (not isinstance(plan,FillPlan) or scheme!='exact-phase' or region is not None or
                leaf_ids is not None or workers!=1 or backend!='threadpool'):
            raise ValueError("plan owns exact-phase coverage and currently executes with one threadpool worker")
        return plan.prepare(source,fields,memory_limit=memory_limit)
    if scheme == "coordinate-phase":
        from .coordinate import prepare as prepare_coordinate
        return prepare_coordinate(source, fields, region=region, leaf_ids=leaf_ids,
                                  workers=workers, backend=backend, memory_limit=memory_limit)
    if backend != "threadpool":
        raise ValueError("exact-phase preparation currently uses the threadpool serial path")
    selection, field_ids, block, capacity, shape, required = _request(
        source, fields, region, leaf_ids, scheme, workers, support_capacity, memory_limit)
    values = np.empty(shape, dtype=float)
    workspace = exact.Workspace.allocate(source.mesh, len(field_ids), block, capacity)
    stats = exact.fill(source, selection, field_ids, values, workspace)
    stats["controlled_upper_bytes"] = required
    del workspace
    return publish(source.mesh, values, selection, tuple(source.fields[i] for i in field_ids),
                   2, 2, exact.SCHEME, source.identity, stats,
                   value_identity=source.value_identity(field_ids,exact.SCHEME))
