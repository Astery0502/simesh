"""Optional retained geometry for the existing exact-phase halo provider.

Plans contain source-leaf IDs, local support ordinals and geometric transfer
boxes. They contain no field values, limiter results, output pointers or runtime
cache slots. Numerical work still uses the retained provider's original kernels.
"""
from dataclasses import dataclass
import sys
import time
import numpy as np

from .fields import PreparedFields
from .mesh import frozen_array, indices


def _freeze(value):
    if isinstance(value, np.ndarray):
        return frozen_array(value, value.dtype)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(x) for x in value)
    return value


def _bytes(value, seen=None):
    """Retained Python objects plus owned ndarray backing, counted once."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, (tuple, list)):
        total += sum(_bytes(x, seen) for x in value)
    return total


def _actions(w, primary_count, selected_count, lo, hi):
    from simesh_rewrite import refined_halo as r
    actions = []
    for primary in range(primary_count):
        for row in range(26):
            if w.physical_masks[primary,row] or not r._nonempty(w.target_lower[row],w.target_upper[row]):
                continue
            lower, upper = w.target_lower[row], w.target_upper[row]
            kind, count = int(w.relation_kinds[primary,row]), int(w.source_counts[primary,row])
            r._prepare_action_rows(w,row,max(count,1))
            if kind == r.RELATION_SAME:
                r.fill_same_level_source_boxes_unchecked(lo,hi,w.action_directions[:1],
                    w.action_target_lower[:1],w.action_target_upper[:1],w.same_source_lower,w.same_source_upper)
                actions.append(_freeze((kind,primary,int(w.source_slots[primary,row,0]),
                                        w.same_source_lower[0],lower,w.same_source_upper[0]-w.same_source_lower[0])))
            elif kind == r.RELATION_FINER:
                w.action_phases[:count] = w.phase_codes[primary,row,:count]
                r.fill_finer_restriction_boxes_unchecked(lo,hi,w.action_directions[:count],
                    w.action_phases[:count],w.action_target_lower[:count],w.action_target_upper[:count],
                    w.finer_target_lower[:count],w.finer_target_upper[:count],w.finer_source_lower[:count],w.finer_source_upper[:count])
                for i in range(count):
                    actions.append(_freeze((kind,primary,int(w.source_slots[primary,row,i]),
                                            w.finer_source_lower[i],w.finer_source_upper[i],w.finer_target_lower[i])))
            elif kind == r.RELATION_COARSER:
                w.action_phases[0] = w.phase_codes[primary,row,0]
                r.fill_coarser_workspace_boxes_unchecked(lo,hi,w.action_directions[:1],w.action_phases[:1],
                    w.action_target_lower[:1],w.action_target_upper[:1],*w.cwp_outputs)
                nt,nr = r.fill_coarser_slope_support_plan_unchecked(lo,hi,primary,w.action_phases[0],
                    w.action_directions[0],w.relation_kinds[primary],w.physical_masks[primary],w.source_slots[primary],
                    *(x[0] for x in w.cwp_outputs),*w.csp_outputs)
                # Unchecked application only traverses the active records.
                actions.append(_freeze((kind,primary,lower,upper,tuple(x[0] for x in w.cwp_outputs),
                                        int(nt),int(nr),tuple(x[:nr] for x in w.csp_outputs))))
    # All nonphysical base regions complete before any physical widening.
    for primary in range(primary_count):
        count = r._prepare_primary_pwa(w,primary,lo,hi)
        if count:
            arrays = tuple(getattr(w,"pwa_"+name)[:count] for name in
                ("logical_lower","logical_upper","offsets","directions","masks",
                 "base_lower","base_upper","target_lower","target_upper"))
            actions.append(_freeze((r.RELATION_PHYSICAL,primary,arrays)))
    return tuple(actions)


def _execute(actions, payload, coarse, lo, hi, modes, normals):
    from simesh_rewrite import refined_halo as r
    for action in actions:
        kind, target = action[:2]
        if kind == r.RELATION_SAME:
            _,_,src,src_lo,dst_lo,extent = action
            r.copy_region_into_unchecked(payload[src:src+1],src_lo,payload[target:target+1],dst_lo,extent)
        elif kind == r.RELATION_FINER:
            _,_,src,src_lo,src_hi,dst_lo = action
            r.restrict_cartesian_2to1_into_unchecked(payload[src:src+1],src_lo,src_hi,payload[target:target+1],dst_lo)
        elif kind == r.RELATION_COARSER:
            _,_,dst_lo,dst_hi,cwp,nt,nr,csp = action
            r._apply_coarser_workspace_plan_unchecked(payload,coarse,lo,hi,cwp[4],cwp[5],nt,nr,*csp,modes,normals)
            r.prolong_cartesian_2to1_into_unchecked(coarse,cwp[6],payload[target:target+1],dst_lo,dst_hi,lo)
        else:
            r.apply_cartesian_physical_widening_unchecked(payload,target,*action[2],modes,normals)


@dataclass(frozen=True, eq=False)
class FillPlan:
    mesh: object
    strategy: str
    leaf_ids: np.ndarray
    chunks: tuple
    capacity: int
    nbytes: int
    build_seconds: float

    def prepare(self, source, field_ids, *, budget_bytes=2*1024**3):
        """Bind fresh fields/private execution storage and detach complete output.

        Reuse across values/field sets is allowed on this immutable mesh and
        strategy. A different mesh, halo, boundary rule, selection or partition
        needs a new plan. Any read/fill failure publishes no product. No shared
        mutable workspace is retained, so the plan itself is safe to share.
        """
        if source.mesh is not self.mesh or source.strategy != self.strategy or source.read_interiors is None:
            raise ValueError("fill plan mesh/strategy does not match source")
        if source.validate_values is not None:
            source.validate_values()
        fields = indices(field_ids,len(source.fields),"field_ids")
        if not len(fields):
            raise ValueError("at least one field is required")
        block = np.array(self.mesh.block_shape,dtype=np.int64)
        padded = block+4
        k = len(fields)
        shape = (len(self.leaf_ids),*padded,k)
        scratch = 8*k*(self.capacity*(int(np.prod(padded))+int(np.prod(block)))+int(np.prod(block+1)))
        required = (self.nbytes+self.mesh.nbytes+source.resident_bytes+8*int(np.prod(shape))+
                    8*(self.mesh.leaf_count+len(self.leaf_ids))+scratch+source.scratch_bytes(fields,0))
        if required > budget_bytes:
            raise MemoryError(f"planned preparation needs {required} controlled bytes")
        values = np.empty(shape)
        payload = np.empty((self.capacity,k,*padded))
        raw = np.empty((self.capacity,k,*block))
        coarse = np.empty((1,k,*(block+1)))
        lo,hi = np.full(3,2,dtype=np.int64),block+2
        modes,normals = np.zeros((k,6),dtype=np.uint8),np.full(3,-1,dtype=np.int64)
        offset = loads = 0
        read_value_bytes = requested_value_bytes = 0
        read_seconds = execute_seconds = packing_seconds = 0.
        started = time.perf_counter()
        for ids, count, actions in self.chunks:
            n = len(ids)
            t = time.perf_counter()
            stats = source.read_interiors(ids,fields,raw[:n])
            requested = n*k*8*int(np.prod(block))
            read_value_bytes += stats.get("read_value_bytes",requested)
            requested_value_bytes += stats.get("requested_value_bytes",requested)
            payload[:n,:,2:hi[0],2:hi[1],2:hi[2]] = raw[:n]
            read_seconds += time.perf_counter()-t
            t = time.perf_counter()
            _execute(actions,payload[:n],coarse,lo,hi,modes,normals)
            execute_seconds += time.perf_counter()-t
            t = time.perf_counter()
            values[offset:offset+count] = np.moveaxis(payload[:count],1,-1)
            packing_seconds += time.perf_counter()-t
            offset += count
            loads += n
        directory = np.full(self.mesh.leaf_count,-1,dtype=np.int64)
        directory[self.leaf_ids] = np.arange(len(self.leaf_ids))
        directory.flags.writeable = values.flags.writeable = False
        return PreparedFields(self.mesh,values,self.leaf_ids,directory,tuple(source.fields[i] for i in fields),
            2,self.strategy,source.identity,{"total_seconds":time.perf_counter()-started,
            "read_seconds":read_seconds,"transfer_seconds":execute_seconds,"packing_seconds":packing_seconds,
            "selected_load_count":loads,"read_value_bytes":read_value_bytes,
            "requested_value_bytes":requested_value_bytes,
            "plan_bytes":self.nbytes,"controlled_upper_bytes":required})


def build_fill_plan(source, leaf_ids, *, capacity=128, budget_bytes=128*1024**2):
    """Build a bounded optional plan through an explicit provider geometry hook."""
    if source.plan_builder is None:
        raise ValueError("this source provider has no retained geometry planner")
    return source.plan_builder(source,leaf_ids,capacity,budget_bytes)


def _build(source, leaf_ids, capacity, budget_bytes, root_shape, coord_to_rank, forest):
    from simesh_rewrite import refined_halo as r
    from simesh_rewrite.target_boxes import fill_directed_halo_target_boxes
    start = time.perf_counter()
    ids = np.sort(indices(leaf_ids,source.mesh.leaf_count))
    if type(capacity) is not int or capacity < min(57,source.mesh.leaf_count):
        raise ValueError("plan capacity must fit the all-26 support closure")
    capacity = min(capacity,source.mesh.leaf_count)
    block = np.array(source.mesh.block_shape,dtype=np.int64)
    w,arrays = r._allocate_workspace(capacity,0,tuple(block),tuple(block+4))
    workspace_bytes = sum(a.nbytes for a in arrays)
    if workspace_bytes+ids.nbytes > budget_bytes:
        raise MemoryError("plan build workspace exceeds plan budget")
    lo,hi = np.full(3,2,dtype=np.int64),block+2
    modes,normals = np.zeros((0,6),dtype=np.uint8),np.full(3,-1,dtype=np.int64)
    fill_directed_halo_target_boxes(lo,hi,np.zeros(3,dtype=np.int64),block+4,
                                    r.CANONICAL_DIRECTIONS,w.target_lower,w.target_upper)
    f = forest
    chunks = []
    first = 0
    retained = 0
    while first < len(ids):
        count,n = r._prepare_refined_halo_chunk(w,ids[first:first+capacity],root_shape,coord_to_rank,
            f.root_node_ids,f.node_levels,f.node_coords,f.child_node_ids,f.node_leaf_ids,f.leaf_node_ids,
            lo,hi,modes,normals,validate_actions=True)
        actions = _actions(w,count,n,lo,hi)
        chunk = (_freeze(w.selected_leaf_ids[:n]),count,actions)
        retained += _bytes(chunk)
        if retained+workspace_bytes+ids.nbytes > budget_bytes:
            raise MemoryError("retained geometric actions exceed plan budget; keep unretained provider")
        chunks.append(chunk)
        first += count
    chunks = tuple(chunks)
    ids = frozen_array(ids,np.int64)
    return FillPlan(source.mesh,source.strategy,ids,chunks,capacity,_bytes((ids,chunks))+512,
                    time.perf_counter()-start)
