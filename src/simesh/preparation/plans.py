"""Optional reusable geometry; fresh values and limiter state on every execution."""
from dataclasses import dataclass
import sys
import time
import numpy as np
from .._validation import frozen_array, admit
from ..mesh import resolve_selection
from ..fields import publish
from . import exact

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
    from simesh._amr import halo as r
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
    from simesh._amr import halo as r
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



@dataclass(frozen=True,eq=False)
class FillPlan:
    mesh: object
    selection: object
    order: np.ndarray
    chunks: tuple
    capacity: int
    nbytes: int
    build_seconds: float
    scheme: str = exact.SCHEME

    def prepare(self,source,fields=None,*,memory_limit=None):
        if source.mesh is not self.mesh:
            raise ValueError("plan belongs to a different immutable Mesh")
        source.validate()
        fields=source.field_ids(fields)
        k=len(fields)
        block=np.array(self.mesh.block_shape,dtype=np.int64)
        padded=block+4
        shape=(len(self.selection.leaf_ids),*padded,k)
        scratch=8*k*(self.capacity*int(np.prod(padded))+int(np.prod(block+1)))
        required=(self.nbytes+self.mesh.nbytes+source.nbytes+source.read_footprint(self.capacity,k)+
                  8*int(np.prod(shape))+8*(self.mesh.leaf_count+len(self.selection.leaf_ids))+scratch)
        admit(required,memory_limit,"planned preparation")
        values=np.empty(shape)
        payload=np.empty((self.capacity,k,*padded))
        coarse=np.empty((1,k,*(block+1)))
        lo,hi=np.full(3,2,dtype=np.int64),block+2
        modes,normals=np.zeros((k,6),dtype=np.uint8),np.full(3,-1,dtype=np.int64)
        offset=loads=0
        reading=arithmetic=packing=0.
        start=time.perf_counter()
        cache_before=source.io_stats.get("value_cache_misses")
        for ids,count,actions in self.chunks:
            n=len(ids)
            stamp=time.perf_counter()
            rows=self.order[offset:offset+count]
            exact.read_targets(source,fields,ids,count,values,rows,payload,lo,hi)
            reading+=time.perf_counter()-stamp
            stamp=time.perf_counter()
            _execute(actions,payload[:n],coarse,lo,hi,modes,normals)
            arithmetic+=time.perf_counter()-stamp
            stamp=time.perf_counter()
            exact.copy_halos(payload,values,rows)
            packing+=time.perf_counter()-stamp
            offset+=count
            loads+=n
        source.validate()
        stats={"total_seconds":time.perf_counter()-start,"read_seconds":reading,
               "transfer_seconds":arithmetic,"packing_seconds":packing,"selected_load_count":loads,
               "read_value_bytes":(loads if cache_before is None else source.io_stats['value_cache_misses']-cache_before)*k*8*int(np.prod(block)),
               "requested_value_bytes":loads*k*8*int(np.prod(block)),"plan_bytes":self.nbytes,
               "controlled_upper_bytes":required}
        return publish(self.mesh,values,self.selection,tuple(source.fields[i] for i in fields),
                       2,2,self.scheme,source.identity,stats,
                       value_identity=source.value_identity(fields,self.scheme))


def plan_preparation(mesh,*,region=None,leaf_ids=None,scheme,support_capacity=128,memory_limit=None):
    """Retain field-independent exact-phase geometry for a fixed target selection."""
    if scheme!='exact-phase':
        raise ValueError("retained plans currently implement exact-phase only")
    selected=resolve_selection(mesh,region,leaf_ids)
    block,capacity,scratch=exact.parameters(mesh,0,support_capacity)
    order=np.argsort(selected.leaf_ids,kind='stable')
    ids=np.ascontiguousarray(selected.leaf_ids[order])
    admit(scratch+ids.nbytes+mesh.nbytes,memory_limit,"plan workspace")
    workspace=exact.Workspace.allocate(mesh,0,block,capacity)
    chunks=[]
    first=retained=0
    start=time.perf_counter()
    while first<len(ids):
        count,n=exact.plan_chunk(workspace,ids[first:first+capacity])
        if count<1:
            raise MemoryError("plan support does not fit its workspace")
        actions=_actions(workspace.w,count,n,workspace.lower,workspace.upper)
        chunk=(_freeze(workspace.w.selected_leaf_ids[:n]),count,actions)
        retained+=_bytes(chunk)
        admit(retained+workspace.nbytes+ids.nbytes+mesh.nbytes,memory_limit,"retained plan")
        chunks.append(chunk)
        first+=count
    chunks=tuple(chunks)
    order=frozen_array(order,np.int64)
    size=_bytes((selected.leaf_ids,order,chunks))+512
    return FillPlan(mesh,selected,order,chunks,capacity,size,time.perf_counter()-start)
