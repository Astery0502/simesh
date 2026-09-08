"""Bounded E1 prototype: exact whole-leaf merges and direct mapped consumers.

This is evidence tooling, not a replacement native compute/cache identity.
Packing deliberately uses the retained native two-halo preparation, so storage
savings are never reported as eliminated preparation or I/O work.
"""
from dataclasses import dataclass
from itertools import product
import time
import numpy as np
from simesh.analysis.mesh import frozen_array, indices


@dataclass(frozen=True)
class BrickPlan:
    mesh: object
    merge_shape: tuple
    leaf_to_brick: np.ndarray
    leaf_offsets: np.ndarray
    shapes: np.ndarray
    starts: np.ndarray
    members: np.ndarray
    member_offsets: np.ndarray
    build_seconds: float

    @property
    def nbytes(self):
        return sum(v.nbytes for v in vars(self).values() if isinstance(v,np.ndarray))

    def expand(self, leaf_ids):
        ids = indices(leaf_ids,self.mesh.leaf_count)
        bricks = np.unique(self.leaf_to_brick[ids])
        return np.sort(np.concatenate([self.members[self.member_offsets[b]:self.member_offsets[b+1]] for b in bricks]))

    def storage_report(self):
        physical = self.mesh.leaf_count*int(np.prod(self.mesh.block_shape))
        padded = int(np.prod(self.shapes+4,axis=1).sum())
        counts = np.diff(self.member_offsets)
        return {"bricks":len(counts),"merged_leaves":int(counts[counts>1].sum()),
            "merged_leaf_fraction":float(counts[counts>1].sum()/self.mesh.leaf_count),
            "physical_cells":physical,"padded_cells":padded,"halo_cells":padded-physical,
            "padded_over_interior":padded/physical,"mapping_bytes":self.nbytes,
            "geometry_seconds":self.build_seconds}


def build_bricks(mesh, merge_shape):
    start = time.perf_counter()
    merge = np.asarray(merge_shape,dtype=np.int64)
    if tuple(merge) not in ((1,1,1),(2,1,1),(2,2,2)):
        raise ValueError("bounded E1 candidates are native, x-pairs and octets")
    block = np.asarray(mesh.block_shape)
    levels = np.rint(np.log2(mesh.spacing.max(axis=0)/mesh.spacing)).astype(np.int64)
    coords = np.rint((mesh.bounds[:,0]-mesh.lower)/(mesh.spacing*block)).astype(np.int64)
    buckets = {}
    for leaf in range(mesh.leaf_count):
        key = (*levels[leaf],*(coords[leaf]//merge))
        buckets.setdefault(key,[]).append(leaf)
    groups = []
    for leaves in buckets.values():
        origin = coords[leaves[0]]//merge*merge
        expected = {tuple(origin+offset) for offset in product(*(range(n) for n in merge))}
        if len(leaves)==int(np.prod(merge)) and {tuple(c) for c in coords[leaves]}==expected:
            groups.append((sorted(leaves),origin,merge*block))
        else:
            groups.extend(([leaf],coords[leaf],block) for leaf in leaves)
    groups.sort(key=lambda group:group[0][0])
    leaf_to_brick = np.empty(mesh.leaf_count,dtype=np.int64)
    offsets = np.empty((mesh.leaf_count,3),dtype=np.int64)
    members = []
    member_offsets = [0]
    shapes = []
    for brick,(leaves,origin,shape) in enumerate(groups):
        leaf_to_brick[leaves] = brick
        offsets[leaves] = (coords[leaves]-origin)*block
        shapes.append(shape)
        members.extend(leaves)
        member_offsets.append(len(members))
    shapes = np.asarray(shapes,dtype=np.int64)
    starts = np.r_[0,np.cumsum(np.prod(shapes+4,axis=1))]
    return BrickPlan(mesh,tuple(map(int,merge)),*(frozen_array(x,np.int64) for x in
        (leaf_to_brick,offsets,shapes,starts,members,member_offsets)),time.perf_counter()-start)


@dataclass(frozen=True)
class BrickFields:
    plan: BrickPlan
    values: np.ndarray
    starts: np.ndarray
    leaf_ids: np.ndarray

    @property
    def nbytes(self):
        return self.values.nbytes+self.starts.nbytes+self.leaf_ids.nbytes

    def window(self, leaf, halo=0):
        p = self.plan
        b = p.leaf_to_brick[leaf]
        if self.starts[b] < 0:
            raise ValueError("missing compute brick")
        shape = p.shapes[b]+4
        start = self.starts[b]
        data = self.values[start:start+int(np.prod(shape))].reshape(*shape,self.values.shape[-1])
        lo = p.leaf_offsets[leaf]+2-halo
        hi = p.leaf_offsets[leaf]+2+np.asarray(p.mesh.block_shape)+halo
        return data[tuple(slice(a,b) for a,b in zip(lo,hi))]


def pack_bricks(plan, prepared):
    if prepared.mesh is not plan.mesh or prepared.halo != 2:
        raise ValueError("prototype requires matching mesh and native width two")
    leaves = plan.expand(prepared.leaf_ids)
    if np.any(prepared.slot_of_leaf[leaves]<0):
        raise ValueError("preparation must include whole requested bricks")
    bricks = np.unique(plan.leaf_to_brick[leaves])
    starts = np.full(len(plan.shapes),-1,dtype=np.int64)
    sizes = np.prod(plan.shapes[bricks]+4,axis=1)
    starts[bricks] = np.r_[0,np.cumsum(sizes)[:-1]]
    values = np.empty((int(sizes.sum()),len(prepared.fields)))
    block = np.asarray(plan.mesh.block_shape)
    for leaf in leaves:
        b = plan.leaf_to_brick[leaf]
        offset = plan.leaf_offsets[leaf]
        shape = plan.shapes[b]
        lo = np.where(offset==0,0,2)
        hi = np.where(offset+block==shape,block+4,block+2)
        dst = tuple(slice(a+o,b+o) for a,b,o in zip(lo,hi,offset))
        src = tuple(slice(a,b) for a,b in zip(lo,hi))
        view = values[starts[b]:starts[b]+int(np.prod(shape+4))].reshape(*(shape+4),len(prepared.fields))
        view[dst] = prepared.values[(prepared.slot_of_leaf[leaf],*src,slice(None))]
    return BrickFields(plan,values,starts,leaves)


def sample_bricks(fields, points):
    """Vectorized direct access into compact variable-size bricks, no unpack."""
    p = fields.plan
    points = np.asarray(points,dtype=float)
    owners = p.mesh.locate(points)
    safe = np.maximum(owners,0)
    bricks = p.leaf_to_brick[safe]
    valid = (owners>=0)&(fields.starts[bricks]>=0)
    result = np.full((len(points),fields.values.shape[1]),np.nan)
    selected = np.flatnonzero(valid)
    leaf = owners[selected]
    b = bricks[selected]
    q = (points[selected]-p.mesh.bounds[leaf,0])/p.mesh.spacing[leaf]-.5
    base = np.floor(q).astype(np.int64)
    w = q-base
    base += p.leaf_offsets[leaf]+2
    shape = p.shapes[b]+4
    total = np.zeros((len(selected),fields.values.shape[1]))
    for dx,dy,dz in product((0,1),repeat=3):
        index = base+[dx,dy,dz]
        linear = fields.starts[b]+(index[:,0]*shape[:,1]+index[:,1])*shape[:,2]+index[:,2]
        weight = np.prod(np.where(np.array([dx,dy,dz]),w,1-w),axis=1)
        total += fields.values[linear]*weight[:,None]
    result[selected] = total
    valid &= np.isfinite(result).all(axis=1)
    result[~valid] = np.nan
    return result,owners,valid


def curl_bricks(fields, leaf_ids):
    """Complete requested native-cell curl using views of compact brick storage."""
    result = np.empty((len(leaf_ids),*fields.plan.mesh.block_shape,3))
    for row,leaf in enumerate(leaf_ids):
        x = fields.window(leaf,halo=1)
        dx,dy,dz = fields.plan.mesh.spacing[leaf]
        result[row,...,0] = (x[1:-1,2:,1:-1,2]-x[1:-1,:-2,1:-1,2])/(2*dy)-(x[1:-1,1:-1,2:,1]-x[1:-1,1:-1,:-2,1])/(2*dz)
        result[row,...,1] = (x[1:-1,1:-1,2:,0]-x[1:-1,1:-1,:-2,0])/(2*dz)-(x[2:,1:-1,1:-1,2]-x[:-2,1:-1,1:-1,2])/(2*dx)
        result[row,...,2] = (x[2:,1:-1,1:-1,1]-x[:-2,1:-1,1:-1,1])/(2*dx)-(x[1:-1,2:,1:-1,0]-x[1:-1,:-2,1:-1,0])/(2*dy)
    return result


def trace_samples(sampler, mesh, seeds, step, count):
    """Matched vectorized RK4 benchmark consumer, accepted-prefix summaries."""
    points = seeds.copy()
    alive = np.ones(len(seeds),dtype=bool)
    steps = np.zeros(len(seeds),dtype=np.int64)
    def tangent(x):
        b,_,good = sampler(x)
        norm = np.hypot(np.hypot(b[:,0],b[:,1]),b[:,2])
        good &= np.isfinite(norm)&(norm>0)
        with np.errstate(divide='ignore',invalid='ignore'):
            return b/norm[:,None],good
    for _ in range(count):
        k1,v1 = tangent(points)
        k2,v2 = tangent(points+step/2*k1)
        k3,v3 = tangent(points+step/2*k2)
        k4,v4 = tangent(points+step*k3)
        trial = points+step/6*(k1+2*k2+2*k3+k4)
        alive &= v1&v2&v3&v4&(mesh.locate(trial)>=0)
        points[alive] = trial[alive]
        steps += alive
    return points,steps
