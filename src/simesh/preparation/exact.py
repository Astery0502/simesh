"""Exact-phase preparation with explicit geometry, I/O and workspace boundaries."""

from dataclasses import dataclass
import time
import numpy as np

from .._amr import halo
from .._validation import periodic_mask
from .._amr.target_boxes import fill_directed_halo_target_boxes
from .._kernels.preparation import apply_direct_actions

SCHEME = "rewrite-ratio2-minmod-exactphase-boundary-v1"


def parameters(mesh, field_count, support_capacity):
    block = np.asarray(mesh.block_shape, dtype=np.int64)
    if np.any(block < 4) or np.any(block % 2):
        raise ValueError("exact-phase preparation requires even 3D blocks of at least four cells")
    if type(support_capacity) is not int or support_capacity < 1:
        raise ValueError("support_capacity must be a positive integer")
    capacity = min(max(57, support_capacity), mesh.leaf_count)
    _, fixed = halo._allocate_workspace(0, 0, tuple(block), tuple(block+4))
    fixed_bytes = sum(a.nbytes for a in fixed)
    scratch = (capacity*(8*field_count*int(np.prod(block+4))+1854) +
               8*field_count*int(np.prod(block+1)) + fixed_bytes + 8*field_count +
               32*mesh.leaf_count + 4096)
    return block, capacity, scratch


@dataclass
class Workspace:
    """Private mutable transfer bindings and numerical scratch for one call."""

    mesh: object
    capacity: int
    w: object
    arrays: tuple
    lower: np.ndarray
    upper: np.ndarray
    modes: np.ndarray
    normals: np.ndarray
    periodic_mask: int

    @classmethod
    def allocate(cls, mesh, field_count, block, capacity):
        lower = np.full(3, 2, dtype=np.int64)
        upper = lower + block
        w, arrays = halo._allocate_workspace(capacity, field_count, tuple(block), tuple(block+4))
        zero = np.zeros(3, dtype=np.int64)
        fill_directed_halo_target_boxes(lower, upper, zero, block+4,
            halo.CANONICAL_DIRECTIONS, w.target_lower, w.target_upper)
        return cls(mesh, capacity, w, tuple(arrays), lower, upper,
                   np.zeros((field_count, 6), dtype=np.uint8), np.full(3, -1, dtype=np.int64),
                   periodic_mask=periodic_mask(mesh.periodic))

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (*self.arrays, self.lower, self.upper, self.modes, self.normals))


def plan_chunk(workspace, ordered_targets):
    """Bind source-leaf ordinals and target boxes using geometry alone."""
    mesh, w = workspace.mesh, workspace.w
    f = mesh.forest
    return halo._prepare_refined_halo_chunk(
        w, ordered_targets, mesh.root_shape, mesh.coord_to_rank,
        f.root_node_ids, f.node_levels, f.node_coords, f.child_node_ids,
        f.node_leaf_ids, f.leaf_node_ids, workspace.lower, workspace.upper,
        workspace.modes, workspace.normals, validate_actions=False,
        periodic_mask=workspace.periodic_mask)


def execute_chunk(workspace, count, selected):
    """Execute admitted exact-phase transfers without accessing a source."""
    w = workspace.w
    apply_direct_actions(w.payload, count, w.relation_kinds, w.physical_masks,
        w.source_counts, w.source_slots, w.phase_codes, halo.CANONICAL_DIRECTIONS,
        w.target_lower, w.target_upper, workspace.lower, workspace.upper)
    # Retain the accepted coarse-support/minmod and physical-widening arithmetic.
    halo._apply_chunk_actions_unchecked(w, count, selected, workspace.lower,
        workspace.upper, workspace.modes, workspace.normals, _direct_applied=True)


def read_targets(source, field_ids, ids, count, output, rows, payload, lower, upper):
    """Read primary interiors to their final slots, then bind bounded fill scratch.

    Scratch retains the support values needed by the unchanged exact-phase
    method. Primary values go straight to final storage, never a domain-sized
    field-major staging array. Extra support leaves are read only to scratch.
    """
    first = calls = 0
    while first < count:
        last = first+1
        while last < count and rows[last] == rows[last-1]+1:
            last += 1
        source.read_native_into(ids[first:last], field_ids,
            output[rows[first]:rows[last-1]+1], storage_halo=2)
        calls += 1
        first = last
    box = tuple(slice(int(a),int(b)) for a,b in zip(lower,upper))
    for slot, row in enumerate(rows):
        payload[(slot,slice(None),*box)] = np.moveaxis(output[(row,*box,slice(None))],-1,0)
    if count < len(ids):
        source._read_padded_into(ids[count:],field_ids,payload[count:len(ids)],lower)
        calls += 1
    return calls


def copy_halos(payload, output, rows):
    """Deliver only completed halos; interiors were read into output already."""
    for slot, row in enumerate(rows):
        values = np.moveaxis(payload[slot],0,-1)
        target = output[row]
        target[:2] = values[:2]
        target[-2:] = values[-2:]
        target[2:-2,:2] = values[2:-2,:2]
        target[2:-2,-2:] = values[2:-2,-2:]
        target[2:-2,2:-2,:2] = values[2:-2,2:-2,:2]
        target[2:-2,2:-2,-2:] = values[2:-2,2:-2,-2:]


def fill(source, selection, field_ids, output, workspace):
    """Complete private output; a caller may publish only after this returns."""
    source.validate()
    # Field IDs are validated by the caller; clip avoids NumPy's raise-mode
    # output buffer while gathering directly into the admitted workspace.
    np.take(source._boundary_modes, field_ids, axis=0, out=workspace.modes, mode="clip")
    ids = selection.leaf_ids
    cache_before = source.io_stats.get("value_cache_misses")
    order = np.argsort(ids, kind="stable")
    ordered = np.ascontiguousarray(ids[order])
    w = workspace.w
    first = chunks = loads = maximum = reader_calls = 0
    planning = reading = arithmetic = packing = 0.0
    start = time.perf_counter()
    while first < len(ordered):
        stamp = time.perf_counter()
        count, selected = plan_chunk(workspace, ordered[first:first+workspace.capacity])
        planning += time.perf_counter()-stamp
        if count < 1:
            raise RuntimeError("preparation support does not fit its admitted workspace")
        stamp = time.perf_counter()
        rows = order[first:first+count]
        reader_calls += read_targets(source,field_ids,w.selected_leaf_ids[:selected],count,output,rows,
                     w.payload,workspace.lower,workspace.upper)
        reading += time.perf_counter()-stamp
        stamp = time.perf_counter()
        execute_chunk(workspace, count, selected)
        arithmetic += time.perf_counter()-stamp
        stamp = time.perf_counter()
        copy_halos(w.payload,output,rows)
        packing += time.perf_counter()-stamp
        first += count
        chunks += 1
        loads += selected
        maximum = max(maximum, selected)
    source.validate()
    return {"primary_count": len(ids), "chunk_count": chunks,
            "reader_call_count": reader_calls, "selected_load_count": loads,
            "maximum_selected_slots": maximum,
            "managed_array_bytes": workspace.nbytes,
            "planning_seconds": planning, "reader_seconds": reading,
            "ghost_seconds": arithmetic, "packing_seconds": packing,
            "total_seconds": time.perf_counter()-start,
            "read_value_bytes": (loads if cache_before is None else
                source.io_stats["value_cache_misses"]-cache_before)*len(field_ids)*8*int(np.prod(source.mesh.block_shape)),
            "requested_value_bytes": loads*len(field_ids)*8*int(np.prod(source.mesh.block_shape))}
