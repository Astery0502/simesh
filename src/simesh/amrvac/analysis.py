"""Explicit canonical resident preparation adapter for experimental analysis.

The provider keeps canonical coordinate-phase transfer semantics. It does not
change Dataset defaults or depend on the checkout-local rewrite package.
"""

import time
import numpy as np

from simesh.analysis.fields import PreparedFields
from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh


class _OwnerArray(np.ndarray):
    """Carry the C allocation owner through NumPy views and stripped subclasses."""
    def __array_finalize__(self, parent):
        self._allocation_owner = getattr(parent,"_allocation_owner",None)


def prepare_resident(mesh_index, root_shape, forest_flags, interiors, definitions,
                     *, budget_bytes=2*1024**3):
    """Prepare all leaves with canonical bulk fills; retain native padded backing.

    Explicit provider geometry/leaf order must match interiors and flags. This
    adapter consumes the same validated geometry used to construct MeshIndex.
    Source interiors may be released on return. No padding/repacking copy is
    introduced between the prepared C buffer and analysis consumers.
    """
    mesh = mesh_index
    definitions = tuple(definitions)
    block = np.asarray(mesh.block_shape,dtype=np.uint32)
    root = np.asarray(root_shape)
    if (root.shape != (3,) or root.dtype.kind not in "iu" or np.any(root < 1) or
            np.any(root > np.iinfo(np.uint32).max//block)):
        raise ValueError("invalid canonical root shape")
    root = root.astype(np.uint32)
    if tuple(root) != mesh.roots.shape:
        raise ValueError("root shape disagrees with mesh index")
    if (not isinstance(interiors,np.ndarray) or interiors.dtype != np.float64 or
            not interiors.flags.c_contiguous or
            interiors.shape != (mesh.leaf_count,len(definitions),*mesh.block_shape)):
        raise ValueError("interiors must match provider leaf/field/block layout")
    flags = np.asarray(forest_flags)
    if flags.ndim != 1 or not np.isin(flags,[0,1]).all() or np.count_nonzero(flags) != mesh.leaf_count:
        raise ValueError("forest flags disagree with leaf coverage")
    # Validate traversal shape before calling the canonical constructor, which
    # assumes a well-formed flag stream. Flat provider hierarchy supplies IDs.
    if len(flags) != len(mesh.node_leaves) or not np.array_equal(flags!=0,mesh.node_leaves>=0):
        raise ValueError("forest flag order disagrees with the validated mesh index")
    padded_bytes = mesh.leaf_count*len(definitions)*8*int(np.prod(block.astype(int)+4))
    coarse_bound = mesh.leaf_count*len(definitions)*8*int(np.prod(block.astype(int)//2+4))
    geometry_bound = mesh.leaf_count*1400+len(flags)*256+len(flags)*4+4096
    retained_bound = padded_bytes+coarse_bound+geometry_bound
    total = retained_bound+interiors.nbytes+mesh.nbytes+mesh.leaf_count*16
    if total > budget_bytes:
        raise MemoryError(f"canonical preparation needs {total} controlled bytes, budget {budget_bytes}")
    start = time.perf_counter()
    forest = AMRForest(3,*map(np.uint32,root),flags.astype(np.int32))
    owner = AMRMesh(3,block,root*block,mesh.lower.copy(),mesh.upper.copy(),2,len(definitions),forest)
    owner.load_interior_data(interiors)
    loaded = time.perf_counter()
    owner.apply_ghost_cells()
    finished = time.perf_counter()
    backing = owner.padded_view().view(_OwnerArray)
    backing._allocation_owner = owner
    # Expose a plain ndarray whose base retains the owner-bearing subclass.
    values = backing.view(np.ndarray)
    values.flags.writeable = False
    ids = np.arange(mesh.leaf_count,dtype=np.int64)
    ids.flags.writeable = False
    stats = {"provider":"canonical","construct_and_copy_seconds":loaded-start,
             "ghost_seconds":finished-loaded,"total_seconds":finished-start,
             "controlled_upper_bytes":total,"coarse_upper_bytes":coarse_bound,
             "provider_geometry_upper_bytes":geometry_bound}
    return PreparedFields(mesh,values,ids,ids,definitions,2,
                          "canonical-coordinatephase-cont-v1",object(),stats,
                          owner=owner,owner_extra_bytes=coarse_bound+geometry_bound)
