from simesh import amr_loader
import numpy as np

def load_slab_physbound(filename, load_ghost=True):
    """
    Load a slab of data from an AMR dataset.

    Args:
        filename (str): The path to the AMR dataset file.
        load_ghost (bool): Whether to load ghost cells.

    Returns:
        np.ndarray: A slab of data from the AMR dataset.
    """
    ds = amr_loader(filename, load_ghost=load_ghost)

    data_slab = np.zeros((*(ds.mesh.domain_nx+2*ds.mesh.nghostcells),7))
    for i in range(ds.mesh.nleafs):
        igidx = ds.mesh.forest.sfc_iglevel1[i]
        xyz0 = igidx * ds.mesh.block_nx
        xyz1 = (igidx+1) * ds.mesh.block_nx

        ixMmin = ds.mesh.ixMmin.copy()
        ixMmax = ds.mesh.ixMmax.copy()

        for idx in range(3):
            if xyz0[idx] == 0:
                ixMmin[idx] -= ds.mesh.nghostcells
                continue
            xyz0[idx] += ds.mesh.nghostcells
        
        for idx in range(3):
            if xyz1[idx] == ds.mesh.domain_nx[idx]:
                ixMmax[idx] += ds.mesh.nghostcells
                xyz1[idx] += ds.mesh.nghostcells
            xyz1[idx] += ds.mesh.nghostcells

        data_slab[xyz0[0]:xyz1[0], xyz0[1]:xyz1[1], xyz0[2]:xyz1[2]] = \
            ds.mesh.data[i][ixMmin[0]:ixMmax[0]+1, ixMmin[1]:ixMmax[1]+1, ixMmin[2]:ixMmax[2]+1]
        
        return data_slab