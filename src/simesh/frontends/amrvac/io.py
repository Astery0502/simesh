import numpy as np
from .datio import get_header, get_forest, get_tree_info, get_single_block_data
from simesh.meshes.amr_mesh import AMRMesh
from simesh.geometry.amr.amr_forest import AMRForest
from simesh.dataset.data_set import AMRDataSet

def amr_loader(file_path: str, nghostcells: int = 2, load_ghost: bool = True):

    with open(file_path, 'rb') as fb:
        header = get_header(fb)
        forest = get_forest(fb)
        tree = get_tree_info(fb)

        xmin = header['xmin']
        xmax = header['xmax']
        xrange = (xmin[0], xmax[0])
        yrange = (xmin[1], xmax[1])
        zrange = (xmin[2], xmax[2])

        block_nx = header['block_nx']
        domain_nx = header['domain_nx']
        nblev1 = domain_nx // block_nx

        field_names = header['w_names']
        nleafs = header['nleafs']

        assert isinstance(nblev1, np.ndarray), "nblev1 must be a numpy array"
        assert len(nblev1) == 3 , f"nblev1 should be a 3-element array, {nblev1}"

        forest_amr = AMRForest(nblev1[0], nblev1[1], nblev1[2], nleafs)
        forest_amr.read_forest(forest)
        forest_amr.build_connectivity()

        mesh = AMRMesh(xrange, yrange, zrange, field_names, block_nx, domain_nx, forest_amr, nghostcells)
        ds = AMRDataSet(mesh, file_path, header, forest, tree, field_names)
        ds.load_data(load_ghost=load_ghost)

        return ds
