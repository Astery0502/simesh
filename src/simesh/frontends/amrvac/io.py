import os
import numpy as np
from .datio import get_header, get_forest, get_tree_info, get_tree_size, write_header, write_forest_tree, write_single_block_field_data
from simesh.meshes.amr_mesh import AMRMesh, amrmesh_from_uniform
from simesh.geometry.amr.amr_forest import AMRForest
from simesh.dataset.data_set import AMRDataSet

header_template = {
    'datfile_version': 5,
    'offset_tree': 0,
    'offset_blocks': 0,
    'nw': 7,
    'ndir': 3,
    'ndim': 3,
    'levmax': 1,
    'nleafs': 1,
    'nparents': 0,
    'it': 0,
    'time': 0.0,
    'xmin': np.array([-1., -1., -1.]),
    'xmax': np.array([1., 1., 1.]),
    'domain_nx': np.array([20, 20, 20]),
    'block_nx': np.array([10, 10, 10]),
    'periodic': np.array([False, False, False]),
    'geometry': 'Cartesian_3D',
    'staggered': False,
    'w_names': ['rho', 'm1', 'm2', 'm3', 'b1', 'b2', 'b3'],
    'physics_type': 'mhd',
    'n_par': 1,
    'params': np.array([1.66666667]),
    'param_names': ['gamma'],
    'snapshotnext': 1,
    'slicenext': 0,
    'collapsenext': 0
}

def amr_loader(file_path: str, nghostcells: int = 2, load_ghost: bool = False):

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
        ds = AMRDataSet(mesh, header, forest, tree, file_path)
        ds.load_data(load_ghost=load_ghost)

        return ds

def load_from_uarrays(nw_arrays, w_names, xmin, xmax, block_nx, file_path:str='datfile', **kwargs):

    mesh = amrmesh_from_uniform(nw_arrays, w_names, xmin, xmax, block_nx)

    header = header_template.copy()
    header['nw'] = len(w_names)
    header['w_names'] = list(w_names)
    header['nleafs'] = int(mesh.nleafs)
    header['xmin'] = np.array(xmin)
    header['xmax'] = np.array(xmax)
    header['domain_nx'] = np.array(mesh.domain_nx)
    header['block_nx'] = np.array(block_nx).astype(int)
    header['levmax'] = mesh.forest.max_level

    tree_size, offset_size = get_tree_size(header)
    header['offset_tree'] = tree_size
    header['offset_blocks'] = offset_size

    forest = mesh.forest.write_forest()
    tree = mesh.write_tree()
    tree[2] += offset_size

    ds = AMRDataSet(mesh, header, forest, tuple(tree), file_path)
    ds.update_header(**kwargs)

    return ds

def write_dat_metadata(file_path: str, amr_mesh: AMRMesh, **kwargs):
    """
    Write the metadata of an AMR mesh to a dat file.
    """
    # Check if file already exists
    if os.path.exists(file_path):
        raise FileExistsError(f"File {file_path} already exists")

    header = header_template.copy()
    header['nw'] = len(amr_mesh.field_names)
    header['w_names'] = list(amr_mesh.field_names)
    header['nleafs'] = int(amr_mesh.nleafs)
    header['xmin'] = np.array([amr_mesh.xrange[0], amr_mesh.yrange[0], amr_mesh.zrange[0]])
    header['xmax'] = np.array([amr_mesh.xrange[1], amr_mesh.yrange[1], amr_mesh.zrange[1]])
    header['domain_nx'] = np.array(amr_mesh.domain_nx).astype(int)
    header['block_nx'] = np.array(amr_mesh.block_nx).astype(int)
    header['levmax'] = amr_mesh.forest.max_level

    tree_size, offset_size = get_tree_size(header)
    header['offset_tree'] = tree_size
    header['offset_blocks'] = offset_size

    for key, value in kwargs.items():
        if key in header:
            header[key] = value
        else:
            raise ValueError(f"Key '{key}' not found in header")

    forest = amr_mesh.forest.write_forest()
    tree = amr_mesh.write_tree()
    tree[2] += offset_size

    with open(file_path, 'wb') as fb:
        write_header(fb, header)
        write_forest_tree(fb, header, forest, tree)

    return header, forest, tree

def write_dat_field(fb, field_idx: int, data: np.ndarray, isuniform: bool = False):
    """
    Write a single field to a dat file, already write_dat_metadata must be called first.
    """
    header = get_header(fb)
    tree = get_tree_info(fb)

    data = np.asarray(data)

    assert field_idx < header['nw']
    assert np.all(data.shape == header['domain_nx']), f"data shape {data.shape} must match domain_nx {header['domain_nx']}"

    if isuniform:
        assert header['levmax'] == 1

    for ileaf, ioffset in enumerate(tree[2]):
        if isuniform:
            idx = tree[1][ileaf]-1
            x0, y0, z0 = idx * header['block_nx']
            x1, y1, z1 = (idx+1) * header['block_nx']
            datai = data[x0:x1, y0:y1, z0:z1]
        else:
            datai = data[ileaf]

        write_single_block_field_data(fb, ioffset, header['block_nx'], field_idx, 3, datai)
