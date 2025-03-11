import os
import pytest
import numpy as np
from simesh import amr_loader
from simesh.utils.configurations import TDm_slab
from simesh.meshes.amr_mesh import create_empty_amrmesh
from simesh.frontends.amrvac.io import write_dat_metadata, load_from_uarrays, write_dat_field
from simesh.frontends.amrvac.datio import get_header, get_forest, get_tree_info

@pytest.fixture
def common_tdmds():
    # prepare necessary parameters for initialization of a new AMRDataSet
    xmin = np.array([-4,-4,0])
    xmax = np.array([4,4,8])
    domain_nx = np.array([30,30,30])
    block_nx = np.array([10,10,10])

    # parameters for the TDm model
    r0 = 1.5
    a0 = 0.3
    q0 = 1.0
    L0 = 1.0
    d0 = 0.5
    naxis = 500
    ispositive = True

    # prepare the mesh data of domain size
    rho = np.ones(domain_nx) # uniform density just for display, no physical meaning
    v1 = np.zeros(domain_nx)
    v2 = np.zeros(domain_nx)
    v3 = np.zeros(domain_nx)
    bfield_tdm = TDm_slab(xmin, xmax, domain_nx, r0, a0, ispositive, naxis, q0, L0, d0) # magnetic field b1,b2,b3 from TDm model

    w_arrays = np.stack([rho, v1, v2, v3, bfield_tdm[0], bfield_tdm[1], bfield_tdm[2]], axis=-1)

    w_names = ['rho', 'v1', 'v2', 'v3', 'b1', 'b2', 'b3']

    # here you can specify a datfile name for further output like: file_path='example.dat'; and any other parameters by **kwargs to update the default header
    # you can view the default header from: simesh.header_template
    ds = load_from_uarrays(w_arrays, w_names, xmin, xmax, block_nx, file_path='data/tdm.dat') 

    return ds

def test_write_dat_metadata(common_tdmds):

    meta_file = 'data/meta.dat'
    if os.path.exists(meta_file):
        os.remove(meta_file)

    write_dat_metadata(meta_file, common_tdmds.mesh)
    with open(meta_file, 'rb') as f:    
        header_meta = get_header(f)
        forest_meta = get_forest(f)
        tree_meta = get_tree_info(f)

    for key in header_meta.keys():
        assert np.all(header_meta[key] == common_tdmds.header[key])

    assert np.all(forest_meta == common_tdmds.forest)
    for i in range(3):
        assert np.all(tree_meta[i] == common_tdmds.tree[i])

    os.remove(meta_file)

def test_write_dat_field(common_tdmds):

    """
    Here the mesh parameter into write_dat_metadata can be constructed by:

    from simesh.meshes.amr_mesh import create_empty_uniform_amrmesh
    mesh = create_empty_uniform_amrmesh(domain_nx, block_nx, xmin, xmax, w_names)

    domain_nx: [int, int, int]
    block_nx: [int, int, int]
    xmin: [float, float, float]
    xmax: [float, float, float]
    w_names: list of field names
    """

    # prepare the data to be written and the mesh for metadata
    udata = common_tdmds.mesh.udata
    data_file = 'data/single_field.dat'
    if os.path.exists(data_file):
        os.remove(data_file)
    write_dat_metadata(data_file, common_tdmds.mesh) ## **kwargs for additional update of header

    # open the file in binary mode and write the field data one by one
    with open(data_file, 'rb+') as fb:
        for fi in range(7):
            assert np.all(udata.shape[:-1] == common_tdmds.mesh.domain_nx)
            write_dat_field(fb, fi, udata[...,fi], isuniform=True) # here uniform data is written into the file

    # load the data file and check the written data
    ds = amr_loader(data_file)
    assert np.all(ds.mesh.udata == udata)
    os.remove(data_file)
