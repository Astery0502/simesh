import numpy as np
import pytest
from simesh.meshes.amr_mesh import AMRMesh
from simesh.geometry.amr.amr_forest import AMRForest

@pytest.fixture
def common_mesh():
    """Create a basic AMR mesh with the same forest structure as in test_amr_forest"""
    # Define basic mesh parameters
    xrange = (0., 1.)
    yrange = (0., 1.)
    zrange = (0., 1.)
    field_names = ['density', 'magnetic_field']
    nleafs = 16
    block_nx = np.array([8, 8, 8])
    domain_nx = np.array([16, 16, 16])
    nghostcells = 2

    # Create forest configuration
    forest = AMRForest(ng1=2, ng2=2, ng3=2, nleafs=16)
    forest_config = np.ones(16, dtype=bool)
    forest_config[1] = False  # Second element is non-leaf
    forest.read_forest(forest_config)
    forest.build_connectivity()

    # Create mesh
    mesh = AMRMesh(
        xrange, yrange, zrange,
        field_names, nleafs, block_nx, domain_nx,
        nghostcells=nghostcells,
        forest=forest
    )

    meshi = tuple(slice(mesh.ixMmin[i], mesh.ixMmax[i]+1) for i in range(3))
    z2n = tuple(slice(mesh.ixGmin[i], mesh.ixGmax[i]+1) if i !=2 
                else slice(mesh.ixMmax[i]+1, mesh.ixMmax[i]+3) for i in range(3))
    x2n = tuple(slice(mesh.ixMmin[i], mesh.ixMmax[i]+1) if i !=0 
                else slice(mesh.ixMmax[i]+1, mesh.ixMmax[i]+3) for i in range(3))
    y2n = tuple(slice(mesh.ixMmin[i], mesh.ixMmax[i]+1) if i !=1 
                else slice(mesh.ixMmax[i]+1, mesh.ixMmax[i]+3) for i in range(3))

    x0n = tuple(slice(mesh.ixMmin[i], mesh.ixMmax[i]+1) if i !=0 
                else slice(mesh.ixMmin[i]-1, mesh.ixMmin[i]-3) for i in range(3))
    y0n = tuple(slice(mesh.ixMmin[i], mesh.ixMmax[i]+1) if i !=1 
                else slice(mesh.ixMmin[i]-1, mesh.ixMmin[i]-3) for i in range(3))
    z0b = tuple(slice(mesh.ixMmin[i], mesh.ixMmax[i]+1) if i !=2 
                else slice(mesh.ixMmin[i]-1, mesh.ixMmin[i]-3) for i in range(3))

    # Fill data with ones
    mesh.data.fill(1.0)
    return mesh

def test_bc_phys(common_mesh):
    """Test physical boundary conditions"""
    igrid = 0  # Test with first grid
    idim = 0   # Test x-direction
    
    # Test minimum boundary (iside=0)
    ixBmin = common_mesh.ixGmin.copy()
    ixBmax = common_mesh.ixGmax.copy()
    
    # Apply boundary condition
    common_mesh.bc_phys(0, idim, igrid, ixBmin, ixBmax)
    
    # Check if ghost cells were filled correctly
    ghost_region = tuple(
        slice(0, common_mesh.nghostcells) if i == idim 
        else slice(None) 
        for i in range(3)
    )
    interior_region = tuple(
        slice(common_mesh.nghostcells, common_mesh.nghostcells + 1) if i == idim 
        else slice(None) 
        for i in range(3)
    )
    
    assert np.all(common_mesh.data[igrid][ghost_region] == 
                 common_mesh.data[igrid][interior_region])

    # Test maximum boundary (iside=1)
    common_mesh.bc_phys(1, idim, igrid, ixBmin, ixBmax)
    
    ghost_region = tuple(
        slice(-common_mesh.nghostcells, None) if i == idim 
        else slice(None) 
        for i in range(3)
    )
    interior_region = tuple(
        slice(-common_mesh.nghostcells-1, -common_mesh.nghostcells) if i == idim 
        else slice(None) 
        for i in range(3)
    )
    
    assert np.all(common_mesh.data[igrid][ghost_region] == 
                 common_mesh.data[igrid][interior_region])

def test_coarsen_grid(common_mesh):
    """Test grid coarsening operation"""
    igrid = 0
    
    # Define coarsening regions
    ixFimin = common_mesh.ixMmin
    ixFimax = common_mesh.ixMmax
    ixComin = common_mesh.ixMmin
    ixComax = (common_mesh.ixMmax // 2).astype(int)
    
    # Fill fine grid with known values
    common_mesh.data[igrid].fill(1.0)
    
    # Perform coarsening
    common_mesh.coarsen_grid(igrid, ixFimin, ixFimax, ixComin, ixComax)
    
    # Check results
    # Each coarse cell should be the average of 8 fine cells (2x2x2)
    # Since input is 1.0, output should be 1.0 (8/8 = 1)
    coarse_region = tuple(slice(ixComin[i], ixComax[i]+1) for i in range(3))
    assert np.allclose(common_mesh.datac[igrid][coarse_region], 1.0)

def test_fill_coarse_boundary():

    mesh = common_mesh()
    mesh.coarsen_grid(1, np.array([2,2,2]), np.array([5,5,5]), np.array([2,2,2]), np.array([3,3,3]))
    mesh.fill_coarse_boundary(1, 0, 1, 1)

    assert np.allclose(mesh.datac[1, 2:4, 2:4, 2:4, :], 1.0)
    assert np.allclose(mesh.datac[1, 2:4, 0:2, 2:4, :], 1.0)
    assert np.allclose(mesh.datac[1, 2:4, 2:4, 0:2, :], 1.0)
    assert np.allclose(mesh.datac[1, 2:4, 0:2, 0:2, :], 1.0)


def test_fill_boundary_before_gc():

    mesh = common_mesh()
    mesh.fill_boundary_before_gc(0)

    assert np.allclose(mesh.data[0][:,:,:,0][0:6,0:6,0:6], 1.0)

def test_getbc():
    mesh = common_mesh()
    mesh.getbc()

    assert np.all(mesh.data)
