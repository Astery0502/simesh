import os
import struct
import numpy as np
from simesh.amrvac.amrvac_dataset import AMRVACDataSet
from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh.utils.lib.amr.morton import fill_morton_mapping3D


def load_from_uniform(udata:np.ndarray, w_names:list[str], xmin:np.ndarray, xmax:np.ndarray, 
        block_nx:np.ndarray, **kwargs):
    """
    Load the data in the sfc sequence from the uniform grid
    """
    assert len(w_names) == udata.shape[-1]
    if len(udata.shape) == 3 or udata.shape[-2] == 1:
        ndim = 2

    block_nx = np.array(block_nx).astype(np.uint32)
    domain_nx = np.array(udata.shape[:3]).astype(np.uint32)
    assert np.all(domain_nx % block_nx == 0), "domain_nx must be divisible by block_nx"
    nblev1 = np.array(domain_nx // block_nx).astype(np.uint32)

    is_leaf = np.ones(nblev1[0]*nblev1[1]*nblev1[2], dtype=np.int32)
    forest = AMRForest(np.uint32(ndim), nblev1[0], nblev1[1], nblev1[2], is_leaf)
    mesh = AMRMesh(ndim, block_nx, domain_nx, xmin, xmax, 0, len(w_names), forest)
    ds = AMRVACDataSet(mesh)

    return ds

def uniform_to_vtk(udata: np.ndarray, w_names:list[str], filename: str, xmin:np.ndarray, xmax:np.ndarray = None):

    """
    Convert the uniform grid data to VTK format with binary data and ASCII header
    
    Parameters:
    -----------
    udata : np.ndarray
        Uniform grid data, of size (nw, nx, ny, nz)
    w_names : list[str]
        List of field names, of size (nw,)
    filename : str
        Output filename (should have .vtk extension)
    xmin : np.ndarray
        Minimum coordinates of the domain, shape (3,)
    xmax : np.ndarray, optional
        Maximum coordinates of the domain, shape (3,). If not provided, unit spacing is assumed.
    """

    if os.path.exists(filename):
        raise FileExistsError(f"File {filename} already exists")

    assert len(udata.shape) == 4, f"udata must be a 4D array, but got {len(udata.shape)}"
    assert udata.shape[0] == len(w_names), f"w_names must be of size {udata.shape[0]}, but got {len(w_names)}"
    assert len(xmin) == 3, f"xmin must have 3 elements, but got {len(xmin)}"

    nx, ny, nz = udata.shape[1:]
    nw = udata.shape[0]
    
    # Compute spacing
    if xmax is not None:
        assert len(xmax) == 3, f"xmax must have 3 elements, but got {len(xmax)}"
        dx = (xmax[0] - xmin[0]) / (nx - 1) if nx > 1 else 1.0
        dy = (xmax[1] - xmin[1]) / (ny - 1) if ny > 1 else 1.0
        dz = (xmax[2] - xmin[2]) / (nz - 1) if nz > 1 else 1.0
    else:
        # Default to unit spacing
        dx = dy = dz = 1.0
    
    # Ensure data is float64 and in C-contiguous order for binary writing
    udata = np.ascontiguousarray(udata.astype(np.float64))
    
    with open(filename, 'wb') as f:
        # Write ASCII header
        f.write(b'# vtk DataFile Version 2.0\n')
        f.write(b'Uniform grid data\n')
        f.write(b'BINARY\n')
        f.write(b'DATASET STRUCTURED_POINTS\n')
        
        # Write dimensions (ASCII)
        f.write(f'DIMENSIONS {nx} {ny} {nz}\n'.encode('ascii'))
        
        # Write origin (ASCII)
        f.write(f'ORIGIN {xmin[0]:.6e} {xmin[1]:.6e} {xmin[2]:.6e}\n'.encode('ascii'))
        
        # Write spacing (ASCII)
        f.write(f'SPACING {dx:.6e} {dy:.6e} {dz:.6e}\n'.encode('ascii'))
        
        # Write point data header (ASCII)
        f.write(f'POINT_DATA {nx * ny * nz}\n'.encode('ascii'))
        
        # Write each field as binary data
        for iw in range(nw):
            # Write field name and data type (ASCII)
            f.write(f'SCALARS {w_names[iw]} double 1\n'.encode('ascii'))
            f.write(b'LOOKUP_TABLE default\n')
            
            # Write binary data
            # VTK legacy format expects big-endian format for binary data
            # Transpose to swap x and z axes so that x varies fastest in the flattened C-order stream
            # VTK expects x-fastest (Fortran order for (nx, ny, nz)), but numpy flatten('C') is z-fastest.
            # Transposing (nx, ny, nz) -> (nz, ny, nx) and flattening 'C' makes nx (original x) vary fastest.
            field_data = np.transpose(udata[iw, :, :, :], (2, 1, 0)).flatten(order='C')
            field_data_big_endian = field_data.astype('>f8')  # Big-endian float64
            
            # Write binary data directly (no size prefix in standard VTK legacy format)
            f.write(field_data_big_endian.tobytes())
            f.write(b'\n')  # Add newline after binary data block

