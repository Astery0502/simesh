import os
import struct
import numpy as np
from simesh.amrvac.amrvac_dataset import AMRVACDataSet
from simesh.amrvac.datio import extract_uniform_data, header_template, update_header, write_datfile_from_sfc
from simesh.amrvac.layouts import udata_to_datau
from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh.utils.lib.amr.morton import fill_morton_mapping3D


def load_from_uniform(udata:np.ndarray, w_names:list[str], xmin:np.ndarray, xmax:np.ndarray, 
        block_nx:np.ndarray, **kwargs):
    """
    Load data from a user-facing uniform grid with shape (nx, ny, nz, nw).
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


def _build_uniform_level1_tree(domain_nx: np.ndarray, block_nx: np.ndarray):
    """
    Build level-1 forest/tree metadata for a uniform grid.
    """
    nblev1 = np.array(domain_nx // block_nx, dtype=np.uint32)
    nleafs = int(np.prod(nblev1))
    is_leaf = np.ones(nleafs, dtype=np.int32)

    morton2ig = np.zeros((nleafs, 3), dtype=np.uint32)
    fill_morton_mapping3D(np.zeros(tuple(nblev1), dtype=np.uint32), morton2ig, *nblev1)

    block_lvls = np.ones(nleafs, dtype=np.int32)
    block_ixs = morton2ig.astype(np.int32) + 1
    block_offsets = np.zeros(nleafs, dtype=np.int64)

    return is_leaf, (block_lvls, block_ixs, block_offsets)


def _sfc_data_from_uniform(udata: np.ndarray, block_nx: np.ndarray, xmin: np.ndarray, xmax: np.ndarray):
    """
    Convert uniform cell-centered data with shape (nx, ny, nz, nw) into
    canonical SFC/Morton-ordered block data with shape (nleafs, nw, bx, by, bz).
    """
    udata = np.asarray(udata, dtype=np.double)
    block_nx = np.asarray(block_nx, dtype=np.uint32)
    xmin = np.asarray(xmin, dtype=np.double)
    xmax = np.asarray(xmax, dtype=np.double)

    if udata.ndim != 4:
        raise ValueError(f"udata must have shape (nx, ny, nz, nw), got {udata.shape}")
    if len(block_nx) != 3:
        raise ValueError(f"block_nx must have length 3, got {block_nx}")
    if len(xmin) != 3 or len(xmax) != 3:
        raise ValueError("xmin and xmax must have length 3")

    domain_nx = np.array(udata.shape[:3], dtype=np.uint32)
    nfields = int(udata.shape[3])

    if np.any(domain_nx % block_nx != 0):
        raise ValueError(f"domain_nx must be divisible by block_nx, got {domain_nx} and {block_nx}")

    nblev1 = np.array(domain_nx // block_nx, dtype=np.uint32)
    nleafs = int(np.prod(nblev1))
    is_leaf = np.ones(nleafs, dtype=np.int32)

    forest = AMRForest(np.uint32(3), nblev1[0], nblev1[1], nblev1[2], is_leaf)
    mesh = AMRMesh(np.uint32(3), block_nx, domain_nx, xmin, xmax, np.uint32(0), np.uint32(nfields), forest)

    uniform_data = np.ascontiguousarray(udata_to_datau(udata), dtype=np.double)
    sfc_data = np.zeros((nleafs, nfields, block_nx[0], block_nx[1], block_nx[2]), dtype=np.double)
    mesh.uniform_to_sfc(uniform_data, sfc_data)

    return sfc_data


def write_datfile_from_uniform(file_path: str, udata: np.ndarray, w_names: list[str],
                               xmin: np.ndarray, xmax: np.ndarray, block_nx: np.ndarray,
                               overwrite: bool = False, **header_updates) -> dict:
    """
    Write an AMRVAC .dat file from uniform data by first arranging it into
    canonical Morton/SFC-ordered blocks.
    """
    udata = np.asarray(udata, dtype=np.double)
    if udata.ndim != 4:
        raise ValueError(f"udata must have shape (nx, ny, nz, nw), got {udata.shape}")
    if len(w_names) != udata.shape[3]:
        raise ValueError(f"w_names length must match udata.shape[-1], {len(w_names)} != {udata.shape[3]}")

    domain_nx = np.array(udata.shape[:3], dtype=np.int32)
    block_nx = np.asarray(block_nx, dtype=np.int32)
    xmin = np.asarray(xmin, dtype=np.double)
    xmax = np.asarray(xmax, dtype=np.double)

    sfc_data = _sfc_data_from_uniform(udata, block_nx, xmin, xmax)
    is_leaf, tree_info = _build_uniform_level1_tree(domain_nx, block_nx)

    header = header_template.copy()
    header["nw"] = len(w_names)
    header["w_names"] = list(w_names)
    header["levmax"] = 1
    header["nleafs"] = int(sfc_data.shape[0])
    header["nparents"] = 0
    header["xmin"] = xmin
    header["xmax"] = xmax
    header["domain_nx"] = domain_nx
    header["block_nx"] = block_nx

    if header_updates:
        header = update_header(header, **header_updates)

    return write_datfile_from_sfc(file_path, sfc_data, header, is_leaf, tree_info, overwrite=overwrite)


def load_uniform_data(file_path: str, field_indices: list[int] | None = None,
                      return_geometry: bool = True):
    """
    Load level-1 AMRVAC data directly as a uniform grid.

    Returns uniform data with shape (nx, ny, nz, n_selected_fields). When
    `return_geometry` is True, a compact geometry/metadata dictionary is
    returned alongside the data.
    """
    udata, header = extract_uniform_data(file_path, field_indices=field_indices)

    if not return_geometry:
        return udata

    geometry_info = {
        "xmin": np.asarray(header["xmin"]).copy(),
        "xmax": np.asarray(header["xmax"]).copy(),
        "domain_nx": np.asarray(header["domain_nx"]).copy(),
        "block_nx": np.asarray(header["block_nx"]).copy(),
        "w_names": list(header["w_names"]),
        "geometry": header["geometry"],
        "periodic": np.asarray(header["periodic"]).copy(),
        "ndim": int(header["ndim"]),
        "ndir": int(header["ndir"]),
        "time": float(header["time"]),
        "it": int(header["it"]),
    }
    return udata, geometry_info


def datfile_to_vtk(file_path: str, filename: str, field_indices: list[int] | None = None):
    """
    Convert a level-1 AMRVAC .dat file directly to VTK.

    This is a convenience wrapper that loads user-facing uniform data with
    shape (nx, ny, nz, nw), converts it to the compute-oriented layout
    expected by ``uniform_to_vtk()``, and writes the VTK file using the
    metadata stored in the datfile.
    """
    udata, geometry = load_uniform_data(file_path, field_indices=field_indices, return_geometry=True)
    datau = udata_to_datau(udata)
    uniform_to_vtk(
        datau,
        geometry["w_names"],
        filename,
        geometry["xmin"],
        geometry["xmax"],
    )


def uniform_to_vtk(udata: np.ndarray, w_names:list[str], filename: str, xmin:np.ndarray, xmax:np.ndarray = None):

    """
    Convert the uniform grid data to VTK format with binary data and ASCII header
    
    Parameters:
    -----------
    udata : np.ndarray
        Compute-oriented uniform grid data, of size (nw, nx, ny, nz)
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
