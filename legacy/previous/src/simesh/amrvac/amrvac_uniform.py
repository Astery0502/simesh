import os
import numpy as np
from simesh.amrvac.amrvac_dataset import AMRVACDataSet
from simesh.amrvac.datio import extract_uniform_data, header_template, update_header
from simesh.amrvac.layouts import udata_to_datau
from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh
from simesh.utils.lib.amr.morton import fill_morton_mapping3D


def load_from_uniform(udata:np.ndarray, w_names:list[str], xmin:np.ndarray, xmax:np.ndarray, 
        block_nx:np.ndarray, **kwargs):
    """
    Load data from a user-facing uniform grid with shape (nx, ny, nz, nw).
    """
    udata = _validate_uniform_data_shape(udata, w_names)
    sfc_data, ndim, domain_nx, block_nx, is_leaf, tree_info, forest, mesh = _sfc_data_from_uniform(
        udata,
        block_nx,
        xmin,
        xmax,
    )
    header = _uniform_header(
        w_names,
        ndim,
        domain_nx,
        block_nx,
        np.asarray(xmin, dtype=np.double),
        np.asarray(xmax, dtype=np.double),
        int(sfc_data.shape[0]),
        kwargs,
    )
    return _dataset_from_uniform_components(sfc_data, header, is_leaf, tree_info, forest, mesh)


def _validate_uniform_data_shape(udata: np.ndarray, w_names: list[str]) -> np.ndarray:
    udata = np.asarray(udata)
    if udata.ndim != 4:
        raise ValueError(f"udata must have shape (nx, ny, nz, nw), got {udata.shape}")
    if len(w_names) != udata.shape[-1]:
        raise ValueError(f"w_names length must match udata.shape[-1], {len(w_names)} != {udata.shape[-1]}")
    return udata


def _uniform_header(
    w_names: list[str],
    ndim: int,
    domain_nx: np.ndarray,
    block_nx: np.ndarray,
    xmin: np.ndarray,
    xmax: np.ndarray,
    nleafs: int,
    header_updates: dict | None = None,
) -> dict:
    header = header_template.copy()
    header["ndim"] = ndim
    header["nw"] = len(w_names)
    header["w_names"] = list(w_names)
    header["levmax"] = 1
    header["nleafs"] = int(nleafs)
    header["nparents"] = 0
    header["xmin"] = np.asarray(xmin, dtype=np.double)[:ndim]
    header["xmax"] = np.asarray(xmax, dtype=np.double)[:ndim]
    header["domain_nx"] = np.asarray(domain_nx, dtype=np.int32)
    header["block_nx"] = np.asarray(block_nx, dtype=np.int32)
    if ndim == 2:
        header["periodic"] = np.asarray(header["periodic"][:2], dtype=bool)
        header["geometry"] = "Cartesian_2D"
    if header_updates:
        header = update_header(header, **header_updates)
    return header


def _dataset_from_uniform_components(
    sfc_data: np.ndarray,
    header: dict,
    is_leaf: np.ndarray,
    tree_info: tuple,
    forest: AMRForest,
    mesh: AMRMesh,
) -> AMRVACDataSet:
    ds = AMRVACDataSet.__new__(AMRVACDataSet)
    ds.sfile = None
    ds.ghost_width = 0
    ds.ng = 0
    ds.metadata = header.copy()
    ds.is_leaf = is_leaf.copy().astype(np.int32)
    ds.tree_info = tree_info
    ds.ndim = np.uint32(header["ndim"])
    ds.ndir = np.uint32(header["ndir"])
    ds.nw = np.uint32(header["nw"])
    ds.wnames = list(header["w_names"])
    ds.nleafs = np.uint32(header["nleafs"])
    ds.nparents = np.uint32(header["nparents"])
    ds.levmax = np.uint32(header["levmax"])
    ds.block_nx = header["block_nx"].astype(np.uint32)
    ds.domain_nx = header["domain_nx"].astype(np.uint32)
    ds.physical_domain = np.array((header["xmin"], header["xmax"]))
    ds.periodic = header["periodic"]
    ds.geometry = header["geometry"]
    ds.forest = forest
    ds.mesh = mesh
    ds.data = sfc_data
    ds._init_derived_fields()
    ds._set_field_columns(ds._original_field_columns(range(int(ds.nw))))
    return ds


def _build_uniform_level1_tree(domain_nx: np.ndarray, block_nx: np.ndarray, ndim: int | None = None):
    """
    Build level-1 forest/tree metadata for a uniform grid.
    """
    if ndim is None:
        ndim = len(domain_nx)
    nblev1 = np.array(domain_nx // block_nx, dtype=np.uint32)
    nleafs = int(np.prod(nblev1))
    is_leaf = np.ones(nleafs, dtype=np.int32)

    morton2ig = np.zeros((nleafs, 3), dtype=np.uint32)
    nblev1_3 = np.array([nblev1[0], nblev1[1], 1 if ndim == 2 else nblev1[2]], dtype=np.uint32)
    fill_morton_mapping3D(np.zeros(tuple(nblev1_3), dtype=np.uint32), morton2ig, *nblev1_3)

    block_lvls = np.ones(nleafs, dtype=np.int32)
    block_ixs = morton2ig[:, :ndim].astype(np.int32) + 1
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
    if len(block_nx) not in (2, 3):
        raise ValueError(f"block_nx must have length 2 or 3, got {block_nx}")
    ndim = 2 if len(block_nx) == 2 else 3
    if ndim == 2:
        if udata.shape[2] != 1:
            raise ValueError("2D uniform data must use singleton z, udata.shape[2] == 1")
        if len(xmin) < 2 or len(xmax) < 2:
            raise ValueError("2D xmin and xmax must have at least two entries")
        block_nx_model = np.array(block_nx[:2], dtype=np.uint32)
        domain_nx_model = np.array(udata.shape[:2], dtype=np.uint32)
    else:
        if len(block_nx) != 3:
            raise ValueError(f"3D block_nx must have length 3, got {block_nx}")
        if len(xmin) != 3 or len(xmax) != 3:
            raise ValueError("3D xmin and xmax must have length 3")
        block_nx_model = block_nx
        domain_nx_model = np.array(udata.shape[:3], dtype=np.uint32)

    nfields = int(udata.shape[3])

    if np.any(domain_nx_model % block_nx_model != 0):
        raise ValueError(f"domain_nx must be divisible by block_nx, got {domain_nx_model} and {block_nx_model}")

    is_leaf, tree_info = _build_uniform_level1_tree(domain_nx_model, block_nx_model, ndim=ndim)
    nblev1 = np.array(domain_nx_model // block_nx_model, dtype=np.uint32)
    nblev1_3 = np.array([nblev1[0], nblev1[1], 1 if ndim == 2 else nblev1[2]], dtype=np.uint32)
    nleafs = int(is_leaf.shape[0])

    forest = AMRForest(np.uint32(ndim), nblev1_3[0], nblev1_3[1], nblev1_3[2], is_leaf)
    mesh = AMRMesh(
        np.uint32(ndim),
        block_nx_model,
        domain_nx_model,
        np.asarray(xmin[:ndim], dtype=np.double),
        np.asarray(xmax[:ndim], dtype=np.double),
        np.uint32(0),
        np.uint32(nfields),
        forest,
    )

    uniform_data = np.ascontiguousarray(udata_to_datau(udata), dtype=np.double)
    bz = 1 if ndim == 2 else int(block_nx_model[2])
    sfc_data = np.zeros((nleafs, nfields, block_nx_model[0], block_nx_model[1], bz), dtype=np.double)
    mesh.uniform_to_sfc(uniform_data, sfc_data)

    return sfc_data, ndim, domain_nx_model, block_nx_model, is_leaf, tree_info, forest, mesh


def write_datfile_from_uniform(file_path: str, udata: np.ndarray, w_names: list[str],
                               xmin: np.ndarray, xmax: np.ndarray, block_nx: np.ndarray,
                               overwrite: bool = False, **header_updates) -> dict:
    """
    Write an AMRVAC .dat file from uniform data by first arranging it into
    canonical Morton/SFC-ordered blocks.
    """
    ds = load_from_uniform(udata, w_names, xmin, xmax, block_nx, **header_updates)
    return ds.write_datfile(file_path, overwrite=overwrite)


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
        Minimum coordinates of the domain, shape (3,), or shape (2,) for
        singleton-z 2D data.
    xmax : np.ndarray, optional
        Maximum coordinates of the domain, shape (3,), or shape (2,) for
        singleton-z 2D data. If not provided, unit spacing is assumed.
    """

    if os.path.exists(filename):
        raise FileExistsError(f"File {filename} already exists")

    assert len(udata.shape) == 4, f"udata must be a 4D array, but got {len(udata.shape)}"
    assert udata.shape[0] == len(w_names), f"w_names must be of size {udata.shape[0]}, but got {len(w_names)}"

    nx, ny, nz = udata.shape[1:]
    nw = udata.shape[0]
    xmin = np.asarray(xmin, dtype=np.double)
    if len(xmin) == 2 and nz == 1:
        xmin = np.array([xmin[0], xmin[1], 0.0], dtype=np.double)
    assert len(xmin) == 3, f"xmin must have 3 elements, but got {len(xmin)}"
    
    # Compute spacing
    if xmax is not None:
        xmax = np.asarray(xmax, dtype=np.double)
        if len(xmax) == 2 and nz == 1:
            xmax = np.array([xmax[0], xmax[1], 0.0], dtype=np.double)
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
