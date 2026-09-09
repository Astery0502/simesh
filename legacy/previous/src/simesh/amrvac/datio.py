import os
import numpy as np
import mmap
from concurrent.futures import ThreadPoolExecutor
import struct
from typing import Iterable


# Size of basic types (in bytes)
SIZE_LOGICAL = 4
SIZE_INT = 4
SIZE_DOUBLE = 8
NAME_LEN = 16

# For un-aligned data, use '=' (for aligned data set to '')
ALIGN = "="

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

def get_header(istream):
    """Read header from an MPI-AMRVAC 2.0 snapshot.
    istream' should be a file
    opened in binary mode.
    """
    istream.seek(0)
    h = {}

    fmt = ALIGN + "i"
    [h["datfile_version"]] = struct.unpack(fmt, istream.read(struct.calcsize(fmt)))

    if h["datfile_version"] < 3:
        raise OSError("Unsupported AMRVAC .dat file version: %d", h["datfile_version"])

    # Read scalar data at beginning of file
    fmt = ALIGN + 9 * "i" + "d"
    hdr = struct.unpack(fmt, istream.read(struct.calcsize(fmt)))
    [
        h["offset_tree"],
        h["offset_blocks"],
        h["nw"],
        h["ndir"],
        h["ndim"],
        h["levmax"],
        h["nleafs"],
        h["nparents"],
        h["it"],
        h["time"],
    ] = hdr

    # Read min/max coordinates
    fmt = ALIGN + h["ndim"] * "d"
    h["xmin"] = np.array(struct.unpack(fmt, istream.read(struct.calcsize(fmt))))
    h["xmax"] = np.array(struct.unpack(fmt, istream.read(struct.calcsize(fmt))))

    # Read domain and block size (in number of cells)
    fmt = ALIGN + h["ndim"] * "i"
    h["domain_nx"] = np.array(struct.unpack(fmt, istream.read(struct.calcsize(fmt))))
    h["block_nx"] = np.array(struct.unpack(fmt, istream.read(struct.calcsize(fmt))))

    if h["datfile_version"] >= 4:
        # Read periodicity
        fmt = ALIGN + h["ndim"] * "i"  # Fortran logical is 4 byte int
        h["periodic"] = np.array(
            struct.unpack(fmt, istream.read(struct.calcsize(fmt))), dtype=bool
        )

        # Read geometry name
        fmt = ALIGN + NAME_LEN * "c"
        hdr = struct.unpack(fmt, istream.read(struct.calcsize(fmt)))
        h["geometry"] = b"".join(hdr).strip().decode()

        # Read staggered flag
        fmt = ALIGN + "i"  # Fortran logical is 4 byte int
        h["staggered"] = bool(struct.unpack(fmt, istream.read(struct.calcsize(fmt)))[0])

    # if version > 3
    # Read w_names
    w_names = []
    for _ in range(h["nw"]):
        fmt = ALIGN + NAME_LEN * "c"
        hdr = struct.unpack(fmt, istream.read(struct.calcsize(fmt)))
        w_names.append(b"".join(hdr).strip().decode())
    h["w_names"] = w_names

    # Read physics type
    fmt = ALIGN + NAME_LEN * "c"
    hdr = struct.unpack(fmt, istream.read(struct.calcsize(fmt)))
    h["physics_type"] = b"".join(hdr).strip().decode()

    # Read number of physics-defined parameters
    fmt = ALIGN + "i"
    [n_pars] = struct.unpack(fmt, istream.read(struct.calcsize(fmt)))

    # First physics-parameter values are given, then their names
    fmt = ALIGN + n_pars * "d"
    vals = struct.unpack(fmt, istream.read(struct.calcsize(fmt)))

    fmt = ALIGN + n_pars * NAME_LEN * "c"
    names = struct.unpack(fmt, istream.read(struct.calcsize(fmt)))
    # Split and join the name strings (from one character array)
    names = [
        b"".join(names[i : i + NAME_LEN]).strip().decode()
        for i in range(0, len(names), NAME_LEN)
    ]

    # Store additional physics parameters in header
    h["n_par"] = n_pars
    h["params"] = np.array(vals)
    h["param_names"] = names

    # Read snapshot next if not specified 
    fmt = ALIGN + "i"
    [snapshotnext] = struct.unpack(fmt, istream.read(struct.calcsize(fmt)))
    h["snapshotnext"] = snapshotnext
    [slicenext] = struct.unpack(fmt, istream.read(struct.calcsize(fmt)))
    h['slicenext'] = slicenext
    [collapsenext] = struct.unpack(fmt, istream.read(struct.calcsize(fmt)))
    h["collapsenext"] = collapsenext

    return h

def get_forest(istream, header):

    # get the forest
    istream.seek(header["offset_tree"])
    fmt = ALIGN + (header["nleafs"] + header["nparents"]) * "i"
    forest =  np.array(struct.unpack(fmt, istream.read(struct.calcsize(fmt))), dtype=bool)
    return forest

def get_tree(istream, header):

    # read tree info
    istream.seek(header["offset_tree"] + (header["nleafs"] + header["nparents"]) * SIZE_LOGICAL)
    fmt = ALIGN + header["nleafs"] * "i"
    block_lvls = np.array(struct.unpack(fmt, istream.read(struct.calcsize(fmt))))

    # read block indices
    fmt = ALIGN + header["nleafs"] * header["ndim"] * "i"
    block_ids = np.reshape(
        struct.unpack(fmt, istream.read(struct.calcsize(fmt))), [header["nleafs"], header["ndim"]]
    )

    # read block offsets (not skip ghost cells !)
    fmt = ALIGN + header["nleafs"] * "q"
    block_offsets = np.array(struct.unpack(fmt, istream.read(struct.calcsize(fmt))))
    block_info = (block_lvls, block_ids, block_offsets)

    return block_lvls, block_ids, block_offsets

def get_metadata(file:str):

    with open(file, "rb") as istream:
        header = get_header(istream)

        # read forest
        forest = get_forest(istream, header)

        # read tree info
        block_info = get_tree(istream, header)

    return header, forest, block_info

# SEQUENTIAL MMAP VERSION: No parallel processing, just mmap for performance
def read_blocks_mmap_sequential(filename, field_indices=None):
    """
    Sequential block reading using mmap for performance.
    No parallel processing = no mmap conflicts!
    
    Features:
    - Uses mmap for fast file access
    - Sequential processing (no parallel conflicts)
    - Memory efficient
    - Reliable and simple
    """
    
    # Get metadata first
    header, forest, block_info = get_metadata(filename)
    block_offsets = block_info[2].copy()
    block_shape = header['block_nx'].copy()
    ndim = header['ndim']
    nw = header['nw']
    
    if field_indices is None:
        field_indices = list(range(nw))

    # CRITICAL: Delete all variables that might hold file references
    del header, forest, block_info
    block_offsets = [int(x) for x in block_offsets]
    block_shape = tuple(int(x) for x in block_shape)
    
    nblocks = len(block_offsets)
    print(f"Reading {nblocks} blocks sequentially with mmap...")

    # Pre-allocate memory for results: (nblocks, nfields, *block_shape)
    result_shape = (nblocks, len(field_indices)) + block_shape
    block_fields_all = np.empty(result_shape, dtype=np.float64)
    
    # Open file and create mmap
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            
            for i, offset in enumerate(block_offsets):
                ghost_offset = 2 * ndim * SIZE_INT
                
                # Read ghost cells
                ghostcells_view = np.frombuffer(mm, dtype='=i4', count=2*ndim, offset=offset)
                ghostcells = ghostcells_view.reshape(2, ndim)
                
                # Calculate shape with ghost cells (what's stored on disk)
                bg_shape = block_shape + ghostcells[0] + ghostcells[1]
                count = np.prod(bg_shape)
                byte_size_field = count * SIZE_DOUBLE
                
                # Read requested fields
                for field_idx_idx, field_idx in enumerate(field_indices):
                    byte_offset = offset + ghost_offset + field_idx * byte_size_field
                    
                    # Read full bg_shape from disk
                    arr = np.frombuffer(mm, dtype='=f8', count=count, offset=byte_offset)
                    arr = arr.reshape(bg_shape[::-1]).T
                    
                    # Extract interior region by removing ghost cells
                    # Create slice for each dimension
                    slices = []
                    for dim in range(ndim):
                        start = ghostcells[0, dim]
                        end = bg_shape[dim] - ghostcells[1, dim] if ghostcells[1, dim] > 0 else None
                        slices.append(slice(start, end))
                    
                    # Extract interior region (without ghost cells)
                    interior = arr[tuple(slices)]
                    
                    # Store in pre-allocated array
                    block_fields_all[i, field_idx_idx, ...] = interior.copy()

            # close mmap data view
            del interior, ghostcells_view

    return block_fields_all

def read_blocks_sequential(filename, field_indices=None):
    """
    Parallel block reading using simple file I/O
    Each worker opens its own file handle, so no shared state issues.
    """
    
    # Get metadata first
    header, forest, block_info = get_metadata(filename)
    block_offsets = block_info[2]
    block_shape = tuple(header['block_nx'])
    ndim = header['ndim']
    nw = header['nw']
    
    if field_indices is None:
        field_indices = list(range(nw))
    
    # Pre-allocate memory for results: (nblocks, nfields, *block_shape)
    block_fields_all = np.empty((len(block_offsets), len(field_indices)) + block_shape, dtype=np.float64)
    with open(filename, 'rb') as f:
        for i, offset in enumerate(block_offsets):
            f.seek(offset)
            
            # Read ghost cells
            nghostcells = np.frombuffer(f.read(2 * ndim * SIZE_INT), dtype='=i4').reshape(2, ndim)
            bg_shape = block_shape + nghostcells[0] + nghostcells[1]
            count = np.prod(bg_shape)
            byte_size_field = count * SIZE_DOUBLE
            
            # Read requested fields
            for field_idx_idx, field_idx in enumerate(field_indices):
                f.seek(offset + 2 * ndim * SIZE_INT + field_idx * byte_size_field)
                arr = np.frombuffer(f.read(byte_size_field), dtype='=f8')
                arr = arr.reshape(bg_shape[::-1]).T
                
                # Extract interior region by removing ghost cells
                # Create slice for each dimension
                slices = []
                for dim in range(ndim):
                    start = nghostcells[0, dim]
                    end = bg_shape[dim] - nghostcells[1, dim] if nghostcells[1, dim] > 0 else None
                    slices.append(slice(start, end))
                
                # Extract interior region (without ghost cells)
                interior = arr[tuple(slices)]
                block_fields_all[i, field_idx_idx, ...] = interior.copy()
    
    return block_fields_all


def _normalize_field_indices(header: dict, field_indices=None) -> list[int]:
    """
    Normalize and validate field indices against the original header ordering.
    """
    nw = int(header["nw"])
    if field_indices is None:
        return list(range(nw))

    normalized = [int(i) for i in field_indices]
    if any(i < 0 or i >= nw for i in normalized):
        raise ValueError(f"field_indices must be within [0, {nw - 1}], got {normalized}")
    return normalized


def extract_uniform_data(file_path: str, field_indices=None) -> tuple[np.ndarray, dict]:
    """
    Extract level-1 AMRVAC data directly into a uniform grid.

    Returns
    -------
    udata : np.ndarray
        Uniform data with shape (nx, ny, nz, n_selected_fields).
    header : dict
        Header metadata with `w_names` and `nw` updated to match the selected
        fields when `field_indices` is provided.
    """
    header, _forest, tree_info = get_metadata(file_path)
    if int(header["levmax"]) != 1:
        raise ValueError("extract_uniform_data() is only available when levmax == 1.")

    field_indices = _normalize_field_indices(header, field_indices)
    block_data = read_blocks_sequential(file_path, field_indices)

    domain_nx = np.asarray(header["domain_nx"], dtype=np.int32)
    block_nx = np.asarray(header["block_nx"], dtype=np.int32)
    block_ixs = np.asarray(tree_info[1], dtype=np.int32)

    if int(header["ndim"]) == 2:
        udata = np.zeros((domain_nx[0], domain_nx[1], 1, len(field_indices)), dtype=np.float64)
    else:
        udata = np.zeros((*domain_nx, len(field_indices)), dtype=np.float64)
    for ileaf, block_idx in enumerate(block_ixs):
        if int(header["ndim"]) == 2:
            x0, y0 = (block_idx - 1) * block_nx
            x1, y1 = block_idx * block_nx
            udata[x0:x1, y0:y1, 0, :] = np.transpose(block_data[ileaf], (1, 2, 0))
        else:
            x0, y0, z0 = (block_idx - 1) * block_nx
            x1, y1, z1 = block_idx * block_nx
            udata[x0:x1, y0:y1, z0:z1, :] = np.transpose(block_data[ileaf], (1, 2, 3, 0))

    header_out = header.copy()
    header_out["nw"] = len(field_indices)
    header_out["w_names"] = [header["w_names"][i] for i in field_indices]

    return udata, header_out

def get_tree_size(header):

    if header['datfile_version'] < 3:
        raise OSError("Unsupported AMRVAC .dat file version: %d", header["datfile_version"])

    tree_size = 0
    tree_size += 10 * SIZE_INT # first 10 integers fixed
    tree_size += SIZE_DOUBLE # time
    
    for key, value in header.items():
        if key in ['w_names', 'param_names']:
            tree_size += len(value) * NAME_LEN
        elif key in ['xmin', 'xmax', 'params']:
            tree_size += len(value) * SIZE_DOUBLE
        elif key in ['domain_nx', 'block_nx']:
            tree_size += len(value) * SIZE_INT

    if header['datfile_version'] >= 4:
        tree_size += SIZE_INT * header['ndim'] # periodic conditions
        tree_size += NAME_LEN # geometry name
        tree_size += SIZE_INT # staggered flag

    tree_size += NAME_LEN # physics type
    tree_size += SIZE_INT # number of physics-defined parameters: n_par
    tree_size += 3 * SIZE_INT # snapshotnext, slicenext, collapsenext

    offset_size = tree_size + SIZE_INT*(header['nleafs'] + header['nparents']) # the forest
    offset_size += SIZE_INT*header['nleafs'] # the block levels
    offset_size += SIZE_INT*header['nleafs']*header['ndim'] # the block indices
    offset_size += SIZE_DOUBLE*header['nleafs'] # the block offsets with long long int
    
    return tree_size, offset_size

def write_header(fi, header):
    """
    write the amrvac header to the .dat file
    """
    fi.seek(0)

    fmt = ALIGN + "i"
    size = struct.calcsize(fmt)
    packed_data = struct.pack(fmt, header['datfile_version'])
    fi.write(packed_data)

    fmt = ALIGN + 9 * "i"  + "d"
    packed_data = struct.pack(fmt,
        header["offset_tree"],
        header["offset_blocks"],
        header["nw"],
        header["ndir"],
        header["ndim"],
        header["levmax"],
        header["nleafs"],
        header["nparents"],
        header["it"],
        header["time"],
     )
    fi.write(packed_data)

    # 
    fmt = ALIGN + header['ndim'] * "d"
    packed_data = struct.pack(fmt, *header['xmin'])
    fi.write(packed_data)
    packed_data = struct.pack(fmt, *header['xmax'])
    fi.write(packed_data)

    # 
    fmt = ALIGN + header["ndim"] * "i"
    packed_data = struct.pack(fmt, *header["domain_nx"])
    fi.write(packed_data)
    packed_data = struct.pack(fmt, *header["block_nx"])
    fi.write(packed_data)

    # 
    if header["datfile_version"] >= 5:
        fmt = ALIGN + header["ndim"] * "i"
        # Convert boolean array to integers for struct.pack
        periodic = header["periodic"]
        if isinstance(periodic, np.ndarray) and periodic.dtype == bool:
            periodic = periodic.astype(np.int32)
        packed_data = struct.pack(fmt, *periodic)
        fi.write(packed_data)

        decoded_data = header["geometry"].encode().ljust(NAME_LEN)
        fi.write(decoded_data) 

        fmt = ALIGN + "i"
        # Convert boolean to integer for struct.pack
        staggered = header["staggered"]
        if isinstance(staggered, bool):
            staggered = int(staggered)
        packed_data = struct.pack(fmt, staggered)
        fi.write(packed_data)

    # Write w_names
    for i in range(header['nw']):
        decoded_data = header['w_names'][i].encode().ljust(NAME_LEN)
        fi.write(decoded_data)
    
    # Write physics_type
    decoded_data = header["physics_type"].encode().ljust(NAME_LEN)
    fi.write(decoded_data)

    # Write number of physics-defined parameters
    fmt  = ALIGN + "i"
    packed_data = struct.pack(fmt, header["n_par"]) # n_pars = 1
    fi.write(packed_data)

    # Write physics-parameter values
    fmt = ALIGN + header["n_par"] * "d"
    packed_data = struct.pack(fmt, *header['params'])
    fi.write(packed_data)

    # Write physics-parameter names
    for i in range(header['n_par']):
        decoded_data = header["param_names"][i].encode().ljust(NAME_LEN)
        fi.write(decoded_data)

    # Write snapshotnext, slicenext, and collapsenext
    fmt = ALIGN + 1 * "i"
    packed_data = struct.pack(fmt, header["snapshotnext"])
    fi.write(packed_data)

    packed_data = struct.pack(fmt, header["slicenext"])
    fi.write(packed_data)

    packed_data = struct.pack(fmt, header["collapsenext"])
    fi.write(packed_data)

    assert(fi.tell()) == header['offset_tree'], f"Header is not written correctly, {fi.tell()} != {header['offset_tree']}"
    return fi.tell()

def update_header(header: dict, **kwargs):
    """
    Create a new header dictionary with updated values from kwargs.
    Validates that kwargs only contain standard header keywords from the template.
    
    Args:
        header: Original header dictionary
        **kwargs: Keyword arguments to update in the header
        
    Returns:
        Updated header dictionary
        
    Raises:
        ValueError: If any key in kwargs is not a standard header keyword
    """
    # Get all standard header keywords from the template
    standard_keys = set(header_template.keys())
    
    # Validate that all kwargs are standard header keywords
    for key in kwargs.keys():
        if key not in standard_keys:
            raise ValueError(f"Key '{key}' is not a standard header keyword. "
                           f"Valid keys are: {sorted(standard_keys)}")
    
    # Create a copy of the header and update with kwargs
    header_new = header.copy()
    for key, value in kwargs.items():
        header_new[key] = value

    tree_size, offset_size = get_tree_size(header_new)
    header_new['offset_tree'] = tree_size
    header_new['offset_blocks'] = offset_size
    return header_new


def _normalize_header_for_sfc_write(header: dict, data: np.ndarray) -> dict:
    """
    Normalize header values for writing Morton/SFC-ordered block data.
    """
    data = np.asarray(data)
    if data.ndim != 5:
        raise ValueError(f"data must be a 5D array with shape (nleafs, nfields, bx, by, bz), got {data.shape}")

    nleafs, nfields, bx, by, bz = data.shape
    ndim = int(header["ndim"])
    if ndim not in (2, 3):
        raise ValueError(f"Only 2D and 3D canonical AMRVAC writing is supported for now, got ndim={header['ndim']}")
    if ndim == 2 and bz != 1:
        raise ValueError(f"2D AMRVAC writing expects singleton z block size, got bz={bz}")

    header_new = header.copy()
    header_new["nw"] = int(nfields)
    header_new["nleafs"] = int(nleafs)
    if ndim == 2:
        header_new["block_nx"] = np.array([bx, by], dtype=np.int32)
    else:
        header_new["block_nx"] = np.array([bx, by, bz], dtype=np.int32)

    if "w_names" not in header_new:
        raise ValueError("header must contain 'w_names'")
    if len(header_new["w_names"]) != nfields:
        raise ValueError(
            f"header['w_names'] length must match the number of fields in data, "
            f"{len(header_new['w_names'])} != {nfields}"
        )

    if "nparents" not in header_new:
        raise ValueError("header must contain 'nparents'")

    tree_size, offset_size = get_tree_size(header_new)
    header_new["offset_tree"] = tree_size
    header_new["offset_blocks"] = offset_size

    return header_new


def _block_record_size(header: dict) -> int:
    """
    Size in bytes of one block record written without ghost cells.
    """
    ghost_bytes = 2 * int(header["ndim"]) * SIZE_INT
    field_bytes = int(np.prod(header["block_nx"])) * int(header["nw"]) * SIZE_DOUBLE
    return ghost_bytes + field_bytes


def _build_block_offsets(header: dict, nleafs: int) -> np.ndarray:
    """
    Build contiguous block offsets for a zero-ghostcell SFC write.
    """
    block_bytes = _block_record_size(header)
    start = int(header["offset_blocks"])
    return start + np.arange(nleafs, dtype=np.int64) * block_bytes


def _validate_sfc_tree(header: dict, is_leaf: np.ndarray, tree_info):
    """
    Validate forest/tree metadata for canonical SFC writing.
    """
    if not isinstance(tree_info, tuple) or len(tree_info) != 3:
        raise ValueError("tree_info must be a tuple of (block_lvls, block_ixs, block_offsets)")

    is_leaf = np.asarray(is_leaf, dtype=np.int32)
    block_lvls = np.asarray(tree_info[0], dtype=np.int32)
    block_ixs = np.asarray(tree_info[1], dtype=np.int32)

    nleafs = int(header["nleafs"])
    nparents = int(header["nparents"])
    ndim = int(header["ndim"])

    if is_leaf.shape != (nleafs + nparents,):
        raise ValueError(
            f"is_leaf must have length nleafs + nparents, {is_leaf.shape} != {(nleafs + nparents,)}"
        )
    if block_lvls.shape != (nleafs,):
        raise ValueError(f"block_lvls must have shape ({nleafs},), got {block_lvls.shape}")
    if block_ixs.shape != (nleafs, ndim):
        raise ValueError(f"block_ixs must have shape ({nleafs}, {ndim}), got {block_ixs.shape}")
    if np.any(block_lvls < 1):
        raise ValueError("block_lvls must be positive integers")
    if np.any(block_ixs < 1):
        raise ValueError("block_ixs must use AMRVAC's 1-based indexing")

    return is_leaf, block_lvls, block_ixs


def write_datfile_from_sfc(file_path: str, data: np.ndarray, header: dict, is_leaf: np.ndarray,
                           tree_info, overwrite: bool = False) -> dict:
    """
    Write an AMRVAC .dat file from Morton/SFC-ordered block data.

    Parameters
    ----------
    data : np.ndarray
        Array with shape (nleafs, nfields, bx, by, bz).
    header : dict
        Header metadata. Offsets and a subset of shape-dependent fields are
        normalized from `data` before writing.
    is_leaf : np.ndarray
        Forest flags of length nleafs + nparents.
    tree_info : tuple
        Tuple of (block_lvls, block_ixs, block_offsets). The incoming
        block_offsets are ignored and rebuilt for the output file.
    overwrite : bool
        If False, refuse to overwrite an existing file.
    """
    data = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
    header_new = _normalize_header_for_sfc_write(header, data)
    is_leaf_arr, block_lvls, block_ixs = _validate_sfc_tree(header_new, is_leaf, tree_info)
    block_offsets = _build_block_offsets(header_new, int(header_new["nleafs"]))

    if not overwrite and os.path.exists(file_path):
        raise FileExistsError(f"File {file_path} already exists")

    with open(file_path, "wb") as fb:
        write_header(fb, header_new)
        write_forest_tree(fb, header_new, is_leaf_arr, (block_lvls, block_ixs, block_offsets))
        write_blocks(fb, data, header_new["ndim"], block_offsets)

    return header_new


def write_forest_tree(fi, header, forest, tree):

    fi.seek(header['offset_tree'])
    len_forest = len(forest)
    assert(len_forest == (header['nleafs'] + header['nparents'])), f"Forest data is not written correctly, {len_forest} != {header['nleafs'] + header['nparents']}"

    fmt = ALIGN + len_forest * "i"
    packed_data = struct.pack(fmt, *forest)
    fi.write(packed_data)

    block_lvls, block_ixs, block_offsets = tree
    assert(len(block_lvls) == len(block_ixs) == len(block_offsets))
    assert(len(block_lvls) == header['nleafs'])
    
    fmt = ALIGN + len(block_lvls) * "i"
    packed_data = struct.pack(fmt, *block_lvls)
    fi.write(packed_data)

    fmt = ALIGN + len(block_ixs) * header['ndim'] * "i"
    packed_data = struct.pack(fmt, *(block_ixs.flatten()))
    fi.write(packed_data)

    fmt = ALIGN + len(block_offsets) * "q"
    packed_data = struct.pack(fmt, *block_offsets)
    fi.write(packed_data)

    assert(fi.tell() == header['offset_blocks']), f"Tree data is not written correctly, {fi.tell()} != {header['offset_blocks']}"
    return fi.tell()

def write_blocks(fi, data, ndim, offsets):

    """
    fi: file buffer for input
    data: blocks of data in the morton order
    offsets: list of offsets for each block in fi
    """

    fi.seek(offsets[0])

    for i in range(len(offsets)):

        offset = offsets[i]
        block_array = data[i]

        fmt = ALIGN + 2 * ndim * "i"
        ghostcells = np.zeros(2 * ndim, dtype=np.int32) # no ghostcells written into
        packed_data = struct.pack(fmt, *ghostcells.flatten())
        fi.write(packed_data)

        fmt = ALIGN + np.prod(block_array.shape) * "d"
        if ndim == 2:
            block_data = np.transpose(block_array[:, :, :, 0], (0, 2, 1)).flatten()
        else:
            block_data = np.transpose(block_array, (0,3,2,1)).flatten()
        packed_data = struct.pack(fmt, *block_data)
        fi.write(packed_data)

        if (i < len(offsets)-1):
            assert(fi.tell() == offsets[i+1]), f"Block data is not written correctly, {fi.tell()} != {offsets[i+1]}"
