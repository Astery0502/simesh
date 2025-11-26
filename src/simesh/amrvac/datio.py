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

    # Pre-allocate memory for results: (nblocks, *block_shape, nfields)
    result_shape = (nblocks, ) + block_shape + (len(field_indices),)
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
                    block_fields_all[i, ..., field_idx_idx] = interior.copy()

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
    
    block_fields_all = np.empty((len(block_offsets), ) + block_shape + (len(field_indices),), dtype=np.float64)
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
                block_fields_all[i, ..., field_idx_idx] = interior.copy()
    
    return block_fields_all