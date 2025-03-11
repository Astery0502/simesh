import struct
import copy
import numpy as np
import math
from typing import Union, Iterable


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

def get_forest(istream):
    istream.seek(0)
    header = get_header(istream)
    istream.seek(header["offset_tree"])
    fmt = ALIGN + (header["nleafs"] + header["nparents"]) * "i"
    return np.array(struct.unpack(fmt, istream.read(struct.calcsize(fmt))), dtype=bool)

def get_tree_info(istream):
    """
    Read levels, morton-curve indices, and byte offsets for each block as stored in the
    datfile istream is an open datfile buffer with 'rb' mode
    This can be used as the "first pass" data reading required by YT's interface.
    """
    istream.seek(0)
    header = get_header(istream)
    nleafs = header["nleafs"]
    nparents = header["nparents"]

    # Read tree info. Skip 'leaf' array
    istream.seek(header["offset_tree"] + (nleafs + nparents) * SIZE_LOGICAL)

    # Read block levels
    fmt = ALIGN + nleafs * "i"
    block_lvls = np.array(struct.unpack(fmt, istream.read(struct.calcsize(fmt))))

    # Read block indices
    fmt = ALIGN + nleafs * header["ndim"] * "i"
    block_ixs = np.reshape(
        struct.unpack(fmt, istream.read(struct.calcsize(fmt))), [nleafs, header["ndim"]]
    )

    # Read block offsets (not skip ghost cells !)
    fmt = ALIGN + nleafs * "q"
    block_offsets = (
        np.array(struct.unpack(fmt, istream.read(struct.calcsize(fmt))))
    )
    return block_lvls, block_ixs, block_offsets

# get all data from a single block containging the ghost cells info and ixG^L block data
def get_single_block_data(istream, offset):

    header = get_header(istream)
    block_shape = header['block_nx']
    nw = header['nw']

    istream.seek(offset)
    bcfmt = ALIGN + 2 * header["ndim"] * "i"
    ghostcells = np.reshape(struct.unpack(bcfmt, istream.read(struct.calcsize(bcfmt))), [2, header["ndim"]])
    block_shape = block_shape + ghostcells[0] + ghostcells[1]

    if header['staggered']:
        # Read regular field data
        fmt = ALIGN + (nw * np.prod(block_shape)) * "d"
        field_data = struct.unpack(fmt, istream.read(struct.calcsize(fmt)))
        
        # Read staggered field data
        fmt = ALIGN + (3 * np.prod(block_shape+1)) * "d" 
        staggered_data = struct.unpack(fmt, istream.read(struct.calcsize(fmt)))
        block_data = (field_data, staggered_data)
    else:
        fmt = ALIGN + (nw * np.prod(block_shape)) * "d"
        block_data = struct.unpack(fmt, istream.read(struct.calcsize(fmt)))

    ## check if the block data is read correctly
    # tree = get_tree_info(istream)
    # offsets = tree[2]
    # j = np.where(offsets == offset)[0][0]
    # if j < header['nleafs']:
    #     assert(istream.tell() == offsets[j+1]), f"Block data is not read correctly, from {offset}, {istream.tell()} != {offsets[j+1]}"

    return (ghostcells, block_data)

def get_single_block_field_data(istream, offset, block_shape, field_idx:int, ndim:int=3):
    """Retrieve field data from a grid.

    Parameters
    ----------
    istream: file descriptor (open binary file with read access)

    grid : yt.frontends.amrvac.data_structures.AMRVACGrid
        The grid from which data is to be read.
    field : str
        A field name.

    Returns
    -------
    data : np.ndarray
        A 3D array of float64 type representing grid data.

    """
    count = np.prod(block_shape)
    byte_size_field = count * SIZE_DOUBLE  # size of a double

    istream.seek(offset + byte_size_field * field_idx + 2*ndim*SIZE_INT)
    data = np.fromfile(istream, "=f8", count=count)
    data.shape = block_shape[::-1] # reverse the block_nx sequence
    data = data.T                  # for fortran ordering correction
    # Always convert data to 3D, as grid.ActiveDimensions is always 3D
    while len(data.shape) < 3:
        data = data[..., np.newaxis]
    return data

def write_single_block_field_data(fb, offset: int, block_shape, field_idx: int, ndim: int, data):
    """
    Write a single field to a dat file
    for now ct surface variables are not supported

    Parameters
    ----------
    fb: file buffer
    offset: int
    block_shape: tuple
    field_idx: int
    ndim: int
    data: np.ndarray
    """

    count = np.prod(block_shape)
    byte_size_field = count * SIZE_DOUBLE
    fb.seek(offset + byte_size_field * field_idx + 2*ndim*SIZE_INT)
    block_data = (np.asarray(data).T).flatten()

    fmt = ALIGN + np.prod(block_shape) * "d"
    packed_data = struct.pack(fmt, *block_data)
    fb.write(packed_data)

def find_uniform_fields(fb, header, tree):

    if header['levmax'] != 1:
        raise ValueError("The mesh is not uniform")

    domain_nx = header['domain_nx']
    block_nx = header['block_nx']
    nleafs = header['nleafs']
    lenw = header['nw']

    fields = np.zeros((*domain_nx, lenw), dtype=np.float64)

    for ileaf in range(nleafs):
        block_idx = tree[1][ileaf]
        offset = tree[2][ileaf]

        x0, y0, z0 = (block_idx-1) * block_nx
        x1, y1, z1 = block_idx * block_nx

        _ghostcells, block_data = get_single_block_data(fb, offset)

        if header['staggered']:
            field_data, staggered_data = block_data
        else:
            field_data = block_data

        fields[x0:x1, y0:y1, z0:z1, :] = np.transpose(np.array(field_data).reshape(lenw, *block_nx), (3,2,1,0))
    
    return fields

def extract_uniform_fields(file:str, field_indices:Union[int, Iterable[int]]):

    with open(file, 'rb+') as fi:

        header = get_header(fi)
        assert (header['levmax'] == 1), "The mesh is not uniform"
        domain_nx = header['domain_nx']
        block_nx = header['block_nx']
        nblock_nx = (header['domain_nx'] / header['block_nx']).astype(np.int32)
        nleafs = header['nleafs']
        lenw = len(header['w_names'])

        if isinstance(field_indices, int):
            field_indices = [field_indices]
        assert lenw > max(field_indices), "Field index is out of range"
        assert isinstance(field_indices, list), "Field indices must be a list or iterable"

        tree = get_tree_info(fi)
        block_indices = tree[1]
        block_offsets = tree[2]

        fields = np.zeros((*domain_nx, len(field_indices)), dtype=np.float64)

        for ileaf in range(nleafs):
            block_idx = tree[1][ileaf]
            offset = tree[2][ileaf]

            x0, y0, z0 = (block_idx-1) * block_nx
            x1, y1, z1 = block_idx * block_nx
            for i, field_idx in enumerate(field_indices):
                field = get_single_block_field_data(fi, offset, block_nx, header['ndim'], field_idx)
                fields[x0:x1, y0:y1, z0:z1, i] = field
        
        return fields

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
        packed_data = struct.pack(fmt, *header["periodic"])
        fi.write(packed_data)

        decoded_data = header["geometry"].encode().ljust(NAME_LEN)
        fi.write(decoded_data) 

        fmt = ALIGN + "i"
        packed_data = struct.pack(fmt, header["staggered"])
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
        block_data = np.transpose(block_array, (3,2,1,0)).flatten()
        packed_data = struct.pack(fmt, *block_data)
        fi.write(packed_data)

        if (i < len(offsets)-1):
            assert(fi.tell() == offsets[i+1]), f"Block data is not written correctly, {fi.tell()} != {offsets[i+1]}"

def write_new_header(fi, header, **kwargs):

    header_old = copy.deepcopy(header)

    for key, value in kwargs.items():
        if key in header_old:
            header_old[key] = value
        else:
            raise ValueError(f"Key '{key}' not found in header")

    fi.seek(0)
    fmt = ALIGN + "i"
    size = struct.calcsize(fmt)
    packed_data = struct.pack(fmt, header_old['datfile_version'])
    fi.write(packed_data)

    fmt = ALIGN + 9 * "i"  + "d"
    packed_data = struct.pack(fmt,
        header_old["offset_tree"],
        header_old["offset_blocks"],
        header_old["nw"],
        header_old["ndir"],
        header_old["ndim"],
        header_old["levmax"],
        header_old["nleafs"],
        header_old["nparents"],
        header_old["it"],
        header_old["time"],
     )
    fi.write(packed_data)

    # 
    fmt = ALIGN + header_old['ndim'] * "d"
    packed_data = struct.pack(fmt, *header_old['xmin'])
    fi.write(packed_data)
    packed_data = struct.pack(fmt, *header_old['xmax'])
    fi.write(packed_data)

    # 
    fmt = ALIGN + header_old["ndim"] * "i"
    packed_data = struct.pack(fmt, *header_old["domain_nx"])
    fi.write(packed_data)
    packed_data = struct.pack(fmt, *header_old["block_nx"])
    fi.write(packed_data)

    # 
    if header_old["datfile_version"] >= 5:
        fmt = ALIGN + header_old["ndim"] * "i"
        packed_data = struct.pack(fmt, *header_old["periodic"])
        fi.write(packed_data)

        decoded_data = header_old["geometry"].encode().ljust(NAME_LEN)
        fi.write(decoded_data) 

        fmt = ALIGN + "i"
        packed_data = struct.pack(fmt, header_old["staggered"])
        fi.write(packed_data)

    # Write w_names
    for i in range(header_old['nw']):
        decoded_data = header_old['w_names'][i].encode().ljust(NAME_LEN)
        fi.write(decoded_data)
    
    # Write physics_type
    decoded_data = header_old["physics_type"].encode().ljust(NAME_LEN)
    fi.write(decoded_data)

    # Write number of physics-defined parameters
    fmt  = ALIGN + "i"
    packed_data = struct.pack(fmt, header_old["n_par"]) # n_pars = 1
    fi.write(packed_data)

    # Write physics-parameter values
    fmt = ALIGN + header_old["n_par"] * "d"
    packed_data = struct.pack(fmt, *header_old['params'])
    fi.write(packed_data)

    # Write physics-parameter names
    for i in range(header_old['n_par']):
        decoded_data = header_old["param_names"][i].encode().ljust(NAME_LEN)
        fi.write(decoded_data)

    # Write snapshotnext, slicenext, and collapsenext
    fmt = ALIGN + 1 * "i"
    packed_data = struct.pack(fmt, header_old["snapshotnext"])
    fi.write(packed_data)

    packed_data = struct.pack(fmt, header_old["slicenext"])
    fi.write(packed_data)

    packed_data = struct.pack(fmt, header_old["collapsenext"])
    fi.write(packed_data)

    assert(fi.tell()) == header_old['offset_tree']
    return fi.tell()

def write_block_data(fi, fr, offsets):

    """
    fi: file buffer for input
    fr: file buffer for reading
    offsets: list of offsets for each block in fr
    """

    header = get_header(fi)
    tree = get_tree_info(fi)
    block_offsets = tree[2]
    fi.seek(header['offset_blocks'])

    assert(header['nleafs'] == len(offsets))
    assert(header['offset_blocks'] == block_offsets[0]), f"Block data is not written correctly, {header['offset_blocks']} != {block_offsets[0]}"

    for i in range(len(offsets)):

        offset = offsets[i]
        ghostcells, block_data = get_single_block_data(fr, offset)
        block_len = len(block_data)

        fmt = ALIGN + 2 * header['ndim'] * "i"
        packed_data = struct.pack(fmt, *ghostcells.flatten())
        fi.write(packed_data)

        fmt = ALIGN + block_len * "d"
        packed_data = struct.pack(fmt, *block_data)
        fi.write(packed_data)

        if (i < len(offsets)-1):
            assert(fi.tell() == block_offsets[i+1])

def selected_lev1_indices(fi, nx, ny, nz):

    """
    return the indices of the selected blocks in the lev1 tree
    """

    header = get_header(fi)
    nw, ndir, ndim = header['nw'], header['ndir'], header['ndim']
    domain_nx = header['domain_nx']
    block_nx = header['block_nx']
    numblocks = np.array(domain_nx / block_nx, dtype=int)

    assert((nx[1] <= numblocks[0]) and (ny[1] <= numblocks[1]) and (nz[1] <= numblocks[2]))
    nglev1_morton_all = nglev1_morton([0,numblocks[0]], [0,numblocks[1]], [0,numblocks[2]])
    nglev1_morton_selected = nglev1_morton([nx[0], nx[1]], [ny[0], ny[1]], [nz[0], nz[1]])

    selected_indices = nglev1_selected_indices(nglev1_morton_all, nglev1_morton_selected)

    # sort the normalised selected indices
    nglev1_morton_normalised = nglev1_morton_selected - np.array([nx[0], ny[0], nz[0]])
    morton_normalised = [interleave_bits(index) for index in nglev1_morton_normalised]
    iindex_sorted = np.argsort(morton_normalised)
    return np.array(selected_indices)[iindex_sorted]

def create_new_tree(fi, nx, ny, nz):

    header = get_header(fi)
    forest = get_forest(fi)

    selected_indices = selected_lev1_indices(fi, nx, ny, nz)
    lev1_indices_tree = read_lev1_indices_from_tree(header, forest)
    tree = get_tree_info(fi)

    lev1_index_tree = [lev1_indices_tree[lev1_index] for lev1_index in selected_indices]

    levels = tree[0].tolist()
    indices = tree[1].tolist()
    offsets = tree[2].tolist()

    levels_new = []
    indices_new = []
    offsets_new = []

    for i in range(len(lev1_index_tree)):
        i1 = lev1_index_tree[i]
        i_lev1_indices_tree = lev1_indices_tree.index(i1)
        if i_lev1_indices_tree < len(lev1_indices_tree)-1:
            levels_new.extend(levels[i1:lev1_indices_tree[i_lev1_indices_tree+1]])
            indices_new.extend(indices[i1:lev1_indices_tree[i_lev1_indices_tree+1]])
            offsets_new.extend(offsets[i1:lev1_indices_tree[i_lev1_indices_tree+1]])
        else: # if it is the last block
            levels_new.extend(levels[i1:])
            indices_new.extend(indices[i1:])
            offsets_new.extend(offsets[i1:])

    assert(len(levels_new) == len(indices_new) == len(offsets_new))

    return levels_new, indices_new, offsets_new

def check_new_tree(fi, nx:int, ny:int, nz:int):

    header = get_header(fi)
    tree = get_tree_info(fi)

    lvls, indices, offsets = tree
    lvls_new, indices_new, offsets_new = create_new_tree(fi, nx, ny, nz)

    count = 0

    for i in range(len(lvls)):
        lx = math.ceil(indices[i][0]/2**(lvls[i]-1))-1
        ly = math.ceil(indices[i][1]/2**(lvls[i]-1))-1
        lz = math.ceil(indices[i][2]/2**(lvls[i]-1))-1 
        if ((lx >= nx[0]) and (lx < nx[1]) and (ly >= ny[0]) and (ly < ny[1]) and (lz >= nz[0]) and (lz < nz[1])):
            count += 1
            assert(any(np.all(index_new == indices[i]) for index_new in indices_new)), f"Block index {indices[i]} is not in the new tree"
    
    assert(count == len(lvls_new)), f"Number of blocks in the new tree is not correct, {count} != {len(lvls_new)}"
    # not applied as multiple level blocks in the indices list
    # morton_number_new = [interleave_bits(index) for index in indices_new]
    # assert(all(x < y for x, y in zip(morton_number_new, morton_number_new[1:])))

def write_new_datfile(fi, path, nx, ny, nz):

    # create new forest and tree with new nx,ny,nz
    forest_new = create_new_forest(fi, nx, ny, nz)
    lvls_new, indices_new, offsets_new = create_new_tree(fi, nx, ny, nz)

    # new tree properties: nleafs, nparents, levmax
    nleafs = sum(forest_new)
    nparents = len(forest_new) - nleafs
    levmax = max(lvls_new)
    assert(len(lvls_new) == nleafs)

    # check with new forest and tree
    check_new_tree(fi, nx, ny, nz)

    # read some former properties for modification
    header = get_header(fi)
    offset_tree = header["offset_tree"]
    xmin = header['xmin']
    xmax = header['xmax']
    block_nx = header['block_nx']
    domain_nx = header['domain_nx']
    numblocks = np.array(domain_nx / block_nx, dtype=int)

    tree = get_tree_info(fi)
    offsets_blocks = tree[2]

    # generate some new properties with elder ones
    xmin_new = xmin + np.array([nx[0], ny[0], nz[0]]) * (block_nx/domain_nx) * (xmax - xmin)
    xmax_new = xmin + np.array([nx[1], ny[1], nz[1]]) * (block_nx/domain_nx) * (xmax - xmin)
    domain_nx_new = np.array([nx[1] - nx[0], ny[1] - ny[0], nz[1] - nz[0]]) * block_nx

    # calculate new offsets of all blocks
    offset_tree_new = offset_tree
    offset_block_new = offset_tree + (nleafs + nparents) * SIZE_LOGICAL + \
                        nleafs * (SIZE_INT + header['ndim'] * SIZE_INT + SIZE_DOUBLE)

    relative_offsets = []
    relative_offset = 0
    for i in range(nleafs):
        # the append would run for nleafs times
        relative_offsets.append(relative_offset)
        if i == nleafs-1:
            break
        ioffset = np.where(offsets_blocks == offsets_new[i])[0][0]
        relative_offset += (offsets_blocks[ioffset+1] - offsets_blocks[ioffset])
    offset_blocks_new = np.array(relative_offsets) + offset_block_new
    assert(len(offset_blocks_new) == nleafs)

    # calculate new lvls for all blocks
    for i in range(nleafs):
        indices_new[i] -= np.array([nx[0], ny[0], nz[0]]) * 2**(lvls_new[i]-1)

    with open(path, 'rb+') as fw:
        write_header(fw, header, offset_tree=offset_tree_new, offset_blocks=offset_block_new,
                    levmax=levmax, nleafs=nleafs, nparents=nparents, 
                    xmin=xmin_new, xmax=xmax_new, domain_nx=domain_nx_new)
        
        write_forest_tree(fw, forest_new, (lvls_new, np.array(indices_new), offset_blocks_new))
        write_block_data(fw, fi, offsets_new)

def check_tree(fi):
    """
    Check if the tree is consistent with the header information.
    From 1. The nparents and nleafs derived from the amr levels tree is consistent with the header [From the leaf levels]
         2. The forest derived nparents and nleafs are consistent with the header [From counting the bool list]
    """

    header = get_header(fi)
    forest = get_forest(fi)
    tree   = get_tree_info(fi)

    nparents_header = header['nparents']
    nleafs_header = header['nleafs']

    tlevels = tree[0]
    level_stat = {}
    for level in tlevels:
        level_stat[level] = level_stat.get(level, 0) + 1

    nparent = 0
    nparents = 0
    nleafs_tree = len(tree[0])
    for i in range(max(tlevels), 0, -1):
        if i == 1:
            break
        nparent = (nparent + level_stat[i]) / 8
        nparents += nparent
    nparents_tree = nparents
    
    assert(nparents_tree == nparents_header)
    assert(nleafs_tree == nleafs_header)

    nleafs_forest = len(forest[forest == True])
    nparents_forest = len(forest[forest == False])

    assert(nparents_forest == nparents_header)
    assert(nleafs_forest == nleafs_header)

    return nparents_header, nleafs_header