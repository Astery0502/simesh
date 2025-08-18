import numpy as np
import struct
import copy
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

def get_metadata(file:str):

    with open(file, "rb") as istream:
        header = get_header(istream)

        # read forest
        istream.seek(header["offset_tree"])
        fmt = ALIGN + (header["nleafs"] + header["nparents"]) * "i"
        forest =  np.array(struct.unpack(fmt, istream.read(struct.calcsize(fmt))), dtype=bool)

        # read tree info
        istream.seek(header["offset_tree"] + (header["nleafs"] + header["nparents"]) * SIZE_LOGICAL)

        # read block levels
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

    return header, forest, block_info
