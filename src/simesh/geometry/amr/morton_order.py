import numpy as np
from typing import Tuple, Iterable, Optional

"""
The Morton order is a space-filling curve that is used to sort points in a 3D space.
The file refers to the amrvac/src/amr/mod_space_filling_curve.t
"""

def encode_Morton(ign:Tuple[int, int, int]) -> np.int32:
    """
    input:
    ign: Tuple[int, int, int], the index of the grid

    output:
    morton_order: int, the morton order of the grid
    """
    assert len(ign) == 3, "Input must be a 3-element iterable containing x,y,z coordinates"
    
    x, y, z = ign
    # Ensure inputs are 32-bit integers
    x = np.int32(x)
    y = np.int32(y) 
    z = np.int32(z)
    
    def extract_10bits(x):
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x <<  8)) & 0x0300F00F
        x = (x | (x <<  4)) & 0x030C30C3
        x = (x | (x <<  2)) & 0x09249249
        return x
    x, y, z = extract_10bits(x), extract_10bits(y), extract_10bits(z)
    return np.int32(x | (y << 1) | (z << 2))

def level1_Morton_order(ng1:int, ng2:int, ng3:int):
    """
    input:
    ng1, ng2, ng3: int, the index of the grid
    """

    ngs = np.array([ng1, ng2, ng3])
    ngsq = np.array([np.max(2**(np.ceil(np.log(ngs)/np.log(2))))] * 3).astype(int)

    total_number = np.prod(ngsq)

    gsq_sfc = np.zeros(ngsq).astype(int)
    seq_sfc = np.zeros(total_number).astype(int)
    seq_ig  = np.zeros((total_number, 3)).astype(int)
    in_domain = np.ones(total_number, dtype=bool)

    # in python, the index starts from 0, so we do not require to minus 1 in the input of encode_Morton 
    # and the morton order starts from 0
    for i in range(ngsq[0]):
        for j in range(ngsq[1]):
            for k in range(ngsq[2]):
                gsq_sfc[i, j, k] = encode_Morton((i, j, k))
                seq_sfc[gsq_sfc[i,j,k]] = gsq_sfc[i,j,k]
                seq_ig[gsq_sfc[i,j,k]] = [i, j, k]

    # mark the grids out of the domain and shift the morton order
    for isq in range(total_number):
        if (np.any(seq_ig[isq] >= ngs)): # >= is used for the 0-based index in python
            seq_sfc[isq:] -= 1
            in_domain[isq] = False

    # allocate the modified morton order (scaled to [1,nglev1]) and the index of the grid
    iglevel1_sfc = np.zeros(ngs).astype(int)
    sfc_iglevel1 = np.zeros((np.prod(ngs),3)).astype(int)

    for isq in range(total_number):
        if in_domain[isq]:
            iglevel1_sfc[tuple(seq_ig[isq])] = seq_sfc[isq]
            sfc_iglevel1[seq_sfc[isq]] = seq_ig[isq]

    return iglevel1_sfc, sfc_iglevel1

