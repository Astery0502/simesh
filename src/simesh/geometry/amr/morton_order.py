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
    ng1, ng2, ng3: int, the number of the level 1 blocks

    output:
    bidirection_map: igindex[Tuple[int, int, int]] <-> index [int]

    iglevel1_sfc: from igindex[Tuple[int, int, int]] to index [int]
    sfc_iglevel1: from index [int] to igindex[Tuple[int, int, int]]

    Examples
    --------
    >>> ng1, ng2, ng3 = 3, 3, 3
    >>> iglevel1_sfc, sfc_iglevel1 = level1_Morton_order(ng1, ng2, ng3)
    >>> iglevel1_sfc.shape
    (3, 3, 3)
    >>> sfc_iglevel1.shape
    (27, 3)
    >>> # Test mapping from coordinates to Morton index
    >>> iglevel1_sfc[0,0,0] == 0  # First corner
    np.True_
    >>> iglevel1_sfc[2,2,2] == 26  # Last corner
    np.True_
    >>> # Test mapping from Morton index back to coordinates
    >>> tuple(sfc_iglevel1[0]) == (0, 0, 0)  # First point coordinates
    True
    >>> tuple(sfc_iglevel1[26]) == (2, 2, 2)  # Last point coordinates
    True
    """

    # Calculate smallest power of 2 that fits each dimension
    ngs = np.array([ng1, ng2, ng3])
    ngsq = 2**np.ceil(np.log2(ngs)).astype(int)
    
    # Generate coordinates and Morton codes in sorted order
    coords = np.stack(np.meshgrid(np.arange(ngsq[0]), 
                                 np.arange(ngsq[1]), 
                                 np.arange(ngsq[2]), 
                                 indexing='ij'), axis=-1).reshape(-1, 3)
    morton_codes = np.array([encode_Morton(tuple(coord)) for coord in coords])
    # Sort both arrays based on Morton codes
    sort_idx = np.argsort(morton_codes)
    morton_codes = morton_codes[sort_idx]
    coords = coords[sort_idx]
    
    # Create mapping arrays
    total_number = np.prod(ngsq)
    seq_sfc = morton_codes.copy()
    seq_ig = coords
    
    # Identify points inside/outside domain
    in_domain = ~np.any(coords >= ngs.reshape(1,3), axis=1)
    
    # Adjust Morton codes for points outside domain
    for isq in range(total_number):
        if not in_domain[isq]:
            seq_sfc[isq:] -= 1
    
    # Create final bidirectional mapping
    iglevel1_sfc = np.zeros(ngs, dtype=int)
    sfc_iglevel1 = np.zeros((np.prod(ngs), 3), dtype=int)
    
    # Fill mappings only for points inside domain
    valid_idx = np.where(in_domain)[0]
    for idx in valid_idx:
        coord = tuple(seq_ig[idx])
        iglevel1_sfc[coord] = seq_sfc[idx]
        sfc_iglevel1[seq_sfc[idx]] = seq_ig[idx]
    
    return iglevel1_sfc, sfc_iglevel1

