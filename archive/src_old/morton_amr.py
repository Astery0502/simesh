import numpy as np
from typing import Iterable

def interleave_bits(ign:Iterable):
    answer = 0
    ndim = len(ign)
    for i in range(0,64//ndim):  

        if ndim == 1:
            return ign[0]

        elif ndim == 2:
            bit_x = (ign[0] >> i) & 1
            bit_y = (ign[1] >> i) & 1

            answer |= (bit_x << (2*i)) | (bit_y << (2*i + 1))
        
        elif ndim == 3:
            bit_x = (ign[0] >> i) & 1
            bit_y = (ign[1] >> i) & 1
            bit_z = (ign[2] >> i) & 1

            answer |= (bit_x << (3*i)) | (bit_y << (3*i + 1)) | (bit_z << (3*i + 2))
        
    return answer

def read_lev1_indices_from_forest(header, forest, numblocks=None):

    if numblocks is None:
        domain_nx = header['domain_nx']
        block_nx = header['block_nx']
        numblocks = np.array(domain_nx / block_nx, dtype=int)
    else:
        numblocks = np.array(numblocks, dtype=int)

    lev1_indices_forest = [] # in forest, all lev1 indices; for writing the new forest bool list

    def read_forest(forest, j, level):

        if level == 1:
            lev1_indices_forest.append(j)

        if forest[j]:
            return j+1
        else:
            childlevel = level + 1
            j = j + 1
            for i in range(8):
                j = read_forest(forest, j, childlevel)
            return j

    nglev1 = np.array([[x, y, z] for x in range(0, numblocks[0]) for y in range(0, numblocks[1])
                        for z in range(0, numblocks[2])])
    j = 0
    for i, grid in enumerate(nglev1):
        level = 1
        j = read_forest(forest, j, level)
    
    return lev1_indices_forest

def read_lev1_indices_from_tree(header, forest, numblocks=None):

    lev1_indices = read_lev1_indices_from_forest(header, forest, numblocks)

    lev1_indices_leaf = [] # in leafs, the lev1 block posiiton would be; for indicating the new tree leaf blocks sequence
    j = 0
    for i in range(len(lev1_indices)-1):
        lev1_indices_leaf.append(j)    

        lev1_ng1 = lev1_indices[i]
        lev1_ng2 = lev1_indices[i+1]
        if not forest[lev1_ng1]:
            leaf_num = np.where(forest[lev1_ng1:lev1_ng2] == True)[0].shape[0]
            j += leaf_num
        else:
            j += 1
    lev1_indices_leaf.append(j)

    return lev1_indices_leaf

def nglev1_morton(nx:Iterable[int], ny:Iterable[int], nz:Iterable[int]):
    """
    Generate a sorted array of 3D coordinates using Morton encoding.

    Parameters:
    nx (Iterable[int]): A range of x-coordinates.
    ny (Iterable[int]): A range of y-coordinates.
    nz (Iterable[int]): A range of z-coordinates.

    Returns:
    np.ndarray: A sorted array of 3D coordinates.

    """
    nglev1 = np.array([[x, y, z] for x in range(nx[0], nx[1]) for y in range(ny[0], ny[1])
                        for z in range(nz[0], nz[1])])
    morton_numbers = np.array([interleave_bits(i) for i in nglev1])
    sorted_indices = np.argsort(morton_numbers)
    sorted_nglev1 = nglev1[sorted_indices]

    return sorted_nglev1

def nglev1_selected_indices(nglev1_all, nglev1_selected):
    """
    Find the indices of the selected nglev1 indices in the nglev1_all array

    nglev1_all: array of all the nglev1 indices
    nglev1_selected: array of the selected nglev1 indices

    return: the indices of the selected nglev1 indices in the nglev1_all array
    """
    return [np.where(np.all(nglev1_all == i, axis=1))[0][0] for i in nglev1_selected]
