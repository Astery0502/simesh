import numpy as np
from typing import Iterable
from utils.calculation import spread_32bits

def find_lev1_idx_forest(forest):
    """
    Find the lev1 indices from the forest

    input:
    forest: forest bool list

    output:
    lev1_indices_forest: list of lev1 indices in the forest
    """
    lev1_idx_forest = [] # in forest, all lev1 indices; for writing the new forest bool list

    def iterate_forest(forest, i, level):

        # save the lev1 idx
        if level == 1:
            lev1_idx_forest.append(i)

        # return the next one if the current one is leaf node
        if forest[i]:
            return i+1
        else: # if the current one is parent node, iterate over all children nodes (8 for 3d case)
            childlevel = level + 1 # clean all children nodes over level 1 in the loop
            i = i + 1
            for i in range(8):
                i = iterate_forest(forest, i, childlevel)
            return i

    # start with 0 position and level 1 at first 
    i = 0
    while (i < len(forest)):
        level = 1 # reset level every time when it comes back
        i = iterate_forest(forest, i, level)
    
    return lev1_idx_forest

def find_lev1_idx_leaf(forest):
    """
    Find the lev1 indices from the tree (leaf nodes)

    input:
    tree: tree list
    output:
    lev1_idx_tree: list of lev1 indices in the tree
    """

    lev1_idx_forest = find_lev1_idx_forest(forest)
    lev1_idx_leaf = [] # in leafs, the lev1 block posiiton would be; for indicating the new tree leaf blocks sequence
    j = 0
    for i in range(len(lev1_idx_forest)-1):
        lev1_idx_leaf.append(j)    

        lev1_ng1 = lev1_idx_forest[i]
        lev1_ng2 = lev1_idx_forest[i+1]
        if not forest[lev1_ng1]:
            leaf_num = np.where(forest[lev1_ng1:lev1_ng2] == True)[0].shape[0]
            j += leaf_num
        else:
            j += 1
    lev1_idx_leaf.append(j)

    return lev1_idx_leaf
