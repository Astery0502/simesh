import numpy as np
cimport numpy as np

# python ./setup.py build_ext --inplace
cdef np.int64_t interleave_bits(np.ndarray[np.int32_t, ndim=1] ign):
    cdef np.int64_t value = 0
    cdef size_t ndim = len(ign)
    cdef int i
    cdef np.int64_t bit_x, bit_y, bit_z
    
    for i in range(0, 64 // ndim):
        if ndim == 1:
            return ign[0]

        elif ndim == 2:
            bit_x = (ign[0] >> i) & 1
            bit_y = (ign[1] >> i) & 1

            value |= (bit_x << (2 * i)) | (bit_y << (2 * i + 1))

        elif ndim == 3:
            bit_x = (ign[0] >> i) & 1
            bit_y = (ign[1] >> i) & 1
            bit_z = (ign[2] >> i) & 1

            value |= (bit_x << (3 * i)) | (bit_y << (3 * i + 1)) | (bit_z << (3 * i + 2))

    return value

# def find_leaf_indices(np.ndarray[np.float64_t, ndim=2] points, 
#                       np.ndarray[np.int32_t, ndim=1] block_lev1_idices, 
#                       np.ndarray[np.int32_t, ndim=1] leaf_idx_tree, 
#                       np.ndarray[np.float64_t, ndim=2] leaf_origin_lev1,
#                       np.ndarray[np.float64_t, ndim=1] leaf_dblevel_lev1,
#                       np.ndarray[np.int32_t, ndim=1] leaf_indices):

#     cdef int block_lev1_idx
#     cdef np.float64_t point[3]
#     cdef int start_idx, end_idx
#     cdef np.float64_t dpoint[3]

#     for i in range(points.shape[0]):
#         point = points[i]
#         block_lev1_idx = block_lev1_idices[i]

#         start_idx = leaf_idx_tree[block_lev1_idx] 
#         if (block_lev1_idx == leaf_idx_tree.shape[0] - 1):
#             end_idx = leaf_idx_tree[-1]
#         else:
#             end_idx = leaf_idx_tree[block_lev1_idx + 1] - 1

#         for j in range(start_idx, end_idx + 1):
#             dpoint = point - leaf_origin_lev1[j]
#             for k in range(3):
#                 if (dpoint[k] < 0) or (dpoint[k] >= leaf_dblevel_lev1[j]):
#                     break
#                 if k == 2:
#                     leaf_indices[i] = j


# the leaf_origin_lev1, leaf_end_lev1 are already sliced by lev1 index of the point
cdef int find_leaf_idx(np.ndarray[np.float64_t, ndim=1] point,
                  np.ndarray[np.float64_t, ndim=2] leaf_origin_lev1,
                  np.ndarray[np.float64_t, ndim=2] leaf_end_lev1):

    return np.argmax(np.all((point[np.newaxis, :] >= leaf_origin_lev1) &
           (point[np.newaxis, :] <= leaf_end_lev1), axis=-1))


def find_leaf_indices(np.ndarray[np.float64_t, ndim=2] points, 
                      np.ndarray[np.int32_t, ndim=1] block_lev1_indices, 
                      np.ndarray[np.int32_t, ndim=1] leaf_idx_tree, 
                      np.ndarray[np.float64_t, ndim=2] leaf_origin_lev1,
                      np.ndarray[np.float64_t, ndim=1] leaf_dblevel_lev1,
                      np.ndarray[np.int32_t, ndim=1] leaf_indices):

    cdef np.ndarray[np.int64_t, ndim=1] block_lev1_idx
    cdef np.ndarray[np.float64_t, ndim=2] points_idx
    cdef np.int64_t start_idx, end_idx
    cdef np.float64_t dpoint[3]

    cdef np.ndarray[np.float64_t, ndim=2] leaf_end_lev1

    #cdef np.ndarray[np.bool_t, ndim=2] within_bounds

    for idx_lev1 in range(np.max(block_lev1_indices)+1):

        block_lev1_idx = np.where(block_lev1_indices == idx_lev1)[0]
        points_idx = points[block_lev1_idx]

        start_idx = leaf_idx_tree[idx_lev1] 
        if (idx_lev1 == (leaf_idx_tree.shape[0] - 1)):
            end_idx = leaf_idx_tree[-1]
        else:
            end_idx = leaf_idx_tree[idx_lev1 + 1] - 1

        if start_idx == end_idx:
            leaf_indices[block_lev1_idx] = start_idx
            continue

        leaf_end_lev1 = leaf_origin_lev1[start_idx:end_idx+1] + leaf_dblevel_lev1[start_idx:end_idx+1][:, np.newaxis]

        within_bounds = np.all((points_idx[:, np.newaxis, :] >= leaf_origin_lev1[start_idx:end_idx+1][np.newaxis, :, :]) &
                               (points_idx[:, np.newaxis, :] <= leaf_end_lev1[np.newaxis, :, :]), axis=-1)
        
        leaf_indices[block_lev1_idx] = np.argmax(within_bounds, axis=-1) + start_idx

# def find_nearest_cells(np.ndarray[np.float64_t, ndim=2] points,
#                        np.ndarray[np.int32_t, ndim=1] leaf_indices,
#                        np.ndarray[np.float64_t, ndim=2] leaf_origin_lev1,
#                        np.ndarray[np.float64_t, ndim=2] leaf_dxlevel_lev1,
#                        np.ndarray[np.int32_t, ndim=2] nearest_cells): 

#     nearest_cells[:] = np.floor((points - leaf_origin_lev1[leaf_indices]) / leaf_dxlevel_lev1[leaf_indices]).astype(np.int32)
#     nearest_cells[nearest_cells == ] = 0

def find_cell_from_another_block(np.int32_t level_diff, 
                                 np.ndarray[np.float64_t, ndim=1] ddblevel,
                                 np.ndarray[np.int32_t, ndim=1] cell, 
                                 np.ndarray[np.int32_t, ndim=1] loc, 
                                 np.ndarray[np.int32_t, ndim=1] block_nx):

    cdef np.ndarray[np.float64_t, ndim=1] cell_new = (cell.copy()).astype(float)

    # note the cell varies from 0 to 9, 1-8 are the indices of internal cells
    if level_diff == 0:
        cell_new[loc == 1] = 1
        cell_new[loc == -1] = block_nx[loc == -1]

    # finer to coaser situation, find the nearest cell
    elif level_diff == -1:
        if any(loc == 0):
            cell_new[loc == 0] = np.ceil(cell_new[loc==0]/2) + np.where(ddblevel[loc==0], block_nx[loc==0]/2, 0)
        cell_new[loc == 1] = 1
        cell_new[loc == -1] = block_nx[loc == -1]
    
    # coaser to finer situation, trilinear interpolation from the nearest 8 cells
    else:
        if any(loc == 0):
            cell_new[loc == 0] = (2*(cell_new[loc == 0] - np.where(ddblevel[loc == 0], block_nx[loc==0]/2, 0))-1) + 0.5

        cell_new[loc == 1] = 1.5
        cell_new[loc == -1] = block_nx[loc == -1] - 0.5
    return cell_new

def find_surrounding_cells(np.ndarray[np.float64_t, ndim=2] points,
                           np.ndarray[np.int32_t, ndim=1] leaf_indices,
                           np.ndarray[np.int32_t, ndim=2] nearest_cells,
                           np.ndarray[np.int32_t, ndim=1] block_nx,
                           np.ndarray[np.float64_t, ndim=2] leaf_origin_lev1,
                           np.ndarray[np.float64_t, ndim=1] leaf_dblevel_lev1,
                           dict lookup_lev1, 
                           np.ndarray[np.int32_t, ndim=1] lev1_idx_tree,
                           np.ndarray[np.float64_t, ndim=3] surrounding_cells): 

    cdef np.ndarray[np.float64_t, ndim=1] point, cell_lev1, dblock
    cdef np.ndarray[np.int32_t, ndim=1] cell_lev1_idx
    cdef np.int64_t leaf_idx, leaf_idx_new
    cdef np.int32_t x0, y0, z0, level_diff
    cdef np.ndarray[np.int32_t, ndim=2] cells
    cdef np.int64_t start_idx, end_idx, count
    cdef np.int64_t block_index_lev1_new

    cdef np.ndarray[np.npy_bool, ndim=1] mask1, mask2
    cdef np.ndarray[np.int32_t, ndim=1] loc
    cdef np.ndarray[np.float64_t, ndim=1] cell_new

    for i, point in enumerate(points):

        leaf_idx = leaf_indices[i]
        x0, y0, z0 = nearest_cells[i]

        cells = np.array([[i,j,k] for i in range(x0, x0+2) for j in range(y0, y0+2) for k in range(z0, z0+2)]).astype(np.int32)

        count = 0
        for cell in cells:

            assert(np.all(cell >= 0) and np.all(cell <= block_nx+1)), f"Cell out of bounds: {cell}"

            # loc to indicate in which side the cell beyond the block; -1 means minus, 0 means inside; 1 means plus
            loc = np.zeros(3).astype(np.int32)
            # mask1 to show plus situation; mask2 to show minus situation; loc here only serves as np.zeros(3)
            mask1 = (cell > block_nx)
            mask2 = (cell == loc)

            # cell in lev1 length unit
            cell_lev1 = leaf_origin_lev1[leaf_idx] + (cell-0.5) * leaf_dblevel_lev1[leaf_idx] / block_nx

            # if outside, just use the nearest cell: at least one cell inside the original leaf block
            if any(cell_lev1 < 0) or any(cell_lev1 > max(lookup_lev1.keys())):
                leaf_idx_new = leaf_idx
                for cell1 in cells:
                    if any(cell1 <= 0) or any(cell1 > block_nx):
                        continue
                    cell_new = cell1.copy().astype(float)
            
            elif any(mask1) or any(mask2):

                # project mask to loc
                loc = np.where(mask1, 1, loc)
                loc = np.where(mask2, -1, loc)

                # find cell lev1 block index in lev1 list
                cell_lev1_idx = np.floor(cell_lev1).astype(np.int32)
                # modify the max edge situation
                block_index_lev1_new = lookup_lev1[tuple(cell_lev1_idx)]

                # find the leaf index of cell
                start_idx = lev1_idx_tree[block_index_lev1_new]
                end_idx = lev1_idx_tree[block_index_lev1_new + 1]-1 \
                    if (block_index_lev1_new < len(lev1_idx_tree)-1) else len(leaf_origin_lev1)-1

                if start_idx == end_idx:
                    leaf_idx_new = start_idx
                else:
                    leaf_idx_new = find_leaf_idx(cell_lev1, leaf_origin_lev1[start_idx:end_idx+1], 
                        leaf_origin_lev1[start_idx:end_idx+1]+leaf_dblevel_lev1[start_idx:end_idx+1][:,np.newaxis]) \
                        + start_idx

                # assert(np.all(leaf_origin_lev1[leaf_idx_new] <= cell_lev1) and \
                #        np.all(cell_lev1 <= leaf_origin_lev1[leaf_idx_new]+leaf_dblevel_lev1[leaf_idx_new])), \
                #        f"Cell out of bounds: {cell_lev1}, {cell_lev1_idx}, {leaf_origin_lev1[leaf_idx_new]}, {leaf_dblevel_lev1[leaf_idx_new]}"

                
                # eval the level_diff situation
                if leaf_dblevel_lev1[leaf_idx_new] < leaf_dblevel_lev1[leaf_idx]:
                    level_diff = 1
                elif leaf_dblevel_lev1[leaf_idx_new] > leaf_dblevel_lev1[leaf_idx]:
                    level_diff = -1
                else:
                    level_diff = 0
                dblock = leaf_origin_lev1[leaf_idx_new] - leaf_origin_lev1[leaf_idx]
                # find the new cell 
                cell_new = find_cell_from_another_block(level_diff, dblock, cell, loc, block_nx)
                
            else:
                 leaf_idx_new = leaf_idx
                 cell_new = cell.astype(float)
            if count == 0:
                surrounding_cells[i,-1,:3] = (point-cell_lev1) / leaf_dblevel_lev1[leaf_idx] * block_nx
                
            surrounding_cells[i, count, :3] = cell_new
            surrounding_cells[i, count, 3] = leaf_idx_new

            count += 1
        
        assert(count == 8), f"Count is not 8: {count}"

def fill_coordinates_field_nearest(np.ndarray[np.int32_t, ndim=2] cells,
                           np.ndarray[np.int32_t, ndim=1] leaf_indices,
                           np.ndarray[np.float64_t, ndim=4] raw_field_data,
                           np.ndarray[np.float64_t, ndim=1] field_data):
    
    cdef np.int32_t i, leaf_idx
    cdef np.ndarray[np.int32_t, ndim=1] cell
    
    for i, cell in enumerate(cells):
        leaf_idx = leaf_indices[i]
        field_data[i] = raw_field_data[leaf_idx][cell[0], cell[1], cell[2]]

def fill_coordinates_field_linear(np.ndarray[np.float64_t, ndim=3] surrounding_cells,
                                  np.ndarray[np.float64_t, ndim=4] raw_field_data,
                                  np.ndarray[np.float64_t, ndim=1] field_data):
    
    cdef np.int32_t i, j 
    cdef np.float64_t f1, f2, f3

    cdef np.ndarray[np.float64_t, ndim=2] cells
    cdef np.ndarray[np.float64_t, ndim=1] factors, values, cell

    cdef np.ndarray[np.int32_t, ndim=2] cells_around
    cdef np.ndarray[np.int32_t, ndim=1] cell_floor, leaf_idx

    for i, coords in enumerate(surrounding_cells):
        
        # read 1. 8 cells 2. 8 leaf indices 3. linear interpolation factor
        cells = coords[:8,:3]
        leaf_idx = coords[:8,3].astype(np.int32)
        f1,f2,f3 = coords[-1][:3]
        factors = np.array([(1-f1)*(1-f2)*(1-f3), (1-f1)*(1-f2)*f3, (1-f1)*f2*(1-f3), (1-f1)*f2*f3,
                    f1*(1-f2)*(1-f3), f1*(1-f2)*f3, f1*f2*(1-f3), f1*f2*f3])

        # read 8 values from 8 cells
        values = np.zeros(8)
        for j, cell in enumerate(cells):

            all_integers = np.all(np.equal(np.mod(cell, 1), 0))
            all_half = np.all(np.equal(np.mod(cell, 1), 0.5))

            assert all_integers or all_half, f"Array elements: {cell} are not integers or half integers"

            cell_floor = np.floor(cell).astype(np.int32)-1
        
            if all_integers:
                values[j] = raw_field_data[leaf_idx[j]][cell_floor[0], cell_floor[1], cell_floor[2]]
            else:
                cells_around = np.array([[i,j,k] for i in range(cell_floor[0],cell_floor[0]+2) \
                    for j in range(cell_floor[1],cell_floor[1]+2) for k in range(cell_floor[2],cell_floor[2]+2)]).astype(np.int32)

                assert(cells_around.shape[0] == 8), "Surrounding cells are not 8"

                values[j] = np.sum(raw_field_data[leaf_idx[j]][tuple(cells_around.T)])/8
        
        # do interpolation
        field_data[i] = np.sum(factors * values)
                