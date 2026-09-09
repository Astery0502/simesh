import numpy as np
from functools import cached_property
from typing import Iterable, Union

from .datfile_io import *
from .amr_selection import *

class AMRMesh:

    """
    AMR Mesh information container, read from MPI-AMRVAC generated .dat file
    """

    def __init__(self, datfile:str):

        self.source_file = datfile

        with open(datfile, 'rb') as f:
            header = get_header(f)

            self.ndim = header['ndim'].astype(np.int32)
            self.domain_nx = header['domain_nx']
            self.block_nx = header['block_nx'].astype(np.int32)
            self.xmin = header['xmin']
            self.xmax = header['xmax']
            numblock = (header['domain_nx'] / header['block_nx']).astype(np.int32)
            assert (all(numblock % 1 == 0)), "Number of blocks must be integer"
            self.nblock_nx = numblock.astype(np.int32)
            self.wnames = header['w_names']

            # forest and tree information
            self.forest = get_forest(f)
            tree = get_tree_info(f)
            self.leaf_levels = tree[0]
            self.leaf_indices = tree[1]
            self.block_offsets = tree[2]
            # level 1 block position in leaf about array
            self.lev1_idx_tree = np.asarray(read_lev1_indices_from_tree(header, self.forest), 
                                            dtype=np.int32)

            # block origin and block size in lev1 block length unit
            self.block_origin_lev1 = np.array((self.leaf_indices-1.0)/np.power(2, self.leaf_levels[:, np.newaxis]-1))
            self.dblevel = 1.0 / np.power(2, self.leaf_levels-1)
            nglev1 = nglev1_morton([0,self.nblock_nx[0]], 
                                   [0,self.nblock_nx[1]], 
                                   [0,self.nblock_nx[2]])
            # quick lookup table for lev1 block index 
            self.lookup_lev1 = {tuple(nglev1[i]): i for i in range(nglev1.shape[0])}

            assert(len(self.lev1_idx_tree) == nglev1.shape[0]), "Number of lev1 blocks mismatch"

            self.raw_field_data = {}

    def _determine_field_name(self, field:str):
        if field in self.wnames:
            return field
        else:
            raise ValueError(f"field {field} is not in the field list: \n {self.field_names}")

    def load_raw_field_data(self, field:str):
        """
        Load raw field data from .dat file
        """
        field = self._determine_field_name(field)
        field_idx = self.wnames.index(field)

        self.raw_field_data[field] = np.zeros((self.block_offsets.shape[0], *self.block_nx))
        with open(self.source_file, 'rb') as f:
            for i, offset in enumerate(self.block_offsets):
                data = get_single_block_field_data(f, offset, self.block_nx, self.ndim, field_idx)
                self.raw_field_data[field][i] = data

    def point2lev1(self, point:Iterable[float]):
        """
        Convert point to lev1 block index
        """
        return (point-self.xmin) / (self.xmax-self.xmin) * self.nblock_nx 

class UniformMesh:

    """
    Resampled uniform mesh from AMR mesh
    """

    def __init__(self, am:AMRMesh, region:Iterable, nx:Iterable):
        """
        Always resample from the AMR Mesh,

        am: AMRMesh Object
        region: [xmin, xmax, ymin, ymax, zmin, zmax]
        nx: [nx, ny, nz]
        """
        self.am = am
        self.xmin = np.asarray(region[::2])
        self.xmax = np.asarray(region[1::2])
        self.nx = np.asarray(nx).astype(np.int32)

        assert(len(self.xmin) == self.am.ndim == len(self.nx)), "Region dimension mismatch"

        # uniform coordinates in lev1 block length unit
        coordinates = self.uniform_coordinates(self.xmin, self.xmax, self.nx)
        self.coordinates = (coordinates-self.am.xmin) / (self.am.xmax-self.am.xmin) * self.am.nblock_nx
        self.field_data = {}

    def __getitem__(self, field:str):

        field = self.am._determine_field_name(field)

        if field not in self.am.raw_field_data:
            self.am.load_raw_field_data(field)

        if field not in self.field_data:
            self.load_field_data(field, 'linear')
        
        return self.field_data[field]

    @staticmethod
    def uniform_coordinates(xmin:Iterable[float], xmax:Iterable[float], nx:Iterable[int]):
        """
        Generate uniform coordinates (cell center)
        """
        x, y, z = [np.linspace(xmin[i], xmax[i], nx[i]+1) for i in range(len(xmin))]
        x, y, z = [0.5*(x[:-1]+x[1:]) for x in [x, y, z]]
        mesh = np.meshgrid(x, y, z, indexing='ij')
        coordinates = np.stack(mesh, axis=-1).reshape(-1, 3)

        return coordinates

    @cached_property
    def leaf_indices(self):
        """
        Find the leaf index for each uniform cell
        """
        leaf_indices = np.zeros(self.coordinates.shape[0], dtype=np.int32)

        block_lev1_idx = np.array([self.am.lookup_lev1[tuple(i)] for i in np.floor(self.coordinates).astype(int)]).astype(np.int32)

        find_leaf_indices(self.coordinates, 
                          block_lev1_idx,
                          self.am.lev1_idx_tree,
                          self.am.block_origin_lev1,
                          self.am.dblevel,
                          leaf_indices
                          )
        return leaf_indices
        # return  (self.coordinates, 
        #         block_lev1_idx,
        #         np.asarray(self.am.lev1_idx_tree),
        #         self.am.block_origin_lev1,
        #         self.am.dblevel,
        #         leaf_indices)
                          
    @cached_property
    def coordinates_cell_idx(self):
        return (self.coordinates-self.am.block_origin_lev1[self.leaf_indices]) / \
                        self.am.dblevel[self.leaf_indices][:,np.newaxis] * self.am.block_nx

    @cached_property
    def nearest_cells(self):

        nearest_cells = np.zeros(self.coordinates.shape)

        nearest_cells[:] = np.floor(self.coordinates_cell_idx)

        nearest_cells[nearest_cells == self.am.block_nx] -= 1
        nearest_cells = nearest_cells.astype(np.int32)
        
        return nearest_cells

    @cached_property
    def surrounding_cells(self):

        surrounding_cells = np.zeros((self.coordinates.shape[0], 9, 4), dtype=float)

        # return (self.coordinates, self.leaf_indices, np.floor(self.coordinates_cell_idx+0.5).astype(int), 
        #                            self.am.block_nx, self.am.block_origin_lev1, self.am.dblevel,
        #                            self.am.lookup_lev1, np.asarray(self.am.lev1_idx_tree), surrounding_cells)

        find_surrounding_cells(self.coordinates, self.leaf_indices, np.floor(self.coordinates_cell_idx+0.5).astype(np.int32), 
                                   self.am.block_nx, self.am.block_origin_lev1, self.am.dblevel,
                                   self.am.lookup_lev1, self.am.lev1_idx_tree, surrounding_cells)

        return surrounding_cells

    def check_surrounding_cells(self):

        for i, coord in enumerate(self.coordinates):

            cells = self.surrounding_cells[i][:8, :3]
            leaf_indices = self.surrounding_cells[i][:8, 3].astype(np.int32)
            factors = self.surrounding_cells[i][8, :3]


            for cell, leaf_idx in zip(cells, leaf_indices):
                if any(coord < self.am.dblevel[leaf_idx]/2):
                    pass
                
                cell_lev1 = self.am.block_origin_lev1[leaf_idx] + (cell-0.5) * self.am.dblevel[leaf_idx] / self.am.block_nx

                dcell = (coord - cell_lev1) / self.am.dblevel[leaf_idx] * self.am.block_nx
                for i in range(3):
                    assert(abs(dcell[i]) == factors[i] or abs(dcell[i]) == 1-factors[i]), \
                        f"Cell {cell} in leaf {leaf_idx} is not correct: {cell_lev1}, with point {coord}, and dcell {dcell} but factors {factors}"


    def load_field_data(self, field:str, interpolation:str='nearest'):
        """
        Load field data from AMR Mesh
        """
        field = self.am._determine_field_name(field)

        if field in self.field_data:
            return

        if interpolation == 'nearest':
            self.field_data[field] = np.zeros(self.coordinates.shape[0])

            fill_coordinates_field_nearest(self.nearest_cells, self.leaf_indices, self.am.raw_field_data[field], self.field_data[field])
            

        elif interpolation == 'linear':
            self.field_data[field] = np.zeros(self.coordinates.shape[0])

            fill_coordinates_field_linear(self.surrounding_cells, self.am.raw_field_data[field], self.field_data[field])

        else:
            raise ValueError(f"Interpolation method {interpolation} is not supported")