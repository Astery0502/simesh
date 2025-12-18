import os
import numpy as np
from functools import cached_property
from datio import get_metadata, read_blocks_sequential
from datio import update_header, write_header, write_forest_tree, write_blocks
from simesh.dataset.data_set import DataSet
from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh

class AMRVACDataSet(DataSet):
    """
    AMRVAC specific implementation of DataSet.
    """
    def __init__(self, sfile: str):
        super().__init__(sfile)

    def load_metadata(self):
        """
        Load the metadata from the data file.
        """

        self.ng = 0

        header, is_leaf, tree_info = get_metadata(self.sfile)
        self.metadata = header.copy()
        self.is_leaf = is_leaf.copy().astype(np.int32)
        self.tree_info = tree_info

        # Basic metadata
        self.ndim = np.uint32(header['ndim'])
        self.ndir = np.uint32(header['ndir'])
        self.nw = np.uint32(header['nw'])
        self.wnames = header['w_names']

        # AMR specific metadata
        self.nleafs = np.uint32(header['nleafs'])
        self.nparents = np.uint32(header['nparents'])
        self.levmax = np.uint32(header['levmax'])
        self.block_nx = header['block_nx'].astype(np.uint32)

        # Domain specific metadata
        self.domain_nx = header['domain_nx'].astype(np.uint32)
        self.physical_domain = np.array((header['xmin'], header['xmax']))
        self.periodic = header['periodic']
        self.geometry = header['geometry']

        # use nghostcells = 0
        self.forest = AMRForest(self.ndim, 
                                np.uint32(self.domain_nx[0]//self.block_nx[0]), 
                                np.uint32(self.domain_nx[1]//self.block_nx[1]), 
                                np.uint32(self.domain_nx[2]//self.block_nx[2]), 
                                self.is_leaf)
        self.mesh = AMRMesh(self.ndim, 
                            self.block_nx, self.domain_nx, 
                            np.array(self.physical_domain[0], dtype=np.double), 
                            np.array(self.physical_domain[1], dtype=np.double), 
                            np.uint32(0), self.nw, self.forest)

    def print_metadata_impl(self):
        """
        furtherly print the metadata
        """
        print("====specific metadata=====")
        print("AMRVAC specific metadata:")
        print("Number of leaves: {self.nleafs}")
        print("Size of each block: {self.block_nx}")
        print("Maximum level: {self.levmax}")

    def load_data(self, field_indices: list[int] = None):
        """
        Load the amr 1d managed block data from the data file
        Can reload the data if the field_indices are different
        """

        data = read_blocks_sequential(self.sfile, field_indices)
        self.data = data

        if field_indices is not None:
            self.field_indices = field_indices


    @cached_property
    def field_indices(self):
        """
        Cached property for field indices.
        If not explicitly set via load_data, defaults to all fields (0 to nw-1).
        """
        # Default to all fields if not declared/defined
        return list(range(self.nw))
    
    def __getitem__(self, key):
        """
        Get data from the dataset on a uniform grid using NumPy-like slicing.

        The indexing uses a convention similar to ``numpy.mgrid`` where a
        complex-valued step encodes the total number of points in that
        direction.

        Examples
        --------
        - ``ds[::100j, ::100j, ::100j]``:
            build a uniform grid over the *entire* domain with a resolution of
            100 × 100 × 100 and return the full grid.

        - ``ds[50:100:200j, 50:100:200j, 50:100:200j]``:
            build a uniform grid over the *entire* domain with a resolution of
            200 × 200 × 200 and then return the sub-box ``50:100`` in each
            direction.

        Notes
        -----
        - Only 3D slicing with complex steps is currently supported.
        - The internal uniform grid is computed with shape ``(n_fields, nx, ny, nz)``.
          This method returns a view transposed to ``(nx, ny, nz, n_fields)`` for
          user-facing convenience.
        """
        # Expect a 3D indexing key
        if not isinstance(key, tuple) or len(key) != 3:
            raise TypeError(
                "Indexing expects a tuple of three slices, e.g. "
                "[::100j, ::100j, ::100j] or [50:100:200j, 50:100:200j, 50:100:200j]"
            )

        slices = []
        nx = []

        for s in key:
            if not isinstance(s, slice) or not isinstance(s.step, complex):
                raise TypeError(
                    "Each index must be a slice with a complex step, e.g. ::100j or 50:100:200j"
                )

            # Total number of points along this axis
            n_axis = int(round(s.step.imag))
            if n_axis <= 0:
                raise ValueError("Complex step imag part must give a positive number of points")

            nx.append(n_axis)

            start = 0 if s.start is None else s.start
            stop = n_axis if s.stop is None else s.stop
            slices.append(slice(start, stop))

        # Build the full-domain uniform grid at the requested resolution.
        # This returns data with shape (n_fields, nx, ny, nz)
        full_grid = self.uniform_grid(nx)

        # Extract the requested sub-box
        return full_grid[slices[0], slices[1], slices[2], :]

    def uniform_grid(self, nx, xmin: list = None, xmax: list = None, field_indices: list[int] = None):
        """
        Get the uniform grid data from the 1d amr managed data (zero order interpolation).
        If xmin or xmax are not provided, they default to the full physical domain of the dataset.
        
        Parameters:
        -----------
        field_indices : list[int], optional
            Indices to select from the already-loaded fields. If None, uses all loaded fields.
            These indices are relative to self.field_indices (the fields that were loaded),
            not the original nw fields in the raw data.
        """
        # Default to full domain if bounds are not specified
        if xmin is None:
            xmin = self.physical_domain[0]
        if xmax is None:
            xmax = self.physical_domain[1]

        # Load data if not already loaded (load all fields to allow selection later)
        if self.data is None:
            self.load_data(None)  # Load all fields
        
        # Select which of the already-loaded fields to use
        # If field_indices is provided, use those to select from the loaded data
        # Otherwise, use all loaded fields
        if field_indices is not None:
            # field_indices are indices into the already-loaded data (self.data)
            # Select the corresponding columns from self.data
            data_to_use = self.data[:, field_indices, :, :, :]
            n_fields = len(field_indices)
        else:
            # Use all loaded fields
            data_to_use = self.data
            n_fields = len(self.field_indices)
        
        uniform_grid = np.zeros((n_fields, nx[0], nx[1], nx[2]), dtype=np.double)
        self.mesh.uniform_grid_zero_order(data_to_use, uniform_grid, np.array(nx, dtype=np.uint32), 
                np.array(xmin, dtype=np.double), np.array(xmax, dtype=np.double))
        return uniform_grid

    def write_datfile(self, sfile: str):

        if os.path.exists(sfile):
            raise FileExistsError(f"File {sfile} already exists")
        with open(sfile, 'wb') as fb:
            # update the header if not fully read original fields
            updated_header = update_header(self.metadata, nw=len(self.field_indices), 
                w_names=[self.wnames[i] for i in self.field_indices])
            print(updated_header)
            print(self.metadata)
            write_header(fb, updated_header)

            # required to rewrite the block offset tree (staggered grid not read here)
            write_forest_tree(fb, updated_header, self.is_leaf, self.tree_info)
            # the non-ghostcells data0, now default ng=0, no ghostcells
            data0 = self.data[:,:,self.ng:self.ng+self.block_nx[0],
                self.ng:self.ng+self.block_nx[1],
                self.ng:self.ng+self.block_nx[2]]
            write_blocks(fb, data0, updated_header['ndim'], self.tree_info[2])


