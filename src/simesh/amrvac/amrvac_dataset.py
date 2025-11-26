import numpy as np
from datio import get_metadata, read_blocks_sequential
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

        header, is_leaf, _ = get_metadata(self.sfile)
        self.metadata = header.copy()
        self.is_leaf = is_leaf.copy().astype(np.int32)

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
        """

        if self.data is None:
            data = read_blocks_sequential(self.sfile, field_indices)
            self.data = data
            self.field_indices = list(range(self.nw)) if field_indices is None else field_indices
        else:
            indices_to_add = []
            for i in field_indices:
                if i not in self.field_indices:
                    indices_to_add.append(i)
            if len(indices_to_add) > 0:
                new_data = read_blocks_sequential(self.sfile, indices_to_add)
                # Concatenate along the last dimension (field indices)
                # old_data: (block_num, nx1, nx2, nx3, fidx1)
                # new_data: (block_num, nx1, nx2, nx3, fidx2)
                # result:   (block_num, nx1, nx2, nx3, fidx1+fidx2)
                self.data = np.concatenate((self.data, new_data), axis=-1)
                self.field_indices.extend(indices_to_add)

    def __getitem__(self, key):
        """
        Get the data from the dataset with uniform grid 
        """
        return

    def uniform_grid(self, xmin, xmax, nx, field_indices:list[int] = None):
        """
        Get the uniform grid data from the 1d amr managed data (zero order interpolation)
        """
        if self.data is None:
            self.load_data(field_indices)
        if field_indices is None:
            field_indices = list(range(self.nw))
        uniform_grid = np.zeros((nx[0], nx[1], nx[2], len(field_indices)), dtype=np.double)
        self.mesh.uniform_grid_zero_order(self.data[:,:,:,:,field_indices], uniform_grid, np.array(nx, dtype=np.uint32), 
                np.array(xmin, dtype=np.double), np.array(xmax, dtype=np.double))
        return uniform_grid
    


