import numpy as np
from .datio import get_metadata, read_blocks_sequential
from .datio import update_header, write_datfile_from_sfc
from .dataset_base import DataSet
from .layouts import datau_to_udata
from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh

class AMRVACDataSet(DataSet):
    """
    AMRVAC specific implementation of DataSet.
    """
    def __init__(self, sfile: str, ghost_width: int = 0):
        self.ghost_width = int(ghost_width)
        if self.ghost_width < 0:
            raise ValueError("ghost_width must be non-negative")
        super().__init__(sfile)

    def _build_mesh(self, nfields: int | None = None):
        if nfields is None:
            nfields = int(self.nw)

        return AMRMesh(
            self.ndim,
            self.block_nx,
            self.domain_nx,
            np.array(self.physical_domain[0], dtype=np.double),
            np.array(self.physical_domain[1], dtype=np.double),
            np.uint32(self.ghost_width),
            np.uint32(nfields),
            self.forest,
        )

    def load_metadata(self):
        """
        Load the metadata from the data file.
        """

        self.ng = self.ghost_width

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
        self.mesh = self._build_mesh()

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
        if field_indices is None:
            field_indices = list(range(int(self.nw)))
        else:
            field_indices = [int(i) for i in field_indices]

        data = read_blocks_sequential(self.sfile, field_indices)
        if self.ghost_width > 0:
            self.mesh = self._build_mesh(len(field_indices))
            self.mesh.load_interior_data(np.ascontiguousarray(data, dtype=np.double))
            self.mesh.apply_ghost_cells()
            self.data = self.mesh.interior_view()
        else:
            self.data = data
        self.loaded_field_indices = field_indices

    def update_ghost_cells(self):
        if self.ghost_width <= 0:
            return
        if self.data is None:
            self.load_data(None)
            return
        self.mesh.apply_ghost_cells()
        self.data = self.mesh.interior_view()

    def exchange_ghost_cells(self):
        """
        Refresh ghost cells after mutating loaded interior block data.
        """
        self.update_ghost_cells()

    @property
    def has_ghost_cells(self):
        """
        Whether this dataset was opened with ghost-cell storage enabled.
        """
        return self.ghost_width > 0

    def blocks(self, include_ghosts: bool = False):
        """
        Return loaded block data in SFC layout.

        The default interior layout is ``(nleafs, nw, bx, by, bz)``. When
        ``include_ghosts`` is true and ``ghost_width > 0``, the returned view
        has shape ``(nleafs, nw, bx + 2g, by + 2g, bz + 2g)``.
        """
        if self.data is None:
            self.load_data(None)

        if not include_ghosts or self.ghost_width <= 0:
            return self.data

        padded = self.mesh.padded_view()
        return np.transpose(padded, (0, 4, 1, 2, 3))

    @property
    def loaded_field_indices(self):
        """
        Original file/header field indices currently loaded in self.data.
        """
        if not hasattr(self, "_loaded_field_indices"):
            self._loaded_field_indices = list(range(int(self.nw)))
        return self._loaded_field_indices

    @loaded_field_indices.setter
    def loaded_field_indices(self, value):
        self._loaded_field_indices = [int(i) for i in value]

    def _loaded_field_map(self):
        """
        Map original file/header field indices to column positions in self.data.
        """
        return {field_idx: i for i, field_idx in enumerate(self.loaded_field_indices)}
    
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
        - The internal compute layout is ``(n_fields, nx, ny, nz)``.
        - This method converts that to the user-facing ``(nx, ny, nz, n_fields)``
          layout before applying the requested slices.
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
        # This returns data with shape (nx, ny, nz, n_fields)
        full_grid = datau_to_udata(self.uniform_grid(nx))

        # Extract the requested sub-box
        return full_grid[slices[0], slices[1], slices[2], :]

    def uniform_grid(self, nx, xmin: list = None, xmax: list = None, field_indices: list[int] = None):
        """
        Get the uniform grid data from the 1d AMR-managed data by zero-order
        interpolation.

        Returns data in compute-oriented ``datau`` layout with shape
        ``(n_fields, nx, ny, nz)``. If you want user-facing ``udata`` layout
        ``(nx, ny, nz, n_fields)``, convert it with
        ``simesh.amrvac.layouts.datau_to_udata``.
        
        Parameters:
        -----------
        field_indices : list[int], optional
            Original file/header field indices to extract, corresponding to self.wnames.
            If None, uses all currently loaded fields.
        """
        # Default to full domain if bounds are not specified
        if xmin is None:
            xmin = self.physical_domain[0]
        if xmax is None:
            xmax = self.physical_domain[1]

        # Load all fields lazily if nothing is loaded yet.
        if self.data is None:
            self.load_data(None)

        loaded_field_map = self._loaded_field_map()

        if field_indices is not None:
            field_indices = [int(i) for i in field_indices]
            missing = [i for i in field_indices if i not in loaded_field_map]
            if missing:
                raise ValueError(
                    f"Requested field indices {missing} are not loaded. "
                    f"Currently loaded original field indices: {self.loaded_field_indices}"
                )

            loaded_columns = [loaded_field_map[i] for i in field_indices]
            data_to_use = self.data[:, loaded_columns, :, :, :]
            n_fields = len(loaded_columns)
        else:
            data_to_use = self.data
            n_fields = len(self.loaded_field_indices)
        
        uniform_grid = np.zeros((n_fields, nx[0], nx[1], nx[2]), dtype=np.double)
        self.mesh.uniform_grid_zero_order(data_to_use, uniform_grid, np.array(nx, dtype=np.uint32), 
                np.array(xmin, dtype=np.double), np.array(xmax, dtype=np.double))
        return uniform_grid

    def uniform_full(self, field_indices: list[int] = None):
        """
        Return the full-domain uniform grid in compute-oriented ``datau``
        layout with shape ``(n_fields, nx, ny, nz)`` for datasets without
        refinement.
        """
        if int(self.levmax) != 1:
            raise ValueError("uniform_full() is only available when levmax == 1.")

        if self.data is None:
            self.load_data(None)

        loaded_field_map = self._loaded_field_map()

        if field_indices is not None:
            field_indices = [int(i) for i in field_indices]
            missing = [i for i in field_indices if i not in loaded_field_map]
            if missing:
                raise ValueError(
                    f"Requested field indices {missing} are not loaded. "
                    f"Currently loaded original field indices: {self.loaded_field_indices}"
                )

            loaded_columns = [loaded_field_map[i] for i in field_indices]
            data_to_use = self.data[:, loaded_columns, :, :, :]
            n_fields = len(loaded_columns)
        else:
            data_to_use = self.data
            n_fields = len(self.loaded_field_indices)

        uniform_grid = np.zeros((n_fields, *self.domain_nx), dtype=np.double)
        self.mesh.uniform_full_level1(data_to_use, uniform_grid)

        expected_shape = (n_fields, int(self.domain_nx[0]), int(self.domain_nx[1]), int(self.domain_nx[2]))
        assert uniform_grid.shape == expected_shape, \
            f"uniform_full result shape mismatch: {uniform_grid.shape} != {expected_shape}"

        return uniform_grid

    def write_datfile(self, sfile: str, overwrite: bool = False):

        if self.data is None:
            self.load_data(None)

        updated_header = update_header(
            self.metadata,
            nw=len(self.loaded_field_indices),
            w_names=[self.wnames[i] for i in self.loaded_field_indices],
        )
        data0 = np.asarray(self.data)
        return write_datfile_from_sfc(sfile, data0, updated_header, self.is_leaf, self.tree_info, overwrite=overwrite)
