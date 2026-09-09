from dataclasses import dataclass

import numpy as np
from .boundary import normalize_boundary_conditions
from .datio import get_metadata, read_blocks_sequential
from .datio import update_header, write_datfile_from_sfc
from .dataset_base import DataSet
from .derived_fields import (
    AMRVACDerivedFieldsMixin,
    FIELD_SOURCE_DERIVED,
    FIELD_SOURCE_ORIGINAL,
)
from .layouts import datau_to_udata
from simesh.utils.lib.amr.forest import AMRForest
from simesh.utils.lib.amr.mesh import AMRMesh


@dataclass(frozen=True)
class LoadedFieldColumn:
    name: str
    source_kind: str
    original_index: int | None
    ghost_valid_layers: int


class AMRVACDataSet(AMRVACDerivedFieldsMixin, DataSet):
    """
    AMRVAC specific implementation of DataSet.
    """
    def __init__(self, sfile: str, ghost_width: int = 0, boundary_conditions=None):
        self.ghost_width = int(ghost_width)
        if self.ghost_width < 0:
            raise ValueError("ghost_width must be non-negative")
        if self.ghost_width == 1:
            raise ValueError("ghost_width must be 0 or >= 2")
        self.boundary_conditions = boundary_conditions
        super().__init__(sfile)

    def _build_mesh(
        self,
        nfields: int | None = None,
        field_indices: list[int] | None = None,
        field_names: list[str] | None = None,
        boundary_table=None,
        normal_velocity_fields=None,
    ):
        if nfields is None:
            nfields = int(self.nw)
        if boundary_table is None or normal_velocity_fields is None:
            if field_names is not None:
                field_names = list(field_names)
            elif field_indices is None:
                field_names = list(self.wnames[:nfields])
            else:
                field_names = [self.wnames[i] for i in field_indices]
            boundary_table, normal_velocity_fields = normalize_boundary_conditions(
                self.boundary_conditions,
                field_names,
                int(self.ndim),
            )

        return AMRMesh(
            self.ndim,
            self.block_nx,
            self.domain_nx,
            np.array(self.physical_domain[0], dtype=np.double),
            np.array(self.physical_domain[1], dtype=np.double),
            np.uint32(self.ghost_width),
            np.uint32(nfields),
            self.forest,
            boundary_table,
            normal_velocity_fields,
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
        self._init_derived_fields()
        self._set_field_columns(self._original_field_columns(range(int(self.nw))))

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
        ng3 = np.uint32(1)
        if int(self.ndim) == 3:
            ng3 = np.uint32(self.domain_nx[2]//self.block_nx[2])
        self.forest = AMRForest(self.ndim, 
                                np.uint32(self.domain_nx[0]//self.block_nx[0]), 
                                np.uint32(self.domain_nx[1]//self.block_nx[1]), 
                                ng3, 
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

    def load_data(self, field_indices: list[int] = None, boundary_conditions=None):
        """
        Load the amr 1d managed block data from the data file
        Can reload the data if the field_indices are different
        """
        if field_indices is None:
            field_indices = list(range(int(self.nw)))
        else:
            field_indices = [int(i) for i in field_indices]

        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions

        field_names = [self.wnames[i] for i in field_indices]
        boundary_table, normal_velocity_fields = normalize_boundary_conditions(
            self.boundary_conditions,
            field_names,
            int(self.ndim),
        )
        self.boundary_condition_table = boundary_table
        self.normal_velocity_fields = normal_velocity_fields

        data = read_blocks_sequential(self.sfile, field_indices)
        if int(self.ndim) == 2:
            data = data[:, :, :, :, np.newaxis]
        if self.ghost_width > 0:
            self.mesh = self._build_mesh(
                len(field_indices),
                field_indices,
                boundary_table=boundary_table,
                normal_velocity_fields=normal_velocity_fields,
            )
            self.mesh.load_interior_data(np.ascontiguousarray(data, dtype=np.double))
            self.mesh.apply_ghost_cells()
            self.data = self.mesh.interior_view()
        else:
            self.data = data
        self._set_field_columns(self._original_field_columns(field_indices))

    def update_ghost_cells(self):
        if self.ghost_width <= 0:
            return
        if self.data is None or self.mesh is None:
            self.load_data(None)
            return
        if all(column.source_kind == FIELD_SOURCE_ORIGINAL for column in self._field_columns):
            self.mesh.apply_ghost_cells()
        else:
            self._exchange_original_ghost_cells()
        self.data = self.mesh.interior_view()

    def _expanded_boundary_storage(self, field_names, previous_field_names=None):
        field_names = list(field_names)
        boundary_table, normal_velocity_fields = normalize_boundary_conditions(
            None,
            field_names,
            int(self.ndim),
        )
        existing_table = getattr(self, "boundary_condition_table", None)
        if existing_table is not None and previous_field_names is not None:
            existing_table = np.asarray(existing_table, dtype=np.int32)
            previous_field_map = {
                field_name: column
                for column, field_name in enumerate(previous_field_names)
            }
            for column, field_name in enumerate(field_names):
                previous_column = previous_field_map.get(field_name)
                if previous_column is not None and previous_column < int(existing_table.shape[0]):
                    boundary_table[column, :] = existing_table[previous_column, :]

            boundary_table, normal_velocity_fields = normalize_boundary_conditions(
                boundary_table,
                field_names,
                int(self.ndim),
            )
        return boundary_table, normal_velocity_fields

    def _exchange_original_ghost_cells(self):
        original_positions = [
            (column_index, column)
            for column_index, column in enumerate(self._field_columns)
            if column.source_kind == FIELD_SOURCE_ORIGINAL
        ]
        if not original_positions:
            return

        padded = self.mesh.padded_view()
        derived_positions = [
            column_index
            for column_index, column in enumerate(self._field_columns)
            if column.source_kind == FIELD_SOURCE_DERIVED
        ]
        derived_padded = padded[..., derived_positions].copy() if derived_positions else None
        self.mesh.apply_ghost_cells()
        if derived_positions:
            padded[..., derived_positions] = derived_padded

    def _refresh_mesh_after_field_axis_change(self, previous_field_names=None):
        if self.ghost_width <= 0:
            return

        field_names = self.loaded_field_names
        if self.mesh is not None and self.mesh.padded_view() is not None:
            previous_padded = self.mesh.padded_view().copy()
            if previous_field_names is None:
                previous_field_names = list(field_names[:previous_padded.shape[-1]])
        else:
            previous_padded = None
            previous_field_names = [] if previous_field_names is None else list(previous_field_names)

        boundary_table, normal_velocity_fields = self._expanded_boundary_storage(
            field_names,
            previous_field_names=previous_field_names,
        )

        self.boundary_condition_table = boundary_table
        self.normal_velocity_fields = normal_velocity_fields
        self.mesh = self._build_mesh(
            len(field_names),
            field_names=field_names,
            boundary_table=boundary_table,
            normal_velocity_fields=normal_velocity_fields,
        )
        self.mesh.load_interior_data(np.ascontiguousarray(self.data, dtype=np.double))
        if previous_padded is not None:
            previous_field_map = {
                field_name: column
                for column, field_name in enumerate(previous_field_names)
            }
            padded = self.mesh.padded_view()
            for column, field_name in enumerate(field_names):
                previous_column = previous_field_map.get(field_name)
                if previous_column is not None:
                    padded[..., column] = previous_padded[..., previous_column]
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

    def blocks(self, include_ghosts: bool = False, field_indices: list[int] = None, field_names: list[str] = None):
        """
        Return loaded block data in SFC layout.

        The default interior layout is ``(nleafs, nw, bx, by, bz)``. When
        ``include_ghosts`` is true and ``ghost_width >= 2``, the returned view
        has shape ``(nleafs, nw, bx + 2g, by + 2g, bz + 2g)``.
        """
        if self.data is None:
            self.load_data(None)

        if not include_ghosts or self.ghost_width <= 0:
            return self._selected_sfc_data(field_indices=field_indices, field_names=field_names)

        padded = self.mesh.padded_view()
        if field_indices is None and field_names is None:
            return np.transpose(padded, (0, 4, 1, 2, 3))
        columns = self._columns_for_field_selectors(
            field_indices=field_indices,
            field_names=field_names,
        )
        return np.transpose(padded[..., columns], (0, 4, 1, 2, 3))

    @property
    def loaded_field_indices(self):
        """
        Original file/header field indices currently loaded in self.data.
        """
        return [
            column.original_index
            for column in self._field_columns
            if column.source_kind == FIELD_SOURCE_ORIGINAL
        ]

    @property
    def loaded_field_names(self):
        """
        Field names corresponding to the loaded columns in self.data.
        """
        return [column.name for column in self._field_columns]

    @property
    def derived_field_names(self):
        return [
            column.name
            for column in self._field_columns
            if column.source_kind == FIELD_SOURCE_DERIVED
        ]

    @property
    def derived_field_ghost_valid_layers(self):
        return {
            column.name: column.ghost_valid_layers
            for column in self._field_columns
            if column.source_kind == FIELD_SOURCE_DERIVED
        }

    def _original_ghost_valid_layers(self):
        return self.ghost_width if self.ghost_width >= 2 else 0

    def _original_field_columns(self, field_indices):
        return [
            LoadedFieldColumn(
                name=str(self.wnames[int(original_index)]),
                source_kind=FIELD_SOURCE_ORIGINAL,
                original_index=int(original_index),
                ghost_valid_layers=self._original_ghost_valid_layers(),
            )
            for original_index in field_indices
        ]

    def _set_field_columns(self, columns):
        columns = list(columns)
        self._validate_field_columns(columns)
        self._field_columns = columns

    def _validate_field_columns(self, columns):
        names = [column.name for column in columns]
        if len(names) != len(set(names)):
            raise ValueError(f"Loaded column names must be unique, got {names}")

        for column in columns:
            if column.source_kind == FIELD_SOURCE_ORIGINAL:
                if not isinstance(column.original_index, int):
                    raise ValueError(
                        f"Original field column {column.name!r} requires an integer original_index."
                    )
            elif column.source_kind == FIELD_SOURCE_DERIVED:
                if column.original_index is not None:
                    raise ValueError(
                        f"Derived field column {column.name!r} must use original_index=None."
                    )
            else:
                raise ValueError(
                    f"Field column {column.name!r} has invalid source_kind {column.source_kind!r}."
                )
            if int(column.ghost_valid_layers) < 0:
                raise ValueError(
                    f"Field column {column.name!r} has negative ghost_valid_layers."
                )

        data = getattr(self, "data", None)
        if data is not None and int(data.shape[1]) != len(columns):
            raise ValueError(
                f"Field column metadata length {len(columns)} does not match "
                f"data field axis {int(data.shape[1])}."
            )

    def _append_derived_field_columns(self, names, ghost_valid_layers):
        columns = [
            *self._field_columns,
            *[
                LoadedFieldColumn(
                    name=str(name),
                    source_kind=FIELD_SOURCE_DERIVED,
                    original_index=None,
                    ghost_valid_layers=int(ghost_valid_layers),
                )
                for name in names
            ],
        ]
        self._set_field_columns(columns)

    def _drop_derived_field_columns(self, names):
        names_to_drop = set(str(name) for name in names)
        previous_field_names = list(self.loaded_field_names)
        keep_columns = [
            column_index
            for column_index, column in enumerate(self._field_columns)
            if column.source_kind != FIELD_SOURCE_DERIVED or column.name not in names_to_drop
        ]
        if self.data is not None:
            self.data = np.ascontiguousarray(self.data[:, keep_columns, :, :, :], dtype=np.double)
        self._set_field_columns([self._field_columns[column_index] for column_index in keep_columns])
        self._refresh_mesh_after_field_axis_change(previous_field_names=previous_field_names)

    def _column_for_loaded_field_name(self, name):
        field_map = self._loaded_field_name_map()
        try:
            return field_map[str(name)]
        except KeyError as exc:
            raise ValueError(
                f"Requested field names {[str(name)]} are not loaded. "
                f"Currently loaded field names: {self.loaded_field_names}"
            ) from exc

    def _field_column_for_name(self, name):
        return self._field_columns[self._column_for_loaded_field_name(name)]

    def _loaded_field_map(self):
        """
        Map original file/header field indices to column positions in self.data.
        """
        return {
            column.original_index: column_index
            for column_index, column in enumerate(self._field_columns)
            if column.source_kind == FIELD_SOURCE_ORIGINAL
        }

    def _loaded_field_name_map(self):
        """
        Map loaded field names to column positions in self.data.
        """
        return {
            column.name: column_index
            for column_index, column in enumerate(self._field_columns)
        }

    def _validate_field_selectors(self, field_indices=None, field_names=None):
        if field_indices is not None and field_names is not None:
            raise ValueError("field_indices and field_names cannot both be supplied.")

    def _columns_for_field_names(self, field_names):
        loaded_field_map = self._loaded_field_name_map()
        field_names = [str(name) for name in field_names]
        missing = [name for name in field_names if name not in loaded_field_map]
        if missing:
            raise ValueError(
                f"Requested field names {missing} are not loaded. "
                f"Currently loaded field names: {self.loaded_field_names}"
            )
        return [loaded_field_map[name] for name in field_names]

    def _columns_for_field_indices(self, field_indices):
        loaded_field_map = self._loaded_field_map()
        field_indices = [int(i) for i in field_indices]
        missing = [i for i in field_indices if i not in loaded_field_map]
        if missing:
            raise ValueError(
                f"Requested field indices {missing} are not loaded. "
                f"Currently loaded original field indices: {self.loaded_field_indices}"
            )
        return [loaded_field_map[i] for i in field_indices]

    def _columns_for_field_selectors(self, field_indices=None, field_names=None):
        self._validate_field_selectors(field_indices=field_indices, field_names=field_names)
        if field_names is not None:
            return self._columns_for_field_names(field_names)
        if field_indices is not None:
            return self._columns_for_field_indices(field_indices)
        return list(range(len(self.loaded_field_names)))

    def _selected_sfc_data(self, field_indices=None, field_names=None):
        columns = self._columns_for_field_selectors(
            field_indices=field_indices,
            field_names=field_names,
        )
        if field_indices is None and field_names is None:
            return self.data
        return self.data[:, columns, :, :, :]

    def _selected_sfc_data_and_count(self, field_indices=None, field_names=None):
        data = self._selected_sfc_data(
            field_indices=field_indices,
            field_names=field_names,
        )
        return data, data.shape[1]

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

    def uniform_grid(
        self,
        nx,
        xmin: list = None,
        xmax: list = None,
        field_indices: list[int] = None,
        field_names: list[str] = None,
        interpolation: str = "zero",
    ):
        """
        Get the uniform grid data from the 1d AMR-managed data.

        Returns data in compute-oriented ``datau`` layout with shape
        ``(n_fields, nx, ny, nz)``. If you want user-facing ``udata`` layout
        ``(nx, ny, nz, n_fields)``, convert it with
        ``simesh.amrvac.layouts.datau_to_udata``.
        
        Parameters:
        -----------
        field_indices : list[int], optional
            Original file/header field indices to extract, corresponding to self.wnames.
            If None, uses all currently loaded fields.
        field_names : list[str], optional
            Loaded field names to extract, including materialized derived fields.
        interpolation : {"zero", "linear"}, optional
            ``"zero"`` uses piecewise-constant sampling. ``"linear"`` uses
            trilinear interpolation from ghost-cell-padded mesh storage.
        """
        # Default to full domain if bounds are not specified
        if xmin is None:
            xmin = self.physical_domain[0]
        if xmax is None:
            xmax = self.physical_domain[1]

        nx = np.asarray(nx, dtype=np.uint32)
        if int(self.ndim) == 2:
            if nx.shape == (2,):
                nx = np.array([nx[0], nx[1], 1], dtype=np.uint32)
            elif nx.shape != (3,) or int(nx[2]) != 1:
                raise ValueError(f"2D resolution must have shape (nx, ny) or (nx, ny, 1), got {tuple(nx)}")
        elif nx.shape != (3,):
            raise ValueError(f"3D resolution must have three entries, got {tuple(nx)}")

        # Load all fields lazily if nothing is loaded yet.
        if self.data is None:
            self.load_data(None)

        loaded_columns = self._columns_for_field_selectors(
            field_indices=field_indices,
            field_names=field_names,
        )
        n_fields = len(loaded_columns)
        
        uniform_grid = np.zeros((n_fields, int(nx[0]), int(nx[1]), int(nx[2])), dtype=np.double)

        interpolation = interpolation.lower()
        if interpolation in ("zero", "nearest", "piecewise_constant"):
            data_to_use, _ = self._selected_sfc_data_and_count(
                field_indices=field_indices,
                field_names=field_names,
            )
            self.mesh.uniform_grid_zero_order(
                data_to_use,
                uniform_grid,
                nx,
                np.array(xmin, dtype=np.double),
                np.array(xmax, dtype=np.double),
            )
        elif interpolation in ("linear", "trilinear"):
            if self.ghost_width <= 0:
                raise ValueError("Linear interpolation requires opening the dataset with ghost_width >= 2.")
            if field_indices is None and field_names is None:
                field_positions = np.arange(n_fields, dtype=np.uint32)
            else:
                field_positions = np.array(loaded_columns, dtype=np.uint32)
            self.mesh.uniform_grid_linear(
                uniform_grid,
                nx,
                np.array(xmin, dtype=np.double),
                np.array(xmax, dtype=np.double),
                field_positions,
            )
        else:
            raise ValueError(f"Unknown interpolation mode: {interpolation}")
        return uniform_grid

    def uniform_full(self, field_indices: list[int] = None, field_names: list[str] = None):
        """
        Return the full-domain uniform grid in compute-oriented ``datau``
        layout with shape ``(n_fields, nx, ny, nz)`` for datasets without
        refinement.
        """
        if int(self.levmax) != 1:
            raise ValueError("uniform_full() is only available when levmax == 1.")

        if self.data is None:
            self.load_data(None)

        data_to_use, n_fields = self._selected_sfc_data_and_count(
            field_indices=field_indices,
            field_names=field_names,
        )

        if int(self.ndim) == 2:
            uniform_shape = (n_fields, int(self.domain_nx[0]), int(self.domain_nx[1]), 1)
        else:
            uniform_shape = (n_fields, int(self.domain_nx[0]), int(self.domain_nx[1]), int(self.domain_nx[2]))
        uniform_grid = np.zeros(uniform_shape, dtype=np.double)
        self.mesh.uniform_full_level1(data_to_use, uniform_grid)

        expected_shape = uniform_shape
        assert uniform_grid.shape == expected_shape, \
            f"uniform_full result shape mismatch: {uniform_grid.shape} != {expected_shape}"

        return uniform_grid

    def write_datfile(self, sfile: str, overwrite: bool = False, field_names: list[str] = None):

        if self.data is None:
            self.load_data(None)

        if field_names is None:
            original_columns = [
                (column_index, column)
                for column_index, column in enumerate(self._field_columns)
                if column.source_kind == FIELD_SOURCE_ORIGINAL
            ]
            output_columns = [column_index for column_index, _ in original_columns]
            output_names = [
                self.wnames[column.original_index]
                for _, column in original_columns
            ]
        else:
            field_names = [str(name) for name in field_names]
            output_columns = self._columns_for_field_names(field_names)
            output_names = field_names

        updated_header = update_header(
            self.metadata,
            nw=len(output_names),
            w_names=output_names,
        )
        data0 = np.asarray(self.data[:, output_columns, :, :, :])
        return write_datfile_from_sfc(sfile, data0, updated_header, self.is_leaf, self.tree_info, overwrite=overwrite)
