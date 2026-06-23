from dataclasses import dataclass

import numpy as np


FIELD_SOURCE_ORIGINAL = "original"
FIELD_SOURCE_DERIVED = "derived"


@dataclass(frozen=True)
class DerivedFieldDefinition:
    func: object
    dependencies: tuple[str, ...]
    requires_ghosts: bool = False


@dataclass(frozen=True)
class DerivativeTerm:
    field_name: str
    axis: int
    coefficient: float


@dataclass(frozen=True)
class DerivativeFieldDefinition:
    terms: tuple[DerivativeTerm, ...]
    dependencies: tuple[str, ...]
    requires_ghosts: bool = True


class DerivedFieldContext:
    def __init__(self, dataset):
        self._dataset = dataset

    @property
    def spacing(self):
        if self._dataset.mesh is None:
            return None
        ndim = int(self._dataset.ndim)
        return np.asarray(self._dataset.mesh.rnode)[:, 2 * ndim:3 * ndim]

    def field(self, name):
        columns = self._dataset._columns_for_field_names([name])
        return self._dataset.data[:, columns[0], :, :, :]

    def padded_field(self, name):
        if self._dataset.ghost_width <= 0:
            raise ValueError(f"Padded field {name!r} requires ghost cells.")
        if self._dataset.mesh is None:
            raise ValueError(f"Padded field {name!r} requires an initialized mesh.")
        if not self._dataset.mesh.has_padded_data():
            raise ValueError(f"Padded field {name!r} requires padded ghost-cell storage.")
        column = self._dataset._field_column_for_name(name)
        if column.source_kind != FIELD_SOURCE_ORIGINAL:
            raise ValueError(
                f"Padded field {name!r} is a materialized derived field, but only "
                "original loaded fields currently have a ghost-cell exchange contract."
            )
        if column.ghost_valid_layers < 1:
            raise ValueError(f"Padded field {name!r} requires at least one valid ghost layer.")

        columns = self._dataset._columns_for_field_names([name])
        padded = self._dataset.mesh.padded_view()
        if padded is None:
            raise ValueError(f"Padded field {name!r} requires padded ghost-cell storage.")
        return padded[..., columns[0]]


class AMRVACDerivedFieldsMixin:
    def _init_derived_fields(self):
        self.derived_definitions = {}
        self._clear_materialized_derived_fields()

    def _clear_materialized_derived_fields(self):
        if not hasattr(self, "_field_columns"):
            self._field_columns = []
            return
        self._set_field_columns([
            column
            for column in self._field_columns
            if column.source_kind == FIELD_SOURCE_ORIGINAL
        ])

    def _drop_materialized_derived_field(self, name):
        if name not in self.derived_field_names:
            return
        self._drop_derived_field_columns([name])

    def drop_derived_fields(self, names):
        """
        Remove materialized derived fields from the loaded data columns.
        """
        if isinstance(names, str):
            names = [names]
        names = list(names)
        for name in names:
            if name not in self.derived_field_names:
                raise KeyError(f"Derived field {name!r} is not materialized.")

        self._drop_derived_field_columns(names)

    def _derived_context(self):
        return DerivedFieldContext(self)

    def _normalize_derivative_axis(self, name, axis):
        axis_names = {"x": 0, "y": 1, "z": 2}
        if isinstance(axis, str):
            normalized = axis_names.get(axis.lower())
        else:
            try:
                normalized = int(axis)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Derivative field {name!r} has invalid axis {axis!r}."
                ) from exc
        if normalized is None or normalized < 0 or normalized >= int(self.ndim):
            raise ValueError(
                f"Derivative field {name!r} has invalid axis {axis!r} "
                f"for ndim={int(self.ndim)}."
            )
        return normalized

    def _normalize_derivative_terms(self, name, terms):
        if terms is None or isinstance(terms, (str, bytes)):
            raise ValueError(f"Derivative field {name!r} terms must be a non-empty sequence.")
        try:
            raw_terms = list(terms)
        except TypeError as exc:
            raise ValueError(f"Derivative field {name!r} terms must be a non-empty sequence.") from exc
        if not raw_terms:
            raise ValueError(f"Derivative field {name!r} terms must be a non-empty sequence.")

        normalized_terms = []
        dependencies = []
        for term in raw_terms:
            try:
                field_name, axis, coefficient = term
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Derivative field {name!r} terms must have "
                    "(field_name, axis, coefficient)."
                ) from exc

            if not isinstance(field_name, str) or not field_name:
                raise ValueError(
                    f"Derivative field {name!r} terms must use non-empty field names, "
                    f"got {field_name!r}."
                )
            axis_index = self._normalize_derivative_axis(name, axis)
            try:
                coefficient_value = float(coefficient)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Derivative field {name!r} coefficient for {field_name!r} "
                    f"along axis {axis!r} cannot be converted to float64."
                ) from exc

            normalized_terms.append(DerivativeTerm(field_name, axis_index, coefficient_value))
            if field_name not in dependencies:
                dependencies.append(field_name)

        return tuple(normalized_terms), tuple(dependencies)

    def _expected_derived_field_shape(self):
        if self.data is None:
            self.load_data(None)
        return (int(self.nleafs), *self.data.shape[2:])

    def _normalize_derived_result(self, name, result):
        expected_shape = self._expected_derived_field_shape()
        try:
            result = np.ascontiguousarray(result, dtype=np.double)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Derived field {name!r} result cannot be converted to float64.") from exc
        if result.shape != expected_shape:
            raise ValueError(
                f"Derived field {name!r} returned shape {result.shape}; "
                f"expected {expected_shape}."
            )
        return result

    def _check_derived_dependencies(self, name, definition, *, require_original_ghosts=False):
        loaded_names = self._loaded_field_name_map()
        missing = [
            dependency
            for dependency in definition.dependencies
            if dependency not in loaded_names
        ]
        if missing:
            raise ValueError(
                f"Derived field {name!r} dependencies are not loaded: {missing}."
            )

        if require_original_ghosts or definition.requires_ghosts:
            materialized = []
            insufficient_ghost_layers = []
            required_layers = 2 if require_original_ghosts else 1
            for dependency in definition.dependencies:
                column = self._field_column_for_name(dependency)
                if column.source_kind != FIELD_SOURCE_ORIGINAL:
                    materialized.append(dependency)
                elif column.ghost_valid_layers < required_layers:
                    insufficient_ghost_layers.append(dependency)
            if materialized:
                raise ValueError(
                    f"Derived field {name!r} requires exchanged ghost cells for "
                    f"dependencies {materialized}, but only original loaded fields "
                    "currently have a ghost-cell exchange contract."
                )
            if insufficient_ghost_layers:
                raise ValueError(
                    f"Derived field {name!r} requires at least {required_layers} valid "
                    f"ghost layer{'s' if required_layers != 1 else ''} "
                    f"for dependencies {insufficient_ghost_layers}."
                )

    def _check_ghost_required_recipe(self, name):
        if self.ghost_width <= 0:
            raise ValueError(f"Derived field {name!r} requires ghost cells.")
        if self.mesh is None:
            raise ValueError(f"Derived field {name!r} requires an initialized mesh.")
        if not self.mesh.has_padded_data():
            raise ValueError(f"Derived field {name!r} requires padded ghost-cell storage.")

    def _append_interior_derived_results(self, names, results, ghost_valid_layers=0):
        names = list(names)
        if not names:
            return
        self.data = np.concatenate(
            (self.data, np.stack(results, axis=1)),
            axis=1,
        )
        self._append_derived_field_columns(names, ghost_valid_layers)
        if self.ghost_width > 0:
            self._refresh_mesh_after_field_axis_change()

    def _materialize_python_derived_fields(self, names):
        names = list(names)
        results = []

        for name in names:
            definition = self.derived_definitions[name]
            if definition.requires_ghosts:
                self._check_ghost_required_recipe(name)
            self._check_derived_dependencies(
                name,
                definition,
                require_original_ghosts=definition.requires_ghosts,
            )

            result = definition.func(self._derived_context())
            result = self._normalize_derived_result(name, result)
            results.append(result)

        self._append_interior_derived_results(names, results)

    def _build_derivative_batch_arrays(self, names):
        loaded_names = self._loaded_field_name_map()
        term_output_positions = []
        term_field_positions = []
        term_axes = []
        term_coefficients = []

        for output_position, name in enumerate(names):
            definition = self.derived_definitions[name]
            for term in definition.terms:
                term_output_positions.append(output_position)
                term_field_positions.append(loaded_names[term.field_name])
                term_axes.append(term.axis)
                term_coefficients.append(term.coefficient)

        return (
            np.asarray(term_output_positions, dtype=np.uint32),
            np.asarray(term_field_positions, dtype=np.uint32),
            np.asarray(term_axes, dtype=np.uint32),
            np.asarray(term_coefficients, dtype=np.double),
        )

    def _append_derivative_batch_results(self, names, padded_results):
        interior_shape = self.data.shape[2:]
        interior = np.transpose(
            padded_results[
                :,
                self.ghost_width:self.ghost_width + int(interior_shape[0]),
                self.ghost_width:self.ghost_width + int(interior_shape[1]),
                self.ghost_width:self.ghost_width + int(interior_shape[2]),
                :,
            ],
            (0, 4, 1, 2, 3),
        )
        self.data = np.concatenate((self.data, interior), axis=1)
        valid_layers = max(self.ghost_width - 1, 0)
        self._append_derived_field_columns(names, valid_layers)

        if self.ghost_width <= 0:
            return

        output_count = len(names)
        output_start = len(self.loaded_field_names) - output_count
        self._refresh_mesh_after_field_axis_change()
        padded = self.mesh.padded_view()
        padded[..., output_start:output_start + output_count] = padded_results
        self.data = self.mesh.interior_view()

    def _materialize_derivative_batch(self, names):
        for name in names:
            definition = self.derived_definitions[name]
            self._check_derived_dependencies(name, definition, require_original_ghosts=True)
            if self.ghost_width < 2:
                raise ValueError(
                    f"Derivative field {name!r} requires ghost_width >= 2; "
                    f"got {self.ghost_width}."
                )
            self._check_ghost_required_recipe(name)

        (
            term_output_positions,
            term_field_positions,
            term_axes,
            term_coefficients,
        ) = self._build_derivative_batch_arrays(names)

        padded = self.mesh.padded_view()
        result_shape = (*padded.shape[:4], len(names))
        padded_results = np.empty(result_shape, dtype=np.double)
        self.mesh.first_derivative_fields(
            padded_results,
            term_output_positions,
            term_field_positions,
            term_axes,
            term_coefficients,
        )
        self._append_derivative_batch_results(names, padded_results)

    def materialize_fields(self, names):
        """
        Compute registered derived fields and append them to self.data.
        """
        if isinstance(names, str):
            names = [names]
        if self.data is None:
            self.load_data(None)

        pending_names = []
        for name in names:
            name = str(name)
            if name in self.derived_field_names or name in pending_names:
                continue
            if name not in self.derived_definitions:
                raise KeyError(f"Derived field {name!r} is not registered.")
            pending_names.append(name)

        python_names = []
        derivative_names = []
        for name in pending_names:
            definition = self.derived_definitions[name]
            if isinstance(definition, DerivativeFieldDefinition):
                derivative_names.append(name)
            else:
                python_names.append(name)

        if python_names:
            self._materialize_python_derived_fields(python_names)
        if derivative_names:
            self._materialize_derivative_batch(derivative_names)

    def register_derived(self, name, func, dependencies, requires_ghosts: bool = False):
        """
        Register a named derived-field recipe without computing it.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Derived field name must be a non-empty string.")
        if name in self.wnames:
            raise ValueError(f"Derived field name {name!r} collides with an original field name.")
        if not callable(func):
            raise ValueError(f"Derived field {name!r} requires a callable function.")
        if dependencies is None or isinstance(dependencies, str):
            raise ValueError(f"Derived field {name!r} dependencies must be field names.")

        try:
            normalized_dependencies = tuple(dependencies)
        except TypeError as exc:
            raise ValueError(f"Derived field {name!r} dependencies must be field names.") from exc

        for dependency in normalized_dependencies:
            if not isinstance(dependency, str) or not dependency:
                raise ValueError(
                    f"Derived field {name!r} dependencies must be non-empty field names, "
                    f"got {dependency!r}."
                )

        if name in self.derived_field_names:
            self._drop_materialized_derived_field(name)

        self.derived_definitions[name] = DerivedFieldDefinition(
            func=func,
            dependencies=normalized_dependencies,
            requires_ghosts=bool(requires_ghosts),
        )

    def register_derivative(self, name, terms):
        """
        Register a named first-derivative stencil recipe without computing it.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Derivative field name must be a non-empty string.")
        if name in self.wnames:
            raise ValueError(f"Derivative field name {name!r} collides with an original field name.")

        normalized_terms, dependencies = self._normalize_derivative_terms(name, terms)

        if name in self.derived_field_names:
            self._drop_materialized_derived_field(name)

        self.derived_definitions[name] = DerivativeFieldDefinition(
            terms=normalized_terms,
            dependencies=dependencies,
        )
