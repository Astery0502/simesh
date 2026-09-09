"""Owned composition, provenance, and multi-output scientific workflows."""

from dataclasses import replace
import gc
import re
import weakref

import numpy as np
import pytest
import simesh as sm
from simesh.field_ops import select_fields, merge_fields
from simesh.operators.derived import derive_many
from fixtures import mixed_source


@pytest.fixture
def prepared():
    with mixed_source()[0] as source:
        return sm.prepare(source, scheme="exact-phase")


def test_noncontiguous_selection_rename_units_and_ownership(prepared):
    fields = replace(prepared, fields=(sm.FieldDefinition("rho", "g/cm^3"),
                                      sm.FieldDefinition("vx", "cm/s"),
                                      sm.FieldDefinition("p", "dyn/cm^2")))
    result = select_fields(fields, ("p", np.int64(0)), names=("pressure", "density"))
    assert [d.name for d in result.fields] == ["pressure", "density"]
    assert [d.units for d in result.fields] == ["dyn/cm^2", "g/cm^3"]
    np.testing.assert_array_equal(result.values, fields.values[..., [2, 0]])
    assert result.values.flags.owndata and result.values.base is None
    assert result.values.flags.c_contiguous and not result.values.flags.writeable
    assert not np.shares_memory(result.values, fields.values)
    assert result.source is fields.source and result.scheme == fields.scheme
    assert result.value_identity is not fields.value_identity and result.derivation is None
    assert select_fields(fields, "vx").fields == (fields.fields[1],)
    assert select_fields(fields, 0).fields == (fields.fields[0],)
    np.testing.assert_array_equal(select_fields(fields).values, fields.values)


@pytest.mark.parametrize("components", [(), (0, 0), ("b1", 0), (-1,), (3,), (1.5,), (True,), ("missing",)])
def test_invalid_selections(prepared, components):
    with pytest.raises(ValueError):
        select_fields(prepared, components)


@pytest.mark.parametrize("names", [("same", "same"), ("only",), ("", "valid")])
def test_invalid_output_names(prepared, names):
    with pytest.raises(ValueError):
        select_fields(prepared, (0, 1), names=names)


def test_ambiguous_input_names_require_indices_and_explicit_rename(prepared):
    ambiguous = replace(prepared, fields=(prepared.fields[0], prepared.fields[0], prepared.fields[2]))
    with pytest.raises(ValueError, match="ambiguous"):
        select_fields(ambiguous, "b1")
    with pytest.raises(ValueError, match="duplicate"):
        select_fields(ambiguous, (0, 1))
    assert len(select_fields(ambiguous, (0, 1), names=("first", "second")).fields) == 2


def test_merge_aligns_physical_slots_and_selection_order(prepared):
    a = select_fields(prepared, (2, 0), names=("z", "x"))
    b = select_fields(prepared, (1,), names=("y",))
    permutation = np.roll(np.arange(len(b.leaf_ids))[::-1], 2)
    directory = np.full(b.mesh.leaf_count, -1, dtype=np.int64)
    directory[b.leaf_ids[permutation]] = np.arange(len(permutation))
    b = replace(b, _values=b.values[permutation].copy(), slot_of_leaf=directory,
                selection=sm.Selection(b.mesh, b.leaf_ids[::-1]))
    a = replace(a, selection=sm.Selection(a.mesh, a.leaf_ids[::-1]))
    result = merge_fields((a, b))
    assert [d.name for d in result.fields] == ["z", "x", "y"]
    assert result.selection is a.selection
    np.testing.assert_array_equal(result.slot_of_leaf[result.leaf_ids], np.arange(len(result.leaf_ids)))
    for leaf in result.leaf_ids:
        np.testing.assert_array_equal(result.values[result.slot_of_leaf[leaf]],
                                      prepared.values[prepared.slot_of_leaf[leaf]][..., [2, 0, 1]])
    assert result.interior().shape[-1] == 3
    assert result.source == (a.source, b.source)
    assert result.values.flags.owndata and result.derivation is None


def test_common_valid_halo_ignores_padding(prepared):
    backing = np.full((len(prepared.leaf_ids), 14, 14, 14, 3), np.nan)
    backing[:, 1:-1, 1:-1, 1:-1] = prepared.values
    padded = replace(prepared, _values=backing, storage_halo=3)
    cropped = select_fields(padded, 0)
    assert cropped.valid_halo == cropped.storage_halo == 2
    np.testing.assert_array_equal(cropped.values[..., 0], prepared.values[..., 0])
    narrow = replace(prepared, valid_halo=1)
    merged = merge_fields((padded, narrow), names=tuple(f"f{i}" for i in range(6)))
    assert merged.storage_halo == merged.valid_halo == 1
    assert np.isfinite(merged.values).all()
    np.testing.assert_array_equal(merged.values[..., :3], prepared.values[:, 1:-1, 1:-1, 1:-1])
    raw = replace(narrow, valid_halo=0)
    merged = merge_fields((padded, raw), names=tuple(f"f{i}" for i in range(6)))
    assert merged.valid_halo == merged.storage_halo == 0
    with pytest.raises(ValueError, match="halo"):
        sm.curl(select_fields(merged, (0, 1, 2)))


def test_merge_conflicts_mesh_and_coverage(prepared):
    with pytest.raises(ValueError, match="duplicate"):
        merge_fields((prepared, prepared))
    with pytest.raises(ValueError):
        merge_fields(())
    with pytest.raises(TypeError):
        merge_fields({"a": prepared})
    with mixed_source()[0] as other:
        other = sm.prepare(other, scheme="exact-phase")
    with pytest.raises(ValueError, match="Mesh"):
        merge_fields((prepared, other))
    directory = np.full(prepared.mesh.leaf_count, -1, dtype=np.int64)
    directory[0] = 0
    partial = replace(prepared, _values=prepared.values[:1].copy(), slot_of_leaf=directory,
                      selection=sm.Selection(prepared.mesh, [0]))
    with pytest.raises(ValueError, match="coverage"):
        merge_fields((prepared, partial))


def test_multi_output_once_per_leaf_order_shape_and_shared_formula(prepared):
    visited = []

    def recipe(ctx):
        value = ctx.field("b1")
        assert not value.flags.writeable
        visited.append(value.shape)
        squared = value**2
        return {"constant": 2., "squared": squared, "offset": squared + 3.}

    result = derive_many(prepared, {"offset": "code^2", "squared": "code^2", "constant": "1"}, recipe)
    assert len(visited) == len(prepared.leaf_ids)
    assert [d.name for d in result.fields] == ["offset", "squared", "constant"]
    assert {d.interpretation for d in result.fields} == {"pointwise-derived"}
    np.testing.assert_array_equal(result.values[..., 0], prepared.values[..., 0]**2 + 3)
    np.testing.assert_array_equal(result.values[..., 1], prepared.values[..., 0]**2)
    np.testing.assert_array_equal(result.values[..., 2], 2.)
    assert result.values.flags.owndata and result.value_identity is not prepared.value_identity
    scalar = sm.derive(prepared, "squared", lambda ctx: ctx.field(0)**2, units="code^2")
    np.testing.assert_array_equal(scalar.values[..., 0], result.values[..., 1])
    gradient = sm.derivative(result, [[("squared", "x", 1.)]], [sm.FieldDefinition("dx")])
    assert gradient.valid_halo == result.valid_halo - 1


def test_multi_output_aligns_groups_and_common_support(prepared):
    b = replace(prepared, _values=prepared.values[::-1].copy(),
                slot_of_leaf=(len(prepared.leaf_ids)-1-prepared.slot_of_leaf).copy(), valid_halo=1)
    result = derive_many({"a": prepared, "b": b}, {"difference": "code", "copy": "code"},
                         lambda ctx: {"copy": ctx.field("b2", group="b"),
                                      "difference": ctx.field(0, group="a")-ctx.field(0, group="b")})
    assert result.storage_halo == result.valid_halo == 1
    np.testing.assert_array_equal(result.values[..., 0], 0.)
    np.testing.assert_array_equal(result.values[..., 1], prepared.values[:, 1:-1, 1:-1, 1:-1, 1])


@pytest.mark.parametrize("returned", [{}, {"a": 1, "extra": 2}, [1], {"a": np.zeros((2, 2))}])
def test_multi_output_rejects_incomplete_or_wrong_shapes(prepared, returned):
    with pytest.raises(ValueError):
        derive_many(prepared, {"a": "code"}, lambda ctx: returned)


def test_multi_output_definition_validation_and_nonfinite_values(prepared):
    for definitions in ({}, (sm.FieldDefinition("a"), sm.FieldDefinition("a")), ("a",)):
        with pytest.raises(ValueError):
            derive_many(prepared, definitions, lambda ctx: pytest.fail("invalid definitions ran"))
    result = derive_many(prepared, (sm.FieldDefinition("a", "K", "pointwise-derived"),),
                         lambda ctx: {"a": np.nan})
    assert np.isnan(result.values).all() and result.fields[0].units == "K"


def test_borrow_expiration_and_results_release_parent():
    source, _ = mixed_source()
    batches = sm.iter_prepared(source, scheme="exact-phase", batch_size=1)
    borrowed = next(batches)
    selected = select_fields(borrowed, (2, 0))
    merged = merge_fields((borrowed,))
    derived = derive_many(borrowed, {"a": "code", "b": "code"},
                          lambda ctx: {"a": ctx.field(0), "b": ctx.field(1)})
    expected = [value.values.copy() for value in (selected, merged, derived)]
    next(batches)
    for call in (lambda: select_fields(borrowed, 0), lambda: merge_fields((borrowed,)),
                 lambda: derive_many(borrowed, {"a": "code"}, lambda ctx: {"a": 1.})):
        with pytest.raises(RuntimeError, match="expired"):
            call()
    batches.close()
    source.close()
    for value, data in zip((selected, merged, derived), expected):
        np.testing.assert_array_equal(value.values, data)
    parent_ref = weakref.ref(selected.values)
    detached = select_fields(selected, 0)
    del selected
    gc.collect()
    assert parent_ref() is None
    assert detached.values.flags.owndata


def test_callback_cannot_publish_after_expiring_borrow():
    with mixed_source()[0] as source:
        batches = sm.iter_prepared(source, scheme="exact-phase", batch_size=1)
        borrowed = next(batches)

        def expire(ctx):
            batches.close()
            return {"a": 1.}

        with pytest.raises(RuntimeError, match="expired"):
            derive_many(borrowed, {"a": "code"}, expire)


def test_memory_rejection_counts_parent_backing_and_all_outputs(prepared):
    # A contiguous view can still retain a much larger physical allocation.
    backing = np.empty((len(prepared.leaf_ids)+100, *prepared.values.shape[1:]))
    backing[:len(prepared.leaf_ids)] = prepared.values
    view = replace(prepared, _values=backing[:len(prepared.leaf_ids)])
    for call in (lambda limit: select_fields(view, 0, memory_limit=limit),
                 lambda limit: merge_fields((view,), memory_limit=limit),
                 lambda limit: derive_many(view, {"a": "code", "b": "K"},
                                           lambda ctx: pytest.fail("callback before admission"),
                                           memory_limit=limit)):
        with pytest.raises(MemoryError):
            call(backing.nbytes - 1)
    # Admission is reproducible and covers all output components and scratch.
    def operation(limit):
        return derive_many(prepared, {"a": "code", "b": "K"},
                           lambda ctx: {"a": ctx.field(0), "b": 2.}, memory_limit=limit)
    with pytest.raises(MemoryError) as error:
        operation(1)
    required = int(re.search(r"at most (\d+)", str(error.value)).group(1))
    assert operation(required).values.shape[-1] == 2
    with pytest.raises(MemoryError):
        operation(required-1)


def test_combined_mixed_units_to_curl_qsl_and_selected_bidirectional_trace():
    mesh = sm.mesh_from_forest((1, 1, 1), np.array([True]), lower=(0, 0, 0),
                              upper=(1, 1, 1), block_shape=(8, 8, 8))
    raw = np.zeros((1, 5, 8, 8, 8))
    raw[:, 0], raw[:, 3], raw[:, 4] = 2., 1., 1.e6
    with sm.source_from_arrays(mesh, raw, ("rho", "b1", "b2", "b3", "temperature"),
                                units={"rho": "g/cm^3", "b1": "G", "b2": "G", "b3": "G", "temperature": "K"}) as source:
        prepared = sm.prepare(source, scheme="exact-phase")
    thermo = select_fields(prepared, ("temperature", "rho"))
    vector_parts = select_fields(prepared, (3, 1, 2))
    composed = merge_fields((thermo, vector_parts))
    magnetic = select_fields(composed, ("b1", "b2", "b3"))
    companion = sm.curl(magnetic)
    np.testing.assert_array_equal(companion.values, 0.)
    seeds = sm.PointSet(np.array([[.5, .5, .5]]))
    from simesh import applications as app
    diagnostics = app.surface_diagnostics(magnetic, seeds, curl_field=companion)
    assert diagnostics.q_valid.all() and diagnostics.twist_valid.all()
    np.testing.assert_allclose(diagnostics.data.q, 2.)
    chosen = diagnostics.threshold(q_min=1.9)
    lines = app.trace(magnetic, chosen, step=.025, max_steps=100)
    assert len(lines.branch(chosen.ids[0], -1)) > 1
    assert len(lines.branch(chosen.ids[0], 1)) > 1
    # No selection/merge can accidentally reuse a certificate for different B.
    for other in (select_fields(magnetic, (2, 1, 0)), select_fields(magnetic),
                  merge_fields((magnetic,))):
        with pytest.raises(ValueError, match="derive from this vector"):
            sm.qsl(other, seeds.positions, curl_field=companion)
    # Selecting or merging curl values strips the certificate as well.
    for other_curl in (select_fields(companion, (2, 1, 0)), merge_fields((companion,))):
        with pytest.raises(ValueError, match="derive from this vector"):
            sm.trace(magnetic, seeds.positions, step=.025, twist=True, curl_field=other_curl)
