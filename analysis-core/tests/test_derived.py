"""Pointwise recipes compose with native derivatives and owned consumers."""

from dataclasses import replace
import numpy as np
import pytest
import simesh as sm
from fixtures import mixed_source


def test_recipe_derivative_and_sampling():
    mesh = sm.mesh_from_forest((1, 1, 1), np.array([True]),
                              lower=(0, 0, 0), upper=(1, 1, 1), block_shape=(8, 8, 8))
    x, y, z = (np.indices(mesh.block_shape)+.5)/8
    with sm.source_from_arrays(mesh, np.array([[x, y, z]]), ("x", "y", "z")) as source:
        ready = sm.prepare(source, scheme="exact-phase")
    squared = sm.derive(ready, "squared", lambda ctx: ctx.field("x")**2)
    gradient = sm.derivative(squared, [[("squared", "x", 1.)]],
                             [sm.FieldDefinition("dx")])
    assert squared.valid_halo == 2 and gradient.valid_halo == 1
    values, _, valid = sm.sample(gradient, np.array([[.5, .5, .5]]))
    assert valid.all()
    np.testing.assert_allclose(values, 1.)
    combined = sm.derive({"g": gradient, "q": squared}, "combined",
                         lambda ctx: ctx.field("dx", group="g") + ctx.field("squared", group="q"))
    assert combined.valid_halo == 1
    assert not np.shares_memory(squared.values, ready.values)
    assert not combined.values.flags.writeable
    np.testing.assert_allclose(combined.interior()[0, 2:-2, 2:-2, 2:-2, 0],
                               (2*x+x*x)[2:-2, 2:-2, 2:-2])


def test_groups_align_leaf_order_and_trim_invalid_padding():
    source, _ = mixed_source()
    a = sm.prepare(source, scheme="exact-phase")
    storage = np.full((len(a.leaf_ids), 14, 14, 14, 3), np.nan)
    storage[:, 1:-1, 1:-1, 1:-1] = a.values
    padded = replace(a, _values=storage, storage_halo=3)
    copied = sm.derive(padded, "copy", lambda ctx: ctx.field("b1"))
    np.testing.assert_array_equal(copied.values[..., 0], a.values[..., 0])
    assert copied.storage_halo == copied.valid_halo == 2
    b = sm.prepare(source, leaf_ids=a.leaf_ids[::-1], scheme="exact-phase")
    difference = sm.derive({"a": a, "b": b}, "difference",
                           lambda ctx: ctx.field("b1", group="a")-ctx.field("b1", group="b"))
    np.testing.assert_array_equal(difference.values, 0.)
    raw = sm.read_fields(source)
    interior = sm.derive({"a": a, "raw": raw}, "interior",
                         lambda ctx: ctx.field("b1", group="a")-ctx.field("b1", group="raw"))
    assert interior.valid_halo == interior.storage_halo == 0
    np.testing.assert_array_equal(interior.values, 0.)
    with pytest.raises(ValueError, match="halo"):
        sm.derivative(interior, [[("interior", "x", 1.)]], [sm.FieldDefinition("dx")])
    source.close()


def test_invalid_combinations_and_recipe_failures():
    source, _ = mixed_source()
    ready = sm.prepare(source, scheme="exact-phase")
    partial = sm.prepare(source, leaf_ids=[0], scheme="exact-phase")
    with pytest.raises(ValueError, match="coverage"):
        sm.derive({"a": ready, "b": partial}, "bad", lambda ctx: 0.)
    with pytest.raises(ValueError, match="block shape"):
        sm.derive(ready, "bad", lambda ctx: np.zeros((2, 2)))
    with pytest.raises(ValueError, match="exactly one"):
        sm.derive(ready, "bad", lambda ctx: ctx.field("absent"))
    with pytest.raises(MemoryError):
        sm.derive(ready, "bad", lambda ctx: pytest.fail("callback ran before admission"), memory_limit=1)
    result = sm.derive(ready, "constant", lambda ctx: 3.)
    np.testing.assert_array_equal(result.values, 3.)
    source.close()


def test_named_selectors_preserve_ambiguity_and_repeated_derivative_terms():
    source, _ = mixed_source()
    with source:
        ready = sm.prepare(source, scheme="exact-phase")
        definitions = [sm.FieldDefinition("sum")]
        named = sm.derivative(ready, [[("b1", "x", 1.), ("b1", "y", 2.), ("b2", "z", 3.)]],
                              definitions)
        numbered = sm.derivative(ready, [[(0, 0, 1.), (0, 1, 2.), (1, 2, 3.)]], definitions)
        np.testing.assert_array_equal(named.values, numbered.values)
        ambiguous = replace(ready, fields=(ready.fields[0], ready.fields[0], ready.fields[2]))
        with pytest.raises(ValueError, match="ambiguous"):
            sm.derive(ambiguous, "bad", lambda ctx: ctx.field("b1"))
        with pytest.raises(ValueError, match="ambiguous"):
            sm.derivative(ambiguous, [[("b1", "x", 1.)]], definitions)
        with pytest.raises(ValueError, match="missing"):
            sm.read_fields(source, "absent")


def test_recipe_detaches_borrowed_batch():
    source, _ = mixed_source()
    batches = sm.iter_prepared(source, scheme="exact-phase", batch_size=1)
    borrowed = next(batches)
    result = sm.derive(borrowed, "copy", lambda ctx: ctx.field("b1"))
    expected = result.values.copy()
    next(batches)
    batches.close()
    source.close()
    np.testing.assert_array_equal(result.values, expected)
