"""Array-only operations on explicitly sampled curves."""

import numpy as np


def _samples(values, arclength, *, strict=False):
    values = np.asarray(values, dtype=float)
    arclength = np.asarray(arclength, dtype=float)
    if values.ndim < 1 or arclength.shape != (len(values),):
        raise ValueError('arclength must match the first axis of values')
    if not np.isfinite(values).all() or not np.isfinite(arclength).all():
        raise ValueError('curve samples and arclength must be finite')
    ds = np.diff(arclength)
    if np.any(ds <= 0) if strict else np.any(ds < 0):
        raise ValueError('arclength must increase strictly' if strict else 'arclength must not decrease')
    return values, arclength, ds


def _trapezoid(values, ds):
    weights = ds.reshape((-1,)+(1,)*(values.ndim-1))
    return np.sum(.5*(values[:-1]+values[1:])*weights, axis=0)


def line_integral(values, arclength):
    """Integrate sampled values with the trapezoid rule on a supplied curve.

    Parameters
    ----------
    values : array-like
        Finite samples shaped (n, ...); trailing axes are independent quantities.
        Apply pointwise transformations before calling, for example squared norm.
    arclength : array-like
        Finite nondecreasing distances (n,) in explicitly chosen length units.
        Repeated distances contribute zero length, including a repeated seed.

    Returns
    -------
    scalar or ndarray
        Integral over the supplied samples, in value units times length units.
        Empty and one-point curves return zero. No path completion is inferred.

    Notes
    -----
    This is sampled-curve quadrature, distinct from integrating at RK stages.
    Check trajectory termination and sample validity before interpreting results.
    """
    values, _, ds = _samples(values, arclength)
    return _trapezoid(values, ds)


def line_derivative(values, arclength, *, edge_order=1):
    """Differentiate sampled values on a nonuniform curve-distance grid.

    Parameters
    ----------
    values : array-like
        Finite samples shaped (n, ...), differentiated along the first axis.
    arclength : array-like
        Finite strictly increasing distances (n,). Remove repeated seed samples
        before differentiation. Distances determine orientation and length units.
    edge_order : int
        Endpoint difference order, 1 or 2; at least edge_order+1 samples required.
        Interior samples use three-point differences on the nonuniform grid.

    Returns
    -------
    ndarray
        Derivatives matching values, in value units divided by length units.

    Notes
    -----
    This differentiates the saved sample sequence, not the spatial interpolant
    or a prepared grid gradient. It neither samples fields nor retraces paths.
    """
    if type(edge_order) is not int or edge_order not in (1, 2):
        raise ValueError('edge_order must be 1 or 2')
    values, distance, _ = _samples(values, arclength, strict=True)
    if len(values) < edge_order+1:
        raise ValueError('too few samples for the requested endpoint order')
    return np.gradient(values, distance, axis=0, edge_order=edge_order)
