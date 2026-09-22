"""Analytic, AMR, endpoint, and failure checks for magnetic connectivity."""

from dataclasses import replace
import numpy as np
import pytest
import simesh as sm


def magnetic_source(function, *, cells=16, mixed=False):
    roots = (1,1,2) if mixed else (1,1,1)
    flags = np.array([False]+[True]*9) if mixed else np.array([True])
    mesh = sm.mesh_from_forest(roots, flags, lower=(-1,-1,0), upper=(1,1,1),
                              block_shape=(cells,cells,cells))
    local = np.indices(mesh.block_shape)+.5
    values = np.empty((mesh.leaf_count,3,*mesh.block_shape))
    for leaf in range(mesh.leaf_count):
        xyz = (mesh.bounds[leaf,0,:,None,None,None] +
               local*mesh.spacing[leaf,:,None,None,None])
        values[leaf] = function(*xyz)
    return sm.source_from_arrays(mesh, values, ("b1","b2","b3"))


BOX = ([-.75,-.75,.125],[.75,.75,.875])


@pytest.mark.parametrize("axis", [0,1,2])
def test_constant_field_faces_boundary_seeds_and_streaming(axis):
    def field(x,y,z):
        b = np.zeros((3,*x.shape))
        b[axis] = 1.
        return b
    with magnetic_source(field) as source:
        ready = sm.prepare(source, scheme="exact-phase")
    seeds = np.array([[.1,.1,.4],[.1,.1,.4],[.1,.1,.4]])
    seeds[1,axis], seeds[2,axis] = ready.mesh.lower[axis], ready.mesh.upper[axis]
    result = sm.qsl(ready, seeds, workers=2, seed_batch=1)
    assert result.complete.all() and result.valid.all()
    np.testing.assert_allclose(result.q, 2., atol=2e-14)
    np.testing.assert_allclose(result.q_perp, 2., atol=2e-14)
    np.testing.assert_array_equal(result.twist, 0.)
    np.testing.assert_allclose(result.length, ready.mesh.upper[axis]-ready.mesh.lower[axis], atol=1e-9)
    np.testing.assert_array_equal(result.footpoints[:,0,axis], ready.mesh.lower[axis])
    np.testing.assert_array_equal(result.footpoints[:,1,axis], ready.mesh.upper[axis])
    np.testing.assert_array_equal(result.boundary, np.tile([5-2*axis,6-2*axis], (3,1)))
    streamed = list(sm.iter_qsl(ready, seeds, seed_batch=2))
    np.testing.assert_allclose(np.concatenate([r.q for r in streamed]), result.q)


@pytest.mark.parametrize("mixed", [False,True])
@pytest.mark.parametrize("method", ["finite-difference","variational"])
@pytest.mark.parametrize("workers", [1,2])
def test_hyperbolic_q_and_independent_endpoint_differences(mixed, method, workers):
    a = .5
    with magnetic_source(lambda x,y,z: np.array([a*x,-a*y,np.ones_like(x)]), mixed=mixed) as source:
        ready = sm.prepare(source, scheme="exact-phase")
    delta = 1e-4
    seeds = np.array([[0.,0.,.5],[delta,0.,.5],[-delta,0.,.5],[0.,delta,.5],[0.,-delta,.5]])
    result = sm.qsl(ready, seeds, bounds=BOX, twist=False, method=method, workers=workers)
    assert result.valid.all()
    expected = 2*np.cosh(2*a*(BOX[1][2]-BOX[0][2]))
    np.testing.assert_allclose(result.q, expected, rtol=.004)
    maps = np.stack(((result.footpoints[1,:,:2]-result.footpoints[2,:,:2])/(2*delta),
                     (result.footpoints[3,:,:2]-result.footpoints[4,:,:2])/(2*delta)), axis=-1)
    jacobian = maps[1] @ np.linalg.inv(maps[0])
    q_mapping = np.sum(jacobian**2)/abs(np.linalg.det(jacobian))
    np.testing.assert_allclose(result.q[0], q_mapping, rtol=.004)
    np.testing.assert_allclose(result.q_perp[0], expected, rtol=.004)
    permuted = replace(ready, selection=sm.Selection(ready.mesh, ready.leaf_ids[::-1]))
    again = sm.qsl(permuted, seeds[:1], bounds=BOX, twist=False, normalization="flux", method=method)
    np.testing.assert_allclose(again.q, result.q[:1], rtol=.003)


@pytest.mark.parametrize("a", [-.5,.5])
@pytest.mark.parametrize("method", ["finite-difference","variational"])
def test_helical_twist_whole_line_direction_sign_and_local_sphere(a, method):
    with magnetic_source(lambda x,y,z: np.array([-a*y,a*x,np.ones_like(x)]), mixed=True) as source:
        ready = sm.prepare(source, scheme="exact-phase")
    seeds = np.array([[0.,0.,.5],[.1,.1,.5]])
    companion = sm.curl(ready)
    result = sm.qsl(ready, seeds, bounds=BOX, curl_field=companion, method=method)
    length_z = BOX[1][2]-BOX[0][2]
    r2 = np.sum(seeds[:,:2]**2, axis=1)
    expected = a*length_z/(2*np.pi*np.sqrt(1+a*a*r2))
    assert result.valid.all()
    np.testing.assert_allclose(result.q, 2., atol=1e-5)
    np.testing.assert_allclose(result.twist, expected, rtol=2e-5)
    local = sm.qsl(ready, seeds[:1], local_radius=.2, method=method)
    assert local.valid.all() and local.q_local is local.q
    np.testing.assert_array_equal(local.boundary, [[sm.Boundary.LOCAL_SPHERE]*2])
    np.testing.assert_allclose(local.length, .4, atol=1e-9)
    np.testing.assert_allclose(local.twist, a*.4/(2*np.pi), rtol=2e-5)
    # A nearby box surface replaces one side of the local sphere.
    near = sm.qsl(ready, np.array([[0.,0.,.05]]), local_radius=.2, method=method)
    np.testing.assert_array_equal(near.boundary, [[sm.Boundary.ZMIN,sm.Boundary.LOCAL_SPHERE]])


def test_partial_coverage_limits_nulls_corners_and_budget():
    with magnetic_source(lambda x,y,z: np.array([np.zeros_like(x),np.zeros_like(x),np.ones_like(x)]),
                         mixed=True) as source:
        ready = sm.prepare(source, scheme="exact-phase")
        partial = sm.prepare(source, region=([-1,-1,0],[1,1,.2]), scheme="exact-phase")
    limited = sm.qsl(ready, np.array([[.1,.1,.4]]), max_steps=1)
    assert not limited.complete.any() and np.isnan(limited.q).all()
    np.testing.assert_allclose(limited.q_perp, 2.)
    np.testing.assert_array_equal(limited.twist, 0.)
    missing = sm.qsl(partial, np.array([[.1,.1,.2]]))
    assert sm.ConnectivityTermination.MISSING_COVERAGE in missing.termination
    assert not missing.complete.any() and np.isnan(missing.q).all()
    outside = sm.qsl(ready, np.array([[2.,0.,.2]]))
    np.testing.assert_array_equal(outside.termination, sm.ConnectivityTermination.OUTSIDE_SEED)
    with pytest.raises(MemoryError):
        sm.qsl(ready, np.array([[.1,.1,.4]]), memory_limit=1)
    with magnetic_source(lambda x,y,z: np.zeros((3,*x.shape))) as source:
        null = sm.prepare(source, scheme="exact-phase")
    result = sm.qsl(null, np.array([[0.,0.,.5]]))
    np.testing.assert_array_equal(result.termination, sm.ConnectivityTermination.NULL_FIELD)
    assert np.isnan(result.q).all() and np.isnan(result.twist).all()
    with magnetic_source(lambda x,y,z: np.array([np.ones_like(x),np.zeros_like(x),np.ones_like(x)])) as source:
        diagonal = sm.prepare(source, scheme="exact-phase")
    corner = sm.qsl(diagonal, np.array([[0.,0.,.5]]), bounds=([-.5,-.5,0],[.5,.5,1]))
    assert corner.complete.all() and not corner.valid.any()
    np.testing.assert_array_equal(corner.boundary, sm.Boundary.EDGE)
    assert np.isnan(corner.q).all()


def test_q_converges_with_spatial_resolution():
    expected = 2*np.cosh(.75)
    errors = []
    for cells in (8,16,32):
        with magnetic_source(lambda x,y,z: np.array([.5*x,-.5*y,np.ones_like(x)]), cells=cells) as source:
            ready = sm.prepare(source, scheme="exact-phase")
        result = sm.qsl(ready, np.array([[0.,0.,.5]]), bounds=BOX, twist=False, method="variational")
        errors.append(abs(result.q[0]-expected))
    assert errors[1] < errors[0]*.4 and errors[2] < errors[1]*.4


def test_nonlinear_amr_map_is_stable_under_step_and_stencil_refinement():
    def quadrupole(x,y,z):
        b = np.zeros((3,*x.shape))
        for position, strength in zip((-1.5,-.5,.5,1.5), (1,-1,1,-1)):
            r = np.array([x-position,y,z+.5])
            b += strength*r/np.sum(r*r,axis=0)**1.5
        return b
    with magnetic_source(quadrupole, cells=32, mixed=True) as source:
        ready = sm.prepare(source, scheme="exact-phase")
    seeds = np.array([[.15,.23,.4],[.35,.23,.6],[.55,.23,.6]])
    coarse = sm.qsl(ready, seeds, bounds=BOX, method="finite-difference", delta=1e-4, step_fraction=.125)
    fine = sm.qsl(ready, seeds, bounds=BOX, method="finite-difference", delta=5e-5, step_fraction=.0625)
    assert coarse.valid.all() and fine.valid.all()
    np.testing.assert_allclose(coarse.q, fine.q, rtol=.01)
    np.testing.assert_allclose(coarse.footpoints, fine.footpoints, atol=1e-4)


def test_requested_surface_at_partial_coverage_edge_and_input_contracts():
    with magnetic_source(lambda x,y,z: np.array([np.zeros_like(x),np.zeros_like(x),np.ones_like(x)]),
                         mixed=True) as source:
        partial = sm.prepare(source, region=([-1,-1,0],[1,1,.2]), scheme="exact-phase")
    seeds = np.array([[.1,.1,.2]])
    upper_z = float(partial.mesh.bounds[partial.leaf_ids,1,2].max())
    result = sm.qsl(partial, seeds, bounds=([-1,-1,0],[1,1,upper_z]))
    assert result.complete.all() and result.valid.all()
    np.testing.assert_array_equal(result.footpoints[0,:,2], [0.,upper_z])
    for options in ({"method":"unknown"}, {"delta":-1}, {"step_fraction":2},
                    {"boundary_tolerance":0}, {"normalization":"unknown"},
                    {"local_radius":1e-12}, {"seed_batch":0}):
        with pytest.raises(ValueError):
            sm.qsl(partial, seeds, **options)
    empty = sm.qsl(partial, np.empty((0,3)))
    assert empty.q.shape == (0,) and empty.footpoints.shape == (0,2,3)
    interior_only_twist_disabled = sm.qsl(replace(partial, valid_halo=1), seeds,
        bounds=([-1,-1,0],[1,1,upper_z]), twist=False, method="finite-difference")
    assert interior_only_twist_disabled.valid.all() and interior_only_twist_disabled.twist is None


@pytest.mark.parametrize("method", ["finite-difference", "variational"])
@pytest.mark.parametrize("seed_z", [.249, .24949999])
def test_local_sphere_stops_before_missing_coverage(method, seed_z):
    with magnetic_source(lambda x,y,z: np.array([np.zeros_like(x),np.zeros_like(x),np.ones_like(x)]),
                         mixed=True) as source:
        partial = sm.prepare(source, region=([-1,-1,0],[1,1,.2]), scheme="exact-phase")
    radius = .0005
    result = sm.qsl(partial, np.array([[.1,.1,seed_z]]), local_radius=radius,
                    twist=False, method=method)
    assert result.valid.all() and result.complete.all()
    np.testing.assert_array_equal(result.termination, sm.ConnectivityTermination.LOCAL_EXIT)
    np.testing.assert_allclose(result.footpoints[0,:,2], [seed_z-radius,seed_z+radius], atol=1e-10)
    np.testing.assert_allclose(result.q, 2., atol=1e-8)
    missing = sm.qsl(partial, np.array([[.1,.1,seed_z]]), local_radius=.002,
                     twist=False, method=method)
    assert not missing.complete.any()
    assert sm.ConnectivityTermination.MISSING_COVERAGE in missing.termination


@pytest.mark.parametrize("method", ["variational", "finite-difference"])
def test_valid_stencils_keep_their_rows_among_failed_seeds(method):
    with magnetic_source(lambda x,y,z: np.array([np.zeros_like(x),np.zeros_like(x),np.ones_like(x)])) as source:
        ready = sm.prepare(source, scheme="exact-phase")
    seeds = np.array([[.1,.1,.4],[2.,0.,.5],[-1.,-1.,0.],[.2,.3,.6]])
    result = sm.qsl(ready, seeds, twist=False, method=method, workers=2, seed_batch=3)
    assert result.valid.tolist() == [True,False,False,True]
    assert not result.stencil_valid[1:3].any()
    direct = sm.qsl(ready, seeds[[0,3]], twist=False, method=method)
    np.testing.assert_allclose(result.q[[0,3]], direct.q)
    np.testing.assert_allclose(result.footpoints[[0,3]], direct.footpoints)


@pytest.mark.parametrize("entry", [sm.qsl, sm.iter_qsl, sm.line_diagnostics, sm.iter_line_diagnostics])
def test_default_variational_avoids_neighbor_tracing(entry, monkeypatch):
    import simesh.connectivity as implementation
    with magnetic_source(lambda x,y,z: np.array([.5*x,-.5*y,np.ones_like(z)])) as source:
        fields = sm.prepare(source, scheme="exact-phase")
    def unavailable(*args, **kwargs):
        raise AssertionError("default variational Q must not launch neighboring seeds")
    monkeypatch.setattr(implementation, "_stencil", unavailable)
    result = entry(fields, np.array([[0.,0.,.5]]), bounds=BOX)
    if entry in (sm.iter_qsl, sm.iter_line_diagnostics):
        result = next(result)
    assert result.method == "variational" and result.valid.all()
    np.testing.assert_allclose(result.q, 2*np.cosh(.75), rtol=.004)


def test_method_support_and_delta_contract(monkeypatch):
    import simesh.connectivity as implementation
    import simesh.applications as app
    with magnetic_source(lambda x,y,z: np.array([0*x,0*y,np.ones_like(z)])) as source:
        fields = sm.prepare(source, scheme="exact-phase")
    points = sm.PointSet(np.array([[0.,0.,.5]]))
    one_halo = replace(fields, valid_halo=1)
    for invoke in (lambda: sm.qsl(one_halo, points.positions, twist=False),
                   lambda: app.connectivity(one_halo, points, quantities="q")):
        with pytest.raises(ValueError, match="requires at least 2"):
            invoke()
    with pytest.raises(ValueError, match="delta only applies to finite-difference"):
        sm.qsl(fields, points.positions, delta=1e-4)
    with pytest.raises(ValueError, match="method must"):
        sm.qsl(fields, points.positions, method="unknown")
    def unavailable(*args, **kwargs):
        raise AssertionError("finite-difference Q must not prepare unit gradients")
    monkeypatch.setattr(implementation, "_unit_gradient", unavailable)
    result = app.connectivity(one_halo, points, quantities="q",
                              method="finite-difference", delta=1e-4)
    assert result.data.method == "finite-difference" and result.data.valid.all()
    np.testing.assert_allclose(result.data.q, 2.)
