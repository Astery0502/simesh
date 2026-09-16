"""Composed line calculations across AMR retries, leases and worker schedules."""

from dataclasses import replace
import numpy as np
import pytest
import simesh as sm
from simesh.bounded import PreparedPool, trace_bounded
from test_connectivity import magnetic_source


def rates(mesh, *, region=None, width=4):
    local = np.indices(mesh.block_shape)+.5
    values = np.empty((mesh.leaf_count,width,*mesh.block_shape))
    for leaf in range(mesh.leaf_count):
        z = mesh.bounds[leaf,0,2]+local[2]*mesh.spacing[leaf,2]
        values[leaf,0] = 1.
        values[leaf,1] = z
        values[leaf,2] = 2*z
        values[leaf,3:] = 2.
    with sm.source_from_arrays(mesh,values,tuple(f'rate{i}' for i in range(width))) as source:
        return sm.prepare(source,scheme='exact-phase',region=region)


@pytest.mark.parametrize('direction', [-1,1])
@pytest.mark.parametrize('seed_batch', [1,2])
def test_rk_integrals_share_geometry_and_are_independent_of_batches(direction,seed_batch):
    with magnetic_source(lambda x,y,z: np.array([0*x,0*y,np.ones_like(z)]),cells=4,mixed=True) as source:
        field = sm.prepare(source,scheme='exact-phase')
    integrands = rates(field.mesh,width=17)
    seeds = np.array([[.13,.17,.51],[.21,.27,.49]])
    options = dict(direction=direction,max_length=.2,max_steps=1000,trajectories=True)
    geometric = sm.trace(field,seeds,**options)
    serial = sm.trace(field,seeds,integrands=integrands,**options)
    parallel = sm.trace(field,seeds,integrands=integrands,workers=2,schedule='dynamic',seed_batch=seed_batch,**options)
    for name in ('positions','length','steps','termination','samples','trajectories'):
        np.testing.assert_array_equal(getattr(serial,name),getattr(geometric,name))
        np.testing.assert_array_equal(getattr(serial,name),getattr(parallel,name))
    np.testing.assert_array_equal(serial.integrals,parallel.integrals)
    length = serial.length
    expected = seeds[:,2]*length+direction*.5*length**2
    np.testing.assert_allclose(serial.integrals[:,0],length,rtol=2e-14)
    np.testing.assert_allclose(serial.integrals[:,1],expected,rtol=2e-14)
    np.testing.assert_allclose(serial.integrals[:,2]/serial.integrals[:,3],expected/length,rtol=2e-14)
    assert serial.integral_fields == integrands.fields
    assert geometric.integrals is None


def test_bounded_rk_resume_keeps_all_integral_stages_with_twist():
    with magnetic_source(lambda x,y,z: np.array([0*x,0*y,np.ones_like(z)]),cells=4,mixed=True) as source:
        field = sm.prepare(source,scheme='exact-phase')
        integrands = rates(field.mesh)
        seeds = np.array([[.13,.17,.51],[.21,.27,.49]])
        options = dict(direction=-1,max_steps=200,step_fraction=.0625,
                       trajectories=True,twist=True,integrands=integrands,workers=2)
        direct = sm.trace(field,seeds,**options)
        with PreparedPool(source,('b1','b2','b3'),scheme='exact-phase',capacity=1) as pool:
            resumed = trace_bounded(pool,seeds,**options)
            partial = rates(field.mesh,region=([-1,-1,0],[1,1,.2]))
            with pytest.raises(ValueError,match='entire pool Mesh'):
                trace_bounded(pool,seeds,integrands=partial)
    assert resumed.misses.sum() > 0
    for name in ('positions','length','steps','termination','trajectories','twist','integrals'):
        np.testing.assert_array_equal(getattr(resumed,name),getattr(direct,name))


def test_missing_or_invalid_integrands_preserve_only_accepted_prefix():
    with magnetic_source(lambda x,y,z: np.array([0*x,0*y,np.ones_like(z)]),cells=4,mixed=True) as source:
        field = sm.prepare(source,scheme='exact-phase')
    partial = rates(field.mesh,region=([-1,-1,0],[1,1,.2]))
    result = sm.trace(field,np.array([[.13,.17,.2]]),integrands=partial,max_steps=1000)
    assert result.termination[0] == sm.Termination.MISSING_COVERAGE
    assert result.length[0] > 0
    np.testing.assert_allclose(result.integrals[:,0],result.length)
    full = rates(field.mesh)
    values = full.values.copy()
    upper_leaves = field.mesh.bounds[:,0,2] >= .5
    values[full.slot_of_leaf[upper_leaves]] = np.nan
    invalid = replace(full,_values=values)
    stopped = sm.trace(field,np.array([[.13,.17,.49]]),integrands=invalid,
                       step=.04,step_fraction=None,max_steps=1)
    assert stopped.termination[0] == sm.Termination.NONFINITE_DIAGNOSTIC
    assert stopped.steps[0] == 0
    np.testing.assert_array_equal(stopped.integrals,0.)


def test_output_pause_does_not_terminate_or_recount_integrals():
    from simesh.tracing import _new_state, _advance_state
    with magnetic_source(lambda x,y,z: np.array([0*x,0*y,np.ones_like(z)]),cells=4) as source:
        field = sm.prepare(source,scheme='exact-phase')
    integrands = rates(field.mesh)
    seeds, ids = np.array([[.13,.17,.25]]), np.array([4],np.int64)
    state = _new_state(field.mesh,seeds,ids,4,np.inf,True,False,path_capacity=3,integral_count=4)
    options = dict(step=.01,step_fraction=None,max_steps=4,max_length=np.inf,
                   null_threshold=0.,direction=1,workers=1,executor=None,integrands=integrands)
    _advance_state(field,None,state,**options)
    assert state.steps[0] == 2 and state.status[0] == sm.Termination.RUNNING
    _advance_state(field,None,state,path_start=2,**options)
    direct = sm.trace(field,seeds,step=.01,step_fraction=None,max_steps=4,integrands=integrands)
    np.testing.assert_array_equal(state.integrals,direct.integrals)
    np.testing.assert_array_equal(state.positions,direct.positions)


def test_sampled_curve_derivative_nonuniform_grid_and_seed_join():
    distance = np.array([-.7,-.2,0.,.1,.4,1.])
    values = np.column_stack((distance**2,3*distance+2))
    derivatives = sm.line_derivative(values,distance,edge_order=2)
    np.testing.assert_allclose(derivatives[:,0],2*distance,atol=2e-14)
    np.testing.assert_allclose(derivatives[:,1],3.,atol=2e-14)
    joined = np.array([-.5,0.,0.,.5])
    np.testing.assert_allclose(sm.line_integral(2+joined,joined),2.)
    with pytest.raises(ValueError,match='strictly'):
        sm.line_derivative(2+joined,joined)
    with pytest.raises(ValueError,match='finite'):
        sm.line_integral([1,np.nan],[0,1])


@pytest.mark.parametrize('schedule', ['static','dynamic'])
def test_openmp_integral_buffers_are_independent(schedule):
    from simesh._kernels.native import openmp_build_info
    if not openmp_build_info()['enabled']:
        pytest.skip('ordinary build')
    with magnetic_source(lambda x,y,z: np.array([0*x,0*y,np.ones_like(z)]),cells=4,mixed=True) as source:
        field = sm.prepare(source,scheme='exact-phase')
    integrands = rates(field.mesh,width=17)
    seeds = np.column_stack((np.linspace(.1,.3,19),np.full(19,.17),np.linspace(.3,.6,19)))
    options = dict(integrands=integrands,max_length=.2,max_steps=1000,twist=True,seed_batch=19)
    expected = sm.trace(field,seeds,**options)
    actual = sm.trace(field,seeds,workers=3,backend='openmp',schedule=schedule,**options)
    for name in ('positions','length','steps','termination','twist','integrals'):
        np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
