"""Local-cell RK step limits across AMR, bounded execution and applications."""

import numpy as np
import pytest
import simesh as sm
from simesh import applications as app
from simesh.bounded import PreparedPool, trace_bounded
from test_connectivity import magnetic_source


@pytest.fixture
def source():
    with magnetic_source(lambda x,y,z: np.array([0*x,0*y,np.ones_like(z)]),
                         cells=4, mixed=True) as source:
        yield source


@pytest.mark.parametrize("direction,start", [(-1,.51),(1,.45)])
def test_steps_follow_local_cells_and_restart_at_finer_stages(source, direction, start):
    fields = sm.prepare(source, scheme="exact-phase")
    seeds = np.array([[.13,.17,start]])
    result = sm.trace(fields, seeds, direction=direction, max_steps=12, trajectories=True)
    path = result.trajectories[0,:result.point_counts[0]]
    distances = np.linalg.norm(np.diff(path,axis=0),axis=1)
    owners = fields.mesh.locate(path[:-1])
    caps = .25*fields.mesh.spacing[owners].min(axis=1)
    assert np.all(distances <= caps*(1+1e-12))
    np.testing.assert_allclose(result.length[0], distances.sum(), atol=1e-14)
    np.testing.assert_allclose(distances[0], .015625, atol=1e-14)
    if direction == -1:
        assert distances[0] < caps[0]  # An intermediate stage already enters fine coverage.
        assert result.samples[0] > 4*result.steps[0]
    else:
        assert distances[-1] > distances[0]


@pytest.mark.parametrize("method", ["variational","finite-difference"])
def test_qsl_and_trace_share_local_and_fixed_step_controls(source, method):
    fields = sm.prepare(source, scheme="exact-phase")
    seeds = np.array([[.13,.17,.51]])
    for controls in ({}, {"step":.008}, {"step":.02,"step_fraction":None}):
        q = sm.qsl(fields,seeds,max_steps=1,twist=False,method=method,**controls)
        for side,direction in enumerate((-1,1)):
            traced = sm.trace(fields,seeds,max_steps=1,direction=direction,**controls)
            np.testing.assert_allclose(traced.positions,q.footpoints[:,side],atol=1e-14)
    fixed = sm.trace(fields,seeds,step=.02,step_fraction=None,max_steps=3,trajectories=True)
    np.testing.assert_allclose(np.diff(fixed.trajectories[0,:,2]),.02,atol=1e-14)


def test_bounded_resume_and_segmented_paths_keep_the_reduced_step(source):
    fields = sm.prepare(source, scheme="exact-phase")
    seeds = np.array([[.13,.17,.51],[.21,.27,.49]])
    controls = dict(direction=-1,max_steps=200,trajectories=True,workers=2,step_fraction=.0625)
    direct = sm.trace(fields,seeds,**controls)
    with PreparedPool(source,("b1","b2","b3"),scheme="exact-phase",capacity=1) as pool:
        bounded = trace_bounded(pool,seeds,**controls)
    for name in ("positions","length","steps","termination","trajectories"):
        np.testing.assert_array_equal(getattr(bounded,name),getattr(direct,name))
    assert bounded.misses.sum() > 0
    packed = app.trace(fields,sm.PointSet(seeds),direction="against",max_steps=200,
                       workers=2,step_fraction=.0625)
    for row in range(len(seeds)):
        start,stop = packed.offsets[2*row:2*row+2]
        np.testing.assert_array_equal(packed.positions[start:stop],
                                     direct.trajectories[row,:direct.point_counts[row]])
    assert direct.steps.min() > 64


@pytest.mark.parametrize("controls", [
    {"step_fraction":0}, {"step_fraction":1.1}, {"step_fraction":np.nan},
    {"step_fraction":None}, {"step":0}, {"step":np.inf},
])
def test_shared_step_validation(source, controls):
    fields = sm.prepare(source, scheme="exact-phase")
    seeds = np.array([[.13,.17,.51]])
    for invoke in (lambda: sm.trace(fields,seeds,**controls),
                   lambda: sm.qsl(fields,seeds,**controls),
                   lambda: app.trace(fields,sm.PointSet(seeds),**controls)):
        with pytest.raises(ValueError,match="step"):
            invoke()


def test_unrepresentable_step_stops_without_accepting_duplicate_points(source):
    fields = sm.prepare(source, scheme="exact-phase")
    seeds = np.array([[.13,.17,.51]])
    result = sm.trace(fields,seeds,step=1e-300,max_steps=100,trajectories=True)
    assert result.termination[0] == sm.Termination.UNREPRESENTABLE_STEP
    assert result.steps[0] == 0 and result.length[0] == 0
    np.testing.assert_array_equal(result.positions,seeds)
