"""Associations across geometry, diagnostics, selected lines and ray images."""

import numpy as np
import pytest
import simesh as sm
from simesh import applications as app
from fixtures import mixed_source
from test_connectivity import magnetic_source


def test_point_layout_selection_and_native_sampling_agree():
    with mixed_source()[0] as source:
        fields = sm.prepare(source,scheme="exact-phase")
    plane = sm.Plane([.1,.1,.3],[.6,0,0],[0,.6,0],(4,3))
    points = sm.PointSet.from_plane(plane,ids=np.arange(100,112,dtype=np.int64))
    sampled = app.sample(fields,points,workers=2)
    reference = sm.sample_plane(fields,plane)
    np.testing.assert_array_equal(sampled.image,reference.values)
    np.testing.assert_array_equal(points.reshape(sampled.valid),reference.valid)
    mask = np.zeros(plane.shape,dtype=bool)
    mask[1,2] = mask[3,0] = True
    selected = sampled.select(mask)
    np.testing.assert_array_equal(selected.ids,[105,109])
    np.testing.assert_array_equal(selected.positions,points.positions[[5,9]])
    assert not points.positions.flags.writeable and sampled.source_identity is fields.value_identity
    assert selected.plane is points.plane
    batches = list(sm.PointSet.iter_plane(plane,batch_size=5))
    np.testing.assert_array_equal(np.concatenate([batch.ids for batch in batches]),np.arange(12))
    np.testing.assert_array_equal(np.concatenate([batch.positions for batch in batches]),points.positions)
    arbitrary = sm.PointSet([[.123,.345,.456]],ids=np.array([-7],dtype=np.int64))
    np.testing.assert_array_equal(app.sample(fields,arbitrary).values,sm.sample(fields,arbitrary.positions)[0])
    with pytest.raises(ValueError):
        sm.PointSet([[0,0,0],[1,1,1]],ids=np.array([1,1],dtype=np.int64))
    with pytest.raises(ValueError):
        sm.PointSet([[0,0,0]],ids=np.array([2**64-1],dtype=np.uint64))
    np.testing.assert_array_equal(sm.PointSet([[0,0,0]],ids=np.array([3],dtype=np.int32)).ids,[3])
    with pytest.raises(MemoryError):
        sm.PointSet.from_plane(plane,memory_limit=1)


def test_bottom_diagnostics_selection_and_separate_compact_traces():
    k,alpha = 1.2,.6
    decay = np.sqrt(k*k-alpha*alpha)
    with magnetic_source(lambda x,y,z: np.array([decay*np.cos(k*x),alpha*np.cos(k*x),-k*np.sin(k*x)])*
                         np.exp(-decay*z),mixed=True) as source:
        fields = sm.prepare(source,scheme="exact-phase")
    points = sm.PointSet.boundary(fields.mesh,"zmin",(8,4),ids=np.arange(200,232,dtype=np.int64))
    diagnostic = app.connectivity(fields,points,workers=2)
    assert diagnostic.image("q").shape == (8,4)
    selected = diagnostic.threshold(q_min=4.,abs_twist_min=.03)
    mask = (diagnostic.data.valid & (diagnostic.data.q >= 4.)) | (
        diagnostic.data.complete & np.isfinite(diagnostic.data.twist) & (np.abs(diagnostic.data.twist) >= .03))
    np.testing.assert_array_equal(selected.ids,points.ids[mask])
    assert len(selected) > 0
    lines = app.trace(fields,selected,direction="inward",step=.003,max_steps=2000,workers=2,seed_batch=4)
    np.testing.assert_array_equal(lines.seeds.ids,selected.ids)
    assert np.all(np.sum(lines.termination != -1,axis=1) == 1)
    assert np.all(lines.termination[lines.termination != -1] == sm.Termination.DOMAIN_EXIT)
    assert len(lines.positions) < len(selected)*2001
    assert np.isfinite(lines.positions).all()
    for index,seed_id in enumerate(selected.ids):
        side = int(np.flatnonzero(lines.termination[index] != -1)[0])
        branch = lines.branch(seed_id,2*side-1)
        assert len(branch) > 1
        np.testing.assert_array_equal(branch[0],selected.positions[index])
        assert len(lines.line(seed_id)) == len(branch)
    assert lines.source_identity is diagnostic.source_identity
    batches = list(app.iter_connectivity(fields,points,seed_batch=7,workers=2))
    np.testing.assert_array_equal(np.concatenate([b.points.ids for b in batches]),points.ids)
    np.testing.assert_allclose(np.concatenate([b.data.q for b in batches]),diagnostic.data.q,equal_nan=True)
    assert len(diagnostic.threshold(q_min=1e100,abs_twist_min=1e100)) == 0


def test_upper_boundary_and_tangent_direction_status():
    with magnetic_source(lambda x,y,z: np.array([np.zeros_like(x),np.zeros_like(x),np.ones_like(x)])) as source:
        fields = sm.prepare(source,scheme="exact-phase")
    points = sm.PointSet.boundary(fields.mesh,"zmax",(2,2))
    lines = app.trace(fields,points,direction="inward",step=.02,max_steps=100)
    np.testing.assert_array_equal(lines.termination[:,0],sm.Termination.DOMAIN_EXIT)
    np.testing.assert_array_equal(lines.termination[:,1],-1)
    for index in points.ids:
        np.testing.assert_array_equal(lines.branch(index,-1)[0],points.positions[index])
    tangent = sm.PointSet.boundary(fields.mesh,"xmin",(2,2))
    skipped = app.trace(fields,tangent,direction="inward",step=.02)
    np.testing.assert_array_equal(skipped.termination,-2)
    assert len(skipped.positions) == 0
    with pytest.raises(ValueError,match="physical box face"):
        app.trace(fields,sm.PointSet([[0,0,.5]]),direction="inward",step=.02)
    with pytest.raises(MemoryError):
        app.trace(fields,points,step=.02,memory_limit=1)
    empty = app.trace(fields,sm.PointSet(np.empty((0,3))),step=.02)
    assert empty.offsets.tolist() == [0]


def test_inward_weak_tangent_field_respects_null_threshold():
    with magnetic_source(lambda x,y,z: np.array([np.zeros_like(x),np.zeros_like(x),np.full_like(x,1e-9)])) as source:
        fields = sm.prepare(source,scheme="exact-phase")
    points = sm.PointSet.boundary(fields.mesh,"xmin",(1,1))
    lines = app.trace(fields,points,direction="inward",step=.02,null_threshold=1e-8)
    assert lines.termination[0,1] == sm.Termination.NULL_FIELD
    assert lines.termination[0,0] == sm.LineSet.NOT_REQUESTED
    assert len(lines.branch(points.ids[0],1)) == 1


@pytest.mark.parametrize("workers", [1,4])
def test_ray_set_matches_individual_plane_rays_and_keeps_ids(workers):
    with magnetic_source(lambda x,y,z: np.array([np.full_like(x,2.),x,y]),mixed=True) as source:
        fields = sm.prepare(source,scheme="exact-phase")
    points = sm.PointSet([[0,0,-1],[0,0,.5],[3,0,.5],[-2,0,.5]],
                         ids=np.array([11,9,27,2],dtype=np.int64),shape=(2,2))
    rays = sm.RaySet(points,[[0,0,2],[1,0,0],[1,0,0],[1,.2,0]],near=0.,far=np.inf)
    result = app.los(fields,rays,workers=workers,ray_batch=2)
    assert result.complete and result.image.shape == (2,2)
    for row,origin in enumerate(points.positions):
        plane = sm.Plane(origin-[.5,.5,0],[1,0,0],[0,1,0],(1,1))
        expected = sm.integrate_los(fields,plane,rays.directions[row])
        np.testing.assert_allclose(result.values[row],expected.values[0,0],rtol=2e-14)
        assert result.status[row] == expected.status[0,0]
    chosen = result.select(np.array([[True,False],[False,True]]))
    np.testing.assert_array_equal(chosen.origins.ids,[11,2])
    np.testing.assert_array_equal(chosen.directions,rays.directions[[0,3]])
    np.testing.assert_allclose(app.los(fields,chosen).values,result.values[[0,3]])
    clipped = sm.RaySet(points,rays.directions,near=np.zeros((2,2)),far=np.ones((2,2))*.5)
    assert np.all(app.los(fields,clipped).values <= result.values)
    with pytest.raises(MemoryError):
        app.los(fields,rays,memory_limit=1)


@pytest.mark.parametrize("order", ["thermodynamics-first","emissivity-first"])
def test_thermal_ray_set_preserves_existing_reconstruction(order):
    with magnetic_source(lambda x,y,z: np.array([np.ones_like(x),x,y])) as source:
        fields = sm.prepare(source,scheme="exact-phase")
    thermal = sm.thermal_fields(fields,1e6,density_unit_g_cm3=1e-15,temperature_label="test isothermal")
    plane = sm.Plane([-.5,-.5,-1],[1,0,0],[0,1,0],(4,3))
    rays = sm.RaySet.from_plane(plane,[.1,.2,1])
    result = app.thermal_los(thermal,rays,length_unit_cm=1e8,order=order,workers=3,ray_batch=5)
    expected = sm.integrate_thermal_los(thermal,plane,[.1,.2,1],length_unit_cm=1e8,order=order)
    np.testing.assert_allclose(result.image,expected.values,rtol=3e-14)
    np.testing.assert_array_equal(rays.origins.reshape(result.status),expected.status)
    assert result.metadata["length_unit_cm"] == 1e8
    varied = sm.RaySet(rays.origins,np.array([[.01*i,.1,1.] for i in range(len(rays.origins))]))
    varied_result = app.thermal_los(thermal,varied,length_unit_cm=1e8,order=order,workers=2)
    assert varied_result.complete and np.isfinite(varied_result.values).all()
    for row in (0,5,11):
        origin = varied.origins.positions[row]
        single = sm.Plane(origin-[.5,.5,0],[1,0,0],[0,1,0],(1,1))
        expected = sm.integrate_thermal_los(thermal,single,varied.directions[row],length_unit_cm=1e8,order=order)
        np.testing.assert_allclose(varied_result.values[row],expected.values[0,0],rtol=3e-14)


def test_ray_failure_status_and_empty_geometry():
    with magnetic_source(lambda x,y,z: np.array([np.ones_like(x),x,y]),mixed=True) as source:
        partial = sm.prepare(source,region=([-1,-1,0],[1,1,.2]),scheme="exact-phase")
    rays = sm.RaySet(sm.PointSet([[.1,.1,-1.]]),[0,0,1])
    result = app.los(partial,rays)
    assert result.status[0] == sm.LOSStatus.MISSING_COVERAGE and np.isnan(result.values[0])
    empty = sm.RaySet(sm.PointSet(np.empty((0,3))),[0,0,1])
    assert app.los(partial,empty).values.shape == (0,)
    with pytest.raises(ValueError):
        sm.RaySet(rays.origins,[0,0,0])


def test_nonfinite_sample_values_do_not_become_usable():
    mesh = sm.mesh_from_forest((1,1,1),np.array([True]),lower=(0,0,0),upper=(1,1,1),block_shape=(8,8,8))
    with sm.source_from_arrays(mesh,np.full((1,1,8,8,8),np.nan),("f",)) as source:
        fields = sm.prepare(source,scheme="exact-phase")
    points = sm.PointSet([[.5,.5,.5]])
    sampled = app.sample(fields,points)
    assert sampled.valid.all() and not sampled.usable.any()
    volume = app.uniform_grid(fields,(3,3,3))
    assert volume.valid.all() and not volume.usable.any()
