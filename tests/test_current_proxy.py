"""Native seed weights, closed-loop selection and display deposition checks."""
import numpy as np
import pytest
import simesh as sm
from simesh import applications as app
from simesh.current_proxy import _visited_cells


def arcade():
    mesh = sm.mesh_from_forest((1,1,1),np.array([True]),lower=(-2,-1,0),upper=(2,1,2),block_shape=(32,8,24))
    local = np.indices(mesh.block_shape)+.5
    x,y,z = mesh.lower[:,None,None,None]+local*mesh.spacing[0,:,None,None,None]
    # Concentric circles about an axis below zmin, with constant curl_y = 2.
    values = np.array([z+.5,np.zeros_like(y),-x])[None]
    return sm.source_from_arrays(mesh,values,('b1','b2','b3'))


def test_native_mixed_bottom_areas_and_centers():
    mesh = sm.mesh_from_forest((2,1,1),np.array([False]+[True]*9),lower=(0,0,0),upper=(2,1,1),block_shape=(4,4,4))
    points,area = sm.native_bottom_seeds(mesh)
    assert len(points)==80 and np.all(points.positions[:,2]==0)
    np.testing.assert_allclose(area.sum(),2)
    assert len(np.unique(area))==2
    assert len(np.unique(points.positions,axis=0))==len(points)


def test_closed_arcade_weights_batches_and_custom_seeds():
    with arcade() as source:
        field = sm.prepare(source,scheme='exact-phase')
    points = sm.PointSet([[-.9,0,0],[-1.3,.25,0]],ids=np.array([8,19]))
    options = dict(points=points,seed_areas=[.02,.04],step=.0125,max_steps=1000,workers=1)
    a = sm.current_proxy(field,(24,8,16),seed_batch=1,**options)
    b = sm.current_proxy(field,(24,8,16),seed_batch=2,**options)
    np.testing.assert_allclose(a.emissivity,b.emissivity,rtol=1e-14)
    np.testing.assert_array_equal(a.visits,b.visits)
    assert b.batches[0].closed.all()
    assert np.all(b.batches[0].endpoint_faces==1)
    # Boundary continuation perturbs curl near the photosphere; the interior
    # circular arc still provides a broad independent scale check.
    assert np.all((b.batches[0].mean_current_squared>3)&(b.batches[0].mean_current_squared<4.01))
    doubled = sm.current_proxy(field,(24,8,16),**{**options,'seed_areas':[.04,.08]})
    np.testing.assert_allclose(doubled.emissivity,2*a.emissivity)
    truncated = sm.current_proxy(field,(24,8,16),**{**options,'max_steps':2})
    assert not truncated.batches[0].closed.any() and not truncated.emissivity.any()
    with pytest.raises(ValueError,match='seed_areas'):
        sm.current_proxy(field,(24,8,16),points=points,step=.01)


def test_open_uniform_lines_make_no_proxy_and_default_seeds_work():
    mesh = sm.mesh_from_forest((1,1,1),np.array([True]),lower=(0,0,0),upper=(1,1,1),block_shape=(4,4,4))
    values = np.zeros((1,3,4,4,4)); values[:,2]=1
    with sm.source_from_arrays(mesh,values,('b1','b2','b3')) as source:
        field = sm.prepare(source,scheme='exact-phase')
    result = sm.current_proxy(field,(8,8,8),step=.05,max_steps=30,seed_batch=7)
    assert sum(len(batch.points) for batch in result.batches)==16
    assert not result.emissivity.any()
    assert all(not batch.closed.any() for batch in result.batches)
    assert all(np.all(batch.endpoint_faces[:,1]==2) for batch in result.batches)


def test_custom_interior_seed_traces_both_complete_branches():
    with arcade() as source:
        field = sm.prepare(source,scheme='exact-phase')
    result = sm.current_proxy(field,(24,8,16),points=sm.PointSet([[0,0,.5]],ids=np.array([91])),
                              seed_areas=[.03],step=.01,max_steps=1000)
    batch = result.batches[0]
    assert batch.closed[0]
    np.testing.assert_array_equal(batch.termination,[[int(sm.Termination.DOMAIN_EXIT)]*2])
    np.testing.assert_array_equal(batch.endpoint_faces,[[1,1]])
    assert 1.9<batch.length[0]<2.3
    assert result.emissivity.sum()>0


def test_voxel_clipping_corner_crossings_and_revisits():
    lo,hi,shape = np.zeros(3),np.ones(3)*2,np.array([2,2,2])
    path = np.array([[-.2,.5,.5],[.4,.5,.5],[1.2,.5,.5],[1.9,.5,.5],[2.2,.5,.5],[1.5,.5,.5],[.8,.5,.5]])
    cells = _visited_cells(path,lo,hi,shape)
    np.testing.assert_array_equal(cells,[0,4])
    corner = _visited_cells(np.array([[.5,.5,.5],[1.5,1.5,1.5]]),lo,hi,shape)
    np.testing.assert_array_equal(corner,[0,7])
    assert not len(_visited_cells(np.array([[.2,2.,.5],[.8,2.,.5]]),lo,hi,shape))


def test_reuse_loaded_paths_matches_convenience_flow_and_reweights(tmp_path):
    with arcade() as source:
        fields = sm.prepare(source,scheme='exact-phase')
    points = sm.PointSet([[-.9,0,0],[-1.3,.25,0]],ids=np.array([19,8]))
    lines = app.trace(fields,points,direction='inward',step=.0125,max_steps=1000)
    sm.save_result(tmp_path/'lines.npz',lines)
    loaded = sm.load_result(tmp_path/'lines.npz').result
    result = sm.current_proxy_from_lines(fields,loaded,(24,8,16),seed_areas=[.02,.04],step=.0125)
    direct = sm.current_proxy(fields,(24,8,16),points=points,seed_areas=[.02,.04],step=.0125,max_steps=1000)
    expected = np.zeros(direct.emissivity.size)
    expected[result.voxel_ids] = result.increments
    np.testing.assert_allclose(expected.reshape(direct.emissivity.shape),direct.emissivity,rtol=1e-14)
    np.testing.assert_array_equal(result.points.ids,[19,8])
    assert result.accepted.all()
    assert all(not hasattr(batch,'increments') for batch in direct.batches)
    selected = loaded.select(np.array([False,True]))
    assert len(selected.seeds)==1 and selected.seeds.ids[0]==8
    np.testing.assert_array_equal(selected.line(8),loaded.line(8))
    assert not np.shares_memory(selected.positions,loaded.positions)
    smaller = sm.current_proxy_from_lines(fields,selected,(12,4,8),seed_areas=[.08],step=.0125)
    base = sm.current_proxy_from_lines(fields,selected,(12,4,8),seed_areas=[.04],step=.0125)
    np.testing.assert_allclose(smaller.increments,2*base.increments)
    empty = loaded.select(np.array([False,False]))
    assert empty.positions.shape==(0,3) and empty.offsets.tolist()==[0]


def test_geometric_closure_survives_invalid_curl_samples():
    from dataclasses import replace
    with arcade() as source:
        fields = sm.prepare(source,scheme='exact-phase')
    lines = app.trace(fields,sm.PointSet([[-.9,0,0]]),direction='inward',step=.0125,max_steps=1000)
    curl = sm.curl(fields)
    invalid = replace(curl,_values=np.full_like(curl.values,np.nan))
    result = sm.current_proxy_from_lines(fields,lines,(24,8,16),seed_areas=[.02],step=.0125,curl_field=invalid)
    assert result.closed[0] and not result.accepted[0]
    assert np.isnan(result.mean_current_squared[0]) and len(result.voxel_ids)==0
    np.testing.assert_array_equal(result.termination,lines.termination)


def test_mixed_interior_and_boundary_seeds_keep_order():
    with arcade() as source:
        fields = sm.prepare(source,scheme='exact-phase')
    points = sm.PointSet([[0,0,.5],[-.9,0,0]],ids=np.array([7,3]))
    options = dict(points=points,seed_areas=[.02,.04],step=.01,max_steps=1000)
    joined = sm.current_proxy(fields,(24,8,16),seed_batch=2,**options)
    separate = sm.current_proxy(fields,(24,8,16),seed_batch=1,**options)
    np.testing.assert_array_equal(joined.batches[0].points.ids,[7,3])
    assert joined.batches[0].accepted.all()
    np.testing.assert_allclose(joined.emissivity,separate.emissivity,rtol=1e-14)
