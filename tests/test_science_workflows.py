"""Independent science and ownership across retained and bounded N3 consumers."""

import gc
import weakref
import numpy as np
import pytest

import simesh as sm
from simesh.bounded import (PreparedPool, CurlPool, trace_bounded, integrate_los_bounded,
                            integrate_los_views_bounded, iter_uniform_bounded)
from fixtures import mixed_source, write_dat


def helical_source():
    m=sm.mesh_from_forest((2,2,2),np.ones(8,dtype=bool),lower=(-2,-2,0),upper=(2,2,4),block_shape=(8,8,8))
    values=np.empty((8,3,8,8,8))
    local=np.indices((8,8,8))+.5
    for leaf in range(8):
        x,y,z=m.bounds[leaf,0,:,None,None,None]+local*m.spacing[leaf,:,None,None,None]
        values[leaf]=[-y,x,np.full_like(z,.4)]
    return sm.source_from_arrays(m,values,('b1','b2','b3'))


def test_helical_twist_retrace_and_bounded_stage_continuation():
    source=helical_source()
    primary=sm.prepare(source,scheme='exact-phase')
    curl=sm.curl(primary)
    angles=np.linspace(0,2*np.pi,8,endpoint=False)
    radius=.7
    seeds=np.ascontiguousarray(np.column_stack((radius*np.cos(angles),radius*np.sin(angles),np.ones(8))))
    one=sm.trace(primary,seeds,step=.01,max_steps=200,twist=True,curl_field=curl)
    four=sm.trace(primary,seeds,step=.01,max_steps=200,twist=True,curl_field=curl,
                  workers=4,trajectories=True)
    np.testing.assert_array_equal(one.positions,four.positions)
    np.testing.assert_array_equal(one.twist,four.twist)
    expected=.8/(4*np.pi*(radius*radius+.16))*one.length
    np.testing.assert_allclose(one.twist,expected,rtol=0,atol=2e-10)
    selected=np.array([6,1],dtype=np.int64)
    again=sm.retrace(primary,one,selected,step=.01,max_steps=200,twist=True,curl_field=curl)
    np.testing.assert_array_equal(again.positions,one.positions[selected])
    np.testing.assert_array_equal(again.trajectories,four.trajectories[selected])
    with PreparedPool(source,capacity=2,scheme='exact-phase') as pool:
        with CurlPool(pool) as coupled:
            bounded=trace_bounded(coupled,seeds,step=.01,max_steps=200,twist=True,
                                   trajectories=True,workers=4,seed_batch=2)
            np.testing.assert_array_equal(bounded.positions,one.positions)
            np.testing.assert_array_equal(bounded.twist,one.twist)
            np.testing.assert_array_equal(bounded.trajectories,four.trajectories)
            assert coupled.derived_count>2
    zero=sm.trace(primary,seeds,step=.01,max_steps=0,twist=True)
    np.testing.assert_array_equal(zero.twist,np.zeros(8))


def test_retained_curl_matches_logical_fields_without_pinning_primary():
    source,_=mixed_source()
    primary=sm.prepare(source,scheme='exact-phase')
    global_result=sm.global_curl(source,batch_size=2)
    np.testing.assert_array_equal(global_result.values,sm.curl(primary).values)
    sm.trace(primary,np.array([[.75,.25,.25]]),step=.001,max_steps=4,twist=True,curl_field=global_result)
    wrong=sm.curl(primary,components=(1,2,0))
    with pytest.raises(ValueError,match='derive from'):
        sm.trace(primary,np.array([[.75,.25,.25]]),step=.001,max_steps=4,twist=True,curl_field=wrong)
    ref=weakref.ref(primary.values)
    retained=sm.curl(primary)
    del primary,wrong
    source.close()
    gc.collect()
    assert ref() is None
    assert np.isfinite(retained.values).all()


def test_scalar_los_views_depths_and_pool_eviction():
    source,_=mixed_source(lambda x,y,z:np.full((3,*x.shape),2.))
    ready=sm.prepare(source,['b1'],scheme='exact-phase')
    directions=([0,0,1],[.3,.2,1])
    planes=[sm.orthographic_plane(source.mesh.lower,source.mesh.upper,d,(10,8)) for d in directions]
    images=sm.integrate_los_views(ready,planes,directions,workers=4)
    for image in images:
        assert image.complete
        np.testing.assert_allclose(image.values,2*image.depth,rtol=0,atol=4e-13)
    with PreparedPool(source,['b1'],capacity=2,scheme='exact-phase') as pool:
        bounded=integrate_los_views_bounded(pool,planes,directions,workers=4)
        for a,b in zip(images,bounded):
            np.testing.assert_array_equal(a.values,b.values)
            np.testing.assert_array_equal(a.status,b.status)
    limited=sm.integrate_los(ready,planes[0],directions[0],max_samples=1)
    assert not limited.complete
    assert np.isnan(limited.values[~limited.valid]).all()


def test_explicit_thermal_models_and_two_reconstruction_orders():
    source,raw=mixed_source(lambda x,y,z:np.array([2+.1*x,np.ones_like(x),np.ones_like(x)]))
    density=sm.prepare(source,['b1'],scheme='exact-phase')
    temperature=np.empty((source.mesh.leaf_count,1,*source.mesh.block_shape))
    local=np.indices(source.mesh.block_shape)+.5
    for leaf in range(source.mesh.leaf_count):
        y=source.mesh.bounds[leaf,0,1]+local[1]*source.mesh.spacing[leaf,1]
        temperature[leaf,0]=.8e6+.4e6*y*y
    with sm.source_from_arrays(source.mesh,temperature,['T'],units='K') as thermal_source:
        t=sm.prepare(thermal_source,scheme='exact-phase')
    thermo=sm.thermal_fields(density,t,density_unit_g_cm3=1e-15,temperature_label='manufactured test')
    direction=[.3,.2,1]
    plane=sm.orthographic_plane(source.mesh.lower,source.mesh.upper,direction,(8,8))
    native=sm.integrate_thermal_los(thermo,plane,direction,length_unit_cm=1e8,workers=4)
    reference=sm.integrate_thermal_los(thermo,plane,direction,length_unit_cm=1e8,implementation='reference')
    assert native.complete and reference.complete
    np.testing.assert_allclose(native.values,reference.values,rtol=1e-10,atol=1e-10)
    epsilon=sm.emissivity_fields(thermo)
    alternate=sm.integrate_thermal_los(thermo,plane,direction,length_unit_cm=1e8,order='emissivity-first')
    scalar=sm.integrate_los(epsilon,plane,direction)
    np.testing.assert_array_equal(alternate.values,scalar.values*1e8)
    assert not np.allclose(alternate.values,native.values,rtol=1e-6,atol=1e-6)
    with pytest.raises(ValueError,match='matching physical'):
        sm.integrate_thermal_los(thermo,plane,direction,length_unit_cm=1e8,
                                 model=sm.AIA171(density_convention='amrvac-hydrogen'))


def test_geometric_plan_reuses_geometry_but_recomputes_values_and_cache_lifetime():
    source,raw=mixed_source(lambda x,y,z:np.array([np.sin(x)+y,np.cos(y)+z,np.sin(x*z)]))
    ids=[8,2,1,5]
    plan=sm.plan_preparation(source.mesh,leaf_ids=ids,scheme='exact-phase')
    cached=sm.cache_source(source,capacity=9)
    storage=weakref.ref(cached._reader.state.values)
    first=sm.prepare(cached,['b3','b1'],scheme='exact-phase',plan=plan)
    second=sm.prepare(cached,['b3','b1'],scheme='exact-phase',plan=plan)
    assert second.preparation_stats['read_value_bytes']==0
    np.testing.assert_array_equal(first.values,second.values)
    single=sm.prepare(cached,['b2'],scheme='exact-phase',plan=plan)
    np.testing.assert_array_equal(single.values,sm.prepare(source,['b2'],leaf_ids=ids,scheme='exact-phase').values)
    with sm.source_from_arrays(source.mesh,raw*1.25,['b1','b2','b3']) as changed:
        planned=sm.prepare(changed,['b3','b1'],scheme='exact-phase',plan=plan)
        direct=sm.prepare(changed,['b3','b1'],leaf_ids=ids,scheme='exact-phase')
        np.testing.assert_array_equal(planned.values,direct.values)
    source.close()
    with pytest.raises(OSError,match='closed'):
        sm.prepare(cached,['b2'],scheme='exact-phase',plan=plan)
    cached.close()
    gc.collect()
    assert storage() is None
    assert np.isfinite(first.values).all()


def test_streamed_owned_slabs_and_file_task_curl(tmp_path):
    source,values=mixed_source()
    ready=sm.prepare(source,scheme='exact-phase')
    slabs=list(sm.iter_uniform(ready,(12,10,4),workers=4))
    with PreparedPool(source,capacity=2,scheme='exact-phase') as pool:
        bounded=list(iter_uniform_bounded(pool,(12,10,4),workers=4))
    for (i,a),(j,b) in zip(slabs,bounded):
        assert i==j
        np.testing.assert_array_equal(a.values,b.values)
        np.testing.assert_array_equal(a.valid,b.valid)
    assert not np.shares_memory(slabs[0][1].values,slabs[1][1].values)
    path=tmp_path/'file.dat'
    write_dat(path,source.mesh,values,staggered=True)
    expected=sm.curl(ready)
    for backend in ('thread','process'):
        actual=sm.global_curl_file(path,backend=backend,workers=2,task_size=4,batch_size=2)
        np.testing.assert_array_equal(actual.values,expected.values)


def test_openmp_scientific_consumers_when_available():
    from simesh._kernels.native import openmp_build_info
    if not openmp_build_info()['enabled']:
        pytest.skip('ordinary build')
    source=helical_source()
    ready=sm.prepare(source,scheme='exact-phase')
    seeds=np.array([[.7,.1,1.],[.6,.3,1.]])
    reference=sm.trace(ready,seeds,step=.01,max_steps=50,twist=True)
    actual=sm.trace(ready,seeds,step=.01,max_steps=50,twist=True,backend='openmp',workers=4)
    np.testing.assert_array_equal(actual.twist,reference.twist)
    np.testing.assert_array_equal(actual.positions,reference.positions)
    plane=sm.orthographic_plane(ready.mesh.lower,ready.mesh.upper,[.3,.2,1],(8,8))
    a=sm.integrate_los(ready,plane,[.3,.2,1],component=2)
    b=sm.integrate_los(ready,plane,[.3,.2,1],component=2,backend='openmp',workers=4)
    np.testing.assert_array_equal(a.values,b.values)
    with sm.source_from_arrays(ready.mesh,np.ones((ready.mesh.leaf_count,1,*ready.mesh.block_shape)),['rho']) as density_source:
        density=sm.prepare(density_source,scheme='exact-phase')
    thermal=sm.thermal_fields(density,1e6,density_unit_g_cm3=1e-15,temperature_label='isothermal test')
    a=sm.integrate_thermal_los(thermal,plane,[.3,.2,1],length_unit_cm=1e8)
    b=sm.integrate_thermal_los(thermal,plane,[.3,.2,1],length_unit_cm=1e8,workers=4,backend='openmp')
    np.testing.assert_array_equal(a.values,b.values)
