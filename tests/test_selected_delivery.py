"""Selected I/O, valid support and direct/sharded delivery integration checks."""
from dataclasses import replace
import json

import numpy as np
import pytest
import simesh as sm
from simesh import applications as app
from fixtures import mixed_source, write_dat
from test_mhd_thermodynamics import source_for, model


@pytest.mark.parametrize('endian', ['<', '>'])
@pytest.mark.parametrize('partial', [False, True])
def test_final_storage_read_preserves_bits_and_order(tmp_path, endian, partial):
    source, raw = mixed_source()
    raw.view(np.uint64)[0, 0, 0, 0, 0] = 0x7ff8000000001234
    raw[1, 2, 0, 0, 0] = -0.
    path = tmp_path/'source.dat'
    write_dat(path, source.mesh, raw, byte_order=endian, saved_ghosts=True, staggered=True)
    ids = [7, 1, 0] if partial else np.arange(source.mesh.leaf_count)
    with sm.open_amrvac(path, fields=('b3','b1')) as selected:
        interior = sm.read_fields(selected, leaf_ids=ids)
        target = np.full((len(ids),12,12,12,2), 99.)
        selected.read_native_into(ids,[0,1],target,storage_halo=2)
        np.testing.assert_array_equal(target[:,2:-2,2:-2,2:-2].view('u8'),interior.values.view('u8'))
        assert np.all(target[:,:2] == 99.)
    expected = np.moveaxis(raw[np.asarray(ids)][:,[2,0]],1,-1)
    np.testing.assert_array_equal(interior.values.view('u8'), expected.view('u8'))
    assert interior.valid_halo == interior.storage_halo == 0
    assert interior.preparation_stats['publication_copy_bytes'] == 0


def test_mapping_cache_order_identity_budget_and_parent_lifetime():
    source, raw = mixed_source()
    mapped = sm.select_source(source, ('b3','b1'))
    assert mapped.nbytes == source.nbytes + 16
    assert mapped.identity is source.identity and mapped.field_origins == (2,0)
    with sm.cache_source(mapped, fields=('b1',), capacity=3) as cached:
        assert cached._reader.state.values.shape[1] == 1
        for _ in range(2):
            result = sm.read_fields(cached, leaf_ids=[1,0])
        assert cached.io_stats['value_cache_hits'] > 0
        np.testing.assert_array_equal(result.values[...,0], raw[[1,0],0])
        assert result.value_identity == sm.read_fields(source,('b1',),leaf_ids=[1,0]).value_identity
        source.close()
        with pytest.raises(OSError, match='closed'):
            sm.read_fields(cached,leaf_ids=[1])
    mapped.close()
    assert np.isfinite(result.values).all()


def test_source_subset_certificate_and_exact_plan():
    with mixed_source()[0] as source:
        mapped=sm.select_source(source,('b3','b1','b2'))
        subset=sm.select_source(mapped,('b1','b2','b3'))
        plan=sm.plan_preparation(source.mesh,scheme='exact-phase',leaf_ids=[5,0,8])
        planned=sm.prepare(subset,scheme='exact-phase',plan=plan)
        direct=sm.prepare(source,scheme='exact-phase',leaf_ids=[5,0,8])
        np.testing.assert_array_equal(planned.values,direct.values)
        assert planned.value_identity == direct.value_identity
        curl=sm.curl(direct)
        sm.trace(planned,[[.2,.2,.2]],step=.01,max_steps=1,twist=True,curl_field=curl)


def test_interior_quantities_and_common_support():
    with source_for()[0] as source:
        interior=sm.read_fields(source)
        ready=sm.prepare(source,scheme='exact-phase')
    a=sm.mhd_fields(interior,model=model(),outputs=('pressure','density'))
    b=sm.mhd_fields(ready,model=model(),outputs=('pressure','density'))
    np.testing.assert_array_equal(a.values,b.interior())
    assert a.valid_halo == 0
    assert sm.volume_integral(a,0).value == pytest.approx(sm.volume_integral(b,0).value)
    assert sm.extrema(a,0).minimum == sm.extrema(b,0).minimum
    derived=sm.derive(a,'twice',lambda ctx: 2*ctx.field('pressure'))
    merged=sm.merge_fields((derived, sm.select_fields(b,'density')))
    assert merged.valid_halo == 0
    thermal=sm.thermal_fields(sm.select_fields(interior,'rho'),1.e6,
        density_unit_g_cm3=1.e-15,temperature_label='isothermal')
    emissivity=sm.emissivity_fields(thermal)
    assert thermal.valid_halo == emissivity.valid_halo == 0
    assert np.isfinite(sm.volume_integral(emissivity).value)
    with pytest.raises(ValueError,match='current valid_halo=0.*1 valid halo'):
        sm.integrate_thermal_los(thermal,sm.Plane([0,0,0],[1,0,0],[0,1,0],(1,1)),[0,0,1],length_unit_cm=1.)


@pytest.mark.parametrize('allocated',[False,True])
def test_invalid_support_rejected_before_output_allocation(allocated, monkeypatch):
    with mixed_source()[0] as source:
        fields=sm.read_fields(source)
        if allocated:
            padded=np.zeros((len(fields.leaf_ids),12,12,12,3))
            padded[:,2:-2,2:-2,2:-2]=fields.values
            fields=replace(fields,_values=padded,storage_halo=2)
    plane=sm.Plane([0,0,.5],[2,0,0],[0,1,0],(10000,10000))
    consumers=[lambda: sm.sample(fields,[[.2,.2,.2]],memory_limit=1),
        lambda: sm.sample_plane(fields,plane,memory_limit=1),
        lambda: next(sm.iter_uniform(fields,(10000,10000,10000),memory_limit=1)),
        lambda: app.uniform_grid(fields,(10000,10000,10000),memory_limit=1),
        lambda: sm.curl(fields,memory_limit=1),
        lambda: sm.trace(fields,[[.2,.2,.2]],step=.1,memory_limit=1),
        lambda: sm.integrate_los(fields,plane,[0,0,1],memory_limit=1),
        lambda: app.field_map(fields,plane,memory_limit=1),
        lambda: app.bottom_diagnostics(fields,memory_limit=1),
        lambda: sm.qsl(fields,[[.2,.2,.2]],twist=False,memory_limit=1)]
    for consumer in consumers:
        with pytest.raises(ValueError,match='current valid_halo=0.*requires at least 1.*prepare'):
            consumer()


def test_derivative_support_and_categorical_selections():
    with mixed_source()[0] as source:
        ready=sm.prepare(source,scheme='exact-phase')
    first=sm.curl(ready)
    target=np.empty((len(first.leaf_ids),8,8,8,3))
    second=sm.curl(first,output=target)
    assert second.valid_halo == 0 and np.shares_memory(second.values,target)
    assert target.flags.writeable and not second.values.flags.writeable
    for invoke in (lambda: sm.sample(second,[[.2,.2,.2]]),
                   lambda: sm.qsl(second,[[.2,.2,.2]],twist=False),
                   lambda: sm.trace(first,[[.2,.2,.2]],step=.1,twist=True)):
        with pytest.raises(ValueError,match='valid_halo=.*requires at least'):
            invoke()
    categorical=replace(ready,fields=(sm.FieldDefinition('status',interpretation='categorical-node'),
                                    *ready.fields[1:]))
    with pytest.raises(ValueError,match='categorical'):
        sm.sample(categorical,[[.2,.2,.2]])
    sm.sample(categorical,[[.2,.2,.2]],components='b2')
    with pytest.raises(ValueError,match='categorical'):
        sm.integrate_los(categorical,sm.Plane([0,0,0],[2,0,0],[0,1,0],(1,1)),[0,0,1])


def test_selected_strided_sampling_and_caller_volume(tmp_path):
    with mixed_source()[0] as source:
        ready=sm.prepare(source,scheme='coordinate-phase')
    points=np.array([[.2,.3,.4],[1.5,.4,.3],[3.,.2,.3]])
    reference=sm.sample(ready,points)
    backing=np.full((3,4),-77.)
    target=(backing[:,::2],np.empty(6,dtype='i8')[::2],np.empty(6,dtype=bool)[::2])
    actual=sm.sample(ready,points,components=('b3','b1'),output=target)
    np.testing.assert_array_equal(actual[0],reference[0][:,[2,0]])
    assert np.shares_memory(actual[0],backing) and np.all(backing[:,1::2] == -77.)
    shape=(7,9,5)
    values=np.memmap(tmp_path/'volume.bin',mode='w+',dtype='f8',shape=(*shape,2))
    valid=np.zeros(shape,dtype=bool)
    result=app.uniform_grid(ready,shape,components=('b3','b1'),output=(values,valid),tile_rows=3,workers=2)
    assert result.values is values and result.valid is valid
    for iz,slab in sm.iter_uniform(ready,shape,components=('b3','b1'),tile_rows=2):
        np.testing.assert_array_equal(result.values[:,:,iz],slab.values)
        np.testing.assert_array_equal(result.valid[:,:,iz],slab.valid)
    plane=sm.Plane([0,0,.5],[2,0,0],[0,1,0],(5,7))
    target=(np.empty((5,7,1)),np.empty((5,7),dtype='i8'),np.empty((5,7),dtype=bool))
    sm.sample_plane(ready,plane,components='b2',output=target)
    np.testing.assert_array_equal(target[0],sm.sample_plane(ready,plane).values[...,1:2])


def test_caller_output_partial_failure_is_not_published(monkeypatch):
    import simesh.operators.sampling as sampling
    with mixed_source()[0] as source:
        ready=sm.prepare(source,scheme='exact-phase')
    values=np.full((3,4,2,3),-123.)
    valid=np.zeros((3,4,2),dtype=bool)
    real=sampling._sample
    calls=0
    def fail(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:
            raise RuntimeError('injected failure')
        return real(*args,**kwargs)
    monkeypatch.setattr(sampling,'_sample',fail)
    with pytest.raises(RuntimeError,match='injected'):
        app.uniform_grid(ready,(3,4,2),output=(values,valid),tile_rows=1)
    assert np.any(values!=-123.) and np.any(values==-123.)


def test_lines_profiles_shards_roundtrip_and_failed_manifest(tmp_path):
    with mixed_source(lambda x,y,z: np.array([0*x,0*y,1+0*z]))[0] as source:
        fields=sm.prepare(source,scheme='exact-phase')
    points=sm.PointSet(np.array([[.2,.3,.4],[.6,.4,.5],[1.2,.3,.4]]),np.array([19,2,88],dtype='i8'))
    controls=dict(step=.03,max_steps=4)
    full=app.trace(fields,points,**controls)
    batches=app.iter_lines(fields,points,seed_batch=2,**controls)
    result=sm.save_result_shards(tmp_path/'lines',batches,seed_ids=points.ids)
    assert result.complete and len(result)==2
    for index,(start,stop) in enumerate(((0,2),(2,3))):
        loaded=result.load(index).result
        np.testing.assert_array_equal(loaded.seeds.ids,points.ids[start:stop])
        np.testing.assert_array_equal(loaded.termination,full.termination[start:stop])
        for seed in loaded.seeds.ids:
            for direction in (-1,1):
                np.testing.assert_array_equal(loaded.branch(seed,direction),full.branch(seed,direction))
    profiles=sm.iter_line_profiles(fields,(result.load(i).result for i in range(len(result))),('b3',),point_batch=3)
    saved=sm.save_result_shards(tmp_path/'profiles',profiles,seed_ids=points.ids)
    loaded=saved.load(1).result
    assert loaded.definitions == (fields.fields[2],) and loaded.component_indices==(2,)
    assert loaded.source_identity is None and loaded.line_source_identity is None
    np.testing.assert_allclose(loaded.values,1.)
    assert loaded.length_units == sm.LengthUnits(1.,'coordinate-length')
    assert not loaded.values.flags.writeable
    def broken():
        yield result.load(0).result
        raise RuntimeError('producer failed')
    with pytest.raises(RuntimeError,match='producer failed'):
        sm.save_result_shards(tmp_path/'broken',broken(),seed_ids=points.ids)
    partial=sm.open_result_shards(tmp_path/'broken')
    assert not partial.complete and len(partial)==1
    partial.load(0)
    path=tmp_path/'profiles'/'shard-000001.npz'
    with path.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(sm.ResultFileError,match='checksum'):
        saved.load(1)


def test_selected_cache_rebinding_and_file_mutation(tmp_path):
    source, raw=mixed_source()
    path=tmp_path/'cache.dat'
    write_dat(path,source.mesh,raw)
    with sm.open_amrvac(path,fields=('b3','b1')) as source:
        with sm.cache_source(source,capacity=2) as cache:
            for order in ((0,1),(1,0),(1,)):
                result=sm.read_fields(cache,order,leaf_ids=[0,1])
                expected=np.moveaxis(raw[:2][:,np.array([2,0])[list(order)]],1,-1)
                np.testing.assert_array_equal(result.values,expected)
            with path.open('ab') as stream:
                stream.write(b'changed')
            with pytest.raises(OSError,match='changed'):
                sm.read_fields(cache,(1,),leaf_ids=[0,1])


def test_native_read_failure_never_publishes(monkeypatch):
    import importlib
    module=importlib.import_module('simesh.io.source')
    with mixed_source()[0] as source:
        def broken(ids,components,target):
            target[0]=5.
            raise OSError('failed native read')
        source._read_native=broken
        def publish_forbidden(*args,**kwargs):
            raise AssertionError('partial read must not be published')
        monkeypatch.setattr(module,'publish',publish_forbidden)
        with pytest.raises(OSError,match='failed native'):
            sm.read_fields(source)
        output=np.full((2,8,8,8,1),-1.)
        with pytest.raises(OSError,match='failed native'):
            source.read_native_into([0,1],[0],output)
        assert np.all(output[0]==5.) and np.all(output[1]==-1.)


@pytest.mark.parametrize('corruption',['gap','complete','path','checksum','version'])
def test_shard_manifest_rejects_false_completeness_and_bad_ranges(tmp_path,corruption):
    data=dict(format='simesh-result-shards',schema_version=1,complete=False,
        seed_ids=[7,3],metadata={},source=None,
        shards=[dict(file='shard-000000.npz',start=0,stop=1,kind='LineSet',sha256='0'*64)])
    if corruption=='gap':
        data['shards'][0]['start']=1
    elif corruption=='complete':
        data['complete']=True
    elif corruption=='path':
        data['shards'][0]['file']='../outside.npz'
    elif corruption=='checksum':
        data['shards'][0]['sha256']='z'*64
    else:
        data['schema_version']=True
    (tmp_path/'manifest.json').write_text(json.dumps(data))
    with pytest.raises(sm.ResultFileError):
        sm.open_result_shards(tmp_path)


def test_shard_delivery_freezes_requested_ids(tmp_path):
    ids=np.array([10,20],dtype=np.int64)
    def batches():
        for row in range(2):
            if row:
                ids[row]=99
            seeds=sm.PointSet([[.2,.2,.2]],ids[row:row+1])
            yield sm.LineSet(seeds,seeds.positions,np.array([0,0,1],dtype=np.int64),
                             np.array([[-1,1]],dtype=np.int64),None)
    with pytest.raises(ValueError,match='seed IDs'):
        sm.save_result_shards(tmp_path/'shards',batches(),seed_ids=ids)
    saved=sm.open_result_shards(tmp_path/'shards')
    assert not saved.complete
    np.testing.assert_array_equal(saved.seed_ids,[10,20])
    np.testing.assert_array_equal(saved.load(0).result.seeds.ids,[10])


def test_volume_rejects_output_aliasing_bounds():
    with mixed_source()[0] as source:
        fields=sm.prepare(source,scheme='exact-phase')
    values=np.zeros((2,2,2,1))
    values.ravel()[:6]=[0,0,0,2,1,1]
    before=values.copy()
    with pytest.raises(ValueError,match='alias'):
        app.uniform_grid(fields,(2,2,2),components='b1',
            bounds=(values.ravel()[:3],values.ravel()[3:6]),
            output=(values,np.empty((2,2,2),dtype=bool)),tile_rows=1)
    np.testing.assert_array_equal(values,before)


def test_one_shot_component_selectors_survive_maps_and_shards():
    with mixed_source()[0] as source:
        fields=sm.prepare(source,scheme='exact-phase')
    seeds=sm.PointSet([[.2,.2,.2],[.3,.3,.3]])
    mapped=app.field_map(fields,seeds,components=iter(('b3','b1')))
    np.testing.assert_array_equal(mapped.values,sm.sample(fields,seeds.positions,components=('b3','b1'))[0])
    batches=app.iter_lines(fields,seeds,step=.01,max_steps=2,seed_batch=1)
    results=list(sm.iter_line_profiles(fields,batches,components=iter(('b3',))))
    assert len(results)==2
    for result in results:
        assert result.component_indices==(2,)
        np.testing.assert_array_equal(result.values,sm.sample(fields,result.lines.positions,components='b3')[0])


def test_thermal_common_support_allows_temperature_coverage_superset():
    with mixed_source()[0] as source:
        ready=sm.prepare(source,('b1',),scheme='exact-phase')
        region=sm.prepare(source,('b1',),leaf_ids=[2,0],scheme='exact-phase')
    density=replace(region,valid_halo=1)
    temperature=sm.derive(ready,'temperature',lambda ctx: 1.e6,units='K')
    thermal=sm.thermal_fields(density,temperature,density_unit_g_cm3=1.e-15,
                             temperature_label='constant temperature')
    assert thermal.valid_halo==thermal.storage_halo==1
    np.testing.assert_array_equal(thermal.leaf_ids,[2,0])
    np.testing.assert_array_equal(thermal.values[...,1],1.e6)
