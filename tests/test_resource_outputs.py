"""Resource fixes preserve accepted paths and recoverable shard publication."""

import json
import tracemalloc

import numpy as np
import pytest
import simesh as sm
from simesh import applications as app, result_shards
from simesh.tracing import _PATH_SEGMENT_STEPS
from fixtures import mixed_source


def vector_fields(*, subset=False, null=False):
    def vector(x,y,z):
        return np.array([.1*y,.2*x,1.+0*z]) * (0. if null else 1.)
    with mixed_source(vector)[0] as source:
        return sm.prepare(source,scheme='exact-phase',leaf_ids=[0] if subset else None)


@pytest.mark.parametrize('limits', [
    dict(max_steps=0), dict(max_steps=1),
    dict(max_steps=_PATH_SEGMENT_STEPS), dict(max_steps=_PATH_SEGMENT_STEPS+1),
    dict(max_steps=1000), dict(max_steps=1000,max_length=0.),
    dict(max_steps=1000,max_length=.33333),
])
@pytest.mark.parametrize('workers,schedule', [(1,'static'),(2,'dynamic')])
def test_segments_match_dense_accepted_paths(limits,workers,schedule):
    fields=vector_fields()
    points=sm.PointSet([[.2,.2,.2],[.3,.2,.95],[1.4,.4,.6],[3.,.2,.2]],ids=[9,-2,0,83])
    options=dict(step=.0017,schedule=schedule,workers=workers,**limits)
    actual=app.trace(fields,points,seed_batch=3,**options)
    for side in (-1,1):
        expected=sm.trace(fields,points.positions,seed_ids=points.ids,trajectories=True,direction=side,**options)
        np.testing.assert_array_equal(actual.termination[:,(side+1)//2],expected.termination)
        for row,seed_id in enumerate(points.ids):
            np.testing.assert_array_equal(actual.branch(seed_id,side),
                expected.trajectories[row,:expected.point_counts[row]])
    shards=list(app.iter_lines(fields,points,seed_batch=2,**options))
    for shard in shards:
        for seed_id in shard.seeds.ids:
            for side in (-1,1):
                np.testing.assert_array_equal(shard.branch(seed_id,side),actual.branch(seed_id,side))


@pytest.mark.parametrize('subset,null', [(True,False),(False,True)])
def test_segment_termination_for_missing_and_null_fields(subset,null):
    fields=vector_fields(subset=subset,null=null)
    points=sm.PointSet([[.2,.2,.2]])
    actual=app.trace(fields,points,direction='along',step=.001,max_steps=1000)
    expected=sm.trace(fields,points.positions,step=.001,max_steps=1000,trajectories=True)
    assert expected.termination[0] == (sm.Termination.NULL_FIELD if null else sm.Termination.MISSING_COVERAGE)
    np.testing.assert_array_equal(actual.termination[:,1],expected.termination)
    np.testing.assert_array_equal(actual.branch(0,1),expected.trajectories[0,:expected.point_counts[0]])


def test_short_trace_memory_is_independent_of_max_steps():
    fields=vector_fields()
    points=sm.PointSet(np.tile([.2,.2,.5],(32,1)))
    peaks=[]
    results=[]
    for limit in (1000,10**8):
        tracemalloc.start()
        try:
            # This budget cannot admit a dense path array for the larger limit.
            result=app.trace(fields,points,step=.03,max_steps=limit,
                memory_limit=fields.mesh.nbytes+fields.nbytes+1_000_000)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        results.append(result)
    np.testing.assert_array_equal(results[0].positions,results[1].positions)
    np.testing.assert_array_equal(results[0].termination,results[1].termination)
    assert peaks[1] < peaks[0]+65536
    # Actual long output is still subject to the budget; no silent truncation.
    with pytest.raises(MemoryError):
        app.trace(fields,points,step=.00001,max_steps=100000,
            memory_limit=fields.mesh.nbytes+fields.nbytes+300_000)


def empty_batch(ids):
    points=sm.PointSet(np.full((len(ids),3),.5),ids)
    return sm.LineSet(points,np.empty((0,3)),np.zeros(2*len(ids)+1,dtype=np.int64),
                      np.full((len(ids),2),-1,dtype=np.int64),None)


@pytest.mark.parametrize('version',[1,2])
def test_shard_versions_preserve_extreme_ids_and_metadata(tmp_path,version):
    ids=np.array([-(2**63),2**63-1,0],dtype=np.int64)
    path=tmp_path/'shards'
    saved=sm.save_result_shards(path,[empty_batch(ids[:2]),empty_batch(ids[2:])],
        seed_ids=ids,metadata={'title':'measurement'},source={'run':7})
    if version==1:
        # Construct the prior on-disk protocol around the same registered NPZs.
        data=saved.manifest.copy()
        data.update(schema_version=1,seed_ids=ids.tolist())
        (path/'manifest.json').write_text(json.dumps(data))
    opened=sm.open_result_shards(path)
    assert opened.complete and len(opened)==2
    assert opened.manifest['metadata']=={'title':'measurement'}
    np.testing.assert_array_equal(opened.seed_ids,ids)
    np.testing.assert_array_equal(opened.load(0).result.seeds.ids,ids[:2])
    np.testing.assert_array_equal(opened.load(1).result.seeds.ids,ids[2:])
    assert not opened.seed_ids.flags.writeable


def test_empty_shards_and_incomplete_v1(tmp_path):
    saved=sm.save_result_shards(tmp_path/'empty',[],seed_ids=np.empty(0,dtype=np.int64))
    opened=sm.open_result_shards(saved.path)
    assert opened.complete and len(opened)==0 and len(opened.seed_ids)==0
    path=tmp_path/'v1'
    path.mkdir()
    (path/'manifest.json').write_text(json.dumps(dict(format='simesh-result-shards',
        schema_version=1,complete=False,seed_ids=[4],shards=[],metadata={},source=None)))
    assert not sm.open_result_shards(path).complete


@pytest.mark.parametrize('failure',['producer','index','summary'])
def test_interrupted_v2_exposes_only_committed_shards(tmp_path,monkeypatch,failure):
    ids=np.array([7,2],dtype=np.int64)
    original=result_shards._write_json
    def fail_write(path,data):
        if ((failure=='index' and path.name=='index-000001.json') or
                (failure=='summary' and data.get('complete'))):
            raise OSError('interrupted')
        original(path,data)
    def batches():
        yield empty_batch(ids[:1])
        if failure=='producer':
            raise OSError('interrupted')
        yield empty_batch(ids[1:])
    monkeypatch.setattr(result_shards,'_write_json',fail_write)
    path=tmp_path/'shards'
    with pytest.raises(OSError,match='interrupted'):
        sm.save_result_shards(path,batches(),seed_ids=ids)
    partial=sm.open_result_shards(path)
    assert not partial.complete and len(partial)==(2 if failure=='summary' else 1)
    for index in range(len(partial)):
        partial.load(index)


@pytest.mark.parametrize('corruption',['ids','gap','range','kind','checksum','missing_payload'])
def test_v2_rejects_corrupt_identity_and_indices(tmp_path,corruption):
    ids=np.array([7,2],dtype=np.int64)
    path=tmp_path/'shards'
    def batches():
        yield empty_batch(ids[:1])
        yield empty_batch(ids[1:])
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):
        sm.save_result_shards(path,batches(),seed_ids=ids)
    if corruption=='ids':
        np.save(path/'seed_ids.npy',ids[::-1])
    elif corruption=='gap':
        (path/'index-000000.json').unlink()
    elif corruption=='missing_payload':
        (path/'seed_ids.npy').unlink()
    else:
        index=path/'index-000001.json'
        entry=json.loads(index.read_text())
        entry.update({'range':{'start':0},'kind':{'kind':'wrong'},'checksum':{'sha256':'bad'}}[corruption])
        index.write_text(json.dumps(entry))
    with pytest.raises(sm.ResultFileError):
        sm.open_result_shards(path)


def test_shard_metadata_written_bytes_scale_linearly(tmp_path,monkeypatch):
    original=result_shards._write_json
    sizes=[]
    def measure(path,data):
        original(path,data)
        sizes.append(path.stat().st_size)
    monkeypatch.setattr(result_shards,'_write_json',measure)
    totals=[]
    for count in (20,40):
        ids=np.arange(count*10,dtype=np.int64)
        sizes.clear()
        path=tmp_path/str(count)
        sm.save_result_shards(path,(empty_batch(ids[i:i+10]) for i in range(0,len(ids),10)),seed_ids=ids)
        totals.append(sum(sizes)+(path/'seed_ids.npy').stat().st_size)
    assert totals[1] < 2.2*totals[0]
