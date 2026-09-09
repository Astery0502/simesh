"""Selective diagnostics and reduced save copies preserve scientific contracts."""

from dataclasses import replace
import tracemalloc

import numpy as np
import pytest
import simesh as sm
from simesh import applications as app, results_io as rio
from simesh.physics.mhd import _OUTPUTS, MHDStatus, MHDStateError
from test_mhd_thermodynamics import model, source_for


@pytest.mark.parametrize('energy_kind',['total','internal'])
def test_selected_mhd_outputs_match_complete_recovery(energy_kind):
    config=model(energy_kind)
    with source_for(configuration=config,mixed=True)[0] as source:
        fields=sm.prepare(source,scheme='coordinate-phase')
    full=sm.mhd_fields(fields,model=config,outputs=tuple(_OUTPUTS))
    columns={definition.name:i for i,definition in enumerate(full.fields)}
    for name in _OUTPUTS:
        selected=sm.mhd_fields(fields,model=config,outputs=name)
        expected=full.values[...,[columns[definition.name] for definition in selected.fields]]
        np.testing.assert_array_equal(selected.values,expected)
        assert selected.preparation_stats['invalid_state_counts']==full.preparation_stats['invalid_state_counts']


def test_mhd_diagnostic_scope_does_not_report_unchecked_as_passing():
    with source_for()[0] as source:
        fields=sm.read_fields(source)
    values=fields.values.copy()
    values[...,5:8]=1e-200  # Finite state; beta overflows because magnetic pressure underflows.
    fields=replace(fields,_values=values)
    config=model('internal')
    velocity=sm.mhd_fields(fields,model=config,outputs='velocity')
    full=sm.mhd_fields(fields,model=config,outputs='status')
    beta=sm.mhd_fields(fields,model=config,outputs='beta')
    mach=sm.mhd_fields(fields,model=config,outputs='alfven_mach')
    sonic=sm.mhd_fields(fields,model=config,outputs='sonic_mach')
    np.testing.assert_array_equal(full.values,int(MHDStatus.UNREPRESENTABLE_DIAGNOSTIC))
    assert velocity.preparation_stats['evaluated_diagnostics']==()
    assert beta.preparation_stats['evaluated_diagnostics']==('beta',)
    assert mach.preparation_stats['evaluated_diagnostics']==('alfven_speed','alfven_mach')
    assert sonic.preparation_stats['evaluated_diagnostics']==('sound_speed','sonic_mach')
    for region in ('interior','evaluated'):
        assert velocity.preparation_stats['status_counts'][region]['UNREPRESENTABLE_DIAGNOSTIC'] is None
        assert beta.preparation_stats['status_counts'][region]['UNREPRESENTABLE_DIAGNOSTIC']>0
        assert mach.preparation_stats['status_counts'][region]['UNREPRESENTABLE_DIAGNOSTIC']==0
    assert np.isnan(beta.values).all() and np.isfinite(mach.values).all()


@pytest.mark.parametrize('output',['density','velocity','beta'])
@pytest.mark.parametrize('component,value',[(0,0.),(4,-1.),(6,np.nan)])
def test_selected_mhd_still_checks_complete_physical_state(output,component,value):
    with source_for()[0] as source:
        fields=sm.read_fields(source)
    values=fields.values.copy()
    values[0,1,2,3,component]=value
    fields=replace(fields,_values=values)
    with pytest.raises(MHDStateError) as expected:
        sm.mhd_fields(fields,model=model())
    with pytest.raises(MHDStateError) as actual:
        sm.mhd_fields(fields,model=model(),outputs=output)
    assert actual.value.cell_index==expected.value.cell_index
    assert actual.value.status==expected.value.status
    masked=sm.mhd_fields(fields,model=model(),outputs=output,invalid='nan')
    assert np.isnan(masked.values[0,1,2,3]).all()
    assert masked.preparation_stats['invalid_state_counts']=={'interior':1,'evaluated':1}


def test_mhd_keeps_raw_dependency_before_nonfinite_normalization():
    config=replace(model('internal'),gamma=1e308)
    with source_for()[0] as source:
        fields=sm.read_fields(source)
    values=np.zeros_like(fields.values)
    values[...,0]=1e-310
    values[...,4]=1e-306
    fields=replace(fields,_values=values)
    result=sm.mhd_fields(fields,model=config,outputs=('sound_speed','sonic_mach','status'))
    assert np.isnan(result.values[...,0]).all()
    np.testing.assert_array_equal(result.values[...,1],0.)
    np.testing.assert_array_equal(result.values[...,2],
        int(MHDStatus.ZERO_MAGNETIC_FIELD|MHDStatus.UNREPRESENTABLE_DIAGNOSTIC))
    selected=sm.mhd_fields(fields,model=config,outputs='sonic_mach')
    np.testing.assert_array_equal(selected.values,result.values[...,1:2])


def sampled(points,values):
    return app.SampledPoints(points,values,np.zeros(len(points),dtype=np.int64),
        np.ones(len(points),dtype=bool),(sm.FieldDefinition('value','code'),),None)


def test_save_snapshots_writable_outputs_and_readonly_views(tmp_path,monkeypatch):
    points=sm.PointSet([[.2,.3,.4],[.4,.5,.6]])
    backing=np.array([[1.,-1.],[2.,-1.]])
    view=backing[:,::2]
    view.flags.writeable=False
    result=sampled(points,view)
    original=rio._write_archive
    def mutate_after_capture(stream,arrays):
        backing[:,0]=99.
        result.owners[:]=7
        original(stream,arrays)
    monkeypatch.setattr(rio,'_write_archive',mutate_after_capture)
    path=sm.save_result(tmp_path/'result.npz',result)
    restored=sm.load_result(path).result
    np.testing.assert_array_equal(restored.values[:,0],[1.,2.])
    np.testing.assert_array_equal(restored.owners,0)
    assert backing.flags.writeable and result.owners.flags.writeable
    assert not view.flags.writeable


def test_readonly_line_owner_with_existing_writable_alias_is_snapshotted(tmp_path,monkeypatch):
    seeds=sm.PointSet([[.2,.3,.4]])
    positions=np.array([[.2,.3,.4],[.2,.3,.5]])
    alias=positions.view()  # Writable alias survives LineSet setting its owner read-only.
    lines=sm.LineSet(seeds,positions,np.array([0,0,2],dtype=np.int64),
        np.array([[-1,int(sm.Termination.MAX_STEPS)]],dtype=np.int64),None)
    expected=positions.copy()
    original=rio._write_archive
    def mutate_after_capture(stream,arrays):
        alias[1,2]=.9
        original(stream,arrays)
    monkeypatch.setattr(rio,'_write_archive',mutate_after_capture)
    restored=sm.load_result(sm.save_result(tmp_path/'lines.npz',lines)).result
    np.testing.assert_array_equal(restored.positions,expected)
    assert positions[1,2]==.9 and not positions.flags.writeable


def test_save_preserves_strided_endian_and_nonfinite_payload_bits(tmp_path):
    points=sm.PointSet([[.2,.3,.4],[.4,.5,.6],[.6,.6,.6]])
    bits=np.array([0x8000000000000000,0x7ff8000000001234,0x7ff0000000000000],dtype=np.uint64)
    backing=np.empty((3,2),dtype='>f8')
    backing[:,0]=bits.view(np.float64)
    result=sampled(points,backing[:,::2])
    restored=sm.load_result(sm.save_result(tmp_path/'bits.npz',result)).result
    np.testing.assert_array_equal(restored.values[:,0].view(np.uint64),bits)
    assert backing.flags.writeable and not restored.values.flags.writeable


def test_save_owned_geometry_avoids_a_second_full_snapshot(tmp_path):
    # Isolate geometry from writable result arrays and measure incremental space.
    points=sm.PointSet(np.random.default_rng(7).uniform(size=(100000,3)))
    original=points.positions.copy()
    tracemalloc.start()
    try:
        path=sm.save_result(tmp_path/'geometry.npz',points)
        peak=tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < points.nbytes  # A complete geometry snapshot alone exceeds this bound.
    restored=sm.load_result(path).result
    np.testing.assert_array_equal(restored.positions,original)
    assert not points.positions.flags.writeable
    assert not np.shares_memory(restored.positions,points.positions)


def test_save_validation_rejects_degenerate_plane_before_publication(tmp_path):
    plane=sm.Plane([0,0,0],[1,0,0],[0,1,0],(1,1))
    points=sm.PointSet.from_plane(plane)
    plane.v.flags.writeable=True
    plane.v[:]=plane.u
    plane.v.flags.writeable=False
    with pytest.raises(rio.ResultFileError,match='independent'):
        sm.save_result(tmp_path/'invalid.npz',points)
    assert not list(tmp_path.iterdir())
