"""Archived-grid paired recipe: exact arithmetic and support checks."""
import numpy as np
import pytest
import xarray as xr
from data.matched_reference import CHANNELS,FACTORS,CLIPPED,inspect_source,aggregate_archived,independent_reconstruction,reference_roundtrip
from data.preprocessing import write_zarr_new


def test_chunk_accumulation_matches_independent_six_hour_windows(tmp_path):
    values=np.arange(12*8*3*4,dtype=np.float32).reshape(12,8,3,4)/17+10
    values[:,2,1,1]=-.125 # Original postaggregation clipping, no hourly repair.
    lat,lon=np.meshgrid(np.arange(3.)+30,np.arange(4.)-115,indexing='ij')
    source=xr.Dataset({'aorc':(('time','channel','y','x'),values)},coords={
        'time':np.arange(1,13),'channel':np.arange(8),'lat':(('y','x'),lat),'lon':(('y','x'),lon),
        'aorc_variable':('channel',np.array(CHANNELS,dtype='S64'))})
    source.time.attrs['units']='hours since 2012-01-01 00:00:00'
    path=tmp_path/'hourly.zarr'
    write_zarr_new(source.chunk({'time':4,'channel':3,'y':3,'x':4}),path)
    requested=np.array(['2012-01-01T06','2012-01-01T12'],dtype='datetime64[h]')
    indices,meta,_=inspect_source(path,requested,lat,lon)
    mask=np.ones((3,4),bool);mask[0,0]=False
    got,audit=aggregate_archived(path,indices,meta,mask)
    expected=np.empty_like(got)
    for i in range(2):
        expected[i,0]=values[i*6:(i+1)*6,0].sum(axis=0,dtype=np.float64)
        expected[i,1:]=values[i*6:(i+1)*6,1:].mean(axis=0,dtype=np.float64)
    np.testing.assert_array_equal(got,expected)
    assert audit['channel_checks']['DSWRF_surface']['negative']==12
    assert got[0,2,1,1]==-.125
    independent_reconstruction(path,got,indices,mask)
    with pytest.raises(ValueError):inspect_source(path,requested+np.timedelta64(12,'h'),lat,lon)


@pytest.mark.integration
def test_reference_roundtrip_matches_archived_loader_arithmetic():
    # Exercise the actual preserved legacy loader, not a second implementation
    # of the new helper. This optional test requires the model environment.
    pytest.importorskip('physicsnemo')
    from models.datasets.aorc_legacy import AORCDownscalingDataset,DataFilters
    rng=np.random.default_rng(312)
    raw=rng.uniform(-2,100,size=(8,3,4)).astype(np.float32)
    raw[0,0,:]=[.5,.9999999,1.,1.0000001]
    mask=np.ones((3,4),bool);mask[-1,-1]=False
    stats={'output':{name:{'mean':float(i+.321),'std':float(i*.5+.123)} for i,name in enumerate(CHANNELS)}}
    model=object.__new__(AORCDownscalingDataset)
    model.stats=stats;model.output_variables=CHANNELS;model.clip_negative=True;model.precip_threshold=.1
    expected=DataFilters.apply_scale_factors(raw,CHANNELS)
    expected=DataFilters.clip_negative_values(expected,CHANNELS)
    expected=DataFilters.apply_precip_threshold(expected,CHANNELS,.1)
    expected=model.normalize_output(expected)
    expected[:,~mask]=np.nan
    expected=model.denormalize_output(expected)
    np.testing.assert_array_equal(reference_roundtrip(raw,stats,mask),expected)
