"""Small scientific invariants; no downloaded data or GPUs."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import xarray as xr
from data.source_profiles import AORC_CHANNELS,decode_aorc,mixing_ratio_to_specific_humidity,rotate_grid_wind
from data.preprocessing import block_mean,interpolate_index_grid,aggregate_six_hour,cumulative_to_increment,write_zarr_new,training_statistics
from data.acquisition import subset,ingest_manifest

ROOT=Path(__file__).resolve().parents[1]


def load(name,folder='preprocess'):
    spec=importlib.util.spec_from_file_location('fixture_'+name,ROOT/'scripts'/folder/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_fill_masking_precedes_block_mean_and_precip_zero_is_data():
    raw=xr.DataArray([[3000,-32767],[3000,3000]],dims=('y','x'))
    valid=decode_aorc(raw,'TMP_2maboveground')
    mean,coverage=block_mean(valid,2)
    assert np.isnan(mean.item()) and coverage.item()==.75
    partial,_=block_mean(valid,2,min_coverage=.75)
    assert partial.item()==300
    rain=decode_aorc(np.array([0,-32767,10]),'APCP_surface')
    np.testing.assert_allclose(rain,[0,np.nan,1],equal_nan=True)


def test_physical_and_legacy_humidity_are_explicit():
    assert decode_aorc(np.array([100]),'SPFH_2maboveground').item()==pytest.approx(.01)
    assert decode_aorc(np.array([100]),'SPFH_2maboveground',profile='legacy_model').item()==pytest.approx(.1)
    assert decode_aorc(np.array([100]),'SPFH_2maboveground',profile='legacy_packed').item()==100
    np.testing.assert_allclose(mixing_ratio_to_specific_humidity(np.array([.01,np.nan])),[.01/1.01,np.nan])
    with pytest.raises(ValueError):mixing_ratio_to_specific_humidity(np.array([-.01]))
    u,v=rotate_grid_wind(3,4,0,1)
    assert (u,v)==(-4,3)


def test_spatial_mean_preserves_equal_weight_domain_mean():
    field=xr.DataArray(np.arange(64).reshape(8,8),dims=('y','x'))
    coarse,_=block_mean(field,4)
    assert coarse.mean().item()==field.mean().item()
    constant=xr.DataArray(np.full((3,3),7.),dims=('y','x'),coords={'y':[1.5,5.5,9.5],'x':[1.5,5.5,9.5]})
    masked,support=interpolate_index_grid(constant,np.arange(12),np.arange(12))
    assert np.isnan(masked[0,0]) and not support[0,0]
    assert masked[5,5]==7
    for boundary in ('nearest','extrapolate'):
        result,_=interpolate_index_grid(constant,np.arange(12),np.arange(12),boundary=boundary)
        np.testing.assert_allclose(result,7)


def test_six_hour_end_window_coverage_zero_and_missing():
    times=np.arange('2012-01-01T01','2012-01-01T13',dtype='datetime64[h]')
    values=xr.DataArray(np.ones((12,1,3)),dims=('time','y','x'),coords={'time':times})
    values[:,:,0]=0;values[2,0,1]=np.nan
    rain,count=aggregate_six_hour(values,precipitation=True)
    np.testing.assert_array_equal(rain.time,np.array(['2012-01-01T06','2012-01-01T12'],dtype='datetime64[h]'))
    assert rain[0,0,0]==0 and np.isnan(rain[0,0,1]) and rain[0,0,2]==6
    assert count[0,0,1]==5
    missing_hour=values.isel(time=[i for i in range(12) if i!=8])
    gap,_=aggregate_six_hour(missing_hour,precipitation=True,drop_boundary_partial=False)
    assert np.isnan(gap[1]).all()
    with pytest.raises(ValueError):aggregate_six_hour(values.isel(time=[0,0,1]))


def test_historical_grouping_is_not_end_alignment():
    times=np.arange('2012-01-01T00','2012-01-01T12',dtype='datetime64[h]')
    field=xr.DataArray(np.arange(12.),dims='time',coords={'time':times})
    ending,_=aggregate_six_hour(field,precipitation=True)
    forward,_=aggregate_six_hour(field,precipitation=True,temporal_profile='historical_forward')
    assert ending.item()==21 and forward[0].item()==15
    assert ending.time.values[0]==np.datetime64('2012-01-01T06')
    assert forward.time.values[0]==np.datetime64('2012-01-01T00')


def test_cumulative_resets_require_explicit_policy():
    field=xr.DataArray([1.,3.,.5,2.],dims='time')
    with pytest.raises(ValueError):cumulative_to_increment(field)
    np.testing.assert_allclose(cumulative_to_increment(field,reset='restart'),[np.nan,2,.5,1.5])


def encoded_fixture():
    times=np.arange('2012-01-01T01','2012-01-01T19',dtype='datetime64[h]')
    values=[10,3000,2000,9000,100,2900,30,40]
    fields={n:(('time','latitude','longitude'),np.full((18,24,24),v,dtype='int16')) for n,v in zip(AORC_CHANNELS,values)}
    ds=xr.Dataset(fields,coords={'time':times,'latitude':30+np.arange(24)/120,'longitude':-115+np.arange(24)/120})
    # Variation ensures finite, positive normalization scales.
    for n in AORC_CHANNELS:ds[n].values+=np.arange(18,dtype='int16')[:,None,None]
    return ds


def test_end_to_end_packed_native_to_zarr_and_common_statistics(tmp_path):
    aorc=load('aorc')
    source=encoded_fixture()
    inp,target=aorc.prepare(source)
    assert inp.attrs['numeric_profile']=='physical'
    assert inp.attrs['humidity_representation']=='specific_humidity'
    assert inp.attrs['precipitation_interval']=='(t-6h,t]'
    np.testing.assert_allclose(target.aorc[0,0],7.5) # sum 1.0,...,1.5 mm
    np.testing.assert_allclose(target.aorc[0,4],.01025,rtol=1e-6)
    assert int(inp.spatial_support.sum())==16
    for name,ds in [('input',inp),('target',target)]:write_zarr_new(ds,tmp_path/(name+'.zarr'))
    loaded=[xr.open_zarr(tmp_path/(n+'.zarr')) for n in ('input','target')]
    stats=training_statistics(*loaded,np.ones((8,8),bool),input_array='aorc',target_array='aorc',start='2012-01-01',end='2012-01-01T13')
    assert stats['metadata']['samples']==2 and stats['metadata']['mask_cells']==16
    assert stats['input']['APCP_surface']['mean']==pytest.approx(9.3)
    assert stats['input']['APCP_surface']['std']==pytest.approx(1.8)
    for name in AORC_CHANNELS:
        for metric in ('mean','std'):
            assert stats['input'][name][metric]==pytest.approx(stats['output'][name][metric],abs=1e-4)
    with pytest.raises(FileExistsError):write_zarr_new(inp,tmp_path/'input.zarr')
    # Missing native source is never diluted into a valid model pixel.
    source['TMP_2maboveground'].values[0,12,12]=-32767
    broken,_=aorc.prepare(source)
    assert np.isnan(broken.aorc[0,5]).all()


def test_subset_exclusive_end_and_manifest_checks(tmp_path):
    ds=encoded_fixture()
    selected=subset(ds,start='2012-01-01T06',end='2012-01-01T12',bbox=(-115,30,-114.8,30.2))
    assert selected.sizes['time']==6 and selected.time[-1]==np.datetime64('2012-01-01T11')
    import hashlib
    source=tmp_path/'source.txt';source.write_text('fixture')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps([{'source':str(source),'path':'nested/a.txt','sha256':hashlib.sha256(source.read_bytes()).hexdigest()}]))
    ingest_manifest(manifest,tmp_path/'dest')
    assert (tmp_path/'dest/nested/a.txt').read_text()=='fixture'
    with pytest.raises(FileExistsError):ingest_manifest(manifest,tmp_path/'dest')
    manifest.write_text(json.dumps([{'source':str(source),'path':'../outside.txt','sha256':'0'*64}]))
    with pytest.raises(ValueError):ingest_manifest(manifest,tmp_path/'dest')


def test_era5_plan_requires_no_credentials_and_retains_all_channels():
    plan=load('era5','download').requests_for(2012,2,(-118,28,-105,40.5))
    assert len(plan)==14
    assert all(len(request['day'])==29 for _,request,_ in plan)
    assert any(request['variable']==['surface_thermal_radiation_downwards'] for _,request,_ in plan)
    assert sum('pressure_level' in request for _,request,_ in plan)==7


def test_era5_preprocess_keeps_all_35_fields_and_physical_units():
    from data.source_profiles import ERA5_SURFACE,ERA5_PRESSURE,PRESSURE_LEVELS
    time=np.arange('2012-01-01T01','2012-01-01T13',dtype='datetime64[h]')
    coord={'time':time,'latitude':[30.,31.,32.],'longitude':[-115.,-114.,-113.]}
    source_values=dict(u10=3,v10=4,t2m=290,sp=90000,tp=.001,ssrd=360000,strd=1080000)
    surface=xr.Dataset({name:(('time','latitude','longitude'),np.full((12,3,3),value)) for name,value in source_values.items()},coords=coord)
    pcoord=dict(coord,pressure_level=list(PRESSURE_LEVELS))
    pressure=xr.Dataset({name:(('time','pressure_level','latitude','longitude'),np.full((12,4,3,3),.01 if name=='q' else 1.)) for name in ERA5_PRESSURE.values()},coords=pcoord)
    lat,lon=np.meshgrid([30.5,31.5],[-114.5,-113.5],indexing='ij')
    reference=xr.Dataset({'lat':(('y','x'),lat),'lon':(('y','x'),lon)})
    result=load('era5').prepare(surface,pressure,reference)
    names=[x.decode() for x in result.era5_variable.values]
    assert len(names)==35 and 'Q2' not in names
    np.testing.assert_allclose(result.era5[:,names.index('PSFC')],900)
    np.testing.assert_allclose(result.era5[:,names.index('PREC_ACC_NC')],6)
    np.testing.assert_allclose(result.era5[:,names.index('GLW')],300)
    np.testing.assert_allclose(result.era5[:,names.index('q_925')],.01)
    with pytest.raises(KeyError):load('era5').prepare(surface.drop_vars('strd'),pressure,reference)


def test_wrf_same_grid_source_conventions_are_converted_before_means():
    mod=load('paired_fields')
    time=np.arange('2012-01-01T01','2012-01-01T13',dtype='datetime64[h]')
    values=dict(GLW=300,SWDOWN=100,PSFC=90000,Q2=.02,T2=290,U10=3,V10=4,PREC_ACC_NC=1)
    lat,lon=np.meshgrid([30.,31.],[-115.,-114.],indexing='ij')
    source=xr.Dataset({name:(('time','y','x'),np.full((12,2,2),value)) for name,value in values.items()},coords={'time':time,'lat':(('y','x'),lat),'lon':(('y','x'),lon)})
    reference=source[['lat','lon']]
    kwargs=dict(array_name='wrf',variable_map={name:name for name in mod.NAMES},humidity_kind='mixing_ratio',pressure_units='Pa',wind_frame='earth',source_interval='hourly_end',precipitation_kind='interval')
    result=mod.prepare(source,reference,**kwargs)
    names=[x.decode() for x in result.wrf_variable.values]
    np.testing.assert_allclose(result.wrf[:,names.index('Q2')],.02/1.02)
    np.testing.assert_allclose(result.wrf[:,names.index('PSFC')],900)
    np.testing.assert_allclose(result.wrf[:,names.index('RAIN_TOTAL')],6)
    assert result.attrs['source_humidity_representation']=='mixing_ratio'
    assert result.attrs['humidity_representation']=='specific_humidity'
    with pytest.raises(ValueError):mod.prepare(source,reference,**dict(kwargs,wind_frame='grid'))


def test_statistics_threshold_and_common_drop_membership():
    inp,target=load('aorc').prepare(encoded_fixture())
    inp=inp.compute();target=target.compute()
    for ds in (inp,target):
        ds['aorc'].values[0,0,:,:]=.05
        ds['aorc'].values[1,0,:,:]=.1
        ds['aorc'].values[2,0,:,:]=.2
    inp['aorc'].values[1,5,3,3]=np.nan
    options=dict(input_array='aorc',target_array='aorc',start='2012-01-01',end='2012-01-02')
    with pytest.raises(ValueError):training_statistics(inp,target,np.ones((8,8),bool),**options)
    stats=training_statistics(inp,target,np.ones((8,8),bool),invalid_times='drop',**options)
    assert stats['metadata']['samples']==2 and len(stats['metadata']['dropped_timestamps'])==1
    assert stats['input']['APCP_surface']['mean']==pytest.approx(.1)
    assert stats['metadata']['precip_threshold']==.1


def test_landmask_geometry_and_initial_buffer_are_explicit():
    shapely=pytest.importorskip('shapely')
    lat,lon=np.meshgrid(np.arange(5.),np.arange(5.),indexing='ij')
    geom=shapely.geometry.box(.5,.5,3.5,3.5)
    mod=load('landmask')
    raw=mod.polygon_mask(lat,lon,[geom])
    buffered=mod.polygon_mask(lat,lon,[geom],initial_coastal_buffer=1)
    assert int(raw.sum())==9 and int(buffered.sum())==1
    with pytest.raises(ValueError):mod.polygon_mask(lat,lon,[geom],initial_coastal_buffer=-1)
