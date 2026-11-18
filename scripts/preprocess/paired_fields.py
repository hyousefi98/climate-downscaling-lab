#!/usr/bin/env python3
"""Prepare WRF/CONUS404 fields with explicit source units, wind frame, and intervals."""
import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr
from data.preprocessing import aggregate_six_hour,cumulative_to_increment,stack_fields,write_zarr_new
from data.source_profiles import mixing_ratio_to_specific_humidity,rotate_grid_wind
from data.regridding import target_coordinates,rectilinear,projected,curvilinear_regridder

CONUS404_PROJECTION='+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +a=6370000 +b=6370000 +units=m'
NAMES=('GLW','SWDOWN','PSFC','Q2','T2','U10','V10','PREC_ACC_NC')
UNITS=dict(zip(NAMES,('W m-2','W m-2','hPa','kg kg-1','K','m s-1','m s-1','mm')))


def prepare(source,reference,*,array_name,variable_map,humidity_kind,pressure_units,wind_frame,
            source_interval,precipitation_kind,regrid_method='same_grid',projection=None,
            weights_path=None,boundary='mask',temporal_profile='ending',reset='raise',
            latitude='lat',longitude='lon',y_dim='y',x_dim='x'):
    if humidity_kind not in ('mixing_ratio','specific_humidity'):raise ValueError(humidity_kind)
    if pressure_units not in ('Pa','hPa'):raise ValueError(pressure_units)
    if wind_frame not in ('earth','grid'):raise ValueError(wind_frame)
    if source_interval not in ('hourly_end','six_hourly_end'):raise ValueError(source_interval)
    if precipitation_kind not in ('interval','cumulative'):raise ValueError(precipitation_kind)
    if source_interval=='six_hourly_end' and temporal_profile!='ending':
        raise ValueError('Already six-hourly end-labelled fields cannot use historical forward grouping')
    time=np.asarray(source.time.values).astype('datetime64[ns]')
    step=np.timedelta64(1 if source_interval=='hourly_end' else 6,'h')
    if len(time)<2 or np.isnat(time).any() or np.any(np.diff(time)<=np.timedelta64(0,'ns')):
        raise ValueError('Need unique increasing decoded source times')
    if source_interval=='six_hourly_end' and (np.any(np.diff(time)!=step) or np.any(time.astype('datetime64[h]').astype('int64')%6)):
        raise ValueError('Six-hourly records must be contiguous and end at 00/06/12/18 UTC')
    if precipitation_kind=='cumulative' and np.any(np.diff(time)!=step):
        raise ValueError('Cannot difference cumulative precipitation across missing source times')
    lat,lon=target_coordinates(reference)
    fields={}
    for name in NAMES:
        src=variable_map[name]
        if src not in source:raise KeyError(f'Missing required field {src}')
        field=source[src]
        # xarray CF decoding handles declared fill/scale; also mask documented
        # missing_value attrs retained by some Zarr producers before any remap.
        fill=field.attrs.get('missing_value')
        if fill is not None:field=field.where(field!=fill)
        field=field.where(np.isfinite(field)).astype('float32')
        rename={old:new for old,new in ((y_dim,'y'),(x_dim,'x')) if old!=new and old in field.dims}
        field=field.rename(rename)
        extra=set(field.dims)-{'time','y','x','latitude','longitude'}
        if extra:raise ValueError(f'{src}: unexpected dimensions {extra}; select one source field explicitly')
        fields[name]=field
    fields['PSFC']=fields['PSFC']*(.01 if pressure_units=='Pa' else 1)
    if humidity_kind=='mixing_ratio':fields['Q2']=mixing_ratio_to_specific_humidity(fields['Q2'])
    if wind_frame=='grid':
        if not {'COSALPHA','SINALPHA'}<=set(source):raise ValueError('Grid-relative winds require supplied COSALPHA and SINALPHA')
        angles=[source[n].rename({d:v for d,v in ((y_dim,'y'),(x_dim,'x')) if d!=v and d in source[n].dims}) for n in ('COSALPHA','SINALPHA')]
        fields['U10'],fields['V10']=rotate_grid_wind(fields['U10'],fields['V10'],*angles)
    if precipitation_kind=='cumulative':fields['PREC_ACC_NC']=cumulative_to_increment(fields['PREC_ACC_NC'],reset=reset)
    source_lat,source_lon=np.asarray(source[latitude]),np.asarray(source[longitude])
    if source_lat.ndim==3 and source_lat.shape[0]==1:source_lat=source_lat[0]
    if source_lon.ndim==3 and source_lon.shape[0]==1:source_lon=source_lon[0]
    if regrid_method=='same_grid':
        if not np.array_equal(source_lat,lat) or not np.array_equal(source_lon,lon):
            raise ValueError('same_grid requires exact source/target coordinates')
        remap=lambda f:f
    elif regrid_method=='rectilinear':remap=lambda f:rectilinear(f,lat,lon,boundary=boundary)
    elif regrid_method=='projected':
        if not projection:raise ValueError('Provide documented source projection')
        remap=lambda f:projected(f,source_lat,source_lon,lat,lon,projection=projection,boundary=boundary)
    elif regrid_method=='esmf_bilinear':
        if boundary!='mask':raise ValueError('ESMF path does not extrapolate')
        if weights_path is None:raise ValueError('Provide a new weights path')
        engine=curvilinear_regridder(source_lat,source_lon,lat,lon,weights_path=weights_path)
        remap=lambda f:engine(f,skipna=False)
    else:raise ValueError(regrid_method)
    # A field of ones tests geometric support independently of weather/missingness.
    support=np.isfinite(remap(xr.ones_like(fields['T2'].isel(time=0,drop=True))))
    result={};counts={}
    for name,field in fields.items():
        field=remap(field).assign_coords(y=np.arange(lat.shape[0]),x=np.arange(lat.shape[1]))
        if source_interval=='hourly_end':
            result[name],counts[name]=aggregate_six_hour(field,precipitation=name=='PREC_ACC_NC',temporal_profile=temporal_profile)
        else:result[name]=field
    support=xr.DataArray(np.asarray(support),dims=('y','x'))
    units=dict(UNITS)
    if array_name=='wrf':
        result={'RAIN_TOTAL':result.pop('PREC_ACC_NC'),**result}
        if counts:counts={'RAIN_TOTAL':counts.pop('PREC_ACC_NC'),**counts}
        units={'RAIN_TOTAL':units.pop('PREC_ACC_NC'),**units}
    out=stack_fields(result,lat,lon,array_name=array_name,numeric_profile='physical',spatial_support=support)
    if counts:out['hourly_valid_count']=xr.concat(list(counts.values()),dim='channel').transpose('time','channel','y','x').astype('uint8')
    out.attrs.update(temporal_profile=temporal_profile,precipitation_interval='(t-6h,t]' if temporal_profile=='ending' else '(t-1h,t+5h]',
                     time_label='interval_end' if temporal_profile=='ending' else 'first_hour_label',
                     humidity_representation='specific_humidity',source_humidity_representation=humidity_kind,
                     pressure_units='hPa',source_pressure_units=pressure_units,wind_frame='earth',source_wind_frame=wind_frame,
                     regrid_method=regrid_method,boundary_policy=boundary,units_json=json.dumps(units),
                     source_interval=source_interval,precipitation_kind=precipitation_kind,reset_policy=reset)
    return out.chunk({'time':24,'channel':1,'y':-1,'x':-1})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','reference-grid','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--array-name',choices=('wrf','conus404'),required=True)
    p.add_argument('--variable-map',help='JSON mapping canonical names to source names; default names unchanged')
    p.add_argument('--humidity-kind',choices=('mixing_ratio','specific_humidity'),required=True)
    p.add_argument('--pressure-units',choices=('Pa','hPa'),required=True)
    p.add_argument('--wind-frame',choices=('earth','grid'),required=True)
    p.add_argument('--source-interval',choices=('hourly_end','six_hourly_end'),required=True)
    p.add_argument('--precipitation-kind',choices=('interval','cumulative'),required=True)
    p.add_argument('--regrid-method',choices=('same_grid','rectilinear','projected','esmf_bilinear'),default='same_grid')
    p.add_argument('--projection',help='Explicit PROJ string; use documented source projection, not a generic WRF guess')
    p.add_argument('--weights-path')
    p.add_argument('--boundary',choices=('mask','extrapolate'),default='mask')
    p.add_argument('--temporal-profile',choices=('ending','historical_forward'),default='ending')
    p.add_argument('--reset',choices=('raise','restart'),default='raise')
    for name,default in (('latitude','lat'),('longitude','lon'),('y-dim','y'),('x-dim','x')):p.add_argument('--'+name,default=default)
    a=p.parse_args()
    if Path(a.output).exists() or Path(a.output+'.partial').exists():raise FileExistsError(a.output)
    source=xr.open_zarr(a.source) if a.source.endswith('.zarr') else xr.open_dataset(a.source,chunks={})
    mapping=json.loads(Path(a.variable_map).read_text()) if a.variable_map else {name:('RAIN_TOTAL' if name=='PREC_ACC_NC' and a.array_name=='wrf' else name) for name in NAMES}
    result=prepare(source,xr.open_dataset(a.reference_grid),array_name=a.array_name,variable_map=mapping,
                   humidity_kind=a.humidity_kind,pressure_units=a.pressure_units,wind_frame=a.wind_frame,
                   source_interval=a.source_interval,precipitation_kind=a.precipitation_kind,
                   regrid_method=a.regrid_method,projection=a.projection,weights_path=a.weights_path,boundary=a.boundary,
                   temporal_profile=a.temporal_profile,reset=a.reset,latitude=a.latitude,longitude=a.longitude,y_dim=a.y_dim,x_dim=a.x_dim)
    write_zarr_new(result,a.output)


if __name__=='__main__':main()
