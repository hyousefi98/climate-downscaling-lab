#!/usr/bin/env python3
"""Prepare all 35 ERA5 conditioning channels on the explicit target grid."""
import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr
from data.source_profiles import ERA5_SURFACE,ERA5_PRESSURE,PRESSURE_LEVELS
from data.preprocessing import aggregate_six_hour,stack_fields,write_zarr_new
from data.regridding import target_coordinates,rectilinear


def prepare(surface,pressure,reference,*,radiation_profile='all_sky',temporal_profile='ending',boundary='mask'):
    def normalize(ds):
        if 'valid_time' in ds.dims:ds=ds.rename(valid_time='time')
        if 'level' in ds.dims:ds=ds.rename(level='pressure_level')
        return ds
    surface,pressure=normalize(surface),normalize(pressure)
    xr.align(surface.time,pressure.time,join='exact')
    lat,lon=target_coordinates(reference)
    mapping=dict(ERA5_SURFACE)
    if radiation_profile=='historical_clear_sky':
        del mapping['surface_thermal_radiation_downwards']
        mapping['surface_thermal_radiation_downward_clear_sky']=('strdc','GLW',1/3600)
    elif radiation_profile!='all_sky':raise ValueError(radiation_profile)
    fields={};counts={}
    for _,(src,dst,scale) in mapping.items():
        if src not in surface:raise KeyError(f'Required ERA5 surface channel {src} missing')
        fields[dst]=surface[src]*scale
    for src in ERA5_PRESSURE.values():
        if src not in pressure:raise KeyError(f'Required ERA5 pressure variable {src} missing')
        for level in PRESSURE_LEVELS:fields[f'{src}_{level}']=pressure[src].sel(pressure_level=level,drop=True)
    support=None;outputs={}
    for name,field in fields.items():
        if set(field.dims)!={'time','latitude','longitude'}:raise ValueError(f'{name}: select extra dimensions explicitly: {field.dims}')
        if support is None:support=np.isfinite(rectilinear(xr.ones_like(field.isel(time=0,drop=True)),lat,lon,boundary=boundary))
        field=rectilinear(field.where(np.isfinite(field)),lat,lon,boundary=boundary).assign_coords(y=np.arange(lat.shape[0]),x=np.arange(lat.shape[1]))
        outputs[name],counts[name]=aggregate_six_hour(field,precipitation=name=='PREC_ACC_NC',temporal_profile=temporal_profile)
    result=stack_fields(outputs,lat,lon,array_name='era5',numeric_profile='physical',spatial_support=xr.DataArray(np.asarray(support),dims=('y','x')))
    result['hourly_valid_count']=xr.concat(list(counts.values()),dim='channel').transpose('time','channel','y','x').astype('uint8')
    units={'U10':'m s-1','V10':'m s-1','T2':'K','PSFC':'hPa','PREC_ACC_NC':'mm','SWDOWN':'W m-2','GLW':'W m-2'}
    pl_units=dict(t='K',u='m s-1',v='m s-1',q='kg kg-1',z='m2 s-2',cc='1',r='%')
    units.update({f'{v}_{lev}':unit for v,unit in pl_units.items() for lev in PRESSURE_LEVELS})
    result.attrs.update(temporal_profile=temporal_profile,precipitation_interval='(t-6h,t]' if temporal_profile=='ending' else '(t-1h,t+5h]',
                        time_label='interval_end' if temporal_profile=='ending' else 'first_hour_label',radiation_profile=radiation_profile,
                        humidity_representation='specific_humidity',pressure_units='hPa',wind_frame='earth',
                        source_interval='hourly_end',regrid_method='rectilinear_bilinear',boundary_policy=boundary,units_json=json.dumps(units))
    return result.chunk({'time':24,'channel':1,'y':-1,'x':-1})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--surface-files',nargs='+',required=True)
    p.add_argument('--pressure-files',nargs='+',required=True)
    p.add_argument('--reference-grid',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--radiation-profile',choices=('all_sky','historical_clear_sky'),default='all_sky')
    p.add_argument('--temporal-profile',choices=('ending','historical_forward'),default='ending')
    p.add_argument('--boundary',choices=('mask','extrapolate'),default='mask')
    a=p.parse_args()
    if Path(a.output).exists() or Path(a.output+'.partial').exists():raise FileExistsError(a.output)
    surface=xr.open_mfdataset(a.surface_files,combine='by_coords',join='exact')
    pressure=xr.open_mfdataset(a.pressure_files,combine='by_coords',join='exact')
    result=prepare(surface,pressure,xr.open_dataset(a.reference_grid),radiation_profile=a.radiation_profile,
                   temporal_profile=a.temporal_profile,boundary=a.boundary)
    write_zarr_new(result,a.output)


if __name__=='__main__':main()
