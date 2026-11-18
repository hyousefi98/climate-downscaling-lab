"""Exact arithmetic of the 2026-11-18 archived-grid paired preparation.

This compatibility path consumes already spatially processed hourly packed
AORC stores. It cannot repair their original spatial fill-value handling, and
is deliberately separate from corrected native preprocessing. It validates
every selected hour on the original fixed verification mask.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import netCDF4 as nc
import zarr
from numcodecs import get_codec
from data.source_profiles import AORC_CHANNELS

CHANNELS=list(AORC_CHANNELS)
CLIPPED={'APCP_surface','DLWRF_surface','DSWRF_surface','SPFH_2maboveground'}
FACTORS=[.1,.1,.1,.1,.001,.1,.1,.1]
PHYSICAL_FACTORS=[.1,.1,.1,10.,.0001,.1,.1,.1]
PHYSICAL_UNITS=['mm','W m-2','W m-2','Pa','kg kg-1','K','m s-1','m s-1']
OUTPUT_NAMES=['APCPsurface','DLWRFsurface','DSWRFsurface','PRESsurface','SPFH2maboveground','TMP2maboveground','UGRD10maboveground','VGRD10maboveground']


def sha_file(path):
    sha=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):sha.update(block)
    return sha.hexdigest()


def inspect_source(path,requested,lat,lon):
    path=Path(path);store=zarr.open_group(str(path),mode='r')
    names=[x.decode() if isinstance(x,bytes) else str(x) for x in store['aorc_variable'][:]]
    if names!=CHANNELS:raise ValueError('Archived AORC channel order differs')
    h,w=lat.shape
    if not np.array_equal(store['lat'][:h,:w],lat) or not np.array_equal(store['lon'][:h,:w],lon):
        raise ValueError('Hourly source crop does not exactly match baseline coordinates/orientation')
    tv=store['time'];units=tv.attrs.get('units','')
    if not units.startswith('hours since '):raise ValueError('Archived hourly time must declare hours since its origin')
    numeric=np.asarray(tv[:])
    if not np.equal(numeric,np.floor(numeric)).all():raise ValueError('Noninteger hourly source labels')
    times=np.datetime64(units.split(' since ',1)[1],'h')+numeric.astype('timedelta64[h]')
    if not np.all(np.diff(times)==np.timedelta64(1,'h')):raise ValueError('Hourly source labels must be contiguous')
    requested=np.asarray(requested,dtype='datetime64[h]')
    windows=requested[:,None]+np.arange(-5,1,dtype='timedelta64[h]')[None,:]
    indices=np.searchsorted(times,windows)
    if np.any(indices>=len(times)) or not np.array_equal(times[indices],windows):raise ValueError('A requested six-hour window is missing')
    meta=json.loads((path/'aorc/.zarray').read_text())
    if meta['shape'][1]!=8 or meta['chunks'][2:]!=meta['shape'][2:]:
        raise ValueError('Archived replay requires one full spatial chunk, as in the verified source stores')
    if meta.get('filters') or meta.get('dimension_separator','.')!='.' or meta.get('order','C')!='C':
        raise ValueError('Unsupported archived chunk encoding; use the verified source layout')
    report={'source_name':path.name,'times':len(times),'first_time':str(times[0]),'last_time':str(times[-1]),
            'hourly_indices':indices.tolist(),'metadata_sha256':{key:sha_file(path/key) for key in ['aorc/.zarray','time/.zattrs','time/.zarray','aorc_variable/.zarray','lat/.zarray','lon/.zarray']}}
    return indices,meta,report


def aggregate_archived(path,indices,meta,mask):
    """Same chunk traversal/reduction order as the original verified builder."""
    path=Path(path);shape=tuple(meta['chunks']);h,w=mask.shape
    codec=get_codec(meta['compressor']);n=len(indices)
    sums=np.zeros((n,8,h,w),dtype=np.float64);counts=np.zeros((n,8),dtype=np.int16)
    checks={name:{'hourly_values_checked':0,'nonfinite':0,'fill_sentinel':0,'negative':0,'minimum':None,'maximum':None} for name in CHANNELS}
    chunks=[]
    for chunk_id in sorted(set((indices//shape[0]).ravel().tolist())):
        start=chunk_id*shape[0]
        selections=[(case,row[(row>=start)&(row<start+shape[0])]-start) for case,row in enumerate(indices)]
        for channel_chunk in range(int(np.ceil(8/shape[1]))):
            channel_start=channel_chunk*shape[1]
            key=f'{chunk_id}.{channel_chunk}.0.0';encoded=(path/'aorc'/key).read_bytes()
            chunks.append({'key':key,'sha256':hashlib.sha256(encoded).hexdigest(),'bytes':len(encoded)})
            block=np.frombuffer(codec.decode(encoded),dtype=meta['dtype']).reshape(shape)
            for case,hours in selections:
                if not len(hours):continue
                for local_channel in range(min(shape[1],8-channel_start)):
                    channel=channel_start+local_channel;name=CHANNELS[channel]
                    field=block[hours,local_channel,:h,:w];selected=field[:,mask]
                    nonfinite=~np.isfinite(selected);fill=selected<=-32000;negative=selected<0
                    record=checks[name];record['hourly_values_checked']+=int(selected.size)
                    for label,bad in [('nonfinite',nonfinite),('fill_sentinel',fill),('negative',negative)]:record[label]+=int(bad.sum())
                    finite=selected[np.isfinite(selected)]
                    if finite.size:
                        lower,upper=float(finite.min()),float(finite.max())
                        record['minimum']=lower if record['minimum'] is None else min(record['minimum'],lower)
                        record['maximum']=upper if record['maximum'] is None else max(record['maximum'],upper)
                    if nonfinite.any() or fill.any() or (name in {'PRES_surface','TMP_2maboveground'} and negative.any()):
                        raise ValueError(f'Invalid archived hourly values on fixed support: case {case}, {name}, chunk {key}; {record}')
                    # Preserve raw signed/positive-variable undershoots through
                    # aggregation; original clipping belongs after scaling.
                    sums[case,channel]+=field.sum(axis=0,dtype=np.float64)
                    counts[case,channel]+=len(hours)
    if not np.all(counts==6):raise ValueError('Incomplete source-hour count')
    sums[:,1:]/=6.
    values=sums.astype(np.float32)
    if not np.isfinite(values[:,:,mask]).all():raise ValueError('Nonfinite aggregate on fixed support')
    aggregate_checks={name:{'negative_count':int((values[:,i][:,mask]<0).sum()),'minimum_raw':float(values[:,i][:,mask].min()),'maximum_raw':float(values[:,i][:,mask].max())} for i,name in enumerate(CHANNELS)}
    return values,{'channel_checks':checks,'aggregated_value_checks':aggregate_checks,'compressed_chunk_manifest':chunks,
                   'arithmetic':'chunk-order float64 accumulation, float32 output; precipitation sum, others mean',
                   'each_required_chunk_read_once':True}


def reference_roundtrip(raw,statistics,mask):
    """Exact original float32 writer transform, including second clipping pass."""
    result=np.asarray(raw,dtype=np.float32).copy()
    for i,name in enumerate(CHANNELS):
        result[i]=result[i]*FACTORS[i]
        if name in CLIPPED:result[i]=np.maximum(result[i],0)
    result[0]=np.where(result[0]<.1,0.,result[0])
    for i,name in enumerate(CHANNELS):
        mean,std=statistics['output'][name]['mean'],statistics['output'][name]['std']
        if not np.isfinite([mean,std]).all() or std<=0:raise ValueError(f'Invalid frozen normalization for {name}')
        result[i]=(result[i]-mean)/std
    result[:,~mask]=np.nan
    for i,name in enumerate(CHANNELS):
        mean,std=statistics['output'][name]['mean'],statistics['output'][name]['std']
        result[i]=result[i]*std+mean
        if name in CLIPPED:result[i]=np.maximum(result[i],0)
    result[0]=np.where(result[0]<.1,0.,result[0])
    return result


def independent_reconstruction(path,values,indices,mask):
    source=zarr.open_group(str(path),mode='r')['aorc'];h,w=mask.shape;rows=[]
    for case in sorted({0,len(indices)//2,len(indices)-1}):
        hourly=np.asarray(source[int(indices[case,0]):int(indices[case,-1])+1,:,:h,:w])
        expected=np.empty((8,h,w),np.float32)
        expected[0]=hourly[:,0].sum(axis=0,dtype=np.float64)
        expected[1:]=hourly[:,1:].mean(axis=0,dtype=np.float64)
        np.testing.assert_array_equal(values[case,:,mask],expected[:,mask].T)
        rows.append({'case':case,'exact_float32_equal_on_reference_mask':True})
    return rows


def prepare(synthetic_hourly,target_hourly,baseline_netcdf,statistics_path,output,reference_output,audit_output,times_output=None):
    paths=[Path(p) for p in (output,reference_output,audit_output)]
    if times_output is not None:paths.append(Path(times_output))
    if len(set(p.resolve() for p in paths))!=len(paths):raise ValueError('Output paths must differ')
    for p in paths:
        if p.exists() or p.with_name(p.name+'.partial').exists():raise FileExistsError(p)
        p.parent.mkdir(parents=True,exist_ok=True)
    with nc.Dataset(baseline_netcdf) as base:
        t=base['time'];dates=nc.num2date(t[:],t.units,getattr(t,'calendar','standard'))
        date_strings=[d.strftime('%Y-%m-%dT%H:%M:%S') for d in dates]
        requested=np.array(date_strings,dtype='datetime64[s]')
        if np.any(requested!=requested.astype('datetime64[h]')):raise ValueError('Baseline labels must be exact hours')
        lat=np.asarray(base['lat'][:]);lon=np.asarray(base['lon'][:])
        mask=np.isfinite(np.ma.filled(base['truth']['APCPsurface'][0],np.nan))
    if len(dates)!=200 or len(set(date_strings))!=200 or lat.shape!=lon.shape or lat.shape!=mask.shape or lat.shape!=(288,480) or mask.sum()!=128619:
        raise ValueError('Current-paper replay requires the original 200-date,288x480,128619-cell baseline contract')
    epoch=requested.astype(np.int64);statistics=json.loads(Path(statistics_path).read_text())
    report={'profile':'current_paper_matched_packed','dates':date_strings,'mask_cells':int(mask.sum()),
            'baseline_sha256':sha_file(baseline_netcdf),'statistics_sha256':sha_file(statistics_path),
            'numeric_convention':'raw stored AORC numbers; downstream legacy model pressure hPa and humidity10q',
            'time_semantics':'hourly labels t-5,...,t; precipitation(t-6h,t], other fields mean of same6records',
            'sources':{},'limitations':['Already spatially processed hourly sources retain their upstream processing history.',
            'Finite negative radiation/precipitation/humidity undershoots are reported and retained until original postaggregation clipping.']}
    partials=[p.with_name(p.name+'.partial') for p in paths[:2]]
    with nc.Dataset(partials[0],'w') as aligned,nc.Dataset(partials[1],'w') as ref:
        for ds in (aligned,ref):
            for name,size in [('time',200),('y',288),('x',480),('bounds',2)]:ds.createDimension(name,size)
            t=ds.createVariable('time','i8',('time',));t[:]=epoch;t.units='seconds since 1970-01-01 00:00:00';t.calendar='proleptic_gregorian';t.bounds='time_bounds'
            b=ds.createVariable('time_bounds','i8',('time','bounds'));b[:]=np.stack([epoch-21600,epoch],axis=-1);b.units=t.units
            for name,a in [('lat',lat),('lon',lon)]:ds.createVariable(name,'f4',('y','x'))[:]=a
            ds.createVariable('verification_mask','i1',('y','x'))[:]=mask
            ds.aggregation=report['time_semantics'];ds.preparation_profile=report['profile']
        aligned.numeric_convention=report['numeric_convention'];ref.humidity_convention='10 times physical specific humidity';ref.pressure_convention='hPa'
        for group_name,path in [('synthetic_input',synthetic_hourly),('reference_aorc',target_hourly)]:
            indices,meta,source_report=inspect_source(path,requested,lat,lon)
            values,checks=aggregate_archived(path,indices,meta,mask)
            checks['independent_reconstruction']=independent_reconstruction(path,values,indices,mask)
            report['sources'][group_name]={**source_report,**checks}
            group=aligned.createGroup(group_name)
            for i,name in enumerate(CHANNELS):
                v=group.createVariable(name,'f4',('time','y','x'),zlib=True,complevel=1,shuffle=True,chunksizes=(1,288,480),fill_value=np.nan)
                v.raw_to_physical_factor=PHYSICAL_FACTORS[i];v.raw_to_legacy_model_factor=FACTORS[i];v.physical_units_after_conversion=PHYSICAL_UNITS[i]
                v[:]=values[:,i]
            if group_name=='reference_aorc':
                truth=ref.createGroup('truth')
                variables=[truth.createVariable(name,'f4',('time','y','x'),zlib=True,complevel=1,chunksizes=(1,288,480),fill_value=np.nan) for name in OUTPUT_NAMES]
                for case in range(200):
                    physical=reference_roundtrip(values[case],statistics,mask)
                    for i,var in enumerate(variables):var[case]=physical[i]
            del values
    for partial,dest in zip(partials,paths):partial.rename(dest)
    report['artifacts']={p.name:{'sha256':sha_file(p),'bytes':p.stat().st_size} for p in paths[:2]}
    if times_output is not None:
        with paths[3].open('x') as handle:json.dump(date_strings,handle,indent=2)
        report['times_sha256']=sha_file(paths[3])
    with paths[2].open('x') as handle:json.dump(report,handle,indent=2,allow_nan=False)
    return report
