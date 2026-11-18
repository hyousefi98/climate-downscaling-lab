"""Explicit rectilinear/projected/curvilinear bilinear interpolation.

Bilinear interpolation is not a conservative remap. Source fill values must
already be decoded to NaN; the default never extrapolates at domain edges.
"""
import numpy as np
import xarray as xr


def target_coordinates(reference):
    lat=reference['LAT2D'] if 'LAT2D' in reference else reference['lat']
    lon=reference['LON2D'] if 'LON2D' in reference else reference['lon']
    if lat.ndim==lon.ndim==1:
        lat,lon=np.meshgrid(lat,lon,indexing='ij')
    if np.shape(lat)!=np.shape(lon) or np.ndim(lat)!=2:
        raise ValueError('Target latitude/longitude must describe one 2D grid')
    return np.asarray(lat),np.asarray(lon)


def rectilinear(field, lat, lon, *, boundary='mask'):
    if boundary not in ('mask','extrapolate'):raise ValueError(boundary)
    field=field.assign_coords(longitude=(field.longitude+180)%360-180).sortby('longitude').sortby('latitude')
    result=field.interp(latitude=xr.DataArray(lat,dims=('y','x')),longitude=xr.DataArray(lon,dims=('y','x')),
                        method='linear',kwargs={'fill_value':None if boundary=='extrapolate' else np.nan,'bounds_error':False})
    return result.drop_vars(['latitude','longitude'],errors='ignore')


def projected(field, source_lat, source_lon, lat, lon, *, projection, boundary='mask', tolerance_m=30):
    """Rectilinear interpolation in a documented projection, with geometry check.

    CONUS404 uses '+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97
    +a=6370000 +b=6370000 +units=m'. Require caller to choose it explicitly.
    """
    if boundary not in ('mask','extrapolate'):raise ValueError(boundary)
    from pyproj import Proj
    p=Proj(projection)
    sx,sy=p(np.asarray(source_lon),np.asarray(source_lat))
    if np.max(np.abs(sx-sx[0:1,:]))>tolerance_m or np.max(np.abs(sy-sy[:,0:1]))>tolerance_m:
        raise ValueError('Source is not rectilinear in the supplied projection')
    field=field.assign_coords(x=sx[0],y=sy[:,0]).sortby('x').sortby('y')
    tx,ty=p(lon,lat)
    # Output axes differ from source dimensions until interpolation is complete.
    result=field.interp(x=xr.DataArray(tx,dims=('target_y','target_x')),y=xr.DataArray(ty,dims=('target_y','target_x')),
                        method='linear',kwargs={'fill_value':None if boundary=='extrapolate' else np.nan,'bounds_error':False})
    return result.drop_vars(['x','y'],errors='ignore').rename(target_y='y',target_x='x')


def curvilinear_regridder(source_lat, source_lon, lat, lon, *, weights_path):
    """ESMF bilinear weights; no extrapolation and no silent weight-file reuse."""
    from pathlib import Path
    import xesmf as xe
    path=Path(weights_path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    src=xr.Dataset({'lat':(('y','x'),source_lat),'lon':(('y','x'),source_lon)})
    dst=xr.Dataset({'lat':(('y','x'),lat),'lon':(('y','x'),lon)})
    return xe.Regridder(src,dst,'bilinear',filename=str(path),unmapped_to_nan=True,ignore_degenerate=False)
