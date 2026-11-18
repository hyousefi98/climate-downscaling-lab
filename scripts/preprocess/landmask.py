#!/usr/bin/env python3
"""Create a point-in-polygon land mask on the exact supplied grid.

The recovered paper script used Natural Earth 10m land polygons and a one-pixel
initial ocean dilation, followed by further loader buffering. New masks default
to no initial dilation; record each stage explicitly to avoid double buffering.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.ndimage import binary_dilation,generate_binary_structure
from data.regridding import target_coordinates


def polygon_mask(lat,lon,geometries,*,initial_coastal_buffer=0):
    from shapely import contains_xy,union_all
    if initial_coastal_buffer<0:raise ValueError('Buffer cannot be negative')
    mask=contains_xy(union_all(list(geometries)),lon,lat)
    if initial_coastal_buffer:mask=~binary_dilation(~mask,structure=generate_binary_structure(2,1),iterations=initial_coastal_buffer)
    return mask


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-grid',required=True,help='NetCDF reference grid or corrected Zarr store')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--shapefile',help='Supplied WGS84 land-polygon shapefile; no reprojection is guessed')
    source.add_argument('--natural-earth-resolution',choices=('10m','50m','110m'),help='Explicitly permits Cartopy to fetch/cache Natural Earth polygons')
    p.add_argument('--initial-coastal-buffer',type=int,default=0)
    p.add_argument('--output',required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists() or output.with_name(output.name+'.partial').exists():raise FileExistsError(output)
    from cartopy.io import shapereader
    path=Path(a.shapefile or shapereader.natural_earth(resolution=a.natural_earth_resolution,category='physical',name='land'))
    from pyproj import CRS
    projection=path.with_suffix('.prj')
    if not projection.exists() or not CRS.from_wkt(projection.read_text()).equals(CRS.from_epsg(4326),ignore_axis_order=True):
        raise ValueError('Land polygons must include a WGS84 longitude/latitude .prj file')
    reference=xr.open_zarr(a.reference_grid) if a.reference_grid.endswith('.zarr') else xr.open_dataset(a.reference_grid)
    lat,lon=target_coordinates(reference)
    mask=polygon_mask(lat,lon,shapereader.Reader(str(path)).geometries(),initial_coastal_buffer=a.initial_coastal_buffer)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.parent.glob(path.stem+'.*') if p.is_file()}
    ds=xr.Dataset({'LANDMASK':(('y','x'),mask.astype('uint8'))},coords={'lat':(('y','x'),lat),'lon':(('y','x'),lon)},
                  attrs={'method':'land polygon contains cell center; polygon boundary excluded',
                         'initial_coastal_buffer_pixels':a.initial_coastal_buffer,'polygon_files_sha256':json.dumps(hashes),
                         'polygon_source':path.name,'natural_earth_resolution':a.natural_earth_resolution or 'user-supplied'})
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_name(output.name+'.partial');ds.to_netcdf(partial);partial.rename(output)


if __name__=='__main__':main()
