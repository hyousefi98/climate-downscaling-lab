#!/usr/bin/env python3
"""Recreate explicit historical paper reference coordinates; no data included."""
import argparse
from pathlib import Path
import numpy as np
import xarray as xr


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--bbox',nargs=4,type=float,default=[-117.5,30.5,-105.,38.],metavar=('WEST','SOUTH','EAST','NORTH'))
    p.add_argument('--factor',type=int,choices=[1,3,4,12,28],required=True)
    p.add_argument('--base-degrees',type=float,default=.008333,help='Historical paper approximation, not an exact kilometer spacing')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    west,south,east,north=a.bbox;step=a.base_degrees*a.factor
    lat=np.round(south+step*np.arange(int(np.floor((north-south)/step))+1),10)
    lon=np.round(west+step*np.arange(int(np.floor((east-west)/step))+1),10)
    y,x=np.meshgrid(lat,lon,indexing='ij')
    ds=xr.Dataset({'LAT2D':(('y','x'),y.astype('float32')),'LON2D':(('y','x'),x.astype('float32'))},
                  attrs={'grid_profile':'historical_labels','nominal_resolution_factor':a.factor,'base_degrees':a.base_degrees})
    out.parent.mkdir(parents=True,exist_ok=True);ds.to_netcdf(out,mode='w')


if __name__=='__main__':main()
