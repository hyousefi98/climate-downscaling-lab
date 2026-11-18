#!/usr/bin/env python3
"""Request monthly ERA5 surface and pressure fields from CDS; dry-run needs no key."""
import argparse
import calendar
import json
from pathlib import Path
from data.source_profiles import ERA5_SURFACE, ERA5_PRESSURE, PRESSURE_LEVELS


def requests_for(year, month, bbox, radiation_profile='all_sky'):
    west,south,east,north=bbox
    common=dict(product_type=['reanalysis'],year=[str(year)],month=[f'{month:02d}'],
                day=[f'{day:02d}' for day in range(1,calendar.monthrange(year,month)[1]+1)],
                time=[f'{hour:02d}:00' for hour in range(24)],area=[north,west,south,east],
                data_format='netcdf',download_format='unarchived')
    surface=list(ERA5_SURFACE)
    if radiation_profile=='historical_clear_sky':
        surface[-1]='surface_thermal_radiation_downward_clear_sky'
    # Separate variables avoid mixed accumulated/instantaneous streams returning
    # multiple files in a ZIP from current CDS conversion services.
    return [('reanalysis-era5-single-levels',dict(common,variable=[v]),f'era5_surface_{v}_{year}_{month:02d}.nc') for v in surface]+[
            ('reanalysis-era5-pressure-levels',dict(common,variable=[v],pressure_level=[str(x) for x in PRESSURE_LEVELS]),f'era5_pressure_{v}_{year}_{month:02d}.nc') for v in ERA5_PRESSURE]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--years',nargs='+',type=int,required=True)
    p.add_argument('--months',nargs='+',type=int,default=list(range(1,13)))
    p.add_argument('--bbox',nargs=4,type=float,required=True,metavar=('WEST','SOUTH','EAST','NORTH'))
    p.add_argument('--output-dir',required=True)
    p.add_argument('--radiation-profile',choices=('all_sky','historical_clear_sky'),default='all_sky')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    plan=[item for year in args.years for month in args.months for item in requests_for(year,month,args.bbox,args.radiation_profile)]
    if len({item[2] for item in plan})!=len(plan):raise ValueError('Duplicate years/months would repeat an output path')
    if args.dry_run:print(json.dumps(plan,indent=2));return
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    for _,_,name in plan:
        if any((out/n).exists() for n in (name,name+'.partial',name+'.request.json')):raise FileExistsError(out/name)
    import cdsapi
    client=cdsapi.Client() # External ~/.cdsapirc; never store credentials here.
    for dataset,request,name in plan:
        partial=out/(name+'.partial')
        client.retrieve(dataset,request,str(partial))
        # Current CDS may return ZIP for mixed stream types despite requested format.
        with partial.open('rb') as handle:signature=handle.read(2)
        if signature==b'PK':
            raise ValueError(f'CDS returned ZIP at {partial}; extract explicitly and inspect before preprocessing')
        partial.rename(out/name)
        with (out/(name+'.request.json')).open('x') as handle:
            json.dump({'dataset':dataset,'request':request,'radiation_profile':args.radiation_profile},handle,indent=2)


if __name__=='__main__':main()
