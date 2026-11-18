#!/usr/bin/env python3
"""Download a bounded CONUS404 subset through its Planetary Computer STAC asset."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
VARS=('GLW','SWDOWN','PSFC','Q2','T2','U10','V10','PREC_ACC_NC')
CATALOG='https://planetarycomputer.microsoft.com/api/stac/v1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('start','end','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--bbox',nargs=4,type=float,required=True,metavar=('WEST','SOUTH','EAST','NORTH'))
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    plan=dict(catalog=CATALOG,collection='conus404',asset='zarr-abfs',variables=VARS,
              start_inclusive=args.start,end_exclusive=args.end,bbox_wsen=args.bbox,output=args.output)
    if args.dry_run:print(json.dumps(plan,indent=2));return
    if Path(args.output).exists() or Path(args.output+'.partial').exists():raise FileExistsError(args.output)
    import planetary_computer as pc
    import pystac_client
    import xarray as xr
    from data.acquisition import subset
    from data.preprocessing import write_zarr_new
    catalog=pystac_client.Client.open(CATALOG,modifier=pc.sign_inplace)
    asset=catalog.get_collection('conus404').assets['zarr-abfs']
    ds=xr.open_zarr(asset.href,storage_options=asset.extra_fields['xarray:storage_options'],**asset.extra_fields['xarray:open_kwargs'])
    absent=set(VARS)-set(ds)
    if absent:raise ValueError(f'Required CONUS404 variables missing: {absent}')
    ds=subset(ds,start=args.start,end=args.end,bbox=args.bbox,latitude='lat',longitude='lon')
    ds=ds[list(VARS)+[name for name in ('lat','lon','COSALPHA','SINALPHA') if name in ds and name not in VARS]]
    # Signed storage credentials are deliberately excluded from provenance.
    ds.attrs.update(source_catalog=CATALOG,source_collection='conus404',source_asset='zarr-abfs',
                    source_request=json.dumps(plan),retrieved_utc=datetime.now(timezone.utc).isoformat())
    for name in ds.variables:ds[name].encoding={}
    write_zarr_new(ds.chunk({'time':24}),args.output)


if __name__=='__main__':main()
