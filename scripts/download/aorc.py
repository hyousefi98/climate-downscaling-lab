#!/usr/bin/env python3
"""Download a bounded anonymous NOAA AORC native-packed subset, not whole years."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from data.source_profiles import AORC_CHANNELS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("start", "end", "output"):p.add_argument("--"+name,required=True)
    p.add_argument("--bbox",nargs=4,type=float,required=True,metavar=("WEST","SOUTH","EAST","NORTH"))
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    import numpy as np
    if np.datetime64(args.start)>=np.datetime64(args.end):raise ValueError('start must precede end')
    years=range(int(args.start[:4]), int(str(np.datetime64(args.end)-np.timedelta64(1,'ns'))[:4])+1)
    plan={"sources":[f"s3://noaa-nws-aorc-v1-1-1km/{year}.zarr" for year in years],
          "start_inclusive":args.start,"end_exclusive":args.end,"bbox_wsen":args.bbox,
          "variables":list(AORC_CHANNELS),"output":args.output,"numeric_profile":"native_packed"}
    if args.dry_run:print(json.dumps(plan,indent=2));return
    from pathlib import Path
    if Path(args.output).exists() or Path(args.output+'.partial').exists():raise FileExistsError(args.output)
    import fsspec
    import xarray as xr
    from data.acquisition import subset
    from data.preprocessing import write_zarr_new
    pieces=[]; hashes={}
    for uri in plan['sources']:
        mapper=fsspec.get_mapper(uri,anon=True)
        hashes[uri]=hashlib.sha256(mapper['.zmetadata']).hexdigest()
        raw=xr.open_zarr(mapper,consolidated=True,mask_and_scale=False)
        raw=raw[list(AORC_CHANNELS)]
        pieces.append(subset(raw,start=args.start,end=args.end,bbox=args.bbox))
    ds=xr.concat(pieces,dim='time',join='exact') if len(pieces)>1 else pieces[0]
    ds.attrs.update(source_catalog='https://registry.opendata.aws/noaa-nws-aorc/',
                    source_metadata_sha256=json.dumps(hashes),retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    numeric_profile='native_packed',source_request=json.dumps(plan))
    # Clear native chunk encodings; preserve scale/fill attributes for explicit decode.
    for name in ds.variables:ds[name].encoding={}
    write_zarr_new(ds.chunk({'time':24,'latitude':128,'longitude':256}),args.output)


if __name__=='__main__':main()
