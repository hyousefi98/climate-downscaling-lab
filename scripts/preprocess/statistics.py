#!/usr/bin/env python3
"""Compute explicit common-period, common-mask training statistics."""
import argparse
import json
from pathlib import Path
import xarray as xr
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure
from data.preprocessing import training_statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("input", "target", "mask", "output", "start", "end"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--input-array", default="aorc")
    p.add_argument("--target-array", default="aorc")
    p.add_argument("--mask-variable", default="LANDMASK")
    p.add_argument("--coastal-buffer", type=int, default=0)
    p.add_argument("--invalid-times", choices=("reject", "drop"), default="reject")
    p.add_argument('--precip-threshold',type=float,default=.1)
    p.add_argument('--crop-shape',nargs=2,type=int,required=True,metavar=('Y','X'),help='Exact model crop before coastal dilation/masking')
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    if args.coastal_buffer<0:raise ValueError('Coastal buffer cannot be negative')
    inp,tgt=xr.open_zarr(args.input),xr.open_zarr(args.target)
    mask_ds=xr.open_dataset(args.mask)
    land=mask_ds[args.mask_variable].values.astype(bool)
    h,w=args.crop_shape
    if h<=0 or w<=0 or h>land.shape[0] or w>land.shape[1]:raise ValueError('Crop outside supplied mask')
    land=land[:h,:w]
    from data.regridding import target_coordinates
    lat,lon=target_coordinates(mask_ds)
    if not np.array_equal(lat[:h,:w],tgt.lat.values[:h,:w]) or not np.array_equal(lon[:h,:w],tgt.lon.values[:h,:w]):
        raise ValueError('Mask coordinates differ from the model grid; rebuild the mask on the selected grid')
    if args.coastal_buffer:
        land=~binary_dilation(~land,structure=generate_binary_structure(2,1),iterations=args.coastal_buffer)
    result=training_statistics(inp,tgt,land,input_array=args.input_array,target_array=args.target_array,
                               start=args.start,end=args.end,invalid_times=args.invalid_times,precip_threshold=args.precip_threshold,crop_shape=args.crop_shape)
    result['metadata']['coastal_buffer_pixels']=args.coastal_buffer
    result['metadata']['initial_mask_buffer_pixels']=mask_ds.attrs.get('initial_coastal_buffer_pixels','not recorded')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
