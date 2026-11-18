#!/usr/bin/env python3
"""Build corrected AORC synthetic pairs from a native packed Zarr subset."""
import argparse
import json
import numpy as np
import xarray as xr
from data.source_profiles import AORC_CHANNELS, AORC_UNITS, NUMERIC_PROFILES, decode_aorc
from data.preprocessing import block_mean, interpolate_index_grid, aggregate_six_hour, stack_fields, write_zarr_new


def prepare(source, *, fine_factor=3, coarse_factor=12, numeric_profile="physical", temporal_profile="ending", boundary="mask", reference_grid=None, grid_profile="native_centers"):
    if coarse_factor % fine_factor or coarse_factor <= fine_factor:
        raise ValueError("Coarse factor must be an integer multiple of fine factor")
    ratio = coarse_factor // fine_factor
    source = source.sortby("latitude").sortby("longitude")
    if not np.issubdtype(source.time.dtype, np.datetime64):
        raise ValueError("Source time must decode to datetime64; supply its documented CF time units")
    if reference_grid is not None:
        lat_ref = np.asarray(reference_grid["LAT2D"])
        lon_ref = np.asarray(reference_grid["LON2D"])
        ny, nx = lat_ref.shape
        iy = int(np.abs(source.latitude.values - lat_ref.min()).argmin())
        ix = int(np.abs(source.longitude.values - lon_ref.min()).argmin())
        h, w = int(np.ceil(ny / ratio) * coarse_factor), int(np.ceil(nx / ratio) * coarse_factor)
        source = source.isel(latitude=slice(iy, iy+h), longitude=slice(ix, ix+w))
        if source.sizes["latitude"] != h or source.sizes["longitude"] != w:
            raise ValueError("Source subset lacks the padded native cells required by reference grid")
    else:
        if grid_profile == "historical_labels":
            raise ValueError("historical_labels requires --reference-grid")
        h = source.sizes["latitude"] // coarse_factor * coarse_factor
        w = source.sizes["longitude"] // coarse_factor * coarse_factor
        source = source.isel(latitude=slice(h), longitude=slice(w))
        ny, nx = h // fine_factor, w // fine_factor
    if min(h, w) < 2 * coarse_factor:
        raise ValueError("Need at least two coarse cells per spatial direction")
    lat = source.latitude.values[:ny*fine_factor].reshape(ny, fine_factor).mean(1)
    lon = source.longitude.values[:nx*fine_factor].reshape(nx, fine_factor).mean(1)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    if grid_profile == "historical_labels":
        lat2, lon2 = lat_ref, lon_ref
    fine, coarse, hours_fine, hours_coarse = {}, {}, {}, {}
    for name in AORC_CHANNELS:
        if name not in source:
            raise KeyError(f"Required source variable missing: {name}")
        raw = source[name].rename(latitude="y", longitude="x")
        fill = raw.attrs.get("missing_value", raw.attrs.get('_FillValue',raw.encoding.get("_FillValue", -32767)))
        physical = decode_aorc(raw, name, profile=numeric_profile, fill_value=fill)
        target, _ = block_mean(physical, fine_factor)
        target = target.isel(y=slice(ny), x=slice(nx)).assign_coords(y=np.arange(ny), x=np.arange(nx))
        low, _ = block_mean(physical, coarse_factor)
        low = low.assign_coords(y=(ratio-1)/2 + ratio*np.arange(low.sizes["y"]),
                                x=(ratio-1)/2 + ratio*np.arange(low.sizes["x"]))
        input_field, support = interpolate_index_grid(low, np.arange(ny), np.arange(nx), boundary=boundary)
        # Avoid unphysical negative rainfall from an explicitly selected extrapolation.
        # where/notnull preserves missing data; historical np.where(NaN>=0,NaN,0) did not.
        if name == "APCP_surface":
            input_field = input_field.clip(min=0)
        fine[name], hours_fine[name] = aggregate_six_hour(target, precipitation=name == "APCP_surface", temporal_profile=temporal_profile)
        coarse[name], hours_coarse[name] = aggregate_six_hour(input_field, precipitation=name == "APCP_surface", temporal_profile=temporal_profile)
    outputs = []
    for fields, counts in ((coarse, hours_coarse), (fine, hours_fine)):
        ds = stack_fields(fields, lat2, lon2, array_name="aorc", numeric_profile=numeric_profile, spatial_support=support)
        ds["hourly_valid_count"] = xr.concat(list(counts.values()), dim="channel").transpose("time", "channel", "y", "x").astype("uint8")
        units=dict(AORC_UNITS)
        if numeric_profile=='legacy_model':units['SPFH_2maboveground']='10 * kg kg-1'
        elif numeric_profile=='legacy_packed':units={name:'native packed numerical code' for name in AORC_CHANNELS}
        ds.attrs.update(temporal_profile=temporal_profile, boundary_policy=boundary, grid_profile=grid_profile,
                        fine_factor=fine_factor, coarse_factor=coarse_factor, source_fill_masked_before_aggregation=True,
                        spatial_coverage_required=1.0, units_json=json.dumps(units),
                        humidity_convention="10q" if numeric_profile == "legacy_model" else numeric_profile,
                        humidity_representation='specific_humidity' if numeric_profile=='physical' else numeric_profile,
                        pressure_units='hPa' if numeric_profile!='legacy_packed' else 'native packed numerical code',
                        wind_frame='earth',time_label='interval_end' if temporal_profile=='ending' else 'first_hour_label',
                        precipitation_interval='(t-6h,t]' if temporal_profile=='ending' else '(t-1h,t+5h]',
                        historical_missing_value_bug_reproduced=False)
        ds = ds.chunk({"time": 24, "channel": 1, "y": -1, "x": -1})
        outputs.append(ds)
    return tuple(outputs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, nargs="+", help="Packed native Zarr store(s); include previous-year hours when required")
    p.add_argument("--input-output", required=True)
    p.add_argument("--target-output", required=True)
    p.add_argument("--fine-factor", type=int, default=3)
    p.add_argument("--coarse-factor", type=int, default=12)
    p.add_argument("--numeric-profile", choices=NUMERIC_PROFILES, default="physical")
    p.add_argument("--temporal-profile", choices=("ending", "historical_forward"), default="ending")
    p.add_argument("--boundary", choices=("mask", "nearest", "extrapolate"), default="mask")
    p.add_argument("--grid-profile", choices=("native_centers", "historical_labels"), default="native_centers")
    p.add_argument("--reference-grid", help="Optional historical LAT2D/LON2D reference; use explicit historical_labels to retain its labels")
    args = p.parse_args()
    stores = [xr.open_zarr(path, mask_and_scale=False) for path in args.source]
    source = xr.concat(stores, dim="time", join="exact") if len(stores) > 1 else stores[0]
    reference = xr.open_dataset(args.reference_grid) if args.reference_grid else None
    pair = prepare(source, fine_factor=args.fine_factor, coarse_factor=args.coarse_factor,
                   numeric_profile=args.numeric_profile, temporal_profile=args.temporal_profile,
                   boundary=args.boundary, reference_grid=reference, grid_profile=args.grid_profile)
    # Preflight BOTH destinations before any expensive output write.
    from pathlib import Path
    if Path(args.input_output).resolve()==Path(args.target_output).resolve():
        raise ValueError('Input and target destinations must differ')
    for value in (args.input_output, args.target_output):
        path = Path(value)
        if path.exists() or path.with_name(path.name+".partial").exists():
            raise FileExistsError(path)
    for ds, path in zip(pair, (args.input_output, args.target_output)):
        write_zarr_new(ds, path)


if __name__ == "__main__":
    main()
