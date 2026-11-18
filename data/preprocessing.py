"""Missing-aware block means, interpolation, time windows, and paired statistics."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
import xarray as xr


def block_mean(field: xr.DataArray, factor: int, *, min_coverage: float = 1.0):
    """Equal-weight means; never sum precipitation over spatial pixels.

    Returns (mean, valid_fraction). Partial blocks at the domain boundary are
    trimmed. Missing source cells remain missing unless partial coverage is
    explicitly requested; valid zeros still count as observations.
    """
    if factor < 1 or not 0 < min_coverage <= 1:
        raise ValueError("factor>=1 and 0<min_coverage<=1 required")
    group = dict(y=factor, x=factor, boundary="trim")
    field = field.where(np.isfinite(field))
    coverage = field.notnull().coarsen(**group).mean()
    values = field.coarsen(**group).mean(skipna=True).where(coverage >= min_coverage)
    return values, coverage


def interpolate_index_grid(field, y_target, x_target, *, boundary="mask"):
    """Bilinear interpolation on known cell centers, with explicit edge policy.

    mask: no invented values beyond source centers. nearest: clamp coordinates
    at domain edges. extrapolate: historical linear edge extrapolation, which
    can overshoot. NaNs are retained, never replaced with zero.
    """
    if boundary not in ("mask", "nearest", "extrapolate"):
        raise ValueError(boundary)
    yy, xx = np.asarray(y_target), np.asarray(x_target)
    support = ((yy[:, None] >= field.y.min().item()) & (yy[:, None] <= field.y.max().item()) &
               (xx[None, :] >= field.x.min().item()) & (xx[None, :] <= field.x.max().item()))
    if boundary == "nearest":
        yi, xi = np.clip(yy, field.y.min().item(), field.y.max().item()), np.clip(xx, field.x.min().item(), field.x.max().item())
    else:
        yi, xi = yy, xx
    out = field.interp(y=xr.DataArray(yi, dims="y"), x=xr.DataArray(xi, dims="x"),
                       method="linear", kwargs={"fill_value": "extrapolate" if boundary == "extrapolate" else np.nan})
    out = out.assign_coords(y=yy, x=xx)
    return out, xr.DataArray(support if boundary == "mask" else np.ones_like(support), dims=("y", "x"), coords={"y": yy, "x": xx})


def aggregate_six_hour(field, *, precipitation=False, temporal_profile="ending", drop_boundary_partial=True):
    """Aggregate hourly END-labelled records, retaining per-cell coverage.

    ending: timestamp t uses labels t-5,...,t, precipitation interval(t-6h,t].
    historical_forward: timestamp t uses labels t,...,t+5; physical interval
    (t-1h,t+5]. This reproduces the old grouping, not end-aligned WRF intervals.
    A missing hour or pixel makes a six-hour value NaN. All-zero valid rainfall
    stays zero. Duplicate, unordered, or subhourly source labels are rejected.
    """
    if temporal_profile not in ("ending", "historical_forward"):
        raise ValueError(temporal_profile)
    field = field.where(np.isfinite(field))
    times = np.asarray(field.time.values).astype("datetime64[ns]")
    if not len(times) or np.isnat(times).any() or np.any(np.diff(times) <= np.timedelta64(0, "ns")):
        raise ValueError("Source time must be unique, increasing, and nonempty")
    if np.any(times != times.astype("datetime64[h]")):
        raise ValueError("Expected hourly labels")
    side = "right" if temporal_profile == "ending" else "left"
    options = dict(time="6h", closed=side, label=side, origin="start_day")
    count = field.notnull().resample(**options).sum()
    presence = xr.DataArray(np.ones(len(times), dtype="int16"), dims="time", coords={"time": field.time}).resample(**options).sum()
    value = field.resample(**options).sum(skipna=True) if precipitation else field.resample(**options).mean(skipna=True)
    value = value.where((count == 6) & (presence == 6))
    if drop_boundary_partial:
        # Drop incomplete calendar endpoints only; interior gaps stay visible.
        labels = value.time.values
        keep = np.ones(len(labels), bool)
        if len(labels):
            keep[0] = presence.values[0] == 6
            keep[-1] = presence.values[-1] == 6
        value, count = value.isel(time=keep), count.isel(time=keep)
    value.attrs.update(temporal_profile=temporal_profile, source_interval="hourly ending at label",
                       aggregation="sum" if precipitation else "mean", required_hour_count=6)
    return value, count.astype("uint8")


def cumulative_to_increment(field, *, reset="raise"):
    """Difference cumulative precipitation; first record remains missing.

    reset='restart' explicitly treats a negative jump as an accumulation reset.
    Never clip a reset silently: historical max(diff,0) could discard rainfall.
    """
    if reset not in ("raise", "restart"):
        raise ValueError(reset)
    previous = field.shift(time=1)
    diff = field - previous
    if reset == "raise" and bool((diff < 0).any().compute().item()):
        raise ValueError("Negative accumulation increment; supply reset='restart' only for documented restarts")
    return xr.where(diff < 0, field, diff) if reset == "restart" else diff


def stack_fields(fields, lat, lon, *, array_name, numeric_profile, spatial_support=None):
    names = list(fields)
    arrays = xr.align(*(fields[n] for n in names), join="exact")
    a = xr.concat(arrays, dim=xr.IndexVariable("channel", np.arange(len(names)))).transpose("time", "channel", "y", "x")
    ds = xr.Dataset({array_name: a.astype("float32")}, coords={
        "lat": (("y", "x"), np.asarray(lat)), "lon": (("y", "x"), np.asarray(lon)),
        f"{array_name}_variable": ("channel", np.array(names, dtype="S64")),
    })
    ds.attrs.update(numeric_profile=numeric_profile, missing_value_policy="NaN; never zero imputation", schema_version="1")
    if spatial_support is not None:
        ds["spatial_support"] = spatial_support.astype("uint8")
    return ds


def write_zarr_new(dataset, output):
    """Refuse existing output; keep failures in an identifiable partial store."""
    from numcodecs import Blosc
    output = Path(output).expanduser()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Refusing overwrite: {output} or {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoding = {n: {"compressor": Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)} for n in dataset.data_vars}
    dataset.to_zarr(partial, mode="w-", consolidated=True, encoding=encoding)
    partial.rename(output)


def training_statistics(input_ds, target_ds, mask, *, input_array, target_array, start, end, invalid_times="reject", precip_threshold=.1,crop_shape=None):
    """Statistics on identical times/support, [start,end), excluding held-out data.

    Optional explicit 'drop' removes entire common timestamps containing any
    invalid channel on the selected mask. It never filters each arm separately.
    """
    if invalid_times not in ("reject", "drop"):
        raise ValueError(invalid_times)
    if not np.isfinite(precip_threshold) or precip_threshold<0:
        raise ValueError('Precipitation threshold must be finite and nonnegative')
    if crop_shape is not None:
        h,w=map(int,crop_shape)
        if h<=0 or w<=0 or h>input_ds.sizes['y'] or w>input_ds.sizes['x'] or h>target_ds.sizes['y'] or w>target_ds.sizes['x']:
            raise ValueError('Crop must lie inside both source grids')
        input_ds=input_ds.isel(y=slice(h),x=slice(w));target_ds=target_ds.isel(y=slice(h),x=slice(w))
        mask=np.asarray(mask)[:h,:w]
    if input_ds.attrs.get("numeric_profile") not in ('physical','legacy_model','legacy_packed'):
        raise ValueError('Explicit recognized numeric_profile metadata required')
    if input_ds.attrs.get("numeric_profile") != target_ds.attrs.get("numeric_profile"):
        raise ValueError("Input and target numeric profiles differ")
    for key in ('temporal_profile','precipitation_interval','humidity_representation','pressure_units','wind_frame'):
        if input_ds.attrs.get(key) != target_ds.attrs.get(key):
            raise ValueError(f'Input and target {key} metadata differ')
    for coord in ("lat", "lon"):
        if not np.array_equal(input_ds[coord], target_ds[coord]):
            raise ValueError(f"Input and target {coord} grids differ")
    a, b = xr.align(input_ds[input_array], target_ds[target_array], join="exact", exclude={"channel"})
    select = (a.time >= np.datetime64(start)) & (a.time < np.datetime64(end))
    a, b = a.sel(time=select), b.sel(time=select)
    if not a.sizes["time"]:
        raise ValueError("Empty training period")
    if np.shape(mask) != (a.sizes['y'],a.sizes['x']):
        raise ValueError('Mask shape differs from paired grid')
    mask = xr.DataArray(np.asarray(mask, bool), dims=("y", "x"),coords={'y':a.y,'x':a.x})
    for ds in (input_ds, target_ds):
        if "spatial_support" in ds:
            mask = mask & ds.spatial_support.astype(bool)
    if not bool(mask.any()):
        raise ValueError("Empty verification/training support")
    good = ((np.isfinite(a) | ~mask).all(("channel", "y", "x")) &
            (np.isfinite(b) | ~mask).all(("channel", "y", "x"))).compute()
    dropped = a.time.values[~good.values].astype(str).tolist()
    if dropped and invalid_times == "reject":
        raise ValueError(f"{len(dropped)} timestamps have missing values on selected support; use explicit drop policy")
    a, b = a.sel(time=good), b.sel(time=good)
    if not a.sizes["time"]:
        raise ValueError("No complete common training timestamps")
    result = {}
    for label, arr, ds, array_name in [("input", a, input_ds, input_array), ("output", b, target_ds, target_array)]:
        names = [x.decode() if isinstance(x, bytes) else str(x) for x in ds[f"{array_name}_variable"].values]
        values = arr.where(mask).astype("float64")
        threshold=precip_threshold/(.1 if ds.attrs['numeric_profile']=='legacy_packed' else 1.)
        rainfall=xr.DataArray([name in {'APCP_surface','RAIN_TOTAL','PREC_ACC_NC'} for name in names],dims='channel',coords={'channel':arr.channel})
        values=xr.where(rainfall & (values<threshold),0.,values)
        means = values.mean(("time", "y", "x"), skipna=True).compute().values
        stds = values.std(("time", "y", "x"), skipna=True, ddof=0).compute().values
        result[label] = {n: {"mean": float(m), "std": float(s)} for n, m, s in zip(names, means, stds)}
    result["metadata"] = {"numeric_profile": input_ds.attrs.get("numeric_profile"), "start_inclusive": start,
                          "end_exclusive": end, "samples": a.sizes["time"], "mask_cells": int(mask.sum()),
                          "invalid_times": invalid_times, "dropped_timestamps": dropped,
                          'included_timestamps':a.time.values.astype(str).tolist(),'precip_threshold':float(precip_threshold),
                          'mask_shape':list(mask.shape),'mask_sha256':hashlib.sha256(np.asarray(mask,dtype=bool).tobytes(order='C')).hexdigest(),
                          'crop_shape':list(mask.shape)}
    for key in ('temporal_profile','precipitation_interval','humidity_representation','pressure_units','wind_frame'):
        result['metadata'][key]=input_ds.attrs.get(key)
    return result
