#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""
Plot a single sample from an RF NetCDF output.

Example:
  python plot_single_validation_rf.py --netcdf-file rf_aorc_validation.nc --sample 0 --out rf_val_0.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

try:
    import cftime
except Exception:
    cftime = None

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False


VARIABLE_PLOT_SETTINGS = {
    "APCP": ("Blues", 0, None, "Precipitation"),
    "TMP": ("RdYlBu_r", None, None, "Temperature"),
    "PRES": ("viridis", None, None, "Pressure"),
    "SPFH": ("YlGnBu", 0, None, "Specific Humidity"),
    "UGRD": ("RdBu_r", None, None, "U-Wind"),
    "VGRD": ("RdBu_r", None, None, "V-Wind"),
    "DLWRF": ("YlOrRd", None, None, "Downward LW"),
    "DSWRF": ("YlOrRd", None, None, "Downward SW"),
}


def _get_plot_settings(channel: str) -> Tuple[str, Optional[float], Optional[float], str]:
    for pattern, settings in VARIABLE_PLOT_SETTINGS.items():
        if pattern in channel:
            return settings
    return ("viridis", None, None, channel)


def _time_str(ds: nc.Dataset, idx: int) -> str:
    v = ds["time"]
    if hasattr(v, "units") and hasattr(v, "calendar") and cftime is not None:
        time_val = cftime.num2date(v[idx], units=v.units, calendar=v.calendar)
        return time_val.strftime("%Y-%m-%d %H:%M")
    return str(v[idx])


def _to_mesh(lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if lat.ndim == 1 and lon.ndim == 1:
        lon2, lat2 = np.meshgrid(lon, lat)
        return lat2, lon2
    return lat, lon


def _collect_limits(arrays):
    data = []
    for arr in arrays:
        if arr is None:
            continue
        valid = arr[np.isfinite(arr)]
        if valid.size:
            data.append(valid)
    if not data:
        return 0.0, 1.0
    combined = np.concatenate(data)
    return float(np.min(combined)), float(np.max(combined))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RF NetCDF outputs.")
    parser.add_argument("--netcdf-file", required=True, help="Path to NetCDF output file.")
    parser.add_argument("--sample", type=int, default=0, help="Time index to plot.")
    parser.add_argument("--ensemble", type=int, default=0, help="Ensemble index to plot.")
    parser.add_argument("--out", default="rf_validation.png", help="Output image file.")
    args = parser.parse_args()

    ds = nc.Dataset(args.netcdf_file, "r")

    lat = ds["lat"][:]
    lon = ds["lon"][:]
    lat, lon = _to_mesh(lat, lon)

    pred_group = ds["prediction"]
    truth_group = ds["truth"]
    input_group = ds["input"]

    channels = list(pred_group.variables.keys())
    n_channels = len(channels)
    if n_channels == 0:
        raise RuntimeError("No prediction variables found in NetCDF file.")

    time_label = _time_str(ds, args.sample)

    if _HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        fig, axs = plt.subplots(
            nrows=n_channels,
            ncols=3,
            figsize=(13, 3.4 * n_channels),
            constrained_layout=True,
            subplot_kw={"projection": proj},
        )
    else:
        fig, axs = plt.subplots(
            nrows=n_channels,
            ncols=3,
            figsize=(12, 3.2 * n_channels),
            constrained_layout=True,
        )

    if n_channels == 1:
        axs = np.expand_dims(axs, axis=0)

    for i, name in enumerate(channels):
        pred = pred_group[name][args.ensemble, args.sample]
        truth = truth_group[name][args.sample]
        inp = input_group[name][args.sample] if name in input_group.variables else None

        cmap, vmin, vmax, label = _get_plot_settings(name)
        auto_vmin, auto_vmax = _collect_limits([pred, truth, inp])
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax

        row = axs[i]
        titles = ["Input", "Prediction", "Truth"]
        panels = [inp, pred, truth]

        for ax, title, panel in zip(row, titles, panels):
            if _HAS_CARTOPY:
                im = ax.pcolormesh(
                    lon,
                    lat,
                    panel,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3)
                ax.set_title(f"{name} | {title}")
            else:
                im = ax.imshow(panel, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"{name} | {title}")
                ax.set_xticks([])
                ax.set_yticks([])

        cbar = fig.colorbar(im, ax=row.tolist(), shrink=0.85)
        cbar.set_label(label)

    fig.suptitle(f"Sample {args.sample} | {time_label}")
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
