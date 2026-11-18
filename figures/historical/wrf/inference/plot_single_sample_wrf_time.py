#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import cftime
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.io import shapereader
    from cartopy.io.shapereader import Reader

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False
    ccrs = None
    cfeature = None
    shapereader = None
    Reader = None


RAIN_WHITE_START_STOPS = [
    "#ffffff",
    "#f3faff",
    "#d9efff",
    "#7fc4ff",
    "#2b83ba",
    "#abdda4",
    "#ffffbf",
    "#fdae61",
    "#d7191c",
    "#7a0403",
]
RAIN_WHITE_START_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "rain_white_start",
    RAIN_WHITE_START_STOPS,
    N=256,
)
RAIN_WHITE_START_CMAP.set_bad("#f2f2f2")


VARIABLE_PLOT_SETTINGS = {
    "RAIN_TOTAL": (RAIN_WHITE_START_CMAP, 0.0, None, "mm/6hr"),
    "GLW": ("YlOrRd", None, None, "W/m²"),
    "SWDOWN": ("YlOrRd", 0.0, None, "W/m²"),
    "PSFC": ("viridis", None, None, "Pa"),
    "Q2": ("YlGnBu", 0.0, None, "kg/kg"),
    "T2": ("RdYlBu_r", None, None, "K"),
    "U10": ("RdBu_r", None, None, "m/s"),
    "V10": ("RdBu_r", None, None, "m/s"),
}

VARIABLE_DISPLAY_NAMES = {
    "RAIN_TOTAL": "Precipitation",
    "GLW": "LW Radiation",
    "SWDOWN": "SW Radiation",
    "PSFC": "Surface Pressure",
    "Q2": "Specific Humidity",
    "T2": "2m Temperature",
    "U10": "10m U Wind",
    "V10": "10m V Wind",
}

VARIABLE_ORDER = ["RAIN_TOTAL", "GLW", "SWDOWN", "PSFC", "Q2", "T2", "U10", "V10"]


def _normalize_time_text(text):
    out = str(text).strip().replace("T", " ")
    if "." in out:
        out = out.split(".", 1)[0]
    return out


def _decode_times(ds):
    v = ds["time"]
    calendar_name = getattr(v, "calendar", "standard")
    return cftime.num2date(v[:], units=v.units, calendar=calendar_name)


def _find_time_index(times, target_time, label):
    target_key = _normalize_time_text(target_time)
    matches = [i for i, t in enumerate(times) if _normalize_time_text(t) == target_key]
    if not matches:
        first = _normalize_time_text(times[0])
        last = _normalize_time_text(times[-1])
        raise ValueError(
            f"Time '{target_key}' not found in {label}. Available range: {first} .. {last}"
        )
    return int(matches[0]), target_key


def _ordered_common_variables(reg_ds, diff_ds):
    reg_pred = set(reg_ds["prediction"].variables.keys())
    reg_truth = set(reg_ds["truth"].variables.keys())
    reg_input = set(reg_ds["input"].variables.keys())
    diff_pred = set(diff_ds["prediction"].variables.keys())
    common = reg_pred & reg_truth & reg_input & diff_pred
    ordered = [v for v in VARIABLE_ORDER if v in common]
    extras = sorted(common - set(ordered))
    return ordered + extras


def _is_precip_channel(channel):
    return channel.upper() == "RAIN_TOTAL"


def _get_plot_settings(channel):
    if channel in VARIABLE_PLOT_SETTINGS:
        return VARIABLE_PLOT_SETTINGS[channel]
    return ("viridis", None, None, channel)


def _log_norm_for_arrays(arrays, floor=1e-3):
    finite = [a[np.isfinite(a)] for a in arrays if np.isfinite(a).any()]
    if finite:
        finite_all = np.concatenate(finite)
        positive = finite_all[finite_all > 0]
    else:
        positive = np.array([], dtype=np.float64)
        finite_all = np.array([], dtype=np.float64)

    log_vmin = max(float(np.nanmin(positive)) if positive.size else floor, floor)
    finite_max = float(np.nanmax(finite_all)) if finite_all.size else log_vmin * 10.0
    log_vmax = max(float(np.ceil(finite_max)), log_vmin * 10.0)

    clipped = [np.where(np.isfinite(a), np.maximum(a, log_vmin), np.nan) for a in arrays]
    return clipped, mcolors.LogNorm(vmin=log_vmin, vmax=log_vmax, clip=True), log_vmin, log_vmax


def _choose_colorbar_ticks(vmax, max_ticks=6, log_scale=False, log_vmin=0.05):
    if log_scale:
        candidates = np.array(
            [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200],
            dtype=float,
        )
        ticks = candidates[(candidates >= float(log_vmin)) & (candidates <= float(vmax))].tolist()
        if not ticks:
            ticks = [float(log_vmin)]
        if ticks[-1] < float(vmax):
            ticks.append(float(vmax))
    else:
        candidates = np.array([0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100], dtype=float)
        ticks = candidates[candidates <= vmax].tolist()
        if not ticks:
            ticks = [0.0]
        if not np.isclose(ticks[-1], vmax, rtol=0, atol=max(1e-6, 0.02 * max(vmax, 1.0))):
            ticks.append(float(vmax))

    ticks = sorted(set(float(t) for t in ticks))
    if len(ticks) > max_ticks:
        idx = np.linspace(0, len(ticks) - 1, num=max_ticks, dtype=int)
        ticks = [ticks[i] for i in idx]
        ticks = sorted(set(ticks))

    if len(ticks) >= 2:
        if log_scale:
            if ticks[-2] > 0 and (ticks[-1] / ticks[-2]) < 1.9:
                ticks.pop(-2)
        else:
            span = ticks[-1] - ticks[0]
            min_gap = 0.12 * span if span > 0 else 0.0
            if (ticks[-1] - ticks[-2]) < min_gap:
                ticks.pop(-2)
    return ticks


def _format_tick_label(t):
    t = float(t)
    if np.isclose(t, round(t), atol=1e-8):
        return str(int(round(t)))
    if t < 1.0:
        return f"{t:.2f}".rstrip("0").rstrip(".")
    return f"{t:.1f}".rstrip("0").rstrip(".")


def _prediction_slice(ds, channel, time_idx, sample_idx=0, ensemble_mean=False):
    pred = ds["prediction"][channel]
    if pred.ndim == 4:
        n_ens = pred.shape[0]
        if ensemble_mean:
            return np.nanmean(pred[:, time_idx], axis=0)
        pick = max(0, min(int(sample_idx), n_ens - 1))
        return pred[pick, time_idx]
    if pred.ndim == 3:
        return pred[time_idx]
    raise ValueError(f"Unexpected prediction dims for '{channel}': {pred.shape}")


def _best_ensemble_index_by_precip_rmse(reg_ds, diff_ds, reg_idx, diff_idx, precip_channel="RAIN_TOTAL"):
    truth = np.asarray(reg_ds["truth"][precip_channel][reg_idx], dtype=np.float32)
    pred = diff_ds["prediction"][precip_channel]

    if pred.ndim != 4:
        return 0

    n_ens = int(pred.shape[0])
    best_idx = 0
    best_rmse = np.inf
    for ens in range(n_ens):
        field = np.asarray(pred[ens, diff_idx], dtype=np.float32)
        mask = np.isfinite(truth) & np.isfinite(field)
        if not np.any(mask):
            continue
        rmse = float(np.sqrt(np.mean((field[mask] - truth[mask]) ** 2)))
        if np.isfinite(rmse) and rmse < best_rmse:
            best_rmse = rmse
            best_idx = ens
    return int(best_idx)


def _add_map_features(ax, highlight_state="Arizona"):
    if not HAS_CARTOPY:
        return
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="black")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="gray")

    try:
        shpfilename = shapereader.natural_earth(
            resolution="50m",
            category="cultural",
            name="admin_1_states_provinces",
        )
        reader = Reader(shpfilename)
        for record in reader.records():
            if record.attributes.get("name") == highlight_state:
                ax.add_geometries(
                    [record.geometry],
                    ccrs.PlateCarree(),
                    facecolor="none",
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=10,
                )
                break
    except Exception:
        pass


def _plot_panel(ax, data, extent, cmap, vmin, vmax, norm=None):
    kwargs = {
        "origin": "lower",
        "extent": extent,
        "cmap": cmap,
    }
    if norm is None:
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
    else:
        kwargs["norm"] = norm

    if HAS_CARTOPY:
        proj = ccrs.PlateCarree()
        im = ax.imshow(data, transform=proj, **kwargs)
        ax.set_extent(extent, crs=proj)
        _add_map_features(ax, highlight_state="Arizona")
        return im

    im = ax.imshow(data, **kwargs)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return im


def plot_wrf_time_all_variables(
    regression_file,
    corrdiff_file,
    target_time,
    output_dir,
    sample_unet=0,
    sample_corrdiff=0,
    corrdiff_ensemble_mean=False,
    dpi=300,
    font_scale=1.0,
    precip_log_vmin=0.05,
):
    if float(font_scale) <= 0:
        raise ValueError("--font_scale must be positive.")
    if float(precip_log_vmin) <= 0:
        raise ValueError("--precip_log_vmin must be > 0.")
    fs = float(font_scale)
    title_fs = 24 * fs
    var_name_fs = 22 * fs
    cbar_label_fs = 20 * fs
    cbar_tick_fs = 18 * fs

    plt.rcParams.update({
        "font.family": "serif",
        "axes.linewidth": 1.2 * fs,
        "xtick.major.width": 1.0 * fs,
        "ytick.major.width": 1.0 * fs,
    })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reg = nc.Dataset(regression_file, "r")
    diff = nc.Dataset(corrdiff_file, "r")
    try:
        reg_times = _decode_times(reg)
        diff_times = _decode_times(diff)
        reg_idx, target_key = _find_time_index(reg_times, target_time, "regression file")
        diff_idx, _ = _find_time_index(diff_times, target_time, "corrdiff file")
        best_ens_idx = _best_ensemble_index_by_precip_rmse(
            reg_ds=reg,
            diff_ds=diff,
            reg_idx=reg_idx,
            diff_idx=diff_idx,
            precip_channel="RAIN_TOTAL",
        )

        variables = _ordered_common_variables(reg, diff)
        if not variables:
            raise RuntimeError(
                "No common variables across regression input/truth/prediction and corrdiff prediction."
            )

        lon = reg["lon"][:]
        lat = reg["lat"][:]
        lat_1d = lat[:, 0] if lat.ndim == 2 else lat
        flip_lat = (len(lat_1d) > 1) and (lat_1d[0] > lat_1d[-1])
        extent = [
            float(np.nanmin(lon)),
            float(np.nanmax(lon)),
            float(np.nanmin(lat)),
            float(np.nanmax(lat)),
        ]

        subplot_kw = {"projection": ccrs.PlateCarree()} if HAS_CARTOPY else {}
        n_cols = 5
        fig, axs = plt.subplots(
            nrows=len(variables),
            ncols=n_cols,
            figsize=(27.5 * max(1.0, fs * 0.98), 3.6 * len(variables) * max(1.0, fs * 0.92)),
            subplot_kw=subplot_kw,
            constrained_layout=True,
        )

        if len(variables) == 1:
            axs = axs.reshape(1, -1)

        col_titles = ["Input", "Unet", "CorrDiff mean", f"Best Ensemble (ens={best_ens_idx})", "Target"]
        for c, title in enumerate(col_titles):
            axs[0, c].set_title(title, fontsize=title_fs, fontweight="bold", pad=28)

        for r, channel in enumerate(variables):
            x = np.array(reg["input"][channel][reg_idx], dtype=np.float32)
            y_unet = np.array(
                _prediction_slice(reg, channel, reg_idx, sample_idx=sample_unet, ensemble_mean=False),
                dtype=np.float32,
            )
            y_corrdiff_mean = np.array(
                _prediction_slice(
                    diff,
                    channel,
                    diff_idx,
                    sample_idx=sample_corrdiff,
                    ensemble_mean=True,
                ),
                dtype=np.float32,
            )
            y_corrdiff_best = np.array(
                _prediction_slice(
                    diff,
                    channel,
                    diff_idx,
                    sample_idx=best_ens_idx,
                    ensemble_mean=False,
                ),
                dtype=np.float32,
            )
            target = np.array(reg["truth"][channel][reg_idx], dtype=np.float32)

            if flip_lat:
                x = np.flipud(x)
                y_unet = np.flipud(y_unet)
                y_corrdiff_mean = np.flipud(y_corrdiff_mean)
                y_corrdiff_best = np.flipud(y_corrdiff_best)
                target = np.flipud(target)

            row_data = [x, y_unet, y_corrdiff_mean, y_corrdiff_best, target]
            cmap, forced_vmin, forced_vmax, cbar_label = _get_plot_settings(channel)

            finite_chunks = [d[np.isfinite(d)] for d in row_data if np.isfinite(d).any()]
            if finite_chunks:
                valid = np.concatenate(finite_chunks)
                row_vmin = float(np.nanmin(valid))
                row_vmax = float(np.nanmax(valid))
            else:
                row_vmin = 0.0
                row_vmax = 1.0

            vmin = forced_vmin if forced_vmin is not None else row_vmin
            vmax = forced_vmax if forced_vmax is not None else row_vmax
            norm = None

            if channel in {"U10", "V10"} and vmin < 0 < vmax:
                bound = max(abs(vmin), abs(vmax))
                vmin = -bound
                vmax = bound

            precip_log_bounds = None
            if _is_precip_channel(channel):
                row_data, norm, p_vmin, p_vmax = _log_norm_for_arrays(row_data, floor=float(precip_log_vmin))
                precip_log_bounds = (float(p_vmin), float(p_vmax))

            im = None
            for c, data in enumerate(row_data):
                im = _plot_panel(axs[r, c], data, extent, cmap, vmin, vmax, norm=norm)

            axs[r, 0].text(
                -0.15,
                0.5,
                VARIABLE_DISPLAY_NAMES.get(channel, channel),
                transform=axs[r, 0].transAxes,
                fontsize=var_name_fs,
                fontweight="bold",
                va="center",
                ha="center",
                rotation=90,
            )

            cb = plt.colorbar(im, ax=axs[r, :].tolist(), shrink=0.82, pad=0.02, fraction=0.022, aspect=18)
            if _is_precip_channel(channel) and precip_log_bounds is not None:
                p_vmin, p_vmax = precip_log_bounds
                ticks = _choose_colorbar_ticks(
                    vmax=float(p_vmax),
                    max_ticks=6,
                    log_scale=True,
                    log_vmin=float(p_vmin),
                )
                cb.set_ticks(ticks)
                cb.set_ticklabels([_format_tick_label(t) for t in ticks])
                cb.ax.minorticks_off()
            cb.set_label(cbar_label, fontsize=cbar_label_fs, labelpad=12, fontweight="bold")
            cb.ax.tick_params(labelsize=cbar_tick_fs, pad=6, width=1.5 * fs, length=6 * fs)
            for lbl in cb.ax.get_yticklabels():
                lbl.set_fontweight("bold")

        time_token = target_key.replace(" ", "_").replace(":", "-")
        out_file = output_dir / f"{time_token}.wrf_allvars_4col.png"
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_file}")
    finally:
        reg.close()
        diff.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot Input | Unet | CorrDiff | Target for one WRF timestamp across all common variables."
        )
    )
    parser.add_argument(
        "--regression_file",
        default="artifacts/wrf/reg_wrf_12km_to_3km_200sam.nc",
        help="Path to Unet/regression netCDF file",
    )
    parser.add_argument(
        "--corrdiff_file",
        default="artifacts/wrf/all_wrf_12km_to_3km_32ens_200sam.nc",
        help="Path to CorrDiff netCDF file",
    )
    parser.add_argument(
        "--target_time",
        default="2012-01-16 06:00:00",
        help="Timestamp to plot (e.g. '2012-01-16 06:00:00')",
    )
    parser.add_argument(
        "--output_dir",
        default="artifacts/wrf/plots_single_time_wrf_allvars",
        help="Directory for output figure",
    )
    parser.add_argument("--sample_unet", type=int, default=0, help="Unet ensemble member index")
    parser.add_argument("--sample_corrdiff", type=int, default=0, help="CorrDiff ensemble member index")
    parser.add_argument(
        "--corrdiff_ensemble_mean",
        action="store_true",
        help="Use CorrDiff ensemble mean instead of one member",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI")
    parser.add_argument(
        "--font_scale",
        type=float,
        default=1.0,
        help="Scale factor for panel titles, variable labels, and colorbar fonts.",
    )
    parser.add_argument(
        "--precip_log_vmin",
        type=float,
        default=0.05,
        help="Minimum precipitation value for log colorbar (RAIN_TOTAL only).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_wrf_time_all_variables(
        regression_file=args.regression_file,
        corrdiff_file=args.corrdiff_file,
        target_time=args.target_time,
        output_dir=args.output_dir,
        sample_unet=args.sample_unet,
        sample_corrdiff=args.sample_corrdiff,
        corrdiff_ensemble_mean=args.corrdiff_ensemble_mean,
        dpi=args.dpi,
        font_scale=args.font_scale,
        precip_log_vmin=args.precip_log_vmin,
    )
