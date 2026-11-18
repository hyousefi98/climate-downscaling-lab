import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "This script requires cartopy. Run with an environment that has cartopy "
        "(for example: conda run -n physicsnemo1 ...)."
    ) from exc


REGRESSION_FILE = Path("reg_wrf_12km_to_3km_200sam.nc")
DIFFUSION_FILE = Path("all_wrf_12km_to_3km_32ens_200sam.nc")
OUTPUT_DIR = Path("plots_rainiest_validation_logcolorbar_hires_custom_times")
RAIN_VAR = "RAIN_TOTAL"

DEFAULT_TIMES = [
    "2012-01-16 06:00:00",
    "2012-03-18 18:00:00",
    "2012-08-23 00:00:00",
    "2012-09-12 00:00:00",
]

ROW_LABELS = ["Input", "UNet", "CorrDiff Ensemble Mean", "Best Ensemble", "Target"]
PROFILE_COLORS = {
    "Input": "#1f77b4",
    "UNet": "#ff7f0e",
    "CorrDiff Ensemble Mean": "#2ca02c",
    "Best Ensemble": "#d62728",
    "Target": "#111111",
}


plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 21,
        "axes.labelsize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
    }
)


def _safe_token(text):
    return str(text).replace(":", "-").replace(" ", "_").replace("|", "_")


def _format_time_label(time_value):
    ts = pd.to_datetime(time_value)
    return ts.strftime("%Y-%m-%d | %H:%M:%S")


def _normalize_time_text(time_text):
    ts = pd.to_datetime(str(time_text))
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _build_time_lookup(time_values):
    lookup = {}
    for idx, t in enumerate(pd.to_datetime(np.asarray(time_values))):
        key = t.strftime("%Y-%m-%d %H:%M:%S")
        lookup[key] = int(idx)
    return lookup


def _kge(obs, pred):
    obs = np.asarray(obs).ravel()
    pred = np.asarray(pred).ravel()
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]
    pred = pred[mask]
    if obs.size < 2:
        return np.nan

    mean_obs = float(np.mean(obs))
    mean_pred = float(np.mean(pred))
    std_obs = float(np.std(obs))
    std_pred = float(np.std(pred))
    if mean_obs <= 0 or std_obs <= 0 or std_pred <= 0:
        return np.nan

    r = float(np.corrcoef(obs, pred)[0, 1])
    if not np.isfinite(r):
        return np.nan
    alpha = std_pred / std_obs
    beta = mean_pred / mean_obs
    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def _score_best_ensemble_for_index(truth_ds, diff_ds, index, var_name):
    n_ens = int(diff_ds.sizes["ensemble"])
    obs = truth_ds[var_name].isel(time=index).values
    best_ens = 0
    best_kge = -np.inf
    for ens in range(n_ens):
        pred = diff_ds[var_name].isel(ensemble=ens, time=index).values
        val = _kge(obs, pred)
        if np.isfinite(val) and val > best_kge:
            best_kge = float(val)
            best_ens = int(ens)
    return best_ens, best_kge


def _mask_ocean_as_nan(field, ocean_nan_mask):
    out = np.array(field, dtype=np.float32, copy=True)
    out[ocean_nan_mask] = np.nan
    return out


def _extract_maps(input_ds, truth_ds, reg_ds, diff_ds, var_name, idx, best_ens):
    target_raw = np.array(truth_ds[var_name].isel(time=idx).values, dtype=np.float32)
    ocean_nan_mask = ~np.isfinite(target_raw)

    input_raw = np.array(input_ds[var_name].isel(time=idx).values, dtype=np.float32)
    input_map = _mask_ocean_as_nan(input_raw, ocean_nan_mask)

    reg_var = reg_ds[var_name]
    if "ensemble" in reg_var.dims:
        reg_raw = np.array(reg_var.isel(ensemble=0, time=idx).values, dtype=np.float32)
    else:
        reg_raw = np.array(reg_var.isel(time=idx).values, dtype=np.float32)
    reg_map = _mask_ocean_as_nan(reg_raw, ocean_nan_mask)

    diff_slice = diff_ds[var_name].isel(time=idx)
    diff_mean_raw = np.array(diff_slice.mean("ensemble").values, dtype=np.float32)
    best_raw = np.array(diff_slice.isel(ensemble=best_ens).values, dtype=np.float32)
    diff_mean_map = _mask_ocean_as_nan(diff_mean_raw, ocean_nan_mask)
    best_map = _mask_ocean_as_nan(best_raw, ocean_nan_mask)

    target_map = _mask_ocean_as_nan(target_raw, ocean_nan_mask)

    # Order must match ROW_LABELS
    return [input_map, reg_map, diff_mean_map, best_map, target_map]


def _precip_cmap():
    stops = [
        (0.00, "#ffffff"),
        (0.06, "#f3faff"),
        (0.16, "#d9efff"),
        (0.32, "#7fc4ff"),
        (0.48, "#2b83ba"),
        (0.64, "#abdda4"),
        (0.78, "#ffffbf"),
        (0.90, "#fdae61"),
        (0.97, "#d7191c"),
        (1.00, "#7a0403"),
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("rain_white_start", stops, N=256)
    cmap.set_bad(color="#f2f2f2")
    return cmap


def _draw_context(ax, extent, az_geoms):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="#deefff", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f9f9f9", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAKES, facecolor="#deefff", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#444444", linewidth=0.5, zorder=3)
    states = cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lines", "10m", facecolor="none"
    )
    ax.add_feature(states, edgecolor="#666666", linewidth=0.28, zorder=3)
    if az_geoms:
        ax.add_geometries(
            az_geoms,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="#111111",
            linewidth=1.0,
            zorder=4,
        )


def _choose_colorbar_ticks(vmax, max_ticks=7, log_scale=False, log_vmin=0.05):
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
        candidates = np.array(
            [0, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200],
            dtype=float,
        )
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

    # Avoid the last two labels crowding together (common on compact horizontal bars).
    if len(ticks) >= 2:
        if log_scale:
            # If top two ticks are too close multiplicatively, drop the second-last.
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


def _profile_axis_from_lat(lat):
    lat_arr = np.asarray(lat)
    if lat_arr.ndim == 1:
        return lat_arr.astype(np.float32)
    if lat_arr.ndim == 2:
        return np.nanmean(lat_arr, axis=1).astype(np.float32)
    raise ValueError("Unsupported latitude array shape for profile plotting.")


def _row_label_display(label):
    if label == "CorrDiff Ensemble Mean":
        return "CorrDiff\nEnsemble Mean"
    return label


def main(times, reg_file, diff_file, out_file, dpi, log_colorbar, log_vmin):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    root = xr.open_dataset(reg_file)
    input_ds = xr.open_dataset(reg_file, group="input")
    truth_ds = xr.open_dataset(reg_file, group="truth")
    reg_ds = xr.open_dataset(reg_file, group="prediction")
    diff_ds = xr.open_dataset(diff_file, group="prediction")

    try:
        for ds in (input_ds, truth_ds, reg_ds):
            ds.coords.update(root.coords)

        lookup = _build_time_lookup(root["time"].values)

        selections = []
        for t_in in times:
            normalized = _normalize_time_text(t_in)
            if normalized not in lookup:
                avail = list(lookup.keys())
                sample_msg = ", ".join(avail[:5])
                raise ValueError(
                    f"Time not found: {t_in} (normalized: {normalized}). "
                    f"Example available timestamps: {sample_msg}"
                )
            idx = int(lookup[normalized])
            best_ens, best_kge = _score_best_ensemble_for_index(truth_ds, diff_ds, idx, RAIN_VAR)
            maps = _extract_maps(
                input_ds=input_ds,
                truth_ds=truth_ds,
                reg_ds=reg_ds,
                diff_ds=diff_ds,
                var_name=RAIN_VAR,
                idx=idx,
                best_ens=int(best_ens),
            )
            sample_max = max(
                [float(np.nanmax(m)) for m in maps if np.isfinite(m).any()],
                default=np.nan,
            )
            if np.isfinite(sample_max):
                vmax = float(max(np.ceil(sample_max), 1.0))
            else:
                vmax = 1.0

            selections.append(
                {
                    "time_index": idx,
                    "time_value": root["time"].values[idx],
                    "best_ensemble": int(best_ens),
                    "best_kge": float(best_kge) if np.isfinite(best_kge) else np.nan,
                    "maps": maps,
                    "vmax": vmax,
                }
            )

        lon = root["lon"].values
        lat = root["lat"].values
        extent = [
            float(np.nanmin(lon)),
            float(np.nanmax(lon)),
            float(np.nanmin(lat)),
            float(np.nanmax(lat)),
        ]

        shp_path = shpreader.natural_earth(
            resolution="10m", category="cultural", name="admin_1_states_provinces_lakes"
        )
        az_geoms = []
        for rec in shpreader.Reader(shp_path).records():
            if (
                rec.attributes.get("name") == "Arizona"
                and rec.attributes.get("admin") == "United States of America"
            ):
                az_geoms.append(rec.geometry)

        n_cols = len(selections)
        n_map_rows = len(ROW_LABELS)
        fig = plt.figure(
            figsize=(5.51 * n_cols, 3.15 * n_map_rows + 6.2),
            constrained_layout=False,
        )
        gs = fig.add_gridspec(
            nrows=n_map_rows + 4,
            ncols=n_cols,
            height_ratios=[1, 1, 1, 1, 1, 0.10, 0.10, 0.68, 0.42],
            hspace=0.16,
            wspace=0.08,
        )

        cmap = _precip_cmap()
        profile_x = _profile_axis_from_lat(lat)

        metadata_rows = []
        legend_handles = None
        legend_labels = None

        for cidx, sel in enumerate(selections):
            col_maps = sel["maps"]

            if log_colorbar:
                used_vmin = max(float(log_vmin), 1.0e-6)
                used_vmax = max(float(sel["vmax"]), used_vmin * 1.01)
                norm = mcolors.LogNorm(vmin=used_vmin, vmax=used_vmax)
            else:
                used_vmin = 0.0
                used_vmax = float(sel["vmax"])
                norm = mcolors.PowerNorm(gamma=0.45, vmin=0.0, vmax=used_vmax)

            col_im = None
            for ridx, (row_label, field) in enumerate(zip(ROW_LABELS, col_maps)):
                ax = fig.add_subplot(gs[ridx, cidx], projection=ccrs.PlateCarree())
                _draw_context(ax, extent, az_geoms)

                plot_field = np.array(field, dtype=np.float32, copy=True)
                if log_colorbar:
                    finite = np.isfinite(plot_field)
                    plot_field[finite & (plot_field <= 0)] = used_vmin

                col_im = ax.pcolormesh(
                    lon,
                    lat,
                    plot_field,
                    transform=ccrs.PlateCarree(),
                    shading="auto",
                    cmap=cmap,
                    norm=norm,
                    zorder=2,
                )

                if ridx == 0:
                    ts = _format_time_label(sel["time_value"])
                    date_str, time_str = ts.split(" | ")
                    col_tag = chr(ord("a") + cidx)
                    ax.set_title(
                        f"({col_tag}) {date_str}\n{time_str}",
                        fontsize=20,
                        fontweight="bold",
                        pad=8,
                    )

                if cidx == 0:
                    ax.set_ylabel(
                        _row_label_display(row_label),
                        fontsize=19,
                        fontweight="bold",
                        labelpad=16,
                    )
                else:
                    ax.set_ylabel("")

                ax.set_xticks([])
                ax.set_yticks([])

                mean_val = float(np.nanmean(field)) if np.isfinite(field).any() else np.nan
                max_val = float(np.nanmax(field)) if np.isfinite(field).any() else np.nan
                txt = f"mean={mean_val:.2f}\nmax={max_val:.2f}"
                if row_label == "Best Ensemble":
                    txt = f"{txt}\nens={int(sel['best_ensemble'])}"
                ax.text(
                    0.01,
                    0.01,
                    txt,
                    transform=ax.transAxes,
                    fontsize=16,
                    va="bottom",
                    ha="left",
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
                    zorder=5,
                )

            cbar_sub = gs[n_map_rows, cidx].subgridspec(1, 10, wspace=0.0)
            cax = fig.add_subplot(cbar_sub[0, 1:9])
            cbar = fig.colorbar(col_im, cax=cax, orientation="horizontal", extend="max")
            ticks = _choose_colorbar_ticks(
                vmax=used_vmax,
                max_ticks=6,
                log_scale=bool(log_colorbar),
                log_vmin=float(used_vmin),
            )
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([_format_tick_label(t) for t in ticks])
            cbar.ax.tick_params(labelsize=15)
            cbar.set_label("Precipitation (mm/6hr)", fontsize=14, labelpad=4)
            cbar.outline.set_linewidth(0.8)

            prof_sub = gs[n_map_rows + 2, cidx].subgridspec(1, 10, wspace=0.0)
            pax = fig.add_subplot(prof_sub[0, 1:9])
            for row_label, field in zip(ROW_LABELS, col_maps):
                profile_vals = np.nanmean(np.asarray(field, dtype=np.float32), axis=1)
                pax.plot(
                    profile_x,
                    profile_vals,
                    linewidth=2.6,
                    color=PROFILE_COLORS[row_label],
                    label=row_label,
                )

            pax.grid(True, alpha=0.35, linewidth=0.8)
            pax.set_xlabel("Latitude", fontsize=14, fontweight="bold")
            if cidx == 0:
                pax.set_ylabel("Profile\n(mm/6hr)", fontsize=16, fontweight="bold")
            else:
                pax.set_ylabel("")
            pax.tick_params(labelsize=16)

            y_max = max(
                [
                    float(np.nanmax(np.nanmean(np.asarray(f, dtype=np.float32), axis=1)))
                    for f in col_maps
                    if np.isfinite(f).any()
                ],
                default=1.0,
            )
            if not np.isfinite(y_max) or y_max <= 0:
                y_max = 1.0
            pax.set_ylim(0.0, y_max * 1.1)

            if cidx == 0:
                legend_handles, legend_labels = pax.get_legend_handles_labels()

            metadata_rows.append(
                {
                    "time": _normalize_time_text(sel["time_value"]),
                    "time_index": int(sel["time_index"]),
                    "best_ensemble": int(sel["best_ensemble"]),
                    "best_kge": float(sel["best_kge"]),
                    "vmax": float(sel["vmax"]),
                }
            )

        legend_ax = fig.add_subplot(gs[n_map_rows + 3, :])
        legend_ax.axis("off")
        if legend_handles and legend_labels:
            legend_ax.legend(
                legend_handles,
                legend_labels,
                loc="center",
                ncol=len(ROW_LABELS),
                frameon=True,
                fontsize=14.5,
                handlelength=2.8,
                columnspacing=1.0,
            )

        if out_file is None:
            token = "__".join(_safe_token(_normalize_time_text(t)) for t in times)
            out_file = OUTPUT_DIR / f"paper_grid_selected_times_profile_{token}.png"
        fig.savefig(out_file, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)

        meta_path = Path(out_file).with_suffix(".csv")
        pd.DataFrame(metadata_rows).to_csv(meta_path, index=False)

        print(f"Saved figure: {Path(out_file).resolve()}")
        print(f"Saved metadata: {meta_path.resolve()}")
    finally:
        input_ds.close()
        truth_ds.close()
        reg_ds.close()
        diff_ds.close()
        root.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Paper-style multi-date panel: columns are dates, rows are Input/UNet/"
            "CorrDiff mean/Best ensemble/Target + bottom profile row."
        )
    )
    parser.add_argument(
        "--times",
        nargs="+",
        default=DEFAULT_TIMES,
        help="List of timestamps (e.g. '2012-03-18 18:00:00'). Each becomes one column.",
    )
    parser.add_argument("--reg-file", type=str, default=str(REGRESSION_FILE))
    parser.add_argument("--diff-file", type=str, default=str(DIFFUSION_FILE))
    parser.add_argument("--out-file", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=420)
    parser.add_argument("--log-colorbar", action="store_true", help="Use log-scaled colorbars.")
    parser.add_argument(
        "--log-vmin",
        type=float,
        default=0.05,
        help="Minimum value for log color scale; values <= 0 are clipped to this.",
    )
    args = parser.parse_args()

    out = Path(args.out_file) if args.out_file else None
    main(
        times=args.times,
        reg_file=Path(args.reg_file),
        diff_file=Path(args.diff_file),
        out_file=out,
        dpi=int(args.dpi),
        log_colorbar=bool(args.log_colorbar),
        log_vmin=float(args.log_vmin),
    )
