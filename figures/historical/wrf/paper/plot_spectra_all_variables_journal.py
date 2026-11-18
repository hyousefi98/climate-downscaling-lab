#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import periodogram
from scipy.spatial import cKDTree

DIFF_FILE_DEFAULT = Path("all_wrf_12km_to_3km_32ens_200sam.nc")
REG_FILE_DEFAULT = Path("reg_wrf_12km_to_3km_200sam.nc")
OUT_DIR_DEFAULT = Path("paper/figures")
OUT_STEM_DEFAULT = "spectra_all_variables_4x2_journal"
VAR_ORDER_DEFAULT = ["RAIN_TOTAL", "GLW", "SWDOWN", "PSFC", "Q2", "T2", "U10", "V10"]

LANDMASK_PATH_DEFAULT = Path("data/grids/aorc_land_mask_3km.nc")
LANDMASK_VAR_DEFAULT = "LANDMASK"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot 4x2 power spectra for all shared variables.")
    p.add_argument("--diff-file", type=Path, default=DIFF_FILE_DEFAULT)
    p.add_argument("--reg-file", type=Path, default=REG_FILE_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--out-stem", type=str, default=OUT_STEM_DEFAULT)
    p.add_argument("--variables", nargs="+", default=VAR_ORDER_DEFAULT)
    p.add_argument("--mask-mode", choices=["land", "ocean", "none"], default="land")
    p.add_argument("--landmask-path", type=Path, default=LANDMASK_PATH_DEFAULT)
    p.add_argument("--landmask-var", type=str, default=LANDMASK_VAR_DEFAULT)
    p.add_argument("--band-alpha", type=float, default=0.20)
    p.add_argument("--dpi", type=int, default=600)
    return p.parse_args()


def _open_grouped_with_coords(file_path: Path, group: str) -> xr.Dataset:
    root = xr.open_dataset(file_path)
    try:
        ds = xr.open_dataset(file_path, group=group)
        ds.coords.update(root.coords)
    finally:
        root.close()
    return ds


def _make_domain_mask(root_ds: xr.Dataset, landmask_path: Path, landmask_var: str, mode: str) -> xr.DataArray | None:
    mode = mode.lower()
    if mode == "none":
        return None
    if mode not in {"land", "ocean"}:
        raise ValueError("mask_mode must be one of: land, ocean, none")

    model_lat = np.asarray(root_ds["lat"].values, dtype=np.float64)
    model_lon = np.asarray(root_ds["lon"].values, dtype=np.float64)

    mask_ds = xr.open_dataset(landmask_path)
    try:
        src_lat = np.asarray(mask_ds["LAT2D"].values, dtype=np.float64)
        src_lon = np.asarray(mask_ds["LON2D"].values, dtype=np.float64)
        src_mask = np.asarray(mask_ds[landmask_var].values, dtype=np.float64)
    finally:
        mask_ds.close()

    if (
        src_mask.shape == model_lat.shape
        and np.nanmax(np.abs(src_lat - model_lat)) < 1e-6
        and np.nanmax(np.abs(src_lon - model_lon)) < 1e-6
    ):
        mapped = src_mask
    else:
        src_points = np.column_stack((src_lat.ravel(), src_lon.ravel()))
        tgt_points = np.column_stack((model_lat.ravel(), model_lon.ravel()))
        tree = cKDTree(src_points)
        _, idx = tree.query(tgt_points, k=1)
        mapped = src_mask.ravel()[idx].reshape(model_lat.shape)

    keep = mapped >= 0.5 if mode == "land" else mapped < 0.5
    dims = root_ds["lat"].dims
    coords = {d: root_ds.coords[d] for d in dims if d in root_ds.coords}
    return xr.DataArray(keep, dims=dims, coords=coords, name=f"{mode}_mask")


def _apply_mask(da: xr.DataArray, mask: xr.DataArray | None) -> xr.DataArray:
    return da if mask is None else da.where(mask)


def _haversine(lat1, lon1, lat2, lon2):
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    earth_radius = 6371000.0
    dlat_rad = lat2_rad - lat1_rad
    dlon_rad = lon2_rad - lon1_rad
    a = (
        np.sin(dlat_rad / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon_rad / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return earth_radius * c


def _estimate_grid_spacing_km(lat2d: np.ndarray, lon2d: np.ndarray) -> float:
    dx = _haversine(lat2d[0, 0], lon2d[0, 0], lat2d[1, 0], lon2d[1, 0])
    dy = _haversine(lat2d[0, 0], lon2d[0, 0], lat2d[0, 1], lon2d[0, 1])
    return float(0.5 * (dx + dy) / 1000.0)


def _avg_power_spectrum(arr_3d: np.ndarray, d_km: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr_3d, dtype=np.float64)
    freqs, psd = periodogram(arr, fs=1.0 / d_km, axis=-1)
    psd_mean = np.nanmean(psd, axis=(0, 1))
    return freqs, psd_mean


def _reduce_diffusion_mean_field(diff_da: xr.DataArray) -> np.ndarray:
    if "ensemble" not in diff_da.dims:
        return np.asarray(diff_da.values, dtype=np.float32)

    n_time = int(diff_da.sizes["time"])
    ny = int(diff_da.sizes["y"])
    nx = int(diff_da.sizes["x"])
    out = np.empty((n_time, ny, nx), dtype=np.float32)
    for t in range(n_time):
        slab = np.asarray(diff_da.isel(time=t).values, dtype=np.float32)
        out[t] = np.nanmean(slab, axis=0, dtype=np.float32)
    return out


def _spectrum_curves_for_var(
    truth_da: xr.DataArray,
    input_da: xr.DataArray,
    unet_da: xr.DataArray,
    diff_da: xr.DataArray,
    d_km: float,
) -> dict:
    truth = np.asarray(truth_da.values, dtype=np.float32)
    inp = np.asarray(input_da.values, dtype=np.float32)
    unet = np.asarray(unet_da.values, dtype=np.float32)

    freqs, s_truth = _avg_power_spectrum(truth, d_km)
    _, s_input = _avg_power_spectrum(inp, d_km)
    _, s_unet = _avg_power_spectrum(unet, d_km)

    diff_mean_field = _reduce_diffusion_mean_field(diff_da)
    _, s_diff_mean = _avg_power_spectrum(diff_mean_field, d_km)

    if "ensemble" in diff_da.dims:
        n_members = int(diff_da.sizes["ensemble"])
        member_specs = []
        for m in range(n_members):
            mem = np.asarray(diff_da.isel(ensemble=m).values, dtype=np.float32)
            _, s_mem = _avg_power_spectrum(mem, d_km)
            member_specs.append(np.asarray(s_mem, dtype=np.float64))
        member_specs = np.asarray(member_specs, dtype=np.float64)
    else:
        _, s_single = _avg_power_spectrum(np.asarray(diff_da.values, dtype=np.float32), d_km)
        member_specs = np.asarray(s_single, dtype=np.float64)[None, :]

    keep = np.isfinite(freqs) & (freqs > 0)
    for s in (s_truth, s_input, s_unet, s_diff_mean):
        keep &= np.isfinite(s) & (s > 0)

    f = np.asarray(freqs, dtype=np.float64)[keep]
    s_truth = np.asarray(s_truth, dtype=np.float64)[keep]
    s_input = np.asarray(s_input, dtype=np.float64)[keep]
    s_unet = np.asarray(s_unet, dtype=np.float64)[keep]
    s_diff_mean = np.asarray(s_diff_mean, dtype=np.float64)[keep]

    member_curves = []
    for m in range(member_specs.shape[0]):
        y = np.asarray(member_specs[m], dtype=np.float64)[keep]
        y[(~np.isfinite(y)) | (y <= 0)] = np.nan
        member_curves.append(y)
    member_curves = np.asarray(member_curves, dtype=np.float64)
    y_p10 = np.nanquantile(member_curves, 0.10, axis=0)
    y_p90 = np.nanquantile(member_curves, 0.90, axis=0)

    return {
        "freq": f,
        "target": s_truth,
        "input": s_input,
        "unet": s_unet,
        "corrdiff_mean": s_diff_mean,
        "corrdiff_p10": y_p10,
        "corrdiff_p90": y_p90,
    }


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 13.5,
            "ytick.labelsize": 13.5,
            "legend.fontsize": 12,
            "axes.linewidth": 0.8,
        }
    )

    root_diff = xr.open_dataset(args.diff_file)
    truth_diff = _open_grouped_with_coords(args.diff_file, "truth")
    input_diff = _open_grouped_with_coords(args.diff_file, "input")
    pred_diff = _open_grouped_with_coords(args.diff_file, "prediction")
    pred_reg = _open_grouped_with_coords(args.reg_file, "prediction")

    try:
        mask_da = _make_domain_mask(root_diff, args.landmask_path, args.landmask_var, args.mask_mode)

        available = set(truth_diff.data_vars) & set(input_diff.data_vars) & set(pred_diff.data_vars) & set(pred_reg.data_vars)
        vars_requested = [v for v in args.variables if v in available]
        if len(vars_requested) != 8:
            raise RuntimeError(
                f"Expected exactly 8 variables for a 4x2 grid, got {len(vars_requested)}: {vars_requested}"
            )

        lat = np.asarray(root_diff["lat"].values, dtype=np.float64)
        lon = np.asarray(root_diff["lon"].values, dtype=np.float64)
        d_km = _estimate_grid_spacing_km(lat, lon)

        styles = {
            "Input": {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.0},
            "UNet": {"color": "#d62728", "linestyle": "-", "linewidth": 2.0},
            "CorrDiff mean": {"color": "#2ca02c", "linestyle": "-", "linewidth": 2.25},
            "Target": {"color": "#111111", "linestyle": "--", "linewidth": 1.8},
        }
        band_color = styles["CorrDiff mean"]["color"]
        band_alpha = float(np.clip(args.band_alpha, 0.0, 1.0))

        fig, axes = plt.subplots(4, 2, figsize=(13.2, 20.0))
        flat_axes = axes.ravel()

        for i, var in enumerate(vars_requested):
            ax = flat_axes[i]

            truth_da = _apply_mask(truth_diff[var], mask_da)
            input_da = _apply_mask(input_diff[var], mask_da)
            unet_da = pred_reg[var]
            if "ensemble" in unet_da.dims:
                unet_da = unet_da.isel(ensemble=0)
            unet_da = _apply_mask(unet_da, mask_da)
            diff_da = _apply_mask(pred_diff[var], mask_da)

            spec = _spectrum_curves_for_var(
                truth_da=truth_da,
                input_da=input_da,
                unet_da=unet_da,
                diff_da=diff_da,
                d_km=d_km,
            )

            f = spec["freq"]
            ax.fill_between(
                f,
                spec["corrdiff_p10"],
                spec["corrdiff_p90"],
                color=band_color,
                alpha=band_alpha,
                linewidth=0.0,
                zorder=1,
            )
            ax.loglog(f, spec["input"], zorder=3, **styles["Input"])
            ax.loglog(f, spec["unet"], zorder=3, **styles["UNet"])
            ax.loglog(f, spec["corrdiff_mean"], zorder=4, **styles["CorrDiff mean"])
            ax.loglog(f, spec["target"], zorder=2, **styles["Target"])
            ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
            ax.set_title(var, fontweight="normal", pad=8)
            ax.set_xlabel("Spatial frequency (1/km)")
            ax.set_ylabel("Power spectral density")

        legend_handles = [
            Line2D([0], [0], **styles["Input"], label="Input"),
            Line2D([0], [0], **styles["UNet"], label="UNet"),
            Line2D([0], [0], **styles["CorrDiff mean"], label="CorrDiff mean"),
            Patch(facecolor=band_color, edgecolor="none", alpha=band_alpha, label="CorrDiff 10-90% band"),
            Line2D([0], [0], **styles["Target"], label="Target"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=5,
            frameon=True,
            bbox_to_anchor=(0.5, 0.02),
            columnspacing=1.4,
            handlelength=2.5,
        )

        fig.subplots_adjust(left=0.08, right=0.985, top=0.96, bottom=0.09, wspace=0.20, hspace=0.36)

        out_png = args.out_dir / f"{args.out_stem}.png"
        out_pdf = args.out_dir / f"{args.out_stem}.pdf"
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {out_png.resolve()}")
        print(f"Saved: {out_pdf.resolve()}")
        print(f"Variables: {', '.join(vars_requested)}")

    finally:
        truth_diff.close()
        input_diff.close()
        pred_diff.close()
        pred_reg.close()
        root_diff.close()


if __name__ == "__main__":
    main()
