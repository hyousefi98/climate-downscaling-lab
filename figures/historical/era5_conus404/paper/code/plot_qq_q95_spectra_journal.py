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
OUT_STEM_DEFAULT = "qq_all_q95_and_spectra_journal"

LANDMASK_PATH_DEFAULT = Path("data/grids/aorc_land_mask_3km.nc")
LANDMASK_VAR_DEFAULT = "LANDMASK"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create a 2x2 journal panel: QQ(all) + QQ(q95) on top row, "
            "precip/temp spectra on bottom row, shared legend at bottom."
        )
    )
    p.add_argument("--diff-file", type=Path, default=DIFF_FILE_DEFAULT)
    p.add_argument("--reg-file", type=Path, default=REG_FILE_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--out-stem", type=str, default=OUT_STEM_DEFAULT)
    p.add_argument("--precip-var", type=str, default="RAIN_TOTAL")
    p.add_argument("--temp-var", type=str, default="T2")
    p.add_argument("--extreme-quantile", type=float, default=0.95)
    p.add_argument("--min-target-mean", type=float, default=0.1)
    p.add_argument("--max-samples", type=int, default=400000)
    p.add_argument("--n-quantiles", type=int, default=161)
    p.add_argument("--mask-mode", choices=["land", "ocean", "none"], default="land")
    p.add_argument("--landmask-path", type=Path, default=LANDMASK_PATH_DEFAULT)
    p.add_argument("--landmask-var", type=str, default=LANDMASK_VAR_DEFAULT)
    p.add_argument("--band-alpha", type=float, default=0.20, help="Alpha for CorrDiff 10-90 uncertainty band.")
    p.add_argument(
        "--qq-axis-max",
        type=float,
        default=None,
        help="Optional upper limit for QQ axes (applied to both x/y in panels a,b).",
    )
    p.add_argument(
        "--qq-axis-max-all",
        type=float,
        default=None,
        help="Optional upper limit for QQ(all) axis only.",
    )
    p.add_argument(
        "--qq-axis-max-q95",
        type=float,
        default=None,
        help="Optional upper limit for QQ(q95) axis only.",
    )
    p.add_argument(
        "--qq-tick-step-all",
        type=float,
        default=5.0,
        help="Tick interval for QQ(all) axes.",
    )
    p.add_argument(
        "--qq-tick-step-q95",
        type=float,
        default=10.0,
        help="Tick interval for QQ(q95) axes.",
    )
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
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


def _select_time_indices(truth_precip: xr.DataArray, extreme_quantile: float, min_target_mean: float) -> tuple[np.ndarray, np.ndarray]:
    ts = truth_precip.mean(dim=("y", "x"), skipna=True).values.astype(np.float64)
    finite = np.isfinite(ts)
    all_idx = np.where(finite)[0]
    if all_idx.size == 0:
        raise RuntimeError("No valid timesteps in truth precipitation.")

    qv = float(np.quantile(ts[finite], extreme_quantile))
    extreme = finite & (ts >= qv) & (ts >= float(min_target_mean))
    q95_idx = np.where(extreme)[0]
    if q95_idx.size == 0:
        raise RuntimeError("No q95 extreme timesteps selected; lower threshold/quantile.")

    return all_idx.astype(int), np.sort(q95_idx).astype(int)


def _safe_quantile(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.full_like(probs, np.nan, dtype=np.float64)
    return np.quantile(x, probs)


def _qq_curves_for_selection(
    truth_da: xr.DataArray,
    input_da: xr.DataArray,
    unet_da: xr.DataArray,
    diff_da: xr.DataArray,
    selected_idx: np.ndarray,
    probs: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> dict:
    truth_sel = np.asarray(truth_da.isel(time=selected_idx).values, dtype=np.float32)
    input_sel = np.asarray(input_da.isel(time=selected_idx).values, dtype=np.float32)
    unet_sel = np.asarray(unet_da.isel(time=selected_idx).values, dtype=np.float32)

    valid = np.isfinite(truth_sel) & np.isfinite(input_sel) & np.isfinite(unet_sel)
    valid_idx = np.flatnonzero(valid.ravel())
    if valid_idx.size == 0:
        raise RuntimeError("No valid points for QQ after filtering/masking.")

    if max_samples > 0 and valid_idx.size > max_samples:
        sample_idx = rng.choice(valid_idx, size=int(max_samples), replace=False)
    else:
        sample_idx = valid_idx

    truth_flat = truth_sel.ravel()[sample_idx].astype(np.float64)
    input_flat = input_sel.ravel()[sample_idx].astype(np.float64)
    unet_flat = unet_sel.ravel()[sample_idx].astype(np.float64)

    diff_var = diff_da
    if "ensemble" in diff_var.dims:
        n_members = int(diff_var.sizes["ensemble"])
        member_samples = np.empty((n_members, sample_idx.size), dtype=np.float32)
        for m in range(n_members):
            mem = np.asarray(diff_var.isel(ensemble=m, time=selected_idx).values, dtype=np.float32)
            member_samples[m, :] = mem.ravel()[sample_idx]
    else:
        n_members = 1
        mem = np.asarray(diff_var.isel(time=selected_idx).values, dtype=np.float32)
        member_samples = mem.ravel()[sample_idx][None, :]

    diff_mean_flat = np.nanmean(member_samples, axis=0).astype(np.float64)

    q_obs = _safe_quantile(truth_flat, probs)
    q_input = _safe_quantile(input_flat, probs)
    q_unet = _safe_quantile(unet_flat, probs)
    q_diff_mean = _safe_quantile(diff_mean_flat, probs)

    member_curves = []
    for m in range(member_samples.shape[0]):
        q_mem = _safe_quantile(member_samples[m, :], probs)
        member_curves.append(q_mem)
    member_curves_arr = np.asarray(member_curves, dtype=np.float64)
    q_diff_p10 = np.nanquantile(member_curves_arr, 0.10, axis=0)
    q_diff_p90 = np.nanquantile(member_curves_arr, 0.90, axis=0)

    all_vals = [q_obs, q_input, q_unet, q_diff_mean] + member_curves
    lo = min(float(np.nanmin(v)) for v in all_vals if np.isfinite(v).any())
    hi = max(float(np.nanmax(v)) for v in all_vals if np.isfinite(v).any())
    pad = max(0.04 * (hi - lo), 0.05)

    return {
        "q_obs": q_obs,
        "q_input": q_input,
        "q_unet": q_unet,
        "q_diff_mean": q_diff_mean,
        "q_diff_p10": q_diff_p10,
        "q_diff_p90": q_diff_p90,
        "axis_lo": lo - pad,
        "axis_hi": hi + pad,
        "n_points": int(sample_idx.size),
        "n_members": int(n_members),
        "n_times": int(selected_idx.size),
    }


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


def _reduce_diffusion_field(diff_da: xr.DataArray, reducer: str) -> np.ndarray:
    """
    Build a deterministic 3D field (time, y, x) from diffusion ensemble.
    reducer in {"mean", "median"}.
    Uses time-wise reduction to avoid loading full 4D ensemble into memory.
    """
    if "ensemble" not in diff_da.dims:
        return np.asarray(diff_da.values, dtype=np.float32)

    if reducer not in {"mean", "median"}:
        raise ValueError(f"Unsupported reducer '{reducer}'")

    n_time = int(diff_da.sizes["time"])
    ny = int(diff_da.sizes["y"])
    nx = int(diff_da.sizes["x"])
    out = np.empty((n_time, ny, nx), dtype=np.float32)

    for t in range(n_time):
        slab = np.asarray(diff_da.isel(time=t).values, dtype=np.float32)  # (ensemble, y, x)
        if reducer == "mean":
            out[t] = np.nanmean(slab, axis=0, dtype=np.float32)
        else:
            out[t] = np.nanmedian(slab, axis=0).astype(np.float32)
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

    # Deterministic diffusion mean field (not from averaging spectra).
    diff_mean_field = _reduce_diffusion_field(diff_da, reducer="mean")
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
    member_curves_arr = np.asarray(member_curves, dtype=np.float64)
    y_p10 = np.nanquantile(member_curves_arr, 0.10, axis=0)
    y_p90 = np.nanquantile(member_curves_arr, 0.90, axis=0)

    return {
        "freq": f,
        "target": s_truth,
        "input": s_input,
        "unet": s_unet,
        "corrdiff_mean": s_diff_mean,
        "corrdiff_p10": y_p10,
        "corrdiff_p90": y_p90,
    }


def _qq_plot_limits(qq: dict, forced_hi: float | None = None) -> tuple[float, float]:
    # Use target-quantile extent for QQ axes so the panel ends where x-curve ends.
    q_obs = np.asarray(qq["q_obs"], dtype=np.float64)
    finite_obs = q_obs[np.isfinite(q_obs)]
    if finite_obs.size == 0:
        raise RuntimeError("No finite target quantiles available to build QQ axis limits.")
    lo_raw = float(np.nanmin(finite_obs))
    hi_raw = float(np.nanmax(finite_obs))

    lo = float(np.floor(min(lo_raw, 0.0)))
    if forced_hi is None:
        hi = float(np.ceil(hi_raw))
    else:
        hi = float(forced_hi)
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _apply_linear_ticks(ax, lo: float, hi: float, step: float) -> None:
    if step <= 0:
        return
    start = step * np.ceil(lo / step)
    ticks = np.arange(start, hi + 0.5 * step, step, dtype=np.float64)
    if ticks.size == 0:
        ticks = np.array([lo, hi], dtype=np.float64)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 13.5,
            "ytick.labelsize": 13.5,
            "legend.fontsize": 11.5,
            "axes.linewidth": 0.8,
        }
    )

    root_diff = xr.open_dataset(args.diff_file)
    truth_diff = _open_grouped_with_coords(args.diff_file, "truth")
    input_diff = _open_grouped_with_coords(args.diff_file, "input")
    pred_diff = _open_grouped_with_coords(args.diff_file, "prediction")

    root_reg = xr.open_dataset(args.reg_file)
    pred_reg = _open_grouped_with_coords(args.reg_file, "prediction")

    try:
        mask_da = _make_domain_mask(root_diff, args.landmask_path, args.landmask_var, args.mask_mode)

        truth_p = _apply_mask(truth_diff[args.precip_var], mask_da)
        input_p = _apply_mask(input_diff[args.precip_var], mask_da)

        unet_p = pred_reg[args.precip_var]
        if "ensemble" in unet_p.dims:
            unet_p = unet_p.isel(ensemble=0)
        unet_p = _apply_mask(unet_p, mask_da)

        diff_p = _apply_mask(pred_diff[args.precip_var], mask_da)

        idx_all, idx_q95 = _select_time_indices(
            truth_precip=truth_p,
            extreme_quantile=float(args.extreme_quantile),
            min_target_mean=float(args.min_target_mean),
        )

        probs = np.linspace(0.001, 0.999, int(args.n_quantiles))

        qq_all = _qq_curves_for_selection(
            truth_da=truth_p,
            input_da=input_p,
            unet_da=unet_p,
            diff_da=diff_p,
            selected_idx=idx_all,
            probs=probs,
            max_samples=int(args.max_samples),
            rng=rng,
        )
        qq_q95 = _qq_curves_for_selection(
            truth_da=truth_p,
            input_da=input_p,
            unet_da=unet_p,
            diff_da=diff_p,
            selected_idx=idx_q95,
            probs=probs,
            max_samples=int(args.max_samples),
            rng=rng,
        )

        lat = np.asarray(root_diff["lat"].values, dtype=np.float64)
        lon = np.asarray(root_diff["lon"].values, dtype=np.float64)
        d_km = _estimate_grid_spacing_km(lat, lon)

        truth_t2 = truth_diff[args.temp_var]
        input_t2 = input_diff[args.temp_var]
        unet_t2 = pred_reg[args.temp_var]
        if "ensemble" in unet_t2.dims:
            unet_t2 = unet_t2.isel(ensemble=0)
        diff_t2 = pred_diff[args.temp_var]

        spec_precip = _spectrum_curves_for_var(
            truth_da=truth_diff[args.precip_var],
            input_da=input_diff[args.precip_var],
            unet_da=unet_p,
            diff_da=pred_diff[args.precip_var],
            d_km=d_km,
        )
        spec_temp = _spectrum_curves_for_var(
            truth_da=truth_t2,
            input_da=input_t2,
            unet_da=unet_t2,
            diff_da=diff_t2,
            d_km=d_km,
        )

        styles = {
            "Input": {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.0},
            "UNet": {"color": "#d62728", "linestyle": "-", "linewidth": 2.0},
            "CorrDiff mean": {"color": "#2ca02c", "linestyle": "-", "linewidth": 2.25},
            "Target": {"color": "#111111", "linestyle": "--", "linewidth": 1.8},
        }
        one_to_one_style = {"color": "#777777", "linestyle": ":", "linewidth": 1.6}
        band_color = styles["CorrDiff mean"]["color"]
        band_alpha = float(np.clip(args.band_alpha, 0.0, 1.0))

        fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.4))
        ax_a, ax_b, ax_c, ax_d = axes.ravel()

        # (a) QQ all-samples
        hi_all = args.qq_axis_max_all if args.qq_axis_max_all is not None else args.qq_axis_max
        lo, hi = _qq_plot_limits(qq_all, forced_hi=hi_all)
        ax_a.plot([lo, hi], [lo, hi], zorder=2, **one_to_one_style)
        ax_a.fill_between(
            qq_all["q_obs"],
            qq_all["q_diff_p10"],
            qq_all["q_diff_p90"],
            color=band_color,
            alpha=band_alpha,
            linewidth=0.0,
            zorder=1,
        )
        ax_a.plot(qq_all["q_obs"], qq_all["q_input"], zorder=3, **styles["Input"])
        ax_a.plot(qq_all["q_obs"], qq_all["q_unet"], zorder=3, **styles["UNet"])
        ax_a.plot(qq_all["q_obs"], qq_all["q_diff_mean"], zorder=4, **styles["CorrDiff mean"])
        ax_a.set_xlim(lo, hi)
        ax_a.set_ylim(lo, hi)
        _apply_linear_ticks(ax_a, lo, hi, float(args.qq_tick_step_all))
        ax_a.grid(True, alpha=0.35, linewidth=0.6)
        ax_a.set_xlabel("Target quantiles (mm)")
        ax_a.set_ylabel("Predicted quantiles (mm)")
        ax_a.set_title("(a) QQ: All Samples", loc="left", fontweight="bold")

        # (b) QQ q95
        hi_q95 = args.qq_axis_max_q95 if args.qq_axis_max_q95 is not None else args.qq_axis_max
        lo, hi = _qq_plot_limits(qq_q95, forced_hi=hi_q95)
        ax_b.plot([lo, hi], [lo, hi], zorder=2, **one_to_one_style)
        ax_b.fill_between(
            qq_q95["q_obs"],
            qq_q95["q_diff_p10"],
            qq_q95["q_diff_p90"],
            color=band_color,
            alpha=band_alpha,
            linewidth=0.0,
            zorder=1,
        )
        ax_b.plot(qq_q95["q_obs"], qq_q95["q_input"], zorder=3, **styles["Input"])
        ax_b.plot(qq_q95["q_obs"], qq_q95["q_unet"], zorder=3, **styles["UNet"])
        ax_b.plot(qq_q95["q_obs"], qq_q95["q_diff_mean"], zorder=4, **styles["CorrDiff mean"])
        ax_b.set_xlim(lo, hi)
        ax_b.set_ylim(lo, hi)
        _apply_linear_ticks(ax_b, lo, hi, float(args.qq_tick_step_q95))
        ax_b.grid(True, alpha=0.35, linewidth=0.6)
        ax_b.set_xlabel("Target quantiles (mm)")
        ax_b.set_ylabel("Predicted quantiles (mm)")
        ax_b.set_title("(b) QQ: q95 Extreme Times", loc="left", fontweight="bold")

        # (c) precip spectrum
        f = spec_precip["freq"]
        ax_c.fill_between(
            f,
            spec_precip["corrdiff_p10"],
            spec_precip["corrdiff_p90"],
            color=band_color,
            alpha=band_alpha,
            linewidth=0.0,
            zorder=1,
        )
        ax_c.loglog(f, spec_precip["input"], zorder=3, **styles["Input"])
        ax_c.loglog(f, spec_precip["unet"], zorder=3, **styles["UNet"])
        ax_c.loglog(f, spec_precip["corrdiff_mean"], zorder=4, **styles["CorrDiff mean"])
        ax_c.loglog(f, spec_precip["target"], zorder=2, **styles["Target"])
        ax_c.grid(True, which="both", alpha=0.25, linewidth=0.6)
        ax_c.set_xlabel("Spatial frequency (1/km)")
        ax_c.set_ylabel("Power spectral density")
        ax_c.set_title("(c) Spectrum: Precipitation", loc="left", fontweight="bold")

        # (d) temperature spectrum
        f = spec_temp["freq"]
        ax_d.fill_between(
            f,
            spec_temp["corrdiff_p10"],
            spec_temp["corrdiff_p90"],
            color=band_color,
            alpha=band_alpha,
            linewidth=0.0,
            zorder=1,
        )
        ax_d.loglog(f, spec_temp["input"], zorder=3, **styles["Input"])
        ax_d.loglog(f, spec_temp["unet"], zorder=3, **styles["UNet"])
        ax_d.loglog(f, spec_temp["corrdiff_mean"], zorder=4, **styles["CorrDiff mean"])
        ax_d.loglog(f, spec_temp["target"], zorder=2, **styles["Target"])
        ax_d.grid(True, which="both", alpha=0.25, linewidth=0.6)
        ax_d.set_xlabel("Spatial frequency (1/km)")
        ax_d.set_ylabel("Power spectral density")
        ax_d.set_title("(d) Spectrum: Temperature", loc="left", fontweight="bold")

        legend_handles = [
            Line2D([0], [0], **one_to_one_style, label="1:1 line"),
            Line2D([0], [0], **styles["Input"], label="Input"),
            Line2D([0], [0], **styles["UNet"], label="UNet"),
            Line2D([0], [0], **styles["CorrDiff mean"], label="CorrDiff mean"),
            Patch(facecolor=band_color, edgecolor="none", alpha=band_alpha, label="CorrDiff 10-90% band"),
            Line2D([0], [0], **styles["Target"], label="Target"),
        ]

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=6,
            frameon=True,
            bbox_to_anchor=(0.5, 0.02),
            columnspacing=1.3,
            handlelength=2.4,
        )

        fig.subplots_adjust(left=0.08, right=0.985, top=0.94, bottom=0.17, wspace=0.22, hspace=0.24)

        out_png = args.out_dir / f"{args.out_stem}.png"
        out_pdf = args.out_dir / f"{args.out_stem}.pdf"
        fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)

        print(f"Selected times: all={len(idx_all)}, q95={len(idx_q95)}")
        print(f"QQ sample points: all={qq_all['n_points']}, q95={qq_q95['n_points']}")
        print(f"Saved: {out_png.resolve()}")
        print(f"Saved: {out_pdf.resolve()}")

    finally:
        truth_diff.close()
        input_diff.close()
        pred_diff.close()
        pred_reg.close()
        root_diff.close()
        root_reg.close()


if __name__ == "__main__":
    main()
