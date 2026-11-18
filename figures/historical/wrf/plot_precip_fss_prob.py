#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree


DIFF_FILE_DEFAULT = Path("all_wrf_12km_to_3km_32ens_200sam.nc")
REG_FILE_DEFAULT = Path("reg_wrf_12km_to_3km_200sam.nc")
OUT_DIR_DEFAULT = Path("plots_fss_prob")

PRECIP_VAR_DEFAULT = "RAIN_TOTAL"
LANDMASK_PATH_DEFAULT = Path("data/grids/aorc_land_mask_3km.nc")
LANDMASK_VAR_DEFAULT = "LANDMASK"


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


def _area_mean_timeseries(da: xr.DataArray) -> xr.DataArray:
    return da.mean(dim=("y", "x"), skipna=True)


def _parse_scales_to_cells(scales_km: list[float], dx_km: float) -> tuple[np.ndarray, np.ndarray]:
    cells = []
    eff_km = []
    for s in scales_km:
        c = max(1, int(round(float(s) / float(dx_km))))
        if c % 2 == 0:
            c += 1
        cells.append(c)
        eff_km.append(c * dx_km)
    return np.asarray(cells, dtype=int), np.asarray(eff_km, dtype=np.float64)


def _neighborhood_fraction(field: np.ndarray, valid2d: np.ndarray, window_cells: int) -> tuple[np.ndarray, np.ndarray]:
    area = float(window_cells * window_cells)
    valid3d = valid2d[None, :, :].astype(np.float32)

    finite = np.isfinite(field)
    data = np.where(finite, field, 0.0).astype(np.float32)
    support = finite.astype(np.float32)

    data *= valid3d
    support *= valid3d

    num = uniform_filter(data, size=(1, window_cells, window_cells), mode="constant", cval=0.0) * area
    den = uniform_filter(support, size=(1, window_cells, window_cells), mode="constant", cval=0.0) * area

    frac = np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=den > 0)
    return frac, den > 0


def _fss_from_fraction(obs_frac: np.ndarray, pred_frac: np.ndarray, valid3d: np.ndarray) -> float:
    if not np.any(valid3d):
        return float("nan")

    w = valid3d.astype(np.float32)
    n = float(np.sum(w))
    if n <= 0:
        return float("nan")

    num = float(np.sum(((pred_frac - obs_frac) ** 2) * w) / n)
    den = float(np.sum((pred_frac**2 + obs_frac**2) * w) / n)
    if den <= 0:
        return float("nan")

    return float(1.0 - (num / den))


def _flatten_valid(obs: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    o = np.asarray(obs).ravel()
    s = np.asarray(score).ravel()
    keep = np.isfinite(o) & np.isfinite(s)
    return o[keep], s[keep]


def _reliability_curve(prob: np.ndarray, obs01: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.clip(np.asarray(prob, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(obs01, dtype=np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    freq = np.full(n_bins, np.nan, dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.int64)

    for i in range(n_bins):
        if i < n_bins - 1:
            sel = (p >= edges[i]) & (p < edges[i + 1])
        else:
            sel = (p >= edges[i]) & (p <= edges[i + 1])
        c = int(np.sum(sel))
        count[i] = c
        if c > 0:
            freq[i] = float(np.mean(y[sel]))

    return centers, freq, count


def _roc_curve_auc(obs01: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(obs01, dtype=np.int8)
    s = np.asarray(score, dtype=np.float64)
    keep = np.isfinite(s)
    y = y[keep]
    s = s[keep]
    if y.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]

    distinct = np.where(np.diff(s))[0]
    thresh_idx = np.r_[distinct, y.size - 1]

    tps = np.cumsum(y)[thresh_idx].astype(np.float64)
    fps = (1 + thresh_idx).astype(np.float64) - tps

    tps = np.r_[0.0, tps]
    fps = np.r_[0.0, fps]

    tpr = tps / float(pos)
    fpr = fps / float(neg)

    if hasattr(np, "trapezoid"):
        auc = float(np.trapezoid(tpr, fpr))
    else:
        auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def _pr_curve_ap(obs01: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(obs01, dtype=np.int8)
    s = np.asarray(score, dtype=np.float64)
    keep = np.isfinite(s)
    y = y[keep]
    s = s[keep]
    if y.size == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 0.0]), float("nan")

    pos = int(np.sum(y == 1))
    if pos == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), float("nan")

    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]

    tp = np.cumsum(y).astype(np.float64)
    fp = np.cumsum(1 - y).astype(np.float64)

    distinct = np.where(np.diff(s))[0]
    thresh_idx = np.r_[distinct, y.size - 1]

    tp_t = tp[thresh_idx]
    fp_t = fp[thresh_idx]

    precision = tp_t / np.maximum(tp_t + fp_t, 1.0)
    recall = tp_t / float(pos)

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    ap = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
    return recall, precision, ap


def _plot_fss_grid(
    out_path_base: Path,
    thresholds: list[float],
    scales_km: np.ndarray,
    fss_input: dict[float, list[float]],
    fss_reg: dict[float, list[float]],
    fss_diff: dict[float, list[float]],
    fss_diff_members: dict[float, np.ndarray] | None,
    member_line_alpha: float,
    dpi: int,
):
    n_thr = len(thresholds)
    n_cols = min(2, n_thr)
    n_rows = int(np.ceil(n_thr / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.2 * n_cols, 6.5 * n_rows), squeeze=False)
    axes = axes.ravel()

    colors = {
        "Input": "#1f77b4",
        "UNet": "#d62728",
        "CorrDiff (ensemble prob)": "#2ca02c",
    }

    for i, thr in enumerate(thresholds):
        ax = axes[i]
        if fss_diff_members is not None and float(thr) in fss_diff_members:
            member_curves = np.asarray(fss_diff_members[float(thr)], dtype=np.float64)
            if member_curves.ndim == 2 and member_curves.shape[1] == scales_km.size:
                for m_idx, curve in enumerate(member_curves):
                    ax.plot(
                        scales_km,
                        curve,
                        linewidth=1.5,
                        color="#6abf69",
                        alpha=float(member_line_alpha),
                        label="CorrDiff members" if m_idx == 0 else None,
                        zorder=1,
                    )

        ax.plot(scales_km, fss_input[thr], marker="o", linewidth=2.8, color=colors["Input"], label="Input")
        ax.plot(scales_km, fss_reg[thr], marker="s", linewidth=2.8, color=colors["UNet"], label="UNet")
        ax.plot(
            scales_km,
            fss_diff[thr],
            marker="^",
            linewidth=3.1,
            color=colors["CorrDiff (ensemble prob)"],
            label="CorrDiff (ensemble prob)",
        )

        ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Spatial scale (km)")
        ax.set_ylabel("FSS")
        ax.set_title(f"Threshold = {thr:g} mm", pad=8)
        ax.grid(True, color="#bdbdbd", alpha=0.30, linewidth=0.7)
        ax.legend(loc="lower right", frameon=True)

    for j in range(n_thr, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    png = out_path_base.with_suffix(".png")
    pdf = out_path_base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def _plot_prob_skill_panels(
    out_path_base: Path,
    threshold_mm: float,
    rel_curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    brier_scores: dict[str, float],
    brier_ref: float,
    brier_skill: dict[str, float],
    roc_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    roc_auc: dict[str, float],
    pr_curves: dict[str, tuple[np.ndarray, np.ndarray]],
    pr_auc: dict[str, float],
    climatology: float,
    n_samples: int,
    dpi: int,
):
    fig, axes = plt.subplots(1, 3, figsize=(23, 6.8))

    colors = {
        "CorrDiff (ensemble prob)": "#2ca02c",
        "UNet": "#d62728",
        "Input": "#1f77b4",
    }
    markers = {"CorrDiff (ensemble prob)": "o", "UNet": "s", "Input": "^"}
    line_styles = {"CorrDiff (ensemble prob)": "-", "UNet": "--", "Input": "-."}

    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle="--", color="#222222", linewidth=1.5, label="Perfect")
    for name in ("CorrDiff (ensemble prob)", "UNet", "Input"):
        rel_centers, rel_freq, _ = rel_curves[name]
        valid = np.isfinite(rel_freq)
        ax.plot(
            rel_centers[valid],
            rel_freq[valid],
            marker=markers[name],
            linestyle=line_styles[name],
            linewidth=2.8,
            color=colors[name],
            label=f"{name} (BS={brier_scores[name]:.4f}, BSS={brier_skill[name]:.3f})",
        )
    ax.axhline(climatology, linestyle=":", linewidth=1.3, color="#7f7f7f", label=f"Climatology={climatology:.3f}")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability ({threshold_mm:g} mm)")
    ax.grid(True, color="#bdbdbd", alpha=0.30, linewidth=0.7)

    txt = (
        f"BS_ref={brier_ref:.4f}\n"
        f"N={n_samples:,}"
    )
    ax.text(
        0.03,
        0.97,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=14,
        bbox=dict(facecolor="white", edgecolor="#999999", boxstyle="round,pad=0.35", alpha=0.9),
    )
    ax.legend(loc="lower right", frameon=True)

    ax = axes[1]
    for name in ("CorrDiff (ensemble prob)", "UNet", "Input"):
        fpr, tpr = roc_curves[name]
        ax.plot(fpr, tpr, linewidth=2.8, color=colors[name], label=f"{name} (AUC={roc_auc[name]:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#222222", linewidth=1.2, alpha=0.9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC")
    ax.grid(True, color="#bdbdbd", alpha=0.30, linewidth=0.7)
    ax.legend(loc="lower right", frameon=True)

    ax = axes[2]
    for name in ("CorrDiff (ensemble prob)", "UNet", "Input"):
        rec, prec = pr_curves[name]
        ax.plot(rec, prec, linewidth=2.8, color=colors[name], label=f"{name} (AP={pr_auc[name]:.3f})")
    ax.axhline(climatology, linestyle="--", color="#222222", linewidth=1.2, alpha=0.9, label="No-skill")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall")
    ax.grid(True, color="#bdbdbd", alpha=0.30, linewidth=0.7)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    png = out_path_base.with_suffix(".png")
    pdf = out_path_base.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot precipitation skill diagnostics:\n"
            "1) FSS vs spatial scale (Input / UNet / CorrDiff probabilistic)\n"
            "2) Reliability + ROC + PR for threshold exceedance"
        )
    )
    p.add_argument("--diff-file", type=Path, default=DIFF_FILE_DEFAULT)
    p.add_argument("--reg-file", type=Path, default=REG_FILE_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--var", default=PRECIP_VAR_DEFAULT)

    p.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 5.0, 10.0, 20.0])
    p.add_argument("--scales-km", type=float, nargs="+", default=[3.0, 9.0, 15.0, 27.0, 45.0, 75.0, 120.0])
    p.add_argument("--dx-km", type=float, default=3.0, help="Model grid spacing in km.")

    p.add_argument("--time-selection", choices=["all", "extreme"], default="all")
    p.add_argument("--extreme-quantile", type=float, default=0.99)
    p.add_argument("--min-target-mean", type=float, default=0.1)
    p.add_argument("--top-n", type=int, default=0, help="Keep top-N wettest times when --time-selection extreme (0=all).")
    p.add_argument("--max-times", type=int, default=0, help="Random cap on selected times for speed (0=all).")

    p.add_argument("--mask-mode", choices=["land", "ocean", "none"], default="land")
    p.add_argument("--landmask-path", type=Path, default=LANDMASK_PATH_DEFAULT)
    p.add_argument("--landmask-var", default=LANDMASK_VAR_DEFAULT)

    p.add_argument("--reliability-bins", type=int, default=12)
    p.add_argument("--max-samples", type=int, default=1200000, help="Random cap for ROC/PR/Reliability points (0=all).")
    p.add_argument(
        "--fss-member-lines",
        action="store_true",
        help="Also compute and plot one FSS-vs-scale line per diffusion ensemble member.",
    )
    p.add_argument(
        "--fss-member-line-alpha",
        type=float,
        default=0.2,
        help="Alpha for per-member FSS lines when --fss-member-lines is enabled.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main():
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 19,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "figure.titlesize": 20,
        }
    )

    print(f"Loading diffusion file: {args.diff_file}")
    root_diff = xr.open_dataset(args.diff_file)
    truth_ds = xr.open_dataset(args.diff_file, group="truth")
    input_ds = xr.open_dataset(args.diff_file, group="input")
    pred_diff_ds = xr.open_dataset(args.diff_file, group="prediction")
    for ds in (truth_ds, input_ds, pred_diff_ds):
        ds.coords.update(root_diff.coords)

    print(f"Loading regression file: {args.reg_file}")
    root_reg = xr.open_dataset(args.reg_file)
    pred_reg_ds = xr.open_dataset(args.reg_file, group="prediction")
    pred_reg_ds.coords.update(root_reg.coords)

    mask_da = _make_domain_mask(root_diff, args.landmask_path, args.landmask_var, args.mask_mode)
    valid2d = np.ones(root_diff["lat"].shape, dtype=bool) if mask_da is None else np.asarray(mask_da.values, dtype=bool)

    truth_da = _apply_mask(truth_ds[args.var], mask_da)
    input_da = _apply_mask(input_ds[args.var], mask_da)

    reg_da = pred_reg_ds[args.var]
    if "ensemble" in reg_da.dims:
        reg_da = reg_da.isel(ensemble=0)
    reg_da = _apply_mask(reg_da, mask_da)

    diff_da = _apply_mask(pred_diff_ds[args.var], mask_da)

    truth_ts = _area_mean_timeseries(truth_da).values
    finite_ts = np.isfinite(truth_ts)
    if not np.any(finite_ts):
        raise RuntimeError("No valid target area-mean values after masking.")

    qv = float(np.quantile(truth_ts[finite_ts], args.extreme_quantile))
    if args.time_selection == "all":
        selected_idx = np.where(finite_ts)[0]
        print(f"Selected all timesteps: {selected_idx.size}")
    else:
        extreme = finite_ts & (truth_ts >= qv) & (truth_ts >= args.min_target_mean)
        selected_idx = np.where(extreme)[0]
        if selected_idx.size == 0:
            raise RuntimeError("No extreme events selected. Lower --extreme-quantile or --min-target-mean.")
        order = np.argsort(truth_ts[selected_idx])[::-1]
        selected_idx = selected_idx[order]
        if args.top_n > 0:
            selected_idx = selected_idx[: args.top_n]
        selected_idx = np.sort(selected_idx)
        print(
            f"Selected {selected_idx.size} extreme timesteps "
            f"(quantile >= {args.extreme_quantile:.2f}, threshold={qv:.3f} mm)."
        )

    if args.max_times > 0 and selected_idx.size > args.max_times:
        selected_idx = np.sort(rng.choice(selected_idx, size=args.max_times, replace=False))
        print(f"Randomly downsampled timesteps to {selected_idx.size} using seed={args.seed}.")

    truth_sel = truth_da.isel(time=selected_idx).values.astype(np.float32)
    input_sel = input_da.isel(time=selected_idx).values.astype(np.float32)
    reg_sel = reg_da.isel(time=selected_idx).values.astype(np.float32)
    diff_sel_da = diff_da.isel(time=selected_idx)

    cells, scales_eff_km = _parse_scales_to_cells(args.scales_km, args.dx_km)
    print("FSS scales (cells):", cells.tolist())
    print("FSS scales (km):", [round(v, 2) for v in scales_eff_km.tolist()])

    fss_rows: list[dict] = []
    metrics_rows: list[dict] = []
    rel_rows: list[dict] = []

    fss_input: dict[float, list[float]] = {}
    fss_reg: dict[float, list[float]] = {}
    fss_diff: dict[float, list[float]] = {}
    fss_diff_members: dict[float, np.ndarray] = {}

    for thr in args.thresholds:
        print(f"Computing threshold {thr:g} mm ...")
        obs_event = np.where(np.isfinite(truth_sel), truth_sel >= thr, np.nan).astype(np.float32)
        inp_event = np.where(np.isfinite(input_sel), input_sel >= thr, np.nan).astype(np.float32)
        reg_event = np.where(np.isfinite(reg_sel), reg_sel >= thr, np.nan).astype(np.float32)

        diff_prob_da = (diff_sel_da >= thr).mean("ensemble", skipna=True)
        diff_prob = diff_prob_da.values.astype(np.float32)
        diff_prob = np.where(valid2d[None, :, :], diff_prob, np.nan).astype(np.float32)

        fss_input_vals = []
        fss_reg_vals = []
        fss_diff_vals = []

        for c, s_km in zip(cells, scales_eff_km):
            obs_frac, obs_ok = _neighborhood_fraction(obs_event, valid2d, int(c))
            inp_frac, inp_ok = _neighborhood_fraction(inp_event, valid2d, int(c))
            reg_frac, reg_ok = _neighborhood_fraction(reg_event, valid2d, int(c))
            dif_frac, dif_ok = _neighborhood_fraction(diff_prob, valid2d, int(c))

            fss_i = _fss_from_fraction(obs_frac, inp_frac, obs_ok & inp_ok)
            fss_r = _fss_from_fraction(obs_frac, reg_frac, obs_ok & reg_ok)
            fss_d = _fss_from_fraction(obs_frac, dif_frac, obs_ok & dif_ok)

            fss_input_vals.append(fss_i)
            fss_reg_vals.append(fss_r)
            fss_diff_vals.append(fss_d)

            fss_rows.extend(
                [
                    {"threshold_mm": thr, "scale_km": float(s_km), "scale_cells": int(c), "method": "Input", "fss": fss_i},
                    {"threshold_mm": thr, "scale_km": float(s_km), "scale_cells": int(c), "method": "UNet", "fss": fss_r},
                    {
                        "threshold_mm": thr,
                        "scale_km": float(s_km),
                        "scale_cells": int(c),
                        "method": "CorrDiff ensemble prob",
                        "fss": fss_d,
                    },
                ]
            )

        fss_input[float(thr)] = fss_input_vals
        fss_reg[float(thr)] = fss_reg_vals
        fss_diff[float(thr)] = fss_diff_vals

        if args.fss_member_lines and "ensemble" in diff_sel_da.dims:
            n_members = int(diff_sel_da.sizes["ensemble"])
            print(
                f"Computing per-member FSS lines for threshold {thr:g} mm "
                f"({n_members} members, alpha={args.fss_member_line_alpha:.2f}) ..."
            )
            member_curves = np.full((n_members, len(cells)), np.nan, dtype=np.float64)

            for m_idx in range(n_members):
                member_field = diff_sel_da.isel(ensemble=m_idx).values.astype(np.float32)
                member_event = np.where(np.isfinite(member_field), member_field >= thr, np.nan).astype(np.float32)

                for s_idx, (c, s_km) in enumerate(zip(cells, scales_eff_km)):
                    obs_frac, obs_ok = _neighborhood_fraction(obs_event, valid2d, int(c))
                    mem_frac, mem_ok = _neighborhood_fraction(member_event, valid2d, int(c))
                    fss_m = _fss_from_fraction(obs_frac, mem_frac, obs_ok & mem_ok)
                    member_curves[m_idx, s_idx] = fss_m

                    fss_rows.append(
                        {
                            "threshold_mm": float(thr),
                            "scale_km": float(s_km),
                            "scale_cells": int(c),
                            "method": f"CorrDiff member {m_idx}",
                            "fss": fss_m,
                        }
                    )

            fss_diff_members[float(thr)] = member_curves

        y_field = np.where(np.isfinite(truth_sel), truth_sel >= thr, np.nan)
        diff_score_field = diff_prob

        y_flat = y_field[:, valid2d].ravel()
        d_flat = diff_score_field[:, valid2d].ravel()
        r_flat = reg_sel[:, valid2d].ravel()
        i_flat = input_sel[:, valid2d].ravel()

        keep = np.isfinite(y_flat) & np.isfinite(d_flat) & np.isfinite(r_flat) & np.isfinite(i_flat)
        y_true = y_flat[keep].astype(np.int8)
        d_score = d_flat[keep].astype(np.float32)
        r_score = r_flat[keep].astype(np.float32)
        i_score = i_flat[keep].astype(np.float32)

        if args.max_samples > 0 and y_true.size > args.max_samples:
            idx = rng.choice(y_true.size, size=args.max_samples, replace=False)
            y_true = y_true[idx]
            d_score = d_score[idx]
            r_score = r_score[idx]
            i_score = i_score[idx]

        if y_true.size < 2:
            raise RuntimeError(f"Not enough valid samples for threshold {thr:g} mm.")

        clim = float(np.mean(y_true))
        brier_ref = float(clim * (1.0 - clim))
        rel_curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        brier_scores: dict[str, float] = {}
        brier_skill: dict[str, float] = {}

        reg_prob = (r_score >= float(thr)).astype(np.float32)
        input_prob = (i_score >= float(thr)).astype(np.float32)

        prob_for_reliability = {
            "CorrDiff (ensemble prob)": d_score,
            "UNet": reg_prob,
            "Input": input_prob,
        }
        for name, prob in prob_for_reliability.items():
            bs = float(np.mean((prob - y_true) ** 2))
            bss = float(1.0 - bs / brier_ref) if brier_ref > 0 else float("nan")
            brier_scores[name] = bs
            brier_skill[name] = bss
            rel_curves[name] = _reliability_curve(prob, y_true, n_bins=args.reliability_bins)

        roc_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        roc_auc: dict[str, float] = {}
        pr_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        pr_auc: dict[str, float] = {}

        for name, score in (
            ("CorrDiff (ensemble prob)", d_score),
            ("UNet", r_score),
            ("Input", i_score),
        ):
            fpr, tpr, auc_roc = _roc_curve_auc(y_true, score)
            roc_curves[name] = (fpr, tpr)
            roc_auc[name] = auc_roc

            rec, prec, ap = _pr_curve_ap(y_true, score)
            pr_curves[name] = (rec, prec)
            pr_auc[name] = ap

            metrics_rows.append(
                {
                    "threshold_mm": float(thr),
                    "method": name,
                    "event_rate": clim,
                    "roc_auc": roc_auc[name],
                    "pr_auc": pr_auc[name],
                    "n_samples": int(y_true.size),
                    "brier_score": brier_scores[name],
                    "brier_ref": brier_ref,
                    "brier_skill_score": brier_skill[name],
                }
            )

        for name in ("CorrDiff (ensemble prob)", "UNet", "Input"):
            rel_x, rel_y, rel_n = rel_curves[name]
            for x, y, n in zip(rel_x, rel_y, rel_n):
                rel_rows.append(
                    {
                        "threshold_mm": float(thr),
                        "method": name,
                        "prob_bin_center": float(x),
                        "obs_frequency": float(y) if np.isfinite(y) else np.nan,
                        "bin_count": int(n),
                    }
                )

        thr_tag = str(thr).replace(".", "p")
        prob_fig_base = args.out_dir / f"prob_skill_threshold_{thr_tag}mm"
        _plot_prob_skill_panels(
            out_path_base=prob_fig_base,
            threshold_mm=float(thr),
            rel_curves=rel_curves,
            brier_scores=brier_scores,
            brier_ref=brier_ref,
            brier_skill=brier_skill,
            roc_curves=roc_curves,
            roc_auc=roc_auc,
            pr_curves=pr_curves,
            pr_auc=pr_auc,
            climatology=clim,
            n_samples=int(y_true.size),
            dpi=args.dpi,
        )

    fss_base = args.out_dir / "fss_vs_scale"
    _plot_fss_grid(
        out_path_base=fss_base,
        thresholds=[float(v) for v in args.thresholds],
        scales_km=scales_eff_km,
        fss_input=fss_input,
        fss_reg=fss_reg,
        fss_diff=fss_diff,
        fss_diff_members=None,
        member_line_alpha=args.fss_member_line_alpha,
        dpi=args.dpi,
    )

    if args.fss_member_lines and len(fss_diff_members) > 0:
        fss_members_base = args.out_dir / "fss_vs_scale_with_members"
        _plot_fss_grid(
            out_path_base=fss_members_base,
            thresholds=[float(v) for v in args.thresholds],
            scales_km=scales_eff_km,
            fss_input=fss_input,
            fss_reg=fss_reg,
            fss_diff=fss_diff,
            fss_diff_members=fss_diff_members,
            member_line_alpha=args.fss_member_line_alpha,
            dpi=args.dpi,
        )

    fss_df = pd.DataFrame(fss_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    rel_df = pd.DataFrame(rel_rows)

    fss_csv = args.out_dir / "fss_vs_scale_values.csv"
    metrics_csv = args.out_dir / "prob_skill_summary.csv"
    rel_csv = args.out_dir / "reliability_curve_points.csv"

    fss_df.to_csv(fss_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    rel_df.to_csv(rel_csv, index=False)

    print(f"Saved: {fss_csv}")
    print(f"Saved: {metrics_csv}")
    print(f"Saved: {rel_csv}")


if __name__ == "__main__":
    main()
