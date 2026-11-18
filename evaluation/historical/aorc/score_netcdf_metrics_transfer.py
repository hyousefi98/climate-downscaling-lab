#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage


CSI_THRESHOLDS = [0.1, 1.0, 5.0, 10.0]
FSS_WINDOWS = [3, 5, 9]


def rmse(obs, pred):
    obs_flat = obs.values.ravel()
    pred_flat = pred.values.ravel()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    if not np.any(mask):
        return np.nan
    return np.sqrt(np.mean((obs_flat[mask] - pred_flat[mask]) ** 2))


def mae(obs, pred):
    obs_flat = obs.values.ravel()
    pred_flat = pred.values.ravel()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs(obs_flat[mask] - pred_flat[mask]))


def pbias(obs, pred):
    obs_flat = obs.values.ravel()
    pred_flat = pred.values.ravel()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask]
    if obs_valid.size == 0 or np.sum(np.abs(obs_valid)) == 0:
        return np.nan
    return 100.0 * np.sum(pred_valid - obs_valid) / np.sum(np.abs(obs_valid))


def pearson_r(obs, pred):
    obs_flat = obs.values.ravel()
    pred_flat = pred.values.ravel()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask]
    if obs_valid.size < 2:
        return np.nan
    if np.std(obs_valid) == 0 or np.std(pred_valid) == 0:
        return np.nan
    return np.corrcoef(obs_valid, pred_valid)[0, 1]


def pearson_r_by_time(truth_da, pred_da):
    rows = []
    if "time" not in truth_da.dims:
        rows.append({"time_index": 0, "time": np.nan, "Pearson_r": pearson_r(truth_da, pred_da)})
        return rows

    n_times = truth_da.sizes.get("time", 1)
    time_vals = truth_da["time"].values if "time" in truth_da.coords else np.arange(n_times)

    for t in range(n_times):
        rows.append(
            {
                "time_index": int(t),
                "time": time_vals[t],
                "Pearson_r": pearson_r(truth_da.isel(time=t), pred_da.isel(time=t)),
            }
        )
    return rows


def csi(obs, pred, threshold):
    obs_flat = obs.values.ravel()
    pred_flat = pred.values.ravel()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask]

    if obs_valid.size == 0:
        return np.nan

    obs_binary = obs_valid >= threshold
    pred_binary = pred_valid >= threshold

    hits = np.sum(obs_binary & pred_binary)
    misses = np.sum(obs_binary & ~pred_binary)
    false_alarms = np.sum(~obs_binary & pred_binary)

    denom = hits + misses + false_alarms
    if denom == 0:
        return np.nan
    return hits / denom


def fss(obs, pred, threshold, window_size):
    obs_arr = obs.values if hasattr(obs, "values") else obs
    pred_arr = pred.values if hasattr(pred, "values") else pred

    if obs_arr.ndim != 2 or pred_arr.ndim != 2:
        return np.nan

    valid = np.isfinite(obs_arr) & np.isfinite(pred_arr)
    if not np.any(valid):
        return np.nan

    obs_arr = np.where(valid, obs_arr, 0.0)
    pred_arr = np.where(valid, pred_arr, 0.0)

    obs_binary = (obs_arr >= threshold).astype(float)
    pred_binary = (pred_arr >= threshold).astype(float)

    kernel = np.ones((window_size, window_size), dtype=np.float64)
    obs_frac = ndimage.convolve(obs_binary, kernel, mode="constant", cval=0.0)
    pred_frac = ndimage.convolve(pred_binary, kernel, mode="constant", cval=0.0)
    valid_count = ndimage.convolve(valid.astype(float), kernel, mode="constant", cval=0.0)

    obs_frac = np.divide(obs_frac, valid_count, out=np.full(obs_frac.shape, np.nan), where=valid_count > 0)
    pred_frac = np.divide(pred_frac, valid_count, out=np.full(pred_frac.shape, np.nan), where=valid_count > 0)

    valid_frac = np.isfinite(obs_frac) & np.isfinite(pred_frac)
    if not np.any(valid_frac):
        return np.nan

    mse_frac = np.mean((obs_frac[valid_frac] - pred_frac[valid_frac]) ** 2)
    mse_ref = np.mean(obs_frac[valid_frac] ** 2) + np.mean(pred_frac[valid_frac] ** 2)
    if mse_ref == 0:
        return np.nan
    return 1.0 - mse_frac / mse_ref


def compute_fss_over_time(truth_da, pred_da, threshold, window_size):
    n_times = truth_da.sizes.get("time", 1)
    vals = []
    for t in range(n_times):
        obs_2d = truth_da.isel(time=t) if "time" in truth_da.dims else truth_da
        pred_2d = pred_da.isel(time=t) if "time" in pred_da.dims else pred_da
        score = fss(obs_2d, pred_2d, threshold, window_size)
        if np.isfinite(score):
            vals.append(score)
    if not vals:
        return np.nan
    return float(np.mean(vals))


def apply_precip_threshold(da, threshold):
    if threshold is None:
        return da
    return da.where((da >= threshold) | da.isnull(), 0.0)


def collapse_ensemble(pred_var, reducer, quantile):
    if "ensemble" not in pred_var.dims:
        return pred_var

    if reducer == "mean":
        return pred_var.mean("ensemble")
    if reducer == "median":
        return pred_var.median("ensemble")
    if reducer == "max":
        return pred_var.max("ensemble")
    if reducer == "quantile":
        out = pred_var.quantile(quantile, dim="ensemble")
        if "quantile" in out.dims:
            out = out.squeeze("quantile", drop=True)
        return out

    raise ValueError(f"Unsupported ensemble reducer: {reducer}")


def looks_like_precip(name):
    n = name.upper()
    return ("APCP" in n) or ("RAIN" in n) or ("PREC" in n)


def score_file(
    netcdf_file,
    output_basic,
    output_precip,
    output_pearson_time,
    ensemble_reducer,
    ensemble_quantile,
    precip_zero_threshold,
):
    print(f"Loading: {netcdf_file}")
    root_ds = xr.open_dataset(netcdf_file)
    pred_ds = xr.open_dataset(netcdf_file, group="prediction")
    truth_ds = xr.open_dataset(netcdf_file, group="truth")

    for ds in (pred_ds, truth_ds):
        ds.coords.update(root_ds.coords)

    basic_rows = []
    precip_rows = []
    pearson_time_rows = []

    shared_vars = [v for v in truth_ds.data_vars if v in pred_ds.data_vars]
    if not shared_vars:
        raise RuntimeError("No common variables between truth and prediction groups.")

    for var in shared_vars:
        truth_var = truth_ds[var]
        pred_var = collapse_ensemble(pred_ds[var], ensemble_reducer, ensemble_quantile)

        is_precip = looks_like_precip(var)
        if is_precip:
            truth_eval = apply_precip_threshold(truth_var, precip_zero_threshold)
            pred_eval = apply_precip_threshold(pred_var, precip_zero_threshold)
        else:
            truth_eval = truth_var
            pred_eval = pred_var

        basic_rows.append(
            {
                "variable": var,
                "RMSE": rmse(truth_eval, pred_eval),
                "MAE": mae(truth_eval, pred_eval),
                "PBIAS_%": pbias(truth_eval, pred_eval),
                "Pearson_r": pearson_r(truth_eval, pred_eval),
            }
        )

        time_rows = pearson_r_by_time(truth_eval, pred_eval)
        for row in time_rows:
            row["variable"] = var
        pearson_time_rows.extend(time_rows)

        if is_precip:
            row = {"variable": var}
            for th in CSI_THRESHOLDS:
                row[f"CSI_{th}"] = csi(truth_eval, pred_eval, th)
            for w in FSS_WINDOWS:
                row[f"FSS_w{w}"] = compute_fss_over_time(truth_eval, pred_eval, threshold=0.1, window_size=w)
            precip_rows.append(row)

    basic_df = pd.DataFrame(basic_rows).sort_values("variable").reset_index(drop=True).round(4)
    precip_df = pd.DataFrame(precip_rows).sort_values("variable").reset_index(drop=True).round(4)
    pearson_df = pd.DataFrame(pearson_time_rows).round(4)

    basic_df.to_csv(output_basic, index=False)
    precip_df.to_csv(output_precip, index=False)
    pearson_df.to_csv(output_pearson_time, index=False)

    print(f"Saved basic: {output_basic}")
    print(f"Saved precip: {output_precip}")
    print(f"Saved pearson-by-time: {output_pearson_time}")


def parse_args():
    parser = argparse.ArgumentParser(description="Score CorrDiff NetCDF outputs.")
    parser.add_argument("--netcdf", type=Path, required=True)
    parser.add_argument("--output-basic", type=Path, required=True)
    parser.add_argument("--output-precip", type=Path, required=True)
    parser.add_argument("--output-pearson-time", type=Path, required=True)
    parser.add_argument("--ensemble-reducer", type=str, default="mean", choices=["mean", "median", "max", "quantile"])
    parser.add_argument("--ensemble-quantile", type=float, default=0.9)
    parser.add_argument("--precip-zero-threshold", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score_file(
        netcdf_file=args.netcdf,
        output_basic=args.output_basic,
        output_precip=args.output_precip,
        output_pearson_time=args.output_pearson_time,
        ensemble_reducer=args.ensemble_reducer,
        ensemble_quantile=args.ensemble_quantile,
        precip_zero_threshold=args.precip_zero_threshold,
    )
