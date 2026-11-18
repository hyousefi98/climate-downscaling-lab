#%% ───────────────────────────── IMPORTS ─────────────────────────────
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from scipy.spatial import cKDTree

#%% ─────────────────────────── CONFIGURATION ─────────────────────────
# Dataset paths
REG_NETCDF_FILE = Path("reg_wrf_12km_to_3km_200sam.nc")
REGDIFF_NETCDF_FILE = Path("all_wrf_12km_to_3km_32ens_200sam.nc")

# Sources to score (prediction group vs truth group)
SOURCE_SPECS = [
    {
        "source": "Input",
        "netcdf_file": REG_NETCDF_FILE,
        "pred_group": "input",
        "truth_group": "truth",
        "n_ensemble": -1,
    },
    {
        "source": "Reg",
        "netcdf_file": REG_NETCDF_FILE,
        "pred_group": "prediction",
        "truth_group": "truth",
        "n_ensemble": -1,
    },
    {
        "source": "Reg+Diff",
        "netcdf_file": REGDIFF_NETCDF_FILE,
        "pred_group": "prediction",
        "truth_group": "truth",
        "n_ensemble": -1,
    },
]

# Variable detection / mapping
PRECIP_VAR_ALIASES = ("RAIN_TOTAL", "PREC_ACC_NC", "APCP_surface")
VAR_NAME_ALIASES = {
    "PREC_ACC_NC": ["RAIN_TOTAL", "APCP_surface"],
    "RAIN_TOTAL": ["PREC_ACC_NC", "APCP_surface"],
    "APCP_surface": ["RAIN_TOTAL", "PREC_ACC_NC"],
}

# Precipitation preprocessing
PRECIP_ZERO_THRESHOLD = 0.1

# Domain mask
LANDMASK_PATH = Path("data/grids/aorc_land_mask_3km.nc")
LANDMASK_VAR = "LANDMASK"
LANDMASK_MODE = "land"  # one of: land, ocean, none

# Deterministic field selection for spatial metrics
# - member_mean: score each member separately, then average scores over members
# - ensemble_reducer: reduce ensemble to one deterministic field first
DETERMINISTIC_METRICS_MODE = "member_mean"  # one of: member_mean, ensemble_reducer
ENSEMBLE_REDUCER = "mean"  # one of: mean, median, max, quantile
ENSEMBLE_QUANTILE = 0.9  # used only when ENSEMBLE_REDUCER == "quantile"

# Ensemble sharpness interval
ENS_Q_LOW = 0.10
ENS_Q_HIGH = 0.90
ENS_PI_COL = f"EnsPIWidth_q{int(round(100 * ENS_Q_LOW))}_q{int(round(100 * ENS_Q_HIGH))}"

# Debugging control
MAX_TIMESTEPS = -1  # -1 = all time steps

# Output
OUTPUT_CSV = Path("precip_threeway_selected_metrics.csv")

#%% ──────────────────────────── UTILITIES ────────────────────────────
def _normalize_var_name(name):
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _looks_like_precip_name(var_name):
    upper_name = str(var_name).upper()
    return ("PREC" in upper_name) or ("RAIN" in upper_name) or ("APCP" in upper_name)


def _build_normalized_lookup(ds):
    lookup = {}
    for var in ds.data_vars:
        key = _normalize_var_name(var)
        if key not in lookup:
            lookup[key] = var
    return lookup


def _resolve_predictor_var_name(truth_var_name, pred_ds, norm_lookup):
    if truth_var_name in pred_ds.data_vars:
        return truth_var_name

    for candidate in VAR_NAME_ALIASES.get(truth_var_name, []):
        if candidate in pred_ds.data_vars:
            return candidate

    for canonical, aliases in VAR_NAME_ALIASES.items():
        if truth_var_name in aliases and canonical in pred_ds.data_vars:
            return canonical

    norm_truth = _normalize_var_name(truth_var_name)
    if norm_truth in norm_lookup:
        return norm_lookup[norm_truth]

    for candidate in VAR_NAME_ALIASES.get(truth_var_name, []):
        norm_candidate = _normalize_var_name(candidate)
        if norm_candidate in norm_lookup:
            return norm_lookup[norm_candidate]

    if _looks_like_precip_name(truth_var_name):
        for candidate in PRECIP_VAR_ALIASES:
            if candidate in pred_ds.data_vars:
                return candidate
            norm_candidate = _normalize_var_name(candidate)
            if norm_candidate in norm_lookup:
                return norm_lookup[norm_candidate]

        for pred_var in pred_ds.data_vars:
            if _looks_like_precip_name(pred_var):
                return pred_var

    return None


def apply_precip_threshold(da, threshold):
    if threshold is None:
        return da
    return da.where((da >= threshold) | da.isnull(), 0.0)


def _make_domain_mask(root_ds):
    mode = str(LANDMASK_MODE).lower()
    if mode == "none":
        return None
    if mode not in {"land", "ocean"}:
        raise ValueError("LANDMASK_MODE must be one of: land, ocean, none")

    if "lat" not in root_ds.variables or "lon" not in root_ds.variables:
        raise ValueError("Dataset must contain root variables 'lat' and 'lon' for mask remapping.")

    model_lat = np.asarray(root_ds["lat"].values, dtype=np.float64)
    model_lon = np.asarray(root_ds["lon"].values, dtype=np.float64)

    if not LANDMASK_PATH.exists():
        print(f"Warning: LANDMASK_PATH not found ({LANDMASK_PATH}). Disabling mask.")
        return None

    mask_ds = xr.open_dataset(LANDMASK_PATH)
    try:
        if LANDMASK_VAR not in mask_ds:
            raise KeyError(
                f"Mask variable '{LANDMASK_VAR}' not found. Available: {list(mask_ds.data_vars)}"
            )
        if "LAT2D" not in mask_ds or "LON2D" not in mask_ds:
            raise KeyError("Mask dataset must include 'LAT2D' and 'LON2D'.")

        src_lat = np.asarray(mask_ds["LAT2D"].values, dtype=np.float64)
        src_lon = np.asarray(mask_ds["LON2D"].values, dtype=np.float64)
        src_mask = np.asarray(mask_ds[LANDMASK_VAR].values, dtype=np.float64)
    finally:
        mask_ds.close()

    if (
        src_mask.shape == model_lat.shape
        and src_lat.shape == model_lat.shape
        and src_lon.shape == model_lon.shape
        and np.nanmax(np.abs(src_lat - model_lat)) < 1e-6
        and np.nanmax(np.abs(src_lon - model_lon)) < 1e-6
    ):
        mapped_mask = src_mask
    else:
        src_points = np.column_stack((src_lat.ravel(), src_lon.ravel()))
        tgt_points = np.column_stack((model_lat.ravel(), model_lon.ravel()))
        tree = cKDTree(src_points)
        _, nn_idx = tree.query(tgt_points, k=1)
        mapped_mask = src_mask.ravel()[nn_idx].reshape(model_lat.shape)

    keep = mapped_mask >= 0.5 if mode == "land" else mapped_mask < 0.5
    mask_dims = root_ds["lat"].dims
    mask_coords = {d: root_ds.coords[d] for d in mask_dims if d in root_ds.coords}
    return xr.DataArray(keep, dims=mask_dims, coords=mask_coords, name=f"{mode}_mask")


def apply_domain_mask(da, spatial_mask):
    if spatial_mask is None:
        return da
    return da.where(spatial_mask)


def reduce_ensemble(pred_da):
    if "ensemble" not in pred_da.dims:
        return pred_da

    if ENSEMBLE_REDUCER == "mean":
        return pred_da.mean("ensemble")
    if ENSEMBLE_REDUCER == "median":
        return pred_da.median("ensemble")
    if ENSEMBLE_REDUCER == "max":
        return pred_da.max("ensemble")
    if ENSEMBLE_REDUCER == "quantile":
        out = pred_da.quantile(ENSEMBLE_QUANTILE, dim="ensemble")
        if "quantile" in out.dims:
            out = out.squeeze("quantile", drop=True)
        return out

    raise ValueError(
        f"Unsupported ENSEMBLE_REDUCER='{ENSEMBLE_REDUCER}'. "
        "Use one of: mean, median, max, quantile"
    )


def _limit_n_times(truth_da):
    n_times = truth_da.sizes.get("time", 1)
    if MAX_TIMESTEPS is not None and MAX_TIMESTEPS > 0:
        n_times = min(n_times, MAX_TIMESTEPS)
    return n_times


def _as_ensemble_values(pred_arr):
    if pred_arr.ndim == 2:
        return pred_arr[np.newaxis, ...]
    if pred_arr.ndim == 3:
        return pred_arr
    raise ValueError(f"Expected 2D or 3D prediction array, got shape={pred_arr.shape}")


#%% ───────────────────────── METRIC FUNCTIONS ─────────────────────────
def spatial_pattern_corr_2d(obs_arr, pred_arr):
    if obs_arr.ndim != 2 or pred_arr.ndim != 2:
        return np.nan

    obs_flat = obs_arr.ravel()
    pred_flat = pred_arr.ravel()
    valid = np.isfinite(obs_flat) & np.isfinite(pred_flat)

    if np.count_nonzero(valid) < 2:
        return np.nan

    obs_valid = obs_flat[valid]
    pred_valid = pred_flat[valid]
    obs_valid = obs_valid - np.mean(obs_valid)
    pred_valid = pred_valid - np.mean(pred_valid)

    std_obs = np.std(obs_valid)
    std_pred = np.std(pred_valid)
    if std_obs == 0 or std_pred == 0:
        return np.nan

    return float(np.corrcoef(obs_valid, pred_valid)[0, 1])


def gradient_magnitude(arr):
    gy, gx = np.gradient(arr)
    return np.sqrt(gx**2 + gy**2)


def sharpness_ratio_2d(obs_arr, pred_arr):
    if obs_arr.ndim != 2 or pred_arr.ndim != 2:
        return np.nan

    valid = np.isfinite(obs_arr) & np.isfinite(pred_arr)
    if np.count_nonzero(valid) < 4:
        return np.nan

    obs_fill = np.where(valid, obs_arr, 0.0)
    pred_fill = np.where(valid, pred_arr, 0.0)

    grad_obs = gradient_magnitude(obs_fill)
    grad_pred = gradient_magnitude(pred_fill)

    valid_core = ndimage.binary_erosion(valid, structure=np.ones((3, 3), dtype=bool), border_value=0)
    if not np.any(valid_core):
        valid_core = valid

    mean_grad_obs = float(np.mean(grad_obs[valid_core]))
    mean_grad_pred = float(np.mean(grad_pred[valid_core]))
    if mean_grad_obs == 0:
        return np.nan

    return mean_grad_pred / mean_grad_obs


def laplacian_variance_ratio_2d(obs_arr, pred_arr):
    if obs_arr.ndim != 2 or pred_arr.ndim != 2:
        return np.nan

    valid = np.isfinite(obs_arr) & np.isfinite(pred_arr)
    if np.count_nonzero(valid) < 9:
        return np.nan

    obs_fill = np.where(valid, obs_arr, 0.0)
    pred_fill = np.where(valid, pred_arr, 0.0)

    lap_obs = ndimage.laplace(obs_fill, mode="nearest")
    lap_pred = ndimage.laplace(pred_fill, mode="nearest")

    valid_core = ndimage.binary_erosion(valid, structure=np.ones((3, 3), dtype=bool), border_value=0)
    if np.count_nonzero(valid_core) < 9:
        valid_core = valid

    var_obs = float(np.var(lap_obs[valid_core]))
    var_pred = float(np.var(lap_pred[valid_core]))
    if var_obs == 0:
        return np.nan

    return var_pred / var_obs


def haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius_km * c


def peak_location_error_2d(obs_arr, pred_arr, lat2d=None, lon2d=None):
    if obs_arr.ndim != 2 or pred_arr.ndim != 2:
        return np.nan, np.nan

    valid = np.isfinite(obs_arr) & np.isfinite(pred_arr)
    if not np.any(valid):
        return np.nan, np.nan

    obs_work = np.where(valid, obs_arr, -np.inf)
    pred_work = np.where(valid, pred_arr, -np.inf)

    max_obs = float(np.max(obs_work))
    max_pred = float(np.max(pred_work))

    # Both fields dry (after precipitation thresholding): treat as zero location error.
    if max_obs <= 0.0 and max_pred <= 0.0:
        return 0.0, 0.0

    # If one field has no event and the other does, no meaningful peak location match.
    if max_obs <= 0.0 or max_pred <= 0.0:
        return np.nan, np.nan

    obs_idx_flat = int(np.argmax(obs_work))
    pred_idx_flat = int(np.argmax(pred_work))

    obs_iy, obs_ix = np.unravel_index(obs_idx_flat, obs_work.shape)
    pred_iy, pred_ix = np.unravel_index(pred_idx_flat, pred_work.shape)

    dist_px = float(np.hypot(pred_ix - obs_ix, pred_iy - obs_iy))

    if lat2d is None or lon2d is None:
        return dist_px, np.nan

    lat_obs = lat2d[obs_iy, obs_ix]
    lon_obs = lon2d[obs_iy, obs_ix]
    lat_pred = lat2d[pred_iy, pred_ix]
    lon_pred = lon2d[pred_iy, pred_ix]

    if not (np.isfinite(lat_obs) and np.isfinite(lon_obs) and np.isfinite(lat_pred) and np.isfinite(lon_pred)):
        return dist_px, np.nan

    dist_km = float(haversine_km(lat_obs, lon_obs, lat_pred, lon_pred))
    return dist_px, dist_km


def crps_ensemble_mean_2d(obs_arr, pred_arr):
    pred_ens = _as_ensemble_values(pred_arr)
    n_members = pred_ens.shape[0]

    valid = np.isfinite(obs_arr) & np.all(np.isfinite(pred_ens), axis=0)
    if not np.any(valid):
        return np.nan

    obs_valid = obs_arr[valid].astype(np.float64)
    ens_valid = pred_ens[:, valid].astype(np.float64)

    term1 = np.mean(np.abs(ens_valid - obs_valid[np.newaxis, :]), axis=0)

    if n_members == 1:
        crps = term1
    else:
        ens_sorted = np.sort(ens_valid, axis=0)
        ranks = np.arange(1, n_members + 1, dtype=np.float64)[:, np.newaxis]
        coeff = 2.0 * ranks - n_members - 1.0
        term2 = np.sum(coeff * ens_sorted, axis=0) / (n_members**2)
        crps = term1 - term2

    return float(np.mean(crps))


def ensemble_spread_mean_2d(pred_arr, obs_arr):
    pred_ens = _as_ensemble_values(pred_arr)
    valid = np.isfinite(obs_arr) & np.all(np.isfinite(pred_ens), axis=0)
    if not np.any(valid):
        return np.nan

    spread = np.std(pred_ens[:, valid], axis=0)
    return float(np.mean(spread))


def ensemble_interval_width_mean_2d(pred_arr, obs_arr, q_low, q_high):
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("q_low and q_high must satisfy 0 <= q_low < q_high <= 1")

    pred_ens = _as_ensemble_values(pred_arr)
    valid = np.isfinite(obs_arr) & np.all(np.isfinite(pred_ens), axis=0)
    if not np.any(valid):
        return np.nan

    vals = pred_ens[:, valid]
    ql = np.quantile(vals, q_low, axis=0)
    qh = np.quantile(vals, q_high, axis=0)
    width = qh - ql
    return float(np.mean(width))


#%% ───────────────────────── SCORING HELPERS ─────────────────────────
def score_deterministic_field_over_time(truth_da, pred_da, lat2d, lon2d):
    n_times = _limit_n_times(truth_da)

    pattern_vals = []
    sharp_vals = []
    lap_vals = []
    peak_px_vals = []
    peak_km_vals = []

    for t in range(n_times):
        obs_2d = truth_da.isel(time=t) if "time" in truth_da.dims else truth_da
        pred_2d = pred_da.isel(time=t) if "time" in pred_da.dims else pred_da

        obs_arr = np.asarray(obs_2d.values, dtype=np.float64)
        pred_arr = np.asarray(pred_2d.values, dtype=np.float64)

        pattern = spatial_pattern_corr_2d(obs_arr, pred_arr)
        sharp = sharpness_ratio_2d(obs_arr, pred_arr)
        lap = laplacian_variance_ratio_2d(obs_arr, pred_arr)
        peak_px, peak_km = peak_location_error_2d(obs_arr, pred_arr, lat2d=lat2d, lon2d=lon2d)

        if np.isfinite(pattern):
            pattern_vals.append(pattern)
        if np.isfinite(sharp):
            sharp_vals.append(sharp)
        if np.isfinite(lap):
            lap_vals.append(lap)
        if np.isfinite(peak_px):
            peak_px_vals.append(peak_px)
        if np.isfinite(peak_km):
            peak_km_vals.append(peak_km)

    return {
        "SpatialPatternCorr": float(np.mean(pattern_vals)) if pattern_vals else np.nan,
        "SharpnessRatio": float(np.mean(sharp_vals)) if sharp_vals else np.nan,
        "LaplacianVarRatio": float(np.mean(lap_vals)) if lap_vals else np.nan,
        "PeakLocationError_px": float(np.mean(peak_px_vals)) if peak_px_vals else np.nan,
        "PeakLocationError_km": float(np.mean(peak_km_vals)) if peak_km_vals else np.nan,
    }


def score_probabilistic_over_time(truth_da, pred_da):
    n_times = _limit_n_times(truth_da)

    crps_vals = []
    spread_vals = []
    pi_vals = []

    for t in range(n_times):
        obs_2d = truth_da.isel(time=t) if "time" in truth_da.dims else truth_da
        pred_2d = pred_da.isel(time=t) if "time" in pred_da.dims else pred_da

        obs_arr = np.asarray(obs_2d.values, dtype=np.float64)
        pred_arr = np.asarray(pred_2d.values, dtype=np.float64)

        crps = crps_ensemble_mean_2d(obs_arr, pred_arr)
        spread = ensemble_spread_mean_2d(pred_arr, obs_arr)
        pi_width = ensemble_interval_width_mean_2d(pred_arr, obs_arr, ENS_Q_LOW, ENS_Q_HIGH)

        if np.isfinite(crps):
            crps_vals.append(crps)
        if np.isfinite(spread):
            spread_vals.append(spread)
        if np.isfinite(pi_width):
            pi_vals.append(pi_width)

    return {
        "CRPS": float(np.mean(crps_vals)) if crps_vals else np.nan,
        "EnsSpreadMean": float(np.mean(spread_vals)) if spread_vals else np.nan,
        ENS_PI_COL: float(np.mean(pi_vals)) if pi_vals else np.nan,
    }


def summarize_member_metrics(member_metric_rows):
    keys = [
        "SpatialPatternCorr",
        "SharpnessRatio",
        "LaplacianVarRatio",
        "PeakLocationError_px",
        "PeakLocationError_km",
    ]

    out = {}
    for key in keys:
        vals = np.array([row.get(key, np.nan) for row in member_metric_rows], dtype=float)
        out[key] = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan
        out[f"{key}_member_std"] = float(np.nanstd(vals)) if np.any(np.isfinite(vals)) else np.nan
    return out


def score_precip_pair(truth_da, pred_da, lat2d, lon2d):
    prob_summary = score_probabilistic_over_time(truth_da, pred_da)

    if "ensemble" in pred_da.dims and DETERMINISTIC_METRICS_MODE == "member_mean":
        n_members = pred_da.sizes["ensemble"]
        member_rows = []
        for m in range(n_members):
            pred_member = pred_da.isel(ensemble=m)
            member_rows.append(score_deterministic_field_over_time(truth_da, pred_member, lat2d, lon2d))
        det_summary = summarize_member_metrics(member_rows)
        det_mode_used = "member_mean"
        n_members_scored = n_members
    else:
        pred_det = reduce_ensemble(pred_da)
        det_row = score_deterministic_field_over_time(truth_da, pred_det, lat2d, lon2d)
        det_summary = {**det_row}
        det_summary.update(
            {
                "SpatialPatternCorr_member_std": 0.0,
                "SharpnessRatio_member_std": 0.0,
                "LaplacianVarRatio_member_std": 0.0,
                "PeakLocationError_px_member_std": 0.0,
                "PeakLocationError_km_member_std": 0.0,
            }
        )
        det_mode_used = "ensemble_reducer"
        n_members_scored = 1

    out = {**prob_summary, **det_summary}
    out["deterministic_mode_used"] = det_mode_used
    out["n_members_for_det_metrics"] = n_members_scored
    out["n_times_scored"] = _limit_n_times(truth_da)
    return out


def _prepare_precip_pairs(truth_ds, pred_ds):
    norm_lookup = _build_normalized_lookup(pred_ds)
    pairs = []

    for truth_var in truth_ds.data_vars:
        if not _looks_like_precip_name(truth_var):
            continue
        pred_var = _resolve_predictor_var_name(truth_var, pred_ds, norm_lookup)
        if pred_var is None:
            continue
        if not _looks_like_precip_name(pred_var):
            continue
        pairs.append((truth_var, pred_var))

    # Deduplicate while preserving order
    seen = set()
    unique_pairs = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)

    return unique_pairs


def _open_source_datasets(netcdf_file, pred_group, truth_group):
    root_ds = xr.open_dataset(netcdf_file)
    pred_ds = xr.open_dataset(netcdf_file, group=pred_group)
    truth_ds = xr.open_dataset(netcdf_file, group=truth_group)
    for ds in (pred_ds, truth_ds):
        ds.coords.update(root_ds.coords)
    return root_ds, pred_ds, truth_ds


def score_source(spec):
    source_name = spec["source"]
    netcdf_file = Path(spec["netcdf_file"])
    pred_group = spec["pred_group"]
    truth_group = spec["truth_group"]
    n_ensemble = int(spec.get("n_ensemble", -1))

    print("\n" + "-" * 80)
    print(f"Scoring source: {source_name}")
    print(f"File: {netcdf_file}")
    print(f"Groups: pred='{pred_group}' vs truth='{truth_group}'")
    print("-" * 80)

    root_ds, pred_ds, truth_ds = _open_source_datasets(netcdf_file, pred_group, truth_group)

    try:
        if n_ensemble > 0 and "ensemble" in pred_ds.dims:
            pred_ds = pred_ds.isel(ensemble=slice(0, n_ensemble))

        spatial_mask = _make_domain_mask(root_ds)
        if spatial_mask is None:
            print("Domain mask: disabled")
        else:
            selected = int(spatial_mask.sum().item())
            total = int(spatial_mask.size)
            print(
                f"Domain mask: mode='{LANDMASK_MODE}', selected {selected}/{total} "
                f"({100.0 * selected / total:.2f}%) grid points"
            )

        precip_pairs = _prepare_precip_pairs(truth_ds, pred_ds)
        if not precip_pairs:
            print("No precipitation variable pairs found. Skipping source.")
            return []

        lat2d = np.asarray(root_ds["lat"].values, dtype=np.float64) if "lat" in root_ds else None
        lon2d = np.asarray(root_ds["lon"].values, dtype=np.float64) if "lon" in root_ds else None

        rows = []
        for truth_var_name, pred_var_name in precip_pairs:
            print(f"\nProcessing precipitation pair: truth='{truth_var_name}' pred='{pred_var_name}'")
            truth_var = truth_ds[truth_var_name]
            pred_var = pred_ds[pred_var_name]

            truth_eval = apply_precip_threshold(truth_var, PRECIP_ZERO_THRESHOLD)
            pred_eval = apply_precip_threshold(pred_var, PRECIP_ZERO_THRESHOLD)

            truth_eval = apply_domain_mask(truth_eval, spatial_mask)
            pred_eval = apply_domain_mask(pred_eval, spatial_mask)

            metric_row = score_precip_pair(truth_eval, pred_eval, lat2d=lat2d, lon2d=lon2d)

            row = {
                "source": source_name,
                "file": netcdf_file.name,
                "truth_variable": truth_var_name,
                "pred_variable": pred_var_name,
                "n_ensemble_available": int(pred_eval.sizes.get("ensemble", 1)),
                **metric_row,
            }
            rows.append(row)

            print(f"  CRPS: {row['CRPS']:.6f}")
            print(f"  SpatialPatternCorr: {row['SpatialPatternCorr']:.6f}")
            print(f"  SharpnessRatio: {row['SharpnessRatio']:.6f}")
            print(f"  LaplacianVarRatio: {row['LaplacianVarRatio']:.6f}")
            print(f"  {ENS_PI_COL}: {row[ENS_PI_COL]:.6f}")
            print(f"  EnsSpreadMean: {row['EnsSpreadMean']:.6f}")
            print(f"  PeakLocationError_km: {row['PeakLocationError_km']:.6f}")

        return rows
    finally:
        pred_ds.close()
        truth_ds.close()
        root_ds.close()


#%% ───────────────────────────── MAIN ────────────────────────────────
all_rows = []
for spec in SOURCE_SPECS:
    all_rows.extend(score_source(spec))

if not all_rows:
    print("\nNo rows were scored. Nothing to save.")
else:
    output_cols = [
        "source",
        "file",
        "truth_variable",
        "pred_variable",
        "CRPS",
        "SpatialPatternCorr",
        "SpatialPatternCorr_member_std",
        "SharpnessRatio",
        "SharpnessRatio_member_std",
        "LaplacianVarRatio",
        "LaplacianVarRatio_member_std",
        ENS_PI_COL,
        "EnsSpreadMean",
        "PeakLocationError_px",
        "PeakLocationError_px_member_std",
        "PeakLocationError_km",
        "PeakLocationError_km_member_std",
        "n_ensemble_available",
        "n_members_for_det_metrics",
        "deterministic_mode_used",
        "n_times_scored",
    ]

    df = pd.DataFrame(all_rows)
    for col in output_cols:
        if col not in df.columns:
            df[col] = np.nan
    df = df[output_cols]
    df = df.sort_values(["source", "truth_variable", "pred_variable"]).reset_index(drop=True)

    num_cols = [c for c in output_cols if c not in {"source", "file", "truth_variable", "pred_variable", "deterministic_mode_used"}]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.round(6)

    print("\n" + "=" * 80)
    print("PRECIPITATION THREE-WAY METRICS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80 + "\n")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV.resolve()}")

# %%
