#%% ───────────────────────────── IMPORTS ─────────────────────────────
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import ndimage
from scipy.spatial import cKDTree

#%% ─────────────────────────── CONFIGURATION ─────────────────────────
# NETCDF_FILE  = Path("reg_wrf_random200_seed40.nc")  # your NetCDF with groups
NETCDF_FILE  = Path("all_wrf_12km_to_3km_32ens_200sam.nc")
# NETCDF_FILE  = Path("reg_wrf_12km_to_3km_200sam.nc")

OUTPUT_TABLE_BASIC = Path("forecast_scores_basic.csv")   # Basic metrics for all variables
OUTPUT_TABLE_PRECIP = Path("forecast_scores_precip.csv") # CSI/FSS for precipitation only
OUTPUT_TABLE_PEARSON_BY_TIME = Path("forecast_scores_pearson_by_time.csv")  # Pearson r per time step
OUTPUT_TABLE_BASIC_COMBINED = Path("forecast_scores_basic_combined.csv")
OUTPUT_TABLE_PRECIP_COMBINED = Path("forecast_scores_precip_combined.csv")
OUTPUT_TABLE_PEARSON_BY_TIME_COMBINED = Path("forecast_scores_pearson_by_time_combined.csv")

REG_NETCDF_FILE = Path("reg_wrf_12km_to_3km_200sam.nc")
SCORE_REG_INPUT = True
N_ENSEMBLE   = -1                               # -1 = use all members
ENSEMBLE_REDUCER = "mean"                      # one of: mean, median, max, quantile
ENSEMBLE_QUANTILE = 0.9                        # used only when ENSEMBLE_REDUCER == "quantile"

# Precipitation variable name pattern (for CSI/FSS metrics)
PRECIP_VAR_PATTERN = "RAIN_TOTAL"
PRECIP_VAR_ALIASES = ("RAIN_TOTAL", "PREC_ACC_NC", "APCP_surface")

# Optional aliases when truth and predictor variable names differ.
# Keys are truth variable names; values are preferred predictor alternatives.
VAR_NAME_ALIASES = {
    "PREC_ACC_NC": ["RAIN_TOTAL", "APCP_surface"],
    "RAIN_TOTAL": ["PREC_ACC_NC", "APCP_surface"],
    "APCP_surface": ["RAIN_TOTAL", "PREC_ACC_NC"],
    "Q2": ["q_925"],
}

# For precipitation metrics: values below this are set to 0 before scoring
PRECIP_ZERO_THRESHOLD = 0.1

# Land-ocean masking
LANDMASK_PATH = Path("data/grids/aorc_land_mask_3km.nc")
LANDMASK_VAR = "LANDMASK"
LANDMASK_MODE = "land"  # one of: land, ocean, none

# CSI thresholds (mm)
CSI_THRESHOLDS = [0.1, 1.0, 5.0, 10.0]

# FSS window sizes (pixels)
FSS_WINDOWS = [3, 5, 9]

# Ensemble sharpness interval quantiles (for probabilistic sharpness)
SHARPNESS_INTERVAL_LOW_Q = 0.10
SHARPNESS_INTERVAL_HIGH_Q = 0.90
SHARPNESS_INTERVAL_COL = (
    f"EnsPIWidth_q{int(round(100 * SHARPNESS_INTERVAL_LOW_Q))}_"
    f"q{int(round(100 * SHARPNESS_INTERVAL_HIGH_Q))}"
)

#%% ────────────────────────── METRIC FUNCTIONS ───────────────────────

def rmse(obs, pred):
    """Root Mean Square Error - flattened calculation."""
    obs_flat = obs.values.flatten()
    pred_flat = pred.values.flatten()
    # Remove NaN pairs
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask]
    if len(obs_valid) == 0:
        return np.nan
    return np.sqrt(np.mean((obs_valid - pred_valid) ** 2))


def mae(obs, pred):
    """Mean Absolute Error - flattened calculation."""
    obs_flat = obs.values.flatten()
    pred_flat = pred.values.flatten()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask]
    if len(obs_valid) == 0:
        return np.nan
    return np.mean(np.abs(obs_valid - pred_valid))


def pbias(obs, pred):
    """Percent Bias (%) - measures average tendency to over/under predict.
    PBIAS = 100 * sum(pred - obs) / sum(obs)
    Positive = overestimation, Negative = underestimation
    """
    obs_flat = obs.values.flatten()
    pred_flat = pred.values.flatten()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask]
    if len(obs_valid) == 0 or np.sum(np.abs(obs_valid)) == 0:
        return np.nan
    return 100.0 * np.sum(pred_valid - obs_valid) / np.sum(np.abs(obs_valid))


def pearson_r(obs, pred):
    """Pearson correlation coefficient - flattened calculation."""
    obs_flat = obs.values.flatten()
    pred_flat = pred.values.flatten()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask]
    if len(obs_valid) < 2:
        return np.nan
    if np.std(obs_valid) == 0 or np.std(pred_valid) == 0:
        return np.nan
    corr_matrix = np.corrcoef(obs_valid, pred_valid)
    return corr_matrix[0, 1]


def pearson_r_by_time(truth_da, pred_da):
    """Compute Pearson r separately for each time step."""
    rows = []

    if 'time' not in truth_da.dims:
        rows.append({
            'time_index': 0,
            'time': np.nan,
            'Pearson_r': pearson_r(truth_da, pred_da)
        })
        return rows

    n_times = truth_da.sizes.get('time', 1)
    if 'time' in truth_da.coords:
        time_vals = truth_da['time'].values
    else:
        time_vals = np.arange(n_times)

    for t in range(n_times):
        obs_t = truth_da.isel(time=t)
        pred_t = pred_da.isel(time=t)
        rows.append({
            'time_index': int(t),
            'time': time_vals[t],
            'Pearson_r': pearson_r(obs_t, pred_t)
        })

    return rows


def spatial_pattern_corr_2d(obs, pred):
    """Centered spatial pattern correlation for one 2D field.
    Subtracts spatial mean first, then computes correlation.
    """
    obs_arr = obs.values if hasattr(obs, 'values') else obs
    pred_arr = pred.values if hasattr(pred, 'values') else pred

    if obs_arr.ndim != 2 or pred_arr.ndim != 2:
        return np.nan

    obs_flat = obs_arr.flatten()
    pred_flat = pred_arr.flatten()

    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask]

    if len(obs_valid) < 2:
        return np.nan

    # Center spatially
    obs_valid = obs_valid - np.mean(obs_valid)
    pred_valid = pred_valid - np.mean(pred_valid)

    std_obs = np.std(obs_valid)
    std_pred = np.std(pred_valid)
    if std_obs == 0 or std_pred == 0:
        return np.nan

    return np.corrcoef(obs_valid, pred_valid)[0, 1]


def spatial_pattern_corr(truth_da, pred_da):
    """Compute centered spatial pattern correlation averaged over time."""
    if 'time' not in truth_da.dims:
        return spatial_pattern_corr_2d(truth_da, pred_da)

    vals = []
    n_times = truth_da.sizes.get('time', 1)

    for t in range(n_times):
        obs_2d = truth_da.isel(time=t)
        pred_2d = pred_da.isel(time=t)
        val = spatial_pattern_corr_2d(obs_2d, pred_2d)
        if not np.isnan(val):
            vals.append(val)

    if len(vals) == 0:
        return np.nan
    return np.mean(vals)


def gradient_magnitude(arr):
    """Gradient magnitude for a 2D array."""
    gy, gx = np.gradient(arr)
    return np.sqrt(gx**2 + gy**2)


def sharpness_ratio_2d(obs, pred):
    """Gradient-based texture sharpness ratio for one 2D field.
    Ratio = mean(|grad(pred)|) / mean(|grad(obs)|)

    Interpretation:
      1.0  -> similar sharpness
      <1.0 -> prediction is smoother / blurrier
      >1.0 -> prediction is sharper / possibly noisier
    """
    obs_arr = obs.values if hasattr(obs, 'values') else obs
    pred_arr = pred.values if hasattr(pred, 'values') else pred

    if obs_arr.ndim != 2 or pred_arr.ndim != 2:
        return np.nan

    valid = np.isfinite(obs_arr) & np.isfinite(pred_arr)
    if np.count_nonzero(valid) < 4:
        return np.nan

    # Fill invalid values so gradient can be computed; boundary artifacts are
    # handled below by evaluating gradients only on an eroded valid core.
    obs_fill = np.where(valid, obs_arr, 0.0)
    pred_fill = np.where(valid, pred_arr, 0.0)

    grad_obs = gradient_magnitude(obs_fill)
    grad_pred = gradient_magnitude(pred_fill)

    valid_core = ndimage.binary_erosion(
        valid, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    if not np.any(valid_core):
        valid_core = valid

    mean_grad_obs = float(np.mean(grad_obs[valid_core]))
    mean_grad_pred = float(np.mean(grad_pred[valid_core]))

    if mean_grad_obs == 0:
        return np.nan

    return mean_grad_pred / mean_grad_obs


def sharpness_ratio(truth_da, pred_da):
    """Compute gradient-based texture sharpness ratio averaged over time."""
    if 'time' not in truth_da.dims:
        return sharpness_ratio_2d(truth_da, pred_da)

    vals = []
    n_times = truth_da.sizes.get('time', 1)

    for t in range(n_times):
        obs_2d = truth_da.isel(time=t)
        pred_2d = pred_da.isel(time=t)
        val = sharpness_ratio_2d(obs_2d, pred_2d)
        if not np.isnan(val):
            vals.append(val)

    if len(vals) == 0:
        return np.nan
    return np.mean(vals)


def laplacian_variance_ratio_2d(obs, pred):
    """Laplacian-variance sharpness ratio for one 2D field.

    Ratio = var(laplace(pred)) / var(laplace(obs))

    Interpretation:
      1.0  -> similar high-frequency edge content
      <1.0 -> blurrier / less edge detail than truth
      >1.0 -> sharper / more edge detail (can indicate noise)
    """
    obs_arr = obs.values if hasattr(obs, 'values') else obs
    pred_arr = pred.values if hasattr(pred, 'values') else pred

    if obs_arr.ndim != 2 or pred_arr.ndim != 2:
        return np.nan

    valid = np.isfinite(obs_arr) & np.isfinite(pred_arr)
    if np.count_nonzero(valid) < 9:
        return np.nan

    obs_fill = np.where(valid, obs_arr, 0.0)
    pred_fill = np.where(valid, pred_arr, 0.0)

    lap_obs = ndimage.laplace(obs_fill, mode='nearest')
    lap_pred = ndimage.laplace(pred_fill, mode='nearest')

    valid_core = ndimage.binary_erosion(
        valid, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    if np.count_nonzero(valid_core) < 9:
        valid_core = valid

    lap_obs_valid = lap_obs[valid_core]
    lap_pred_valid = lap_pred[valid_core]

    var_obs = float(np.var(lap_obs_valid))
    var_pred = float(np.var(lap_pred_valid))

    if var_obs == 0:
        return np.nan

    return var_pred / var_obs


def laplacian_variance_ratio(truth_da, pred_da):
    """Compute Laplacian-variance sharpness ratio averaged over time."""
    if 'time' not in truth_da.dims:
        return laplacian_variance_ratio_2d(truth_da, pred_da)

    vals = []
    n_times = truth_da.sizes.get('time', 1)

    for t in range(n_times):
        obs_2d = truth_da.isel(time=t)
        pred_2d = pred_da.isel(time=t)
        val = laplacian_variance_ratio_2d(obs_2d, pred_2d)
        if not np.isnan(val):
            vals.append(val)

    if len(vals) == 0:
        return np.nan
    return np.mean(vals)


def ensemble_spread_mean(pred_da):
    """Probabilistic sharpness proxy: mean ensemble standard deviation.

    Lower values indicate a sharper (more concentrated) forecast distribution.
    """
    if 'ensemble' not in pred_da.dims:
        return np.nan

    spread = pred_da.std('ensemble', skipna=True)
    spread_vals = spread.values.flatten()
    spread_vals = spread_vals[np.isfinite(spread_vals)]
    if spread_vals.size == 0:
        return np.nan
    return float(np.mean(spread_vals))


def ensemble_interval_width_mean(pred_da, q_low, q_high):
    """Probabilistic sharpness proxy: mean central prediction-interval width."""
    if 'ensemble' not in pred_da.dims:
        return np.nan
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("q_low and q_high must satisfy 0 <= q_low < q_high <= 1")

    q_low_field = pred_da.quantile(q_low, dim='ensemble', skipna=True)
    q_high_field = pred_da.quantile(q_high, dim='ensemble', skipna=True)

    if 'quantile' in q_low_field.dims:
        q_low_field = q_low_field.squeeze('quantile', drop=True)
    if 'quantile' in q_high_field.dims:
        q_high_field = q_high_field.squeeze('quantile', drop=True)

    width = q_high_field - q_low_field
    width_vals = width.values.flatten()
    width_vals = width_vals[np.isfinite(width_vals)]
    if width_vals.size == 0:
        return np.nan
    return float(np.mean(width_vals))


def csi(obs, pred, threshold):
    """Critical Success Index (Threat Score) for a given threshold.
    CSI = hits / (hits + misses + false_alarms)
    """
    obs_flat = obs.values.flatten()
    pred_flat = pred.values.flatten()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask]

    if len(obs_valid) == 0:
        return np.nan

    obs_binary = obs_valid >= threshold
    pred_binary = pred_valid >= threshold

    hits = np.sum(obs_binary & pred_binary)
    misses = np.sum(obs_binary & ~pred_binary)
    false_alarms = np.sum(~obs_binary & pred_binary)

    denominator = hits + misses + false_alarms
    if denominator == 0:
        return np.nan

    return hits / denominator


def fss(obs, pred, threshold, window_size):
    """Fractions Skill Score for a given threshold and window size."""
    if hasattr(obs, 'values'):
        obs_arr = obs.values
    else:
        obs_arr = obs
    if hasattr(pred, 'values'):
        pred_arr = pred.values
    else:
        pred_arr = pred

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

    obs_frac = ndimage.convolve(obs_binary, kernel, mode='constant', cval=0.0)
    pred_frac = ndimage.convolve(pred_binary, kernel, mode='constant', cval=0.0)
    valid_count = ndimage.convolve(valid.astype(float), kernel, mode='constant', cval=0.0)

    obs_frac = np.where(valid_count > 0, obs_frac / valid_count, np.nan)
    pred_frac = np.where(valid_count > 0, pred_frac / valid_count, np.nan)

    valid_frac = np.isfinite(obs_frac) & np.isfinite(pred_frac)
    if not np.any(valid_frac):
        return np.nan

    mse_frac = np.mean((obs_frac[valid_frac] - pred_frac[valid_frac]) ** 2)
    mse_ref = np.mean(obs_frac[valid_frac] ** 2) + np.mean(pred_frac[valid_frac] ** 2)

    if mse_ref == 0:
        return np.nan

    fss_value = 1.0 - mse_frac / mse_ref
    return fss_value


def compute_fss_over_time(truth_da, pred_da, threshold, window_size):
    """Compute FSS averaged over all time steps."""
    fss_values = []

    n_times = truth_da.sizes.get('time', 1)

    for t in range(n_times):
        if 'time' in truth_da.dims:
            obs_2d = truth_da.isel(time=t)
            pred_2d = pred_da.isel(time=t)
        else:
            obs_2d = truth_da
            pred_2d = pred_da

        fss_val = fss(obs_2d, pred_2d, threshold, window_size)
        if not np.isnan(fss_val):
            fss_values.append(fss_val)

    if len(fss_values) == 0:
        return np.nan
    return np.mean(fss_values)


def apply_precip_threshold(da, threshold):
    """Set precip values below threshold to 0, preserving NaNs."""
    if threshold is None:
        return da
    return da.where((da >= threshold) | da.isnull(), 0.0)


def _make_domain_mask(root_ds):
    """Load and remap land/ocean mask to model grid as a boolean DataArray."""
    mode = str(LANDMASK_MODE).lower()
    if mode == "none":
        return None
    if mode not in {"land", "ocean"}:
        raise ValueError("LANDMASK_MODE must be one of: land, ocean, none")

    if "lat" not in root_ds.variables or "lon" not in root_ds.variables:
        raise ValueError("Model dataset must contain root variables 'lat' and 'lon' for mask remapping.")

    model_lat = np.asarray(root_ds["lat"].values, dtype=np.float64)
    model_lon = np.asarray(root_ds["lon"].values, dtype=np.float64)

    if not LANDMASK_PATH.exists():
        raise FileNotFoundError(f"LANDMASK_PATH not found: {LANDMASK_PATH}")

    mask_ds = xr.open_dataset(LANDMASK_PATH)
    try:
        if LANDMASK_VAR not in mask_ds:
            raise KeyError(
                f"Mask variable '{LANDMASK_VAR}' not found. Available: {list(mask_ds.data_vars)}"
            )
        if "LAT2D" not in mask_ds or "LON2D" not in mask_ds:
            raise KeyError("Mask dataset must include 'LAT2D' and 'LON2D' variables.")

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

    if mode == "land":
        keep = mapped_mask >= 0.5
    else:
        keep = mapped_mask < 0.5

    mask_dims = root_ds["lat"].dims
    mask_coords = {d: root_ds.coords[d] for d in mask_dims if d in root_ds.coords}
    return xr.DataArray(keep, dims=mask_dims, coords=mask_coords, name=f"{mode}_mask")


def apply_domain_mask(da, spatial_mask):
    """Mask out points outside selected domain (land or ocean)."""
    if spatial_mask is None:
        return da
    return da.where(spatial_mask)


def _normalize_var_name(name):
    """Lowercase alphanumeric-only representation for fuzzy var matching."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _looks_like_precip_name(var_name):
    upper_name = str(var_name).upper()
    return ("PREC" in upper_name) or ("RAIN" in upper_name) or ("APCP" in upper_name)


def _build_normalized_lookup(ds):
    lookup = {}
    for var in ds.data_vars:
        key = _normalize_var_name(var)
        # Keep first occurrence deterministically.
        if key not in lookup:
            lookup[key] = var
    return lookup


def _resolve_predictor_var_name(truth_var_name, pred_ds, norm_lookup):
    """Resolve predictor var name for a given truth var with robust fallbacks."""
    if truth_var_name in pred_ds.data_vars:
        return truth_var_name

    # Explicit alias map: truth -> preferred predictor names
    for candidate in VAR_NAME_ALIASES.get(truth_var_name, []):
        if candidate in pred_ds.data_vars:
            return candidate

    # Reverse alias map: if truth name is an alias of some canonical key
    for canonical, aliases in VAR_NAME_ALIASES.items():
        if truth_var_name in aliases and canonical in pred_ds.data_vars:
            return canonical

    # Normalized exact-name fallback
    norm_truth = _normalize_var_name(truth_var_name)
    if norm_truth in norm_lookup:
        return norm_lookup[norm_truth]

    # Normalized alias fallback
    for candidate in VAR_NAME_ALIASES.get(truth_var_name, []):
        norm_candidate = _normalize_var_name(candidate)
        if norm_candidate in norm_lookup:
            return norm_lookup[norm_candidate]

    # Precipitation fallback across common precip aliases
    if _looks_like_precip_name(truth_var_name):
        for candidate in PRECIP_VAR_ALIASES:
            if candidate in pred_ds.data_vars:
                return candidate
            norm_candidate = _normalize_var_name(candidate)
            if norm_candidate in norm_lookup:
                return norm_lookup[norm_candidate]

        # Last resort: first variable whose name looks like precipitation
        for pred_var in pred_ds.data_vars:
            if _looks_like_precip_name(pred_var):
                return pred_var

    return None


def _score_dataset_pair(root_ds, pred_ds, truth_ds, source_tag):
    """Compute all metric tables for one predictor-vs-truth dataset pair."""
    print("\n" + "-" * 80)
    print(f"Scoring source: {source_tag}")
    print("-" * 80)
    print(f"Variables: {list(truth_ds.data_vars)}")
    print(f"Time steps: {truth_ds.sizes.get('time', 1)}")
    print(f"Ensemble reducer: {ENSEMBLE_REDUCER}")

    spatial_mask = _make_domain_mask(root_ds)
    if spatial_mask is None:
        print("Domain mask: disabled (LANDMASK_MODE='none')")
    else:
        selected_points = int(spatial_mask.sum().item())
        total_points = int(spatial_mask.size)
        print(
            f"Domain mask: mode='{LANDMASK_MODE}', selected {selected_points}/{total_points} "
            f"({100.0 * selected_points / total_points:.2f}%) grid points"
        )

    basic_rows = []
    precip_rows = []
    pearson_time_rows = []
    pred_norm_lookup = _build_normalized_lookup(pred_ds)

    for var in truth_ds.data_vars:
        pred_var_name = _resolve_predictor_var_name(var, pred_ds, pred_norm_lookup)
        if pred_var_name is None:
            print(f"\nSkipping variable '{var}' (no matching variable found in predictor dataset)")
            continue
        if pred_var_name != var:
            print(f"\nProcessing variable: {var} (using predictor variable '{pred_var_name}')")
        else:
            print(f"\nProcessing variable: {var}")

        truth_var = truth_ds[var]
        pred_var_full = pred_ds[pred_var_name]
        pred_var = pred_var_full

        if 'ensemble' in pred_var.dims:
            if ENSEMBLE_REDUCER == "mean":
                pred_var = pred_var.mean('ensemble')
            elif ENSEMBLE_REDUCER == "median":
                pred_var = pred_var.median('ensemble')
            elif ENSEMBLE_REDUCER == "max":
                pred_var = pred_var.max('ensemble')
            elif ENSEMBLE_REDUCER == "quantile":
                pred_var = pred_var.quantile(ENSEMBLE_QUANTILE, dim='ensemble')
                if 'quantile' in pred_var.dims:
                    pred_var = pred_var.squeeze('quantile', drop=True)
            else:
                raise ValueError(
                    f"Unsupported ENSEMBLE_REDUCER='{ENSEMBLE_REDUCER}'. "
                    "Use one of: mean, median, max, quantile"
                )

        is_precip = (
            (PRECIP_VAR_PATTERN in str(var))
            or _looks_like_precip_name(var)
            or _looks_like_precip_name(pred_var_name)
        )
        if is_precip:
            truth_eval = apply_precip_threshold(truth_var, PRECIP_ZERO_THRESHOLD)
            pred_eval = apply_precip_threshold(pred_var, PRECIP_ZERO_THRESHOLD)
            pred_eval_ens = apply_precip_threshold(pred_var_full, PRECIP_ZERO_THRESHOLD)
        else:
            truth_eval = truth_var
            pred_eval = pred_var
            pred_eval_ens = pred_var_full

        truth_eval = apply_domain_mask(truth_eval, spatial_mask)
        pred_eval = apply_domain_mask(pred_eval, spatial_mask)
        pred_eval_ens = apply_domain_mask(pred_eval_ens, spatial_mask)

        basic_row = {'source': source_tag, 'variable': var}
        basic_row['RMSE'] = rmse(truth_eval, pred_eval)
        basic_row['MAE'] = mae(truth_eval, pred_eval)
        basic_row['PBIAS_%'] = pbias(truth_eval, pred_eval)
        basic_row['Pearson_r'] = pearson_r(truth_eval, pred_eval)
        basic_row['SpatialPatternCorr'] = spatial_pattern_corr(truth_eval, pred_eval)
        basic_row['SharpnessRatio'] = sharpness_ratio(truth_eval, pred_eval)
        basic_row['LaplacianVarRatio'] = laplacian_variance_ratio(truth_eval, pred_eval)
        basic_row['EnsSpreadMean'] = ensemble_spread_mean(pred_eval_ens)
        basic_row[SHARPNESS_INTERVAL_COL] = ensemble_interval_width_mean(
            pred_eval_ens, SHARPNESS_INTERVAL_LOW_Q, SHARPNESS_INTERVAL_HIGH_Q
        )

        print(f"  RMSE: {basic_row['RMSE']:.4f}")
        print(f"  MAE: {basic_row['MAE']:.4f}")
        print(f"  PBIAS: {basic_row['PBIAS_%']:.2f}%")
        print(f"  Pearson r: {basic_row['Pearson_r']:.4f}")
        print(f"  Spatial pattern corr: {basic_row['SpatialPatternCorr']:.4f}")
        print(f"  Sharpness ratio: {basic_row['SharpnessRatio']:.4f}")
        print(f"  Laplacian variance ratio: {basic_row['LaplacianVarRatio']:.4f}")
        print(f"  Ensemble spread mean: {basic_row['EnsSpreadMean']:.4f}")
        print(f"  {SHARPNESS_INTERVAL_COL}: {basic_row[SHARPNESS_INTERVAL_COL]:.4f}")

        basic_rows.append(basic_row)

        time_rows = pearson_r_by_time(truth_eval, pred_eval)
        for row in time_rows:
            row['source'] = source_tag
            row['variable'] = var
            pearson_time_rows.append(row)

        if is_precip:
            print("  Computing precipitation-specific metrics...")
            precip_row = {'source': source_tag, 'variable': var}
            for thresh in CSI_THRESHOLDS:
                col_name = f'CSI_{thresh}'
                precip_row[col_name] = csi(truth_eval, pred_eval, thresh)
                print(f"    CSI@{thresh}mm: {precip_row[col_name]:.4f}")

            fss_threshold = 0.1
            for win in FSS_WINDOWS:
                col_name = f'FSS_w{win}'
                precip_row[col_name] = compute_fss_over_time(truth_eval, pred_eval, fss_threshold, win)
                print(f"    FSS w={win}: {precip_row[col_name]:.4f}")
            precip_rows.append(precip_row)

    return basic_rows, precip_rows, pearson_time_rows


def _build_basic_df(rows, include_source=False):
    base_cols = [
        'variable',
        'RMSE',
        'MAE',
        'PBIAS_%',
        'Pearson_r',
        'SpatialPatternCorr',
        'SharpnessRatio',
        'LaplacianVarRatio',
        'EnsSpreadMean',
        SHARPNESS_INTERVAL_COL,
    ]
    cols = (['source'] + base_cols) if include_source else base_cols
    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)[cols]
    if include_source:
        df = df.set_index(['source', 'variable']).round(4).sort_index()
    else:
        df = df.drop(columns=['source'], errors='ignore').set_index('variable').round(4).sort_index()
    return df


def _build_precip_df(rows, include_source=False):
    csi_cols = [f'CSI_{t}' for t in CSI_THRESHOLDS]
    fss_cols = [f'FSS_w{w}' for w in FSS_WINDOWS]
    base_cols = ['variable'] + csi_cols + fss_cols
    cols = (['source'] + base_cols) if include_source else base_cols
    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)[cols]
    if include_source:
        df = df.set_index(['source', 'variable']).round(4).sort_index()
    else:
        df = df.drop(columns=['source'], errors='ignore').set_index('variable').round(4).sort_index()
    return df


def _build_pearson_time_df(rows, include_source=False):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if 'Pearson_r' in df.columns:
        df['Pearson_r'] = df['Pearson_r'].round(6)
    if include_source:
        order_cols = ['source', 'variable', 'time_index']
    else:
        order_cols = ['variable', 'time_index']
        df = df.drop(columns=['source'], errors='ignore')
    df = df.sort_values(order_cols).reset_index(drop=True)
    return df


#%% ──────────────────────────── LOAD DATA ────────────────────────────
print(f"Loading forecast data from: {NETCDF_FILE}")
root_ds = xr.open_dataset(NETCDF_FILE)
pred_ds = xr.open_dataset(NETCDF_FILE, group="prediction")
truth_ds = xr.open_dataset(NETCDF_FILE, group="truth")
for ds in (pred_ds, truth_ds):
    ds.coords.update(root_ds.coords)
if N_ENSEMBLE > 0 and 'ensemble' in pred_ds.dims:
    pred_ds = pred_ds.isel(ensemble=slice(0, N_ENSEMBLE))

# Primary score pass: prediction vs truth from NETCDF_FILE
basic_rows_main, precip_rows_main, pearson_rows_main = _score_dataset_pair(
    root_ds=root_ds,
    pred_ds=pred_ds,
    truth_ds=truth_ds,
    source_tag='prediction',
)

# Build/save original single-source outputs (backward compatible)
basic_df = _build_basic_df(basic_rows_main, include_source=False)
precip_df = _build_precip_df(precip_rows_main, include_source=False)
pearson_time_df = _build_pearson_time_df(pearson_rows_main, include_source=False)

print("\n" + "=" * 80)
print("TABLE 1: BASIC METRICS (All Variables)")
print("=" * 80)
print(basic_df.to_string())
print("=" * 80 + "\n")
basic_df.to_csv(OUTPUT_TABLE_BASIC)
print(f"Saved basic metrics to: {OUTPUT_TABLE_BASIC.resolve()}")

if not precip_df.empty:
    print("\n" + "=" * 80)
    print("TABLE 2: PRECIPITATION METRICS (CSI / FSS)")
    print("=" * 80)
    print(precip_df.to_string())
    print("=" * 80 + "\n")
    precip_df.to_csv(OUTPUT_TABLE_PRECIP)
    print(f"Saved precipitation metrics to: {OUTPUT_TABLE_PRECIP.resolve()}")
else:
    print("\nNo precipitation variables found for CSI/FSS metrics.")

if not pearson_time_df.empty:
    pearson_time_df.to_csv(OUTPUT_TABLE_PEARSON_BY_TIME, index=False)
    print(f"Saved Pearson r per time step to: {OUTPUT_TABLE_PEARSON_BY_TIME.resolve()}")

# Optional second score pass: input vs truth from regression NetCDF
all_basic_rows = list(basic_rows_main)
all_precip_rows = list(precip_rows_main)
all_pearson_rows = list(pearson_rows_main)

if SCORE_REG_INPUT:
    if REG_NETCDF_FILE.exists():
        print(f"\nLoading regression-input data from: {REG_NETCDF_FILE}")
        root_reg = xr.open_dataset(REG_NETCDF_FILE)
        try:
            reg_input_ds = xr.open_dataset(REG_NETCDF_FILE, group="input")
            reg_truth_ds = xr.open_dataset(REG_NETCDF_FILE, group="truth")
        except Exception as e:
            print(f"Warning: could not open input/truth groups from {REG_NETCDF_FILE}: {e}")
            reg_input_ds = None
            reg_truth_ds = None

        if reg_input_ds is not None and reg_truth_ds is not None:
            for ds in (reg_input_ds, reg_truth_ds):
                ds.coords.update(root_reg.coords)

            b2, p2, t2 = _score_dataset_pair(
                root_ds=root_reg,
                pred_ds=reg_input_ds,
                truth_ds=reg_truth_ds,
                source_tag='reg_input',
            )
            all_basic_rows.extend(b2)
            all_precip_rows.extend(p2)
            all_pearson_rows.extend(t2)

            reg_input_ds.close()
            reg_truth_ds.close()
        root_reg.close()
    else:
        print(f"Warning: SCORE_REG_INPUT=True but REG_NETCDF_FILE does not exist: {REG_NETCDF_FILE}")

# Save combined outputs (prediction + reg_input when available)
basic_df_combined = _build_basic_df(all_basic_rows, include_source=True)
precip_df_combined = _build_precip_df(all_precip_rows, include_source=True)
pearson_time_df_combined = _build_pearson_time_df(all_pearson_rows, include_source=True)

basic_df_combined.to_csv(OUTPUT_TABLE_BASIC_COMBINED)
print(f"Saved combined basic metrics to: {OUTPUT_TABLE_BASIC_COMBINED.resolve()}")

if not precip_df_combined.empty:
    precip_df_combined.to_csv(OUTPUT_TABLE_PRECIP_COMBINED)
    print(f"Saved combined precipitation metrics to: {OUTPUT_TABLE_PRECIP_COMBINED.resolve()}")

if not pearson_time_df_combined.empty:
    pearson_time_df_combined.to_csv(OUTPUT_TABLE_PEARSON_BY_TIME_COMBINED, index=False)
    print(f"Saved combined Pearson r per time step to: {OUTPUT_TABLE_PEARSON_BY_TIME_COMBINED.resolve()}")

pred_ds.close()
truth_ds.close()
root_ds.close()

# %%
