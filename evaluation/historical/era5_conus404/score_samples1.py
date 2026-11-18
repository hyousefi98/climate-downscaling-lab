#%% ───────────────────────────── IMPORTS ─────────────────────────────
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import ndimage

#%% ─────────────────────────── CONFIGURATION ─────────────────────────
# NETCDF_FILE  = Path("reg_wrf_random200_seed40.nc")  # your NetCDF with groups
# NETCDF_FILE  = Path("reg_era5_to_conus404_32ens_200sam.nc")
NETCDF_FILE  = Path("all_era5_to_conus404_200sam_32ens.nc")
NETCDF_FILE  = Path("reg_era5_to_conus404_200sam.nc")

OUTPUT_TABLE_BASIC = Path("forecast_scores_basic.csv")   # Basic metrics for all variables
OUTPUT_TABLE_PRECIP = Path("forecast_scores_precip.csv") # CSI/FSS for precipitation only
OUTPUT_TABLE_PEARSON_BY_TIME = Path("forecast_scores_pearson_by_time.csv")  # Pearson r per time step
N_ENSEMBLE   = -1                               # -1 = use all members
ENSEMBLE_REDUCER = "mean"                      # one of: mean, median, max, quantile
ENSEMBLE_QUANTILE = 0.9                        # used only when ENSEMBLE_REDUCER == "quantile"

# Precipitation variable name pattern (for CSI/FSS metrics)
PRECIP_VAR_PATTERN = "PREC_ACC_NC"

# For precipitation metrics: values below this are set to 0 before scoring
PRECIP_ZERO_THRESHOLD = 0.1

# CSI thresholds (mm)
CSI_THRESHOLDS = [0.1, 1.0, 5.0, 10.0]

# FSS window sizes (pixels)
FSS_WINDOWS = [3, 5, 9]

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
    # Use numpy corrcoef
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


def csi(obs, pred, threshold):
    """Critical Success Index (Threat Score) for a given threshold.
    CSI = hits / (hits + misses + false_alarms)

    Args:
        obs: Observed values
        pred: Predicted values
        threshold: Threshold value (e.g., 0.1 mm for precipitation)

    Returns:
        CSI value between 0 and 1
    """
    obs_flat = obs.values.flatten()
    pred_flat = pred.values.flatten()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask]

    if len(obs_valid) == 0:
        return np.nan

    # Binary classification
    obs_binary = obs_valid >= threshold
    pred_binary = pred_valid >= threshold

    hits = np.sum(obs_binary & pred_binary)
    misses = np.sum(obs_binary & ~pred_binary)
    false_alarms = np.sum(~obs_binary & pred_binary)

    denominator = hits + misses + false_alarms
    if denominator == 0:
        return np.nan  # No events above threshold

    return hits / denominator


def fss(obs, pred, threshold, window_size):
    """Fractions Skill Score for a given threshold and window size.

    FSS compares the fractional coverage of events within neighborhoods.
    FSS = 1 - MSE_frac / (MSE_frac_ref)

    Args:
        obs: 2D observed field
        pred: 2D predicted field
        threshold: Threshold for binary conversion
        window_size: Size of the neighborhood window (must be odd)

    Returns:
        FSS value between 0 and 1
    """
    # Handle xarray DataArray
    if hasattr(obs, 'values'):
        obs_arr = obs.values
    else:
        obs_arr = obs
    if hasattr(pred, 'values'):
        pred_arr = pred.values
    else:
        pred_arr = pred

    # Ensure 2D
    if obs_arr.ndim != 2 or pred_arr.ndim != 2:
        return np.nan

    # Replace NaN with 0 for convolution (will affect edges)
    obs_arr = np.nan_to_num(obs_arr, nan=0.0)
    pred_arr = np.nan_to_num(pred_arr, nan=0.0)

    # Binary fields
    obs_binary = (obs_arr >= threshold).astype(float)
    pred_binary = (pred_arr >= threshold).astype(float)

    # Create uniform kernel for averaging
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)

    # Compute fractional coverage using convolution
    obs_frac = ndimage.convolve(obs_binary, kernel, mode='constant', cval=0.0)
    pred_frac = ndimage.convolve(pred_binary, kernel, mode='constant', cval=0.0)

    # MSE of fractions
    mse_frac = np.mean((obs_frac - pred_frac) ** 2)

    # Reference MSE (worst case - no overlap)
    mse_ref = np.mean(obs_frac ** 2) + np.mean(pred_frac ** 2)

    if mse_ref == 0:
        return np.nan  # No events

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


#%% ──────────────────────────── LOAD DATA ────────────────────────────
print(f"Loading data from: {NETCDF_FILE}")
root_ds  = xr.open_dataset(NETCDF_FILE)
pred_ds  = xr.open_dataset(NETCDF_FILE, group="prediction")
truth_ds = xr.open_dataset(NETCDF_FILE, group="truth")

# Add lon/lat to sub-datasets
for ds in (pred_ds, truth_ds):
    ds.coords.update(root_ds.coords)

# Optionally thin the ensemble
if N_ENSEMBLE > 0 and 'ensemble' in pred_ds.dims:
    pred_ds = pred_ds.isel(ensemble=slice(0, N_ENSEMBLE))

print(f"Variables: {list(truth_ds.data_vars)}")
print(f"Time steps: {truth_ds.sizes.get('time', 1)}")
print(f"Ensemble reducer: {ENSEMBLE_REDUCER}")

#%% ────────────────────────── COMPUTE METRICS ────────────────────────
basic_rows = []
precip_rows = []
pearson_time_rows = []

for var in truth_ds.data_vars:
    print(f"\nProcessing variable: {var}")

    truth_var = truth_ds[var]
    pred_var = pred_ds[var]

    # If prediction has ensemble dimension, collapse it using configured reducer.
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

    # Identify precipitation variables early (used for thresholding)
    is_precip = PRECIP_VAR_PATTERN in var

    # Apply precip thresholding (values < PRECIP_ZERO_THRESHOLD -> 0) for metrics
    if is_precip:
        truth_eval = apply_precip_threshold(truth_var, PRECIP_ZERO_THRESHOLD)
        pred_eval = apply_precip_threshold(pred_var, PRECIP_ZERO_THRESHOLD)
    else:
        truth_eval = truth_var
        pred_eval = pred_var

    # Basic metrics dictionary (for ALL variables)
    basic_row = {'variable': var}

    # RMSE
    basic_row['RMSE'] = rmse(truth_eval, pred_eval)
    print(f"  RMSE: {basic_row['RMSE']:.4f}")

    # MAE
    basic_row['MAE'] = mae(truth_eval, pred_eval)
    print(f"  MAE: {basic_row['MAE']:.4f}")

    # PBIAS
    basic_row['PBIAS_%'] = pbias(truth_eval, pred_eval)
    print(f"  PBIAS: {basic_row['PBIAS_%']:.2f}%")

    # Pearson r
    basic_row['Pearson_r'] = pearson_r(truth_eval, pred_eval)
    print(f"  Pearson r: {basic_row['Pearson_r']:.4f}")

    basic_rows.append(basic_row)

    # Pearson r per time step (for ALL variables)
    time_rows = pearson_r_by_time(truth_eval, pred_eval)
    for row in time_rows:
        row['variable'] = var
        pearson_time_rows.append(row)

    # CSI and FSS only for precipitation variables
    if is_precip:
        print(f"  Computing precipitation-specific metrics...")
        precip_row = {'variable': var}

        # CSI at different thresholds
        for thresh in CSI_THRESHOLDS:
            col_name = f'CSI_{thresh}'
            precip_row[col_name] = csi(truth_eval, pred_eval, thresh)
            print(f"    CSI@{thresh}mm: {precip_row[col_name]:.4f}")

        # FSS at different window sizes (using threshold 0.1 mm)
        fss_threshold = 0.1
        for win in FSS_WINDOWS:
            col_name = f'FSS_w{win}'
            precip_row[col_name] = compute_fss_over_time(truth_eval, pred_eval, fss_threshold, win)
            print(f"    FSS w={win}: {precip_row[col_name]:.4f}")

        precip_rows.append(precip_row)

#%% ────────────────────────── CREATE DATAFRAMES ──────────────────────
# Table 1: Basic metrics for ALL variables
basic_cols = ['variable', 'RMSE', 'MAE', 'PBIAS_%', 'Pearson_r']
basic_df = pd.DataFrame(basic_rows)[basic_cols]
basic_df = basic_df.set_index('variable').round(4).sort_index()

# Table 2: Precipitation-specific metrics (CSI, FSS) - only for precip variables
csi_cols = [f'CSI_{t}' for t in CSI_THRESHOLDS]
fss_cols = [f'FSS_w{w}' for w in FSS_WINDOWS]
precip_cols = ['variable'] + csi_cols + fss_cols

if precip_rows:
    precip_df = pd.DataFrame(precip_rows)[precip_cols]
    precip_df = precip_df.set_index('variable').round(4).sort_index()
else:
    precip_df = pd.DataFrame()

# Table 3: Pearson r per time step
if pearson_time_rows:
    pearson_time_df = pd.DataFrame(pearson_time_rows)
    if 'Pearson_r' in pearson_time_df.columns:
        pearson_time_df['Pearson_r'] = pearson_time_df['Pearson_r'].round(6)
    pearson_time_df = pearson_time_df.sort_values(['variable', 'time_index']).reset_index(drop=True)
else:
    pearson_time_df = pd.DataFrame()

#%% ──────────────────────────── OUTPUT ───────────────────────────────
print("\n" + "=" * 80)
print("TABLE 1: BASIC METRICS (All Variables)")
print("=" * 80)
print(basic_df.to_string())
print("=" * 80 + "\n")

# Save basic metrics to CSV
basic_df.to_csv(OUTPUT_TABLE_BASIC)
print(f"Saved basic metrics to: {OUTPUT_TABLE_BASIC.resolve()}")

if not precip_df.empty:
    print("\n" + "=" * 80)
    print("TABLE 2: PRECIPITATION METRICS (CSI / FSS)")
    print("=" * 80)
    print(precip_df.to_string())
    print("=" * 80 + "\n")

    # Save precip metrics to CSV
    precip_df.to_csv(OUTPUT_TABLE_PRECIP)
    print(f"Saved precipitation metrics to: {OUTPUT_TABLE_PRECIP.resolve()}")
else:
    print("\nNo precipitation variables found for CSI/FSS metrics.")

if not pearson_time_df.empty:
    pearson_time_df.to_csv(OUTPUT_TABLE_PEARSON_BY_TIME, index=False)
    print(f"Saved Pearson r per time step to: {OUTPUT_TABLE_PEARSON_BY_TIME.resolve()}")

# %%
