#%% ───────────────────────────── IMPORTS ─────────────────────────────
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from scipy.spatial import cKDTree

#%% ─────────────────────────── CONFIGURATION ─────────────────────────
# NetCDF file with groups (expects at least: input + truth)
NETCDF_FILE = Path("reg_wrf_12km_to_3km_200sam.nc")
INPUT_GROUP = "input"
TARGET_GROUP = "truth"

# Output files
OUTPUT_TABLE_BASIC = Path("input_vs_target_scores_basic.csv")
OUTPUT_TABLE_PRECIP = Path("input_vs_target_scores_precip.csv")
OUTPUT_TABLE_PEARSON_BY_TIME = Path("input_vs_target_scores_pearson_by_time.csv")

# Optional mapping when variable names differ between input and truth.
# Example: {"RAIN_TOTAL": "PREC_ACC_NC"}
TRUTH_TO_INPUT_VAR_MAP = {}
REQUIRE_ALL_TRUTH_VARS = False

# Precipitation identification / preprocessing
PRECIP_VAR_NAMES = {"PREC_ACC_NC", "RAIN_TOTAL", "APCP_surface"}
PRECIP_ZERO_THRESHOLD = 0.1
CSI_THRESHOLDS = [0.1, 1.0, 5.0, 10.0]
FSS_WINDOWS = [3, 5, 9]

# Land-ocean masking
LANDMASK_PATH = Path("data/grids/aorc_land_mask_3km.nc")
LANDMASK_VAR = "LANDMASK"
LANDMASK_MODE = "land"  # one of: land, ocean, none


#%% ────────────────────────── METRIC FUNCTIONS ───────────────────────
def rmse(obs, pred):
    obs_flat = obs.values.flatten()
    pred_flat = pred.values.flatten()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask]
    if len(obs_valid) == 0:
        return np.nan
    return np.sqrt(np.mean((obs_valid - pred_valid) ** 2))


def mae(obs, pred):
    obs_flat = obs.values.flatten()
    pred_flat = pred.values.flatten()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask]
    if len(obs_valid) == 0:
        return np.nan
    return np.mean(np.abs(obs_valid - pred_valid))


def pbias(obs, pred):
    obs_flat = obs.values.flatten()
    pred_flat = pred.values.flatten()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask]
    if len(obs_valid) == 0 or np.sum(np.abs(obs_valid)) == 0:
        return np.nan
    return 100.0 * np.sum(pred_valid - obs_valid) / np.sum(np.abs(obs_valid))


def pearson_r(obs, pred):
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
    rows = []

    if "time" not in truth_da.dims:
        rows.append(
            {
                "time_index": 0,
                "time": np.nan,
                "Pearson_r": pearson_r(truth_da, pred_da),
            }
        )
        return rows

    n_times = truth_da.sizes.get("time", 1)
    if "time" in truth_da.coords:
        time_vals = truth_da["time"].values
    else:
        time_vals = np.arange(n_times)

    for t in range(n_times):
        obs_t = truth_da.isel(time=t)
        pred_t = pred_da.isel(time=t)
        rows.append(
            {
                "time_index": int(t),
                "time": time_vals[t],
                "Pearson_r": pearson_r(obs_t, pred_t),
            }
        )

    return rows


def csi(obs, pred, threshold):
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
    if hasattr(obs, "values"):
        obs_arr = obs.values
    else:
        obs_arr = obs
    if hasattr(pred, "values"):
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
    obs_frac = ndimage.convolve(obs_binary, kernel, mode="constant", cval=0.0)
    pred_frac = ndimage.convolve(pred_binary, kernel, mode="constant", cval=0.0)
    valid_count = ndimage.convolve(valid.astype(float), kernel, mode="constant", cval=0.0)
    obs_frac = np.divide(
        obs_frac,
        valid_count,
        out=np.full(obs_frac.shape, np.nan, dtype=np.float64),
        where=valid_count > 0,
    )
    pred_frac = np.divide(
        pred_frac,
        valid_count,
        out=np.full(pred_frac.shape, np.nan, dtype=np.float64),
        where=valid_count > 0,
    )

    valid_frac = np.isfinite(obs_frac) & np.isfinite(pred_frac)
    if not np.any(valid_frac):
        return np.nan

    mse_frac = np.mean((obs_frac[valid_frac] - pred_frac[valid_frac]) ** 2)
    mse_ref = np.mean(obs_frac[valid_frac] ** 2) + np.mean(pred_frac[valid_frac] ** 2)
    if mse_ref == 0:
        return np.nan

    return 1.0 - mse_frac / mse_ref


def compute_fss_over_time(truth_da, pred_da, threshold, window_size):
    fss_values = []
    n_times = truth_da.sizes.get("time", 1)

    for t in range(n_times):
        if "time" in truth_da.dims:
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
    if threshold is None:
        return da
    return da.where((da >= threshold) | da.isnull(), 0.0)


def _looks_like_precip(var_name):
    if var_name in PRECIP_VAR_NAMES:
        return True
    upper_name = var_name.upper()
    return ("PREC" in upper_name) or ("RAIN" in upper_name) or ("APCP" in upper_name)


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


#%% ──────────────────────────── LOAD DATA ────────────────────────────
print(f"Loading data from: {NETCDF_FILE}")
root_ds = xr.open_dataset(NETCDF_FILE)
input_ds = xr.open_dataset(NETCDF_FILE, group=INPUT_GROUP)
truth_ds = xr.open_dataset(NETCDF_FILE, group=TARGET_GROUP)

for ds in (input_ds, truth_ds):
    ds.coords.update(root_ds.coords)

print(f"Input variables ({len(input_ds.data_vars)}): {list(input_ds.data_vars)}")
print(f"Truth variables ({len(truth_ds.data_vars)}): {list(truth_ds.data_vars)}")
print(f"Time steps: {truth_ds.sizes.get('time', 1)}")

spatial_mask = _make_domain_mask(root_ds)
if spatial_mask is None:
    print("Domain mask: disabled")
else:
    selected_points = int(spatial_mask.sum().item())
    total_points = int(spatial_mask.size)
    print(
        f"Domain mask: mode='{LANDMASK_MODE}', selected {selected_points}/{total_points} "
        f"({100.0 * selected_points / total_points:.2f}%) grid points"
    )

# Build truth->input variable pairs.
pairs = []
missing = []
for truth_var in truth_ds.data_vars:
    input_var = TRUTH_TO_INPUT_VAR_MAP.get(truth_var, truth_var)
    if input_var in input_ds.data_vars:
        pairs.append((truth_var, input_var))
    else:
        missing.append((truth_var, input_var))

if missing:
    print("\nVariables skipped because no matching input variable was found:")
    for truth_var, input_var in missing:
        print(f"  truth='{truth_var}' -> input='{input_var}' (missing)")
    if REQUIRE_ALL_TRUTH_VARS:
        raise ValueError("Missing mapped input variables for one or more truth variables.")

if not pairs:
    raise ValueError("No overlapping variables between input and truth groups.")

print("\nScoring pairs:")
for truth_var, input_var in pairs:
    print(f"  truth='{truth_var}' vs input='{input_var}'")


#%% ────────────────────────── COMPUTE METRICS ────────────────────────
basic_rows = []
precip_rows = []
pearson_time_rows = []

for truth_var_name, input_var_name in pairs:
    print(f"\nProcessing truth='{truth_var_name}' vs input='{input_var_name}'")
    truth_var = truth_ds[truth_var_name]
    input_var = input_ds[input_var_name]

    is_precip = _looks_like_precip(truth_var_name) or _looks_like_precip(input_var_name)
    if is_precip:
        truth_eval = apply_precip_threshold(truth_var, PRECIP_ZERO_THRESHOLD)
        input_eval = apply_precip_threshold(input_var, PRECIP_ZERO_THRESHOLD)
    else:
        truth_eval = truth_var
        input_eval = input_var

    truth_eval = apply_domain_mask(truth_eval, spatial_mask)
    input_eval = apply_domain_mask(input_eval, spatial_mask)

    basic_row = {
        "truth_variable": truth_var_name,
        "input_variable": input_var_name,
        "RMSE": rmse(truth_eval, input_eval),
        "MAE": mae(truth_eval, input_eval),
        "PBIAS_%": pbias(truth_eval, input_eval),
        "Pearson_r": pearson_r(truth_eval, input_eval),
    }
    basic_rows.append(basic_row)
    print(f"  RMSE: {basic_row['RMSE']:.4f}")
    print(f"  MAE: {basic_row['MAE']:.4f}")
    print(f"  PBIAS: {basic_row['PBIAS_%']:.2f}%")
    print(f"  Pearson r: {basic_row['Pearson_r']:.4f}")

    time_rows = pearson_r_by_time(truth_eval, input_eval)
    for row in time_rows:
        row["truth_variable"] = truth_var_name
        row["input_variable"] = input_var_name
        pearson_time_rows.append(row)

    if is_precip:
        precip_row = {
            "truth_variable": truth_var_name,
            "input_variable": input_var_name,
        }

        for thresh in CSI_THRESHOLDS:
            col_name = f"CSI_{thresh}"
            precip_row[col_name] = csi(truth_eval, input_eval, thresh)
            print(f"  CSI@{thresh}mm: {precip_row[col_name]:.4f}")

        for win in FSS_WINDOWS:
            col_name = f"FSS_w{win}"
            precip_row[col_name] = compute_fss_over_time(
                truth_eval, input_eval, threshold=0.1, window_size=win
            )
            print(f"  FSS w={win}: {precip_row[col_name]:.4f}")

        precip_rows.append(precip_row)


#%% ────────────────────────── CREATE DATAFRAMES ──────────────────────
basic_cols = ["truth_variable", "input_variable", "RMSE", "MAE", "PBIAS_%", "Pearson_r"]
basic_df = pd.DataFrame(basic_rows)[basic_cols]
basic_df = basic_df.sort_values(["truth_variable", "input_variable"]).reset_index(drop=True).round(4)

csi_cols = [f"CSI_{t}" for t in CSI_THRESHOLDS]
fss_cols = [f"FSS_w{w}" for w in FSS_WINDOWS]
precip_cols = ["truth_variable", "input_variable"] + csi_cols + fss_cols

if precip_rows:
    precip_df = pd.DataFrame(precip_rows)[precip_cols]
    precip_df = (
        precip_df.sort_values(["truth_variable", "input_variable"])
        .reset_index(drop=True)
        .round(4)
    )
else:
    precip_df = pd.DataFrame()

if pearson_time_rows:
    pearson_time_df = pd.DataFrame(pearson_time_rows)
    if "Pearson_r" in pearson_time_df.columns:
        pearson_time_df["Pearson_r"] = pearson_time_df["Pearson_r"].round(6)
    pearson_time_df = pearson_time_df.sort_values(
        ["truth_variable", "input_variable", "time_index"]
    ).reset_index(drop=True)
else:
    pearson_time_df = pd.DataFrame()


#%% ──────────────────────────── OUTPUT ───────────────────────────────
print("\n" + "=" * 80)
print("TABLE 1: INPUT VS TARGET BASIC METRICS")
print("=" * 80)
print(basic_df.to_string(index=False))
print("=" * 80 + "\n")
basic_df.to_csv(OUTPUT_TABLE_BASIC, index=False)
print(f"Saved basic metrics to: {OUTPUT_TABLE_BASIC.resolve()}")

if not precip_df.empty:
    print("\n" + "=" * 80)
    print("TABLE 2: INPUT VS TARGET PRECIPITATION METRICS (CSI / FSS)")
    print("=" * 80)
    print(precip_df.to_string(index=False))
    print("=" * 80 + "\n")
    precip_df.to_csv(OUTPUT_TABLE_PRECIP, index=False)
    print(f"Saved precipitation metrics to: {OUTPUT_TABLE_PRECIP.resolve()}")
else:
    print("\nNo precipitation variables found for CSI/FSS metrics.")

if not pearson_time_df.empty:
    pearson_time_df.to_csv(OUTPUT_TABLE_PEARSON_BY_TIME, index=False)
    print(f"Saved Pearson r per time step to: {OUTPUT_TABLE_PEARSON_BY_TIME.resolve()}")

# %%
