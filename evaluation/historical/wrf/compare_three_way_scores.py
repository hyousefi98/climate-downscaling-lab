#%% ───────────────────────────── IMPORTS ─────────────────────────────
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from scipy import ndimage
from scipy.spatial import cKDTree


#%% ─────────────────────────── CONFIGURATION ─────────────────────────
# Existing score tables
INPUT_BASIC_CSV = Path("input_vs_target_scores_basic.csv")
INPUT_PRECIP_CSV = Path("input_vs_target_scores_precip.csv")

REGRESSION_BASIC_CSV = Path("regression_vs_target_scores_basic.csv")
REGRESSION_PRECIP_CSV = Path("regression_vs_target_scores_precip.csv")

REGDIFF_BASIC_CSV = Path("forecast_scores_basic.csv")
REGDIFF_PRECIP_CSV = Path("forecast_scores_precip.csv")

# If regression score files are missing, compute them from this NetCDF.
REGRESSION_NETCDF = Path("reg_wrf_12km_to_3km_200sam.nc")

# Output artifacts
OUTPUT_LONG_CSV = Path("comparison_three_way_long.csv")
OUTPUT_WIDE_CSV = Path("comparison_three_way_wide.csv")
OUTPUT_SUMMARY_CSV = Path("comparison_three_way_metric_summary.csv")
OUTPUT_PLOT = Path("comparison_three_way_skill_heatmap.png")
OUTPUT_VARIABLE_TRAJECTORY_CSV = Path("comparison_three_way_variable_trajectory.csv")
OUTPUT_VARIABLE_TRAJECTORY_PLOT = Path("comparison_three_way_variable_trajectory.png")
OUTPUT_VARIABLE_TRAJECTORY_BY_METRIC_CSV = Path(
    "comparison_three_way_variable_trajectory_by_metric.csv"
)
OUTPUT_VARIABLE_TRAJECTORY_BY_METRIC_PLOT = Path(
    "comparison_three_way_variable_trajectory_by_metric.png"
)
OUTPUT_TRIPLET_TABLE_CSV = Path("comparison_three_way_triplet_table.csv")
OUTPUT_TRIPLET_TABLE_TXT = Path("comparison_three_way_triplet_table.txt")
OUTPUT_TRIPLET_TABLE_LOGS_CSV = Path(
    "diffusion_pbias_sweep_beststeps_n20/logs/comparison_three_way_triplet_table.csv"
)
OUTPUT_TRIPLET_PRECIP_TABLE_CSV = Path("comparison_three_way_triplet_table_precip.csv")
OUTPUT_TRIPLET_PRECIP_TABLE_TXT = Path("comparison_three_way_triplet_table_precip.txt")
OUTPUT_TRIPLET_PRECIP_TABLE_LOGS_CSV = Path(
    "diffusion_pbias_sweep_beststeps_n20/logs/comparison_three_way_triplet_table_precip.csv"
)
OUTPUT_VARIABLE_TRAJECTORY_CSV_LOGS = Path(
    "diffusion_pbias_sweep_beststeps_n20/logs/comparison_three_way_variable_trajectory.csv"
)
OUTPUT_VARIABLE_TRAJECTORY_PLOT_LOGS = Path(
    "diffusion_pbias_sweep_beststeps_n20/logs/comparison_three_way_variable_trajectory.png"
)
OUTPUT_VARIABLE_TRAJECTORY_BY_METRIC_CSV_LOGS = Path(
    "diffusion_pbias_sweep_beststeps_n20/logs/comparison_three_way_variable_trajectory_by_metric.csv"
)
OUTPUT_VARIABLE_TRAJECTORY_BY_METRIC_PLOT_LOGS = Path(
    "diffusion_pbias_sweep_beststeps_n20/logs/comparison_three_way_variable_trajectory_by_metric.png"
)
TRIPLET_DECIMALS = 2

# Metrics and interpretation
BASIC_METRICS = ["RMSE", "MAE", "PBIAS_%", "Pearson_r"]
PRECIP_METRICS = ["CSI_0.1", "CSI_1.0", "CSI_5.0", "CSI_10.0", "FSS_w3", "FSS_w5", "FSS_w9"]
LOWER_BETTER = {"RMSE", "MAE", "PBIAS_%"}
HIGHER_BETTER = {"Pearson_r", "CSI_0.1", "CSI_1.0", "CSI_5.0", "CSI_10.0", "FSS_w3", "FSS_w5", "FSS_w9"}
TRAJECTORY_METRICS = BASIC_METRICS.copy()
TRAJECTORY_MODELS = ["Input", "Regression", "Reg+Diffusion"]

# Regression scoring settings (kept aligned with score_samples2.py)
PRECIP_VAR_NAMES = {"PREC_ACC_NC", "RAIN_TOTAL", "APCP_surface"}
PRECIP_ZERO_THRESHOLD = 0.1
CSI_THRESHOLDS = [0.1, 1.0, 5.0, 10.0]
FSS_WINDOWS = [3, 5, 9]
LANDMASK_PATH = Path("data/grids/aorc_land_mask_3km.nc")
LANDMASK_VAR = "LANDMASK"
LANDMASK_MODE = "land"  # one of: land, ocean, none


#%% ────────────────────────── METRIC FUNCTIONS ───────────────────────
def rmse(obs, pred):
    obs_flat = obs.values.flatten()
    pred_flat = pred.values.flatten()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    if not np.any(mask):
        return np.nan
    return np.sqrt(np.mean((obs_flat[mask] - pred_flat[mask]) ** 2))


def mae(obs, pred):
    obs_flat = obs.values.flatten()
    pred_flat = pred.values.flatten()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs(obs_flat[mask] - pred_flat[mask]))


def pbias(obs, pred):
    obs_flat = obs.values.flatten()
    pred_flat = pred.values.flatten()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask]
    if obs_valid.size == 0 or np.sum(np.abs(obs_valid)) == 0:
        return np.nan
    return 100.0 * np.sum(pred_valid - obs_valid) / np.sum(np.abs(obs_valid))


def pearson_r(obs, pred):
    obs_flat = obs.values.flatten()
    pred_flat = pred.values.flatten()
    mask = ~(np.isnan(obs_flat) | np.isnan(pred_flat))
    obs_valid = obs_flat[mask]
    pred_valid = pred_flat[mask]
    if obs_valid.size < 2:
        return np.nan
    if np.std(obs_valid) == 0 or np.std(pred_valid) == 0:
        return np.nan
    return np.corrcoef(obs_valid, pred_valid)[0, 1]


def csi(obs, pred, threshold):
    obs_flat = obs.values.flatten()
    pred_flat = pred.values.flatten()
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
    denominator = hits + misses + false_alarms
    if denominator == 0:
        return np.nan
    return hits / denominator


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

    obs_frac = np.divide(
        obs_frac, valid_count, out=np.full(obs_frac.shape, np.nan), where=valid_count > 0
    )
    pred_frac = np.divide(
        pred_frac, valid_count, out=np.full(pred_frac.shape, np.nan), where=valid_count > 0
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
    n_times = truth_da.sizes.get("time", 1)
    values = []
    for t in range(n_times):
        obs_2d = truth_da.isel(time=t) if "time" in truth_da.dims else truth_da
        pred_2d = pred_da.isel(time=t) if "time" in pred_da.dims else pred_da
        score = fss(obs_2d, pred_2d, threshold, window_size)
        if np.isfinite(score):
            values.append(score)
    if not values:
        return np.nan
    return float(np.mean(values))


def apply_precip_threshold(da, threshold):
    if threshold is None:
        return da
    return da.where((da >= threshold) | da.isnull(), 0.0)


def looks_like_precip(var_name):
    if var_name in PRECIP_VAR_NAMES:
        return True
    upper_name = var_name.upper()
    return ("PREC" in upper_name) or ("RAIN" in upper_name) or ("APCP" in upper_name)


def make_domain_mask(root_ds):
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
    mask_coords = {dim: root_ds.coords[dim] for dim in mask_dims if dim in root_ds.coords}
    return xr.DataArray(keep, dims=mask_dims, coords=mask_coords, name=f"{mode}_mask")


def apply_domain_mask(da, spatial_mask):
    if spatial_mask is None:
        return da
    return da.where(spatial_mask)


#%% ─────────────────────────── HELPERS ───────────────────────────────
def compute_prediction_vs_target_scores(netcdf_file, basic_csv, precip_csv):
    print(f"Computing regression-vs-target scores from: {netcdf_file}")

    root_ds = xr.open_dataset(netcdf_file)
    pred_ds = xr.open_dataset(netcdf_file, group="prediction")
    truth_ds = xr.open_dataset(netcdf_file, group="truth")
    try:
        pred_ds = pred_ds.assign_coords(root_ds.coords)
        truth_ds = truth_ds.assign_coords(root_ds.coords)
        spatial_mask = make_domain_mask(root_ds)

        shared_vars = [v for v in truth_ds.data_vars if v in pred_ds.data_vars]
        if not shared_vars:
            raise ValueError("No overlapping variables found between prediction and truth groups.")

        basic_rows = []
        precip_rows = []

        for var in shared_vars:
            truth_var = truth_ds[var]
            pred_var = pred_ds[var]

            if "ensemble" in pred_var.dims:
                pred_var = pred_var.mean("ensemble")

            is_precip = looks_like_precip(var)
            if is_precip:
                truth_eval = apply_precip_threshold(truth_var, PRECIP_ZERO_THRESHOLD)
                pred_eval = apply_precip_threshold(pred_var, PRECIP_ZERO_THRESHOLD)
            else:
                truth_eval = truth_var
                pred_eval = pred_var

            truth_eval = apply_domain_mask(truth_eval, spatial_mask)
            pred_eval = apply_domain_mask(pred_eval, spatial_mask)

            basic_rows.append(
                {
                    "variable": var,
                    "RMSE": rmse(truth_eval, pred_eval),
                    "MAE": mae(truth_eval, pred_eval),
                    "PBIAS_%": pbias(truth_eval, pred_eval),
                    "Pearson_r": pearson_r(truth_eval, pred_eval),
                }
            )

            if is_precip:
                row = {"variable": var}
                for threshold in CSI_THRESHOLDS:
                    row[f"CSI_{threshold}"] = csi(truth_eval, pred_eval, threshold)
                for window in FSS_WINDOWS:
                    row[f"FSS_w{window}"] = compute_fss_over_time(
                        truth_eval, pred_eval, threshold=0.1, window_size=window
                    )
                precip_rows.append(row)

        basic_df = pd.DataFrame(basic_rows).sort_values("variable").reset_index(drop=True).round(4)
        basic_df.to_csv(basic_csv, index=False)
        print(f"Saved: {basic_csv.resolve()}")

        if precip_rows:
            precip_df = (
                pd.DataFrame(precip_rows).sort_values("variable").reset_index(drop=True).round(4)
            )
            precip_df.to_csv(precip_csv, index=False)
            print(f"Saved: {precip_csv.resolve()}")
        else:
            pd.DataFrame(columns=["variable"] + PRECIP_METRICS).to_csv(precip_csv, index=False)
            print(f"Saved empty precip table: {precip_csv.resolve()}")
    finally:
        pred_ds.close()
        truth_ds.close()
        root_ds.close()


def normalize_variable_column(df):
    if "variable" in df.columns:
        return df
    if "truth_variable" in df.columns:
        return df.rename(columns={"truth_variable": "variable"})
    raise ValueError("Expected a variable column ('variable' or 'truth_variable').")


def load_model_scores(basic_csv, precip_csv):
    basic_df = normalize_variable_column(pd.read_csv(basic_csv))
    keep_basic = ["variable"] + [m for m in BASIC_METRICS if m in basic_df.columns]
    basic_df = basic_df[keep_basic].copy()

    if precip_csv.exists():
        precip_df = normalize_variable_column(pd.read_csv(precip_csv))
        keep_precip = ["variable"] + [m for m in PRECIP_METRICS if m in precip_df.columns]
        precip_df = precip_df[keep_precip].copy()
    else:
        precip_df = pd.DataFrame(columns=["variable"] + PRECIP_METRICS)

    merged = basic_df.merge(precip_df, on="variable", how="left")
    return merged


def directional_skill(metric, baseline, candidate):
    if not np.isfinite(baseline) or not np.isfinite(candidate):
        return np.nan
    eps = 1e-12

    if metric in LOWER_BETTER:
        base = abs(baseline) if metric == "PBIAS_%" else baseline
        cand = abs(candidate) if metric == "PBIAS_%" else candidate
        # PBIAS can be near zero; use a 1%-point floor to keep skill stable.
        if metric == "PBIAS_%":
            denom = max(abs(base), 1.0)
        else:
            denom = max(abs(base), eps)
        return 100.0 * (base - cand) / denom

    if metric in HIGHER_BETTER:
        denom = max(1.0 - baseline, eps)
        return 100.0 * (candidate - baseline) / denom

    return np.nan


def best_model(metric, values):
    valid = {name: val for name, val in values.items() if np.isfinite(val)}
    if not valid:
        return "NA"

    if metric in LOWER_BETTER:
        if metric == "PBIAS_%":
            return min(valid, key=lambda name: abs(valid[name]))
        return min(valid, key=lambda name: valid[name])

    if metric in HIGHER_BETTER:
        return max(valid, key=lambda name: valid[name])

    return "NA"


def safe_get(indexed_df, variable, metric):
    if metric not in indexed_df.columns:
        return np.nan
    if variable not in indexed_df.index:
        return np.nan
    value = indexed_df.loc[variable, metric]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    return float(value) if pd.notna(value) else np.nan


def format_triplet_value(value):
    if not np.isfinite(value):
        return "NA"
    return f"{value:.{TRIPLET_DECIMALS}f}"


def format_triplet(input_value, regression_value, regdiff_value):
    return (
        f"{format_triplet_value(input_value)}/"
        f"{format_triplet_value(regression_value)}/"
        f"{format_triplet_value(regdiff_value)}"
    )


def normalize_metric_triplet(metric, values):
    out = {name: np.nan for name in values}
    valid = {name: float(val) for name, val in values.items() if np.isfinite(val)}
    if not valid:
        return out

    if metric in LOWER_BETTER:
        transformed = {
            name: (abs(val) if metric == "PBIAS_%" else val) for name, val in valid.items()
        }
        best = min(transformed.values())
        worst = max(transformed.values())
        if np.isclose(best, worst):
            for name in transformed:
                out[name] = 0.5
        else:
            denom = worst - best
            for name, val in transformed.items():
                out[name] = (worst - val) / denom
        return out

    if metric in HIGHER_BETTER:
        best = max(valid.values())
        worst = min(valid.values())
        if np.isclose(best, worst):
            for name in valid:
                out[name] = 0.5
        else:
            denom = best - worst
            for name, val in valid.items():
                out[name] = (val - worst) / denom
        return out

    return out


def build_variable_trajectory(long_df, metrics):
    df = long_df.copy()
    df["variable"] = df["variable"].astype(str)
    df["metric"] = df["metric"].astype(str)
    metrics = [metric for metric in metrics if metric in set(df["metric"])]

    rows = []
    for variable in sorted(df["variable"].unique()):
        var_df = df[(df["variable"] == variable) & (df["metric"].isin(metrics))]
        collected = {name: [] for name in TRAJECTORY_MODELS}

        for row in var_df.itertuples(index=False):
            scores = normalize_metric_triplet(
                row.metric,
                {
                    "Input": row.input,
                    "Regression": row.regression,
                    "Reg+Diffusion": row.regdiff,
                },
            )
            for model_name, score in scores.items():
                if np.isfinite(score):
                    collected[model_name].append(score)

        if all(len(collected[name]) == 0 for name in TRAJECTORY_MODELS):
            continue

        rows.append(
            {
                "variable": variable,
                "input_norm": np.mean(collected["Input"]) if collected["Input"] else np.nan,
                "regression_norm": (
                    np.mean(collected["Regression"]) if collected["Regression"] else np.nan
                ),
                "regdiff_norm": (
                    np.mean(collected["Reg+Diffusion"]) if collected["Reg+Diffusion"] else np.nan
                ),
                "metrics_used": len(metrics),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("variable").reset_index(drop=True)
    return out_df


def build_metric_trajectory(long_df, metrics):
    df = long_df.copy()
    df["variable"] = df["variable"].astype(str)
    df["metric"] = df["metric"].astype(str)
    metrics = [metric for metric in metrics if metric in set(df["metric"])]

    rows = []
    for metric in metrics:
        metric_df = df[df["metric"] == metric]
        for row in metric_df.itertuples(index=False):
            scores = normalize_metric_triplet(
                metric,
                {
                    "Input": row.input,
                    "Regression": row.regression,
                    "Reg+Diffusion": row.regdiff,
                },
            )
            rows.append(
                {
                    "variable": row.variable,
                    "metric": metric,
                    "input_norm": scores["Input"],
                    "regression_norm": scores["Regression"],
                    "regdiff_norm": scores["Reg+Diffusion"],
                }
            )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df

    out_df["metric"] = pd.Categorical(out_df["metric"], categories=metrics, ordered=True)
    out_df = out_df.sort_values(["metric", "variable"]).reset_index(drop=True)
    return out_df


def write_variable_trajectory_plot(trajectory_df):
    if trajectory_df.empty:
        print("Skipping variable trajectory plot: no rows.")
        return

    x = np.array([0, 1, 2], dtype=float)
    x_labels = ["Input", "Regression", "Regression + Diffusion"]

    fig, ax = plt.subplots(figsize=(12.5, 7.2))

    for row in trajectory_df.itertuples(index=False):
        y = np.array([row.input_norm, row.regression_norm, row.regdiff_norm], dtype=float)
        ax.plot(x, y, marker="o", linewidth=2, alpha=0.95, label=row.variable)

    mean_y = np.array(
        [
            trajectory_df["input_norm"].mean(skipna=True),
            trajectory_df["regression_norm"].mean(skipna=True),
            trajectory_df["regdiff_norm"].mean(skipna=True),
        ],
        dtype=float,
    )
    ax.plot(
        x,
        mean_y,
        marker="o",
        linewidth=3,
        linestyle="--",
        color="black",
        label="Mean (all variables)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(-0.1, 2.1)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Composite normalized performance score")
    ax.set_title("Stage-wise normalized performance trajectory by variable")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)

    fig.text(
        0.5,
        0.01,
        "Normalization is per variable and metric across the three stages (0=worst, 1=best). "
        f"Metrics used: {', '.join(TRAJECTORY_METRICS)}",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 0.82, 1.0])
    fig.savefig(OUTPUT_VARIABLE_TRAJECTORY_PLOT, dpi=180)
    plt.close(fig)
    print(f"Saved: {OUTPUT_VARIABLE_TRAJECTORY_PLOT.resolve()}")

    if OUTPUT_VARIABLE_TRAJECTORY_PLOT_LOGS.parent.exists():
        fig2, ax2 = plt.subplots(figsize=(12.5, 7.2))
        for row in trajectory_df.itertuples(index=False):
            y = np.array([row.input_norm, row.regression_norm, row.regdiff_norm], dtype=float)
            ax2.plot(x, y, marker="o", linewidth=2, alpha=0.95, label=row.variable)
        ax2.plot(
            x,
            mean_y,
            marker="o",
            linewidth=3,
            linestyle="--",
            color="black",
            label="Mean (all variables)",
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(x_labels)
        ax2.set_xlim(-0.1, 2.1)
        ax2.set_ylim(0.0, 1.02)
        ax2.set_ylabel("Composite normalized performance score")
        ax2.set_title("Stage-wise normalized performance trajectory by variable")
        ax2.grid(True, alpha=0.25)
        ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)
        fig2.text(
            0.5,
            0.01,
            "Normalization is per variable and metric across the three stages (0=worst, 1=best). "
            f"Metrics used: {', '.join(TRAJECTORY_METRICS)}",
            ha="center",
            fontsize=9,
        )
        fig2.tight_layout(rect=[0, 0.05, 0.82, 1.0])
        fig2.savefig(OUTPUT_VARIABLE_TRAJECTORY_PLOT_LOGS, dpi=180)
        plt.close(fig2)
        print(f"Saved: {OUTPUT_VARIABLE_TRAJECTORY_PLOT_LOGS.resolve()}")


def write_metric_separated_trajectory_plot(metric_df, metrics):
    if metric_df.empty:
        print("Skipping metric-separated trajectory plot: no rows.")
        return

    metric_names = [metric for metric in metrics if metric in set(metric_df["metric"].astype(str))]
    if not metric_names:
        print("Skipping metric-separated trajectory plot: no matching metrics.")
        return

    variable_names = sorted(metric_df["variable"].astype(str).unique())
    cmap_name = "tab10" if len(variable_names) <= 10 else "tab20"
    cmap = plt.get_cmap(cmap_name, len(variable_names))
    color_map = {name: cmap(idx) for idx, name in enumerate(variable_names)}

    x = np.array([0, 1, 2], dtype=float)
    x_labels = ["Input", "Regression", "Regression + Diffusion"]

    n_metrics = len(metric_names)
    n_cols = 2 if n_metrics > 1 else 1
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(8.8 * n_cols, 3.9 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        sub = metric_df[metric_df["metric"].astype(str) == metric]

        for var_name in variable_names:
            row = sub[sub["variable"].astype(str) == var_name]
            if row.empty:
                continue
            y = row[["input_norm", "regression_norm", "regdiff_norm"]].iloc[0].to_numpy(dtype=float)
            if np.all(np.isnan(y)):
                continue
            ax.plot(x, y, marker="o", linewidth=1.9, alpha=0.95, color=color_map[var_name])

        mean_y = sub[["input_norm", "regression_norm", "regdiff_norm"]].mean(skipna=True).to_numpy(
            dtype=float
        )
        ax.plot(
            x,
            mean_y,
            marker="o",
            linewidth=2.8,
            linestyle="--",
            color="black",
        )

        ax.set_title(metric_label(metric))
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlim(-0.1, 2.1)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)

    for idx in range(n_metrics, len(axes)):
        axes[idx].axis("off")

    for ax in axes[::n_cols]:
        ax.set_ylabel("Normalized score")

    legend_handles = [
        Line2D([0], [0], color=color_map[name], linewidth=2.0, marker="o", label=name)
        for name in variable_names
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.8,
            linestyle="--",
            marker="o",
            label="Mean (metric)",
        )
    )

    fig.suptitle("Stage-wise normalized trajectories by metric", fontsize=14, y=0.995)
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=True,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.01,
        "Normalization is done per variable+metric across the three stages (0=worst, 1=best).",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 0.84, 0.96])
    fig.savefig(OUTPUT_VARIABLE_TRAJECTORY_BY_METRIC_PLOT, dpi=180)
    print(f"Saved: {OUTPUT_VARIABLE_TRAJECTORY_BY_METRIC_PLOT.resolve()}")
    if OUTPUT_VARIABLE_TRAJECTORY_BY_METRIC_PLOT_LOGS.parent.exists():
        fig.savefig(OUTPUT_VARIABLE_TRAJECTORY_BY_METRIC_PLOT_LOGS, dpi=180)
        print(f"Saved: {OUTPUT_VARIABLE_TRAJECTORY_BY_METRIC_PLOT_LOGS.resolve()}")
    plt.close(fig)


def build_triplet_table(input_df, reg_df, regdiff_df, variables, metrics, drop_all_na_rows=False):
    rows = []
    for variable in variables:
        row = {"variable": variable}
        for metric in metrics:
            row[metric] = format_triplet(
                safe_get(input_df, variable, metric),
                safe_get(reg_df, variable, metric),
                safe_get(regdiff_df, variable, metric),
            )
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    keep_cols = ["variable"] + [metric for metric in metrics if metric in table.columns]
    table = table[keep_cols]

    if drop_all_na_rows and metrics:
        metric_cols = [metric for metric in metrics if metric in table.columns]
        if metric_cols:
            mask = ~(table[metric_cols] == "NA/NA/NA").all(axis=1)
            table = table[mask].reset_index(drop=True)

    return table


def metric_label(metric):
    if metric in LOWER_BETTER:
        return f"{metric} [lower better]"
    if metric in HIGHER_BETTER:
        return f"{metric} [higher better]"
    return metric


def write_plot(long_df):
    plot_cols = [
        "regression_vs_input_skill_pct",
        "regdiff_vs_input_skill_pct",
        "regdiff_vs_regression_skill_pct",
    ]
    col_labels = [
        "Regression vs Input",
        "Reg+Diff vs Input",
        "Reg+Diff vs Regression",
    ]

    # Plot only core/basic metrics; precipitation skill metrics stay in CSV tables.
    df = long_df[long_df["metric"].isin(BASIC_METRICS)].copy()
    metric_order = [m for m in BASIC_METRICS if m in df["metric"].unique()]
    if df.empty:
        raise ValueError("No basic metrics available to plot.")
    variable_order = [v for v in df["variable"].drop_duplicates().tolist()]

    df["metric"] = df["metric"].astype(str)
    df["variable"] = df["variable"].astype(str)
    df["metric"] = pd.Categorical(df["metric"], categories=metric_order, ordered=True)
    df["variable"] = pd.Categorical(df["variable"], categories=variable_order, ordered=True)
    df = df.sort_values(["variable", "metric"]).reset_index(drop=True)

    row_labels = [f"{r.variable} | {metric_label(r.metric)}" for r in df.itertuples(index=False)]
    values = df[plot_cols].to_numpy(dtype=float)
    means = np.nanmean(values, axis=0)
    clipped = np.clip(values, -100.0, 100.0)
    masked = np.ma.masked_invalid(clipped)

    metric_summary = (
        df.groupby("metric", observed=True)[plot_cols]
        .mean(numeric_only=True)
        .reindex(metric_order)
        .dropna(how="all")
    )
    metric_labels = [metric_label(m) for m in metric_summary.index.tolist()]

    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#ececec")

    fig_height = max(8.0, 0.28 * len(row_labels) + 2.0)
    fig, (ax_heat, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(17, fig_height),
        gridspec_kw={"width_ratios": [2.7, 1.3]},
    )

    image = ax_heat.imshow(masked, aspect="auto", cmap=cmap, vmin=-100, vmax=100)

    xlabels = [f"{label}\nmean={means[i]:+.1f}%" for i, label in enumerate(col_labels)]
    ax_heat.set_xticks(np.arange(len(col_labels)))
    ax_heat.set_xticklabels(xlabels, fontsize=10)
    ax_heat.set_yticks(np.arange(len(row_labels)))
    ax_heat.set_yticklabels(row_labels, fontsize=8)
    ax_heat.set_title("Per-variable Skill (%)")

    # Annotate values in each heatmap cell for readability.
    n_rows, n_cols = masked.shape
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                text_color = "black" if abs(value) < 60 else "white"
                ax_heat.text(
                    col_idx,
                    row_idx,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=text_color,
                )

    # Draw separators between variables.
    group_sizes = df.groupby("variable", observed=True).size().tolist()
    cumulative = 0
    for group_size in group_sizes[:-1]:
        cumulative += group_size
        ax_heat.axhline(cumulative - 0.5, color="black", linewidth=0.8, alpha=0.5)

    cbar = fig.colorbar(image, ax=ax_heat, pad=0.01, fraction=0.028)
    cbar.set_label("Skill (%)")

    # Summary panel: average improvement per metric.
    y = np.arange(len(metric_summary))
    bar_h = 0.35
    reg_vals = metric_summary["regression_vs_input_skill_pct"].to_numpy(dtype=float)
    regdiff_vals = metric_summary["regdiff_vs_input_skill_pct"].to_numpy(dtype=float)

    ax_bar.barh(y - bar_h / 2, reg_vals, height=bar_h, color="#4c78a8", label="Regression vs Input")
    ax_bar.barh(y + bar_h / 2, regdiff_vals, height=bar_h, color="#f58518", label="Reg+Diff vs Input")
    ax_bar.axvline(0.0, color="black", linewidth=1.0)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(metric_labels, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Mean skill (%)")
    ax_bar.set_title("Metric-level Mean Skill")
    ax_bar.legend(loc="lower right", fontsize=9, frameon=True)

    for y_idx, val in enumerate(reg_vals):
        if np.isfinite(val):
            ax_bar.text(
                val + (1.0 if val >= 0 else -1.0),
                y_idx - bar_h / 2,
                f"{val:+.1f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8,
            )
    for y_idx, val in enumerate(regdiff_vals):
        if np.isfinite(val):
            ax_bar.text(
                val + (1.0 if val >= 0 else -1.0),
                y_idx + bar_h / 2,
                f"{val:+.1f}",
                va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8,
            )

    winner_counts = df["best_model"].value_counts()
    wins_text = (
        f"Wins - Input: {winner_counts.get('Input', 0)}, "
        f"Regression: {winner_counts.get('Regression', 0)}, "
        f"Reg+Diffusion: {winner_counts.get('Reg+Diffusion', 0)}"
    )
    fig.suptitle(
        "Three-way Forecast Comparison (Input vs Regression vs Reg+Diffusion)\n"
        "Basic metrics only (RMSE, MAE, PBIAS, Pearson_r). "
        "Positive skill means improvement over baseline. " + wins_text,
        fontsize=13,
        y=0.995,
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=180)
    plt.close(fig)
    print(f"Saved plot: {OUTPUT_PLOT.resolve()}")


#%% ────────────────────────────── MAIN ───────────────────────────────
if not REGRESSION_BASIC_CSV.exists():
    compute_prediction_vs_target_scores(
        netcdf_file=REGRESSION_NETCDF,
        basic_csv=REGRESSION_BASIC_CSV,
        precip_csv=REGRESSION_PRECIP_CSV,
    )
elif not REGRESSION_PRECIP_CSV.exists():
    # Keep both regression tables in sync.
    compute_prediction_vs_target_scores(
        netcdf_file=REGRESSION_NETCDF,
        basic_csv=REGRESSION_BASIC_CSV,
        precip_csv=REGRESSION_PRECIP_CSV,
    )
else:
    print("Using existing regression-vs-target score files.")

input_df = load_model_scores(INPUT_BASIC_CSV, INPUT_PRECIP_CSV).set_index("variable")
reg_df = load_model_scores(REGRESSION_BASIC_CSV, REGRESSION_PRECIP_CSV).set_index("variable")
regdiff_df = load_model_scores(REGDIFF_BASIC_CSV, REGDIFF_PRECIP_CSV).set_index("variable")

ordered_metrics = BASIC_METRICS + PRECIP_METRICS
all_variables = sorted(set(input_df.index) | set(reg_df.index) | set(regdiff_df.index))

rows = []
for variable in all_variables:
    for metric in ordered_metrics:
        input_value = safe_get(input_df, variable, metric)
        reg_value = safe_get(reg_df, variable, metric)
        regdiff_value = safe_get(regdiff_df, variable, metric)

        if not (np.isfinite(input_value) or np.isfinite(reg_value) or np.isfinite(regdiff_value)):
            continue

        values = {
            "Input": input_value,
            "Regression": reg_value,
            "Reg+Diffusion": regdiff_value,
        }
        rows.append(
            {
                "variable": variable,
                "metric": metric,
                "input": input_value,
                "regression": reg_value,
                "regdiff": regdiff_value,
                "regression_vs_input_skill_pct": directional_skill(metric, input_value, reg_value),
                "regdiff_vs_input_skill_pct": directional_skill(metric, input_value, regdiff_value),
                "regdiff_vs_regression_skill_pct": directional_skill(metric, reg_value, regdiff_value),
                "best_model": best_model(metric, values),
            }
        )

long_df = pd.DataFrame(rows)
metric_order = [m for m in ordered_metrics if m in long_df["metric"].unique()]
variable_order = [v for v in all_variables if v in long_df["variable"].unique()]
long_df["metric"] = pd.Categorical(long_df["metric"], categories=metric_order, ordered=True)
long_df["variable"] = pd.Categorical(long_df["variable"], categories=variable_order, ordered=True)
long_df = long_df.sort_values(["variable", "metric"]).reset_index(drop=True)
long_df.to_csv(OUTPUT_LONG_CSV, index=False, float_format="%.6f")
print(f"Saved: {OUTPUT_LONG_CSV.resolve()}")

wide_df = long_df.pivot(index="variable", columns="metric", values=["input", "regression", "regdiff"])
wide_df.columns = [f"{model}__{metric}" for model, metric in wide_df.columns]
wide_df = wide_df.reset_index()
wide_df.to_csv(OUTPUT_WIDE_CSV, index=False, float_format="%.6f")
print(f"Saved: {OUTPUT_WIDE_CSV.resolve()}")

summary_rows = []
for metric, gdf in long_df.groupby("metric"):
    summary_rows.append(
        {
            "metric": metric,
            "mean_input": gdf["input"].mean(skipna=True),
            "mean_regression": gdf["regression"].mean(skipna=True),
            "mean_regdiff": gdf["regdiff"].mean(skipna=True),
            "mean_regression_vs_input_skill_pct": gdf["regression_vs_input_skill_pct"].mean(skipna=True),
            "mean_regdiff_vs_input_skill_pct": gdf["regdiff_vs_input_skill_pct"].mean(skipna=True),
            "mean_regdiff_vs_regression_skill_pct": gdf[
                "regdiff_vs_regression_skill_pct"
            ].mean(skipna=True),
        }
    )
summary_df = pd.DataFrame(summary_rows).sort_values("metric").reset_index(drop=True)
summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, float_format="%.6f")
print(f"Saved: {OUTPUT_SUMMARY_CSV.resolve()}")

# Triplet tables with one value per box: input/regression/reg+diffusion.
# 1) Basic metrics table
triplet_basic_df = build_triplet_table(
    input_df=input_df,
    reg_df=reg_df,
    regdiff_df=regdiff_df,
    variables=all_variables,
    metrics=BASIC_METRICS,
    drop_all_na_rows=False,
)
triplet_basic_df.to_csv(OUTPUT_TRIPLET_TABLE_CSV, index=False)
print(f"Saved: {OUTPUT_TRIPLET_TABLE_CSV.resolve()}")
OUTPUT_TRIPLET_TABLE_TXT.write_text(triplet_basic_df.to_string(index=False) + "\n", encoding="utf-8")
print(f"Saved: {OUTPUT_TRIPLET_TABLE_TXT.resolve()}")

if OUTPUT_TRIPLET_TABLE_LOGS_CSV.parent.exists():
    triplet_basic_df.to_csv(OUTPUT_TRIPLET_TABLE_LOGS_CSV, index=False)
    print(f"Saved: {OUTPUT_TRIPLET_TABLE_LOGS_CSV.resolve()}")

# 2) Separate precipitation metrics table (CSI/FSS only)
triplet_precip_df = build_triplet_table(
    input_df=input_df,
    reg_df=reg_df,
    regdiff_df=regdiff_df,
    variables=all_variables,
    metrics=PRECIP_METRICS,
    drop_all_na_rows=True,
)
triplet_precip_df.to_csv(OUTPUT_TRIPLET_PRECIP_TABLE_CSV, index=False)
print(f"Saved: {OUTPUT_TRIPLET_PRECIP_TABLE_CSV.resolve()}")
OUTPUT_TRIPLET_PRECIP_TABLE_TXT.write_text(
    triplet_precip_df.to_string(index=False) + "\n", encoding="utf-8"
)
print(f"Saved: {OUTPUT_TRIPLET_PRECIP_TABLE_TXT.resolve()}")

if OUTPUT_TRIPLET_PRECIP_TABLE_LOGS_CSV.parent.exists():
    triplet_precip_df.to_csv(OUTPUT_TRIPLET_PRECIP_TABLE_LOGS_CSV, index=False)
    print(f"Saved: {OUTPUT_TRIPLET_PRECIP_TABLE_LOGS_CSV.resolve()}")

# Variable-level normalized trajectory data + plot.
trajectory_df = build_variable_trajectory(long_df=long_df, metrics=TRAJECTORY_METRICS)
trajectory_df.to_csv(OUTPUT_VARIABLE_TRAJECTORY_CSV, index=False, float_format="%.6f")
print(f"Saved: {OUTPUT_VARIABLE_TRAJECTORY_CSV.resolve()}")
if OUTPUT_VARIABLE_TRAJECTORY_CSV_LOGS.parent.exists():
    trajectory_df.to_csv(OUTPUT_VARIABLE_TRAJECTORY_CSV_LOGS, index=False, float_format="%.6f")
    print(f"Saved: {OUTPUT_VARIABLE_TRAJECTORY_CSV_LOGS.resolve()}")
write_variable_trajectory_plot(trajectory_df)

# Metric-separated normalized trajectory data + plot.
metric_traj_df = build_metric_trajectory(long_df=long_df, metrics=TRAJECTORY_METRICS)
metric_traj_df.to_csv(OUTPUT_VARIABLE_TRAJECTORY_BY_METRIC_CSV, index=False, float_format="%.6f")
print(f"Saved: {OUTPUT_VARIABLE_TRAJECTORY_BY_METRIC_CSV.resolve()}")
if OUTPUT_VARIABLE_TRAJECTORY_BY_METRIC_CSV_LOGS.parent.exists():
    metric_traj_df.to_csv(
        OUTPUT_VARIABLE_TRAJECTORY_BY_METRIC_CSV_LOGS, index=False, float_format="%.6f"
    )
    print(f"Saved: {OUTPUT_VARIABLE_TRAJECTORY_BY_METRIC_CSV_LOGS.resolve()}")
write_metric_separated_trajectory_plot(metric_traj_df, metrics=TRAJECTORY_METRICS)

write_plot(long_df)

print("\nTop rows from combined table:")
print(long_df.head(12).to_string(index=False))

# %%
