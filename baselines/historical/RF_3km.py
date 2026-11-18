#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""
Random Forest downscaling using the CorrDiff 3km data pipeline.

This script lives in examples/weather and reuses the exact same dataset
processing as corrdiff_3km (scaling, clipping, masks, coastal buffer, cropping).

Example:
  python RF_3km.py --config conf/config_training_aorc.yaml \
    --n-estimators 200 --max-samples 200000 --samples-per-image 2000 \
    --model-out rf_3km.joblib --metrics-out rf_3km_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception as exc:
    raise RuntimeError("PyTorch is required to load datasets in this repo.") from exc

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from exc

try:
    import joblib
except Exception as exc:
    raise RuntimeError("joblib is required. Install with: pip install joblib") from exc

try:
    import yaml

    _HAS_YAML = True
except Exception:
    _HAS_YAML = False
    try:
        from omegaconf import OmegaConf
    except Exception:
        OmegaConf = None

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# Paths
THIS_DIR = Path(__file__).resolve().parent
LOCAL_DIR = THIS_DIR
SHARED_DIR = THIS_DIR.parent / "corrdiff_3km"

# Ensure local datasets/ is importable first.
if LOCAL_DIR.exists() and str(LOCAL_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_DIR))
# Optional fallback to shared corrdiff_3km
if SHARED_DIR.exists() and str(SHARED_DIR) not in sys.path:
    sys.path.append(str(SHARED_DIR))


def _find_repo_root(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        if (parent / "physicsnemo").is_dir():
            return parent
    return None


repo_root = _find_repo_root(THIS_DIR)
if repo_root and str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
# Resolve the packaged shared physical loader when running an entry point directly.
PACKAGE_ROOT = THIS_DIR.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from physicsnemo.distributed import DistributedManager  # noqa: E402
from datasets.dataset import known_datasets, register_dataset  # noqa: E402
from rf_feature_state import make_state, normalized_times, restore_state, check_feature_count  # noqa: E402


def _load_yaml(path: Path) -> Dict[str, Any]:
    if _HAS_YAML:
        with path.open("r") as f:
            return yaml.safe_load(f)
    if OmegaConf is None:
        raise RuntimeError("Neither PyYAML nor OmegaConf is available to read configs.")
    return OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    if base is None:
        base = {}
    if override is None:
        return dict(base)
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _find_dataset_name(cfg: Dict[str, Any]) -> str:
    explicit = cfg.get('dataset', {}).get('type')
    defaults = cfg.get("defaults", [])
    for item in defaults:
        if isinstance(item, dict) and "dataset" in item:
            base = item['dataset']
            if explicit and explicit != base and not (explicit=='aorc_legacy' and base=='aorc'):
                raise ValueError('dataset.type disagrees with the dataset defaults entry')
            return explicit or base
    return explicit or 'physical_pair'


def _resolve_config_path(config_arg: str) -> Path:
    p = Path(config_arg)
    if p.is_file():
        return p
    for base in (PACKAGE_ROOT, LOCAL_DIR, SHARED_DIR):
        if base.exists():
            p2 = base / config_arg
            if p2.is_file():
                return p2
    raise FileNotFoundError(f"Config not found: {config_arg}")


def _load_config(config_path: Path) -> Dict[str, Any]:
    cfg = _load_yaml(config_path)
    dataset_name = _find_dataset_name(cfg)
    supplied = cfg.get('dataset', {})
    required = {'input_data_path','target_data_path','stats_path'}
    if required <= set(supplied):
        cfg['dataset'] = dict(supplied)
        cfg['validation_dataset'] = _merge_dicts(supplied,cfg['validation']) if cfg.get('validation') is not None else cfg.get('validation_dataset')
        return cfg
    if dataset_name == 'physical_pair':
        raise ValueError(f"Self-contained physical_pair config is missing: {sorted(required-set(supplied))}")
    if dataset_name == 'aorc_legacy':
        dataset_name = 'aorc'

    base_dataset_path = (
        config_path.parent / "conf" / "base" / "dataset" / f"{dataset_name}.yaml"
    )
    if not base_dataset_path.exists():
        for base in (LOCAL_DIR, SHARED_DIR):
            if base.exists():
                candidate = base / "conf" / "base" / "dataset" / f"{dataset_name}.yaml"
                if candidate.exists():
                    base_dataset_path = candidate
                    break
    if not base_dataset_path.exists() and repo_root is not None:
        base_dataset_path = repo_root / "conf" / "base" / "dataset" / f"{dataset_name}.yaml"

    if not base_dataset_path.exists():
        raise FileNotFoundError(f"Base dataset config not found for {dataset_name}.")

    base_dataset_cfg = _load_yaml(base_dataset_path)
    merged_dataset_cfg = _merge_dicts(base_dataset_cfg, cfg.get("dataset", {}))

    cfg["dataset"] = merged_dataset_cfg
    if "validation" in cfg:
        cfg["validation_dataset"] = _merge_dicts(merged_dataset_cfg, cfg.get("validation", {}))
    else:
        cfg["validation_dataset"] = None
    return cfg


def _build_dataset(dataset_cfg: Dict[str, Any]):
    dataset_type = dataset_cfg.get("type", "physical_pair")
    if dataset_type == 'aorc':
        raise ValueError("Choose type: physical_pair for corrected physical stores, or explicitly type: aorc_legacy for archived packed-number stores")
    if dataset_type == 'aorc_legacy':
        import zarr
        for key in ('input_data_path','target_data_path'):
            profile=zarr.open_group(str(dataset_cfg[key]),mode='r').attrs.get('numeric_profile')
            if profile not in (None,'legacy_packed'):
                raise ValueError(f"Legacy AORC scaling cannot consume numeric_profile={profile!r}; use physical_pair for physical data")
        stats=json.loads(Path(dataset_cfg['stats_path']).read_text())
        if stats.get('metadata',{}).get('numeric_profile')=='physical':
            raise ValueError('Physical normalization statistics cannot be used by the legacy packed-number loader')
    register_dataset(dataset_type)
    dataset_init = known_datasets.get(dataset_type)
    if dataset_init is None:
        raise RuntimeError(f"Unknown dataset type: {dataset_type}")
    cfg = dict(dataset_cfg)
    cfg.pop("type", None)
    return dataset_init(**cfg)


def _get_mask(dataset) -> Optional[np.ndarray]:
    mask = None
    if hasattr(dataset, "get_loss_mask"):
        try:
            mask = dataset.get_loss_mask()
        except Exception:
            mask = None
    if mask is None and hasattr(dataset, "get_loss_mask_tensor"):
        try:
            mask_t = dataset.get_loss_mask_tensor()
            if mask_t is not None:
                mask = mask_t.squeeze().cpu().numpy()
        except Exception:
            mask = None
    if mask is None:
        return None
    return mask.astype(bool)


def _maybe_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _coord_features(h: int, w: int) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.stack([yy, xx], axis=0)


def _normalize_minmax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val - min_val < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_val) / (max_val - min_val) * 2.0 - 1.0


def _prepare_latlon_features(dataset, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not hasattr(dataset, "latitude") or not hasattr(dataset, "longitude"):
        return None
    try:
        lat = _maybe_to_numpy(dataset.latitude()).astype(np.float32)
        lon = _maybe_to_numpy(dataset.longitude()).astype(np.float32)
    except Exception:
        return None

    if lat.ndim == 1 and lon.ndim == 1:
        lat, lon = np.meshgrid(lat, lon, indexing="ij")
    elif lat.ndim != 2 or lon.ndim != 2:
        try:
            lat = np.broadcast_to(lat, shape)
            lon = np.broadcast_to(lon, shape)
        except Exception:
            return None

    h, w = shape
    lat = lat[:h, :w]
    lon = lon[:h, :w]
    lat = _normalize_minmax(lat)
    lon = _normalize_minmax(lon)
    return np.stack([lat, lon], axis=0)


def _prepare_landmask_feature(dataset, shape: Tuple[int, int]) -> Optional[np.ndarray]:
    mask = None
    if hasattr(dataset, "land_mask") and dataset.land_mask is not None:
        mask = dataset.land_mask
    elif hasattr(dataset, "get_loss_mask"):
        try:
            mask = dataset.get_loss_mask()
        except Exception:
            mask = None
    elif hasattr(dataset, "ocean_mask") and dataset.ocean_mask is not None:
        mask = ~dataset.ocean_mask

    if mask is None:
        return None
    mask = _maybe_to_numpy(mask).astype(bool)
    if mask.ndim != 2:
        return None
    h, w = shape
    mask = mask[:h, :w]
    return mask.astype(np.float32)


def _get_time_values(dataset) -> Optional[Sequence[Any]]:
    if not hasattr(dataset, "time"):
        return None
    try:
        times = dataset.time()
    except Exception:
        return None
    return times


def _time_to_fields(time_value: Any) -> Tuple[int, int]:
    if isinstance(time_value, datetime):
        return time_value.month, time_value.timetuple().tm_yday
    if isinstance(time_value, np.datetime64):
        dt = time_value.astype("datetime64[us]").astype(datetime)
        return dt.month, dt.timetuple().tm_yday
    if hasattr(time_value, "to_pydatetime"):
        dt = time_value.to_pydatetime()
        return dt.month, dt.timetuple().tm_yday
    if all(hasattr(time_value,key) for key in ('month','timetuple')):
        return time_value.month, time_value.timetuple().tm_yday
    raise ValueError("Unsupported time value for climatology features.")


def _compute_daily_normals(
    dataset,
    mask: Optional[np.ndarray],
    *,
    max_images: int,
    samples_per_image: int,
    seed: int,
    use_physical: bool,
    time_values: Sequence[Any],
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    sums = None
    counts = None
    global_sum = None
    global_count = 0

    for count, idx in enumerate(indices):
        if max_images > 0 and count >= max_images:
            break
        sample = dataset[idx]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise ValueError("Dataset sample must be a tuple (hr, lr, [lead_time]).")
        img_lr = _maybe_to_numpy(sample[1]).astype(np.float32)

        if use_physical and hasattr(dataset, "denormalize_input"):
            img_lr = dataset.denormalize_input(img_lr)

        c_in, h, w = img_lr.shape
        x = img_lr.reshape(c_in, -1).transpose(1, 0)

        if mask is not None:
            x = x[mask.reshape(-1)]

        if samples_per_image > 0 and samples_per_image < x.shape[0]:
            pick = rng.choice(x.shape[0], size=samples_per_image, replace=False)
            x = x[pick]

        month, doy = _time_to_fields(time_values[idx])

        if sums is None:
            sums = np.zeros((367, c_in), dtype=np.float64)
            counts = np.zeros(367, dtype=np.int64)
            global_sum = np.zeros(c_in, dtype=np.float64)

        sums[doy] += x.sum(axis=0)
        counts[doy] += x.shape[0]
        global_sum += x.sum(axis=0)
        global_count += x.shape[0]

    if sums is None or global_count == 0:
        raise RuntimeError("Failed to compute climatology normals (no samples).")

    global_mean = global_sum / float(global_count)
    normals = np.zeros_like(sums, dtype=np.float32)
    for doy in range(1, sums.shape[0]):
        if counts[doy] > 0:
            normals[doy] = (sums[doy] / counts[doy]).astype(np.float32)
        else:
            normals[doy] = global_mean.astype(np.float32)
    return normals


def _extract_pixels(
    dataset,
    sample,
    mask: Optional[np.ndarray],
    rng: np.random.Generator,
    samples_per_image: int,
    add_coords: bool,
    add_latlon: bool,
    add_landmask: bool,
    add_climatology: bool,
    include_lead_time: bool,
    lead_time_scale: float,
    use_physical: bool,
    coords_cache: Optional[np.ndarray],
    latlon_cache: Optional[np.ndarray],
    landmask_cache: Optional[np.ndarray],
    time_value: Optional[Any],
    climatology: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if isinstance(sample, (tuple, list)) and len(sample) >= 2:
        img_hr = sample[0]
        img_lr = sample[1]
        lead_time = sample[2] if len(sample) > 2 else None
    else:
        raise ValueError("Dataset sample must be a tuple (hr, lr, [lead_time]).")

    img_hr = _maybe_to_numpy(img_hr).astype(np.float32)
    img_lr = _maybe_to_numpy(img_lr).astype(np.float32)

    if use_physical:
        if hasattr(dataset, "denormalize_input"):
            img_lr = dataset.denormalize_input(img_lr)
        if hasattr(dataset, "denormalize_output"):
            img_hr = dataset.denormalize_output(img_hr)

    c_in, h, w = img_lr.shape
    c_out, h2, w2 = img_hr.shape
    if (h, w) != (h2, w2):
        raise ValueError("Input/Output shapes mismatch after processing.")

    x_base = img_lr.reshape(c_in, -1).transpose(1, 0)
    y = img_hr.reshape(c_out, -1).transpose(1, 0)

    feature_blocks = []
    coords = coords_cache
    if add_coords:
        if coords is None or coords.shape[1:] != (h, w):
            coords = _coord_features(h, w)
        coords_flat = coords.reshape(2, -1).transpose(1, 0)
        feature_blocks.append(coords_flat)

    latlon = latlon_cache
    if add_latlon:
        if latlon is None or latlon.shape[1:] != (h, w):
            latlon = _prepare_latlon_features(dataset, (h, w))
        if latlon is not None:
            latlon_flat = latlon.reshape(2, -1).transpose(1, 0)
            feature_blocks.append(latlon_flat)

    landmask = landmask_cache
    if add_landmask:
        if landmask is None or landmask.shape != (h, w):
            landmask = _prepare_landmask_feature(dataset, (h, w))
        if landmask is not None:
            land_flat = landmask.reshape(-1, 1)
            feature_blocks.append(land_flat)

    if include_lead_time:
        if lead_time is None:
            lead_val = 0.0
        else:
            lead_val = float(_maybe_to_numpy(lead_time)) * lead_time_scale
        lead_feat = np.full((x_base.shape[0], 1), lead_val, dtype=np.float32)
        feature_blocks.append(lead_feat)

    month = None
    doy = None
    if add_climatology:
        if time_value is None:
            raise RuntimeError("Time values are required for climatology features.")
        month, doy = _time_to_fields(time_value)
        month_norm = (float(month) - 1.0) / 11.0
        doy_norm = (float(doy) - 1.0) / 365.0
        time_feat = np.array([month_norm, doy_norm], dtype=np.float32)
        time_block = np.repeat(time_feat.reshape(1, -1), x_base.shape[0], axis=0)
        feature_blocks.append(time_block)

    if mask is not None:
        mask_flat = mask.reshape(-1)
        x_base = x_base[mask_flat]
        y = y[mask_flat]
        feature_blocks = [block[mask_flat] for block in feature_blocks]

    if samples_per_image > 0 and samples_per_image < x_base.shape[0]:
        idx = rng.choice(x_base.shape[0], size=samples_per_image, replace=False)
        x_base = x_base[idx]
        y = y[idx]
        feature_blocks = [block[idx] for block in feature_blocks]

    if add_climatology:
        if climatology is None:
            raise RuntimeError("Climatology normals are missing.")
        daily_norm = climatology[doy]
        normal_block = np.repeat(daily_norm.reshape(1, -1), x_base.shape[0], axis=0)
        anomaly_block = x_base - normal_block
        feature_blocks.extend([normal_block, anomaly_block])

    if feature_blocks:
        x = np.concatenate([x_base] + feature_blocks, axis=1)
    else:
        x = x_base

    return x, y, coords, latlon, landmask


def _collect_samples(
    dataset,
    mask: Optional[np.ndarray],
    *,
    max_images: int,
    samples_per_image: int,
    max_samples: int,
    seed: int,
    add_coords: bool,
    add_latlon: bool,
    add_landmask: bool,
    add_climatology: bool,
    climatology: Optional[np.ndarray],
    time_values: Optional[Sequence[Any]],
    include_lead_time: bool,
    lead_time_scale: float,
    use_physical: bool,
    shuffle: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    if shuffle:
        rng.shuffle(indices)

    xs = []
    ys = []
    total = 0
    coords_cache = None
    latlon_cache = None
    landmask_cache = None

    iterator = indices
    if _HAS_TQDM:
        iterator = tqdm(indices, total=len(indices), desc="Sampling images")

    for count, idx in enumerate(iterator):
        if max_images > 0 and count >= max_images:
            break
        sample = dataset[idx]
        time_value = None
        if add_climatology:
            if time_values is None:
                raise RuntimeError("Time values are required for climatology features.")
            time_value = time_values[idx]
        x, y, coords_cache, latlon_cache, landmask_cache = _extract_pixels(
            dataset=dataset,
            sample=sample,
            mask=mask,
            rng=rng,
            samples_per_image=samples_per_image,
            add_coords=add_coords,
            add_latlon=add_latlon,
            add_landmask=add_landmask,
            add_climatology=add_climatology,
            include_lead_time=include_lead_time,
            lead_time_scale=lead_time_scale,
            use_physical=use_physical,
            coords_cache=coords_cache,
            latlon_cache=latlon_cache,
            landmask_cache=landmask_cache,
            time_value=time_value,
            climatology=climatology,
        )

        if max_samples > 0 and total + x.shape[0] > max_samples:
            keep = max_samples - total
            if keep <= 0:
                break
            x = x[:keep]
            y = y[:keep]

        xs.append(x)
        ys.append(y)
        total += x.shape[0]

        if max_samples > 0 and total >= max_samples:
            break

    if not xs:
        raise RuntimeError("No samples collected. Check dataset or sampling settings.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _predict_in_chunks(model, x: np.ndarray, chunk_size: int) -> np.ndarray:
    if chunk_size <= 0 or x.shape[0] <= chunk_size:
        return model.predict(x)
    preds = []
    for start in range(0, x.shape[0], chunk_size):
        preds.append(model.predict(x[start : start + chunk_size]))
    return np.concatenate(preds, axis=0)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    diff = y_pred - y_true
    mse = np.mean(diff**2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff), axis=0)
    return {
        "rmse_per_channel": rmse.tolist(),
        "mae_per_channel": mae.tolist(),
        "rmse_mean": float(np.mean(rmse)),
        "mae_mean": float(np.mean(mae)),
    }


def _parse_max_features(value: Optional[str]) -> Any:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"none", "null"}:
        return None
    if text in {"auto"}:
        return 1.0
    if text in {"sqrt", "log2"}:
        return text
    try:
        if "." in text or "e" in text:
            return float(text)
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid max_features value: {value}. Use int, float, 'sqrt', 'log2', or 'none'."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Forest training for CorrDiff 3km data.")
    parser.add_argument(
        "--config",
        default="configs/baselines/train_physical.yaml",
        help="Hydra training config to reuse dataset processing.",
    )
    parser.add_argument("--model-out", default="rf_3km.joblib", help="Path to save RF model.")
    parser.add_argument("--metrics-out", default=None, help="Optional JSON metrics output path.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; only evaluate.")
    parser.add_argument("--load-model", default=None, help="Load a saved RF model.")

    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-features", type=_parse_max_features, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--oob-score", action="store_true")

    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--samples-per-image", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--shuffle", action="store_true", default=True)

    parser.add_argument("--eval-max-images", type=int, default=50)
    parser.add_argument("--eval-samples-per-image", type=int, default=2000)
    parser.add_argument("--eval-max-samples", type=int, default=50000)
    parser.add_argument("--predict-chunk-size", type=int, default=50000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--add-coords", action="store_true", help="Add normalized (y,x) coords as features.")
    parser.add_argument("--add-latlon", action="store_true", help="Add normalized lat/lon features.")
    parser.add_argument("--add-landmask", action="store_true", help="Add land/sea mask feature.")
    parser.add_argument(
        "--add-climatology",
        action="store_true",
        help="Add month/day-of-year plus daily normals and anomalies.",
    )
    parser.add_argument("--climatology-max-images", type=int, default=200)
    parser.add_argument("--climatology-samples-per-image", type=int, default=2000)
    parser.add_argument("--climatology-seed", type=int, default=123)
    parser.add_argument("--include-lead-time", action="store_true", help="Include lead-time feature if available.")
    parser.add_argument("--lead-time-scale", type=float, default=1.0)
    parser.add_argument("--use-physical", action="store_true", help="Train/eval in physical units.")
    parser.add_argument("--no-validation", action="store_true", help="Skip validation evaluation.")
    parser.add_argument("--allow-legacy-climatology-recompute", action="store_true",
                        help="For old eval-only models without saved feature state, explicitly allow refitting climatology from the configured training dataset.")

    args = parser.parse_args()

    DistributedManager.initialize()

    config_path = _resolve_config_path(args.config)
    cfg = _load_config(config_path)

    train_dataset = _build_dataset(cfg["dataset"])
    train_mask = _get_mask(train_dataset)
    train_time_values = _get_time_values(train_dataset)
    if not cfg['dataset'].get('train',True):
        raise ValueError('RF fitting requires a training dataset configuration')

    if not args.no_validation and cfg.get("validation_dataset"):
        valid_dataset = _build_dataset(cfg["validation_dataset"])
        valid_mask = _get_mask(valid_dataset)
        valid_time_values = _get_time_values(valid_dataset)
        if normalized_times(train_time_values).max()>=normalized_times(valid_time_values).min():
            raise ValueError('RF validation period overlaps the training period')
    else:
        valid_dataset = None
        valid_mask = None
        valid_time_values = None

    if args.eval_only and args.load_model is None:
        raise ValueError("--eval-only requires --load-model.")
    if args.eval_only and valid_dataset is None:
        raise ValueError('--eval-only requires a validation dataset')

    if args.load_model:
        print(f"Loading model from {args.load_model}")
        rf = joblib.load(args.load_model)
    else:
        rf = None

    climatology_normals = None
    restored = args.eval_only and getattr(rf,'downscaling_feature_state_',None) is not None
    if restored:
        climatology_normals = restore_state(rf,args,valid_dataset,cfg['validation_dataset'],valid_time_values)
    elif args.add_climatology:
        if args.eval_only and not args.allow_legacy_climatology_recompute:
            raise ValueError('Old model has no saved training climatology; explicit legacy opt-in is required')
        if train_time_values is None:
            raise RuntimeError("Dataset time values are required for climatology features.")
        print("Computing daily normals for climatology features...")
        climatology_normals = _compute_daily_normals(
            train_dataset,
            train_mask,
            max_images=args.climatology_max_images,
            samples_per_image=args.climatology_samples_per_image,
            seed=args.climatology_seed,
            use_physical=args.use_physical,
            time_values=train_time_values,
        )

    if not args.eval_only:
        print("Collecting training samples...")
        x_train, y_train = _collect_samples(
            train_dataset,
            train_mask,
            max_images=args.max_images,
            samples_per_image=args.samples_per_image,
            max_samples=args.max_samples,
            seed=args.seed,
            add_coords=args.add_coords,
            add_latlon=args.add_latlon,
            add_landmask=args.add_landmask,
            add_climatology=args.add_climatology,
            climatology=climatology_normals,
            time_values=train_time_values,
            include_lead_time=args.include_lead_time,
            lead_time_scale=args.lead_time_scale,
            use_physical=args.use_physical,
            shuffle=args.shuffle,
        )
        print(f"Training samples: {x_train.shape[0]} | features: {x_train.shape[1]} | outputs: {y_train.shape[1]}")

        rf = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
            n_jobs=args.n_jobs,
            random_state=args.seed,
            oob_score=args.oob_score,
        )
        print("Training RandomForest...")
        rf.fit(x_train, y_train)
        rf.downscaling_feature_state_ = make_state(
            args, train_dataset, cfg['dataset'], climatology_normals,
            train_time_values, x_train.shape[1])
        print("Training complete.")

        if args.model_out:
            joblib.dump(rf, args.model_out)
            print(f"Saved model to {args.model_out}")

    metrics = None
    if valid_dataset is not None:
        print("Collecting validation samples...")
        x_val, y_val = _collect_samples(
            valid_dataset,
            valid_mask,
            max_images=args.eval_max_images,
            samples_per_image=args.eval_samples_per_image,
            max_samples=args.eval_max_samples,
            seed=args.seed + 1,
            add_coords=args.add_coords,
            add_latlon=args.add_latlon,
            add_landmask=args.add_landmask,
            add_climatology=args.add_climatology,
            climatology=climatology_normals,
            time_values=valid_time_values,
            include_lead_time=args.include_lead_time,
            lead_time_scale=args.lead_time_scale,
            use_physical=args.use_physical,
            shuffle=False,
        )
        print(f"Validation samples: {x_val.shape[0]}")

        print("Predicting validation samples...")
        check_feature_count(rf,x_val)
        y_pred = _predict_in_chunks(rf, x_val, args.predict_chunk_size)
        metrics = _compute_metrics(y_val, y_pred)

        output_vars = getattr(valid_dataset, "output_variables", None)
        if output_vars:
            metrics["output_variables"] = list(output_vars)
        metrics["use_physical"] = bool(args.use_physical)
        print("Validation metrics:")
        print(json.dumps(metrics, indent=2))

    if args.metrics_out and metrics is not None:
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))
        print(f"Wrote metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()
