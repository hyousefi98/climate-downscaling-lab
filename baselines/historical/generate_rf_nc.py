#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""
Generate RF predictions and save to NetCDF.

Example:
  python generate_rf_nc.py --config conf/config_generate_aorc.yaml --model rf_3km.joblib \\
    --add-latlon --add-landmask --add-climatology
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import netCDF4 as nc
import numpy as np

try:
    import joblib
except Exception as exc:
    raise RuntimeError("joblib is required. Install with: pip install joblib") from exc

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.corrdiff import NetCDFWriter

import RF_3km as rf
from rf_feature_state import restore_state, check_feature_count


def _load_config_with_generation(config_path: Path) -> dict:
    raw = rf._load_yaml(config_path)
    cfg = rf._load_config(config_path)
    cfg["generation"] = raw.get("generation", {})
    return cfg


def _get_generation_settings(cfg: dict) -> Tuple[str, int, int, Optional[Sequence[int]]]:
    gen = cfg.get("generation", {})
    io = gen.get("io", {})
    data = gen.get("data", {})
    output = io.get("output_filename", "rf_output.nc")
    start_index = int(data.get("start_index", 0))
    max_samples = int(data.get("max_samples", -1))
    indices = data.get("indices", None)
    return output, start_index, max_samples, indices


def _build_features(
    img_lr: np.ndarray,
    dataset,
    *,
    add_coords: bool,
    add_latlon: bool,
    add_landmask: bool,
    add_climatology: bool,
    include_lead_time: bool,
    lead_time: Optional[np.ndarray],
    lead_time_scale: float,
    time_value: Optional[Any],
    climatology: Optional[np.ndarray],
    mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    c_in, h, w = img_lr.shape
    x_base = img_lr.reshape(c_in, -1).transpose(1, 0)
    feature_blocks = []

    if add_coords:
        coords = rf._coord_features(h, w)
        coords_flat = coords.reshape(2, -1).transpose(1, 0)
        feature_blocks.append(coords_flat)

    if add_latlon:
        latlon = rf._prepare_latlon_features(dataset, (h, w))
        if latlon is not None:
            latlon_flat = latlon.reshape(2, -1).transpose(1, 0)
            feature_blocks.append(latlon_flat)

    if add_landmask:
        landmask = rf._prepare_landmask_feature(dataset, (h, w))
        if landmask is not None:
            feature_blocks.append(landmask.reshape(-1, 1))

    if include_lead_time:
        if lead_time is None:
            lead_val = 0.0
        else:
            lead_val = float(rf._maybe_to_numpy(lead_time)) * lead_time_scale
        lead_feat = np.full((x_base.shape[0], 1), lead_val, dtype=np.float32)
        feature_blocks.append(lead_feat)

    month = None
    doy = None
    if add_climatology:
        if time_value is None:
            raise RuntimeError("Time values are required for climatology features.")
        if climatology is None:
            raise RuntimeError("Climatology normals are missing.")
        month, doy = rf._time_to_fields(time_value)
        month_norm = (float(month) - 1.0) / 11.0
        doy_norm = (float(doy) - 1.0) / 365.0
        time_feat = np.array([month_norm, doy_norm], dtype=np.float32)
        time_block = np.repeat(time_feat.reshape(1, -1), x_base.shape[0], axis=0)
        feature_blocks.append(time_block)

    if mask is not None:
        mask_flat = mask.reshape(-1)
        x_base = x_base[mask_flat]
        feature_blocks = [block[mask_flat] for block in feature_blocks]
    else:
        mask_flat = None

    if add_climatology:
        daily_norm = climatology[doy]
        normal_block = np.repeat(daily_norm.reshape(1, -1), x_base.shape[0], axis=0)
        anomaly_block = x_base - normal_block
        feature_blocks.extend([normal_block, anomaly_block])

    if feature_blocks:
        x = np.concatenate([x_base] + feature_blocks, axis=1)
    else:
        x = x_base

    return x, mask_flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RF predictions to NetCDF.")
    parser.add_argument("--config", default="configs/baselines/generate_physical.yaml")
    parser.add_argument("--model", required=True, help="Path to saved RF model.")
    parser.add_argument("--output", default=None, help="Override output filename.")
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    parser.add_argument("--predict-chunk-size", type=int, default=50000)

    parser.add_argument("--add-coords", action="store_true")
    parser.add_argument("--add-latlon", action="store_true")
    parser.add_argument("--add-landmask", action="store_true")
    parser.add_argument("--add-climatology", action="store_true")
    parser.add_argument("--include-lead-time", action="store_true")
    parser.add_argument("--lead-time-scale", type=float, default=1.0)
    parser.add_argument("--use-physical", action="store_true")
    parser.add_argument("--allow-legacy-climatology-recompute", action="store_true",
                        help="Archive reproduction only: permit an old model to recompute climatology on generation data. This uses held-out inputs and is not a training-only workflow.")

    parser.add_argument("--climatology-max-images", type=int, default=200)
    parser.add_argument("--climatology-samples-per-image", type=int, default=2000)
    parser.add_argument("--climatology-seed", type=int, default=123)
    args = parser.parse_args()

    DistributedManager.initialize()

    config_path = rf._resolve_config_path(args.config)
    cfg = _load_config_with_generation(config_path)

    output_filename, start_index, max_samples, cfg_indices = _get_generation_settings(cfg)
    if args.output is not None:
        output_filename = args.output
    if args.start_index is not None:
        start_index = args.start_index
    if args.max_samples is not None:
        max_samples = args.max_samples
    if args.indices is not None and len(args.indices) > 0:
        indices = list(args.indices)
    elif cfg_indices is not None:
        indices = list(cfg_indices)
    else:
        indices = None

    dataset = rf._build_dataset(cfg["dataset"])
    mask = rf._get_mask(dataset)
    times = rf._get_time_values(dataset)

    model = joblib.load(args.model)
    has_state = getattr(model,'downscaling_feature_state_',None) is not None
    climatology = restore_state(model,args,dataset,cfg['dataset'],times)
    if args.add_climatology and climatology is None:
        if not args.allow_legacy_climatology_recompute:
            raise ValueError('This model lacks saved training climatology. Train with the packaged workflow, or explicitly opt into --allow-legacy-climatology-recompute for historical reproduction only.')
        if times is None:
            raise RuntimeError("Dataset time values are required for climatology features.")
        climatology = rf._compute_daily_normals(
            dataset,
            mask,
            max_images=args.climatology_max_images,
            samples_per_image=args.climatology_samples_per_image,
            seed=args.climatology_seed,
            use_physical=args.use_physical,
            time_values=times,
        )

    if indices is None:
        end_index = len(dataset) if max_samples <= 0 else min(len(dataset), start_index + max_samples)
        indices = list(range(start_index, end_index))
    if not indices:
        raise RuntimeError("No indices selected for generation.")

    lat = dataset.latitude()
    lon = dataset.longitude()
    h, w = dataset.image_shape()
    lat = lat[:h, :w]
    lon = lon[:h, :w]

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(output_path, "w") as f:
        f.setncattr('feature_state_policy','saved training feature state' if has_state else 'legacy explicit feature flags')
        if args.add_climatology and not has_state:
            f.setncattr('legacy_climatology_recomputed_on_generation_data',1)
        writer = NetCDFWriter(
            f,
            lat,
            lon,
            dataset.input_channels(),
            dataset.output_channels(),
            has_lead_time=False,
        )

        for t_index, idx in enumerate(indices):
            sample = dataset[idx]
            if not isinstance(sample, (tuple, list)) or len(sample) < 2:
                raise ValueError("Dataset sample must be a tuple (hr, lr, [lead_time]).")

            img_hr = rf._maybe_to_numpy(sample[0]).astype(np.float32)
            img_lr = rf._maybe_to_numpy(sample[1]).astype(np.float32)
            lead_time = sample[2] if len(sample) > 2 else None
            time_value = times[idx] if times is not None else None

            if args.use_physical:
                if hasattr(dataset, "denormalize_input"):
                    img_lr_feat = dataset.denormalize_input(img_lr)
                else:
                    img_lr_feat = img_lr
                if hasattr(dataset, "denormalize_output"):
                    img_hr_feat = dataset.denormalize_output(img_hr)
                else:
                    img_hr_feat = img_hr
            else:
                img_lr_feat = img_lr
                img_hr_feat = img_hr

            x, mask_flat = _build_features(
                img_lr_feat,
                dataset,
                add_coords=args.add_coords,
                add_latlon=args.add_latlon,
                add_landmask=args.add_landmask,
                add_climatology=args.add_climatology,
                include_lead_time=args.include_lead_time,
                lead_time=lead_time,
                lead_time_scale=args.lead_time_scale,
                time_value=time_value,
                climatology=climatology,
                mask=mask,
            )

            check_feature_count(model,x)
            y_pred = rf._predict_in_chunks(model, x, args.predict_chunk_size)
            c_out, h_out, w_out = img_hr.shape

            if mask_flat is not None:
                pred_full = np.full((h_out * w_out, c_out), np.nan, dtype=np.float32)
                pred_full[mask_flat] = y_pred.astype(np.float32)
            else:
                pred_full = y_pred.astype(np.float32)
            pred = pred_full.reshape(h_out, w_out, c_out).transpose(2, 0, 1)

            if args.use_physical:
                img_lr_out = img_lr_feat
                img_hr_out = img_hr_feat
                pred_out = pred
            else:
                img_lr_out = dataset.denormalize_input(img_lr) if hasattr(dataset, "denormalize_input") else img_lr
                img_hr_out = dataset.denormalize_output(img_hr) if hasattr(dataset, "denormalize_output") else img_hr
                pred_out = dataset.denormalize_output(pred) if hasattr(dataset, "denormalize_output") else pred

            if mask is not None:
                mask_2d = mask.astype(bool)
                img_lr_out = img_lr_out.copy()
                img_hr_out = img_hr_out.copy()
                pred_out = pred_out.copy()
                img_lr_out[:, ~mask_2d] = np.nan
                img_hr_out[:, ~mask_2d] = np.nan
                pred_out[:, ~mask_2d] = np.nan

            writer.write_time(t_index, time_value)
            for ch_idx, info in enumerate(dataset.output_channels()):
                channel_name = info.name + info.level
                writer.write_truth(channel_name, t_index, img_hr_out[ch_idx])
                writer.write_prediction(channel_name, t_index, 0, pred_out[ch_idx])

            for ch_idx, info in enumerate(dataset.input_channels()):
                channel_name = info.name + info.level
                writer.write_input(channel_name, t_index, img_lr_out[ch_idx])

    print(f"Saved NetCDF to {output_path}")


if __name__ == "__main__":
    main()
