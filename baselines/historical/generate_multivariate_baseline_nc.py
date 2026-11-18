#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""
Train a multivariate linear regression baseline (all inputs -> all outputs)
and generate NetCDF predictions for a validation/generation split.

Example:
  python generate_multivariate_baseline_nc.py \
    --train-config conf/config_training_aorc.yaml \
    --gen-config conf/config_generate_aorc.yaml \
    --output multivariate_regression_aorc_12km_to_3km.nc \
    --coeff-out multivariate_baseline_coeffs.json \
    --max-images 500 --samples-per-image 2000 --max-samples 1000000 \
    --ridge 1e-6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import netCDF4 as nc
import numpy as np

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.corrdiff import NetCDFWriter

import RF_3km as rf
from rf_feature_state import fingerprint, normalized_times


def _channel_key(info) -> str:
    return f"{info.name}{info.level}"


def _load_config_with_generation(config_path: Path) -> Dict[str, Any]:
    raw = rf._load_yaml(config_path)
    cfg = rf._load_config(config_path)
    cfg["generation"] = raw.get("generation", {})
    return cfg


def _get_generation_settings(cfg: Dict[str, Any]) -> Tuple[str, int, int, Optional[Sequence[int]]]:
    gen = cfg.get("generation", {})
    io = gen.get("io", {})
    data = gen.get("data", {})
    output = io.get("output_filename", "multivariate_linear_baseline.nc")
    start_index = int(data.get("start_index", 0))
    max_samples = int(data.get("max_samples", -1))
    indices = data.get("indices", None)
    return output, start_index, max_samples, indices


def _compute_multivariate_coeffs(
    dataset,
    mask: Optional[np.ndarray],
    *,
    max_images: int,
    samples_per_image: int,
    max_samples: int,
    seed: int,
    ridge: float,
) -> Tuple[np.ndarray, int, Dict[str, float], Dict[str, float], list, list]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    input_infos = list(dataset.input_channels())
    output_infos = list(dataset.output_channels())
    input_keys = [_channel_key(info) for info in input_infos]
    output_keys = [_channel_key(info) for info in output_infos]

    p = len(input_keys)
    q = len(output_keys)

    xtx = np.zeros((p + 1, p + 1), dtype=np.float64)
    xty = np.zeros((p + 1, q), dtype=np.float64)

    sums_x = np.zeros(p, dtype=np.float64)
    sums_y = np.zeros(q, dtype=np.float64)
    sums_x2 = np.zeros(p, dtype=np.float64)
    sums_y2 = np.zeros(q, dtype=np.float64)

    coords_cache = None
    latlon_cache = None
    landmask_cache = None

    total = 0

    for count_img, idx in enumerate(indices):
        if max_images > 0 and count_img >= max_images:
            break

        sample = dataset[idx]
        x, y, coords_cache, latlon_cache, landmask_cache = rf._extract_pixels(
            dataset=dataset,
            sample=sample,
            mask=mask,
            rng=rng,
            samples_per_image=samples_per_image,
            add_coords=False,
            add_latlon=False,
            add_landmask=False,
            add_climatology=False,
            include_lead_time=False,
            lead_time_scale=1.0,
            use_physical=True,
            coords_cache=coords_cache,
            latlon_cache=latlon_cache,
            landmask_cache=landmask_cache,
            time_value=None,
            climatology=None,
        )

        if x.size == 0:
            continue

        valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        if not np.any(valid):
            continue

        x = x[valid]
        y = y[valid]

        if max_samples > 0:
            remaining = max_samples - total
            if remaining <= 0:
                break
            if x.shape[0] > remaining:
                pick = rng.choice(x.shape[0], size=remaining, replace=False)
                x = x[pick]
                y = y[pick]

        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)

        xtx += x_aug.T @ x_aug
        xty += x_aug.T @ y

        sums_x += x.sum(axis=0)
        sums_y += y.sum(axis=0)
        sums_x2 += (x * x).sum(axis=0)
        sums_y2 += (y * y).sum(axis=0)

        total += x.shape[0]
        if max_samples > 0 and total >= max_samples:
            break

    if total <= 0:
        raise RuntimeError("No samples collected for multivariate regression.")

    if ridge > 0.0:
        xtx[:p, :p] += ridge * np.eye(p)

    weights = np.linalg.solve(xtx, xty)

    mean_x = sums_x / total
    mean_y = sums_y / total
    std_x = np.sqrt(np.maximum(sums_x2 / total - mean_x * mean_x, 0.0))
    std_y = np.sqrt(np.maximum(sums_y2 / total - mean_y * mean_y, 0.0))

    stats = {
        "mean_x": mean_x.tolist(),
        "std_x": std_x.tolist(),
        "mean_y": mean_y.tolist(),
        "std_y": std_y.tolist(),
    }

    return weights, total, stats, input_keys, output_keys


def main() -> None:
    parser = argparse.ArgumentParser(description="Multivariate linear regression baseline to NetCDF.")
    parser.add_argument("--train-config", default="configs/baselines/train_physical.yaml")
    parser.add_argument("--gen-config", default="configs/baselines/generate_physical.yaml")
    parser.add_argument("--output", default=None, help="Override output filename.")
    parser.add_argument("--coeff-out", default="multivariate_baseline_coeffs.json")

    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--samples-per-image", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge", type=float, default=1e-6)

    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--max-gen-samples", type=int, default=None)
    parser.add_argument("--indices", nargs="*", type=int, default=None)
    args = parser.parse_args()

    DistributedManager.initialize()

    train_config_path = rf._resolve_config_path(args.train_config)
    gen_config_path = rf._resolve_config_path(args.gen_config)

    train_cfg = rf._load_config(train_config_path)
    gen_cfg = _load_config_with_generation(gen_config_path)

    train_dataset = rf._build_dataset(train_cfg["dataset"])
    train_mask = rf._get_mask(train_dataset)
    gen_dataset = rf._build_dataset(gen_cfg['dataset'])
    if not train_cfg['dataset'].get('train',True) or gen_cfg['dataset'].get('train',True):
        raise ValueError('Ridge requires a training configuration and a held-out generation configuration')
    if normalized_times(rf._get_time_values(train_dataset)).max()>=normalized_times(rf._get_time_values(gen_dataset)).min():
        raise ValueError('Ridge generation period must follow training')
    if fingerprint(train_dataset,train_cfg['dataset'])!=fingerprint(gen_dataset,gen_cfg['dataset']):
        raise ValueError('Ridge training and generation variable order, grid, mask policy or normalization differ')

    print("Computing multivariate linear regression coefficients...")
    weights, total, stats, input_keys, output_keys = _compute_multivariate_coeffs(
        train_dataset,
        train_mask,
        max_images=args.max_images,
        samples_per_image=args.samples_per_image,
        max_samples=args.max_samples,
        seed=args.seed,
        ridge=args.ridge,
    )

    coeffs_by_output = {}
    p = len(input_keys)
    for j, out_key in enumerate(output_keys):
        coeffs_by_output[out_key] = {
            "bias": float(weights[p, j]),
            "weights": {input_keys[i]: float(weights[i, j]) for i in range(p)},
        }

    coeff_payload = {
        "mode": "multivariate",
        "samples": int(total),
        "ridge": float(args.ridge),
        "input_keys": input_keys,
        "output_keys": output_keys,
        "weights_matrix": weights.tolist(),
        "stats": stats,
        "coeffs_by_output": coeffs_by_output,
    }

    coeff_out = Path(args.coeff_out)
    coeff_out.write_text(json.dumps(coeff_payload, indent=2))
    print(f"Wrote coefficients to {coeff_out}")

    print("Multivariate regression summary:")
    print(f"  samples={total} inputs={len(input_keys)} outputs={len(output_keys)} ridge={args.ridge}")
    for j, out_key in enumerate(output_keys):
        w = np.asarray(weights[:p, j])
        top_idx = np.argsort(np.abs(w))[-3:][::-1]
        top_str = ", ".join(f"{input_keys[i]}={w[i]:.4f}" for i in top_idx)
        print(f"  {out_key}: bias={weights[p, j]:.4f} top_weights: {top_str}")

    output_filename, start_index, max_samples, cfg_indices = _get_generation_settings(gen_cfg)
    if args.output is not None:
        output_filename = args.output
    if args.start_index is not None:
        start_index = args.start_index
    if args.max_gen_samples is not None:
        max_samples = args.max_gen_samples
    if args.indices is not None and len(args.indices) > 0:
        indices = list(args.indices)
    elif cfg_indices is not None:
        indices = list(cfg_indices)
    else:
        indices = None

    gen_mask = rf._get_mask(gen_dataset)
    times = rf._get_time_values(gen_dataset)

    if indices is None:
        end_index = len(gen_dataset) if max_samples <= 0 else min(len(gen_dataset), start_index + max_samples)
        indices = list(range(start_index, end_index))
    if not indices:
        raise RuntimeError("No indices selected for generation.")

    lat = gen_dataset.latitude()
    lon = gen_dataset.longitude()
    h, w = gen_dataset.image_shape()
    lat = lat[:h, :w]
    lon = lon[:h, :w]

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with nc.Dataset(output_path, "w") as f:
        writer = NetCDFWriter(
            f,
            lat,
            lon,
            gen_dataset.input_channels(),
            gen_dataset.output_channels(),
            has_lead_time=False,
        )

        for t_index, idx in enumerate(indices):
            sample = gen_dataset[idx]
            if not isinstance(sample, (tuple, list)) or len(sample) < 2:
                raise ValueError("Dataset sample must be a tuple (hr, lr, [lead_time]).")

            img_hr = rf._maybe_to_numpy(sample[0]).astype(np.float32)
            img_lr = rf._maybe_to_numpy(sample[1]).astype(np.float32)
            time_value = times[idx] if times is not None else None

            img_lr_out = gen_dataset.denormalize_input(img_lr) if hasattr(gen_dataset, "denormalize_input") else img_lr
            img_hr_out = gen_dataset.denormalize_output(img_hr) if hasattr(gen_dataset, "denormalize_output") else img_hr

            c_in, h_out, w_out = img_lr_out.shape
            c_out = img_hr_out.shape[0]

            x_flat = img_lr_out.reshape(c_in, -1).transpose(1, 0).astype(np.float64)
            pred_flat = x_flat @ weights[:c_in, :] + weights[c_in, :]
            pred_out = pred_flat.reshape(h_out, w_out, c_out).transpose(2, 0, 1).astype(np.float32)

            if gen_mask is not None:
                mask_2d = gen_mask.astype(bool)
                img_lr_out = img_lr_out.copy()
                img_hr_out = img_hr_out.copy()
                pred_out = pred_out.copy()
                img_lr_out[:, ~mask_2d] = np.nan
                img_hr_out[:, ~mask_2d] = np.nan
                pred_out[:, ~mask_2d] = np.nan

            writer.write_time(t_index, time_value)
            for ch_idx, info in enumerate(gen_dataset.output_channels()):
                channel_name = _channel_key(info)
                writer.write_truth(channel_name, t_index, img_hr_out[ch_idx])
                writer.write_prediction(channel_name, t_index, 0, pred_out[ch_idx])

            for ch_idx, info in enumerate(gen_dataset.input_channels()):
                channel_name = _channel_key(info)
                writer.write_input(channel_name, t_index, img_lr_out[ch_idx])

    print(f"Saved NetCDF to {output_path}")


if __name__ == "__main__":
    main()
