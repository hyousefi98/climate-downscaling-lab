#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""
Train a global linear correction (y = a*x + b) on training data
and generate NetCDF predictions for a validation/generation split.

Example:
  python generate_linear_baseline_nc.py \\
    --train-config conf/config_training_aorc.yaml \\
    --gen-config conf/config_generate_aorc.yaml \\
    --output regression_aorc_12km_to_3km.nc \\
    --coeff-out linear_baseline_coeffs.json \\
    --max-images 500 --samples-per-image 2000 --max-samples 1000000
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
    output = io.get("output_filename", "linear_baseline.nc")
    start_index = int(data.get("start_index", 0))
    max_samples = int(data.get("max_samples", -1))
    indices = data.get("indices", None)
    return output, start_index, max_samples, indices


def _compute_linear_coeffs(
    dataset,
    mask: Optional[np.ndarray],
    *,
    max_images: int,
    samples_per_image: int,
    max_samples: int,
    seed: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int], Dict[str, Dict[str, float]]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)

    input_infos = list(dataset.input_channels())
    output_infos = list(dataset.output_channels())
    input_keys = [_channel_key(info) for info in input_infos]
    output_keys = [_channel_key(info) for info in output_infos]

    n_out = len(output_keys)
    input_index = {key: idx for idx, key in enumerate(input_keys)}
    output_to_input = [input_index.get(key, None) for key in output_keys]

    sums_x = np.zeros(n_out, dtype=np.float64)
    sums_y = np.zeros(n_out, dtype=np.float64)
    sums_x2 = np.zeros(n_out, dtype=np.float64)
    sums_xy = np.zeros(n_out, dtype=np.float64)
    sums_y2 = np.zeros(n_out, dtype=np.float64)
    counts = np.zeros(n_out, dtype=np.int64)

    coords_cache = None
    latlon_cache = None
    landmask_cache = None

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

        for out_idx in range(n_out):
            src_idx = output_to_input[out_idx]
            if src_idx is None:
                continue

            xj = x[:, src_idx]
            yj = y[:, out_idx]
            valid_j = np.isfinite(xj) & np.isfinite(yj)
            if not np.any(valid_j):
                continue

            xj = xj[valid_j]
            yj = yj[valid_j]

            if max_samples > 0:
                remaining = max_samples - counts[out_idx]
                if remaining <= 0:
                    continue
                if xj.shape[0] > remaining:
                    pick = rng.choice(xj.shape[0], size=remaining, replace=False)
                    xj = xj[pick]
                    yj = yj[pick]

            sums_x[out_idx] += float(np.sum(xj))
            sums_y[out_idx] += float(np.sum(yj))
            sums_x2[out_idx] += float(np.sum(xj * xj))
            sums_xy[out_idx] += float(np.sum(xj * yj))
            sums_y2[out_idx] += float(np.sum(yj * yj))
            counts[out_idx] += xj.shape[0]

        if max_samples > 0 and np.all(counts >= max_samples):
            break

    coeffs: Dict[str, Dict[str, float]] = {}
    count_map: Dict[str, int] = {}
    stats: Dict[str, Dict[str, float]] = {}
    for out_idx, key in enumerate(output_keys):
        n = counts[out_idx]
        count_map[key] = int(n)
        if n <= 0:
            coeffs[key] = {"a": 1.0, "b": 0.0}
            stats[key] = {
                "mean_x": float("nan"),
                "mean_y": float("nan"),
                "std_x": float("nan"),
                "std_y": float("nan"),
                "corr": float("nan"),
                "a": 1.0,
                "b": 0.0,
            }
            continue
        mean_x = sums_x[out_idx] / n
        mean_y = sums_y[out_idx] / n
        var_x = (sums_x2[out_idx] / n) - mean_x * mean_x
        cov_xy = (sums_xy[out_idx] / n) - mean_x * mean_y
        var_y = (sums_y2[out_idx] / n) - mean_y * mean_y
        if var_x <= 0.0 or not np.isfinite(var_x):
            a = 0.0
        else:
            a = cov_xy / var_x
        b = mean_y - a * mean_x
        coeffs[key] = {"a": float(a), "b": float(b)}
        std_x = float(np.sqrt(var_x)) if var_x > 0 else 0.0
        std_y = float(np.sqrt(var_y)) if var_y > 0 else 0.0
        if std_x > 0.0 and std_y > 0.0:
            corr = float(cov_xy / (std_x * std_y))
        else:
            corr = float("nan")
        stats[key] = {
            "mean_x": float(mean_x),
            "mean_y": float(mean_y),
            "std_x": float(std_x),
            "std_y": float(std_y),
            "corr": float(corr),
            "a": float(a),
            "b": float(b),
        }

    return coeffs, count_map, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Global linear correction baseline to NetCDF.")
    parser.add_argument("--train-config", default="configs/baselines/train_physical.yaml")
    parser.add_argument("--gen-config", default="configs/baselines/generate_physical.yaml")
    parser.add_argument("--output", default=None, help="Override output filename.")
    parser.add_argument("--coeff-out", default="linear_baseline_coeffs.json")

    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--samples-per-image", type=int, default=2000)
    parser.add_argument("--max-samples", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=42)

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

    print("Computing global linear correction coefficients...")
    coeffs, counts, stats = _compute_linear_coeffs(
        train_dataset,
        train_mask,
        max_images=args.max_images,
        samples_per_image=args.samples_per_image,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    coeff_out = Path(args.coeff_out)
    coeff_out.write_text(json.dumps({"coeffs": coeffs, "counts": counts, "stats": stats}, indent=2))
    print(f"Wrote coefficients to {coeff_out}")
    print("Linear baseline stats (physical units):")
    for key, stat in stats.items():
        n = counts.get(key, 0)
        print(
            f"  {key}: n={n} mean_x={stat['mean_x']:.3f} mean_y={stat['mean_y']:.3f} "
            f"std_x={stat['std_x']:.3f} std_y={stat['std_y']:.3f} "
            f"corr={stat['corr']:.4f} a={stat['a']:.4f} b={stat['b']:.4f}"
        )

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

    gen_dataset = rf._build_dataset(gen_cfg["dataset"])
    gen_mask = rf._get_mask(gen_dataset)
    times = rf._get_time_values(gen_dataset)

    if indices is None:
        end_index = len(gen_dataset) if max_samples <= 0 else min(len(gen_dataset), start_index + max_samples)
        indices = list(range(start_index, end_index))
    if not indices:
        raise RuntimeError("No indices selected for generation.")

    input_infos = list(gen_dataset.input_channels())
    output_infos = list(gen_dataset.output_channels())
    input_keys = [_channel_key(info) for info in input_infos]
    output_keys = [_channel_key(info) for info in output_infos]
    input_index = {key: idx for idx, key in enumerate(input_keys)}

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

            c_out, h_out, w_out = img_hr_out.shape
            pred_out = np.full((c_out, h_out, w_out), np.nan, dtype=np.float32)

            for out_idx, key in enumerate(output_keys):
                coeff = coeffs.get(key, {"a": 1.0, "b": 0.0})
                src_idx = input_index.get(key, None)
                if src_idx is None:
                    continue
                pred_out[out_idx] = coeff["a"] * img_lr_out[src_idx] + coeff["b"]

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
