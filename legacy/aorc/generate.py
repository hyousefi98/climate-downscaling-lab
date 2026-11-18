# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import random # Added for random sampling
import threading
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path
import torch
import torch._dynamo
from torch.distributed import gather
import numpy as np
import nvtx
import netCDF4 as nc
import xarray as xr

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.patching import GridPatching2D
from physicsnemo import Module
from physicsnemo.utils.diffusion import deterministic_sampler, stochastic_sampler
from physicsnemo.utils.corrdiff import (
    NetCDFWriter,
    get_time_from_range,
    regression_step,
    diffusion_step,
)

from models.helpers.generate_aorc import (
    get_dataset_and_sampler,
    save_images,
)
from models.helpers.train_helpers import set_patch_shape
from models.datasets.factory import register_dataset
from models.losses_aorc import apply_precip_threshold_normalized

def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}


def _get_debug_settings() -> dict:
    return {
        "enabled": _env_bool("CORRDIFF_GEN_DEBUG", "0"),
        "out_dir": os.getenv("CORRDIFF_GEN_DEBUG_DIR", "debug_plots_generate"),
        "num_samples": int(os.getenv("CORRDIFF_GEN_DEBUG_SAMPLES", "2")),
        "once": _env_bool("CORRDIFF_GEN_DEBUG_ONCE", "1"),
        "seed": int(os.getenv("CORRDIFF_GEN_DEBUG_SEED", "0")),
        "index": os.getenv("CORRDIFF_GEN_DEBUG_INDEX", "").strip(),
        "time": os.getenv("CORRDIFF_GEN_DEBUG_TIME", "").strip(),
        "sigma": os.getenv("CORRDIFF_GEN_DEBUG_SIGMA", "").strip(),
        "noise_seed": os.getenv("CORRDIFF_GEN_DEBUG_NOISE_SEED", "").strip(),
    }


def _maybe_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _vmin_vmax(arr: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    if symmetric:
        vmax = np.percentile(np.abs(finite), 98)
        if vmax == 0:
            vmax = 1.0
        return -float(vmax), float(vmax)
    vmin = np.percentile(finite, 2)
    vmax = np.percentile(finite, 98)
    if vmin == vmax:
        vmin = float(finite.min())
        vmax = float(finite.max())
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
    return float(vmin), float(vmax)


def _plot_debug_grid(
    *,
    out_path: Path,
    rows: list[str],
    cols: list[str],
    data_by_col: dict[str, np.ndarray],
    cmap_by_col: dict[str, str],
    symmetric_cols: set[str] | None = None,
    shared_cols: set[str] | None = None,
):
    plt = _maybe_import_matplotlib()
    if plt is None:
        return False

    symmetric_cols = symmetric_cols or set()
    shared_cols = shared_cols or set()
    n_rows = len(rows)
    n_cols = len(cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 1.8 * n_rows),
        squeeze=False,
    )
    shared_limits = {}
    if shared_cols:
        for r in range(n_rows):
            vals = []
            for col_name in cols:
                if col_name not in shared_cols:
                    continue
                arr = data_by_col[col_name][r]
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    vals.append(finite)
            if vals:
                merged = np.concatenate(vals)
                vmin, vmax = _vmin_vmax(
                    merged,
                    symmetric=any(col in symmetric_cols for col in shared_cols),
                )
                shared_limits[r] = (vmin, vmax)
    for r in range(n_rows):
        for c, col_name in enumerate(cols):
            ax = axes[r, c]
            arr = data_by_col[col_name][r]
            if col_name in shared_cols and r in shared_limits:
                vmin, vmax = shared_limits[r]
            else:
                vmin, vmax = _vmin_vmax(arr, symmetric=col_name in symmetric_cols)
            ax.imshow(arr, cmap=cmap_by_col.get(col_name, "viridis"), vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_name, fontsize=9)
            if c == 0:
                ax.set_ylabel(rows[r], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _tensor_to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def _apply_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x
    if mask.dtype != x.dtype:
        mask = mask.to(dtype=x.dtype)
    return x * mask


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except Exception:
        return None

def load_land_mask(path, img_shape, device):
    """
    Helper to load the land mask for loss weighting.
    Returns a tensor of shape (1, 1, H, W) on the specified device.
    """
    try:
        if not path or not os.path.exists(path):
            return None

        # Load using xarray
        ds = xr.open_dataset(path)
        mask_np = ds['LANDMASK'].values.astype(np.float32)

        # Crop to the image shape used in training (y, x)
        H, W = img_shape
        mask_np = mask_np[:H, :W]

        # Convert to tensor and move to device
        mask_t = torch.from_numpy(mask_np).to(device)

        # Expand dims to (1, 1, H, W) for broadcasting
        mask_t = mask_t.unsqueeze(0).unsqueeze(0)
        return mask_t

    except Exception as e:
        print(f"Warning: Could not load land mask from {path}: {e}")
        return None


def load_residual_std(cfg, dataset, device):
    residual_std_path = None
    if "generation" in cfg and cfg.generation.get("residual_std_path") is not None:
        residual_std_path = cfg.generation.residual_std_path
    if residual_std_path is None:
        return None
    try:
        with open(residual_std_path, "r") as f:
            residual_stats = json.load(f)
        stats_map = residual_stats.get("output", residual_stats)
        # Use the full variable names from the dataset, which match the keys in the JSON file
        residual_std = [stats_map[var_name] for var_name in dataset.output_variables]
        return torch.tensor(residual_std, dtype=torch.float32, device=device)
    except Exception as e:
        print(f"Warning: Could not load residual std from {residual_std_path}: {e}")
        return None


@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # Initialize logger
    logger = PythonLogger("generate")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate.log")

    debug_cfg = _get_debug_settings()
    debug_enabled = debug_cfg["enabled"]
    debug_done = False
    debug_out_dir = Path(debug_cfg["out_dir"])
    if debug_enabled and dist.rank == 0:
        logger0.info(
            f"Generation debug enabled: samples={debug_cfg['num_samples']} out_dir={debug_out_dir}"
        )
    debug_sigma = _parse_float(debug_cfg.get("sigma", ""))
    debug_noise_seed = _parse_int(debug_cfg.get("noise_seed", ""))

    # Handle the batch size
    seeds = list(np.arange(cfg.generation.num_ensembles))
    if len(seeds) == 0:
        raise ValueError("generation.num_ensembles must be >= 1")
    seed_is_real = [1] * len(seeds)
    num_batches = (
        (len(seeds) - 1) // (cfg.generation.seed_batch_size * dist.world_size) + 1
    ) * dist.world_size
    total_needed = num_batches * cfg.generation.seed_batch_size
    if len(seeds) < total_needed:
        pad_count = total_needed - len(seeds)
        pad_seed = seeds[-1]
        seeds += [pad_seed] * pad_count
        seed_is_real += [0] * pad_count

    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    all_real_batches = torch.as_tensor(seed_is_real).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]
    rank_real_batches = all_real_batches[dist.rank :: dist.world_size]
    rank_real_mask = (
        torch.cat(rank_real_batches) if len(rank_real_batches) > 0 else torch.zeros(0, dtype=torch.int64)
    )

    # Synchronize
    if dist.world_size > 1:
        torch.distributed.barrier()

    # --- TIME SELECTION LOGIC ---
    use_random_samples = False
    times = None
    
    force_debug_single = bool(debug_cfg.get("index") or debug_cfg.get("time"))

    # Check if random sampling is requested
    if force_debug_single:
        # Ignore configured times when forcing a specific debug sample.
        times = None
    elif (
        cfg.generation.get("random_samples") is not None
        and cfg.generation.random_samples > 0
    ):
        use_random_samples = True
        logger0.info(f"Random Sampling Enabled: Selecting {cfg.generation.random_samples} random samples from validation set.")
        # We pass None for times initially to load the full dataset
        times = None
    else:
        # Fallback to explicit times or range
        if cfg.generation.times_range and cfg.generation.times:
            raise ValueError("Either times_range or times must be provided, but not both")
        if cfg.generation.times_range:
            times = get_time_from_range(cfg.generation.times_range)
        else:
            times = cfg.generation.times

    # Create dataset object
    dataset_cfg = OmegaConf.to_container(cfg.dataset)

    # Register dataset (if custom dataset)
    register_dataset(cfg.dataset.type)
    logger0.info(f"Using dataset: {cfg.dataset.type}")

    if "has_lead_time" in cfg.generation:
        has_lead_time = cfg.generation["has_lead_time"]
    else:
        has_lead_time = False
    
    # Load dataset
    # If use_random_samples is True, times is None, so it loads ALL valid times.
    dataset, sampler = get_dataset_and_sampler(
        dataset_cfg=dataset_cfg, times=times, has_lead_time=has_lead_time # `times` can be None here
    )

    # --- APPLY RANDOM FILTERING ---
    if use_random_samples:
        total_samples = len(dataset)
        num_to_pick = cfg.generation.random_samples
        
        if num_to_pick > total_samples:
            logger0.warning(f"Requested {num_to_pick} samples but dataset only has {total_samples}. Using all.")
            num_to_pick = total_samples

        # Randomly select indices
        # We use a fixed seed for reproducibility across ranks if needed, 
        # or different if we want variety. Usually consistent is better for debugging.
        rng = np.random.RandomState(42) 
        selected_indices = np.sort(rng.choice(total_samples, size=num_to_pick, replace=False))
        
        logger0.info(f"Randomly selected {num_to_pick} indices: {selected_indices}")
        
        # Modify dataset in-place to shrink it to these samples
        # This relies on AORCDownscalingDataset using _time_indices and _time_values
        if hasattr(dataset, '_time_indices') and hasattr(dataset, '_time_values'):
            dataset._time_indices = dataset._time_indices[selected_indices]
            dataset._time_values = dataset._time_values[selected_indices]
            
            # Print the selected times for verification
            selected_times = dataset.time()
            logger0.info(f"Selected Times: {[t.strftime('%Y-%m-%dT%H:%M:%S') for t in selected_times]}")
        else:
            raise RuntimeError("Dataset does not support in-place random filtering (missing _time_indices).")

        # After modifying the dataset, the sampler must be updated to reflect the new size.
        # The new sampler should just be the indices of the shrunken dataset.
        sampler = range(len(dataset))

    # --- OVERRIDE WITH SINGLE DEBUG SAMPLE (INDEX OR TIME) ---
    if debug_cfg.get("index") or debug_cfg.get("time"):
        debug_index = None
        if debug_cfg.get("index"):
            try:
                debug_index = int(debug_cfg["index"])
            except Exception:
                debug_index = None
        if debug_index is None and debug_cfg.get("time"):
            try:
                all_times = dataset.time()
                for i, t in enumerate(all_times):
                    if t.strftime("%Y-%m-%dT%H:%M:%S") == debug_cfg["time"]:
                        debug_index = i
                        break
            except Exception:
                debug_index = None
        if debug_index is not None:
            if debug_index < 0 or debug_index >= len(dataset):
                logger0.warning(
                    f"Debug index {debug_index} out of range for dataset (0..{len(dataset)-1})."
                )
            elif hasattr(dataset, "_time_indices") and hasattr(dataset, "_time_values"):
                dataset._time_indices = dataset._time_indices[[debug_index]]
                dataset._time_values = dataset._time_values[[debug_index]]
                sampler = range(len(dataset))
                try:
                    t = dataset.time()[0]
                    logger0.info(
                        f"Using debug sample index {debug_index} time={t.strftime('%Y-%m-%dT%H:%M:%S')}"
                    )
                except Exception:
                    logger0.info(f"Using debug sample index {debug_index}")
        else:
            logger0.warning("Debug sample index/time requested but not found in dataset.")

    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())

    # Residual std for per-channel residual normalization (optional)
    residual_std_tensor = load_residual_std(cfg, dataset, device)

    # Precipitation threshold info for normalized-space thresholding (optional)
    precip_threshold_info = None
    if hasattr(dataset, 'get_precip_threshold_info'):
        precip_threshold_info = dataset.get_precip_threshold_info()
        if precip_threshold_info is not None:
            logger0.info(f"Using precip threshold in normalized space: {precip_threshold_info}")

    # --- LOAD MASKS FOR OUTPUT ---
    # We use the expanded ocean mask (ocean + coastal buffer) for NaN masking in output
    # This ensures coastal areas with bad data are marked as NaN in the output
    ocean_mask_with_buffer_cpu = None
    land_mask_tensor = None
    land_mask_cpu = None

    # Try to get the expanded mask from dataset first
    if hasattr(dataset, 'get_ocean_mask_with_buffer') and dataset.get_ocean_mask_with_buffer() is not None:
        ocean_mask_np = dataset.get_ocean_mask_with_buffer()
        ocean_mask_with_buffer_cpu = torch.from_numpy(ocean_mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        buffer_pixels = cfg.dataset.get("coastal_buffer_pixels", 0)
        logger0.info(f"Using ocean mask with coastal buffer ({buffer_pixels} pixels) for output masking")

        # Also create land mask for GPU operations (zeroing during inference)
        loss_mask_np = dataset.get_loss_mask()
        if loss_mask_np is not None:
            land_mask_tensor = torch.from_numpy(loss_mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            land_mask_cpu = land_mask_tensor.cpu()
    elif cfg.dataset.get("landmask_path"):
        # Fallback to loading land mask directly
        logger0.info(f"Loading Land Mask from: {cfg.dataset.landmask_path}")
        land_mask_tensor = load_land_mask(cfg.dataset.landmask_path, img_shape, device)
        if land_mask_tensor is not None:
            land_mask_cpu = land_mask_tensor.cpu()
            # Create ocean mask from land mask (invert it)
            ocean_mask_with_buffer_cpu = 1.0 - land_mask_cpu
            logger0.info("Land mask loaded successfully (no coastal buffer).")

    # Parse the patch shape
    if cfg.generation.patching:
        patch_shape_x = cfg.generation.patch_shape_x
        patch_shape_y = cfg.generation.patch_shape_y
    else:
        patch_shape_x, patch_shape_y = None, None
    patch_shape = (patch_shape_y, patch_shape_x)
    use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    if use_patching:
        patching = GridPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            boundary_pix=cfg.generation.boundary_pix,
            overlap_pix=cfg.generation.overlap_pix,
        )
        logger0.info("Patch-based training enabled")
    else:
        patching = None
        logger0.info("Patch-based training disabled")

    # Parse the inference mode
    if cfg.generation.inference_mode == "regression":
        load_net_reg, load_net_res = True, False
    elif cfg.generation.inference_mode == "diffusion":
        load_net_reg, load_net_res = False, True
    elif cfg.generation.inference_mode == "all":
        load_net_reg, load_net_res = True, True
    else:
        raise ValueError(f"Invalid inference mode {cfg.generation.inference_mode}")

    # Load diffusion network, move to device, change precision
    if load_net_res:
        res_ckpt_filename = cfg.generation.io.res_ckpt_filename
        logger0.info(f'Loading residual network from "{res_ckpt_filename}"...')
        net_res = Module.from_checkpoint(to_absolute_path(res_ckpt_filename))
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)

        # =====================================================================
        # EMA: Load EMA weights if specified (recommended for better quality)
        # =====================================================================
        ema_ckpt_filename = cfg.generation.io.get("ema_ckpt_filename", None)
        if ema_ckpt_filename:
            logger0.info(f'Loading EMA weights from "{ema_ckpt_filename}"...')
            ema_state = torch.load(to_absolute_path(ema_ckpt_filename), map_location=device)
            # EMA checkpoint contains {"shadow": state_dict, "decay": ..., ...}
            if "shadow" in ema_state:
                net_res.load_state_dict(ema_state["shadow"])
                logger0.info("EMA weights loaded successfully (using smoothed weights for inference)")
            else:
                # If it's a raw state_dict (older format)
                net_res.load_state_dict(ema_state)
                logger0.info("EMA weights loaded (raw state_dict format)")
        else:
            logger0.info("No EMA checkpoint specified - using regular training weights")

        if cfg.generation.perf.force_fp16:
            net_res.use_fp16 = True
        # Disable AMP for inference (even if model is trained with AMP)
        if hasattr(net_res, "amp_mode"):
            net_res.amp_mode = False
    else:
        net_res = None

    # load regression network, move to device, change precision
    if load_net_reg:
        reg_ckpt_filename = cfg.generation.io.reg_ckpt_filename
        logger0.info(f'Loading network from "{reg_ckpt_filename}"...')
        net_reg = Module.from_checkpoint(to_absolute_path(reg_ckpt_filename))
        net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
        if cfg.generation.perf.force_fp16:
            net_reg.use_fp16 = True
        # Disable AMP for inference (even if model is trained with AMP)
        if hasattr(net_reg, "amp_mode"):
            net_reg.amp_mode = False
    else:
        net_reg = None

    # Reset since we are using a different mode.
    if cfg.generation.perf.use_torch_compile:
        torch._dynamo.reset()
        # Only compile residual network
        # Overhead of compiling regression network outweights any benefits
        if net_res:
            net_res = torch.compile(net_res, mode="reduce-overhead")

    # Partially instantiate the sampler based on the configs
    if cfg.sampler.type == "deterministic":
        if cfg.generation.hr_mean_conditioning:
            raise NotImplementedError(
                "High-res mean conditioning is not yet implemented for the deterministic sampler"
            )
        sampler_fn = partial(
            deterministic_sampler,
            num_steps=cfg.sampler.num_steps,
            # num_ensembles=cfg.generation.num_ensembles,
            solver=cfg.sampler.solver,
        )
    elif cfg.sampler.type == "stochastic":
        sampler_fn = partial(stochastic_sampler, patching=patching)
    else:
        raise ValueError(f"Unknown sampling method {cfg.sampling.type}")

    # Helper function to print tensor statistics
    def print_tensor_stats(name, tensor, channel_metadata_list):
        """Print detailed statistics for a tensor, per channel."""
        if tensor is None:
            logger0.info(f"  {name}: None")
            return
        logger0.info(f"  {name}: shape={list(tensor.shape)}, dtype={tensor.dtype}")
        logger0.info(f"    Overall: min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, "
                     f"mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}")
        # Per-channel statistics (using provided channel metadata for names)
        for c in range(min(tensor.shape[1], len(channel_metadata_list))):
            ch_data = tensor[:, c, :, :] # Assuming channel dimension is 1
            ch_name = channel_metadata_list[c].name if hasattr(channel_metadata_list[c], 'name') else f"ch{c}"
            logger0.info(f"    [{ch_name}]: min={ch_data.min().item():.6f}, max={ch_data.max().item():.6f}, "
                         f"mean={ch_data.mean().item():.6f}, std={ch_data.std().item():.6f}")

    # Main generation definition
    def generate_fn():
        nonlocal debug_done
        with nvtx.annotate("generate_fn", color="green"):
            # (1, C, H, W)
            img_lr = image_lr # Already converted to channels_last in main loop

            # Debug: print input statistics
            logger0.info("=" * 60)
            logger0.info("DEBUG: Tensor Statistics")
            logger0.info("=" * 60)
            print_tensor_stats("INPUT (img_lr)", img_lr, dataset.input_channels())

            image_reg = None
            image_res = None
            image_res_raw = None
            if net_reg:
                with nvtx.annotate("regression_model", color="yellow"):
                    image_reg = regression_step(
                        net=net_reg,
                        img_lr=img_lr,
                        latents_shape=(
                            cfg.generation.seed_batch_size,
                            img_out_channels,
                            img_shape[0],
                            img_shape[1],
                        ),  # (batch_size, C, H, W)
                        lead_time_label=lead_time_label,
                    )
                # Apply precipitation threshold to regression output
                image_reg = apply_precip_threshold_normalized(image_reg, precip_threshold_info)
                if net_res and land_mask_tensor is not None:
                    image_reg = _apply_mask(image_reg, land_mask_tensor)
                # Debug: print regression output statistics
                print_tensor_stats("REGRESSION OUTPUT (image_reg)", image_reg, dataset.output_channels())

            if net_res:
                if cfg.generation.hr_mean_conditioning:
                    mean_hr = image_reg[0:1]
                else:
                    mean_hr = None
                with nvtx.annotate("diffusion model", color="purple"):
                    image_res = diffusion_step(
                        net=net_res,
                        sampler_fn=sampler_fn,
                        img_shape=img_shape,
                        img_out_channels=img_out_channels,
                        rank_batches=rank_batches,
                        img_lr=img_lr.expand(cfg.generation.seed_batch_size, -1, -1, -1),
                        rank=dist.rank,
                        device=device,
                        mean_hr=mean_hr,
                        lead_time_label=lead_time_label,
                    )
                image_res_raw = image_res
                if residual_std_tensor is not None:
                    image_res = image_res * residual_std_tensor.view(1, -1, 1, 1)
                    logger0.info("Applied per-channel residual std scaling to diffusion residual.")
                # Optional residual scaling/clipping to reduce speckle noise.
                residual_clip = getattr(cfg.generation, "residual_clip", None)
                if residual_clip is not None and residual_clip > 0:
                    image_res = torch.clamp(image_res, -residual_clip, residual_clip)
                    logger0.info(f"Applied residual clipping: ±{residual_clip}")

                residual_scale = getattr(cfg.generation, "residual_scale", 1.0)
                if residual_scale != 1.0:
                    image_res = image_res * residual_scale
                    logger0.info(f"Applied residual scaling: x{residual_scale}")
                if land_mask_tensor is not None:
                    image_res_raw = _apply_mask(image_res_raw, land_mask_tensor)
                    image_res = _apply_mask(image_res, land_mask_tensor)
                # Debug: print diffusion residual statistics
                print_tensor_stats("DIFFUSION RESIDUAL (image_res)", image_res, dataset.output_channels())

                # Debug: print ratio of residual to regression
                if net_reg:
                    ratio_eps = getattr(cfg.generation, "ratio_eps", 1e-3)
                    reg_abs = image_reg.abs()
                    valid = reg_abs > ratio_eps
                    if valid.any():
                        ratio = (image_res.abs() / reg_abs).masked_select(valid)
                        logger0.info(
                            f"  RESIDUAL/REGRESSION RATIO (|reg|>{ratio_eps}): "
                            f"min={ratio.min().item():.6f}, "
                            f"max={ratio.max().item():.6f}, "
                            f"mean={ratio.mean().item():.6f}"
                        )
                    else:
                        logger0.info(
                            f"  RESIDUAL/REGRESSION RATIO skipped (|reg|>{ratio_eps}): no valid pixels"
                        )

            if cfg.generation.inference_mode == "regression":
                image_out = image_reg
            elif cfg.generation.inference_mode == "diffusion":
                image_out = image_res
            else:
                image_out = image_reg + image_res

            # Apply precipitation threshold to final output
            image_out = apply_precip_threshold_normalized(image_out, precip_threshold_info)

            # Debug: print final output statistics (before masking)
            print_tensor_stats("FINAL OUTPUT (before mask)", image_out, dataset.output_channels())
            logger0.info("=" * 60)

            # Debug plots (rank 0 only)
            if debug_enabled and dist.rank == 0 and (not debug_done or not debug_cfg["once"]):
                if _maybe_import_matplotlib() is None:
                    logger0.warning("matplotlib not available; skipping generation debug plots.")
                else:
                    debug_out_dir.mkdir(parents=True, exist_ok=True)
                    real_mask = rank_real_mask.bool()
                    real_indices = (
                        torch.nonzero(real_mask, as_tuple=False).flatten()
                        if real_mask.numel() > 0
                        else torch.tensor([], dtype=torch.int64)
                    )
                    num_real = int(real_indices.numel()) if real_indices.numel() > 0 else 0
                    num_samples = min(debug_cfg["num_samples"], num_real)
                    if num_samples > 0:
                        gen = torch.Generator(device="cpu")
                        gen.manual_seed(debug_cfg["seed"])
                        pick = torch.randperm(num_real, generator=gen)[:num_samples]
                        sample_indices = real_indices[pick].tolist()

                        mask_np = None
                        if land_mask_tensor is not None:
                            mask_np = _tensor_to_np(land_mask_tensor)[0, 0]

                        output_vars = getattr(dataset, "output_variables", None) or [
                            f"ch{i}" for i in range(image_out.shape[1])
                        ]
                        cmap_by_col = {
                            "LR": "viridis",
                            "Target": "viridis",
                            "RegMean": "viridis",
                            "ResidualGT": "RdBu_r",
                            "ResidualTarget": "RdBu_r",
                            "D_yn_1step": "RdBu_r",
                            "DiffResidualRaw": "RdBu_r",
                            "DiffResidual": "RdBu_r",
                            "Output": "viridis",
                            "Mask": "gray",
                        }

                        lr_np = _tensor_to_np(img_lr[0])
                        tar_np = _tensor_to_np(image_tar[0])
                        reg_np = None
                        if net_reg and image_reg is not None:
                            reg_np = _tensor_to_np(image_reg[0])

                        residual_std_np = None
                        if residual_std_tensor is not None:
                            residual_std_np = _tensor_to_np(residual_std_tensor).reshape(-1, 1, 1)

                        d_yn_np = None
                        if net_res and image_reg is not None and image_tar is not None:
                            residual_gt = image_tar - image_reg
                            if residual_std_tensor is not None:
                                residual_gt = residual_gt / (
                                    residual_std_tensor.view(1, -1, 1, 1) + 1e-8
                                )
                            y_cond = img_lr
                            if cfg.generation.hr_mean_conditioning:
                                y_cond = torch.cat((image_reg, img_lr), dim=1)
                            with torch.no_grad():
                                with torch.random.fork_rng(
                                    devices=[image_tar.device.index] if image_tar.device.type == "cuda" else [],
                                    enabled=True,
                                ):
                                    base_seed = debug_cfg["seed"] if debug_noise_seed is None else debug_noise_seed
                                    torch.manual_seed(base_seed)
                                    if debug_sigma is not None:
                                        sigma = torch.full(
                                            [residual_gt.shape[0], 1, 1, 1],
                                            float(debug_sigma),
                                            device=image_tar.device,
                                        )
                                    else:
                                        rnd_normal = torch.randn(
                                            [residual_gt.shape[0], 1, 1, 1],
                                            device=image_tar.device,
                                        )
                                        sigma = (rnd_normal * 1.2 + 0.0).exp()
                                    latent = residual_gt + torch.randn_like(residual_gt) * sigma
                                    if lead_time_label is not None:
                                        d_yn = net_res(
                                            latent,
                                            y_cond,
                                            sigma,
                                            embedding_selector=None,
                                            global_index=None,
                                            lead_time_label=lead_time_label,
                                            augment_labels=None,
                                        )
                                    else:
                                        d_yn = net_res(
                                            latent,
                                            y_cond,
                                            sigma,
                                            embedding_selector=None,
                                            global_index=None,
                                            augment_labels=None,
                                        )
                            if land_mask_tensor is not None:
                                d_yn = _apply_mask(d_yn, land_mask_tensor)
                            d_yn_np = _tensor_to_np(d_yn[0])

                        for sample_idx in sample_indices:
                            out_np = _tensor_to_np(image_out[sample_idx])

                            data_by_col = {
                                "LR": lr_np,
                                "Target": tar_np,
                                "Output": out_np,
                            }
                            cols = ["LR", "Target"]

                            if net_reg:
                                if reg_np is not None:
                                    data_by_col["RegMean"] = reg_np
                                    cols.append("RegMean")
                                if tar_np is not None:
                                    if reg_np is not None:
                                        residual_gt = tar_np - reg_np
                                        if mask_np is not None:
                                            residual_gt = np.where(mask_np, residual_gt, 0.0)
                                        data_by_col["ResidualGT"] = residual_gt
                                        cols.append("ResidualGT")
                                        if residual_std_np is not None:
                                            residual_target = residual_gt / (residual_std_np + 1e-8)
                                            data_by_col["ResidualTarget"] = residual_target
                                            cols.append("ResidualTarget")

                            if net_res:
                                if "image_res_raw" in locals() and image_res_raw is not None:
                                    res_raw_np = _tensor_to_np(image_res_raw[sample_idx])
                                    data_by_col["DiffResidualRaw"] = res_raw_np
                                    cols.append("DiffResidualRaw")
                                if d_yn_np is not None:
                                    data_by_col["D_yn_1step"] = d_yn_np
                                    cols.append("D_yn_1step")
                                res_np = _tensor_to_np(image_res[sample_idx])
                                data_by_col["DiffResidual"] = res_np
                                cols.append("DiffResidual")

                            cols.append("Output")

                            if mask_np is not None:
                                data_by_col["Mask"] = np.repeat(
                                    mask_np[None, ...], out_np.shape[0], axis=0
                                )
                                cols.append("Mask")

                            channel_count = min(arr.shape[0] for arr in data_by_col.values())
                            rows = output_vars[:channel_count]
                            for k in list(data_by_col.keys()):
                                data_by_col[k] = data_by_col[k][:channel_count]

                            out_path = debug_out_dir / f"debug_generate_t{time_index:04d}_s{sample_idx}.png"
                            _plot_debug_grid(
                                out_path=out_path,
                                rows=rows,
                                cols=cols,
                                data_by_col=data_by_col,
                                cmap_by_col=cmap_by_col,
                                symmetric_cols={"ResidualGT", "ResidualTarget", "D_yn_1step", "DiffResidualRaw"},
                                shared_cols={"ResidualGT", "ResidualTarget", "D_yn_1step", "DiffResidualRaw"},
                            )
                            logger0.info(f"Saved generation debug plot: {out_path}")

                        if debug_cfg["once"]:
                            debug_done = True

            # Apply land mask to zero out ocean areas (ON GPU)
            if land_mask_tensor is not None:
                image_out = image_out * land_mask_tensor

            # Gather tensors on rank 0
            if dist.world_size > 1:
                if dist.rank == 0:
                    gathered_tensors = [
                        torch.zeros_like(
                            image_out, dtype=image_out.dtype, device=image_out.device
                        )
                        for _ in range(dist.world_size)
                    ]
                else:
                    gathered_tensors = None

                # Gather masks so we can drop padded ensembles on rank 0.
                rank_real_mask_dev = rank_real_mask.to(device=image_out.device)
                if dist.rank == 0:
                    gathered_masks = [
                        torch.zeros_like(rank_real_mask_dev)
                        for _ in range(dist.world_size)
                    ]
                else:
                    gathered_masks = None

                torch.distributed.barrier()
                gather(
                    image_out,
                    gather_list=gathered_tensors if dist.rank == 0 else None,
                    dst=0,
                )
                gather(
                    rank_real_mask_dev,
                    gather_list=gathered_masks if dist.rank == 0 else None,
                    dst=0,
                )

                if dist.rank == 0:
                    images_cat = torch.cat(gathered_tensors)
                    masks_cat = torch.cat(gathered_masks).bool()
                    if masks_cat.numel() == images_cat.shape[0]:
                        images_cat = images_cat[masks_cat]
                    return images_cat
                else:
                    return None
            else:
                if rank_real_mask.numel() == image_out.shape[0]:
                    image_out = image_out[rank_real_mask.bool().to(image_out.device)]
                return image_out

    # generate images
    output_path = getattr(cfg.generation.io, "output_filename", "corrdiff_output.nc")
    logger0.info(f"Generating images, saving results to {output_path}...")
    # The DataLoader batch_size is always 1 for generation

    # Get the times that will actually be generated (based on sampler indices)
    # This handles both: (1) random sampling / debug filtering where dataset is modified,
    # and (2) explicit times where only the sampler is filtered but dataset is unchanged.
    all_dataset_times = dataset.time()
    if isinstance(sampler, range):
        # Dataset was modified in-place (random sampling or debug filtering)
        times = all_dataset_times
    else:
        # Sampler contains specific indices into the original dataset
        times = [all_dataset_times[i] for i in sampler]
    warmup_steps = min(len(times) - 1, 2)
    
    if dist.rank == 0:
        f = nc.Dataset(output_path, "w")
        # add attributes
        f.cfg = str(cfg)

    torch_cuda_profiler = (
        torch.cuda.profiler.profile()
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    torch_nvtx_profiler = (
        torch.autograd.profiler.emit_nvtx()
        if torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with torch_cuda_profiler:
        with torch_nvtx_profiler:

            data_loader = torch.utils.data.DataLoader(
                dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True
            )
            time_index = -1
            writer_lock = None
            if dist.rank == 0:
                cropped_lat = dataset.latitude()[:img_shape[0], :img_shape[1]]
                cropped_lon = dataset.longitude()[:img_shape[0], :img_shape[1]]
                
                writer = NetCDFWriter(
                    f,
                    # lat=dataset.latitude(),
                    # lon=dataset.longitude(),
                    lat=cropped_lat,
                    lon=cropped_lon,
                    input_channels=dataset.input_channels(),
                    output_channels=dataset.output_channels(),
                    has_lead_time=has_lead_time,
                )
                writer_lock = threading.Lock()

                # Initialize threadpool for writers
                writer_executor = ThreadPoolExecutor(
                    max_workers=cfg.generation.perf.num_writer_workers
                )
                writer_threads = []

            # Create timer objects only if CUDA is available
            use_cuda_timing = torch.cuda.is_available()
            start = torch.cuda.Event(enable_timing=True) if use_cuda_timing else type('DummyEvent', (object,), {'record': lambda self: None, 'synchronize': lambda self: None, 'elapsed_time': lambda self, _: 0})()
            end = torch.cuda.Event(enable_timing=True) if use_cuda_timing else type('DummyEvent', (object,), {'record': lambda self: None, 'synchronize': lambda self: None, 'elapsed_time': lambda self, _: 0})()

            # Track whether start was recorded (only happens after warmup_steps)
            start_recorded = False

            try:
                # Iterate through the (potentially subsetted) dataset
                for index, (image_tar, image_lr, *lead_time_label) in enumerate(
                    iter(data_loader)
                ):
                    time_index += 1
                    if dist.rank == 0:
                        logger0.info(f"starting index: {time_index}")

                    if time_index == warmup_steps:
                        start.record()
                        start_recorded = True

                    # continue
                    if lead_time_label:
                        lead_time_label = lead_time_label[0].to(dist.device).contiguous()
                    else:
                        lead_time_label = None
                    image_lr = (
                        image_lr.to(device=device)
                        .to(torch.float32)
                        .to(memory_format=torch.channels_last)
                    )
                    image_tar = image_tar.to(device=device).to(torch.float32)

                    # Ensure no NaN/Inf values in inputs (same as training)
                    image_lr = torch.nan_to_num(image_lr, nan=0.0, posinf=0.0, neginf=0.0)
                    image_tar = torch.nan_to_num(image_tar, nan=0.0, posinf=0.0, neginf=0.0)

                    # For diffusion inputs, zero ocean+buffer to match training.
                    if net_res is not None and land_mask_tensor is not None:
                        image_lr = _apply_mask(image_lr, land_mask_tensor)
                        image_tar = _apply_mask(image_tar, land_mask_tensor)

                    image_out = generate_fn()

                    if dist.rank == 0:
                        # --- APPLY NAN MASKING ON CPU ---
                        image_out_cpu = image_out.cpu()
                        image_tar_cpu = image_tar.cpu()
                        image_lr_cpu = image_lr.cpu()

                        # Use the expanded ocean mask (includes coastal buffer) for NaN masking
                        # This marks ocean AND coastal buffer areas as NaN in the output
                        if ocean_mask_with_buffer_cpu is not None:
                            is_masked = ocean_mask_with_buffer_cpu == 1

                            # Expand the mask separately for each tensor, since the
                            # generated batch can be larger than the dataloader batch.
                            is_masked_out = is_masked.expand_as(image_out_cpu)
                            is_masked_tar = is_masked.expand_as(image_tar_cpu)
                            is_masked_lr = is_masked.expand_as(image_lr_cpu)

                            image_out_cpu[is_masked_out] = float("nan")
                            image_tar_cpu[is_masked_tar] = float("nan")
                            image_lr_cpu[is_masked_lr] = float("nan")
                        elif land_mask_cpu is not None:
                            # Fallback: use land mask (ocean = where land_mask == 0)
                            is_ocean = land_mask_cpu == 0
                            is_ocean_out = is_ocean.expand_as(image_out_cpu)
                            is_ocean_tar = is_ocean.expand_as(image_tar_cpu)
                            is_ocean_lr = is_ocean.expand_as(image_lr_cpu)

                            image_out_cpu[is_ocean_out] = float("nan")
                            image_tar_cpu[is_ocean_tar] = float("nan")
                            image_lr_cpu[is_ocean_lr] = float("nan")

                        # write out data
                        writer_threads.append(
                            writer_executor.submit(
                                save_images,
                                writer,
                                dataset,
                                list(times),
                                image_out_cpu,
                                image_tar_cpu,
                                image_lr_cpu,
                                time_index,
                                index,
                                has_lead_time,
                                writer_lock,
                            )
                        )
            finally:
                # Ensure end.record() is called even if an exception occurs in the loop
                if use_cuda_timing and start_recorded:
                    end.record()
                    end.synchronize()

            # Only calculate elapsed time if both events were recorded
            if use_cuda_timing and start_recorded:
                elapsed_time = start.elapsed_time(end) / 1000.0  # Convert ms to s
            else:
                elapsed_time = 0
            timed_steps = time_index + 1 - warmup_steps
            if dist.rank == 0 and use_cuda_timing and start_recorded and timed_steps > 0:
                average_time_per_batch_element = elapsed_time / timed_steps / 1 # DataLoader batch_size is 1
                logger.info(
                    f"Total time to run {timed_steps} steps and {1} members = {elapsed_time} s"
                )
                logger.info(
                    f"Average time per batch element = {average_time_per_batch_element} s"
                )

            # make sure all the workers are done writing
            if dist.rank == 0:
                for thread in list(writer_threads):
                    thread.result()
                    writer_threads.remove(thread)
                writer_executor.shutdown()

    if dist.rank == 0:
        f.close()
    logger0.info("Generation Completed.")


if __name__ == "__main__":
    main()
