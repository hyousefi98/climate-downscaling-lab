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

import os
import time
import psutil
from pathlib import Path
from contextlib import nullcontext
import inspect
import json
import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
import xarray as xr  # Added for mask loading
import numpy as np   # Added for mask processing
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import nvtx
import wandb

from physicsnemo import Module
from physicsnemo.models.diffusion import UNet, EDMPrecondSuperResolution
from physicsnemo.distributed import DistributedManager
from physicsnemo.metrics.diffusion import RegressionLoss, ResidualLoss, RegressionLossCE
from models.losses_aorc import (
    MaskedResidualLoss,
    RainAwareRegressionLoss,
    apply_precip_threshold_normalized,
)
from physicsnemo.utils.patching import RandomPatching2D
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.utils import (
    load_checkpoint,
    save_checkpoint,
    get_checkpoint_dir,
)

from models.datasets.factory import init_train_valid_datasets_from_config, register_dataset
from models.helpers.train_helpers import (
    set_patch_shape,
    set_seed,
    configure_cuda_for_consistent_precision,
    compute_num_accumulation_rounds,
    handle_and_clip_gradients,
    is_time_for_periodic_task,
)
from models.helpers.ema import EMA  # Exponential Moving Average for diffusion models

torch._dynamo.reset()
# Increase the cache size limit
torch._dynamo.config.cache_size_limit = 264
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = False
torch._logging.set_logs(recompiles=True, graph_breaks=True)


def checkpoint_list(path, suffix=".mdlus"):
    """Helper function to return sorted list, in ascending order, of checkpoints in a path"""
    checkpoints = []
    for file in os.listdir(path):
        if file.endswith(suffix):
            try:
                index = int(file.split(".")[-2])
                checkpoints.append((index, file))
            except ValueError:
                continue
    checkpoints.sort(key=lambda x: x[0])
    return [file for _, file in checkpoints]


# Define safe CUDA profiler tools that fallback to no-ops when CUDA is not available
def cuda_profiler():
    if torch.cuda.is_available():
        return torch.cuda.profiler.profile()
    else:
        return nullcontext()


def cuda_profiler_start():
    if torch.cuda.is_available():
        torch.cuda.profiler.start()


def cuda_profiler_stop():
    if torch.cuda.is_available():
        torch.cuda.profiler.stop()


def profiler_emit_nvtx():
    if torch.cuda.is_available():
        return torch.autograd.profiler.emit_nvtx()
    else:
        return nullcontext()


def _nan_to_num_(x: torch.Tensor) -> torch.Tensor:
    """Cheap guard against NaN/Inf in tensors. Keeps gradients."""
    if not torch.is_tensor(x):
        return x
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _apply_loss_mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x
    if mask.dtype != x.dtype:
        mask = mask.to(dtype=x.dtype)
    return x * mask


class _MaskedRegressionNet(torch.nn.Module):
    def __init__(self, net: torch.nn.Module, mask: torch.Tensor | None):
        super().__init__()
        self.net = net
        if mask is not None:
            self.register_buffer("mask", mask, persistent=False)
        else:
            self.mask = None

    def forward(self, *args, **kwargs):
        out = self.net(*args, **kwargs)
        if self.mask is None:
            return out
        return _apply_loss_mask(out, self.mask)


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


def _get_debug_settings() -> dict:
    def _env_bool(name: str, default: str = "0") -> bool:
        return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}

    return {
        "enabled": _env_bool("CORRDIFF_DEBUG", "0"),
        "no_ckpt": _env_bool("CORRDIFF_DEBUG_NO_CKPT", "0"),
        "out_dir": os.getenv("CORRDIFF_DEBUG_DIR", "debug_plots"),
        "num_samples": int(os.getenv("CORRDIFF_DEBUG_SAMPLES", "2")),
        "seed": int(os.getenv("CORRDIFF_DEBUG_SEED", "0")),
        "index": os.getenv("CORRDIFF_DEBUG_INDEX", "").strip(),
        "time": os.getenv("CORRDIFF_DEBUG_TIME", "").strip(),
        "sigma": _parse_float(os.getenv("CORRDIFF_DEBUG_SIGMA", "").strip()),
        "noise_seed": _parse_int(os.getenv("CORRDIFF_DEBUG_NOISE_SEED", "").strip()),
        "epoch": _parse_int(os.getenv("CORRDIFF_DEBUG_EPOCH", "").strip()),
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


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _get_debug_sample(
    *,
    dataset,
    debug_index: str,
    debug_time: str,
    logger,
):
    index = None
    if debug_index:
        index = _parse_int(debug_index)
    if index is None and debug_time:
        try:
            times = dataset.time()
            for i, t in enumerate(times):
                t_str = t.strftime("%Y-%m-%dT%H:%M:%S")
                if t_str == debug_time:
                    index = i
                    break
        except Exception:
            index = None

    if index is None:
        return None
    if index < 0 or index >= len(dataset):
        logger.warning(f"Debug index {index} out of range (0..{len(dataset)-1}).")
        return None

    sample = dataset[index]
    if not isinstance(sample, (tuple, list)) or len(sample) < 2:
        logger.warning("Debug sample format unexpected; skipping.")
        return None

    img_clean = sample[0].unsqueeze(0)
    img_lr = sample[1].unsqueeze(0)
    lead_time = sample[2] if len(sample) > 2 else None
    if lead_time is not None and torch.is_tensor(lead_time):
        lead_time = lead_time.unsqueeze(0)

    try:
        times = dataset.time()
        if 0 <= index < len(times):
            t = times[index]
            logger.info(f"Debug sample time: {t.strftime('%Y-%m-%dT%H:%M:%S')}")
    except Exception:
        pass

    return img_clean, img_lr, lead_time


def _run_debug_plots(
    *,
    mode: str,
    out_dir: Path,
    model: torch.nn.Module,
    regression_net: torch.nn.Module | None,
    loss_fn,
    img_clean: torch.Tensor,
    img_lr: torch.Tensor,
    lead_time_label: torch.Tensor | None,
    input_vars: list[str],
    output_vars: list[str],
    loss_mask_tensor: torch.Tensor | None,
    precip_threshold_info: dict | None,
    enable_amp: bool,
    amp_dtype: torch.dtype,
    seed: int,
    num_samples: int,
    sigma_override: float | None = None,
    noise_seed: int | None = None,
    logger,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    if _maybe_import_matplotlib() is None:
        logger.warning("matplotlib not available; skipping debug plots.")
        return
    if loss_mask_tensor is not None and mode == "diffusion":
        img_clean = _apply_loss_mask(img_clean, loss_mask_tensor)
        img_lr = _apply_loss_mask(img_lr, loss_mask_tensor)
    batch_size = img_clean.shape[0]
    num_samples = min(max(1, num_samples), batch_size)
    if num_samples <= 0:
        return

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    sample_indices = torch.randperm(batch_size, generator=gen)[:num_samples].tolist()

    model_was_training = model.training
    model.eval()
    reg_was_training = None
    if regression_net is not None:
        reg_was_training = regression_net.training
        regression_net.eval()

    with torch.no_grad():
        autocast_enabled = enable_amp and img_clean.device.type == "cuda"
        autocast_device = "cuda" if img_clean.device.type == "cuda" else "cpu"
        with torch.autocast(autocast_device, dtype=amp_dtype, enabled=autocast_enabled):
            if mode == "regression":
                zero_input = torch.zeros_like(img_clean, device=img_clean.device)
                if lead_time_label is not None:
                    pred = model(
                        zero_input,
                        img_lr,
                        force_fp32=False,
                        lead_time_label=lead_time_label,
                        augment_labels=None,
                    )
                else:
                    pred = model(
                        zero_input,
                        img_lr,
                        force_fp32=False,
                        augment_labels=None,
                    )
                loss_map = (pred - img_clean) ** 2
                if loss_mask_tensor is not None and loss_map.ndim >= 3:
                    loss_map = loss_map * loss_mask_tensor
            else:
                if regression_net is None:
                    logger.warning("Debug plotting skipped: regression_net is required for diffusion.")
                    return

                zero_input = torch.zeros_like(img_clean, device=img_clean.device)
                if lead_time_label is not None:
                    y_mean = regression_net(
                        zero_input,
                        img_lr,
                        lead_time_label=lead_time_label,
                        augment_labels=None,
                    )
                else:
                    y_mean = regression_net(
                        zero_input,
                        img_lr,
                        augment_labels=None,
                    )

                if loss_mask_tensor is not None:
                    y_mean = _apply_loss_mask(y_mean, loss_mask_tensor)
                # Apply precipitation threshold to regression output and target
                y_mean = apply_precip_threshold_normalized(y_mean, precip_threshold_info)
                img_clean_thresh = apply_precip_threshold_normalized(img_clean, precip_threshold_info)
                residual = img_clean_thresh - y_mean
                if loss_mask_tensor is not None:
                    residual = _apply_loss_mask(residual, loss_mask_tensor)
                residual_for_loss = residual
                residual_std = getattr(loss_fn, "residual_std", None)
                if residual_std is not None:
                    std = residual_std.to(img_clean.device)
                    if std.dim() == 1:
                        std = std.view(1, -1, 1, 1)
                    residual_for_loss = residual / (std + 1e-8)

                hr_mean_conditioning = bool(getattr(loss_fn, "hr_mean_conditioning", False))
                y_cond = img_lr
                if hr_mean_conditioning:
                    y_cond = torch.cat((y_mean, img_lr), dim=1)

                P_mean = float(getattr(loss_fn, "P_mean", 0.0))
                P_std = float(getattr(loss_fn, "P_std", 1.2))
                sigma_data = float(getattr(loss_fn, "sigma_data", 0.5))

                fork_devices = []
                if img_clean.device.type == "cuda":
                    fork_devices = [img_clean.device.index]
                with torch.random.fork_rng(devices=fork_devices, enabled=True):
                    base_seed = seed if noise_seed is None else noise_seed
                    torch.manual_seed(base_seed)
                    if sigma_override is not None:
                        sigma = torch.full(
                            [residual_for_loss.shape[0], 1, 1, 1],
                            float(sigma_override),
                            device=img_clean.device,
                        )
                    else:
                        rnd_normal = torch.randn(
                            [residual_for_loss.shape[0], 1, 1, 1],
                            device=img_clean.device,
                        )
                        sigma = (rnd_normal * P_std + P_mean).exp()
                    weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
                    latent = residual_for_loss + torch.randn_like(residual_for_loss) * sigma

                if lead_time_label is not None:
                    D_yn = model(
                        latent,
                        y_cond,
                        sigma,
                        embedding_selector=None,
                        global_index=None,
                        lead_time_label=lead_time_label,
                        augment_labels=None,
                    )
                else:
                    D_yn = model(
                        latent,
                        y_cond,
                        sigma,
                        embedding_selector=None,
                        global_index=None,
                        augment_labels=None,
                    )

                loss_map = weight * ((D_yn - residual_for_loss) ** 2)
                if loss_mask_tensor is not None and loss_map.ndim >= 3:
                    loss_map = loss_map * loss_mask_tensor

                sigma_np = _tensor_to_np(sigma).reshape(-1)
                weight_np = _tensor_to_np(weight).reshape(-1)

    if reg_was_training is not None:
        regression_net.train(reg_was_training)
    model.train(model_was_training)

    cmap_by_col = {
        "LR": "viridis",
        "Pred": "viridis",
        "Target": "viridis",
        "RegMean": "viridis",
        "ResidualGT": "RdBu_r",
        "ResidualTarget": "RdBu_r",
        "D_yn": "RdBu_r",
        "Loss": "magma",
        "Mask": "gray",
    }

    output_vars = output_vars or [f"ch{i}" for i in range(img_clean.shape[1])]
    mask_np = None
    if loss_mask_tensor is not None:
        mask_np = _tensor_to_np(loss_mask_tensor)[0, 0]

    for sample_idx in sample_indices:
        lr_np = _tensor_to_np(img_lr[sample_idx])
        hr_np = _tensor_to_np(img_clean[sample_idx])

        if mode == "regression":
            pred_np = _tensor_to_np(pred[sample_idx])
            loss_np = _tensor_to_np(loss_map[sample_idx])

            cols = ["LR", "Pred", "Target", "Loss"]
            data_by_col = {
                "LR": lr_np,
                "Pred": pred_np,
                "Target": hr_np,
                "Loss": loss_np,
            }
            if mask_np is not None:
                cols.append("Mask")
                data_by_col["Mask"] = np.repeat(mask_np[None, ...], hr_np.shape[0], axis=0)

            channel_count = min(arr.shape[0] for arr in data_by_col.values())
            rows = output_vars[:channel_count] if output_vars else [f"ch{i}" for i in range(channel_count)]
            for k in list(data_by_col.keys()):
                data_by_col[k] = data_by_col[k][:channel_count]

            out_path = out_dir / f"debug_{mode}_sample{sample_idx}.png"
            _plot_debug_grid(
                out_path=out_path,
                rows=rows,
                cols=cols,
                data_by_col=data_by_col,
                cmap_by_col=cmap_by_col,
                symmetric_cols={"Residual", "D_yn"},
            )

            loss_per_channel = loss_np.reshape(loss_np.shape[0], -1).mean(axis=1).tolist()
            logger.info(
                f"Debug regression sample {sample_idx}: loss_mean={float(loss_np.mean()):.6f} "
                f"loss_per_channel={loss_per_channel}"
            )
        else:
            y_mean_np = _tensor_to_np(y_mean[sample_idx])
            residual_np = _tensor_to_np(residual_for_loss[sample_idx])
            d_yn_plot = D_yn
            if loss_mask_tensor is not None:
                d_yn_plot = _apply_loss_mask(d_yn_plot, loss_mask_tensor)
            d_yn_np = _tensor_to_np(d_yn_plot[sample_idx])
            loss_np = _tensor_to_np(loss_map[sample_idx])

            cols = ["LR", "Target", "RegMean", "ResidualGT", "ResidualTarget", "D_yn", "Loss"]
            data_by_col = {
                "LR": lr_np,
                "Target": hr_np,
                "RegMean": y_mean_np,
                "ResidualGT": _tensor_to_np(residual[sample_idx]),
                "ResidualTarget": residual_np,
                "D_yn": d_yn_np,
                "Loss": loss_np,
            }
            if mask_np is not None:
                cols.append("Mask")
                data_by_col["Mask"] = np.repeat(mask_np[None, ...], hr_np.shape[0], axis=0)

            channel_count = min(arr.shape[0] for arr in data_by_col.values())
            rows = output_vars[:channel_count] if output_vars else [f"ch{i}" for i in range(channel_count)]
            for k in list(data_by_col.keys()):
                data_by_col[k] = data_by_col[k][:channel_count]

            out_path = out_dir / f"debug_{mode}_sample{sample_idx}.png"
            _plot_debug_grid(
                out_path=out_path,
                rows=rows,
                cols=cols,
                data_by_col=data_by_col,
                cmap_by_col=cmap_by_col,
                symmetric_cols={"ResidualGT", "ResidualTarget", "D_yn"},
                shared_cols={"ResidualGT", "ResidualTarget", "D_yn"},
            )

            loss_per_channel = loss_np.reshape(loss_np.shape[0], -1).mean(axis=1).tolist()
            logger.info(
                f"Debug diffusion sample {sample_idx}: sigma={float(sigma_np[sample_idx]):.4f} "
                f"weight={float(weight_np[sample_idx]):.4f} "
                f"loss_mean={float(loss_np.mean()):.6f} loss_per_channel={loss_per_channel} "
                f"residual_std={'on' if residual_std is not None else 'off'}"
            )

# Train the CorrDiff model using the configurations in "conf/config_training.yaml"
@hydra.main(version_base="1.2", config_path="conf", config_name="config_training")
def main(cfg: DictConfig) -> None:

    # Initialize distributed environment for training
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers
    writer = None
    if dist.rank == 0:
        writer = SummaryWriter(log_dir="tensorboard")
    logger = PythonLogger("main")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    debug_cfg = _get_debug_settings()
    debug_enabled = debug_cfg["enabled"]
    debug_no_ckpt = debug_cfg["no_ckpt"]
    debug_done = False
    debug_out_dir = Path(debug_cfg["out_dir"])
    initialize_wandb(
        project="Modulus-Launch",
        entity="Modulus",
        name=f"CorrDiff-Training-{HydraConfig.get().job.name}",
        group="CorrDiff-DDP-Group",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg),
        results_dir=cfg.wandb.results_dir,
    )

    # Resolve and parse configs
    OmegaConf.resolve(cfg)
    dataset_cfg = OmegaConf.to_container(cfg.dataset)

    # Register custom dataset if specified in config
    register_dataset(cfg.dataset.type)
    logger0.info(f"Using dataset: {cfg.dataset.type}")

    if hasattr(cfg, "validation"):
        validation = True
        validation_dataset_cfg = OmegaConf.to_container(cfg.validation)
    else:
        validation = False
        validation_dataset_cfg = None

    fp_optimizations = cfg.training.perf.fp_optimizations
    songunet_checkpoint_level = cfg.training.perf.songunet_checkpoint_level
    fp16 = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16

    logger.info(f"Saving the outputs in {os.getcwd()}")

    checkpoint_dir = get_checkpoint_dir(
        str(cfg.training.io.get("checkpoint_dir", ".")), cfg.model.name
    )
    if cfg.training.hp.batch_size_per_gpu == "auto":
        cfg.training.hp.batch_size_per_gpu = (
            cfg.training.hp.total_batch_size // dist.world_size
        )

    # Load the current number of images for resuming
    try:
        cur_nimg = load_checkpoint(
            path=checkpoint_dir,
        )
    except Exception:
        cur_nimg = 0
    if debug_cfg["epoch"] is not None:
        cur_nimg = debug_cfg["epoch"]

    # Set seeds and configure CUDA and cuDNN settings
    set_seed(dist.rank + cur_nimg)
    configure_cuda_for_consistent_precision()

    # Instantiate the dataset
    num_workers = int(cfg.training.perf.dataloader_workers)
    data_loader_kwargs = {
        "pin_memory": True,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        data_loader_kwargs["prefetch_factor"] = int(
            cfg.training.perf.get("prefetch_factor", 2)
        )
        data_loader_kwargs["persistent_workers"] = bool(
            cfg.training.perf.get("persistent_workers", True)
        )
    # Optional sampler controls for IO/locality tuning.
    for key in (
        "sampler_mode",
        "sampler_shuffle",
        "sampler_window_size",
        "sampler_chunk_size",
        "sampler_shuffle_within_chunk",
    ):
        if key in cfg.training.perf:
            data_loader_kwargs[key] = cfg.training.perf.get(key)
    (
        dataset,
        dataset_iterator,
        validation_dataset,
        validation_dataset_iterator,
    ) = init_train_valid_datasets_from_config(
        dataset_cfg,
        data_loader_kwargs,
        batch_size=cfg.training.hp.batch_size_per_gpu,
        seed=0,
        validation_dataset_cfg=validation_dataset_cfg,
        validation=validation,
        sampler_start_idx=cur_nimg,
    )

    # Parse image configuration
    dataset_in_channels = len(dataset.input_channels())
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())

    # Conditioning channel count (low-res inputs + optional HR mean conditioning)
    img_in_channels = dataset_in_channels
    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels

    # --- LOAD LOSS MASK ---
    # We use the loss mask from the dataset which includes the coastal buffer
    # This mask excludes ocean AND the coastal buffer zone from loss computation
    loss_mask_tensor = None
    if hasattr(dataset, 'get_loss_mask_tensor'):
        loss_mask_tensor = dataset.get_loss_mask_tensor(device=dist.device)
        if loss_mask_tensor is not None:
            buffer_pixels = cfg.dataset.get("coastal_buffer_pixels", 0)
            logger0.info(f"Using loss mask from dataset (coastal buffer: {buffer_pixels} pixels)")
            logger0.info(f"Loss mask shape: {loss_mask_tensor.shape}, valid pixels: {loss_mask_tensor.sum().item():.0f}")

    # Fallback to loading land mask directly if dataset doesn't provide loss mask
    if loss_mask_tensor is None and cfg.dataset.get("landmask_path"):
        logger0.info(f"Falling back to loading Land Mask from: {cfg.dataset.landmask_path}")
        loss_mask_tensor = load_land_mask(cfg.dataset.landmask_path, img_shape, dist.device)
        if loss_mask_tensor is not None:
            logger0.info("Land mask loaded successfully (no coastal buffer).")

    # --- LOAD RESIDUAL STD (OPTIONAL) ---
    residual_std_tensor = None
    residual_std_path = cfg.training.hp.get("residual_std_path", None)
    residual_std_list = cfg.training.hp.get("residual_std", None)
    if residual_std_path:
        try:
            with open(residual_std_path, "r") as f:
                residual_stats = json.load(f)
            # Support either {"output": {var: std}} or {var: std}
            stats_map = residual_stats.get("output", residual_stats)
            residual_std_list = [stats_map[v] for v in dataset.output_variables]
            logger0.info(f"Loaded residual std from {residual_std_path}")
        except Exception as e:
            logger0.warning(f"Could not load residual std from {residual_std_path}: {e}")
            residual_std_list = None
    if residual_std_list is not None:
        residual_std_tensor = torch.tensor(residual_std_list, dtype=torch.float32, device=dist.device)
        logger0.info(f"Using per-channel residual std: {residual_std_list}")

    # --- LOAD PRECIPITATION THRESHOLD INFO (OPTIONAL) ---
    precip_threshold_info = None
    if hasattr(dataset, 'get_precip_threshold_info'):
        precip_threshold_info = dataset.get_precip_threshold_info()
        if precip_threshold_info is not None:
            logger0.info(f"Using precip threshold in normalized space: {precip_threshold_info}")

    # Handle patch shape
    if cfg.model.name == "lt_aware_ce_regression":
        prob_channels = dataset.get_prob_channel_index()
    else:
        prob_channels = None

    if (
        cfg.model.name == "patched_diffusion"
        or cfg.model.name == "lt_aware_patched_diffusion"
    ):
        patch_shape_x = cfg.training.hp.patch_shape_x
        patch_shape_y = cfg.training.hp.patch_shape_y
    else:
        patch_shape_x = None
        patch_shape_y = None

    patch_shape = (patch_shape_y, patch_shape_x)
    use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    
    if use_patching:
        # Utility to perform patches extraction and batching
        patching = RandomPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            patch_num=getattr(cfg.training.hp, "patch_num", 1),
        )
        logger0.info("Patch-based training enabled")
    else:
        patching = None
        logger0.info("Patch-based training disabled")

    # interpolate global channel if patch-based model is used
    if use_patching:
        img_in_channels += dataset_in_channels

    # Instantiate the model
    model_args = {
        "img_out_channels": img_out_channels,
        "img_resolution": list(img_shape),
        "use_fp16": fp16,
        "checkpoint_level": songunet_checkpoint_level,
    }
    if cfg.model.name == "lt_aware_ce_regression":
        model_args["prob_channels"] = prob_channels
    if hasattr(cfg.model, "model_args"):
        model_args.update(OmegaConf.to_container(cfg.model.model_args))

    use_torch_compile = False
    use_apex_gn = False
    profile_mode = False

    if hasattr(cfg.training.perf, "torch_compile"):
        use_torch_compile = cfg.training.perf.torch_compile
    if hasattr(cfg.training.perf, "use_apex_gn"):
        use_apex_gn = cfg.training.perf.use_apex_gn
        model_args["use_apex_gn"] = use_apex_gn
    if hasattr(cfg.training.perf, "profile_mode"):
        profile_mode = cfg.training.perf.profile_mode
        model_args["profile_mode"] = profile_mode
    if enable_amp:
        model_args["amp_mode"] = enable_amp

    # Select Model Architecture
    if cfg.model.name == "regression":
        # Input to UNet is cat(zero_input, img_lr).
        # The UNet adds grid embeddings internally.
        model = UNet(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
    elif cfg.model.name in ["lt_aware_ce_regression", "lt_aware_regression"]:
        # Input is cat(zero_input, img_lr).
        # The UNet adds grid and lead time embeddings internally.
        model = UNet(
            img_in_channels=img_in_channels
            + model_args["N_grid_channels"]
            + model_args["lead_time_channels"],
            **model_args,
        )
    elif cfg.model.name == "lt_aware_patched_diffusion":
        # Input to UNet is cat(latent, y_cond).
        # The UNet adds grid and lead time embeddings internally.
        model = EDMPrecondSuperResolution(
            img_in_channels=img_in_channels
            + model_args["N_grid_channels"]
            + model_args["lead_time_channels"],
            **model_args,
        )
    elif cfg.model.name in ["diffusion", "patched_diffusion"]:
        # Input to UNet is cat(latent, y_cond).
        # The UNet adds grid embeddings internally.
        model = EDMPrecondSuperResolution(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
    else:
        raise ValueError(f"Invalid model: {cfg.model.name}")

    is_diffusion_model = cfg.model.name in {
        "diffusion",
        "patched_diffusion",
        "lt_aware_patched_diffusion",
    }

    model.train().requires_grad_(True).to(dist.device)

    if use_apex_gn:
        model.to(memory_format=torch.channels_last)

    # Distributed Data Parallel
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
            bucket_cap_mb=35,
            gradient_as_bucket_view=True,
        )
        model._set_static_graph()

    if cfg.wandb.watch_model and dist.rank == 0:
        wandb.watch(model)

    try:
        load_checkpoint(
            path=checkpoint_dir,
            models=model,
            epoch=debug_cfg["epoch"],
        )
    except Exception:
        pass

    # Load Regression Net if needed
    if (
        hasattr(cfg.training.io, "regression_checkpoint_path")
        and cfg.training.io.regression_checkpoint_path is not None
    ):
        regression_checkpoint_path = to_absolute_path(
            cfg.training.io.regression_checkpoint_path
        )
        if not os.path.exists(regression_checkpoint_path):
            raise FileNotFoundError(
                f"Expected this regression checkpoint but not found: {regression_checkpoint_path}"
            )
        reg_model_args = {
            "use_apex_gn": use_apex_gn,
            "profile_mode": profile_mode,
            "amp_mode": enable_amp,
        }
        regression_net = Module.from_checkpoint(
            regression_checkpoint_path, reg_model_args
        )
        regression_net.eval().requires_grad_(False).to(dist.device)
        if use_apex_gn:
            regression_net.to(memory_format=torch.channels_last)
        logger0.success("Loaded the pre-trained regression model")
    else:
        regression_net = None

    if is_diffusion_model and regression_net is not None and loss_mask_tensor is not None:
        regression_net = _MaskedRegressionNet(regression_net, loss_mask_tensor)

    if use_torch_compile:
        model = torch.compile(model)
        if regression_net:
            regression_net = torch.compile(regression_net)

    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.training.hp.total_batch_size,
        cfg.training.hp.batch_size_per_gpu,
        dist.world_size,
    )
    batch_size_per_gpu = cfg.training.hp.batch_size_per_gpu
    logger0.info(f"Using {num_accumulation_rounds} gradient accumulation rounds")
    apply_loss_mask_to_inputs = bool(
        cfg.training.perf.get("apply_loss_mask_to_inputs", True)
    )
    if is_diffusion_model and loss_mask_tensor is not None:
        logger0.info(
            f"apply_loss_mask_to_inputs={apply_loss_mask_to_inputs}"
            " (disable to avoid redundant per-step masking overhead)"
        )
    if debug_enabled and dist.rank == 0:
        logger0.info(
            f"Debug plotting enabled: samples={debug_cfg['num_samples']} out_dir={debug_out_dir}"
        )

    # Patch settings
    patch_num = getattr(cfg.training.hp, "patch_num", 1)
    if hasattr(cfg.training.hp, "max_patch_per_gpu"):
        max_patch_per_gpu = cfg.training.hp.max_patch_per_gpu
        if max_patch_per_gpu // batch_size_per_gpu < 1:
            raise ValueError(f"max_patch_per_gpu must be >= batch_size_per_gpu.")
        max_patch_num_per_iter = min(patch_num, (max_patch_per_gpu // batch_size_per_gpu))
        patch_iterations = (patch_num + max_patch_num_per_iter - 1) // max_patch_num_per_iter
        patch_nums_iter = [
            min(max_patch_num_per_iter, patch_num - i * max_patch_num_per_iter)
            for i in range(patch_iterations)
        ]
    else:
        patch_nums_iter = [patch_num]

    # Patch Gradient Accumulation
    if cfg.model.name in {"patched_diffusion", "lt_aware_patched_diffusion"}:
        if len(patch_nums_iter) > 1 and patching:
            use_patch_grad_acc = True
        else:
            use_patch_grad_acc = False
    else:
        use_patch_grad_acc = None

    # Instantiate Loss Function
    if cfg.model.name in ("diffusion", "patched_diffusion", "lt_aware_patched_diffusion"):
        # CRITICAL: sigma_data must match between loss function and model preconditioning
        # Get sigma_data from config, default to 0.5 if not specified (EDM paper default)
        loss_sigma_data = cfg.model.get("model_args", {}).get("sigma_data", 0.5)

        # Get noise schedule parameters from config (with sensible defaults)
        # P_mean, P_std control the log-normal distribution of noise levels
        # Default P_std=1.2 creates a wide range of noise levels
        # Smaller P_std (e.g., 1.0) focuses training on moderate noise levels
        loss_P_mean = cfg.training.hp.get("P_mean", 0.0)
        loss_P_std = cfg.training.hp.get("P_std", 1.2)

        # Stability parameters for loss clipping (IMPROVED)
        # loss_clip_max: Maximum per-pixel loss value (None = no clipping)
        # weight_clip_max: Maximum EDM loss weight (prevents extreme values at low noise)
        loss_clip_max = cfg.training.hp.get("loss_clip_max", None)
        weight_clip_max = cfg.training.hp.get("weight_clip_max", 100.0)

        logger0.info(f"Using sigma_data={loss_sigma_data} for diffusion loss (must match model)")
        logger0.info(f"Using noise schedule: P_mean={loss_P_mean}, P_std={loss_P_std}")
        logger0.info(f"Loss stability: loss_clip_max={loss_clip_max}, weight_clip_max={weight_clip_max}")

        if loss_mask_tensor is not None:
            loss_fn = MaskedResidualLoss(
                regression_net=regression_net,
                hr_mean_conditioning=cfg.model.hr_mean_conditioning,
                sigma_data=loss_sigma_data,
                mask=loss_mask_tensor,
                residual_std=residual_std_tensor,
                P_mean=loss_P_mean,
                P_std=loss_P_std,
                loss_clip_max=loss_clip_max,
                weight_clip_max=weight_clip_max,
                precip_threshold_info=precip_threshold_info,
            )
            logger0.info("Using MaskedResidualLoss (loss mask applied inside loss).")
        else:
            loss_fn = ResidualLoss(
                regression_net=regression_net,
                hr_mean_conditioning=cfg.model.hr_mean_conditioning,
                sigma_data=loss_sigma_data,
                P_mean=loss_P_mean,
                P_std=loss_P_std,
            )
    elif cfg.model.name in ("regression", "lt_aware_regression"):
        use_rain_aware_loss = bool(cfg.training.hp.get("use_rain_aware_loss", False))
        if use_rain_aware_loss and precip_threshold_info is not None:
            dry_false_positive_weight = float(
                cfg.training.hp.get("rain_dry_false_positive_weight", 4.0)
            )
            occurrence_bce_weight = float(
                cfg.training.hp.get("rain_occurrence_bce_weight", 0.0)
            )
            occurrence_logit_scale = float(
                cfg.training.hp.get("rain_occurrence_logit_scale", 12.0)
            )
            loss_fn = RainAwareRegressionLoss(
                precip_threshold_info=precip_threshold_info,
                dry_false_positive_weight=dry_false_positive_weight,
                occurrence_bce_weight=occurrence_bce_weight,
                occurrence_logit_scale=occurrence_logit_scale,
            )
            logger0.info(
                "Using RainAwareRegressionLoss "
                f"(dry_false_positive_weight={dry_false_positive_weight}, "
                f"occurrence_bce_weight={occurrence_bce_weight}, "
                f"occurrence_logit_scale={occurrence_logit_scale})."
            )
        else:
            if use_rain_aware_loss and precip_threshold_info is None:
                logger0.warning(
                    "use_rain_aware_loss=true but precipitation metadata is unavailable. "
                    "Falling back to RegressionLoss."
                )
            loss_fn = RegressionLoss()
    elif cfg.model.name == "lt_aware_ce_regression":
        loss_fn = RegressionLossCE(prob_channels=prob_channels)
    else:
        raise ValueError(f"Unknown model name for loss: {cfg.model.name}")

    # Loss Compatibility Layer
    _loss_fn_params = set(inspect.signature(loss_fn.__call__).parameters.keys())

    def _call_loss(loss_kwargs: dict):
        # We filter args here so we can blindly pass 'mask' or 'patching' 
        # without crashing if the loss doesn't support them.
        return loss_fn(**{k: v for k, v in loss_kwargs.items() if k in _loss_fn_params})

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.training.hp.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
        fused=True,
    )

    # =========================================================================
    # EMA (Exponential Moving Average) Setup
    # =========================================================================
    # EMA maintains smoothed weights for better inference quality.
    # Standard practice for diffusion models.
    ema_enabled = cfg.training.hp.get("ema_enabled", True)  # Enable by default for diffusion
    ema_decay = cfg.training.hp.get("ema_decay", 0.9999)
    ema_warmup = cfg.training.hp.get("ema_warmup", True)
    ema_save_freq = cfg.training.io.get("ema_save_freq", cfg.training.io.save_checkpoint_freq)

    if ema_enabled and is_diffusion_model:
        # Get the underlying model (unwrap DDP if necessary)
        model_for_ema = model.module if hasattr(model, 'module') else model
        ema = EMA(model_for_ema, decay=ema_decay, warmup=ema_warmup)
        logger0.info(f"EMA enabled: decay={ema_decay}, warmup={ema_warmup}")
    else:
        ema = None
        if is_diffusion_model:
            logger0.info("EMA disabled (set training.hp.ema_enabled=true to enable)")

    start_time = time.time()

    if dist.world_size > 1:
        torch.distributed.barrier()
    try:
        load_checkpoint(
            path=checkpoint_dir,
            optimizer=optimizer,
            device=dist.device,
            epoch=debug_cfg["epoch"],
        )
        # Try to load EMA checkpoint if it exists
        if ema is not None:
            ema_checkpoint_path = os.path.join(checkpoint_dir, f"ema.{cur_nimg}.pt")
            if os.path.exists(ema_checkpoint_path):
                ema_state = torch.load(ema_checkpoint_path, map_location=dist.device)
                ema.load_state_dict(ema_state)
                logger0.info(f"Loaded EMA checkpoint from {ema_checkpoint_path}")
    except Exception:
        pass

    ############################################################################
    #                            MAIN TRAINING LOOP                            #
    ############################################################################

    logger0.info(f"Training for {cfg.training.hp.training_duration} images...")
    done = False
    average_loss_running_mean = 0.0
    n_average_loss_running_mean = 1
    start_nimg = cur_nimg
    input_dtype = torch.float32
    if enable_amp:
        input_dtype = torch.float32
    elif fp16:
        input_dtype = torch.float16

    with cuda_profiler():
        with profiler_emit_nvtx():

            while not done:
                tick_start_nimg = cur_nimg
                tick_start_time = time.time()

                if cur_nimg - start_nimg == 24 * cfg.training.hp.total_batch_size:
                    cuda_profiler_start()
                if cur_nimg - start_nimg == 25 * cfg.training.hp.total_batch_size:
                    cuda_profiler_stop()

                with nvtx.annotate("Training iteration", color="green"):
                    optimizer.zero_grad(set_to_none=True)
                    loss_accum = None

                    for n_i in range(num_accumulation_rounds):
                        with nvtx.annotate(f"accumulation round {n_i}", color="Magenta"):
                            with nvtx.annotate("loading data", color="green"):
                                img_clean, img_lr, *lead_time_label = next(dataset_iterator)

                            if use_apex_gn:
                                img_clean = img_clean.to(dist.device, dtype=input_dtype, non_blocking=True).to(memory_format=torch.channels_last)
                                img_lr = img_lr.to(dist.device, dtype=input_dtype, non_blocking=True).to(memory_format=torch.channels_last)
                            else:
                                img_clean = img_clean.to(dist.device).to(input_dtype).contiguous()
                                img_lr = img_lr.to(dist.device).to(input_dtype).contiguous()

                                # Ensure stability (0.0 for Ocean is preserved)
                                img_clean = _nan_to_num_(img_clean)
                                img_lr = _nan_to_num_(img_lr)

                            if (
                                is_diffusion_model
                                and loss_mask_tensor is not None
                                and apply_loss_mask_to_inputs
                            ):
                                img_clean = _apply_loss_mask(img_clean, loss_mask_tensor)
                                img_lr = _apply_loss_mask(img_lr, loss_mask_tensor)

                            if (
                                debug_enabled
                                and not debug_done
                                and dist.rank == 0
                                and n_i == 0
                            ):
                                debug_sample = _get_debug_sample(
                                    dataset=dataset,
                                    debug_index=debug_cfg["index"],
                                    debug_time=debug_cfg["time"],
                                    logger=logger0,
                                )
                                if debug_sample is not None:
                                    dbg_clean, dbg_lr, dbg_lead = debug_sample
                                    if use_apex_gn:
                                        dbg_clean = dbg_clean.to(dist.device, dtype=input_dtype, non_blocking=True).to(
                                            memory_format=torch.channels_last
                                        )
                                        dbg_lr = dbg_lr.to(dist.device, dtype=input_dtype, non_blocking=True).to(
                                            memory_format=torch.channels_last
                                        )
                                    else:
                                        dbg_clean = dbg_clean.to(dist.device).to(input_dtype).contiguous()
                                        dbg_lr = dbg_lr.to(dist.device).to(input_dtype).contiguous()

                                    dbg_clean = _nan_to_num_(dbg_clean)
                                    dbg_lr = _nan_to_num_(dbg_lr)
                                    if (
                                        is_diffusion_model
                                        and loss_mask_tensor is not None
                                        and apply_loss_mask_to_inputs
                                    ):
                                        dbg_clean = _apply_loss_mask(dbg_clean, loss_mask_tensor)
                                        dbg_lr = _apply_loss_mask(dbg_lr, loss_mask_tensor)

                                    debug_img_clean = dbg_clean
                                    debug_img_lr = dbg_lr
                                    debug_lead = dbg_lead.to(dist.device).contiguous() if dbg_lead is not None else None
                                else:
                                    debug_img_clean = img_clean
                                    debug_img_lr = img_lr
                                    debug_lead = lead_time_label[0].to(dist.device).contiguous() if lead_time_label else None

                                debug_mode = (
                                    "diffusion"
                                    if cfg.model.name
                                    in (
                                        "diffusion",
                                        "patched_diffusion",
                                        "lt_aware_patched_diffusion",
                                    )
                                    else "regression"
                                )
                                _run_debug_plots(
                                    mode=debug_mode,
                                    out_dir=debug_out_dir,
                                    model=model,
                                    regression_net=regression_net,
                                    loss_fn=loss_fn,
                                    img_clean=debug_img_clean,
                                    img_lr=debug_img_lr,
                                    lead_time_label=debug_lead,
                                    input_vars=getattr(dataset, "input_variables", []),
                                    output_vars=getattr(dataset, "output_variables", []),
                                    loss_mask_tensor=loss_mask_tensor,
                                    precip_threshold_info=precip_threshold_info,
                                    enable_amp=enable_amp,
                                    amp_dtype=amp_dtype,
                                    seed=debug_cfg["seed"],
                                    num_samples=debug_cfg["num_samples"],
                                    sigma_override=debug_cfg["sigma"],
                                    noise_seed=debug_cfg["noise_seed"],
                                    logger=logger0,
                                )
                                debug_done = True

                            loss_fn_kwargs = {
                                "net": model,
                                "img_clean": img_clean,
                                "img_lr": img_lr,
                                "augment_pipe": None,
                            }
                            if use_patch_grad_acc is not None:
                                loss_fn_kwargs["use_patch_grad_acc"] = use_patch_grad_acc

                            if lead_time_label:
                                loss_fn_kwargs["lead_time_label"] = lead_time_label[0].to(dist.device).contiguous()
                            
                            if use_patch_grad_acc:
                                loss_fn.y_mean = None
                            
                            # PASS MASK TO LOSS FUNCTION (If supported by loss)
                            if loss_mask_tensor is not None:
                                loss_fn_kwargs["mask"] = loss_mask_tensor

                            for patch_num_per_iter in patch_nums_iter:
                                if patching is not None:
                                    patching.set_patch_num(patch_num_per_iter)
                                    loss_fn_kwargs["patching"] = patching

                                with nvtx.annotate("loss forward", color="green"):
                                    with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                                        loss = _call_loss(loss_fn_kwargs)

                                # --- MASKING IS NOW HANDLED INSIDE MaskedResidualLoss ---
                                # DO NOT apply mask manually here - MaskedResidualLoss already:
                                # 1. Applies the mask internally (zeroes ocean pixels)
                                # 2. Scales the loss to account for masked pixels
                                # Double-masking breaks the loss normalization and causes high loss values!
                                #
                                # NOTE: If using standard ResidualLoss (no mask), you may need manual masking,
                                # but this is NOT recommended for datasets with ocean/NaN regions.

                                loss = loss.sum() / batch_size_per_gpu
                                scaled = loss / num_accumulation_rounds / len(patch_nums_iter)
                                loss_accum = scaled if loss_accum is None else (loss_accum + scaled)

                                with nvtx.annotate("loss backward", color="yellow"):
                                    loss.backward()

                    with nvtx.annotate("loss aggregate", color="green"):
                        loss_sum = loss_accum.detach().reshape(1)
                        if dist.world_size > 1:
                            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
                        average_loss = (loss_sum / dist.world_size).cpu().item()

                    average_loss_running_mean += (average_loss - average_loss_running_mean) / n_average_loss_running_mean
                    n_average_loss_running_mean += 1

                    if dist.rank == 0 and writer is not None:
                        writer.add_scalar("training_loss", average_loss, cur_nimg)
                        writer.add_scalar("training_loss_running_mean", average_loss_running_mean, cur_nimg)

                    ptt = is_time_for_periodic_task(
                        cur_nimg, cfg.training.io.print_progress_freq, done, cfg.training.hp.total_batch_size, dist.rank, rank_0_only=True,
                    )
                    if ptt:
                        average_loss_running_mean = 0.0
                        n_average_loss_running_mean = 1

                    with nvtx.annotate("update weights", color="blue"):
                        lr_rampup = cfg.training.hp.lr_rampup
                        for g in optimizer.param_groups:
                            if lr_rampup > 0:
                                g["lr"] = cfg.training.hp.lr * min(cur_nimg / lr_rampup, 1)
                            if cur_nimg >= lr_rampup:
                                g["lr"] *= cfg.training.hp.lr_decay ** ((cur_nimg - lr_rampup) // cfg.training.hp.lr_decay_rate)
                            current_lr = g["lr"]
                            if dist.rank == 0 and writer is not None:
                                writer.add_scalar("learning_rate", current_lr, cur_nimg)

                        handle_and_clip_gradients(model, grad_clip_threshold=cfg.training.hp.grad_clip_threshold)

                    with nvtx.annotate("optimizer step", color="blue"):
                        optimizer.step()

                    # Update EMA weights after optimizer step
                    if ema is not None:
                        model_for_ema = model.module if hasattr(model, 'module') else model
                        ema.update(model_for_ema)

                    cur_nimg += cfg.training.hp.total_batch_size
                    done = cur_nimg >= cfg.training.hp.training_duration

                with nvtx.annotate("validation", color="red"):
                    if validation_dataset_iterator is not None:
                        if is_time_for_periodic_task(
                            cur_nimg, cfg.training.io.validation_freq, done, cfg.training.hp.total_batch_size, dist.rank,
                        ):
                            valid_loss_accum = 0.0
                            with torch.no_grad():
                                for _ in range(cfg.training.io.validation_steps):
                                    img_clean_valid, img_lr_valid, *lead_time_label_valid = next(validation_dataset_iterator)

                                    if use_apex_gn:
                                        img_clean_valid = img_clean_valid.to(dist.device, dtype=input_dtype, non_blocking=True).to(memory_format=torch.channels_last)
                                        img_lr_valid = img_lr_valid.to(dist.device, dtype=input_dtype, non_blocking=True).to(memory_format=torch.channels_last)
                                    else:
                                        img_clean_valid = img_clean_valid.to(dist.device).to(input_dtype).contiguous()
                                        img_lr_valid = img_lr_valid.to(dist.device).to(input_dtype).contiguous()

                                    img_clean_valid = _nan_to_num_(img_clean_valid)
                                    img_lr_valid = _nan_to_num_(img_lr_valid)

                                    if (
                                        is_diffusion_model
                                        and loss_mask_tensor is not None
                                        and apply_loss_mask_to_inputs
                                    ):
                                        img_clean_valid = _apply_loss_mask(img_clean_valid, loss_mask_tensor)
                                        img_lr_valid = _apply_loss_mask(img_lr_valid, loss_mask_tensor)

                                    loss_valid_kwargs = {
                                        "net": model,
                                        "img_clean": img_clean_valid,
                                        "img_lr": img_lr_valid,
                                        "augment_pipe": None,
                                    }
                                    if use_patch_grad_acc is not None:
                                        loss_valid_kwargs["use_patch_grad_acc"] = use_patch_grad_acc

                                    if lead_time_label_valid:
                                        loss_valid_kwargs["lead_time_label"] = lead_time_label_valid[0].to(dist.device).contiguous()

                                    if use_patch_grad_acc:
                                        loss_fn.y_mean = None
                                    
                                    # PASS MASK TO VALIDATION TOO
                                    if loss_mask_tensor is not None:
                                        loss_valid_kwargs["mask"] = loss_mask_tensor

                                    for patch_num_per_iter in patch_nums_iter:
                                        if patching is not None:
                                            patching.set_patch_num(patch_num_per_iter)
                                            loss_valid_kwargs["patching"] = patching

                                        with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                                            loss_valid = _call_loss(loss_valid_kwargs)

                                        # MASKING IS HANDLED INSIDE MaskedResidualLoss - DO NOT DOUBLE-MASK

                                        loss_valid = (loss_valid.sum() / batch_size_per_gpu).cpu().item()
                                        valid_loss_accum += loss_valid / cfg.training.io.validation_steps

                            valid_loss_sum = torch.tensor([valid_loss_accum], device=dist.device)
                            if dist.world_size > 1:
                                torch.distributed.all_reduce(valid_loss_sum, op=torch.distributed.ReduceOp.SUM)
                            average_valid_loss = (valid_loss_sum / dist.world_size).cpu().item()

                            if dist.rank == 0 and writer is not None:
                                writer.add_scalar("validation_loss", average_valid_loss, cur_nimg)

                if is_time_for_periodic_task(
                    cur_nimg, cfg.training.io.print_progress_freq, done, cfg.training.hp.total_batch_size, dist.rank, rank_0_only=True,
                ):
                    tick_end_time = time.time()
                    fields = []
                    fields += [f"samples {cur_nimg:<9.1f}"]
                    fields += [f"training_loss {average_loss:<7.2f}"]
                    fields += [f"training_loss_running_mean {average_loss_running_mean:<7.2f}"]
                    fields += [f"learning_rate {current_lr:<7.8f}"]
                    fields += [f"total_sec {(tick_end_time - start_time):<7.1f}"]
                    fields += [f"sec_per_tick {(tick_end_time - tick_start_time):<7.1f}"]
                    fields += [f"sec_per_sample {((tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg)):<7.2f}"]
                    fields += [f"cpu_mem_gb {(psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
                    if torch.cuda.is_available():
                        fields += [f"peak_gpu_mem_gb {(torch.cuda.max_memory_allocated(dist.device) / 2**30):<6.2f}"]
                        fields += [f"peak_gpu_mem_reserved_gb {(torch.cuda.max_memory_reserved(dist.device) / 2**30):<6.2f}"]
                        torch.cuda.reset_peak_memory_stats()
                    logger0.info(" ".join(fields))

                should_save_ckpt = is_time_for_periodic_task(
                    cur_nimg,
                    cfg.training.io.save_checkpoint_freq,
                    done,
                    cfg.training.hp.total_batch_size,
                    dist.rank,
                    rank_0_only=False,
                )
                if (
                    should_save_ckpt
                    and not debug_no_ckpt
                    and dist.world_size > 1
                ):
                    torch.distributed.barrier()
                if should_save_ckpt and dist.rank == 0 and not debug_no_ckpt:
                    save_checkpoint(path=checkpoint_dir, models=model, optimizer=optimizer, epoch=cur_nimg)

                    # Save EMA checkpoint alongside regular checkpoint
                    if ema is not None:
                        ema_checkpoint_path = os.path.join(checkpoint_dir, f"ema.{cur_nimg}.pt")
                        ema.save(ema_checkpoint_path)
                        logger0.info(f"Saved EMA checkpoint: {ema_checkpoint_path}")

            if cfg.training.io.save_n_recent_checkpoints > 0 and not debug_no_ckpt:
                keep_n = cfg.training.io.save_n_recent_checkpoints

                # Keep only the most recent model state dictionaries (.mdlus).
                model_ckpts = checkpoint_list(checkpoint_dir, suffix=".mdlus")
                while len(model_ckpts) > keep_n:
                    os.remove(os.path.join(checkpoint_dir, model_ckpts[0]))
                    model_ckpts = model_ckpts[1:]

                # Keep only the most recent training checkpoints (optimizer/state).
                state_ckpts = checkpoint_list(checkpoint_dir, suffix=".pt")
                state_ckpts = [f for f in state_ckpts if f.startswith("checkpoint.")]
                while len(state_ckpts) > keep_n:
                    os.remove(os.path.join(checkpoint_dir, state_ckpts[0]))
                    state_ckpts = state_ckpts[1:]

                # Keep EMA checkpoints separately from regular training checkpoints.
                if ema is not None:
                    ema_ckpts = checkpoint_list(checkpoint_dir, suffix=".pt")
                    ema_ckpts = [f for f in ema_ckpts if f.startswith("ema.")]
                    while len(ema_ckpts) > keep_n:
                        os.remove(os.path.join(checkpoint_dir, ema_ckpts[0]))
                        ema_ckpts = ema_ckpts[1:]

    logger0.info("Training Completed.")


if __name__ == "__main__":
    main()
