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

"""
ERA5 -> CONUS404 downscaling dataset for training regression and diffusion models.

Expected zarr layout for both input and target:
  - Root arrays: lat, lon, time, and meteorological variables.
  - Variable arrays shape: (time, y, x)
  - Time array dtype: int64 nanoseconds since Unix epoch.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr
import zarr
from scipy import ndimage
from tqdm import tqdm

from physicsnemo.distributed import DistributedManager

from models.datasets.base import ChannelMetadata, DownscalingDataset
from models.datasets.quality import check_legacy_arrays, runtime_cache_path

logger = logging.getLogger(__name__)


# Variables that need special handling
PRECIP_VARS = ["PREC_ACC_NC"]
POSITIVE_VARS = [
    "PREC_ACC_NC",
    "SWDOWN",
    "GLW",
    "Q2",
    "q_925",
    "q_850",
    "q_700",
    "q_500",
    "r_925",
    "r_850",
    "r_700",
    "r_500",
    "cc_925",
    "cc_850",
    "cc_700",
    "cc_500",
]

# Values below this are set to zero for precipitation channels.
PRECIP_THRESHOLD = 0.1

# Unit harmonization defaults (match WRF pipeline convention):
# - ERA5 PSFC appears to be in hPa already.
# - CONUS404 PSFC appears to be in Pa; convert target to hPa.
DEFAULT_INPUT_SCALE_FACTORS: Dict[str, float] = {}
DEFAULT_OUTPUT_SCALE_FACTORS: Dict[str, float] = {"PSFC": 0.01}


class DataFilters:
    """Collection of data filtering functions used in preprocessing and visualization."""

    @staticmethod
    def apply_scale_factors(
        data: np.ndarray,
        variables: List[str],
        scale_dict: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        if not scale_dict:
            return data

        result = data.copy()
        for i, var in enumerate(variables):
            if var in scale_dict:
                scale = scale_dict[var]
                if result.ndim == 3:
                    result[i] = result[i] * scale
                elif result.ndim == 4:
                    result[:, i] = result[:, i] * scale
        return result

    @staticmethod
    def apply_precip_threshold(
        data: np.ndarray,
        variables: List[str],
        threshold: float = PRECIP_THRESHOLD,
        precip_vars: Optional[List[str]] = None,
    ) -> np.ndarray:
        if precip_vars is None:
            precip_vars = PRECIP_VARS

        result = data.copy()
        for i, var in enumerate(variables):
            if var in precip_vars:
                if result.ndim == 3:
                    result[i] = np.where(result[i] < threshold, 0.0, result[i])
                elif result.ndim == 4:
                    result[:, i] = np.where(result[:, i] < threshold, 0.0, result[:, i])
        return result

    @staticmethod
    def clip_negative_values(
        data: np.ndarray,
        variables: List[str],
        clip_vars: Optional[List[str]] = None,
    ) -> np.ndarray:
        if clip_vars is None:
            clip_vars = POSITIVE_VARS

        result = data.copy()
        for i, var in enumerate(variables):
            if var in clip_vars:
                if result.ndim == 3:
                    result[i] = np.maximum(result[i], 0)
                elif result.ndim == 4:
                    result[:, i] = np.maximum(result[:, i], 0)
        return result


def ns_epoch_to_datetime(time_val: int) -> datetime:
    """Convert int64 nanoseconds since epoch to datetime."""
    return datetime.utcfromtimestamp(int(time_val) / 1e9)


def get_time_indices_for_years(
    time_values: np.ndarray,
    start_year: int,
    end_year: int,
) -> np.ndarray:
    """Get indices of timesteps within the specified inclusive year range."""
    time_dt64 = time_values.astype("datetime64[ns]")
    start_dt64 = np.datetime64(f"{start_year}-01-01T00:00:00")
    end_dt64 = np.datetime64(f"{end_year + 1}-01-01T00:00:00")
    return np.where((time_dt64 >= start_dt64) & (time_dt64 < end_dt64))[0]


class ERA5CONUS404DownscalingDataset(DownscalingDataset):
    """
    Dataset for downscaling ERA5 4km inputs to CONUS404 4km targets.
    """

    def __init__(
        self,
        input_data_path: str,
        target_data_path: str,
        stats_path: str,
        train: bool = True,
        train_years: Union[Tuple[int, int], List[int]] = (2000, 2011),
        valid_years: Union[Tuple[int, int], List[int]] = (2012, 2012),
        apply_scale_factors: bool = True,
        input_scale_factors: Optional[Dict[str, float]] = None,
        output_scale_factors: Optional[Dict[str, float]] = None,
        clip_negative: bool = True,
        precip_threshold: float = 0.1,
        input_variables: Optional[List[str]] = None,
        output_variables: Optional[List[str]] = None,
        landmask_path: Optional[str] = None,
        mask_ocean_in_target: bool = True,
        sanitize_inputs: bool = True,
        keep_ocean_as_nan: bool = False,
        drop_nan_land_timesteps: bool = True,
        nan_check_vars: Optional[List[str]] = None,
        nan_chunk: int = 128,
        debug_nan: bool = False,
        debug_nan_limit: int = 10,
        coastal_buffer_pixels: int = 0,
        crop_to_multiple: int = 32,
        **kwargs,
    ):
        super().__init__()

        self.input_data_path = input_data_path
        self.target_data_path = target_data_path
        self.landmask_path = landmask_path
        self.mask_ocean_in_target = mask_ocean_in_target
        self.sanitize_inputs = sanitize_inputs
        self.keep_ocean_as_nan = keep_ocean_as_nan
        self.drop_nan_land_timesteps = drop_nan_land_timesteps
        self.nan_check_vars = nan_check_vars
        self.nan_chunk = int(nan_chunk)
        self.debug_nan = debug_nan
        self.debug_nan_limit = int(debug_nan_limit)
        self.coastal_buffer_pixels = int(coastal_buffer_pixels)
        self.crop_to_multiple = int(crop_to_multiple) if crop_to_multiple else 0

        self.land_mask = None
        self.ocean_mask = None
        self.ocean_mask_with_buffer = None
        self.loss_mask = None

        self.train = train
        self.train_years = tuple(train_years)
        self.valid_years = tuple(valid_years)
        self.apply_scale_factors = apply_scale_factors
        self.input_scale_factors = (
            DEFAULT_INPUT_SCALE_FACTORS
            if input_scale_factors is None
            else dict(input_scale_factors)
        )
        self.output_scale_factors = (
            DEFAULT_OUTPUT_SCALE_FACTORS
            if output_scale_factors is None
            else dict(output_scale_factors)
        )
        self.clip_negative = clip_negative
        self.precip_threshold = precip_threshold

        logger.info(f"Loading input data from: {input_data_path}")
        logger.info(f"Loading target data from: {target_data_path}")

        self.input_store = zarr.open(input_data_path, mode="r")
        self.target_store = zarr.open(target_data_path, mode="r")

        self._all_input_vars = self._discover_variable_list(self.input_store)
        self._all_target_vars = self._discover_variable_list(self.target_store)

        self.input_variables = input_variables if input_variables else self._all_input_vars
        self.output_variables = output_variables if output_variables else self._all_target_vars

        self._validate_variable_names()

        self.input_var_arrays = {v: self.input_store[v] for v in self.input_variables}
        self.target_var_arrays = {v: self.target_store[v] for v in self.output_variables}

        self._lat = self.target_store["lat"][:]
        self._lon = self.target_store["lon"][:]

        raw_y, raw_x = self._lat.shape
        if self.crop_to_multiple > 1:
            self.img_shape_y = raw_y - (raw_y % self.crop_to_multiple)
            self.img_shape_x = raw_x - (raw_x % self.crop_to_multiple)
            if self.img_shape_y == 0 or self.img_shape_x == 0:
                raise ValueError(
                    f"Invalid crop_to_multiple={self.crop_to_multiple} for shape {raw_y}x{raw_x}"
                )
            if self.img_shape_y != raw_y or self.img_shape_x != raw_x:
                logger.info(
                    "Cropping domain from %dx%d to %dx%d for model compatibility.",
                    raw_y,
                    raw_x,
                    self.img_shape_y,
                    self.img_shape_x,
                )
        else:
            self.img_shape_y = raw_y
            self.img_shape_x = raw_x

        self._lat = self._lat[: self.img_shape_y, : self.img_shape_x]
        self._lon = self._lon[: self.img_shape_y, : self.img_shape_x]

        self._flip_lat = False
        lat_1d = self._lat[:, 0] if self._lat.ndim == 2 else self._lat
        if len(lat_1d) > 1 and lat_1d[0] > lat_1d[-1]:
            self._flip_lat = True
            self._lat = np.flipud(self._lat)
            self._lon = np.flipud(self._lon)
            logger.info("Detected decreasing latitude: enabling y-axis flip.")

        self._load_time_indices()
        self._init_land_ocean_mask()

        if self.drop_nan_land_timesteps:
            self._filter_bad_timesteps_strict()

        with open(stats_path, "r", encoding="utf-8") as f:
            self.stats = json.load(f)

        self._input_channel_metadata = self._create_channel_metadata(self.input_variables)
        self._output_channel_metadata = self._create_channel_metadata(self.output_variables)

        logger.info("Dataset initialized with %d samples", len(self))
        logger.info("Input channels: %d", len(self.input_variables))
        logger.info("Output channels: %d", len(self.output_variables))
        logger.info("Image shape: %s", self.image_shape())

    def _discover_variable_list(self, store: zarr.Group) -> List[str]:
        exclude = {"lat", "lon", "time"}
        return [k for k in store.array_keys() if k not in exclude]

    def _validate_variable_names(self) -> None:
        missing_input = [v for v in self.input_variables if v not in self._all_input_vars]
        missing_output = [v for v in self.output_variables if v not in self._all_target_vars]
        if missing_input:
            raise ValueError(f"Missing input variables in input zarr: {missing_input}")
        if missing_output:
            raise ValueError(f"Missing output variables in target zarr: {missing_output}")

    def _load_time_indices(self) -> None:
        """Build aligned time index pairs between input and target for train/valid years."""
        input_time = self.input_store["time"][:].astype(np.int64)
        target_time = self.target_store["time"][:].astype(np.int64)

        if self.train:
            year_start, year_end = self.train_years
        else:
            year_start, year_end = self.valid_years

        input_indices = get_time_indices_for_years(input_time, year_start, year_end)
        target_indices = get_time_indices_for_years(target_time, year_start, year_end)

        input_time_sel = input_time[input_indices]
        target_time_sel = target_time[target_indices]
        common_times = np.intersect1d(input_time_sel, target_time_sel)

        input_map = {int(t): int(i) for i, t in enumerate(input_time)}
        target_map = {int(t): int(i) for i, t in enumerate(target_time)}

        self._input_time_indices = np.array(
            [input_map[int(t)] for t in common_times], dtype=np.int64
        )
        self._target_time_indices = np.array(
            [target_map[int(t)] for t in common_times], dtype=np.int64
        )
        self._time_values = common_times.astype(np.int64)

        logger.info(
            "%s samples: %d (common timestamps)",
            "Training" if self.train else "Validation",
            len(self._time_values),
        )
        logger.info("Year range: %d-%d", year_start, year_end)

    def _create_channel_metadata(self, variables: List[str]) -> List[ChannelMetadata]:
        return [ChannelMetadata(name=v, level="", auxiliary=False) for v in variables]

    def _init_land_ocean_mask(self) -> None:
        if not self.landmask_path:
            logger.info("No landmask_path provided. Skipping mask initialization.")
            return

        try:
            mask_ds = xr.open_dataset(self.landmask_path)
            if "LANDMASK" in mask_ds:
                full_mask = mask_ds["LANDMASK"].values.astype(bool)
            else:
                first_var = list(mask_ds.data_vars)[0]
                full_mask = mask_ds[first_var].values.astype(bool)

            cropped_mask = full_mask[: self.img_shape_y, : self.img_shape_x]

            if self._flip_lat:
                cropped_mask = np.flipud(cropped_mask)

            self.land_mask = cropped_mask
            self.ocean_mask = ~cropped_mask

            if self.coastal_buffer_pixels > 0:
                structure = ndimage.generate_binary_structure(2, 1)
                self.ocean_mask_with_buffer = ndimage.binary_dilation(
                    self.ocean_mask,
                    structure=structure,
                    iterations=self.coastal_buffer_pixels,
                )
                self.loss_mask = self.land_mask & ~self.ocean_mask_with_buffer
            else:
                self.ocean_mask_with_buffer = self.ocean_mask
                self.loss_mask = self.land_mask

            logger.info("Loaded and cropped land mask successfully.")
        except Exception as e:
            logger.error("Failed to load/process land mask from %s: %s", self.landmask_path, e)
            self.land_mask = None
            self.ocean_mask = None
            self.ocean_mask_with_buffer = None
            self.loss_mask = None

    def _find_precip_variable(self, variables: List[str]) -> Optional[str]:
        for preferred in PRECIP_VARS + ["RAIN_TOTAL", "APCP_surface"]:
            if preferred in variables:
                return preferred
        for var in variables:
            u = var.upper()
            if "PREC" in u or "RAIN" in u:
                return var
        return None

    def _filter_bad_timesteps_strict(self) -> None:
        """
        Scans precipitation channels over LAND in both input and target.
        Drops a timestep if NaN/negative values are found over land.
        """
        import time as time_module

        dist = DistributedManager()

        if self.land_mask is None or len(self._time_values) == 0:
            return

        years = self.train_years if self.train else self.valid_years
        target_name = Path(self.target_data_path).name
        cache_filename = (
            f"{self.input_data_path}.{target_name}.{years[0]}-{years[1]}.valid_indices.npz"
        )
        lock_filename = cache_filename + ".lock"

        cache_filename = runtime_cache_path(cache_filename)
        lock_filename = cache_filename + ".lock"
        if os.path.exists(cache_filename):
            logger.info("Rank %d: Loading filtered timestep cache from %s", dist.rank, cache_filename)
            with np.load(cache_filename) as data:
                self._input_time_indices = data["input_indices"]
                self._target_time_indices = data["target_indices"]
                self._time_values = data["values"]
            return

        input_precip_var = self._find_precip_variable(self.input_variables)
        target_precip_var = self._find_precip_variable(self.output_variables)
        if input_precip_var is None and target_precip_var is None:
            logger.info("No precipitation variable found; skipping strict filtering.")
            return

        if dist.rank == 0:
            try:
                with open(lock_filename, "w", encoding="utf-8") as f:
                    f.write(f"Filtering started at {time_module.time()}")

                num_workers = min(64, os.cpu_count() or 4)
                total_timesteps = len(self._time_values)
                valid_mask = np.ones(total_timesteps, dtype=bool)
                h = self.img_shape_y
                w = self.img_shape_x

                def check_timestep(i: int) -> Tuple[int, bool]:
                    try:
                        input_t = int(self._input_time_indices[i])
                        target_t = int(self._target_time_indices[i])

                        if target_precip_var is not None:
                            t_data = self.target_var_arrays[target_precip_var][target_t, :h, :w]
                            t_land = t_data[self.land_mask]
                            if np.any(np.isnan(t_land)) or np.any(t_land < 0):
                                return (i, True)

                        if input_precip_var is not None:
                            i_data = self.input_var_arrays[input_precip_var][input_t, :h, :w]
                            i_land = i_data[self.land_mask]
                            if np.any(np.isnan(i_land)) or np.any(i_land < 0):
                                return (i, True)

                        return (i, False)
                    except Exception:
                        return (i, True)

                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {
                        executor.submit(check_timestep, i): i for i in range(total_timesteps)
                    }
                    with tqdm(total=total_timesteps, desc=f"Filtering Data ({num_workers} workers)") as pbar:
                        for future in as_completed(futures):
                            idx, is_bad = future.result()
                            if is_bad:
                                valid_mask[idx] = False
                            pbar.update(1)

                dropped = int(np.sum(~valid_mask))
                self._input_time_indices = self._input_time_indices[valid_mask]
                self._target_time_indices = self._target_time_indices[valid_mask]
                self._time_values = self._time_values[valid_mask]
                logger.info(
                    "Filtering complete. Dropped %d timesteps. Remaining: %d",
                    dropped,
                    len(self._time_values),
                )

                np.savez(
                    cache_filename,
                    input_indices=self._input_time_indices,
                    target_indices=self._target_time_indices,
                    values=self._time_values,
                )
            finally:
                if os.path.exists(lock_filename):
                    os.remove(lock_filename)
        else:
            logger.info("Rank %d: Waiting for rank 0 filtering cache...", dist.rank)
            poll_interval = 10
            max_wait = 7200
            waited = 0

            while waited < max_wait:
                if os.path.exists(cache_filename) and not os.path.exists(lock_filename):
                    time_module.sleep(1)
                    break
                time_module.sleep(poll_interval)
                waited += poll_interval

            if not os.path.exists(cache_filename):
                raise RuntimeError(
                    f"Rank {dist.rank}: Timed out waiting for filtering cache after {max_wait}s"
                )

            with np.load(cache_filename) as data:
                self._input_time_indices = data["input_indices"]
                self._target_time_indices = data["target_indices"]
                self._time_values = data["values"]

        if dist.world_size > 1:
            torch.distributed.barrier()

    def __len__(self) -> int:
        return len(self._time_values)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_t = int(self._input_time_indices[idx])
        target_t = int(self._target_time_indices[idx])

        img_lr = np.stack(
            [
                self.input_var_arrays[var][
                    input_t, : self.img_shape_y, : self.img_shape_x
                ]
                for var in self.input_variables
            ],
            axis=0,
        ).astype(np.float32)
        img_hr = np.stack(
            [
                self.target_var_arrays[var][
                    target_t, : self.img_shape_y, : self.img_shape_x
                ]
                for var in self.output_variables
            ],
            axis=0,
        ).astype(np.float32)

        if self._flip_lat:
            img_lr = np.flip(img_lr, axis=1).copy()
            img_hr = np.flip(img_hr, axis=1).copy()

        check_legacy_arrays(self, img_lr, img_hr)
        if self.sanitize_inputs:
            img_lr = np.nan_to_num(img_lr, nan=0.0)
            img_hr = np.nan_to_num(img_hr, nan=0.0)

        if self.apply_scale_factors:
            img_lr = DataFilters.apply_scale_factors(
                img_lr, self.input_variables, self.input_scale_factors
            )
            img_hr = DataFilters.apply_scale_factors(
                img_hr, self.output_variables, self.output_scale_factors
            )

        if self.clip_negative:
            img_lr = DataFilters.clip_negative_values(img_lr, self.input_variables)
            img_hr = DataFilters.clip_negative_values(img_hr, self.output_variables)

        if self.precip_threshold > 0:
            img_lr = DataFilters.apply_precip_threshold(
                img_lr, self.input_variables, self.precip_threshold
            )
            img_hr = DataFilters.apply_precip_threshold(
                img_hr, self.output_variables, self.precip_threshold
            )

        img_lr = self._normalize(img_lr, self.input_variables, "input")
        img_hr = self._normalize(img_hr, self.output_variables, "output")

        if self.ocean_mask_with_buffer is not None:
            img_lr[:, self.ocean_mask_with_buffer] = 0.0
            img_hr[:, self.ocean_mask_with_buffer] = 0.0
        elif self.ocean_mask is not None:
            img_lr[:, self.ocean_mask] = 0.0
            img_hr[:, self.ocean_mask] = 0.0

        return torch.from_numpy(img_hr), torch.from_numpy(img_lr)

    def _extract_mean_std(self, group: str, var: str) -> Optional[Tuple[float, float]]:
        group_stats = self.stats.get(group, {})
        if var not in group_stats:
            return None
        entry = group_stats[var]
        if isinstance(entry, dict):
            mean = float(entry.get("mean", 0.0))
            std = float(entry.get("std", 1.0))
            return mean, std
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            return float(entry[0]), float(entry[1])
        return None

    def _normalize(self, data: np.ndarray, variables: List[str], group: str) -> np.ndarray:
        result = data.copy()
        for i, var in enumerate(variables):
            stats = self._extract_mean_std(group, var)
            if stats is None:
                continue
            mean, std = stats
            if std > 0:
                result[i] = (result[i] - mean) / std
            else:
                result[i] = result[i] - mean
        return result

    def _denormalize(self, data: np.ndarray, variables: List[str], group: str) -> np.ndarray:
        result = data.copy()
        is_batched = data.ndim == 4
        for i, var in enumerate(variables):
            stats = self._extract_mean_std(group, var)
            if stats is None:
                continue
            mean, std = stats
            if is_batched:
                result[:, i] = result[:, i] * std + mean
            else:
                result[i] = result[i] * std + mean
        return result

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        return self._normalize(x, self.input_variables, "input")

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        return self._denormalize(x, self.input_variables, "input")

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        return self._normalize(x, self.output_variables, "output")

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        result = self._denormalize(x, self.output_variables, "output")
        if self.clip_negative:
            result = DataFilters.clip_negative_values(result, self.output_variables)
        if self.precip_threshold > 0:
            result = DataFilters.apply_precip_threshold(
                result, self.output_variables, self.precip_threshold
            )
        return result

    def longitude(self) -> np.ndarray:
        return self._lon

    def latitude(self) -> np.ndarray:
        return self._lat

    def input_channels(self) -> List[ChannelMetadata]:
        return self._input_channel_metadata

    def output_channels(self) -> List[ChannelMetadata]:
        return self._output_channel_metadata

    def time(self) -> List[datetime]:
        return [ns_epoch_to_datetime(int(t)) for t in self._time_values]

    def image_shape(self) -> Tuple[int, int]:
        return (self.img_shape_y, self.img_shape_x)

    def info(self) -> dict:
        input_norm = {}
        output_norm = {}

        for var in self.input_variables:
            if var in self.stats.get("input", {}):
                input_norm[var] = self.stats["input"][var]

        for var in self.output_variables:
            if var in self.stats.get("output", {}):
                output_norm[var] = self.stats["output"][var]

        return {
            "input_normalization": input_norm,
            "output_normalization": output_norm,
            "input_variables": self.input_variables,
            "output_variables": self.output_variables,
            "train": self.train,
            "num_samples": len(self),
        }

    def get_ocean_mask(self, mask_path: Optional[str] = None) -> Optional[np.ndarray]:
        if mask_path is None:
            return None
        try:
            mask_ds = xr.open_dataset(mask_path)
            if "LANDMASK" in mask_ds:
                mask = mask_ds["LANDMASK"].values.astype(bool)
            else:
                first_var = list(mask_ds.data_vars)[0]
                mask = mask_ds[first_var].values.astype(bool)
            return mask
        except Exception as e:
            logger.warning("Could not load mask from %s: %s", mask_path, e)
            return None

    def get_loss_mask(self) -> Optional[np.ndarray]:
        return self.loss_mask

    def get_loss_mask_tensor(self, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if self.loss_mask is None:
            return None
        mask_tensor = torch.from_numpy(self.loss_mask.astype(np.float32))
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        if device is not None:
            mask_tensor = mask_tensor.to(device)
        return mask_tensor

    def get_ocean_mask_with_buffer(self) -> Optional[np.ndarray]:
        return self.ocean_mask_with_buffer
