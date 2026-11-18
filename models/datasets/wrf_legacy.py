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
WRF Downscaling Dataset for training UNet and Diffusion models.

Input: wrf_12km_on_aorc_3km_2000_2012.zarr (coarse WRF regridded to 3km AORC grid)
Target: wrf_3km_on_aorc_3km_2000_2012.zarr (high resolution WRF on 3km AORC grid)

WRF zarr layout:
  - Root keys: lat, lon, time, wrf (group), wrf_variable
  - Data array: store['wrf']['wrf'] with shape (time, channel, y, x)
  - Variable names: T2, Q2, PSFC, U10, V10, SWDOWN, GLW, RAIN_TOTAL
  - Time: epoch seconds (int64)
  - Data is already in physical units (float32) -- NO scale factors needed

Differences from AORC dataset:
  - Data stored under nested group store['wrf']['wrf'] instead of store['aorc']
  - Variable list at store['wrf_variable'] instead of store['aorc_variable']
  - Time is epoch seconds instead of hours-since-base-year
  - No scale factor conversion needed (data already in physical units)
  - Different variable names (WRF convention vs AORC/GRIB convention)
"""

import json
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Union

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
PRECIP_VARS = ['RAIN_TOTAL']
POSITIVE_VARS = ['RAIN_TOTAL', 'SWDOWN', 'GLW', 'Q2']

# Precipitation threshold: values below this (in mm) are set to zero
PRECIP_THRESHOLD = 0.1

# Scale factors for WRF variables to match AORC units
# WRF data is in physical SI units; AORC uses hPa for pressure
SCALE_WRF = {
    'PSFC': 0.01,  # Pa → hPa (to match AORC PRES_surface convention)
}


class DataFilters:
    """Collection of data filtering functions used in preprocessing and visualization."""

    @staticmethod
    def apply_scale_factors(data: np.ndarray, variables: List[str],
                            scale_dict: Optional[dict] = None) -> np.ndarray:
        if scale_dict is None:
            scale_dict = SCALE_WRF

        result = data.copy()
        for i, var in enumerate(variables):
            if var in scale_dict:
                if result.ndim == 3:
                    result[i] = result[i] * scale_dict[var]
                elif result.ndim == 4:
                    result[:, i] = result[:, i] * scale_dict[var]
        return result

    @staticmethod
    def apply_precip_threshold(data: np.ndarray, variables: List[str],
                                threshold: float = PRECIP_THRESHOLD,
                                precip_vars: Optional[List[str]] = None) -> np.ndarray:
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
    def clip_negative_values(data: np.ndarray, variables: List[str],
                             clip_vars: Optional[List[str]] = None) -> np.ndarray:
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


def epoch_to_datetime(time_val) -> datetime:
    """Convert epoch seconds to datetime."""
    return datetime.utcfromtimestamp(int(time_val))


def get_time_indices_for_years(
    time_values: np.ndarray,
    start_year: int,
    end_year: int,
) -> np.ndarray:
    """
    Get indices of timesteps within the specified year range.
    Time values are epoch seconds (int64).
    """
    start_epoch = int(datetime(start_year, 1, 1, 0, 0, 0).timestamp())
    end_epoch = int(datetime(end_year + 1, 1, 1, 0, 0, 0).timestamp())

    mask = (time_values >= start_epoch) & (time_values < end_epoch)
    return np.where(mask)[0]


class WRFDownscalingDataset(DownscalingDataset):
    """
    Dataset for WRF downscaling from 12km to 3km resolution.
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
        **kwargs,
    ):
        super().__init__()

        self.input_data_path = input_data_path
        self.landmask_path = landmask_path
        self.mask_ocean_in_target = mask_ocean_in_target
        self.sanitize_inputs = sanitize_inputs
        self.keep_ocean_as_nan = keep_ocean_as_nan
        self.drop_nan_land_timesteps = drop_nan_land_timesteps
        self.nan_check_vars = nan_check_vars
        self.nan_chunk = int(nan_chunk)
        self.debug_nan = debug_nan
        self.debug_nan_limit = int(debug_nan_limit)
        self._debug_nan_seen = 0
        self.coastal_buffer_pixels = coastal_buffer_pixels

        self.land_mask = None
        self.ocean_mask = None
        self.ocean_mask_with_buffer = None
        self.loss_mask = None

        self.train = train
        self.train_years = tuple(train_years)
        self.valid_years = tuple(valid_years)
        self.apply_scale_factors = apply_scale_factors
        self.clip_negative = clip_negative
        self.precip_threshold = precip_threshold

        # U-Net compatible shape (same as AORC -- same grid)
        self.img_shape_y = 288
        self.img_shape_x = 480

        # Open zarr stores
        logger.info(f"Loading input data from: {input_data_path}")
        logger.info(f"Loading target data from: {target_data_path}")

        self.input_store = zarr.open(input_data_path, mode='r')
        self.target_store = zarr.open(target_data_path, mode='r')

        # WRF zarr: data is at store['wrf']['wrf'], variables at store['wrf_variable']
        self.input_data_array = self.input_store['wrf']['wrf']
        self.target_data_array = self.target_store['wrf']['wrf']

        # Get variable lists
        self._all_input_vars = self._decode_variable_list(self.input_store['wrf_variable'][:])
        self._all_target_vars = self._decode_variable_list(self.target_store['wrf_variable'][:])

        # Set variables to use (all by default)
        self.input_variables = input_variables if input_variables else self._all_input_vars
        self.output_variables = output_variables if output_variables else self._all_target_vars

        # Get channel indices
        self.input_idx = [self._all_input_vars.index(v) for v in self.input_variables]
        self.output_idx = [self._all_target_vars.index(v) for v in self.output_variables]

        # Load coordinates and keep them consistent with model image shape.
        # NetCDFWriter uses lat/lon to create (y, x) dims, so these must match
        # the tensors returned by __getitem__ (img_shape_y, img_shape_x).
        self._lat = self.target_store['lat'][:self.img_shape_y, :self.img_shape_x]
        self._lon = self.target_store['lon'][:self.img_shape_y, :self.img_shape_x]

        # Check latitude orientation
        self._flip_lat = False
        if self._lat.ndim == 2:
            lat_1d = self._lat[:, 0]
        else:
            lat_1d = self._lat
        if len(lat_1d) > 1 and lat_1d[0] > lat_1d[-1]:
            self._flip_lat = True
            self._lat = np.flipud(self._lat)
            self._lon = np.flipud(self._lon)
            logger.info("Detected decreasing latitude - will flip y-axis for correct orientation")

        # Load and filter time indices
        self._load_time_indices()

        # Initialize land/ocean mask
        self._init_land_ocean_mask()

        # Perform strict filtering
        if self.drop_nan_land_timesteps:
            self._filter_bad_timesteps_strict()

        # Load statistics for normalization
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)

        # Create channel metadata
        self._input_channel_metadata = self._create_channel_metadata(self.input_variables)
        self._output_channel_metadata = self._create_channel_metadata(self.output_variables)

        logger.info(f"Dataset initialized with {len(self)} samples")
        logger.info(f"Input channels: {len(self.input_variables)}")
        logger.info(f"Output channels: {len(self.output_variables)}")
        logger.info(f"Image shape: {self.image_shape()}")

    def _decode_variable_list(self, var_array: np.ndarray) -> List[str]:
        return [
            v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in var_array
        ]

    def _load_time_indices(self):
        """Load time coordinates and filter based on train/valid split."""
        input_time = self.input_store['time'][:]
        target_time = self.target_store['time'][:]

        if self.train:
            year_start, year_end = self.train_years
        else:
            year_start, year_end = self.valid_years

        input_indices = get_time_indices_for_years(input_time, year_start, year_end)
        target_indices = get_time_indices_for_years(target_time, year_start, year_end)

        common_indices = np.intersect1d(input_indices, target_indices)
        self._time_indices = common_indices
        self._time_values = input_time[common_indices]

        logger.info(f"{'Training' if self.train else 'Validation'} samples: {len(self._time_indices)}")
        logger.info(f"Year range: {year_start}-{year_end}")

    def _create_channel_metadata(self, variables: List[str]) -> List[ChannelMetadata]:
        metadata = []
        for var_name in variables:
            metadata.append(ChannelMetadata(name=var_name, level="", auxiliary=False))
        return metadata

    def _init_land_ocean_mask(self):
        if not self.landmask_path:
            logger.info("No landmask_path provided. Skipping mask initialization.")
            return

        try:
            mask_ds = xr.open_dataset(self.landmask_path)
            full_mask = mask_ds['LANDMASK'].values.astype(bool)

            cropped_mask = full_mask[:self.img_shape_y, :self.img_shape_x]

            if self._flip_lat:
                cropped_mask = np.flipud(cropped_mask)

            self.land_mask = cropped_mask
            self.ocean_mask = ~cropped_mask

            if self.coastal_buffer_pixels > 0:
                structure = ndimage.generate_binary_structure(2, 1)
                self.ocean_mask_with_buffer = ndimage.binary_dilation(
                    self.ocean_mask,
                    structure=structure,
                    iterations=self.coastal_buffer_pixels
                )
                self.loss_mask = self.land_mask & ~self.ocean_mask_with_buffer

                buffer_pixels = np.sum(self.ocean_mask_with_buffer) - np.sum(self.ocean_mask)
                logger.info(f"Created coastal buffer of {self.coastal_buffer_pixels} pixels "
                           f"(added {buffer_pixels} pixels to ocean mask)")
                logger.info(f"Loss mask: {np.sum(self.loss_mask)} valid pixels "
                           f"({100*np.sum(self.loss_mask)/self.loss_mask.size:.1f}% of domain)")
            else:
                self.ocean_mask_with_buffer = self.ocean_mask
                self.loss_mask = self.land_mask

            logger.info("Loaded and cropped Land Mask successfully.")

        except Exception as e:
            logger.error(f"Failed to load or process land mask from {self.landmask_path}: {e}")
            self.land_mask = None
            self.ocean_mask = None
            self.ocean_mask_with_buffer = None
            self.loss_mask = None

    def _filter_bad_timesteps_strict(self):
        """
        Scans RAIN_TOTAL (if present) over LAND in BOTH Input and Target.
        Removes timestep if NaN or Negative values appear.
        Uses file-based synchronization to avoid NCCL timeouts.
        """
        import time as time_module

        dist = DistributedManager()

        if self.land_mask is None or len(self._time_indices) == 0:
            return

        years = self.train_years if self.train else self.valid_years
        cache_filename = f"{self.input_data_path}.{years[0]}-{years[1]}.valid_indices.npz"
        lock_filename = cache_filename + ".lock"

        cache_filename = runtime_cache_path(cache_filename)
        lock_filename = cache_filename + ".lock"
        if os.path.exists(cache_filename):
            logger.info(f"Rank {dist.rank}: Loading filtered timestep cache from {cache_filename}")
            with np.load(cache_filename) as data:
                self._time_indices = data['indices']
                self._time_values = data['values']
            logger.info(f"Rank {dist.rank}: Loaded {len(self._time_indices)} valid timesteps from cache.")
            return

        if dist.rank == 0:
            try:
                with open(lock_filename, 'w') as f:
                    f.write(f"Filtering started at {time_module.time()}")

                # Find RAIN_TOTAL indices
                t_precip_idx = -1
                for i, var in enumerate(self.output_variables):
                    if "RAIN" in var:
                        t_precip_idx = self.output_idx[i]
                        break

                i_precip_idx = -1
                for i, var in enumerate(self.input_variables):
                    if "RAIN" in var:
                        i_precip_idx = self.input_idx[i]
                        break

                if t_precip_idx == -1 and i_precip_idx == -1:
                    logger.info("No precipitation variable found; skipping strict filtering.")
                else:
                    num_workers = min(64, os.cpu_count() or 4)
                    logger.info(f"Rank 0: Scanning for Bad Timesteps using {num_workers} parallel workers...")

                    H, W = self.img_shape_y, self.img_shape_x
                    total_timesteps = len(self._time_indices)
                    valid_mask = np.ones(total_timesteps, dtype=bool)

                    def check_timestep(args):
                        global_idx, t_idx = args
                        try:
                            is_bad = False
                            if t_precip_idx != -1:
                                t_data = self.target_data_array[int(t_idx), t_precip_idx, :H, :W]
                                land_vals = t_data[self.land_mask]
                                if np.any(np.isnan(land_vals)) or np.any(land_vals < 0):
                                    is_bad = True
                            if not is_bad and i_precip_idx != -1:
                                i_data = self.input_data_array[int(t_idx), i_precip_idx, :H, :W]
                                land_vals = i_data[self.land_mask]
                                if np.any(np.isnan(land_vals)) or np.any(land_vals < 0):
                                    is_bad = True
                            return (global_idx, is_bad)
                        except Exception:
                            return (global_idx, True)

                    work_items = [(i, self._time_indices[i]) for i in range(total_timesteps)]

                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        futures = {executor.submit(check_timestep, item): item for item in work_items}
                        with tqdm(total=total_timesteps, desc=f"Filtering Data ({num_workers} workers)") as pbar:
                            for future in as_completed(futures):
                                global_idx, is_bad = future.result()
                                if is_bad:
                                    valid_mask[global_idx] = False
                                pbar.update(1)

                    total_dropped = np.sum(~valid_mask)
                    self._time_indices = self._time_indices[valid_mask]
                    self._time_values = self._time_values[valid_mask]
                    logger.info(f"Filtering Complete. Dropped {total_dropped} timesteps. Remaining: {len(self)}")

                logger.info(f"Saving filtered timestep cache to {cache_filename}")
                np.savez(cache_filename, indices=self._time_indices, values=self._time_values)

            finally:
                if os.path.exists(lock_filename):
                    os.remove(lock_filename)

        else:
            logger.info(f"Rank {dist.rank}: Waiting for rank 0 to complete data filtering...")
            poll_interval = 10
            max_wait = 7200
            waited = 0

            while waited < max_wait:
                if os.path.exists(cache_filename) and not os.path.exists(lock_filename):
                    time_module.sleep(1)
                    break
                time_module.sleep(poll_interval)
                waited += poll_interval
                if waited % 60 == 0:
                    logger.info(f"Rank {dist.rank}: Still waiting for filtering cache... ({waited}s elapsed)")

            if not os.path.exists(cache_filename):
                raise RuntimeError(f"Rank {dist.rank}: Timed out waiting for filtering cache after {max_wait}s")

            logger.info(f"Rank {dist.rank}: Loading filtered timestep cache from {cache_filename}")
            with np.load(cache_filename) as data:
                self._time_indices = data['indices']
                self._time_values = data['values']
            logger.info(f"Rank {dist.rank}: Loaded {len(self._time_indices)} valid timesteps.")

        if dist.world_size > 1:
            torch.distributed.barrier()

    def __len__(self) -> int:
        return len(self._time_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        time_idx = self._time_indices[idx]

        # Load data from nested wrf group: store['wrf']['wrf'][time, channel, y, x]
        raw_lr = self.input_data_array[time_idx, :, :, :]
        raw_hr = self.target_data_array[time_idx, :, :, :]

        # Select channels
        img_lr = raw_lr[self.input_idx, :, :]
        img_hr = raw_hr[self.output_idx, :, :]

        # Crop to shape
        img_lr = img_lr[:, :self.img_shape_y, :self.img_shape_x]
        img_hr = img_hr[:, :self.img_shape_y, :self.img_shape_x]

        # Flip y-axis if latitude is decreasing
        if self._flip_lat:
            img_lr = np.flip(img_lr, axis=1).copy()
            img_hr = np.flip(img_hr, axis=1).copy()

        # Convert to float32
        img_lr = img_lr.astype(np.float32)
        img_hr = img_hr.astype(np.float32)

        # Sanitize inputs: Replace NaN values with 0.0
        check_legacy_arrays(self, img_lr, img_hr)
        if self.sanitize_inputs:
            img_lr = np.nan_to_num(img_lr, nan=0.0)
            img_hr = np.nan_to_num(img_hr, nan=0.0)

        # Apply WRF scale factors (PSFC: Pa → hPa to match AORC convention)
        if self.apply_scale_factors:
            img_lr = DataFilters.apply_scale_factors(img_lr, self.input_variables)
            img_hr = DataFilters.apply_scale_factors(img_hr, self.output_variables)

        # Clip negative values for positive-only variables
        if self.clip_negative:
            img_lr = DataFilters.clip_negative_values(img_lr, self.input_variables)
            img_hr = DataFilters.clip_negative_values(img_hr, self.output_variables)

        # Apply precipitation threshold (values < threshold → 0)
        if self.precip_threshold > 0:
            img_lr = DataFilters.apply_precip_threshold(img_lr, self.input_variables, self.precip_threshold)
            img_hr = DataFilters.apply_precip_threshold(img_hr, self.output_variables, self.precip_threshold)

        # Normalize
        img_lr = self._normalize(img_lr, self.input_variables, 'input')
        img_hr = self._normalize(img_hr, self.output_variables, 'output')

        # Ocean + Coastal Buffer Handling (FORCE ZERO) - AFTER normalization
        if self.ocean_mask_with_buffer is not None:
            img_lr[:, self.ocean_mask_with_buffer] = 0.0
            img_hr[:, self.ocean_mask_with_buffer] = 0.0
        elif self.ocean_mask is not None:
            img_lr[:, self.ocean_mask] = 0.0
            img_hr[:, self.ocean_mask] = 0.0

        return torch.from_numpy(img_hr), torch.from_numpy(img_lr)

    def _normalize(self, data: np.ndarray, variables: List[str], group: str) -> np.ndarray:
        result = data.copy()
        for i, var in enumerate(variables):
            if var in self.stats[group]:
                mean = self.stats[group][var]['mean']
                std = self.stats[group][var]['std']
                if std > 0:
                    result[i] = (result[i] - mean) / std
                else:
                    result[i] = result[i] - mean
        return result

    def _denormalize(self, data: np.ndarray, variables: List[str], group: str) -> np.ndarray:
        result = data.copy()
        is_batched = data.ndim == 4

        for i, var in enumerate(variables):
            if var in self.stats[group]:
                mean = self.stats[group][var]['mean']
                std = self.stats[group][var]['std']
                if is_batched:
                    result[:, i] = result[:, i] * std + mean
                else:
                    result[i] = result[i] * std + mean
        return result

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        return self._normalize(x, self.input_variables, 'input')

    def denormalize_input(self, x: np.ndarray) -> np.ndarray:
        return self._denormalize(x, self.input_variables, 'input')

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        return self._normalize(x, self.output_variables, 'output')

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        result = self._denormalize(x, self.output_variables, 'output')
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

    def time(self) -> List:
        return [epoch_to_datetime(t) for t in self._time_values]

    def image_shape(self) -> Tuple[int, int]:
        return (self.img_shape_y, self.img_shape_x)

    def info(self) -> dict:
        input_norm = {}
        output_norm = {}

        for var in self.input_variables:
            if var in self.stats['input']:
                input_norm[var] = self.stats['input'][var]

        for var in self.output_variables:
            if var in self.stats['output']:
                output_norm[var] = self.stats['output'][var]

        return {
            'input_normalization': input_norm,
            'output_normalization': output_norm,
            'input_variables': self.input_variables,
            'output_variables': self.output_variables,
            'train': self.train,
            'num_samples': len(self),
        }

    def get_ocean_mask(self, mask_path: Optional[str] = None) -> Optional[np.ndarray]:
        if mask_path is None:
            return None
        try:
            mask_ds = xr.open_dataset(mask_path)
            mask = mask_ds['LANDMASK'].values.astype(bool)
            return mask
        except Exception as e:
            logger.warning(f"Could not load mask from {mask_path}: {e}")
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
