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
AORC Downscaling Dataset for training UNet and Diffusion models.

Input: aorc_12km_2_3km_2000_2012_6h.zarr (coarse resolution upsampled to 3km)
Target: aorc_3km_2000_2012_6h.zarr (high resolution 3km)

Updates:
- Strict filtering: Removes entire timesteps if NaN/Negative values appear over LAND.
- Ocean handling: Explicitly forces Ocean pixels to 0.0 in both Input and Target.
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
PRECIP_VARS = ['APCP_surface']
POSITIVE_VARS = ['APCP_surface', 'DSWRF_surface', 'DLWRF_surface', 'SPFH_2maboveground']

# Scale factors for AORC variables (stored → physical units)
# AORC data is stored with reduced precision; these factors convert to standard units
SCALE_AORC = {
    'TMP_2maboveground': 0.1,       # 0.1 K → K
    'PRES_surface': 0.1,            # 10 Pa → Pa (still needs conversion if hPa desired)
    'SPFH_2maboveground': 0.001,    # 0.001 (stored as 1000x) → kg/kg
    'APCP_surface': 0.1,            # 0.1 mm → mm
    'UGRD_10maboveground': 0.1,     # 0.1 m/s → m/s
    'VGRD_10maboveground': 0.1,     # 0.1 m/s → m/s
    'DLWRF_surface': 0.1,           # 0.1 W/m² → W/m²
    'DSWRF_surface': 0.1,           # 0.1 W/m² → W/m²
}


class DataFilters:
    """Collection of data filtering functions used in preprocessing and visualization."""

    @staticmethod
    def clip_negative_values(data: np.ndarray, variables: List[str],
                             clip_vars: Optional[List[str]] = None) -> np.ndarray:
        """
        Clip negative values to zero for specified variables.
        """
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

    @staticmethod
    def apply_precip_threshold(data: np.ndarray, variables: List[str],
                                threshold: float = 0.05,
                                precip_vars: Optional[List[str]] = None) -> np.ndarray:
        """
        Apply precipitation threshold: values below threshold are set to zero.
        Operates in physical space (mm).
        """
        if threshold <= 0:
            return data
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
    def apply_precipitation_factor(data: np.ndarray, variables: List[str],
                                   factor: float = 0.1,
                                   precip_vars: Optional[List[str]] = None) -> np.ndarray:
        """
        Apply scaling factor to precipitation variables (AORC uses 0.1 mm units).
        DEPRECATED: Use apply_scale_factors instead for all AORC variables.
        """
        if precip_vars is None:
            precip_vars = PRECIP_VARS

        result = data.copy()
        for i, var in enumerate(variables):
            if var in precip_vars:
                if result.ndim == 3:
                    result[i] = result[i] * factor
                elif result.ndim == 4:
                    result[:, i] = result[:, i] * factor
        return result

    @staticmethod
    def apply_scale_factors(data: np.ndarray, variables: List[str],
                            scale_dict: Optional[dict] = None) -> np.ndarray:
        """
        Apply AORC scale factors to convert from stored units to physical units.

        Args:
            data: Input data array (C, H, W) or (B, C, H, W)
            variables: List of variable names corresponding to channels
            scale_dict: Dictionary mapping variable names to scale factors.
                        Defaults to SCALE_AORC.

        Returns:
            Scaled data array
        """
        if scale_dict is None:
            scale_dict = SCALE_AORC

        result = data.copy()
        for i, var in enumerate(variables):
            if var in scale_dict:
                if result.ndim == 3:
                    result[i] = result[i] * scale_dict[var]
                elif result.ndim == 4:
                    result[:, i] = result[:, i] * scale_dict[var]
        return result


def hours_to_datetime(time_val, base_year: int = 2000) -> datetime:
    """Convert time value to datetime.

    Handles two formats:
    1. Hours since base_year (small numeric values)
    2. Nanoseconds since 1970 (large int64 values from datetime64[ns])
    """
    # Detect format based on value magnitude
    if abs(time_val) > 1e15:
        # Format: nanoseconds since 1970 (datetime64[ns] stored as int64)
        dt64 = np.datetime64(int(time_val), 'ns')
        # Convert to Python datetime
        ts = (dt64 - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')
        return datetime.utcfromtimestamp(float(ts))
    else:
        # Format: hours since base_year
        base = datetime(base_year, 1, 1, 0, 0, 0)
        return base + timedelta(hours=int(time_val))


def get_time_indices_for_years(
    time_values: np.ndarray,
    start_year: int,
    end_year: int,
    base_year: int = 2000
) -> np.ndarray:
    """
    Get indices of timesteps within the specified year range.

    Handles two time formats:
    1. Hours since base_year (small numeric values)
    2. Nanoseconds since 1970 (large int64 values from datetime64[ns])
    """
    # Detect time format based on value range
    # datetime64[ns] as int64 would be ~1e18 for year 2000
    # hours since 2000 would be ~0 to ~200000 for 2000-2023
    max_val = np.max(np.abs(time_values))

    if max_val > 1e15:
        # Format: nanoseconds since 1970 (datetime64[ns] stored as int64)
        # Convert to datetime64 for comparison
        time_dt64 = time_values.astype('datetime64[ns]')

        start_dt64 = np.datetime64(f'{start_year}-01-01T00:00:00')
        end_dt64 = np.datetime64(f'{end_year + 1}-01-01T00:00:00')

        mask = (time_dt64 >= start_dt64) & (time_dt64 < end_dt64)
    else:
        # Format: hours since base_year
        start_dt = datetime(start_year, 1, 1, 0, 0, 0)
        base_dt = datetime(base_year, 1, 1, 0, 0, 0)
        start_hours = (start_dt - base_dt).total_seconds() / 3600

        end_dt = datetime(end_year + 1, 1, 1, 0, 0, 0)
        end_hours = (end_dt - base_dt).total_seconds() / 3600

        mask = (time_values >= start_hours) & (time_values < end_hours)

    indices = np.where(mask)[0]

    return indices


class AORCDownscalingDataset(DownscalingDataset):
    """
    Dataset for AORC downscaling from 12km to 3km resolution.
    """

    def __init__(
        self,
        input_data_path: str,
        target_data_path: str,
        stats_path: str,
        train: bool = True,
        train_years: Union[Tuple[int, int], List[int]] = (2000, 2011),
        valid_years: Union[Tuple[int, int], List[int]] = (2012, 2012),
        apply_precip_factor: bool = True,  # DEPRECATED: use apply_scale_factors
        apply_scale_factors: bool = True,  # Apply AORC scale factors to all variables
        clip_negative: bool = True,
        input_variables: Optional[List[str]] = None,
        output_variables: Optional[List[str]] = None,
        base_year: int = 2000,
        landmask_path: Optional[str] = None,
        mask_ocean_in_target: bool = True,
        sanitize_inputs: bool = True,
        keep_ocean_as_nan: bool = False, # CHANGED: Defaults to False so we use 0.0
        drop_nan_land_timesteps: bool = True, # CHANGED: Defaults to True for strict cleaning
        nan_check_vars: Optional[List[str]] = None,
        nan_chunk: int = 128,
        debug_nan: bool = False,
        debug_nan_limit: int = 10,
        coastal_buffer_pixels: int = 0,  # Number of pixels to expand ocean mask into land (coastal buffer)
        precip_threshold: float = 0.05,  # Precipitation threshold in mm (values below → 0)
        **kwargs,
    ):
        super().__init__()

        self.input_data_path = input_data_path
        self.precip_threshold = precip_threshold
        # Missing-data / NaN handling
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

        self.land_mask = None  # True=land
        self.ocean_mask = None  # True=ocean (False=land)
        self.ocean_mask_with_buffer = None  # True=ocean+coastal buffer (for zeroing data)
        self.loss_mask = None  # True=valid for loss computation (land excluding buffer)

        self.train = train
        self.train_years = tuple(train_years)
        self.valid_years = tuple(valid_years)
        self.apply_precip_factor = apply_precip_factor  # DEPRECATED
        self.apply_scale_factors = apply_scale_factors
        self.clip_negative = clip_negative
        self.base_year = base_year

        # Define a U-Net compatible shape by cropping.
        self.img_shape_y = 288 #32*6 # Multiple of 32
        self.img_shape_x = 480 #32*6 # Multiple of 32

        # Open zarr stores
        logger.info(f"Loading input data from: {input_data_path}")
        logger.info(f"Loading target data from: {target_data_path}")

        self.input_store = zarr.open(input_data_path, mode='r')
        self.target_store = zarr.open(target_data_path, mode='r')

        # Get variable lists from zarr
        self._all_input_vars = self._decode_variable_list(self.input_store['aorc_variable'][:])
        self._all_target_vars = self._decode_variable_list(self.target_store['aorc_variable'][:])

        # Set variables to use (all by default)
        self.input_variables = input_variables if input_variables else self._all_input_vars
        self.output_variables = output_variables if output_variables else self._all_target_vars

        # Get channel indices
        self.input_idx = [self._all_input_vars.index(v) for v in self.input_variables]
        self.output_idx = [self._all_target_vars.index(v) for v in self.output_variables]

        # Load coordinates
        self._lat = self.target_store['lat'][:]
        self._lon = self.target_store['lon'][:]

        # Check latitude orientation and set flip flag
        # If latitude is decreasing (North at index 0), we need to flip the y-axis
        # to ensure consistent orientation with origin='lower' plotting
        self._flip_lat = False
        if self._lat.ndim == 2:
            lat_1d = self._lat[:, 0]
        else:
            lat_1d = self._lat
        if len(lat_1d) > 1 and lat_1d[0] > lat_1d[-1]:
            self._flip_lat = True
            # Flip the latitude and longitude arrays
            self._lat = np.flipud(self._lat)
            self._lon = np.flipud(self._lon)
            logger.info("Detected decreasing latitude - will flip y-axis for correct orientation")

        # Load and filter time indices based on train/valid split
        self._load_time_indices()

        # Initialize land/ocean mask (Cropped to image shape)
        self._init_land_ocean_mask()

        # Perform Strict Filtering (NaN/Neg on Land)
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
        """Decode variable names from zarr array."""
        return [
            v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in var_array
        ]

    def _load_time_indices(self):
        """Load time coordinates and filter based on train/valid split."""
        input_time = self.input_store['time'][:]
        target_time = self.target_store['time'][:]

        # Determine year range based on train/valid mode
        if self.train:
            year_start, year_end = self.train_years
        else:
            year_start, year_end = self.valid_years

        # Get indices for the specified years
        input_indices = get_time_indices_for_years(
            input_time, year_start, year_end, self.base_year
        )
        target_indices = get_time_indices_for_years(
            target_time, year_start, year_end, self.base_year
        )

        common_indices = np.intersect1d(input_indices, target_indices)

        self._time_indices = common_indices
        self._time_values = input_time[common_indices]

        logger.info(f"{'Training' if self.train else 'Validation'} samples: {len(self._time_indices)}")
        logger.info(f"Year range: {year_start}-{year_end}")

    def _create_channel_metadata(self, variables: List[str]) -> List[ChannelMetadata]:
        """Create channel metadata from variable names."""
        metadata = []
        for var_name in variables:
            name = var_name
            level = ""
            if '_' in var_name:
                parts = var_name.rsplit('_', 1)
                if parts[1] in ['surface', '2maboveground', '10maboveground']:
                    name = parts[0]
                    level = parts[1]

            metadata.append(ChannelMetadata(name=name, level=level, auxiliary=False))

        return metadata

    def _init_land_ocean_mask(self):
        """
        Initializes and crops the land/ocean mask from the provided path.
        Also creates an expanded ocean mask with coastal buffer if specified.
        """
        if not self.landmask_path:
            logger.info("No landmask_path provided. Skipping mask initialization.")
            return

        try:
            mask_ds = xr.open_dataset(self.landmask_path)
            # LANDMASK is usually 1 for land, 0 for ocean.
            full_mask = mask_ds['LANDMASK'].values.astype(bool)

            # Crop to the dataset's image shape
            cropped_mask = full_mask[:self.img_shape_y, :self.img_shape_x]

            # Flip mask if latitude was flipped (to match data orientation)
            if self._flip_lat:
                cropped_mask = np.flipud(cropped_mask)

            self.land_mask = cropped_mask
            self.ocean_mask = ~cropped_mask

            # Create expanded ocean mask with coastal buffer
            if self.coastal_buffer_pixels > 0:
                # Use binary dilation to expand the ocean mask into land
                # This creates a buffer zone around the coastline
                structure = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
                self.ocean_mask_with_buffer = ndimage.binary_dilation(
                    self.ocean_mask,
                    structure=structure,
                    iterations=self.coastal_buffer_pixels
                )
                # Loss mask: valid land pixels (excluding buffer zone)
                self.loss_mask = self.land_mask & ~self.ocean_mask_with_buffer

                buffer_pixels = np.sum(self.ocean_mask_with_buffer) - np.sum(self.ocean_mask)
                logger.info(f"Created coastal buffer of {self.coastal_buffer_pixels} pixels "
                           f"(added {buffer_pixels} pixels to ocean mask)")
                logger.info(f"Loss mask: {np.sum(self.loss_mask)} valid pixels "
                           f"({100*np.sum(self.loss_mask)/self.loss_mask.size:.1f}% of domain)")
            else:
                # No buffer - use original masks
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
        Scans APCP_surface (if present) over LAND in BOTH Input and Target.
        Removes timestep if NaN or Negative values appear.

        Uses FILE-BASED synchronization to avoid NCCL timeouts during long scans:
        - If cache exists: ALL ranks load independently (no NCCL needed)
        - If cache doesn't exist: Rank 0 scans and creates cache, other ranks poll for it
        """
        import time as time_module

        dist = DistributedManager()

        # Early return if no land mask or no data
        if self.land_mask is None or len(self._time_indices) == 0:
            return

        # Define cache file path based on input data path and year range
        years = self.train_years if self.train else self.valid_years
        cache_filename = f"{self.input_data_path}.{years[0]}-{years[1]}.valid_indices.npz"
        lock_filename = cache_filename + ".lock"  # Lock file to signal rank 0 is working

        # === CASE 1: Cache already exists - ALL ranks load independently ===
        cache_filename = runtime_cache_path(cache_filename)
        lock_filename = cache_filename + ".lock"
        if os.path.exists(cache_filename):
            logger.info(f"Rank {dist.rank}: Loading filtered timestep cache from {cache_filename}")
            with np.load(cache_filename) as data:
                self._time_indices = data['indices']
                self._time_values = data['values']
            logger.info(f"Rank {dist.rank}: Loaded {len(self._time_indices)} valid timesteps from cache.")
            return

        # === CASE 2: Cache doesn't exist - need to create it ===
        if dist.rank == 0:
            # Rank 0: Create lock file, do filtering, save cache, remove lock
            try:
                # Create lock file to signal we're working
                with open(lock_filename, 'w') as f:
                    f.write(f"Filtering started at {time_module.time()}")

                # Identify APCP indices
                t_apcp_idx = -1
                for i, var in enumerate(self.output_variables):
                    if "APCP" in var:
                        t_apcp_idx = self.output_idx[i]
                        break

                i_apcp_idx = -1
                for i, var in enumerate(self.input_variables):
                    if "APCP" in var:
                        i_apcp_idx = self.input_idx[i]
                        break

                if t_apcp_idx == -1 and i_apcp_idx == -1:
                    logger.info("No precipitation variable found; skipping strict filtering.")
                else:
                    num_workers = min(64, os.cpu_count() or 4)
                    logger.info(f"Rank 0: Scanning for Bad Timesteps using {num_workers} parallel workers...")

                    H, W = self.img_shape_y, self.img_shape_x
                    total_timesteps = len(self._time_indices)
                    valid_mask = np.ones(total_timesteps, dtype=bool)

                    def check_timestep(args):
                        """Check if a single timestep has bad data on land."""
                        global_idx, t_idx = args
                        try:
                            is_bad = False
                            if t_apcp_idx != -1:
                                t_data = self.target_store['aorc'][int(t_idx), t_apcp_idx, :H, :W]
                                land_vals = t_data[self.land_mask]
                                if np.any(np.isnan(land_vals)) or np.any(land_vals < 0):
                                    is_bad = True
                            if not is_bad and i_apcp_idx != -1:
                                i_data = self.input_store['aorc'][int(t_idx), i_apcp_idx, :H, :W]
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

                # Save to cache
                logger.info(f"Saving filtered timestep cache to {cache_filename}")
                np.savez(cache_filename, indices=self._time_indices, values=self._time_values)

            finally:
                # Remove lock file
                if os.path.exists(lock_filename):
                    os.remove(lock_filename)

        else:
            # Other ranks: Poll for cache file to appear (no NCCL involved!)
            logger.info(f"Rank {dist.rank}: Waiting for rank 0 to complete data filtering...")
            poll_interval = 10  # seconds
            max_wait = 7200  # 2 hours max
            waited = 0

            while waited < max_wait:
                if os.path.exists(cache_filename) and not os.path.exists(lock_filename):
                    # Cache is ready and lock is removed
                    time_module.sleep(1)  # Small delay to ensure file is fully written
                    break
                time_module.sleep(poll_interval)
                waited += poll_interval
                if waited % 60 == 0:
                    logger.info(f"Rank {dist.rank}: Still waiting for filtering cache... ({waited}s elapsed)")

            if not os.path.exists(cache_filename):
                raise RuntimeError(f"Rank {dist.rank}: Timed out waiting for filtering cache after {max_wait}s")

            # Load from cache
            logger.info(f"Rank {dist.rank}: Loading filtered timestep cache from {cache_filename}")
            with np.load(cache_filename) as data:
                self._time_indices = data['indices']
                self._time_values = data['values']
            logger.info(f"Rank {dist.rank}: Loaded {len(self._time_indices)} valid timesteps.")

        # Quick barrier at the end to ensure all ranks are synchronized before proceeding
        if dist.world_size > 1:
            torch.distributed.barrier()

    def __len__(self) -> int:
        return len(self._time_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        """
        time_idx = self._time_indices[idx]

        # 1. Load Data (Zarr returns numpy array)
        # Zarr shape is (time, channel, y, x)
        raw_lr = self.input_store['aorc'][time_idx, :, :, :]  # (channel, y, x)
        raw_hr = self.target_store['aorc'][time_idx, :, :, :]  # (channel, y, x)

        # Select channels
        img_lr = raw_lr[self.input_idx, :, :]
        img_hr = raw_hr[self.output_idx, :, :]

        # 2. Crop to Shape
        img_lr = img_lr[:, :self.img_shape_y, :self.img_shape_x]
        img_hr = img_hr[:, :self.img_shape_y, :self.img_shape_x]

        # 2.5. Flip y-axis if latitude is decreasing (to ensure consistent orientation)
        if self._flip_lat:
            img_lr = np.flip(img_lr, axis=1).copy()  # Flip along y-axis (axis 1 for C,H,W)
            img_hr = np.flip(img_hr, axis=1).copy()

        # Convert to float32
        img_lr = img_lr.astype(np.float32)
        img_hr = img_hr.astype(np.float32)

        # 3. Sanitize inputs: Replace NaN values with 0.0 BEFORE other processing
        # This handles NaN values that may exist in the raw data
        check_legacy_arrays(self, img_lr, img_hr)
        if self.sanitize_inputs:
            img_lr = np.nan_to_num(img_lr, nan=0.0)
            img_hr = np.nan_to_num(img_hr, nan=0.0)

        # 4. Filters (Scaling / Clipping)
        # Apply AORC scale factors to convert from stored units to physical units
        if self.apply_scale_factors:
            img_lr = DataFilters.apply_scale_factors(img_lr, self.input_variables)
            img_hr = DataFilters.apply_scale_factors(img_hr, self.output_variables)

        if self.clip_negative:
            img_lr = DataFilters.clip_negative_values(img_lr, self.input_variables)
            img_hr = DataFilters.clip_negative_values(img_hr, self.output_variables)

        # 4b. Precipitation threshold (values below threshold → 0)
        if self.precip_threshold > 0:
            img_lr = DataFilters.apply_precip_threshold(img_lr, self.input_variables, self.precip_threshold)
            img_hr = DataFilters.apply_precip_threshold(img_hr, self.output_variables, self.precip_threshold)

        # 5. Normalize
        img_lr = self._normalize(img_lr, self.input_variables, 'input')
        img_hr = self._normalize(img_hr, self.output_variables, 'output')

        # 6. Ocean + Coastal Buffer Handling (FORCE ZERO) - AFTER normalization
        # We zero out ocean pixels AND the coastal buffer zone AFTER normalization
        # so that these regions are truly 0.0 in the normalized space.
        # This ensures the UNet sees zeros for ocean/buffer, not normalized zeros.
        if self.ocean_mask_with_buffer is not None:
            img_lr[:, self.ocean_mask_with_buffer] = 0.0
            img_hr[:, self.ocean_mask_with_buffer] = 0.0
        elif self.ocean_mask is not None:
            img_lr[:, self.ocean_mask] = 0.0
            img_hr[:, self.ocean_mask] = 0.0

        return torch.from_numpy(img_hr), torch.from_numpy(img_lr)

    def _normalize(self, data: np.ndarray, variables: List[str], group: str) -> np.ndarray:
        """Apply z-score normalization."""
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
        """Reverse z-score normalization."""
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
        result = self._denormalize(x, self.input_variables, 'input')
        if self.precip_threshold > 0:
            result = DataFilters.apply_precip_threshold(result, self.input_variables, self.precip_threshold)
        return result

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        return self._normalize(x, self.output_variables, 'output')

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        result = self._denormalize(x, self.output_variables, 'output')

        if self.clip_negative:
            result = DataFilters.clip_negative_values(result, self.output_variables)

        if self.precip_threshold > 0:
            result = DataFilters.apply_precip_threshold(result, self.output_variables, self.precip_threshold)

        return result

    def get_precip_threshold_info(self) -> Optional[dict]:
        """
        Get precipitation threshold info for use in normalized space.

        Returns dict with:
          - channel_idx: index of APCP_surface in output_variables
          - threshold_norm: threshold value in normalized space
          - zero_norm: normalized representation of 0 mm
        Returns None if APCP_surface is not in output_variables or threshold <= 0.
        """
        if self.precip_threshold <= 0:
            return None
        precip_idx = None
        for i, var in enumerate(self.output_variables):
            if var in PRECIP_VARS:
                precip_idx = i
                break
        if precip_idx is None:
            return None
        var = self.output_variables[precip_idx]
        if var not in self.stats['output']:
            return None
        mean = self.stats['output'][var]['mean']
        std = self.stats['output'][var]['std']
        if std <= 0:
            return None
        threshold_norm = (self.precip_threshold - mean) / std
        zero_norm = -mean / std
        return {
            'channel_idx': precip_idx,
            'threshold_norm': threshold_norm,
            'zero_norm': zero_norm,
        }

    def longitude(self) -> np.ndarray:
        return self._lon

    def latitude(self) -> np.ndarray:
        return self._lat

    def input_channels(self) -> List[ChannelMetadata]:
        return self._input_channel_metadata

    def output_channels(self) -> List[ChannelMetadata]:
        return self._output_channel_metadata

    def time(self) -> List:
        return [hours_to_datetime(h, self.base_year) for h in self._time_values]

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
        """
        Returns the loss mask as a numpy array (H, W).
        True = valid pixel for loss computation (land, excluding coastal buffer)
        False = ignore in loss (ocean + coastal buffer)
        """
        return self.loss_mask

    def get_loss_mask_tensor(self, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """
        Returns the loss mask as a torch tensor of shape (1, 1, H, W).
        True/1.0 = valid pixel for loss computation
        False/0.0 = ignore in loss (ocean + coastal buffer)
        """
        if self.loss_mask is None:
            return None
        mask_tensor = torch.from_numpy(self.loss_mask.astype(np.float32))
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        if device is not None:
            mask_tensor = mask_tensor.to(device)
        return mask_tensor

    def get_ocean_mask_with_buffer(self) -> Optional[np.ndarray]:
        """
        Returns the expanded ocean mask (ocean + coastal buffer) as a numpy array (H, W).
        True = ocean or coastal buffer (to be zeroed/masked)
        False = valid land pixel
        """
        return self.ocean_mask_with_buffer