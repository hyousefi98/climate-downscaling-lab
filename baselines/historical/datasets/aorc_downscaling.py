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
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
import xarray as xr
import zarr
import multiprocessing as mp
from scipy import ndimage
from tqdm import tqdm

from physicsnemo.distributed import DistributedManager

from datasets.base import ChannelMetadata, DownscalingDataset

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


def hours_to_datetime(hours: int, base_year: int = 2000) -> datetime:
    """Convert hours from base_year start to datetime."""
    base = datetime(base_year, 1, 1, 0, 0, 0)
    return base + timedelta(hours=int(hours))


def normalize_time_to_hours(time_values: np.ndarray, base_year: int = 2000) -> np.ndarray:
    """
    Normalize time values to hours since base_year.

    Handles:
    - hours since base_year (expected)
    - datetime64 arrays
    - epoch-based seconds / milliseconds / nanoseconds
    """
    time_arr = np.asarray(time_values)
    if time_arr.size == 0:
        return time_arr.astype(np.float64)

    if np.issubdtype(time_arr.dtype, np.datetime64):
        base = np.datetime64(f"{base_year}-01-01T00:00:00", "ns")
        return ((time_arr.astype("datetime64[ns]") - base) / np.timedelta64(1, "h")).astype(np.float64)

    max_val = np.nanmax(time_arr)
    # Heuristic: very large values are likely epoch-based (seconds/ms/ns).
    if max_val > 1e14:
        scale = 1e9  # nanoseconds
    elif max_val > 1e11:
        scale = 1e3  # milliseconds
    elif max_val > 1e8:
        scale = 1.0  # seconds
    else:
        # Assume already hours since base_year.
        return time_arr.astype(np.float64)

    epoch = datetime(1970, 1, 1, 0, 0, 0)
    base = datetime(base_year, 1, 1, 0, 0, 0)
    base_seconds = (base - epoch).total_seconds()
    return ((time_arr.astype(np.float64) / scale) - base_seconds) / 3600.0


def detect_time_first(shape: Tuple[int, ...], nvars: int, label: str) -> bool:
    """
    Detect whether the time dimension is first or second in the AORC array.
    Expected shapes:
      - (C, T, H, W) -> channel-first
      - (T, C, H, W) -> time-first
    """
    if len(shape) < 2:
        logger.warning(f"{label} aorc shape {shape} is too small to infer layout; assuming channel-first.")
        return False
    if shape[0] == nvars and shape[1] != nvars:
        return False
    if shape[1] == nvars and shape[0] != nvars:
        return True
    logger.warning(
        f"{label} aorc shape {shape} is ambiguous for nvars={nvars}; assuming channel-first."
    )
    return False


_SCAN_CTX = {}


def _scan_init(
    input_path: str,
    target_path: str,
    land_mask: np.ndarray,
    H: int,
    W: int,
    i_apcp_idx: int,
    t_apcp_idx: int,
    input_time_first: bool,
    target_time_first: bool,
):
    _SCAN_CTX["input_store"] = zarr.open(input_path, mode="r")
    _SCAN_CTX["target_store"] = zarr.open(target_path, mode="r")
    _SCAN_CTX["land_mask"] = land_mask
    _SCAN_CTX["H"] = H
    _SCAN_CTX["W"] = W
    _SCAN_CTX["i_apcp_idx"] = i_apcp_idx
    _SCAN_CTX["t_apcp_idx"] = t_apcp_idx
    _SCAN_CTX["input_time_first"] = input_time_first
    _SCAN_CTX["target_time_first"] = target_time_first


def _scan_read_channel(store, time_idx: int, channel_idx: int, H: int, W: int, time_first: bool) -> np.ndarray:
    if time_first:
        return store["aorc"][int(time_idx), channel_idx, :H, :W]
    return store["aorc"][channel_idx, int(time_idx), :H, :W]


def _scan_worker(time_idx: int) -> Tuple[int, bool]:
    try:
        land_mask = _SCAN_CTX["land_mask"]
        H = _SCAN_CTX["H"]
        W = _SCAN_CTX["W"]
        i_apcp_idx = _SCAN_CTX["i_apcp_idx"]
        t_apcp_idx = _SCAN_CTX["t_apcp_idx"]
        input_time_first = _SCAN_CTX["input_time_first"]
        target_time_first = _SCAN_CTX["target_time_first"]

        if t_apcp_idx != -1:
            t_data = _scan_read_channel(
                _SCAN_CTX["target_store"], time_idx, t_apcp_idx, H, W, target_time_first
            )
            if np.any(np.isnan(t_data[land_mask])) or np.any(t_data[land_mask] < 0):
                return time_idx, True

        if i_apcp_idx != -1:
            i_data = _scan_read_channel(
                _SCAN_CTX["input_store"], time_idx, i_apcp_idx, H, W, input_time_first
            )
            if np.any(np.isnan(i_data[land_mask])) or np.any(i_data[land_mask] < 0):
                return time_idx, True

        return time_idx, False
    except Exception:
        return time_idx, True


def get_time_indices_for_years(
    time_hours: np.ndarray,
    start_year: int,
    end_year: int,
    base_year: int = 2000
) -> np.ndarray:
    """
    Get indices of timesteps within the specified year range.
    """
    start_dt = datetime(start_year, 1, 1, 0, 0, 0)
    base_dt = datetime(base_year, 1, 1, 0, 0, 0)
    start_hours = (start_dt - base_dt).total_seconds() / 3600

    end_dt = datetime(end_year + 1, 1, 1, 0, 0, 0)
    end_hours = (end_dt - base_dt).total_seconds() / 3600

    mask = (time_hours >= start_hours) & (time_hours < end_hours)
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
        scan_workers: int = 1,
        nan_check_vars: Optional[List[str]] = None,
        nan_chunk: int = 128,
        debug_nan: bool = False,
        debug_nan_limit: int = 10,
        coastal_buffer_pixels: int = 0,  # Number of pixels to expand ocean mask into land (coastal buffer)
        **kwargs,
    ):
        super().__init__()

        self.input_data_path = input_data_path
        self.target_data_path = target_data_path
        # Missing-data / NaN handling
        self.landmask_path = landmask_path
        self.mask_ocean_in_target = mask_ocean_in_target
        self.sanitize_inputs = sanitize_inputs
        self.keep_ocean_as_nan = keep_ocean_as_nan
        self.drop_nan_land_timesteps = drop_nan_land_timesteps
        self.scan_workers = max(1, int(scan_workers))
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

        # Detect aorc array layout (time-first vs channel-first)
        self._input_time_first = detect_time_first(
            self.input_store['aorc'].shape, len(self._all_input_vars), "input"
        )
        self._target_time_first = detect_time_first(
            self.target_store['aorc'].shape, len(self._all_target_vars), "target"
        )

        # Load coordinates
        self._lat = self.target_store['lat'][:]
        self._lon = self.target_store['lon'][:]

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
        input_time_raw = self.input_store['time'][:]
        target_time_raw = self.target_store['time'][:]
        input_time = normalize_time_to_hours(input_time_raw, self.base_year)
        target_time = normalize_time_to_hours(target_time_raw, self.base_year)

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
        This is done on rank 0 and broadcast to other ranks to avoid redundant work.
        A cache file is used to avoid re-scanning on subsequent runs.
        """
        dist = DistributedManager()

        # Early return conditions - must be checked consistently across all ranks
        # to avoid NCCL desync issues
        should_skip = self.land_mask is None or len(self._time_indices) == 0

        # In distributed mode, we must ensure ALL ranks agree on whether to skip
        if dist.world_size > 1:
            # Synchronize the decision across all ranks using a simple tensor broadcast
            skip_tensor = torch.tensor([1 if should_skip else 0], dtype=torch.int32, device='cuda')
            torch.distributed.broadcast(skip_tensor, src=0)
            should_skip = skip_tensor.item() == 1

        if should_skip:
            return

        # Define a cache file path based on the input data path and year range
        years = self.train_years if self.train else self.valid_years
        layout_tag = f"in{'t' if self._input_time_first else 'c'}_tar{'t' if self._target_time_first else 'c'}"
        cache_filename = (
            f"{self.input_data_path}.{years[0]}-{years[1]}.{layout_tag}.timehours.valid_indices.npz"
        )

        # Check if we need to do a long scan (no cache exists)
        # If so, temporarily increase NCCL timeout to avoid timeouts during the scan
        needs_scan = dist.rank == 0 and not os.path.exists(cache_filename)
        old_timeout = None  # Will be set if we need to extend timeout
        if dist.world_size > 1:
            # Broadcast whether we need a long scan so all ranks can adjust timeout
            scan_tensor = torch.tensor([1 if needs_scan else 0], dtype=torch.int32, device='cuda')
            torch.distributed.broadcast(scan_tensor, src=0)
            needs_scan = scan_tensor.item() == 1

            if needs_scan:
                # Increase timeout to 30 minutes for the initial scan
                # This is needed because rank 0 scans while others wait at barrier
                import datetime
                old_timeout = torch.distributed.get_default_timeout()
                torch.distributed.set_default_timeout(datetime.timedelta(minutes=30))
                logger.info(f"Rank {dist.rank}: Extended NCCL timeout to 30 minutes for initial data scan")

        # --- Rank 0 performs the scan or loads from cache ---
        if dist.rank == 0:
            if os.path.exists(cache_filename):
                logger.info(f"Loading filtered timestep cache from {cache_filename}")
                with np.load(cache_filename) as data:
                    self._time_indices = data['indices']
                    self._time_values = data['values']
                logger.info(f"Loaded {len(self._time_indices)} valid timesteps from cache.")
            else:
                # 1. Identify APCP index for Target
                t_apcp_idx = -1
                for i, var in enumerate(self.output_variables):
                    if "APCP" in var:
                        t_apcp_idx = self.output_idx[i]
                        break

                # 2. Identify APCP index for Input
                i_apcp_idx = -1
                for i, var in enumerate(self.input_variables):
                    if "APCP" in var:
                        i_apcp_idx = self.input_idx[i]
                        break

                # If neither has APCP, keep original indices (don't filter)
                if t_apcp_idx == -1 and i_apcp_idx == -1:
                    logger.info("No precipitation variable found; skipping strict filtering (keeping all timesteps).")
                    # Keep self._time_indices and self._time_values as-is
                else:
                    logger.info("Scanning for Bad Timesteps (NaN/Neg on LAND) in Input AND Target... (on rank 0)")
                    valid_indices, valid_times, total_dropped = [], [], 0
                    H, W = self.img_shape_y, self.img_shape_x
                    time_map = {int(k): float(v) for k, v in zip(self._time_indices, self._time_values)}

                    if self.scan_workers > 1:
                        ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
                        with ctx.Pool(
                            processes=self.scan_workers,
                            initializer=_scan_init,
                            initargs=(
                                self.input_data_path,
                                self.target_data_path,
                                self.land_mask,
                                H,
                                W,
                                i_apcp_idx,
                                t_apcp_idx,
                                self._input_time_first,
                                self._target_time_first,
                            ),
                        ) as pool:
                            pbar = tqdm(total=len(self._time_indices), desc="Filtering Data")
                            for t_idx, is_bad in pool.imap_unordered(_scan_worker, list(self._time_indices), chunksize=8):
                                if is_bad:
                                    total_dropped += 1
                                else:
                                    valid_indices.append(t_idx)
                                    valid_times.append(time_map[int(t_idx)])
                                pbar.update(1)
                            pbar.close()
                    else:
                        pbar = tqdm(total=len(self._time_indices), desc="Filtering Data")
                        for i in range(len(self._time_indices)):
                            t_idx, t_val, is_bad = self._time_indices[i], self._time_values[i], False
                            try:
                                if t_apcp_idx != -1:
                                    t_data = self._read_aorc_channel(
                                        self.target_store, t_idx, t_apcp_idx, H, W, self._target_time_first
                                    )
                                    if np.any(np.isnan(t_data[self.land_mask])) or np.any(t_data[self.land_mask] < 0):
                                        is_bad = True
                                if not is_bad and i_apcp_idx != -1:
                                    i_data = self._read_aorc_channel(
                                        self.input_store, t_idx, i_apcp_idx, H, W, self._input_time_first
                                    )
                                    if np.any(np.isnan(i_data[self.land_mask])) or np.any(i_data[self.land_mask] < 0):
                                        is_bad = True
                                if is_bad: total_dropped += 1
                                else: valid_indices.append(t_idx); valid_times.append(t_val)
                            except Exception as e:
                                logger.warning(f"Error reading timestep {t_idx}: {e}"); total_dropped += 1
                            pbar.update(1)
                        pbar.close()

                    self._time_indices = np.array(valid_indices, dtype=np.int64)
                    self._time_values = np.array(valid_times, dtype=np.float64)
                    logger.info(f"Filtering Complete. Dropped {total_dropped} timesteps. Remaining: {len(self)}")

                # Save to cache
                logger.info(f"Saving filtered timestep cache to {cache_filename}")
                try:
                    np.savez(cache_filename, indices=self._time_indices, values=self._time_values)
                except Exception as e:
                    logger.warning(f"Could not write cache file to {cache_filename}: {e}")

        # --- Synchronize and broadcast results from rank 0 ---
        if dist.world_size > 1:
            # All processes wait for rank 0 to finish scanning/loading
            torch.distributed.barrier()

            # Restore original timeout if we extended it
            if needs_scan and old_timeout is not None:
                torch.distributed.set_default_timeout(old_timeout)
                logger.info(f"Rank {dist.rank}: Restored original NCCL timeout")

            # On rank 0, this list contains the filtered numpy arrays.
            # On other ranks, it's a list of None placeholders.
            obj_list = [None, None]
            if dist.rank == 0:
                obj_list = [self._time_indices, self._time_values]

            # Broadcast the list of objects from rank 0 to all other processes.
            # Use default (CPU-based gloo) backend for object broadcast to avoid NCCL issues
            torch.distributed.broadcast_object_list(obj_list, src=0)

            # All ranks (except 0) update their attributes from the received list.
            if dist.rank != 0:
                self._time_indices, self._time_values = obj_list[0], obj_list[1]
                if dist.rank == 1: # Log on one other rank to confirm
                    logger.info(f"Rank {dist.rank} received {len(self._time_indices)} filtered timesteps.")

            # Final barrier to ensure all processes have the updated data before proceeding.
            torch.distributed.barrier()

    def __len__(self) -> int:
        return len(self._time_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        """
        time_idx = self._time_indices[idx]

        # 1. Load Data (Zarr returns numpy array)
        raw_lr = self._read_aorc_frame(self.input_store, time_idx, self._input_time_first)
        raw_hr = self._read_aorc_frame(self.target_store, time_idx, self._target_time_first)

        # Select channels
        img_lr = raw_lr[self.input_idx, :, :]
        img_hr = raw_hr[self.output_idx, :, :]

        # 2. Crop to Shape
        img_lr = img_lr[:, :self.img_shape_y, :self.img_shape_x]
        img_hr = img_hr[:, :self.img_shape_y, :self.img_shape_x]

        # Convert to float32
        img_lr = img_lr.astype(np.float32)
        img_hr = img_hr.astype(np.float32)

        # 3. Sanitize inputs: Replace NaN values with 0.0 BEFORE other processing
        # This handles NaN values that may exist in the raw data
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

    def _read_aorc_frame(self, store, time_idx: int, time_first: bool) -> np.ndarray:
        H, W = self.img_shape_y, self.img_shape_x
        if time_first:
            return store['aorc'][int(time_idx), :, :H, :W]
        return store['aorc'][:, int(time_idx), :H, :W]

    def _read_aorc_channel(
        self,
        store,
        time_idx: int,
        channel_idx: int,
        H: int,
        W: int,
        time_first: bool,
    ) -> np.ndarray:
        if time_first:
            return store['aorc'][int(time_idx), channel_idx, :H, :W]
        return store['aorc'][channel_idx, int(time_idx), :H, :W]

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
        return self._denormalize(x, self.input_variables, 'input')

    def normalize_output(self, x: np.ndarray) -> np.ndarray:
        return self._normalize(x, self.output_variables, 'output')

    def denormalize_output(self, x: np.ndarray) -> np.ndarray:
        result = self._denormalize(x, self.output_variables, 'output')

        if self.clip_negative:
            result = DataFilters.clip_negative_values(result, self.output_variables)

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
