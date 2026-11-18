"""Two input sources, one AORC target and the published AORC model transforms.

The aligned file contains raw stored AORC numbers, already on the baseline
orientation. The WRF fields retain their archived orientation after an explicit
coordinate check. Q2 is interpreted as standard WRF mixing ratio; U10/V10 are
retained as archived because rotation provenance is unavailable.
"""
import json

import netCDF4 as nc
import numpy as np
import torch
import zarr

from models.datasets.aorc_legacy import AORCDownscalingDataset, DataFilters

CHANNELS = ["APCP_surface", "DLWRF_surface", "DSWRF_surface", "PRES_surface",
            "SPFH_2maboveground", "TMP_2maboveground", "UGRD_10maboveground", "VGRD_10maboveground"]
WRF_CHANNELS = ["RAIN_TOTAL", "GLW", "SWDOWN", "PSFC", "Q2", "T2", "U10", "V10"]


class AlignedPairedDataset(AORCDownscalingDataset):
    def __init__(self, aligned_path, stats_path, input_source,
                 wrf_input_path=None, precip_threshold=0.1,
                 coastal_buffer_pixels=10, **kwargs):
        if input_source not in ("synthetic", "wrf"):
            raise ValueError("Unknown input source")
        if coastal_buffer_pixels != 10 or precip_threshold != 0.1:
            raise ValueError("Published baseline mask/threshold must be retained")
        self.input_source = input_source
        self.aligned = nc.Dataset(aligned_path, "r")
        self.aligned.set_auto_mask(False)
        self.img_shape_y, self.img_shape_x = 288, 480
        self.input_variables = list(CHANNELS)
        self.output_variables = list(CHANNELS)
        self._input_channel_metadata = self._create_channel_metadata(CHANNELS)
        self._output_channel_metadata = self._create_channel_metadata(CHANNELS)
        self._lat = np.asarray(self.aligned["lat"][:])
        self._lon = np.asarray(self.aligned["lon"][:])
        self.loss_mask = np.asarray(self.aligned["verification_mask"][:], dtype=bool)
        assert self.loss_mask.shape == (288, 480) and self.loss_mask.sum() == 128619
        self.ocean_mask_with_buffer = ~self.loss_mask
        self.ocean_mask = self.ocean_mask_with_buffer
        self.land_mask = self.loss_mask
        self.keep_ocean_as_nan = True
        self.coastal_buffer_pixels = coastal_buffer_pixels
        self.clip_negative = True
        self.precip_threshold = precip_threshold
        self.train = False
        with open(stats_path) as handle:
            self.stats = json.load(handle)
        for group in ("input", "output"):
            assert all(self.stats[group][name]["std"] > 0 for name in CHANNELS)
        t = self.aligned["time"]
        self._times = list(nc.num2date(t[:], t.units, getattr(t, "calendar", "standard")))
        if len(self._times) != 200 or len(set(map(str, self._times))) != 200:
            raise ValueError("Expected exactly 200 unique reference timestamps")
        # The original generator writes NetCDF in a background thread. Do not
        # access any HDF5-backed source from __getitem__ while that writer runs:
        # the installed netCDF/HDF5 library is not thread-safe across files.
        # This bounded 200-case cache needs ~0.9 GiB per group; values are unchanged.
        groups = ["reference_aorc"] + (["synthetic_input"] if input_source == "synthetic" else [])
        self.raw_groups = {
            group: np.stack([np.asarray(self.aligned[group][v][:], dtype=np.float32)
                             for v in CHANNELS], axis=1)
            for group in groups
        }
        self.aligned.close()
        self.aligned = None
        if input_source == "wrf":
            if not wrf_input_path:
                raise ValueError("WRF input path is required")
            self.wrf_store = zarr.open(wrf_input_path, mode="r")
            self.wrf_data = self.wrf_store["wrf"]["wrf"]
            variables = [v.decode() if isinstance(v, bytes) else str(v)
                         for v in self.wrf_store["wrf_variable"][:]]
            self.wrf_channels = [variables.index(v) for v in WRF_CHANNELS]
            lat = np.asarray(self.wrf_store["lat"][:288, :480])
            lon = np.asarray(self.wrf_store["lon"][:288, :480])
            self.wrf_flip = bool(lat[0, 0] > lat[-1, 0])
            if self.wrf_flip:
                lat, lon = lat[::-1].copy(), lon[::-1].copy()
            if not np.array_equal(lat, self._lat) or not np.array_equal(lon, self._lon):
                raise ValueError("WRF input coordinates differ from the common reference")
            epochs = np.asarray(self.wrf_store["time"][:], dtype=np.int64)
            if len(np.unique(epochs)) != len(epochs):
                raise ValueError("Duplicate WRF input timestamps")
            wanted = nc.date2num(self._times, "seconds since 1970-01-01 00:00:00", "standard").astype(np.int64)
            lookup = {int(epoch): i for i, epoch in enumerate(epochs)}
            self.wrf_indices = [lookup[int(epoch)] for epoch in wanted]

    def __len__(self):
        return len(self._times)

    def time(self):
        return list(self._times)

    def read_aorc_raw(self, group, index):
        return self.raw_groups[group][index]

    def require_finite(self, values, description):
        if not np.isfinite(values[:, self.loss_mask]).all():
            raise ValueError("Nonfinite data on the common support: " + description)

    def aorc_model_values(self, raw):
        values = DataFilters.apply_scale_factors(raw, CHANNELS)
        return self.physical_filters(values)

    def physical_filters(self, values):
        values = DataFilters.clip_negative_values(values, CHANNELS)
        return DataFilters.apply_precip_threshold(values, CHANNELS, self.precip_threshold)

    def normalized_target(self, index):
        raw = self.read_aorc_raw("reference_aorc", index)
        self.require_finite(raw, f"target {index}")
        target = self._normalize(self.aorc_model_values(raw), CHANNELS, "output")
        target[:, ~self.loss_mask] = 0.0
        return target

    def wrf_model_values(self, index):
        raw = np.asarray(self.wrf_data[self.wrf_indices[index], :, :288, :480], dtype=np.float32)
        raw = raw[self.wrf_channels]
        if self.wrf_flip:
            raw = raw[:, ::-1].copy()
        self.require_finite(raw, f"WRF input {index}")
        values = raw.copy()
        values[3] *= np.float32(0.01)  # WRF Pa -> the checkpoint's hPa convention.
        # Fixed interface conversion, not a fitted adjustment of distributions.
        mixing_ratio = np.maximum(values[4], np.float32(0.0))
        values[4] = np.float32(10.0) * (mixing_ratio / (np.float32(1.0) + mixing_ratio))
        return self.physical_filters(values)

    def __getitem__(self, index):
        target = self.normalized_target(index)
        if self.input_source == "synthetic":
            raw = self.read_aorc_raw("synthetic_input", index)
            self.require_finite(raw, f"synthetic input {index}")
            values = self.aorc_model_values(raw)
        else:
            values = self.wrf_model_values(index)
        inputs = self._normalize(values, CHANNELS, "input")
        inputs[:, ~self.loss_mask] = 0.0
        self.require_finite(inputs, f"normalized input {index}")
        return torch.from_numpy(target), torch.from_numpy(inputs)

    def close(self):
        if self.aligned is not None:
            self.aligned.close()
            self.aligned = None
