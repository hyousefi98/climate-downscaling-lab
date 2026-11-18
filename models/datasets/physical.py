"""Strict runtime adapter for newly prepared physical-unit datasets.

Historical loaders remain separate because their scaling and missing-value
handling are part of the original checkpoint interface. This adapter does not
apply their packed-number scale factors or silently repair invalid land data.
"""
import datetime
import hashlib
import json

import cftime
import numpy as np
import torch
import zarr
from scipy.ndimage import binary_dilation, generate_binary_structure

from models.datasets.base import ChannelMetadata, DownscalingDataset


def _decode(values):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


class PhysicalStore:
    def __init__(self, path):
        self.group = zarr.open_group(str(path), mode="r")
        if self.group.attrs.get("numeric_profile") != "physical":
            raise ValueError(f"{path}: expected explicit numeric_profile='physical'; use the legacy workflow for archived stores")
        self.array = None
        for key in ("aorc", "wrf", "era5", "conus404"):
            if key in self.group and key + "_variable" in self.group:
                candidate = self.group[key]
                self.array = candidate[key] if isinstance(candidate, zarr.Group) else candidate
                self.variables = _decode(self.group[key + "_variable"][:])
                break
        if self.array is None:
            self.variables = [key for key in self.group.array_keys() if key not in {"lat", "lon", "time", "spatial_support"}]
        time = self.group["time"]
        if time.attrs.get("calendar", "standard") not in {"standard", "gregorian", "proleptic_gregorian"}:
            raise ValueError("Corrected stores require a Gregorian time calendar")
        units = time.attrs.get("units", "")
        if " since " not in units:
            raise ValueError("Time coordinate must explicitly declare its units and origin")
        unit, origin = units.split(" since ", 1)
        unit = {"nanoseconds": "ns", "microseconds": "us", "milliseconds": "ms", "seconds": "s",
                "minutes": "m", "hours": "h", "days": "D"}.get(unit, unit)
        if unit not in {"ns", "us", "ms", "s", "m", "h", "D"}:
            raise ValueError("Unsupported time unit: " + unit)
        offsets = np.asarray(time[:])
        if offsets.ndim != 1 or not len(offsets) or not np.isfinite(offsets).all() or not np.equal(offsets, np.floor(offsets)).all():
            raise ValueError("Time offsets must be finite integers in the declared unit")
        self.times = (np.datetime64(origin) + offsets.astype(f"timedelta64[{unit}]")).astype("datetime64[ns]")
        if np.isnat(self.times).any() or np.any(self.times[1:] <= self.times[:-1]):
            raise ValueError("Corrected stores require finite, unique, strictly increasing time coordinates")

    def read(self, index, variables, shape):
        h, w = shape
        if self.array is not None:
            indices = [self.variables.index(v) for v in variables]
            return np.asarray(self.array[index, :, :h, :w], dtype=np.float32)[indices]
        return np.stack([np.asarray(self.group[v][index, :h, :w], dtype=np.float32) for v in variables])


def validate_physical(values, variables, support, description):
    """Reject missing values and gross unit/fill-value errors on model support."""
    if not np.isfinite(values[:, support]).all():
        raise ValueError(f"Nonfinite values on model support: {description}")
    for i, name in enumerate(variables):
        x = values[i, support]
        bounds = None
        if name in {"T2", "TMP_2maboveground"} or name.startswith("t_"):
            bounds = (150., 350.)
        elif name in {"PSFC", "PRES_surface"}:
            bounds = (400., 1200.)  # Surface pressure in hPa.
        elif name in {"SPFH_2maboveground", "Q2"} or name.startswith("q_"):
            bounds = (0., .1)
        elif name in {"U10", "V10", "UGRD_10maboveground", "VGRD_10maboveground"} or name.startswith(("u_", "v_")):
            bounds = (-150., 150.)
        elif name in {"APCP_surface", "RAIN_TOTAL", "PREC_ACC_NC", "DLWRF_surface", "DSWRF_surface", "GLW", "SWDOWN"}:
            bounds = (0., float("inf"))
        if bounds and np.any((x < bounds[0]) | (x > bounds[1])):
            raise ValueError(f"{description}/{name}: values outside diagnostic bounds {bounds}; check units and source QC")


class PhysicalPairDataset(DownscalingDataset):
    def __init__(self, input_data_path, target_data_path, stats_path,
                 input_variables, output_variables, train=True,
                 train_years=(2000, 2011), valid_years=(2012, 2012),
                 landmask_path=None, coastal_buffer_pixels=0,
                 crop_shape=None, crop_to_multiple=32, precip_threshold=.1,
                 keep_ocean_as_nan=True, requested_times=None,
                 shared_support_paths=(), **unused):
        self.input_store = PhysicalStore(input_data_path)
        self.target_store = PhysicalStore(target_data_path)
        self.input_variables, self.output_variables = list(input_variables), list(output_variables)
        self.keep_ocean_as_nan = keep_ocean_as_nan
        self.coastal_buffer_pixels = int(coastal_buffer_pixels)
        self.precip_threshold = float(precip_threshold)
        self.train = bool(train)
        a, b = self.input_store.group, self.target_store.group
        for key in ("temporal_profile", "precipitation_interval", "pressure_units", "humidity_representation"):
            if a.attrs.get(key) is None or a.attrs.get(key) != b.attrs.get(key):
                raise ValueError(f"Input and target require matching explicit {key} metadata")
        if a.attrs["temporal_profile"] != "ending":
            raise ValueError("Corrected runtime requires end-labeled six-hour windows; historical grouping belongs to an explicit legacy profile")
        if a.attrs["pressure_units"] != "hPa" or a.attrs["humidity_representation"] != "specific_humidity":
            raise ValueError("Physical runtime requires hPa pressure and physical specific humidity")
        lat, lon = np.asarray(b["lat"][:]), np.asarray(b["lon"][:])
        if lat.ndim != 2 or lon.shape != lat.shape:
            raise ValueError("Reference latitude/longitude must be matching two-dimensional arrays")
        if crop_shape is None:
            crop_shape = tuple((n // crop_to_multiple) * crop_to_multiple for n in lat.shape)
        if len(crop_shape) != 2 or any(int(n) != n for n in crop_shape):
            raise ValueError("Crop shape must contain two integers")
        self.img_shape_y, self.img_shape_x = map(int, crop_shape)
        if min(crop_shape) <= 0:
            raise ValueError("Crop is empty")
        for store in (self.input_store, self.target_store):
            shapes = [store.group[name].shape for name in ("lat", "lon")]
            shapes += [store.array.shape[-2:]] if store.array is not None else [store.group[name].shape[-2:] for name in store.variables]
            if any(len(shape) != 2 or any(n > available for n, available in zip(crop_shape, shape)) for shape in shapes):
                raise ValueError("Crop shape exceeds source grid dimensions")
        cut = np.s_[:self.img_shape_y, :self.img_shape_x]
        if not np.array_equal(np.asarray(a["lat"][:])[cut], lat[cut]) or not np.array_equal(np.asarray(a["lon"][:])[cut], lon[cut]):
            raise ValueError("Input and reference coordinates differ")
        self._lat, self._lon = lat[cut], lon[cut]
        mask = np.ones(self.image_shape(), dtype=bool)
        if landmask_path:
            import netCDF4
            with netCDF4.Dataset(landmask_path) as source:
                lat_name = next((n for n in ("lat", "LAT2D", "latitude", "XLAT") if n in source.variables), None)
                lon_name = next((n for n in ("lon", "LON2D", "longitude", "XLONG") if n in source.variables), None)
                if lat_name is None or lon_name is None:
                    raise ValueError("Corrected land masks must carry explicit latitude/longitude coordinates")
                mask_lat, mask_lon = np.asarray(source[lat_name][:]), np.asarray(source[lon_name][:])
                if mask_lat.ndim == mask_lon.ndim == 1:
                    mask_lat, mask_lon = np.meshgrid(mask_lat, mask_lon, indexing="ij")
                if mask_lat.ndim != 2 or mask_lon.ndim != 2 or not np.array_equal(mask_lat[cut], lat[cut]) or not np.array_equal(mask_lon[cut], lon[cut]):
                    raise ValueError("Land-mask coordinates differ from the corrected dataset grid")
                mask &= np.asarray(np.ma.filled(source["LANDMASK"][cut], 0), dtype=bool)
        if self.coastal_buffer_pixels:
            mask = ~binary_dilation(~mask, structure=generate_binary_structure(2, 1), iterations=self.coastal_buffer_pixels)
        support_stores = [a, b] + [PhysicalStore(p).group for p in shared_support_paths]
        for store in support_stores:
            if not np.array_equal(np.asarray(store["lat"][:])[cut], lat[cut]) or not np.array_equal(np.asarray(store["lon"][:])[cut], lon[cut]):
                raise ValueError("Shared support coordinates differ")
            if "spatial_support" in store:
                mask &= np.asarray(store["spatial_support"][:])[cut].astype(bool)
        if not mask.any():
            raise ValueError("Input/target masks have no common support")
        self.loss_mask = self.land_mask = mask
        self.ocean_mask = self.ocean_mask_with_buffer = ~mask
        self._flip_lat = bool(self._lat[0, 0] > self._lat[-1, 0])
        if self._flip_lat:
            self._lat, self._lon = self._lat[::-1].copy(), self._lon[::-1].copy()
            self.loss_mask = self.land_mask = mask[::-1].copy()
            self.ocean_mask = self.ocean_mask_with_buffer = ~self.loss_mask
        common = np.intersect1d(self.input_store.times, self.target_store.times)
        years = train_years if train else valid_years
        common = common[(common >= np.datetime64(f"{years[0]}-01-01")) & (common < np.datetime64(f"{years[1] + 1}-01-01"))]
        if requested_times is not None:
            wanted = np.asarray(requested_times, dtype="datetime64[ns]")
            if len(np.unique(wanted)) != len(wanted) or not np.isin(wanted, common).all():
                raise ValueError("Requested timestamps must be unique and available in both stores and the requested split")
            common = wanted
        if not len(common):
            raise ValueError("No common timestamps in the requested split")
        maps = [{int(t): i for i, t in enumerate(s.times.astype(np.int64))} for s in (self.input_store, self.target_store)]
        self._pairs = np.asarray([[m[int(t)] for m in maps] for t in common.astype(np.int64)])
        self._time_indices = np.arange(len(common))
        self._time_values = common.astype(np.int64)
        with open(stats_path) as handle:
            self.stats = json.load(handle)
        metadata = self.stats.get("metadata", {})
        self.validate_statistics_profile(metadata)
        if metadata.get("numeric_profile") == "physical":
            for key in ("temporal_profile", "precipitation_interval", "pressure_units", "humidity_representation"):
                if metadata[key] != a.attrs.get(key) or metadata[key] != b.attrs.get(key):
                    raise ValueError("Statistics and source metadata differ: " + key)
            if not np.isclose(float(metadata["precip_threshold"]), self.precip_threshold):
                raise ValueError("Statistics and runtime precipitation thresholds differ")
            if metadata["mask_sha256"] != hashlib.sha256(mask.tobytes()).hexdigest():
                raise ValueError("Statistics and runtime use different spatial support")
            included = np.asarray(metadata["included_timestamps"], dtype="datetime64[ns]")
            if included.ndim != 1 or not len(included) or np.isnat(included).any() or len(np.unique(included)) != len(included):
                raise ValueError("Statistics included_timestamps must be nonempty, finite, and unique")
            included = included.astype(np.int64)
            if self.train and requested_times is not None and not np.isin(self._time_values, included).all():
                raise ValueError("Requested training timestamps are not a subset of the statistics training membership")
            if self.train and metadata.get("dropped_timestamps"):
                dropped = np.asarray(metadata["dropped_timestamps"], dtype="datetime64[ns]").astype(np.int64)
                if np.isin(dropped, included).any():
                    raise ValueError("Statistics included and dropped timestamps overlap")
                keep = ~np.isin(self._time_values, dropped)
                self._time_indices, self._time_values = self._time_indices[keep], self._time_values[keep]
                if not len(self): raise ValueError("No training times remain after the declared statistics drop policy")
            if self.train and requested_times is None and not np.array_equal(np.sort(self._time_values), np.sort(included)):
                raise ValueError("Selected training timestamps differ from the statistics training membership")
            if not self.train and np.isin(self._time_values, included).any():
                raise ValueError("Evaluation timestamps overlap the statistics training membership")
        for group, names in (("input", self.input_variables), ("output", self.output_variables)):
            for name in names:
                entry = self.stats[group][name]
                if not np.isfinite([entry["mean"], entry["std"]]).all() or entry["std"] <= 0:
                    raise ValueError(f"Invalid normalization statistics for {group}/{name}")

    def __len__(self):
        return len(self._time_indices)

    def __getitem__(self, index):
        pair = self._pairs[self._time_indices[index]]
        arrays = []
        for side, source, ti, names in (("input", self.input_store, pair[0], self.input_variables), ("output", self.target_store, pair[1], self.output_variables)):
            values = self.read_values(side, source, int(ti), names)
            if self._flip_lat:
                values = values[:, ::-1].copy()
            validate_physical(values, names, self.loss_mask, f"{side} time index {ti}")
            for j, name in enumerate(names):
                if name in {"APCP_surface", "RAIN_TOTAL", "PREC_ACC_NC"}:
                    values[j] = np.where(values[j] < self.precip_threshold, np.float32(0.), values[j])
            normalized = self._normalize(values, names, side)
            normalized[:, ~self.loss_mask] = 0.
            arrays.append(torch.from_numpy(normalized))
        return arrays[1], arrays[0]

    def read_values(self, side, source, index, names):
        return source.read(index, names, self.image_shape())

    def validate_statistics_profile(self, metadata):
        if metadata.get("numeric_profile") != "physical":
            raise ValueError("Normalization statistics must explicitly declare metadata.numeric_profile='physical'; archived checkpoint statistics are not interchangeable")
        required = ("precip_threshold", "mask_sha256", "included_timestamps", "temporal_profile",
                    "precipitation_interval", "pressure_units", "humidity_representation")
        missing = [name for name in required if name not in metadata]
        if missing:
            raise ValueError("Physical normalization statistics require metadata fields: " + ", ".join(missing))

    def _normalize(self, values, names, group, inverse=False):
        out = np.asarray(values, dtype=np.float32).copy()
        for i, name in enumerate(names):
            mean, std = self.stats[group][name]["mean"], self.stats[group][name]["std"]
            if inverse:
                out[..., i, :, :] = out[..., i, :, :] * std + mean
            else:
                out[..., i, :, :] = (out[..., i, :, :] - mean) / std
        if inverse and self.keep_ocean_as_nan:
            out[..., ~self.loss_mask] = np.nan
        return out

    def latitude(self): return self._lat
    def longitude(self): return self._lon
    def image_shape(self): return self.img_shape_y, self.img_shape_x
    def input_channels(self): return [ChannelMetadata(x) for x in self.input_variables]
    def output_channels(self): return [ChannelMetadata(x) for x in self.output_variables]
    def normalize_input(self, x): return self._normalize(x, self.input_variables, "input")
    def normalize_output(self, x): return self._normalize(x, self.output_variables, "output")
    def denormalize_input(self, x): return self._normalize(x, self.input_variables, "input", True)
    def denormalize_output(self, x): return self._normalize(x, self.output_variables, "output", True)
    def get_loss_mask(self): return self.loss_mask
    def get_ocean_mask_with_buffer(self): return self.ocean_mask_with_buffer
    def get_loss_mask_tensor(self, device=None):
        return torch.from_numpy(self.loss_mask.astype(np.float32))[None, None].to(device=device)
    def get_precip_threshold_info(self):
        for i, name in enumerate(self.output_variables):
            if name in {"APCP_surface", "RAIN_TOTAL", "PREC_ACC_NC"}:
                entry = self.stats["output"][name]
                return {"channel_idx": i, "threshold_norm": (self.precip_threshold - entry["mean"]) / entry["std"], "zero_norm": -entry["mean"] / entry["std"]}
        return None
    def info(self):
        return {"input_normalization": self.stats["input"], "output_normalization": self.stats["output"],
                "input_variables": self.input_variables, "output_variables": self.output_variables,
                "numeric_profile": "physical", "train": self.train, "num_samples": len(self)}
    def time(self):
        return [cftime.DatetimeGregorian(*datetime.datetime.fromtimestamp(int(t) / 1e9, datetime.timezone.utc).timetuple()[:6]) for t in self._time_values]
