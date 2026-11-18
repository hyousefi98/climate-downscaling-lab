"""Use physical inputs with an explicitly bound historical AORC checkpoint.

The checkpoint coordinate is 10 times specific humidity, with hPa pressure.
Input/reference physical values are mapped into that coordinate before applying
the frozen normalization. NetCDF values are denormalized back to physical q.
No statistics are fitted, recomputed, or substituted during this comparison.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from models.datasets.physical import PhysicalPairDataset

CHANNELS = ["APCP_surface", "DLWRF_surface", "DSWRF_surface", "PRES_surface",
            "SPFH_2maboveground", "TMP_2maboveground", "UGRD_10maboveground", "VGRD_10maboveground"]
WRF_CHANNELS = ["RAIN_TOTAL", "GLW", "SWDOWN", "PSFC", "Q2", "T2", "U10", "V10"]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_interface(interface_path, artifacts):
    with open(interface_path) as handle:
        declaration = json.load(handle)
    if declaration.get("model_coordinate") != "aorc_legacy_10q_hpa":
        raise ValueError("Frozen interface must explicitly declare aorc_legacy_10q_hpa")
    for name, path in artifacts.items():
        if file_sha256(path) != declaration.get("sha256", {}).get(name):
            raise ValueError(f"Frozen checkpoint interface hash mismatch: {name}")
    return declaration


class FrozenAORCPairDataset(PhysicalPairDataset):
    def __init__(self, checkpoint_interface_path, input_source, stats_path, **kwargs):
        if input_source not in {"synthetic", "wrf"}:
            raise ValueError("input_source must be synthetic or wrf")
        self.interface = verify_interface(checkpoint_interface_path, {"normalization": stats_path})
        self.input_source = input_source
        kwargs.pop("input_variables", None)
        kwargs.pop("output_variables", None)
        super().__init__(stats_path=stats_path, input_variables=CHANNELS, output_variables=CHANNELS, **kwargs)
        for store in (self.input_store.group, self.target_store.group):
            if store.attrs.get("humidity_representation") != "specific_humidity" or store.attrs.get("pressure_units") != "hPa":
                raise ValueError("Frozen transfer requires explicit specific_humidity and hPa source metadata")
            if store.attrs.get("temporal_profile") != "ending":
                raise ValueError("Frozen transfer requires common end-labeled six-hour windows")

    def validate_statistics_profile(self, metadata):
        # The original external statistics may predate profile metadata. Their
        # bytes must match the explicitly declared, frozen checkpoint interface.
        if metadata.get("numeric_profile") == "physical":
            raise ValueError("New physical normalization cannot replace frozen checkpoint normalization")

    def read_values(self, side, source, index, names):
        source_names = WRF_CHANNELS if side == "input" and self.input_source == "wrf" else CHANNELS
        return source.read(index, source_names, self.image_shape())

    def _normalize(self, values, names, group, inverse=False):
        values = np.asarray(values, dtype=np.float32).copy()
        q_index = names.index("SPFH_2maboveground")
        if not inverse:
            values[..., q_index, :, :] *= np.float32(10.)
        result = super()._normalize(values, names, group, inverse)
        if inverse:
            result[..., q_index, :, :] /= np.float32(10.)
        return result
