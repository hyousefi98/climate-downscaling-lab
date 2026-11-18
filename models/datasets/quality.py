"""Explicit quality policy for historical packed-number readers."""
import hashlib
import os
from pathlib import Path
import shutil

import numpy as np
from models.datasets.physical import validate_physical


def runtime_cache_path(original):
    """Keep loader cache writes away from source data and installed code."""
    source = Path(original)
    root = Path(os.environ.get("DG_CACHE_ROOT", "cache")).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / (hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16] + ".valid_indices.npz")
    if source.is_file() and not destination.exists():
        shutil.copyfile(source, destination)
    return str(destination)


def check_legacy_arrays(dataset, inputs, targets):
    policy = getattr(dataset, "quality_policy", "strict")
    if policy == "archive_replay":
        return
    if policy != "strict":
        raise ValueError(f"Unknown quality policy: {policy}")
    mask = getattr(dataset, "loss_mask", None)
    if mask is None:
        mask = np.ones(inputs.shape[-2:], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    for group, values, names in (("input", inputs, dataset.input_variables), ("output", targets, dataset.output_variables)):
        values = values.copy()
        for i, name in enumerate(names):
            # Check physical units, while leaving original model inputs untouched.
            if name == "TMP_2maboveground": values[i] *= np.float32(.1)
            elif name in {"UGRD_10maboveground", "VGRD_10maboveground"}: values[i] *= np.float32(.1)
            elif name == "SPFH_2maboveground": values[i] *= np.float32(.0001)
            elif name == "PRES_surface": values[i] *= np.float32(.1)
            elif name == "PSFC":
                is_era = dataset.__class__.__module__.endswith("era5_conus404_legacy")
                if not is_era or group == "output": values[i] *= np.float32(.01)
        validate_physical(values, names, mask, "historical " + group)
