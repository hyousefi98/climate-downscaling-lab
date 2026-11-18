"""Strict coordinate/reference validation for CorrDiff-style grouped NetCDF files."""
from __future__ import annotations

import json
import numpy as np
import netCDF4


def read_array(variable, key=slice(None)):
    return np.asarray(np.ma.filled(variable[key], np.nan))


def timestamps(dataset):
    v = dataset.variables["time"]
    t = netCDF4.num2date(v[:], v.units, getattr(v, "calendar", "standard"))
    result = [x.strftime("%Y-%m-%dT%H:%M:%S.%f") for x in t]
    if len(set(result)) != len(result):
        raise ValueError("Duplicate timestamps are not allowed in scored files")
    return result


def validate_grid(reference, candidate):
    calendars = [getattr(d['time'], 'calendar', 'standard') for d in (reference,candidate)]
    calendars = ['standard' if x == 'gregorian' else x for x in calendars]
    if calendars[0] != calendars[1]:
        raise ValueError("Calendar conventions differ")
    if timestamps(reference) != timestamps(candidate):
        raise ValueError("Timestamp values or ordering differ; no implicit intersection is allowed")
    for name in ("lat", "lon"):
        if name not in reference.variables or name not in candidate.variables:
            raise ValueError(f"Missing explicit {name} coordinates")
        a, b = read_array(reference[name]), read_array(candidate[name])
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
            raise ValueError(f"{name} coordinates differ")


def validate_reference(reference, candidate):
    """Require identical values AND finite masks, not just the same product name."""
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
        raise ValueError("Stored truth values or masks differ from the common reference")
    if np.isinf(a).any() or not np.isfinite(a).any():
        raise ValueError("Reference is infinite or has no valid values")


def time_slice(variable, index):
    dims = tuple(variable.dimensions)
    if "time" not in dims:
        raise ValueError("Scored variables require an explicit time dimension")
    key = [slice(None)] * len(dims)
    key[dims.index("time")] = index
    a = read_array(variable, tuple(key))
    remaining = tuple(x for x in dims if x != "time")
    if remaining == ("y", "x"):
        return a[None]
    if remaining == ("ensemble", "y", "x"):
        return a
    raise ValueError(f"Unsupported dimensions {dims}; expected ensemble,time,y,x or time,y,x")


def unit_map(path, variables):
    """Explicit physical scaling; model humidity can require factor 0.1.

    A single declaration is applied identically to reference and predictions.
    Different conventions must be converted before forming a common-reference run.
    """
    mapping = json.loads(path.read_text()) if path is not None else {}
    unknown = set(mapping) - set(variables)
    if unknown:
        raise ValueError(f"Unit declarations for unknown variables: {sorted(unknown)}")
    for v in variables:
        item = mapping.setdefault(v, {"factor": 1.0, "units": "stored units"})
        if not isinstance(item, dict) or not isinstance(item.get('factor'),(int,float)) or isinstance(item.get('factor'),bool) or not np.isfinite(item.get("factor", np.nan)) or item["factor"] <= 0 or not item.get("units"):
            raise ValueError(f"Invalid explicit units for {v}")
    return mapping
