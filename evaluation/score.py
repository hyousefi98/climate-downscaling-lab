"""Stream all prediction files against ONE supplied reference, without re-alignment.

Example: python -m evaluation.score --reference reference.nc --prediction a.nc b.nc
  --output scores.csv --reference-id AORC_3km --require-stored-truth
  --expected-members 4 --units-map units.json

Run from an installed checkout. Output paths are supplied explicitly. Models with
different members may be compared without --expected-members, but paired transfer
experiments should declare their expected common ensemble size. This CLI never
applies a new precipitation cutoff after the ensemble mean.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path

import netCDF4
import numpy as np

from .io import read_array, timestamps, time_slice, unit_map, validate_grid, validate_reference
from .metrics import Accumulator, ensemble_mean


def score_files(reference_path, predictions, output_path, *, reference_id,
                units_path=None, expected_members=None, require_stored_truth=False,
                require_unit_attributes=False):
    reference_path, output_path = Path(reference_path), Path(output_path)
    paths = [Path(p) for p in predictions]
    if not reference_id or not paths or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("A reference identifier and distinct prediction paths are required")
    if output_path.resolve() in {reference_path.resolve(), *(p.resolve() for p in paths)}:
        raise ValueError("Output may not overwrite an input")
    if output_path.exists() or output_path.with_suffix(".audit.json").exists():
        raise FileExistsError("Choose a fresh output path")
    if expected_members is not None and expected_members < 1:
        raise ValueError("Expected members must be positive")
    rows, source_units = [], {}
    with ExitStack() as stack:
        ref = stack.enter_context(netCDF4.Dataset(reference_path))
        if "truth" not in ref.groups:
            raise ValueError("Common reference needs a truth group")
        variables = list(ref.groups["truth"].variables)
        if not variables:
            raise ValueError("Empty truth group")
        ntime = len(timestamps(ref))
        if not ntime:
            raise ValueError("Reference has no timestamps")
        units = unit_map(units_path, variables)
        for path in paths:
            with netCDF4.Dataset(path) as ds:
                validate_grid(ref, ds)
                if "prediction" not in ds.groups or not set(variables) <= set(ds.groups["prediction"].variables):
                    raise ValueError("Predictions are missing reference variables")
                for var in variables:
                    acc = Accumulator()
                    pvar = ds.groups["prediction"][var]
                    reference_units = getattr(ref.groups['truth'][var], 'units', None)
                    prediction_units = getattr(pvar, 'units', None)
                    source_units.setdefault(str(path),{})[var]={'reference':reference_units,'prediction':prediction_units}
                    if require_unit_attributes and (not reference_units or not prediction_units):
                        raise ValueError(f"{var}: source unit attributes are required on both files")
                    if reference_units and prediction_units and reference_units != prediction_units:
                        raise ValueError(f"{var}: source unit attributes differ; convert inputs explicitly")
                    members = pvar.shape[pvar.dimensions.index("ensemble")] if "ensemble" in pvar.dimensions else 1
                    if expected_members is not None and members != expected_members:
                        raise ValueError(f"{path.name}: unexpected ensemble size {members}")
                    for t in range(ntime):
                        o = time_slice(ref.groups["truth"][var], t)
                        if o.shape[0] != 1:
                            raise ValueError("Reference truth must have exactly one field per time")
                        o = o[0]
                        if np.isinf(o).any() or not np.isfinite(o).any():
                            raise ValueError("Invalid reference field")
                        if require_stored_truth:
                            if "truth" not in ds.groups or var not in ds.groups["truth"].variables:
                                raise ValueError("Prediction file lacks required stored truth")
                            candidate = time_slice(ds.groups["truth"][var], t)
                            if candidate.shape[0] != 1:
                                raise ValueError("Stored truth has an ensemble dimension")
                            validate_reference(o, candidate[0])
                        mask = np.isfinite(o)
                        mean = ensemble_mean(time_slice(pvar, t), mask)
                        factor = units[var]["factor"]
                        acc.add(o.astype(np.float64)*factor, mean.astype(np.float64)*factor, mask)
                    rows.append({"prediction_file": str(path), "reference_id": reference_id,
                                 "variable": var, "units": units[var]["units"],
                                 "members": members, **acc.result()})
        audit = {"reference_file": str(reference_path), "reference_id": reference_id,
                 "timestamps": timestamps(ref), "units": units,
                 "stored_truth_required": require_stored_truth,
                 "ensemble_mean_dtype": "float32", "post_mean_threshold": None,
                 "mask": "finite common reference; all members required finite within mask",
                 "PBIAS_denominator": "sum(abs(reference))"}
        audit.update(source_units=source_units, require_unit_attributes=require_unit_attributes,
                     unit_policy='Units map declares one common multiplicative conversion. Missing source attributes are recorded, not independently validated. Differing input conventions must be converted beforehand.',
                     correlation_definition='Pooled time-space Pearson correlation of fields, not anomaly correlation.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with output_path.with_suffix(".audit.json").open("x") as f:
        json.dump(audit, f, indent=2, allow_nan=False)
        f.write("\n")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--prediction", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-id", required=True)
    p.add_argument("--units-map", type=Path)
    p.add_argument("--expected-members", type=int)
    p.add_argument("--require-stored-truth", action="store_true")
    p.add_argument("--require-unit-attributes", action="store_true")
    a = p.parse_args()
    score_files(a.reference, a.prediction, a.output, reference_id=a.reference_id,
                units_path=a.units_map, expected_members=a.expected_members,
                require_stored_truth=a.require_stored_truth, require_unit_attributes=a.require_unit_attributes)


if __name__ == "__main__":
    main()
