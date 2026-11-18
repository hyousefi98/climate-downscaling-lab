#!/usr/bin/env python3
"""Score two new, corrected runs against one exact prepared AORC reference.

The historical baseline score is never reused. Each run must contain the same 200
requested times and four ensemble members. Reference/truth values and masks must
match exactly before scores are calculated. Humidity is converted from the
model's 10-times-specific-humidity convention to kg/kg; pressure remains hPa.
"""
import argparse
import ast
from contextlib import ExitStack
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import netCDF4 as nc
import numpy as np


VARIABLES = {
    "APCPsurface": {"units": "mm", "physical_factor": 1.0, "meaning": "six-hour precipitation accumulation"},
    "DLWRFsurface": {"units": "W m-2", "physical_factor": 1.0, "meaning": "six-hour mean downward longwave radiation"},
    "DSWRFsurface": {"units": "W m-2", "physical_factor": 1.0, "meaning": "six-hour mean downward shortwave radiation"},
    "PRESsurface": {"units": "hPa", "physical_factor": 1.0, "meaning": "six-hour mean surface pressure; retained in model's hPa units"},
    "SPFH2maboveground": {"units": "kg kg-1", "physical_factor": 0.1, "meaning": "six-hour mean specific humidity; divide stored model values by 10"},
    "TMP2maboveground": {"units": "K", "physical_factor": 1.0, "meaning": "six-hour mean 2-m temperature"},
    "UGRD10maboveground": {"units": "m s-1", "physical_factor": 1.0, "meaning": "six-hour mean 10-m eastward wind"},
    "VGRD10maboveground": {"units": "m s-1", "physical_factor": 1.0, "meaning": "six-hour mean 10-m northward wind"},
}


def array(var, selection=slice(None)):
    return np.asarray(np.ma.filled(var[selection], np.nan))


def dates(ds):
    var = ds["time"]
    return [t.strftime("%Y-%m-%dT%H:%M:%S") for t in
            nc.num2date(var[:], var.units, getattr(var, "calendar", "standard"))]


def load_config(ds):
    if "cfg" not in ds.ncattrs():
        raise ValueError("Inference file has no cfg attribute to verify ensemble settings")
    try:
        cfg = ast.literal_eval(ds.getncattr("cfg"))
    except (ValueError, SyntaxError):
        cfg = json.loads(ds.getncattr("cfg"))
    if not isinstance(cfg, dict):
        raise ValueError("cfg must contain a dictionary")
    return cfg


def generation_settings(cfg):
    value = copy.deepcopy(cfg["generation"])
    value.pop("times", None)
    value.setdefault("io", {}).pop("output_filename", None)
    return value


def validate_structure(reference, runs, requested, expected_count):
    if len(requested) != expected_count or len(set(requested)) != expected_count:
        raise ValueError("Expected timestamp list has incorrect length or duplicates")
    if dates(reference) != requested:
        raise ValueError("Canonical reference timestamps do not match the requested order")
    if set(reference.groups["truth"].variables) != set(VARIABLES):
        raise ValueError("Canonical reference must contain precisely the eight expected AORC truth variables")
    configs = {}
    for label, ds in runs.items():
        if dates(ds) != requested:
            raise ValueError(f"{label}: timestamp order differs from canonical reference")
        if len(ds.dimensions["ensemble"]) != 4:
            raise ValueError(f"{label}: expected four ensemble members")
        for coord in ("lat", "lon"):
            if not np.array_equal(array(ds[coord]), array(reference[coord]), equal_nan=True):
                raise ValueError(f"{label}: {coord} grid differs from canonical reference")
        cfg = load_config(ds)
        generation = cfg["generation"]
        if generation.get("num_ensembles") != 4 or generation.get("seed_offset", 0) != 0:
            raise ValueError(f"{label}: expected four members with seeds 0, 1, 2, 3")
        if generation.get("inference_mode") != "all":
            raise ValueError(f"{label}: expected combined regression and diffusion inference")
        for group in ("truth", "prediction"):
            if set(ds.groups[group].variables) != set(VARIABLES):
                raise ValueError(f"{label}: unexpected variables in {group}")
        configs[label] = cfg
    first, second = configs.values()
    if generation_settings(first) != generation_settings(second) or first.get("sampler") != second.get("sampler"):
        raise ValueError("The two runs used different generation/checkpoint/sampler settings")
    return configs


def validate_truth(reference, candidate, label, expected_valid_cells):
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError(f"{label}: truth shape/dtype differs from canonical reference")
    if not np.array_equal(reference, candidate, equal_nan=True):
        raise ValueError(f"{label}: truth values or missing-data mask differ from canonical reference")
    if np.isinf(reference).any():
        raise ValueError(f"{label}: infinite reference values")
    mask = np.isfinite(reference)
    if int(mask.sum()) != expected_valid_cells:
        raise ValueError(f"{label}: reference has {int(mask.sum())} valid cells; expected {expected_valid_cells}")
    return mask


def prediction_mean(var, index, mask, label):
    if var.dimensions != ("ensemble", "time", "y", "x") or var.dtype != np.dtype("float32"):
        raise ValueError(f"{label}: prediction must be float32 (ensemble,time,y,x)")
    members = array(var, (slice(None), index, slice(None), slice(None)))
    if not np.isfinite(members[:, mask]).all():
        raise ValueError(f"{label}: missing/non-finite ensemble values on the fixed reference mask")
    if not np.isnan(members[:, ~mask]).all():
        raise ValueError(f"{label}: prediction mask differs from the canonical reference mask")
    # This float32 mean exactly follows the original baseline scoring convention.
    # No new precipitation threshold, clipping or rounding follows averaging.
    return members[:, mask].mean(axis=0, dtype=np.float32)


class Accumulator:
    def __init__(self):
        self.n = 0
        self.mean_o = self.mean_p = 0.0
        self.m2_o = self.m2_p = self.cross = 0.0
        self.sse = self.sae = self.error_sum = self.abs_obs_sum = 0.0

    def add(self, obs, pred):
        obs, pred = np.asarray(obs, dtype=np.float64), np.asarray(pred, dtype=np.float64)
        if obs.size == 0 or obs.shape != pred.shape or not np.all(np.isfinite(obs) & np.isfinite(pred)):
            raise ValueError("Invalid scoring arrays")
        n = obs.size
        mo, mp = float(obs.mean()), float(pred.mean())
        do, dp = obs - mo, pred - mp
        total, weight = self.n + n, self.n * n / (self.n + n)
        delta_o, delta_p = mo - self.mean_o, mp - self.mean_p
        self.m2_o += float(np.dot(do, do)) + weight * delta_o**2
        self.m2_p += float(np.dot(dp, dp)) + weight * delta_p**2
        self.cross += float(np.dot(do, dp)) + weight * delta_o * delta_p
        self.mean_o += delta_o * n / total
        self.mean_p += delta_p * n / total
        self.n = total
        error = pred - obs
        self.sse += float(np.dot(error, error))
        self.sae += float(np.abs(error).sum())
        self.error_sum += float(error.sum())
        self.abs_obs_sum += float(np.abs(obs).sum())

    def result(self):
        return {
            "valid_count": self.n, "RMSE": float(np.sqrt(self.sse / self.n)),
            "MAE": self.sae / self.n,
            "PBIAS_%": 100 * self.error_sum / self.abs_obs_sum if self.abs_obs_sum else None,
            "Pearson_r": float(np.clip(self.cross / np.sqrt(self.m2_o * self.m2_p), -1.0, 1.0)) if self.m2_o > 0 and self.m2_p > 0 else None,
        }


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_pair(synthetic_path, wrf_path, reference_path, expected_times_path, output_dir,
               expected_count=200, expected_valid_cells=128619):
    paths = {"synthetic_input": Path(synthetic_path), "WRF_input": Path(wrf_path)}
    if paths["synthetic_input"].resolve() == paths["WRF_input"].resolve():
        raise ValueError("Two distinct inference files are required")
    reference_path, output_dir = Path(reference_path), Path(output_dir)
    requested = json.loads(Path(expected_times_path).read_text())
    if not isinstance(requested, list) or not all(isinstance(t, str) for t in requested):
        raise ValueError("Expected-times JSON must be a list of ISO timestamp strings")
    requested = [t.replace(" ", "T") for t in requested]
    rows, per_time, reference_hashes, mask_hashes = [], [], {}, {}
    with ExitStack() as stack:
        reference = stack.enter_context(nc.Dataset(reference_path, "r"))
        runs = {label: stack.enter_context(nc.Dataset(path, "r")) for label, path in paths.items()}
        configs = validate_structure(reference, runs, requested, expected_count)
        for variable, info in VARIABLES.items():
            truth_var = reference.groups["truth"][variable]
            if truth_var.dimensions != ("time", "y", "x") or truth_var.dtype != np.dtype("float32"):
                raise ValueError(f"Canonical {variable}: expected float32 (time,y,x)")
            accumulators = {label: Accumulator() for label in runs}
            reference_hash, mask_hash = hashlib.sha256(), hashlib.sha256()
            for index, timestamp in enumerate(requested):
                obs_model = array(truth_var, index)
                predictions = {}
                for label, run in runs.items():
                    new_truth = run.groups["truth"][variable]
                    if new_truth.dimensions != truth_var.dimensions:
                        raise ValueError(f"{label}/{variable}: truth dimensions differ")
                    mask = validate_truth(obs_model, array(new_truth, index),
                                          f"{label}/{variable}/{timestamp}", expected_valid_cells)
                    predictions[label] = prediction_mean(run.groups["prediction"][variable], index, mask,
                                                         f"{label}/{variable}/{timestamp}")
                reference_hash.update(np.ascontiguousarray(obs_model).tobytes())
                mask_hash.update(np.ascontiguousarray(mask).tobytes())
                factor = info["physical_factor"]
                obs = obs_model[mask].astype(np.float64) * factor
                for label, mean in predictions.items():
                    pred = mean.astype(np.float64) * factor
                    accumulators[label].add(obs, pred)
                    current = Accumulator(); current.add(obs, pred)
                    per_time.append({"experiment": label, "variable": variable, "units": info["units"],
                                     "time_index": index, "time": timestamp, **current.result()})
            reference_hashes[variable], mask_hashes[variable] = reference_hash.hexdigest(), mask_hash.hexdigest()
            for label, accumulator in accumulators.items():
                row = {"experiment": label, "variable": variable, "units": info["units"], **accumulator.result()}
                rows.append(row)
                print(json.dumps(row), flush=True)
    precip = {row["experiment"]: row for row in rows if row["variable"] == "APCPsurface"}
    denominator = precip["synthetic_input"]["RMSE"]
    audit = {
        "inputs": {label: {"path": str(path.resolve()), "bytes": path.stat().st_size} for label, path in paths.items()},
        "reference": {"path": str(reference_path.resolve()), "group": "truth", "bytes": reference_path.stat().st_size},
        "time_count": expected_count, "ensemble_members": 4, "seeds": [0, 1, 2, 3],
        "valid_cells_per_case": expected_valid_cells, "timestamps": requested,
        "reference_model_value_sha256": reference_hashes, "reference_mask_sha256": mask_hashes,
        "exact_truth_reference_check": "passed for both new outputs, every variable and timestamp",
        "identical_generation_settings_check": "passed", "configs": configs,
        "variable_units_and_conversions": VARIABLES,
        "interpretation_limitations": [
            "WRF U10 and V10 are retained as archived; whether the archived winds were rotated to earth-relative coordinates remains unverified.",
            "Conversion of WRF Q2 to specific humidity assumes the archive retains the standard WRF water-vapor mixing-ratio definition; the original archive's conversion history is incomplete.",
            "The temporal averaging semantics of archived WRF nonprecipitation variables remain unverified; matching timestamps does not itself establish identical temporal averaging.",
            "The 2012 WRF precipitation interval alignment is supported by the available provenance and timing checks, but includes an inference about the archived accumulation convention.",
            "Matched reference values, timestamps, masks, weights, and generation settings do not remove every possible difference between the source datasets; the RMSE ratio is a paired performance comparison, not proof that all remaining differences have one cause.",
        ],
        "methods": {
            "temporal_support": "Prepared canonical AORC reference: hour labels t-5 through t inclusive; precipitation sum, other variables mean.",
            "comparison": "Both newly generated runs are scored against identical prepared AORC truth; no historical score enters any calculation.",
            "ensemble": "Arithmetic mean of four float32 members (seeds 0-3); convert mean and truth to documented physical units before float64 metric accumulation.",
            "precipitation": "No additional threshold, clipping or rounding after ensemble averaging; upstream preprocessing is recorded in the run configs.",
            "humidity": "Stored model values represent ten times specific humidity. Divide both predictions and reference by 10 for kg/kg metrics.",
            "pressure": "Stored model pressure is hPa and is scored/labeled as hPa. No conversion to Pa is applied.",
            "support": "Exact truth values and NaN masks must match the canonical reference in both runs; all four prediction members must have the same mask, with no missing reference cells.",
            "global_metrics": "Pool every valid grid-time cell using float64 sums and stable combined covariance; per-case metrics are also supplied.",
            "PBIAS": "100 * sum(prediction - reference) / sum(abs(reference))",
            "RMSE_ratio": "New WRF-input RMSE divided by new synthetic-input RMSE, both using the same reference and support.",
        },
        "results": rows,
        "precipitation_RMSE_ratio_WRF_over_synthetic": precip["WRF_input"]["RMSE"] / denominator if denominator else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "basic.csv", rows)
    write_csv(output_dir / "per_time.csv", per_time)
    (output_dir / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    return audit


def self_test():
    # Independent formulas verify stable accumulation, including nonconstant errors.
    obs, pred = np.array([1., 3., 2., 6.]), np.array([2., 1., 4., 7.])
    acc = Accumulator(); acc.add(obs[:2], pred[:2]); acc.add(obs[2:], pred[2:])
    got = acc.result()
    assert np.isclose(got["RMSE"], np.sqrt(np.mean((pred - obs)**2)))
    assert np.isclose(got["MAE"], np.mean(np.abs(pred - obs)))
    assert np.isclose(got["PBIAS_%"], 100 * np.sum(pred - obs) / np.sum(np.abs(obs)))
    assert np.isclose(got["Pearson_r"], np.corrcoef(obs, pred)[0, 1])
    with tempfile.TemporaryDirectory(prefix="aies-corrected-score-test-") as temporary:
        root = Path(temporary)
        expected = ["2012-01-01T00:00:00", "2012-01-01T06:00:00"]
        expected_path = root / "times.json"; expected_path.write_text(json.dumps(expected))
        def fixture(path, error=None):
            with nc.Dataset(path, "w") as ds:
                for name, size in (("time", 2), ("y", 2), ("x", 3)):
                    ds.createDimension(name, size)
                if error is not None:
                    ds.createDimension("ensemble", 4)
                    ds.cfg = str({"generation": {"times": expected, "num_ensembles": 4,
                                  "seed_batch_size": 1, "inference_mode": "all", "io": {"output_filename": str(path)}},
                                  "sampler": {"type": "stochastic", "num_steps": 18}})
                t = ds.createVariable("time", "i8", ("time",)); t.units = "hours since 2012-01-01 00:00:00"; t[:] = [0, 6]
                for coord in ("lat", "lon"):
                    ds.createVariable(coord, "f4", ("y", "x"))[:] = np.arange(6).reshape(2, 3)
                truth = ds.createGroup("truth")
                prediction = ds.createGroup("prediction") if error is not None else None
                for name in VARIABLES:
                    values = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 8
                    values[:, 1, 2] = np.nan
                    truth.createVariable(name, "f4", ("time", "y", "x"))[:] = values
                    if prediction is not None:
                        prediction.createVariable(name, "f4", ("ensemble", "time", "y", "x"))[:] = np.broadcast_to(values + error, (4, 2, 2, 3))
        reference, synthetic, wrf = (root / name for name in ("reference.nc", "synthetic.nc", "wrf.nc"))
        # Binary-exact errors include precipitation predictions below 0.1 mm;
        # the expected score would fail if an extra threshold were introduced.
        fixture(reference); fixture(synthetic, 0.03125); fixture(wrf, 0.0625)
        audit = score_pair(synthetic, wrf, reference, expected_path, root / "good", 2, 5)
        humidity = next(r for r in audit["results"] if r["experiment"] == "synthetic_input" and r["variable"] == "SPFH2maboveground")
        assert np.isclose(humidity["RMSE"], 0.003125) and humidity["units"] == "kg kg-1"
        precipitation = next(r for r in audit["results"] if r["experiment"] == "synthetic_input" and r["variable"] == "APCPsurface")
        assert np.isclose(precipitation["RMSE"], 0.03125)
        assert np.isclose(audit["precipitation_RMSE_ratio_WRF_over_synthetic"], 2)
        for mutation in ("truth", "mask", "time"):
            fixture(wrf, 0.25)
            with nc.Dataset(wrf, "a") as ds:
                if mutation == "truth":
                    ds["truth"]["APCPsurface"][0, 0, 0] = 1.0
                elif mutation == "mask":
                    ds["prediction"]["APCPsurface"][0, 0, 0, 0] = np.nan
                else:
                    ds["time"][:] = [6, 0]
            try:
                score_pair(synthetic, wrf, reference, expected_path, root / mutation, 2, 5)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid {mutation} comparison was accepted")
    print("Self-test passed: independent metrics, paired reference, humidity conversion, truth/mask/time rejection", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path)
    parser.add_argument("--wrf", type=Path)
    parser.add_argument("--reference", type=Path, help="Canonical reference_model_units.nc with truth group")
    parser.add_argument("--expected-times", type=Path, help="JSON list of the exact 200 requested timestamp strings")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not all((args.synthetic, args.wrf, args.reference, args.expected_times, args.output_dir)):
            parser.error("--synthetic, --wrf, --reference, --expected-times and --output-dir are required")
        result = score_pair(args.synthetic, args.wrf, args.reference, args.expected_times, args.output_dir)
        print(json.dumps({"output_dir": str(args.output_dir), "precipitation_RMSE_ratio_WRF_over_synthetic":
                          result["precipitation_RMSE_ratio_WRF_over_synthetic"]}), flush=True)
