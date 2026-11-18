# Model interfaces and preserved experiments

The three experiment variants share byte-identical training helpers, but retain
their distinct loaders, losses, generators and configurations. Install the pinned
PhysicsNeMo revision and supplied framework patch described in the repository
README before running them. Stock PhysicsNeMo is not assumed equivalent.

| Experiment | Input / output fields | Crop | Preserved implementation |
| --- | --- | --- | --- |
| AORC synthetic | 8 / 8 | 288 × 480 | `legacy/aorc` |
| WRF paired | 8 / 8 | 288 × 480 | `legacy/wrf` |
| ERA5–CONUS404 | 35 / 8 | 224 × 352 | `legacy/era5_conus404` |

ERA5 has seven surface and 28 pressure-level predictors. Ordered channel lists
are explicit in each configuration. Four coordinate channels are added inside
the model; diffusion also receives the eight-channel regression estimate.
Checkpoint architecture metadata is in `configs/checkpoint_architectures.yaml`.
No checkpoint weights, fitted normalization, or fitted residual scales are
included.

## New training with corrected data

`workflows.train` and `workflows.generate` default to `--profile corrected`.
They consume the physical Zarr stores produced by `scripts/preprocess`: explicit
`numeric_profile=physical`, pressure in hPa, specific humidity in kg/kg, common
ending six-hour windows, matching coordinates, and finite values on the common
land/support mask. Coordinate-bearing land masks are checked against the data.
Statistics must declare the same physical profile, precipitation cutoff and
spatial support, along with their included training timestamps and matching
temporal/unit metadata. The selected training timestamps must match that
membership; explicit requested subsets are allowed. Evaluation timestamps must
be disjoint from it. Invalid land values fail before model execution; they are not
silently replaced by zero. Explicitly dropped training timestamps recorded by
the statistics workflow remain excluded during training.
The supplied corrected profiles consistently use a five-pixel AORC coastal
buffer and zero pixels for WRF/ERA5 during both training and generation. An
explicit custom configuration or `--set` may change this, but its statistics
must use the resulting support. Historical AORC generation retains ten pixels.

```bash
python -m workflows.train --experiment aorc --stage regression --dry-run
python -m workflows.train --experiment aorc --stage regression \
  --run-dir /external/runs/aorc_regression \
  --set dataset.input_data_path=/external/data/aorc_input.zarr \
  --set dataset.target_data_path=/external/data/aorc_reference.zarr \
  --set dataset.stats_path=/external/artifacts/aorc_physical_stats.json \
  --set dataset.landmask_path=/external/data/aorc_landmask.nc
```

Use the actual checkpoint written by the regression run when starting diffusion:
`--set training.io.regression_checkpoint_path=/external/runs/.../UNet.0.N.mdlus`.
For AORC, compute new residual RMS scales using `python -m
workflows.residual_scales --help`, then supply the external result as
`training.hp.residual_std_path`. These are the RMS of normalized residuals, not
centered standard deviations. New physical normalization and residual scales
belong with newly trained weights; they must not silently replace the old
checkpoint's artifacts.

`--experiment wrf` and `--experiment era5_conus404` select the corresponding
preserved training variants. `--set KEY=VALUE` accepts repeated configuration
overrides. For multiple GPUs, launch the same module with `torchrun`; all ranks
must see the same run directory. A fresh file rendezvous lets rank 0 atomically
write the configuration before any rank initializes the model runtime. A
nonempty training run requires explicit `--resume`.

## Historical profiles and November 18 matched comparison

`--profile legacy` retains the archived data conventions with strict diagnostic
checks enabled. Only `--allow-invalid-legacy-data` restores the old treatment of
invalid source values for an explicitly requested historical replay. The known
AORC missing-value contamination is present in historical training; a better
inference adapter cannot repair already trained weights.

The November 18 matched comparison has its own exact packed-data route:

1. Build aligned inputs and the canonical reference using
   `scripts/preprocess/matched_reference.py`; see `data/README.md`. This retains
   the historical grid labels and original float64/float32 operation order.
2. Bind the external historical artifacts with `workflows.checkpoint_interface`.
   This hashes the regression checkpoint, diffusion checkpoint, original
   normalization, and original residual RMS file. The declaration establishes
   artifact identity, not training validity or provenance.
3. Run the preserved adapter and generator:

```bash
python -m workflows.reproduce_matched \
  --aligned /external/aligned_inputs.nc \
  --wrf-input /external/wrf_12km_on_aorc_3km.zarr \
  --times /external/requested_times.json \
  --normalization /external/aorc_stats_3km.json \
  --residual-scale /external/residual_std_aorc.json \
  --regression /external/UNet.0.200000.mdlus \
  --diffusion /external/EDMPrecondSuperResolution.0.400000.mdlus \
  --interface /external/checkpoint_interface.json \
  --run-dir /external/runs/matched
```

`--dry-run` prints the plan without opening artifacts. `--validate-only` checks
all 200 paired inputs and the shared target on CPU without launching inference.
The preserved adapter caches about 0.9 GiB per AORC group to avoid concurrent
HDF5 reads and background writes; the two-adapter preflight needs roughly 3 GiB
plus runtime overhead. Inference runs the two arms sequentially on the visible
GPU, using seeds 0–3, one member per batch, and 18 stochastic sampler steps.
This launcher neither selects occupied GPUs nor signals other jobs.

The exact packed route writes humidity in the old **10 × specific-humidity**
coordinate and pressure in hPa. Divide humidity by ten when computing physical
metrics. Both arms share the original AORC normalization, residual scaling,
reference, dates, and mask. WRF pressure is converted from Pa to hPa; archived
WRF Q2 is interpreted as mixing ratio and mapped to `10 × r/(1+r)`. WRF wind
rotation history and the nonprecipitation temporal aggregation history remain
unverified.

`workflows.matched_transfer` is a separate future-data route: it consumes new
physical stores, explicitly maps q to the frozen 10q checkpoint coordinate,
and writes physical q. It does not promise identical results to the historical
grid/packed route.

## Provenance limits and verification

Generation settings were recovered from saved output configuration metadata,
rather than assuming current mutable YAML was the paper configuration. Training
YAML snapshots and architecture metadata do not establish the complete training
history. WRF paper inference was run from the v4 experiment, while its saved
diffusion path names a now-missing v6 checkpoint directory. The available v4
800000-step checkpoint is compatible, but its byte identity to that missing
artifact has not been established. Do not describe the v3 checkpoint currently
named in an old mutable YAML as the verified paper model.

CPU integration tests exercise generated corrected data through statistics,
normalization, an actual U-Net backward/update step, a diffusion forward pass,
and scoring. They also check time/calendar/grid rejection, physical versus
checkpoint humidity units, legacy QC, and multi-rank launch coordination.
`python -m pytest models/tests` runs these tests when the optional model
dependencies are installed; framework tests skip cleanly without them.
