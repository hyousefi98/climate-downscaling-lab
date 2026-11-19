# Climate Downscaling Lab

Research code for **Synthetic benchmarks overstate the real-world skill of
deep-learning climate downscaling**.

This repository brings the paper's data acquisition, preprocessing, statistical
baselines, U-Net/CorrDiff training, inference, evaluation, and figure scripts
together. Shared operations have common implementations; scientifically
different loss functions, samplers, and evaluation conventions remain explicit.

**Code only:** no datasets, normalization outputs, checkpoints, predictions,
figures, result tables, or manuscript files are distributed here. Small scientific tests and
real-checkpoint inference smoke tests are documented below; these do not amount
to retraining every model or regenerating every result in the paper.

## Experiment coverage

| Experiment | Conditioning input | Training/evaluation target | Implementations |
|---|---|---|---|
| Synthetic benchmark | AORC coarsened by 12×12 native pixels, interpolated to the fine grid | AORC aggregated by 3×3 pixels | Ridge, random forest, U-Net, CorrDiff |
| Transfer | Synthetic AORC or WRF 12 km input, with frozen AORC models | One common AORC 3 km reference, matching times and cells | U-Net/CorrDiff transfer workflows |
| WRF pair | WRF 12 km on the common grid | WRF 3 km on that grid | U-Net and CorrDiff |
| ERA5–CONUS404 | ERA5 surface and pressure-level predictors, regridded to the target grid | CONUS404 4 km | U-Net and CorrDiff |

AORC and WRF model crops are 288×480; ERA5–CONUS404 uses 224×352.
The ERA5 model has **35 ordered predictors**: seven surface fields and seven
pressure-level variables at 925, 850, 700, and 500 hPa. It predicts eight target
fields. Configurations and channel-name validation preserve that interface.

The source provenance identifies `corrdiff_3km_v3`, `corrdiff_wrf_3km_v4`,
`corrdiff_conus404_era5_4km_v1`, and `rfmodel_3km` as the relevant research trees.
The earlier WRF v3 directory is not substituted for the paper's v4 source.

## Layout

```text
configs/                  Experiment-specific training and inference settings
data/                     Shared decoding, aggregation, remapping, and data provenance
data/legacy/              Exact recovered historical preprocessing sources, inert text
scripts/download/         Bounded AORC, ERA5, CONUS404, and private WRF acquisition
scripts/preprocess/       Grids, masks, synthetic/dynamical pairs, normalization
models/                   Shared model helpers, losses, and validated data adapters
legacy/                   Preserved experiment-specific training/generation variants
workflows/                Portable launchers, residual scales, matched transfer
baselines/                Ridge/RF code and preserved baseline entry points
evaluation/               Common-reference metrics, FSS, CRPS, quantile diagnostics
figures/                  Paper figure generators and their provenance
patches/                  Pinned PhysicsNeMo source modifications and checksums
tests/, models/tests/     Small scientific and runtime integration tests
```

The `source_manifest.json` and `provenance.json` files record source and adapted
code hashes. They are code provenance, not copies of experiment results.

## Installation

Use a separate Python environment. The research verification environment uses
Python 3.10, PyTorch 2.8, Zarr 2, and the pinned PhysicsNeMo source below.

```bash
git clone https://github.com/hyousefi98/climate-downscaling-lab.git
cd climate-downscaling-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c requirements-tested.txt -e '.[train,baselines,plots,preprocess,download,test]'
python scripts/setup_physicsnemo.py
```

`requirements-tested.txt` records tested package versions; it is not a complete
transitive lockfile. Select a PyTorch CUDA build compatible with your machine.
CPU-only preprocessing, metrics, and tests do not need GPU hardware. The original
training/generation loops expect the PhysicsNeMo/PyTorch runtime and a suitable
CUDA device for production runs.

The setup script checks out PhysicsNeMo commit
`1b1034027f72800d085fd8f01b6945d32e693f8b`, applies the recorded local patch, and
verifies the modified-file hashes. A stock current PhysicsNeMo install is not
assumed equivalent. The patch includes six modified framework files from the
audited environment; it is not proof of the unavailable historical WRF v6
checkpoint's training lineage.

```bash
# Check an existing framework tree without modifying it.
python scripts/setup_physicsnemo.py --verify-only --destination /path/to/physicsnemo
```

Optional native curvilinear remapping requires xESMF **and ESMF**, normally
installed through conda-forge. `pip install -e '.[regrid]'` alone does not supply
ESMF. Rectilinear/projected remaps do not require that optional engine.

## External files and access

Keep scientific files outside this checkout and point the commands at them:

```bash
export DG_DATA_ROOT=/path/to/processed-data
export DG_ARTIFACT_ROOT=/path/to/normalization-and-interfaces
export DG_CHECKPOINT_ROOT=/path/to/checkpoints
export WANDB_MODE=disabled
```

| Source | Access |
|---|---|
| AORC | Anonymous NOAA object storage; bounded date/region downloads |
| ERA5 | CDS account, accepted dataset terms, and an external `.cdsapirc` |
| CONUS404 | Planetary Computer STAC/Zarr collection |
| WRF simulations | Files supplied by the simulation provider; explicit source/hash manifest |

The WRF simulations were supplied by Hsin-I Chang. This repository does not
claim a public WRF archive or recreate the original WRF simulations. Download
manifests, credentials, model parameters, and generated files stay external.
Public archives can change; record retrieval metadata and checksums rather than
assuming a fresh download is identical to an old local copy.

## Prepare new datasets

See [the complete data workflow](data/README.md) for source conventions,
all command options, masks, units, wind rotation, spatial support, time windows,
ERA5 pressure-level fields, and private WRF imports. Every acquisition command
has a credential-free `--dry-run`; remove that flag only for the requested
download. Date ranges and spatial bounds are explicit.

```bash
python scripts/download/aorc.py \
  --start 1999-12-31T19 --end 2013-01-01T01 \
  --bbox -118 30 -104.8 38.3 \
  --output /path/to/raw/aorc.zarr --dry-run

python scripts/preprocess/reference_grid.py --factor 3 \
  --output "$DG_ARTIFACT_ROOT/aorc_paper_domain.nc"

python scripts/preprocess/aorc.py --source /path/to/raw/aorc.zarr \
  --reference-grid "$DG_ARTIFACT_ROOT/aorc_paper_domain.nc" \
  --input-output "$DG_DATA_ROOT/aorc_input_physical.zarr" \
  --target-output "$DG_DATA_ROOT/aorc_target_physical.zarr" \
  --numeric-profile physical --temporal-profile ending --boundary mask

python scripts/preprocess/landmask.py \
  --reference-grid "$DG_DATA_ROOT/aorc_target_physical.zarr" \
  --natural-earth-resolution 10m --initial-coastal-buffer 0 \
  --output "$DG_DATA_ROOT/aorc_landmask_physical.nc"

python scripts/preprocess/statistics.py \
  --input "$DG_DATA_ROOT/aorc_input_physical.zarr" \
  --target "$DG_DATA_ROOT/aorc_target_physical.zarr" \
  --mask "$DG_DATA_ROOT/aorc_landmask_physical.nc" \
  --crop-shape 288 480 --coastal-buffer 5 --precip-threshold 0.1 \
  --start 2000-01-01 --end 2012-01-01 \
  --output "$DG_ARTIFACT_ROOT/aorc_physical_stats.json"
```

The statistics command rejects invalid training dates by default. An explicit
`--invalid-times drop` removes entire **common** dates, records their membership,
and binds that choice to the runtime. The same crop, land mask, coastal buffer,
spatial support, threshold, and training membership must be used by the loader.

The download example is a large, multi-year plan, not a quick-start download.
Use a small region/date range to test acquisition first. A domain must contain
enough source cells for the requested crop, coarse interpolation, and boundary
padding. The reference grid anchors the paper's domain within that padded
download; the default `native_centers` profile still retains corrected block-center
coordinates. Full training needs substantial disk space and compute.

## Corrected processing versus historical replay

The historical AORC pipeline averaged finite missing-data codes before masking
them. This produced invalid temperature and other input values. New processing
masks those codes **before** averaging or interpolation; incomplete rainfall
coverage remains missing instead of becoming a dry observation.

Other conventions are explicit:

- New data use physical specific humidity **q**; historical AORC models learned
  the **10q** numerical convention. Pressure is consistently declared in hPa at
  the model interface.
- `ending` uses six hourly end-labelled fields ending at the output time.
  `historical_forward` preserves the different original grouping explicitly.
- The default interpolation boundary is masked; historical extrapolation is
  available as an explicit choice. Spatial averaging is not described as
  conservative area remapping.
- Corrected model adapters require matching coordinates, support, normalization
  metadata, and valid values before normalization. Shape-only mask agreement is
  insufficient.
- ERA5 all-sky and historical clear-sky thermal radiation are separate profiles.

Corrected data and newly fitted normalization are intended for new training.
They must not silently replace the normalization of an existing checkpoint.
Repairing a table statistic or preprocessing file does not repair learned model
weights. The effect of the historical input defects on model performance still
needs a separate scientific assessment.

## Train U-Net and residual diffusion

Start with a resolved plan. `--dry-run` reads no data or checkpoint; add
`--check-inputs` to validate the external files and their declared interfaces.

```bash
python -m workflows.train --experiment aorc --stage regression --dry-run
python -m workflows.train --experiment wrf --stage diffusion --dry-run
python -m workflows.train --experiment era5_conus404 --stage regression --dry-run
```

For new AORC data, point the corrected profile at the files prepared above:

```bash
python -m workflows.train --experiment aorc --stage regression \
  --run-dir /path/to/runs/aorc_regression \
  --set dataset.input_data_path="$DG_DATA_ROOT/aorc_input_physical.zarr" \
  --set dataset.target_data_path="$DG_DATA_ROOT/aorc_target_physical.zarr" \
  --set dataset.landmask_path="$DG_DATA_ROOT/aorc_landmask_physical.nc"
```

Use the actual saved regression checkpoint to compute the external residual
scales and train diffusion. The historical helper called these “std”, but its
calculation is **RMS normalized residual**, without subtracting the residual
mean. The new command preserves and names that definition.

```bash
python -m workflows.residual_scales \
  --config /path/to/runs/aorc_regression/resolved.yaml \
  --regression /path/to/saved-regression.mdlus --device cuda \
  --output "$DG_ARTIFACT_ROOT/residual_std_aorc_physical.json"

python -m workflows.train --experiment aorc --stage diffusion \
  --run-dir /path/to/runs/aorc_diffusion \
  --set dataset.input_data_path="$DG_DATA_ROOT/aorc_input_physical.zarr" \
  --set dataset.target_data_path="$DG_DATA_ROOT/aorc_target_physical.zarr" \
  --set dataset.landmask_path="$DG_DATA_ROOT/aorc_landmask_physical.nc" \
  --set training.io.regression_checkpoint_path=/path/to/saved-regression.mdlus
```

Select `wrf` or `era5_conus404` with their corresponding prepared stores and
statistics for the other training setups. Check the resolved configuration:
their original paired losses differ from the AORC rain-aware loss, and the
AORC residual-scaling path is not silently imposed on the paired experiments.
Distributed runs use `torchrun --nproc_per_node=... -m workflows.train ...`.
Choose GPUs and resources according to your scheduler; no script stops other
jobs. `--resume` explicitly permits an existing training run directory.

## Generate and compare predictions

```bash
python -m workflows.generate --experiment aorc --dry-run

python -m workflows.generate --experiment aorc \
  --run-dir /path/to/runs/aorc_predictions \
  --set dataset.input_data_path="$DG_DATA_ROOT/aorc_input_physical.zarr" \
  --set dataset.target_data_path="$DG_DATA_ROOT/aorc_target_physical.zarr" \
  --set dataset.landmask_path="$DG_DATA_ROOT/aorc_landmask_physical.nc" \
  --set generation.io.reg_ckpt_filename=/path/to/saved-regression.mdlus \
  --set generation.io.res_ckpt_filename=/path/to/saved-diffusion.mdlus
```

Use `--set generation.inference_mode=regression` for the U-Net-only output.
For a one-case test, supply `generation.times`, set `generation.random_samples=null`,
and set `generation.num_ensembles=1`. Production ensemble size and sampled dates
are experiment-specific; do not infer them from a filename.

`--profile legacy` uses the historical numerical convention and external
historical artifacts. Quality checks remain active. Replaying invalid historical
handling additionally requires the deliberate `--allow-invalid-legacy-data`
flag; it is not a recommended scientific workflow.

For matched transfer with newly prepared physical stores, use
`python -m workflows.checkpoint_interface --help` to bind the frozen AORC
checkpoints, original normalization, and residual scales by hash, then
`python -m workflows.matched_transfer --help`. Both input cases use one reference,
the same requested timestamps, intersection of spatial support, four members,
the same seeds, and the same frozen artifacts. The adapter converts physical q
to the frozen 10q coordinate and converts predictions back to physical q.
It does not fit new normalization or retrain the frozen models.

### Replay the paper's matched comparison

The revised paper used archived spatially processed AORC stores, an aligned
200-date reference, and frozen checkpoints. To replay that calculation, use
`scripts/preprocess/matched_reference.py` followed by
`python -m workflows.reproduce_matched`. The preparation command preserves the
archived grid labels and floating-point operation order; the runner preserves
four members, seeds 0–3, and 18 sampler steps for both input cases.

The full commands and required external artifacts are in
[data preparation](data/README.md#reproduce-the-current-papers-paired-reference)
and [model reproduction](models/README.md#historical-profiles-and-november-18-matched-comparison).
Start with the runner's `--dry-run`, then `--validate-only` to check all 200 dates
and the common reference before launching inference. Its humidity output is
**10q**, so physical humidity metrics require division by ten. This archived
route and the new physical-store workflow have different preprocessing and are
not interchangeable.

## Ridge and random forest

```bash
python -m baselines.run --help
python -m baselines.run ridge -- --help
python -m baselines.run forest -- --help
python -m baselines.run predict-forest -- --help
```

The baseline CLI uses the pinned model/data-loader stack. New runs default to
the shared corrected physical-data loader. Copy
[`train_physical.yaml`](configs/baselines/train_physical.yaml) and
[`generate_physical.yaml`](configs/baselines/generate_physical.yaml) to your
external run directory and set their data, statistics, mask, and output paths.
Use the same training support and cutoff as the normalization calculation.

```bash
python -m baselines.run forest -- \
  --config /path/to/train_physical.yaml --model-out /path/to/models/forest.joblib
python -m baselines.run predict-forest -- \
  --config /path/to/generate_physical.yaml --model /path/to/models/forest.joblib \
  --output /path/to/predictions/forest.nc
python -m baselines.run ridge -- \
  --train-config /path/to/train_physical.yaml \
  --gen-config /path/to/generate_physical.yaml \
  --coeff-out /path/to/models/ridge.json --output /path/to/predictions/ridge.nc
```

Archived packed inputs require an explicit `aorc_legacy` configuration; that
loader rejects physical stores to prevent a second scaling operation. Source manifests
distinguish recovered estimator settings from inferred feature flags or unknown
command histories. Trained forests, ridge coefficients, and climatology arrays
are not bundled.

New RF training persists its training climatology and feature schema. Inference
uses that stored state; it must not fit climatology on the evaluation year.
Historical models without the state require an explicit compatibility decision,
and feature-count checks run before expensive prediction. The
[recovered estimator settings](baselines/experiment_settings.json) record the
2,000-tree RF, depth 35, minimum leaf size eight, square-root feature sampling,
and four-million-point training sample. The CLI defaults provide a smaller run;
set these arguments explicitly when using the recovered estimator size. The
historical feature flags and climatology sample selection are only partly
recovered, so matching hyperparameters alone does not reproduce the old forest.
The same file records the ridge settings and remaining provenance limits.

## Evaluation and figures

Score files against one explicit reference. The scorer checks timestamp order,
coordinates, common finite support, ensemble finiteness, and optionally exact
stored truth equality. It does not silently take a time intersection or apply
another precipitation threshold after averaging members.

```bash
python -m evaluation.score \
  --reference /path/to/common_reference.nc \
  --prediction /path/to/synthetic_predictions.nc /path/to/wrf_predictions.nc \
  --reference-id AORC_3km --expected-members 4 --require-stored-truth \
  --units-map /path/to/units.json --output /path/to/evaluation/scores.csv
```

The unit map is a JSON mapping of exact variable names to `factor` and `units`.
It applies identically to reference and prediction. For example, a legacy
`SPFH2maboveground` channel stored as 10q needs factor 0.1 and units `kg/kg`;
physical-q files need factor 1. Files with different numerical conventions
must be converted first. Without a map the output explicitly says “stored units”.

Shared metrics include RMSE, bias, correlation, CSI, KGE, empirical ensemble
CRPS, and FSS. Historical FSS variants differ in mask treatment and temporal
aggregation, so their names/options remain distinct. Main-map oracle members
selected by KGE and supplementary members selected by precipitation RMSE are
different diagnostics, not deployable member-selection strategies.

```bash
python -m evaluation.percentile_fss --help
python -m figures.paper 2 -- --help
python -m figures.paper S4 -- --help
python -m figures.design --output /path/to/figures/design.pdf
```

Percentile FSS fits exact precipitation quantiles on the supplied **training**
distribution, including zeros, and holds them fixed for evaluation. Figure
commands consume external arrays or tables. Figure 1 has a reconstructed
editable generator. The original supplement reused the WRF spectrum image for
S3; this repository selects the ERA5 generator for S3 explicitly and does not
claim the original duplicated image was a valid ERA5 result.

## Verification and limits

```bash
python -m pytest -m 'not integration and not gpu'
# With the pinned PhysicsNeMo/PyTorch environment:
CUDA_VISIBLE_DEVICES='' python -m pytest
python scripts/check_repository.py
```

Tests exercise missing-code decoding before averaging, complete rainfall
coverage, six-hour windows, grid/mask matching, units, all 35 ERA5 predictors,
normalization membership, frozen checkpoint interfaces, common-reference
scoring, CRPS/CSI/FSS, and training-only climatology. The CPU runtime integration
creates small encoded fields, prepares Zarr stores/statistics, runs a real U-Net
forward/backward update and diffusion forward pass, then checks scoring.
An additional CPU workflow trains and predicts with both ridge and a small RF
from freshly prepared physical stores. Multi-rank tests check shared configuration
initialization and resume behavior without starting GPU training.

Local verification also ran one full-size, one-member U-Net-plus-diffusion case
for each of AORC, WRF, and ERA5–CONUS404 on an available GPU. All eight predicted
variables were finite on valid reference cells. A bounded public NOAA download
preserved the native fill codes and scale metadata. Smoke-test outputs and
logs are external and are not repository contents. CI runs the lightweight
scientific checks without downloading the research datasets or training models.
The archived reference preparation also matches three saved dates exactly across
all eight channels; the preserved adapter passes a full 200-date CPU preflight
with identical targets for both input cases.

These checks do **not** establish complete paper reproduction. In particular:

- The paper's WRF output names a diffusion checkpoint from a deleted v6 tree.
  A current v4 checkpoint candidate loads and runs, but its identity with the
  original v6 bytes is unverified.
- Some original preprocessing invocations, the WRF creation writer, the native
  1 km descriptive-statistics generator, and several historical diagnostic
  outputs were not recovered. Candidate source copies are labelled as such.
- New quality-controlled preprocessing changes data/membership relative to the
  historical pipeline. Its implications for trained models require separate
  experiments before revised scientific claims or publication.
- Full authenticated ERA5 downloads, full native curvilinear remapping, full
  retraining, and all paper-result/figure regeneration have not been completed
  as part of repository assembly.

## License and citation

Code is provided under [Apache-2.0](LICENSE), with NVIDIA notices retained in
derived files and [NOTICE](NOTICE). Data providers' terms and checkpoint rights
are separate. Use [CITATION.cff](CITATION.cff) for the software citation and cite
the associated paper when published. Record the repository commit and external
data/checkpoint hashes for each experiment.
