# Downloading and preparing the paper's data families

This directory contains code and source provenance only. Run commands from the
repository root after `pip install -e '.[download,preprocess]'`. New preprocessing
requires fresh output paths; it never removes or overwrites an existing store.
The commands are designed for new corrected datasets. Exact archived checkpoint
replay is a separate workflow because the historical humidity coordinate,
normalization, time windows, missing-value treatment, and grid labels differ.

## Data access

- **AORC:** [NOAA's public registry](https://registry.opendata.aws/noaa-nws-aorc/)
  identifies the anonymous `noaa-nws-aorc-v1-1-1km` bucket. Download only the
  requested region and dates. The native packed values and fill metadata remain
  intact until preprocessing. NOAA's registry currently notes a 2026 correction
  of incorrectly masked rows; the command records the retrieval time and source
  consolidated-metadata hash. A current download is not assumed identical to a
  years-old local copy.
- **ERA5:** [CDS API setup](https://cds.climate.copernicus.eu/en/how-to-api)
  requires an external CDS account, token in `.cdsapirc`, and acceptance of both
  ERA5 dataset terms. Credentials never belong in this repository. Downloads use
  `reanalysis-era5-single-levels` and `reanalysis-era5-pressure-levels`.
- **CONUS404:** the original acquisition used the
  [Planetary Computer collection](https://planetarycomputer.microsoft.com/dataset/conus404).
  The command resolves its signed `zarr-abfs` asset through STAC. It needs the
  optional `planetary-computer`, `pystac-client`, and `adlfs` dependencies and
  network access. Signed URLs or storage tokens are not written to manifests.
- **WRF:** the simulations were provided privately by Hsin-I Chang; a public
  source and original processing writer remain unverified. Supply accessible
  files and their SHA256 hashes through the manifest importer. The repository
  does not generate or distribute those simulations.

All four download commands provide `--help` and `--dry-run`. ERA5 dry-run does
not instantiate an API client. The bounded endpoint smoke tested NOAA variable
metadata and the CONUS404 collection; full authenticated ERA5 retrieval and
large remote data downloads are not part of the CPU test suite.

```bash
python scripts/download/aorc.py --start 2011-12-31T19 --end 2013-01-01T01 \
  --bbox -118 30 -104.8 38.3 --output /data/raw/aorc_2012_subset.zarr --dry-run
python scripts/download/era5.py --years 2012 --bbox -118 28 -105 40.5 \
  --output-dir /data/raw/era5 --dry-run
python scripts/download/conus404.py --start 2011-12-31T19 --end 2013-01-01T01 \
  --bbox -118 28 -105 40.5 --output /data/raw/conus404_subset.zarr --dry-run
python scripts/download/wrf.py --manifest /data/wrf_files.json \
  --output-dir /data/raw/wrf --dry-run
```

WRF manifest example: a JSON list with `source` (local file or HTTPS URL), `path`
(relative output path), and `sha256` (64 hexadecimal characters) for every file.
Represent a Zarr directory as its individual metadata and chunk files. The
importer rejects parent traversal, duplicate paths, overwrites, and wrong hashes.

## AORC: native packing to synthetic pairs

```bash
python scripts/preprocess/aorc.py --source /data/raw/aorc_2012_subset.zarr \
  --input-output /data/processed/aorc_12to3_2012.zarr \
  --target-output /data/processed/aorc_3_2012.zarr \
  --fine-factor 3 --coarse-factor 12 --numeric-profile physical \
  --temporal-profile ending --boundary mask
```

The default masks each variable's native fill code **before** any spatial mean
or interpolation. Each 3×3 or 12×12 block must contain all its source pixels;
partial coverage remains missing. Spatial means are equal-weight pixel means,
not area-weighted conservative remapping. Precipitation is spatially averaged,
then summed in time. The coarse field is bilinearly interpolated from its cell
centers to the fine centers. Its outside-center rim is missing by default and
recorded in `spatial_support`; use the identical support in both experiment arms.
Explicit `nearest` and historical `extrapolate` edge policies are also available.
No edge policy restores invalid source coverage.

Native block-center latitude/longitude is the default. A supplied historical
reference grid determines crop anchoring and dimensions. Retaining its old
labels requires `--grid-profile historical_labels` explicitly. Those labels
were made with an approximate 0.008333° mesh and need not equal true block
centers. `scripts/preprocess/reference_grid.py` reconstructs those historical
labels without bundling a binary grid. The 4/28 factor pair is also supported.

`physical` stores temperature K, wind m/s, pressure hPa, radiation W/m²,
precipitation mm, and specific humidity q kg/kg. Native AORC humidity has a
1e-4 scale; the old loader used 1e-3 and learned **10q**. `legacy_model` explicitly
retains 10q; `legacy_packed` retains raw numerical codes. These profiles are not
interchangeable and do not authorize using archived normalization for new units.

## Temporal convention and complete coverage

For `ending`, an output at `t` uses hourly labels `t−5,...,t`. Hourly AORC
precipitation ends at its label, so the sum represents `(t−6h,t]`. Other fields
use the mean of those six samples. Include the previous year's hours when a
January 1 00 UTC output is needed. Process consecutive source stores together
to avoid dropping windows at artificial file boundaries.

`historical_forward` instead uses labels `t,...,t+5` and represents precipitation
over `(t−1h,t+5h]`. It is explicit compatibility metadata, not a silent relabel.
Duplicate or unordered labels are rejected. Missing source hours or cells make
the corresponding aggregate missing; complete zero-rainfall windows remain zero.
Only incomplete calendar endpoints are dropped by default. The per-cell
`hourly_valid_count` records coverage. Already six-hourly WRF inputs are never
aggregated a second time.

## ERA5 and CONUS404 / WRF pairs

ERA5 preprocessing requires all seven surface and 28 pressure-level channels;
it fails if a field is missing. The download defaults to **all-sky** downward
thermal radiation. The original script requested clear-sky thermal radiation;
replaying that choice requires `--radiation-profile historical_clear_sky` in
both download and preprocessing. Neither choice is silently substituted.

```bash
python scripts/preprocess/era5.py \
  --surface-files /data/raw/era5/era5_surface_*.nc \
  --pressure-files /data/raw/era5/era5_pressure_*.nc \
  --reference-grid /data/grid4km.nc --output /data/processed/era5.zarr
```

The script expects CDS hourly interval precipitation in meters and radiation
in J/m² over the preceding hour; it converts to mm and W/m² before aggregation.
It converts surface pressure Pa to hPa. Pressure-level specific humidity remains
q, geopotential remains m²/s², and relative humidity remains percent. Input with
additional forecast/member dimensions must be selected explicitly beforehand.

`scripts/preprocess/paired_fields.py` handles CONUS404 and WRF. You must declare
source humidity (`mixing_ratio` or `specific_humidity`), pressure (`Pa` or `hPa`),
wind frame (`grid` or `earth`), precipitation (`interval` or `cumulative`), and
cadence (`hourly_end` or `six_hourly_end`). The output is physical specific q:
mixing ratio r is converted with q=r/(1+r) **before** averaging/interpolation.
Grid winds require supplied `COSALPHA` and `SINALPHA`; absent rotation provenance
is not grounds for assuming earth-relative winds. Raw provider fields may need
their static coordinate/angle dataset merged before this command.

Available remaps are exact `same_grid`, lat/lon `rectilinear`, explicitly
`projected`, and optional `esmf_bilinear` (xESMF plus ESMF installed, e.g. through
conda-forge). None is called conservative. The projected path checks that source
coordinates actually are rectilinear in the supplied projection. The recovered
CONUS404 script used
`+proj=lcc +lat_1=33 +lat_2=45 +lat_0=40 +lon_0=-97 +a=6370000 +b=6370000 +units=m`.
It should not be copied into an unrelated WRF simulation without verification.
ESMF weights must be written to a new explicitly supplied path.

The internal mapping keys are `GLW,SWDOWN,PSFC,Q2,T2,U10,V10,PREC_ACC_NC`.
WRF output uses `RAIN_TOTAL` and assumes that source name by default;
CONUS404 output uses `PREC_ACC_NC`. An optional JSON variable map maps the
internal keys to source names, for example `PREC_ACC_NC` to `RAIN_TOTAL`.
Cumulative precipitation requires contiguous
samples; its first increment is missing. Negative jumps raise by default; use
`--reset restart` only when documented restart accumulation warrants it.

## Normalization and the runtime interface

Build the land mask on the exact new grid; matching array shape alone is not
enough. The recovered mask builder used Natural Earth 10m polygons and one
initial coastal pixel, followed by additional loader buffering. New masks
default to zero initial buffer so that this choice is explicit:

```bash
python scripts/preprocess/landmask.py \
  --reference-grid /data/processed/aorc_target_train.zarr \
  --natural-earth-resolution 10m --initial-coastal-buffer 0 \
  --output /data/landmask.nc
```

This command explicitly permits Cartopy to download/cache the selected Natural
Earth polygons. Alternatively supply `--shapefile /data/ne_10m_land.shp` with
its WGS84 `.prj` and companion files. Polygon file hashes, initial buffering,
and exact grid coordinates are recorded. Changing a polygon release can change
boundary cells; use the same mask file for both arms. For old checkpoint replay,
use the original archived mask rather than assuming a new release is identical.

Each output is a Zarr group with `aorc`, `wrf`, `era5`, or `conus404` shaped
`(time,channel,y,x)`, the corresponding `*_variable(channel)` byte labels,
`lat(y,x)`, `lon(y,x)`, decoded CF time, and `spatial_support(y,x)`. Attributes
record numerical profile, humidity convention, pressure units, wind frame,
time-label/window convention, and regridding policy. The corrected runtime
adapter requires `numeric_profile=physical`. Spatial support is a mask variable,
not a weather channel.

```bash
python scripts/preprocess/statistics.py \
  --input /data/processed/aorc_input_train.zarr \
  --target /data/processed/aorc_target_train.zarr \
  --mask /data/landmask.nc --crop-shape 288 480 --coastal-buffer 5 \
  --start 2000-01-01 --end 2012-01-01 --precip-threshold 0.1 \
  --output /data/processed/aorc_statistics.json
```

The required crop is applied before the coastal buffer, then intersected with
both stores' geometric support. Means and population standard deviations use
identical training dates and mask cells. Values below the stated precipitation
threshold become zero before moments, as in the loader. Held-out years are
excluded by the explicit `[start,end)` interval. Missing support values raise;
`--invalid-times drop` explicitly excludes entire common dates, records them,
and must be respected by training. Statistics include mask shape/SHA256,
threshold, time membership, and unit/profile metadata. These are normalization
statistics for the selected training set, not an automatic recreation of a
manuscript's descriptive table.

Run `pytest tests/test_preprocessing_science.py` for the small scientific
fixtures, including packed→coarsened→six-hour→Zarr→common-statistics integration.

## Reproduce the current paper's paired reference

The revised paper's matched comparison used the original spatially processed
hourly AORC stores, the original 200 timestamps and grid, the original 128,619
verification cells, and the frozen checkpoint normalization. Its preparation
is available separately:

```bash
python scripts/preprocess/matched_reference.py \
  --synthetic-hourly /data/archive/aorc_12km_2_3km_2012.zarr \
  --target-hourly /data/archive/aorc_3km_2012.zarr \
  --baseline-netcdf /data/archive/all_aorc_12km_to_3km.nc \
  --statistics /data/archive/aorc_stats.json \
  --output /data/reproduce/aligned_inputs.nc \
  --reference-output /data/reproduce/reference_model_units.nc \
  --times-output /data/reproduce/requested_times.json \
  --audit-output /data/reproduce/preparation_audit.json
```

This command preserves the verified chunk traversal and float64 summation,
stores float32 raw values, then creates the reference with the original
float32 scale/filter/normalize/denormalize/filter sequence. Reference pressure
is hPa; reference humidity is the checkpoint's **10q** coordinate. It verifies
complete hourly coverage and exact coordinates, rejects invalid hourly values
on the fixed mask, and records finite negative undershoots without changing
their original postaggregation clipping. Original source metadata and consumed
chunks are hashed; three cases are independently reconstructed through Zarr.
The port's writer values were compared against the saved independently produced
reference at three dates with exact equality across all channels and cells.

The source stores' original spatial preprocessing is retained here for
reproduction; this path does not repair historical interpolation/fill defects
or create a new training set. It is CPU-only and needs several GB of RAM for
the original chunk layout. The aligned NetCDF, timestamp JSON, frozen model
artifacts, and supplied WRF input are external inputs to
`python -m workflows.reproduce_matched`. Use the separate
`reference_model_units.nc` with `python -m evaluation.score`. No data or model
artifacts are bundled in the repository. See `tests/test_preprocessing_matched_reference.py` for the
bounded chunk/window fixture and optional actual legacy-loader comparison.
