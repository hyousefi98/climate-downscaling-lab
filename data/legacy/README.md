# Recovered historical preprocessing sources

These are exact, inert text copies of recovered local research scripts, with
original comments and paths retained. Their `.txt` suffix is intentional: they
are evidence, not the runnable default pipeline. SHA256 hashes are recorded in
`../provenance.json`. Original absolute source paths are retained only in the
private audit outside this repository.

Several scripts delete an existing output directory, skip unavailable variables,
or rely on machine-specific paths. In particular, the historical AORC pipeline
averaged packed missing codes before filtering, used `nansum` without complete
rainfall coverage, and grouped hourly labels `t,...,t+5` under label `t`.
Those behaviors are preserved here solely to explain archived results. Do not
execute these copies against an original dataset.

Use `scripts/download/` and `scripts/preprocess/` for explicit paths, fresh output
directories, missing-aware operations, and documented profiles. `legacy_model`
and `legacy_packed` in the new AORC command select numerical conventions only;
they do **not** reproduce the old missing-value defects.

The exact original WRF download and regridding writer was not recovered. The
simulations were supplied by Hsin-I Chang. The new WRF manifest importer and
explicit regridding command are replacements, not a claim that the original
model configuration or upstream archive has been reconstructed. Similarly, a
recovered script is not proof of its exact historical invocation. Multiple
candidate combine scripts are retained and labelled accordingly in provenance.
