# Data sources

All sources are public. Record exact file names/versions and download dates
in the run metadata. Releases are versioned and endpoints change between
invitrodb versions.

## EPA ToxCast / Tox21 (invitrodb)

- Bulk data: EPA Computational Toxicology downloadable data page → invitrodb
  release (CSV/MySQL dump). Tox21 endpoints are a subset of invitrodb assays.
- Key fields: `aenm` (assay endpoint name), `hitc` (hit call), `ac50`,
  `dsstox_substance_id`, plus cytotox flags, since hit calls near the cytotoxicity
  burst need care (see invitrodb docs).
- Chemical IDs: resolve `dsstox_substance_id` → SMILES via the DSSTox
  substance file in the same release.

## PubChem

- PUG-REST base: `https://pubchem.ncbi.nlm.nih.gov/rest/pug`
- Tox21 summary assays have AIDs. Per-compound activity calls available via
  `assay/aid/<AID>/JSON` or bulk CSV.
- Rate-limit politely. Cache raw responses under `data/raw/`.

## ChEMBL

- SQLite dump or web API (`/data/activity.json?...`).
- Useful for hERG/DILI-adjacent bioactivity. Watch unit heterogeneity
  (standard_value/standard_units) when aggregating labels.

## Label reconciliation

- TODO: document the conflicting-call policy chosen in `data/standardize.py`
  (any-active vs majority vs drop-conflicting) and the counts it produced.
