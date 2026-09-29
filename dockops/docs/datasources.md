# Benchmark data sources

`dockops.fetch` implements the DUD-E + RCSB path below for the `egfr`
target (see `fetch:` in config). This doc lists the broader options.

Public actives-vs-decoys benchmarks for evaluating a docking backend once
`VinaEngine` lands. Record the dataset name, version and download date in
the run provenance. Do not report enrichment on the bundled fixture.

## DUD-E

- `dude.docking.org`, ~102 targets, each with actives + property-matched
  decoys. The classic virtual-screening benchmark. Known decoy biases are
  documented in the literature, so mention them if DUD-E numbers are reported.
- Per-target bundles include receptor + co-crystallized ligand for box
  definition.

## LIT-PCBA

- Higher-quality, less-biased successor benchmark (15 targets, actives from
  real screens + matched inactives). Preferred over DUD-E as the less-biased benchmark.
- Distributed via the authors' GitHub / the LIT-PCBA site.

## Receptor structures

- RCSB PDB for apo/holo structures. Prefer holo structures so the binding
  box can be defined from the co-crystallized ligand centroid.

## Ligand libraries (optional, for scale demos)

- ZINC (zinc.docking.org) purchasable subsets. Enamine REAL is huge, so start
  with a small ZINC lead-like tranche if a throughput demo is wanted.
