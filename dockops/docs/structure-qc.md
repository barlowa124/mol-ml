# Structure readiness QC

`python -m dockops.structure_qc` (or `snakemake results/structure_qc.json`,
needs `data/raw/receptor.pdb` from `dockops.fetch`) produces one artifact
answering "is this receptor model defensible to dock into?":

- **AFDB confidence** (`alphafold.plddt_stats`): pLDDT read from the
  model's B-factor column: mean, fraction ≥70 (confident), ≥90 (very high).
- **Agreement with experiment** (`alphafold.ca_rmsd`): Cα superposition
  after constant-offset numbering correction (detects UniProt-vs-crystal
  shifts like EGFR's 24-residue signal peptide. The offset is reported).
  Robust metrics: median Cα deviation + fraction within 2 Å.
- **Mechanics sanity** (`md.minimize`): PDBFixer prepares the structure
  (missing residues/atoms, heterogen removal, protonation at pH 7), then
  amber14 + GBn2 implicit-solvent minimization reports the energy drop.

## Observed result (EGFR, `results/structure_qc.json`)

- AFDB P00533 v6: mean pLDDT **76**, 71% of residues ≥70, 47% ≥90.
- AF model vs 1M17: numbering offset **-24** detected and corrected. All
  312 paired residues identity-matched, but **median Cα deviation 4.6 Å,
  only 9% within 2 Å**. The predicted kinase domain differs in lobe
  orientation from the erlotinib-bound experimental conformation.
- **Conclusion this QC supports:** for EGFR docking, use the experimental
  structure (1M17), not this AF model. Confident pLDDT does not guarantee
  the holo conformation. That is precisely the decision this stage exists
  to automate.
- Minimization: +240.8M → -45.2k kJ/mol (PDBFixer-repaired clashes resolved
  in implicit solvent), RMSD 0.086 nm, so mechanics behave as expected.

## Scope

- The MD stage is a mechanics check (picoseconds, implicit solvent), not a
  conformational study.
- `variants.py` scores mutations with ESM-2 masked marginals, a zero-shot
  variant prioritization, not de-novo protein design.
- AFDB comparison requires same-protein numbering agreement. The offset
  scan handles constant shifts only, not arbitrary renumbering.
