# Endpoint selection

**Decision: Tox21 NR-ER** (nuclear-receptor estrogen agonism).

Rationale: endocrine disruption is a flagship NAMs use case: in-vitro and
in-silico estrogenicity screening is actively argued to displace animal
uterotrophic assays. Tox21 labels are fully public (MoleculeNet mirror),
the ~9% active rate exercises imbalance-aware evaluation, and the dataset
(~6k labeled compounds) is small enough to iterate locally.

Original criteria, for the record:

- **Decision relevance**: endpoint maps to a real preclinical safety decision
  (DILI > hERG > ER agonism roughly, for pharma relevance)
- **Label budget**: enough actives for a scaffold split to leave a usable test set
- **Label quality**: prefer endpoints with consistent assay protocols
- **Narrative**: the NAMs framing is strongest when the endpoint is one where
  in-vitro/in-silico methods are actively argued to displace animal studies

## Candidates

| Endpoint | Source | Positives (approx) | Notes |
|---|---|---|---|
| DILI (hepatotox) | literature sets / ChEMBL | TODO | highest relevance, messiest labels |
| hERG inhibition | ChEMBL / ToxCast | TODO | cardiotox, well-studied |
| Tox21 ER/AR agonism | Tox21 | TODO | clean NAMs story, large screen |
| ToxCast assay AC50 | invitrodb | TODO | most labels, assay-specific |

Decision: **Tox21 NR-ER** (implemented; revisit if expanding to a second
endpoint. hERG or a ToxCast AC50 are the natural next candidates)
