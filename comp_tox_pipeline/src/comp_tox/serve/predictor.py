"""Runtime predictor: SMILES in, calibrated probability + conformal set +
applicability-domain flag out.

Reuses the pipeline's own canonicalization, fingerprint spec, conformal
set rule, and AD distance so served predictions are exactly what the
committed evaluation measured.
"""

from __future__ import annotations

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse

from comp_tox.data.standardize import canonicalize
from comp_tox.eval.applicability import in_domain, nn_tanimoto_distances
from comp_tox.eval.conformal import prediction_sets
from comp_tox.features.build import _parse_fingerprint
from comp_tox.serve.reference import load_reference


class Predictor:
    def __init__(self, model_path: str, reference_path: str):
        bundle = joblib.load(model_path)
        self.ref = load_reference(reference_path)
        self.primary = bundle["primary"]
        self.model = bundle["models"][self.primary]
        self.provenance = {
            k: bundle[k]
            for k in ("fingerprint", "features_sha256", "splits_sha256",
                      "train_rows", "valid_rows", "train_actives")
            if k in bundle
        }
        radius, fp_size = _parse_fingerprint(
            bundle.get("fingerprint") or "morgan-r2-2048"
        )
        self._fp_size = fp_size
        self._gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=fp_size
        )

    def fingerprints(
        self, smiles_list: list[str]
    ) -> tuple[sparse.csr_matrix, list, list]:
        """Returns (X, records) where records[i] carries canonical SMILES or an
        error marker; rows of X align with parseable entries only."""
        rows, records, keep = [], [], []
        for i, smi in enumerate(smiles_list):
            canon, _ = canonicalize(smi)
            mol = Chem.MolFromSmiles(canon) if canon else None
            rec = {"smiles": smi, "canonical_smiles": canon}
            if mol is None:
                rec["error"] = "unparseable_smiles"
            else:
                rows.append(self._gen.GetFingerprintAsNumPy(mol))
                keep.append(i)
            records.append(rec)
        X = (
            sparse.csr_matrix(np.asarray(rows, dtype=np.uint8))
            if rows
            else sparse.csr_matrix((0, self._fp_size), dtype=np.uint8)
        )
        return X, records, keep

    def predict(self, smiles_list: list[str]) -> list[dict]:
        X, records, keep = self.fingerprints(smiles_list)
        if not keep:
            return records
        probs = np.asarray(self.model.predict_proba(X))[:, 1]
        sets = prediction_sets(
            np.column_stack([1.0 - probs, probs]), self.ref["qhat"]
        )
        nn = nn_tanimoto_distances(X, self.ref["train_X"])
        dom = in_domain(nn, self.ref["ad_threshold"])
        for j, i in enumerate(keep):
            records[i].update(
                prob_active=float(probs[j]),
                prediction_set=[int(c) for c in np.flatnonzero(sets[j])],
                nn_tanimoto_distance=float(nn[j]),
                in_domain=bool(dom[j]),
            )
        return records
