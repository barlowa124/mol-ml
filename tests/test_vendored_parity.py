"""Cross-repo vendored parity: esm_cache.py shares a digest with
protein-ml.

The file is vendored (not depended on) in three places —
mol-ml/dti_fusion, protein-ml/active_learning_loop,
protein-ml/protein_design_ops — and protein-ml's test pins the same
digest. An edit on either side trips a pin and forces deliberate sync.
"""
import hashlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ESM_CACHE = (ROOT / "dti_fusion" / "src" / "dti_fusion" / "esm_cache.py")

# Shared with protein-ml/tests/test_vendored_parity.py — keep in sync.
ESM_CACHE_SHA256 = \
    "2c80f1d43fffe47c126ce70e0f7342ced3458a3d902105c6275cacc338295531"

# Canonical conformal helpers, vendored here as eval/conformal_shared.py;
# protein-ml/protein_stability_uncertainty pins the same digest.
CONFORMAL = (ROOT / "comp_tox_pipeline" / "src" / "comp_tox" / "eval" /
             "conformal_shared.py")
CONFORMAL_SHA256 = \
    "fcba54721a3f106e3864ab43ad0cb61caf273c41426d7bfa649b542f5f72af00"

# Claims verifier, vendored from statgen/scrna_qc (bio-qc) and
# llm-posttraining/evals — all four copies pin this digest.
CLAIMS = (ROOT / "comp_tox_pipeline" / "src" / "comp_tox" / "claims.py")
CLAIMS_SHA256 = \
    "475eb4a6a338e363374c0810bf8f3861aec16cd41ae0c4ec2e0fbb54a6cc74a2"


class VendoredParityTests(unittest.TestCase):
    def test_esm_cache_matches_shared_digest(self):
        self.assertEqual(
            hashlib.sha256(ESM_CACHE.read_bytes()).hexdigest(),
            ESM_CACHE_SHA256,
            "dti_fusion esm_cache.py drifted from the vendored copies in "
            "protein-ml — sync all three and update the pin together")

    def test_conformal_shared_matches_shared_digest(self):
        self.assertEqual(
            hashlib.sha256(CONFORMAL.read_bytes()).hexdigest(),
            CONFORMAL_SHA256,
            "comp_tox conformal_shared.py drifted from the vendored copy "
            "in protein-ml/protstab — sync both repos and update the pin "
            "together")

    def test_claims_matches_shared_digest(self):
        self.assertEqual(
            hashlib.sha256(CLAIMS.read_bytes()).hexdigest(),
            CLAIMS_SHA256,
            "comp_tox claims.py drifted from the vendored copies in "
            "bio-qc and llm-posttraining — sync all four and update the "
            "pin together")


if __name__ == "__main__":
    unittest.main()
