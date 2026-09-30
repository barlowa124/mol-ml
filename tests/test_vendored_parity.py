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


class VendoredParityTests(unittest.TestCase):
    def test_esm_cache_matches_shared_digest(self):
        self.assertEqual(
            hashlib.sha256(ESM_CACHE.read_bytes()).hexdigest(),
            ESM_CACHE_SHA256,
            "dti_fusion esm_cache.py drifted from the vendored copies in "
            "protein-ml — sync all three and update the pin together")


if __name__ == "__main__":
    unittest.main()
