import numpy as np
import pytest
from unittest.mock import patch

from dti_fusion.features import (
    drug_fingerprints,
    morgan_fingerprint,
    protein_embeddings,
)


class TestMorganFingerprint:
    def test_valid_smiles(self):
        fp = morgan_fingerprint("CCO", radius=2, nbits=256)
        assert fp is not None

    def test_invalid_smiles_returns_none_not_exception(self):
        assert morgan_fingerprint("not_a_smiles", 2, 256) is None

    @pytest.mark.parametrize("bad", [None, "", 0, 3.14])
    def test_nonstring_input_returns_none(self, bad):
        # MolFromSmiles raises a C++ TypeError on non-str input; the
        # wrapper must convert that to a clean None instead
        assert morgan_fingerprint(bad, 2, 256) is None


class TestDrugFingerprints:
    def test_spec_parsing_and_shape(self):
        out = drug_fingerprints(["CCO", "c1ccccc1"], "morgan-r2-128")
        assert out.shape == (2, 128)
        assert out.dtype == np.float32
        assert out.sum() > 0

    def test_bad_spec_rejected(self):
        with pytest.raises(ValueError, match="unsupported fingerprint"):
            drug_fingerprints(["CCO"], "ecfp4")
        with pytest.raises(ValueError, match="unsupported fingerprint"):
            drug_fingerprints(["CCO"], "morgan-rx-128")

    def test_unparseable_smiles_becomes_zero_row(self):
        out = drug_fingerprints(["CCO", "bogus_smiles"], "morgan-r2-64")
        assert out.shape == (2, 64)
        assert out[0].sum() > 0
        assert np.all(out[1] == 0)


class TestProteinEmbeddings:
    def test_cache_hit_skips_model_load(self):
        # all sequences already in the shared store -> no transformers
        # import, no HF download
        seqs = ["ACDEFG", "HIKLMN"]
        cached = {s: np.full(4, i + 1.0) for i, s in enumerate(seqs)}
        with patch(
            "dti_fusion.esm_cache.get_many",
            return_value=([cached[s] for s in seqs], []),
        ), patch("dti_fusion.esm_cache.put") as put:
            out = protein_embeddings(seqs, "unused-model", max_len=100)
        assert out.shape == (2, 4)
        np.testing.assert_array_equal(out[0], cached[seqs[0]])
        put.assert_not_called()

    def test_truncation_applied_before_cache_key(self):
        long_seq = "A" * 50
        seen_keys = []

        def fake_get_many(kind, model, keys):
            seen_keys.extend(keys)
            return ([np.zeros(4)] * len(keys), [])

        with patch("dti_fusion.esm_cache.get_many",
                   side_effect=fake_get_many):
            protein_embeddings([long_seq], "m", max_len=10)
        assert seen_keys == ["A" * 10]
