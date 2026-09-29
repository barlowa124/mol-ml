import pytest

from comp_tox.features.build import _parse_fingerprint


def test_parse_fingerprint_default_spec():
    assert _parse_fingerprint("morgan-r2-2048") == (2, 2048)


def test_parse_fingerprint_other_spec():
    assert _parse_fingerprint("morgan-r3-1024") == (3, 1024)


def test_parse_fingerprint_rejects_unsupported():
    with pytest.raises(ValueError, match="unsupported fingerprint spec"):
        _parse_fingerprint("avalon-1024")
    with pytest.raises(ValueError):
        _parse_fingerprint("morgan")


def test_build_features_respects_fp_size(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("rdkit")
    from scipy import sparse

    from comp_tox.features.build import build_features

    df = pd.DataFrame(
        {
            "compound_id": ["c1", "c2"],
            "canonical_smiles": ["CCO", "c1ccccc1"],
            "scaffold_id": ["s1", "s2"],
            "label": [0, 1],
        }
    )
    in_path = tmp_path / "compounds.parquet"
    df.to_parquet(in_path, index=False)
    npz = tmp_path / "features.npz"
    meta = tmp_path / "meta.parquet"

    build_features(str(in_path), str(npz), str(meta), fingerprint="morgan-r3-1024")
    X = sparse.load_npz(npz)
    assert X.shape == (2, 1024)
