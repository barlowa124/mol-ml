import pytest

from dockops.ligands import embed_smiles, to_pdbqt


def test_embed_valid_smiles():
    mol = embed_smiles("CC(=O)Oc1ccccc1C(=O)O")
    assert mol is not None
    assert mol.GetNumConformers() == 1


def test_embed_invalid_smiles_returns_none():
    assert embed_smiles("not_a_smiles") is None


def test_pdbqt_conversion():
    try:
        import meeko  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="meeko"):
            to_pdbqt("CCO")
        return
    pdbqt = to_pdbqt("CC(=O)Oc1ccccc1C(=O)O")
    assert pdbqt is not None
    assert "ATOM" in pdbqt
    assert "ROOT" in pdbqt


def test_pdbqt_invalid_smiles():
    pytest.importorskip("meeko")
    assert to_pdbqt("not_a_smiles") is None
