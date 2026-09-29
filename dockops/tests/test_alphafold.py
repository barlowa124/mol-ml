import tempfile
import unittest

import numpy as np

from dockops.alphafold import ca_rmsd, kabsch_rmsd, plddt_stats


NAMES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]


def _atom(serial, resseq, x, y, z, b, resname="ALA"):
    return (
        f"ATOM  {serial:5d}  CA  {resname:>3s} A{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           C\n"
    )


def _pdb(atoms):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
    tmp.writelines(atoms)
    tmp.close()
    return tmp.name


PDB_A = [_atom(i + 1, r, float(r), 0.0, 0.0, 80.0) for i, r in enumerate(range(1, 21))]
PDB_B = [_atom(i + 1, r, float(r), 0.0, 0.0, 55.0) for i, r in enumerate(range(1, 21))]


class AlphaFoldTests(unittest.TestCase):
    def test_plddt_stats(self):
        stats = plddt_stats(_pdb(PDB_A + [_atom(21, 21, 0, 0, 0, 40.0)]))
        self.assertEqual(stats["n_residues"], 21)
        # 20 residues at 80 (confident), 1 at 40
        self.assertAlmostEqual(stats["frac_confident_70"], 20 / 21, places=3)

    def test_kabsch_identical(self):
        pts = np.random.default_rng(0).normal(size=(20, 3))
        self.assertAlmostEqual(kabsch_rmsd(pts, pts), 0.0, places=6)

    def test_kabsch_rotation_invariant(self):
        rng = np.random.default_rng(1)
        pts = rng.normal(size=(30, 3))
        # 90-degree rotation about z + translation
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        moved = pts @ rot.T + np.array([5.0, -3.0, 2.0])
        self.assertAlmostEqual(kabsch_rmsd(pts, moved), 0.0, places=5)

    def test_ca_rmsd_identical_structure(self):
        r = ca_rmsd(_pdb(PDB_A), _pdb(PDB_A))
        self.assertEqual(r["n_paired_residues"], 20)
        self.assertAlmostEqual(r["ca_rmsd"], 0.0, places=3)

    def test_ca_rmsd_numbering_offset(self):
        # Same protein numbered differently: A resseqs 1-20, B resseqs 5-24
        # (i.e. a +4 offset). Distinct residue names make the offset detectable.
        a = [
            _atom(i + 1, r, float(r), 0.0, 0.0, 80.0, NAMES[r - 1])
            for i, r in enumerate(range(1, 21))
        ]
        b = [
            _atom(i + 1, r, float(r - 4), 0.0, 0.0, 80.0, NAMES[r - 5])
            for i, r in enumerate(range(5, 25))
        ]
        r = ca_rmsd(_pdb(a), _pdb(b))
        self.assertEqual(r["numbering_offset"], 4)
        self.assertEqual(r["n_paired_residues"], 20)
        self.assertAlmostEqual(r["ca_rmsd"], 0.0, places=3)

    def test_plddt_stats_rejects_no_ca(self):
        with self.assertRaises(ValueError):
            plddt_stats(_pdb([
                "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N\n"
            ]))

    def test_plddt_stats_filters_nan_bfactor(self):
        # truncated record: coords present, B-factor column empty
        short = (
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n"
        )
        good = _atom(2, 2, 1.0, 0.0, 0.0, 80.0)
        stats = plddt_stats(_pdb([short, good]))
        self.assertEqual(stats["n_residues"], 2)
        self.assertEqual(stats["mean_plddt"], 80.0)


if __name__ == "__main__":
    unittest.main()
