import tempfile
import unittest
from pathlib import Path

from dockops.receptor import box_from_ligand, ligand_atoms


def _het(serial, resname, x, y, z):
    # PDB fixed-width HETATM record
    return (
        f"HETATM{serial:5d}  C1  {resname:>3s} A   1    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C\n"
    )


PDB = (
    "ATOM      1  N   ALA A  10       0.000   0.000   0.000  1.00 20.00           N\n"
    + _het(2, "LIG", 10.0, 20.0, 30.0)
    + _het(3, "LIG", 14.0, 20.0, 30.0)
    + _het(4, "LIG", 12.0, 24.0, 30.0)
    + _het(5, "HOH", 0.0, 0.0, 0.0)
)


class ReceptorTests(unittest.TestCase):
    def _pdb(self):
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".pdb", delete=False
        )
        tmp.write(PDB)
        tmp.close()
        return tmp.name

    def test_ligand_atoms_filters_resname(self):
        coords = ligand_atoms(self._pdb(), "LIG")
        self.assertEqual(len(coords), 3)
        self.assertIn((10.0, 20.0, 30.0), coords)

    def test_missing_resname_raises(self):
        with self.assertRaises(ValueError):
            ligand_atoms(self._pdb(), "XXX")

    def test_box_center_and_size(self):
        center, size = box_from_ligand(self._pdb(), "LIG", padding=5.0)
        self.assertEqual(center, (12.0, 22.0, 30.0))
        # extent 4 on x and y, 0 on z, + 2*padding each side
        self.assertEqual(size, (14.0, 14.0, 10.0))

    def test_multi_site_resname_rejected(self):
        # same resname at two different (chain, resseq) sites -> ambiguous box
        pdb = (
            _het(1, "LIG", 10.0, 20.0, 30.0)
            + _het(2, "LIG", 11.0, 20.0, 30.0)
            + f"HETATM{3:5d}  C1  {'LIG':>3s} A   9    "
              f"{50.0:8.3f}{50.0:8.3f}{50.0:8.3f}  1.00 20.00           C\n"
        )
        tmp = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
        tmp.write(pdb)
        tmp.close()
        with self.assertRaises(ValueError):
            ligand_atoms(tmp.name, "LIG")


if __name__ == "__main__":
    unittest.main()
