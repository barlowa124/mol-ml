import unittest

from dockops.md import minimize, short_nvt

PDB = "tests/fixtures/1L2Y.pdb"


class MDTests(unittest.TestCase):
    def test_minimize_lowers_energy(self):
        r = minimize(PDB)
        self.assertEqual(r["n_atoms"], 304)
        self.assertLess(r["pe_after_kj_mol"], r["pe_before_kj_mol"])
        self.assertLess(r["delta_pe_kj_mol"], 0)

    def test_short_nvt_stable(self):
        r = short_nvt(PDB, steps=1000, report_every=500)
        self.assertEqual(len(r["energies_kj_mol"]), 2)
        # Implicit-solvent trp-cage should not explode or collapse in 1ps
        self.assertLess(abs(r["energy_drift_kj_mol"]), 5000)
        self.assertGreater(r["rmsd_to_start_nm"], 0)


if __name__ == "__main__":
    unittest.main()
