import unittest

from dockops.fetch import parse_ism


ISM = """\
CC(=O)Oc1ccccc1 CHEMBL25
CN1C=NC2=C1C(=O)N(C(=O)N2)C CHEMBL113

c1ccccc1 CHEMBL1000
"""


class FetchTests(unittest.TestCase):
    def test_parse_ism_rows(self):
        rows = parse_ism(ISM, label=1)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], ("CHEMBL25", "CC(=O)Oc1ccccc1", 1))
        self.assertEqual(rows[2], ("CHEMBL1000", "c1ccccc1", 1))

    def test_parse_ism_label(self):
        rows = parse_ism(ISM, label=0)
        self.assertTrue(all(r[2] == 0 for r in rows))

    def test_parse_ism_missing_id(self):
        rows = parse_ism("c1ccccc1\n", label=1)
        self.assertEqual(rows, [("c1ccccc1", "c1ccccc1", 1)])


if __name__ == "__main__":
    unittest.main()
