import tempfile
import unittest

from dockops.pdb import ca_records, records


def _atom(serial, resseq, x, resname="ALA"):
    return (
        f"ATOM  {serial:5d}  CA  {resname:>3s} A{resseq:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 80.00           C\n"
    )


def _pdb(text):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name


class PdbTests(unittest.TestCase):
    def test_first_model_only_for_ensembles(self):
        pdb = (
            "MODEL        1\n"
            + _atom(1, 1, 1.0)
            + "ENDMDL\n"
            + "MODEL        2\n"
            + _atom(1, 1, 9.0)
            + "ENDMDL\n"
        )
        ca = ca_records(_pdb(pdb))
        self.assertEqual(ca[1][1][0], 1.0)  # MODEL 1 coords, not MODEL 2

    def test_short_malformed_atom_line_skipped(self):
        pdb = "ATOM      1  CA  ALA A   1\n" + _atom(2, 2, 5.0)
        recs = list(records(_pdb(pdb)))
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["resseq"], 2)


if __name__ == "__main__":
    unittest.main()
