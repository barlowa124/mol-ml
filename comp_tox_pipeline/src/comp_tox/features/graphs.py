"""Molecular graphs for the GNN baseline.

Each compound -> (x, edge_index, edge_attr) tuple:
    x:          (n_atoms, 8) float: atomic number/100, degree, formal charge,
                implicit Hs, aromatic, in-ring, hybridization index, chiral
    edge_index: (2, n_edges), both directions
    edge_attr:  (n_edges, 6): single/double/triple/aromatic one-hot,
                conjugated, in-ring

Saved as a pickled list via torch.save, row-aligned with the meta parquet.
"""

from __future__ import annotations

import sys

import pandas as pd
import torch
from rdkit import Chem

_HYB = ["SP", "SP2", "SP3", "SP3D", "SP3D2", "S", "UNSPECIFIED"]
_BONDS = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]


def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    x = torch.tensor(
        [
            [
                a.GetAtomicNum() / 100.0,
                float(a.GetDegree()),
                float(a.GetFormalCharge()),
                float(a.GetTotalNumHs()),
                float(a.GetIsAromatic()),
                float(a.IsInRing()),
                float(
                    _HYB.index(str(a.GetHybridization()))
                    if str(a.GetHybridization()) in _HYB
                    else len(_HYB)
                ),
                float(a.HasProp("_ChiralityPossible")),
            ]
            for a in mol.GetAtoms()
        ],
        dtype=torch.float32,
    )
    edges, attrs = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        attr = [float(b.GetBondType() == t) for t in _BONDS] + [
            float(b.GetIsConjugated()),
            float(b.IsInRing()),
        ]
        edges += [(i, j), (j, i)]
        attrs += [attr, attr]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t()
        if edges
        else torch.empty(2, 0, dtype=torch.long)
    )
    edge_attr = (
        torch.tensor(attrs, dtype=torch.float32)
        if attrs
        else torch.empty(0, 6, dtype=torch.float32)
    )
    return x, edge_index, edge_attr


def build_graphs(in_path: str, out_path: str) -> None:
    df = pd.read_parquet(in_path)
    graphs, keep = [], []
    for i, smi in enumerate(df["canonical_smiles"]):
        g = smiles_to_graph(smi)
        if g is not None:
            graphs.append(g)
            keep.append(i)
    if len(keep) != len(df):
        raise ValueError(
            f"{len(df) - len(keep)} compounds failed graph construction; "
            "upstream standardization should have dropped them"
        )
    torch.save(graphs, out_path)
    print(f"graphs: {len(graphs)} molecular graphs -> {out_path}")


def main() -> None:
    in_path, out_path = sys.argv[1], sys.argv[2]
    build_graphs(in_path, out_path)


if __name__ == "__main__":
    main()
